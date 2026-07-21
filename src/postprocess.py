"""Постобработка текста после ASR.

Правила на CPU, никаких моделей. Главный реальный дефект whisper
на длинных файлах — зацикливания, всё остальное косметика.
"""

from __future__ import annotations

import re

from .schema import Segment

# Модель стабильно калечит одни и те же слова: имена, термины предмета.
# Заранее словаря нет — он накапливается: прогнали, прочитали, добавили строку.
# Отсюда же собирается initial_prompt для whisper.
TERMS: dict[str, str] = {
    # "идиолог": "идеолог",
}

FILLERS = ["э-э", "ну вот", "как бы", "то есть вот"]


def initial_prompt() -> str:
    """Подсказка whisper, чтобы термины узнавались сразу правильно."""
    return ", ".join(TERMS.values())


def fix_terms(text: str) -> str:
    for wrong, right in TERMS.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def drop_loops(text: str, n: int = 4, limit: int = 2) -> str:
    """Чистка зацикливаний: подряд повторяющиеся n-граммы схлопываются.

    Whisper на длинных файлах иногда впадает в повтор одной фразы —
    это ловится дедупликацией, а не тюнингом параметров.
    """
    words = text.split()
    if len(words) < n * 2:
        return text

    out: list[str] = []
    i = 0
    while i < len(words):
        gram = words[i:i + n]
        repeats = 1
        j = i + n
        while words[j:j + n] == gram:
            repeats += 1
            j += n
        if repeats > 1:
            for _ in range(min(repeats, limit)):
                out.extend(gram)
            i = j
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def strip_fillers(text: str) -> str:
    for filler in FILLERS:
        text = re.sub(rf"\b{re.escape(filler)}\b[,]?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def join_boundary(prev: str, curr: str) -> str:
    """Склейка предложения, разорванного на стыке чанков.

    Если предыдущий кусок не закончен знаком препинания, а следующий
    начинается со строчной — это одно предложение.
    """
    if not prev or not curr:
        return curr
    if prev.rstrip()[-1:] not in ".!?" and curr[:1].islower():
        return curr
    return curr


def process(segment: Segment, prev_text: str = "") -> Segment:
    """Полная постобработка одной реплики. Чистая функция — можно
    гонять в ProcessPoolExecutor параллельно с ASR следующего чанка."""
    text = segment.text
    text = fix_terms(text)
    text = strip_fillers(text)
    text = join_boundary(prev_text, text)
    segment.text = text
    return segment


def normalize_for_wer(text: str) -> str:
    """⚠️ Прогонять через это ОБА текста перед jiwer.

    Без нормализации получите WER под 40% на ровном месте и полдня
    будете чинить проблему, которой нет.
    """
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
