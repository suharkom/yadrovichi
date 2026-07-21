"""Определение преподавателя и нумерация учеников.

Ни одной модели, GPU не нужен. Основа — граф переходов по таймлайну,
маркеры вторым голосом.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .schema import Segment, TEACHER_ROLE, role_name

# Сильные маркеры: так говорит только ведущий занятие.
# Пополняет МЛщик по ходу прогонов. Держим два уровня, не четыре:
# подбирать много весов не на чем — размеченных записей будет пять штук.
STRONG_MARKERS = [
    "кто ответит", "кто хочет ответить", "поднимите руку", "к доске",
    "запишите", "откройте тетради", "откройте учебник",
    "переходим к следующей теме", "давайте разберем", "давайте рассмотрим",
    "обратите внимание", "попробуйте решить",
]

WEAK_MARKERS = [
    "посмотрите", "прочитайте", "подумайте", "вспомните", "сравните",
    "назовите", "объясните", "приведите пример", "повторите",
]

STRONG_WEIGHT = 3.0
WEAK_WEIGHT = 1.0


def normalize(text: str) -> str:
    """Нижний регистр и ё -> е. Без этого маркеры промахиваются."""
    return text.lower().replace("ё", "е")


def speaker_sequence(segments: list[Segment]) -> list[str]:
    """Последовательность говоривших, соседние одинаковые схлопнуты.

    Схлопывание обязательно: паузы внутри одной реплики иначе
    раздувают счёт переходов.
    """
    seq: list[str] = []
    for seg in sorted(segments, key=lambda s: s.start):
        if not seq or seq[-1] != seg.speaker:
            seq.append(seg.speaker)
    return seq


def neighbour_counts(segments: list[Segment]) -> dict[str, set[str]]:
    """Для каждого спикера — множество тех, с кем он соседствует.

    На занятии переходы идут преподаватель -> ученик -> преподаватель,
    поэтому у преподавателя собеседников больше всех.
    """
    seq = speaker_sequence(segments)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(seq, seq[1:]):
        if a != b:
            neighbours[a].add(b)
            neighbours[b].add(a)
    return dict(neighbours)


def marker_score(segments: list[Segment]) -> dict[str, float]:
    """Счёт маркеров на 1000 слов кластера.

    Нормировка обязательна, иначе побеждает тот, кто просто больше говорил.
    Поиск по границам слова: иначе "тихо" ловит "тихонько".
    """
    texts: dict[str, list[str]] = defaultdict(list)
    for seg in segments:
        texts[seg.speaker].append(normalize(seg.text))

    scores: dict[str, float] = {}
    for speaker, parts in texts.items():
        text = " ".join(parts)
        words = len(text.split())
        if not words:
            scores[speaker] = 0.0
            continue
        hits = 0.0
        for marker, weight in ((m, STRONG_WEIGHT) for m in STRONG_MARKERS):
            hits += weight * len(re.findall(rf"\b{re.escape(marker)}\b", text))
        for marker, weight in ((m, WEAK_WEIGHT) for m in WEAK_MARKERS):
            hits += weight * len(re.findall(rf"\b{re.escape(marker)}\b", text))
        scores[speaker] = hits / words * 1000
    return scores


def find_teacher(segments: list[Segment]) -> str:
    """Кластер преподавателя.

    Граф основной, маркеры вторым голосом. На двух спикерах граф
    бесполезен — у обоих по одному соседу — тогда решают маркеры.
    """
    speakers = {seg.speaker for seg in segments}
    if not speakers:
        return ""
    if len(speakers) == 1:
        return next(iter(speakers))

    scores = marker_score(segments)

    if len(speakers) == 2:
        return max(speakers, key=lambda s: scores.get(s, 0.0))

    neighbours = neighbour_counts(segments)
    by_graph = max(speakers, key=lambda s: len(neighbours.get(s, ())))
    by_markers = max(speakers, key=lambda s: scores.get(s, 0.0))

    if by_graph == by_markers:
        return by_graph
    # Разошлись — берём граф, он не зависит от текста и от типа занятия.
    # TODO(МЛщик): такие случаи отдавать Qwen2.5-3B по 500 слов от кластера.
    return by_graph


def number_speakers(segments: list[Segment], teacher: str) -> list[Segment]:
    """Преподавателю 0, остальным 1..N по первому появлению на таймлайне."""
    mapping = {teacher: TEACHER_ROLE}
    next_id = 1
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.speaker not in mapping:
            mapping[seg.speaker] = next_id
            next_id += 1

    for seg in segments:
        seg.role = mapping.get(seg.speaker)
        if seg.role is not None:
            seg.role_name = role_name(seg.role)
    return segments


def collapse_to_two_roles(segments: list[Segment]) -> list[Segment]:
    """Все ученики в одного — буквальное выполнение требования куратора
    про две роли. Держим как переключатель на случай рваной разбивки."""
    for seg in segments:
        if seg.role is not None and seg.role != TEACHER_ROLE:
            seg.role = 1
            seg.role_name = "Ученик"
    return segments
