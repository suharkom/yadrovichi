"""WER между нашим транскриптом и встроенной транскрипцией Контур Толк.

ВАЖНО про трактовку: это НЕ точность против человека, а РАСХОЖДЕНИЕ двух ASR —
нашей и встроенной в платформу. Обе машинные, человеческого эталона тут нет.
Качественно расхождения в нашу пользу (см. docs/wer.md: где Толк ломает термины
и формулы, мы даём верный вариант), поэтому подавать честно: «расхождение с
встроенной транскрипцией платформы», а не «наша точность X%».

    python -m scripts.wer_vs_talk --ours artifacts/ml_result.json \
        --talk "Транскрибация толк.docx"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Метки спикеров в выгрузке Толка — не текст речи, выкидываем.
_SPEAKER_LABEL = re.compile(
    r"^(участник\s*\d+|аудитория.*|спикер\s*\d+)$", re.IGNORECASE
)


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ours(path: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return " ".join(seg["text"] for seg in data.get("timeline", []))


def load_talk(path: str) -> str:
    file = Path(path)
    if file.suffix.lower() == ".docx":
        import docx

        document = docx.Document(str(file))
        lines = [
            p.text.strip()
            for p in document.paragraphs
            if p.text.strip() and not _SPEAKER_LABEL.match(p.text.strip())
        ]
        return " ".join(lines)
    return file.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="JSON результата пайплайна")
    parser.add_argument("--talk", required=True, help=".docx или .txt транскрипт Толка")
    args = parser.parse_args()

    import jiwer

    ours = normalize(load_ours(args.ours))
    talk = normalize(load_talk(args.talk))

    print(f"Слов у нас:   {len(ours.split())}")
    print(f"Слов у Толка: {len(talk.split())}")
    print(f"WER (эталон=Толк, гипотеза=мы):  {jiwer.wer(talk, ours) * 100:.1f}%")
    print(f"WER (эталон=мы,   гипотеза=Толк): {jiwer.wer(ours, talk) * 100:.1f}%")
    print(
        "\nЭто расхождение двух ASR, не точность против человека. "
        "Трактовать честно (см. docs/wer.md)."
    )


if __name__ == "__main__":
    main()
