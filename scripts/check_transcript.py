"""Проверки транскрипта по чеклисту куратора.

    - язык транскрипта совпадает с языком записи
    - математические формулы записываются в понятном для LLM виде
    - есть таймлайн фрагментов

Запуск на готовом результате:
    python -m scripts.check_transcript data/results/<job_id>.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN = re.compile(r"[a-z]", re.IGNORECASE)


def check_language(segments: list[dict]) -> None:
    """Whisper на тишине и шуме умеет "переключить" язык и выдать
    английский или китайский посреди русской записи. Ловится долей
    кириллицы: у нормального русского сегмента она заметно выше половины.
    """
    suspicious = []
    for seg in segments:
        text = seg.get("text", "")
        letters = CYRILLIC.findall(text) + LATIN.findall(text)
        if len(letters) < 20:
            continue
        share = len(CYRILLIC.findall(text)) / len(letters)
        if share < 0.5:
            suspicious.append((seg["start"], share, text[:60]))

    if not suspicious:
        print("Язык: OK, весь транскрипт русский")
        return
    print(f"Язык: ⚠️ {len(suspicious)} подозрительных сегментов")
    for start, share, preview in suspicious[:5]:
        print(f"   {start:8.2f}s  кириллицы {share:.0%}  {preview}")
    print("   Лечится: language=\"ru\" жёстко, без автоопределения")


def check_timeline(segments: list[dict]) -> None:
    """Таймлайн должен быть у каждого фрагмента, без дыр и наложений."""
    missing = [s for s in segments if s.get("start") is None or s.get("end") is None]
    broken = [s for s in segments if s.get("end", 0) <= s.get("start", 0)]

    overlaps = 0
    for prev, curr in zip(segments, segments[1:]):
        if curr["start"] < prev["end"] - 0.01:
            overlaps += 1

    print(f"Таймлайн: {len(segments)} фрагментов, "
          f"без таймкодов {len(missing)}, битых {len(broken)}, наложений {overlaps}")
    if segments:
        print(f"   покрытие {segments[0]['start']:.1f}s - {segments[-1]['end']:.1f}s")


def check_math(segments: list[dict]) -> None:
    """Формулы должны быть в символьном виде, а не словами."""
    with_math = [s for s in segments if s.get("has_math")]
    unconverted = [s for s in with_math if not s.get("math_text")]

    print(f"Формулы: найдено в {len(with_math)} фрагментах, "
          f"не сконвертировано {len(unconverted)}")
    for seg in with_math[:3]:
        print(f"   было:  {seg['text'][:70]}")
        print(f"   стало: {seg.get('math_text', '')[:70]}")


def check_roles(segments: list[dict]) -> None:
    roles = {s.get("role_name") for s in segments if s.get("role_name")}
    print(f"Роли: {len(roles)} — {sorted(roles)}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Использование: python -m scripts.check_transcript <result.json>")

    segments = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    segments.sort(key=lambda s: s["start"])

    check_language(segments)
    check_timeline(segments)
    check_math(segments)
    check_roles(segments)


if __name__ == "__main__":
    main()
