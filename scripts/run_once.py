"""Один прогон настоящего пайплайна с человекочитаемым выводом.

Показывает то, чего не видно в bench_rtf: сколько спикеров нашла
диаризация, кого назначило преподавателем, как выглядит таймлайн.
Сохраняет полный JSON в data/results/last.json.

Запуск:
    python -m scripts.run_once data/audio.mp3
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from src import audio, stages
from src.roles import find_teacher, marker_score, neighbour_counts


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Использование: python -m scripts.run_once <файл>")

    prepared = audio.prepare(Path(sys.argv[1]))

    print("Диаризация...")
    segments = stages.diarize(str(prepared))
    speakers = sorted({s.speaker for s in segments})
    print(f"  сегментов: {len(segments)}, спикеров: {len(speakers)} — {speakers}")

    print("\nПризнаки для определения роли:")
    neighbours = neighbour_counts(segments)
    print("  (текста ещё нет, маркеры считаем после транскрибации)")
    for sp in speakers:
        secs = sum(s.duration for s in segments if s.speaker == sp)
        print(f"    {sp}: соседей {len(neighbours.get(sp, ())):>2}, речи {secs:6.0f}с")

    print("\nТранскрибация...")
    segments = stages.transcribe(str(prepared), segments)

    print("\nМаркеры на 1000 слов:")
    for sp, score in sorted(marker_score(segments).items()):
        print(f"    {sp}: {score:.1f}")

    teacher = find_teacher(segments)
    print(f"\nОпределён преподаватель: {teacher}")

    segments = stages.assign_roles(segments)
    segments.sort(key=lambda s: s.start)

    dist = Counter(s.role_name for s in segments)
    print("Распределение реплик по ролям:", dict(dist))

    print("\nПервые 25 реплик:")
    for seg in segments[:25]:
        stamp = f"{int(seg.start) // 60:02d}:{int(seg.start) % 60:02d}"
        text = seg.text[:70] if seg.text else "(пусто)"
        print(f"  {stamp} {seg.role_name:<14} | {text}")

    out = Path("data/results/last.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON сохранён: {out} ({len(segments)} реплик)")


if __name__ == "__main__":
    main()
