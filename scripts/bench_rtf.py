"""Замер RTF по стадиям.

Методика зафиксирована здесь, чтобы цифры были сравнимы между прогонами
и между моделями — иначе спорить бессмысленно:

  - прогрев отдельным прогоном, в RTF не входит (первый раз грузятся веса)
  - три прогона, берём медиану, а не среднее
  - мерим на полном файле, а не на коротком куске
  - разбивка по стадиям: узкое место видно сразу

Запуск:
    python -m scripts.bench_rtf data/audio.mp3
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from src import audio, stages

RUNS = 3
BUDGET = {"diarize": 0.2, "transcribe": 0.2, "total": 0.4}


def time_stage(fn, *args) -> tuple[float, object]:
    started = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - started, result


def one_pass(path: str) -> dict[str, float]:
    diar_s, segments = time_stage(stages.diarize, path)
    asr_s, segments = time_stage(stages.transcribe, path, segments)
    roles_s, _ = time_stage(stages.assign_roles, segments)
    return {"diarize": diar_s, "transcribe": asr_s, "roles": roles_s,
            "total": diar_s + asr_s + roles_s}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Использование: python -m scripts.bench_rtf <файл>")

    src = Path(sys.argv[1])
    prepared = audio.prepare(src)
    seconds = audio.duration(prepared)
    print(f"Файл: {src.name}, длительность {seconds / 60:.1f} мин\n")

    print("Прогрев (в RTF не входит)...")
    one_pass(str(prepared))

    runs: list[dict[str, float]] = []
    for i in range(1, RUNS + 1):
        print(f"Прогон {i}/{RUNS}...")
        runs.append(one_pass(str(prepared)))

    print(f"\n{'Стадия':<14}{'сек':>10}{'RTF':>10}{'бюджет':>10}{'':>8}")
    print("-" * 52)
    for stage in ("diarize", "transcribe", "roles", "total"):
        median = statistics.median(r[stage] for r in runs)
        rtf = median / seconds
        budget = BUDGET.get(stage)
        if budget is None:
            mark, shown = "", "-"
        else:
            mark, shown = ("OK" if rtf <= budget else "ПРЕВЫШЕН"), f"{budget:.2f}"
        print(f"{stage:<14}{median:>10.1f}{rtf:>10.4f}{shown:>10}{mark:>8}")

    print("\nЕсли не влезаем — рычаги по убыванию эффекта:")
    print("  1. батчинг сегментов (BatchedInferencePipeline)")
    print("  2. VAD-фильтр: на уроке тишины легко 20-30%")
    print("  3. temperature=0 без fallback (следить за зацикливаниями)")
    print("  4. и только потом beam 5 -> 2 -> 1")


if __name__ == "__main__":
    main()
