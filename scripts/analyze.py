"""Аналитика вовлечённости по готовому результату пайплайна.

    python -m scripts.analyze artifacts/ml_result.json
    python -m scripts.analyze artifacts/ml_result.json --output artifacts/analytics.json

Читает JSON пайплайна, считает метрики (без GPU/LLM), печатает сводку и по
желанию сохраняет полную аналитику в файл.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.analytics import compute_analytics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="JSON результата пайплайна")
    parser.add_argument(
        "--output",
        help="куда сохранить полную аналитику (JSON)",
    )
    parser.add_argument(
        "--bucket-seconds",
        type=float,
        default=60.0,
        help="ширина корзины ленты доминирования, сек",
    )
    args = parser.parse_args()

    result = json.loads(
        Path(args.result).read_text(encoding="utf-8")
    )
    analytics = compute_analytics(
        result, ribbon_bucket_seconds=args.bucket_seconds
    )

    part = analytics["participation"]
    inter = analytics["interactivity"]

    print("Аналитика вовлечённости")
    print(
        f"  длительность:      "
        f"{analytics['lesson_duration_seconds'] / 60:.1f} мин"
    )
    print(
        f"  покрытие речью:    "
        f"{analytics['speech_coverage'] * 100:.0f}%"
    )
    print(f"  спикеров:          {analytics['speaker_count']}")
    print(
        f"  преподаватель:     "
        f"{part['teacher_talk_share'] * 100:.0f}% времени"
    )
    print(
        f"  ученики:           "
        f"{part['student_talk_share'] * 100:.0f}% времени"
    )
    print(
        f"  переключений/мин:  {inter['switches_per_minute']}"
    )
    print(
        f"  реплик с формулой: "
        f"{analytics['math']['utterances_with_formulas']}"
    )
    print("  говорили:")
    for s in analytics["speakers"]:
        print(
            f"    {s['display_name']:<16} "
            f"{s['talk_share'] * 100:5.1f}%  "
            f"{s['utterances']:>4} реплик  "
            f"{s['questions']:>3} вопр."
        )
    if analytics["flags"]:
        print(f"  наблюдения: {', '.join(analytics['flags'])}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(analytics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nПолная аналитика: {out}")


if __name__ == "__main__":
    main()
