from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import load_settings
from app.services.pipeline import AudioProcessingPipeline


def save_json(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запуск ASR, диаризации "
            "и определения ролей."
        )
    )

    parser.add_argument(
        "audio_path",
        type=Path,
        help="Путь к аудиофайлу.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/result.json"),
        help="Путь для сохранения итогового JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        settings = load_settings()

        print("Конфигурация:")
        print(f"  device: {settings.asr_device}")
        print(f"  ASR model: {settings.asr_model_name}")
        print(
            f"  compute type: "
            f"{settings.asr_compute_type}"
        )
        print(
            f"  beam size: {settings.asr_beam_size}"
        )
        print(f"  language: {settings.asr_language or 'auto'}")
        print(
            "  parallel GPU stages: "
            f"{settings.parallel_gpu_stages}"
        )
        print(
            f"  diarization model: "
            f"{settings.diarization_model_name}"
        )
        print()

        pipeline = AudioProcessingPipeline(settings)
        result = pipeline.process(args.audio_path)

        save_json(result, args.output)

        role_detection = result["role_detection"]
        metrics = result["metrics"]

        print()
        print("Готово:")
        print(
            f"  преподаватель: "
            f"{role_detection['teacher_speaker']}"
        )
        print(
            f"  эвристическая уверенность: "
            f"{role_detection['heuristic_confidence']:.3f}"
        )
        print(
            f"  низкая уверенность: "
            f"{role_detection['low_confidence']}"
        )
        print(
            f"  спикеров: "
            f"{result['speaker_count']}"
        )
        print(
            f"  итоговых реплик: "
            f"{metrics['final_utterance_count']}"
        )
        print(
            f"  ASR RTF: "
            f"{metrics['asr_rtf']:.3f}"
        )
        print(
            f"  diarization RTF: "
            f"{metrics['diarization_rtf']:.3f}"
        )
        print(
            f"  pipeline RTF: "
            f"{metrics['pipeline_rtf']:.3f}"
        )
        print(
            f"  слов UNKNOWN: "
            f"{metrics['unknown_word_count']}"
        )
        print(f"  результат: {args.output}")

    except Exception as exc:
        print(
            f"\nОшибка: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
