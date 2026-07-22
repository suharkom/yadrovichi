from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioProcessingError(RuntimeError):
    """Ошибка подготовки или чтения аудиофайла."""


def validate_audio_file(
    audio_path: str | Path,
    max_file_size_mb: int = 200,
) -> Path:
    path = Path(audio_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")

    if not path.is_file():
        raise ValueError(f"Ожидался файл, получено: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb > max_file_size_mb:
        raise ValueError(
            f"Файл занимает {size_mb:.1f} МБ, "
            f"лимит — {max_file_size_mb} МБ."
        )

    return path


def get_audio_duration(audio_path: str | Path) -> float:
    path = Path(audio_path)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "Не найден ffprobe. Установи ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(
            f"ffprobe не смог прочитать файл: {exc.stderr}"
        ) from exc

    payload = json.loads(completed.stdout)
    duration = float(payload["format"]["duration"])

    if duration <= 0:
        raise AudioProcessingError(
            f"Некорректная длительность аудио: {duration}"
        )

    return duration


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16_000,
) -> Path:
    source = Path(input_path)
    destination = Path(output_path)

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "Не найден ffmpeg. На macOS установи: brew install ffmpeg"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(
            f"Не удалось нормализовать аудио:\n{exc.stderr}"
        ) from exc

    return destination
