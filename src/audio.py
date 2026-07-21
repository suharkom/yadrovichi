"""Приведение аудио к 16 кГц моно — один раз на входе.

Пересэмплировать в каждой стадии отдельно нельзя: на часовом файле
это заметный кусок бюджета RTF.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 16_000


def prepare(src: str | Path, dst_dir: str | Path = "data/prepared") -> Path:
    """mp3/mp4/wav -> wav 16 кГц моно. Возвращает путь к готовому файлу."""
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}_16k.wav"

    if dst.exists():
        return dst

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


def duration(path: str | Path) -> float:
    """Длительность в секундах — нужна для расчёта RTF."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())
