"""Экспорт таймлайна в субтитры SRT и VTT.

Из готового результата пайплайна (реплики с таймкодами и ролями) собираем
стандартные субтитры. Имя роли выносим в начало реплики (в SRT — префиксом,
в VTT — голосовым тегом <v>), чтобы было видно, кто говорит.
"""

from __future__ import annotations

from typing import Any


def _timestamp(seconds: float, sep: str) -> str:
    """Секунды -> HH:MM:SS<sep>mmm. sep = ',' для SRT, '.' для VTT."""
    ms_total = int(round(max(0.0, seconds) * 1000))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, ms = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def to_srt(timeline: list[dict[str, Any]], with_speaker: bool = True) -> str:
    """Собрать .srt: пронумерованные блоки с таймкодами через запятую."""
    blocks: list[str] = []
    for index, utt in enumerate(timeline, start=1):
        start = _timestamp(float(utt["start"]), ",")
        end = _timestamp(float(utt["end"]), ",")
        text = str(utt.get("text", "")).strip()
        name = utt.get("display_name")
        body = f"{name}: {text}" if with_speaker and name else text
        blocks.append(f"{index}\n{start} --> {end}\n{body}")
    return "\n\n".join(blocks) + "\n"


def to_vtt(timeline: list[dict[str, Any]], with_speaker: bool = True) -> str:
    """Собрать .vtt: заголовок WEBVTT, таймкоды через точку, роль тегом <v>."""
    lines: list[str] = ["WEBVTT", ""]
    for utt in timeline:
        start = _timestamp(float(utt["start"]), ".")
        end = _timestamp(float(utt["end"]), ".")
        text = str(utt.get("text", "")).strip()
        name = utt.get("display_name")
        body = f"<v {name}>{text}</v>" if with_speaker and name else text
        lines.append(f"{start} --> {end}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
