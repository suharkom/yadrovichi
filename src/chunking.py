"""Нарезка на чанки для ASR.

Режем по границам пауз из диаризации, а не по таймеру: нарезка по
фиксированным 30 секундам рвёт слова посередине и поднимает WER.

Два правила, которые дают почти весь эффект:

  - соседние сегменты одного спикера склеиваем до TARGET секунд.
    Whisper на коротких кусках теряет контекст, а вызовов становится
    в разы больше — это дорого само по себе.
  - чанк никогда не содержит двух спикеров. Тогда привязка "кто сказал"
    получается по построению, а не сшивкой постфактум.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Segment

TARGET = 30.0     # к этой длине стремимся склеивать
MAX = 40.0        # жёсткий потолок: длиннее не склеиваем
MIN_SPEECH = 1.0  # реплики короче — эмбеддинг мусорный, клеим к соседу
GAP = 1.5         # пауза длиннее — режем, даже если спикер тот же


@dataclass
class Chunk:
    """Кусок одного спикера, который уедет в ASR одним вызовом."""

    index: int
    start: float
    end: float
    speaker: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def build(segments: list[Segment]) -> list[Chunk]:
    """Сегменты диаризации -> чанки для транскрибации."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    ordered = _absorb_short(ordered)

    chunks: list[Chunk] = []
    current: Chunk | None = None

    for seg in ordered:
        if current is None or _must_break(current, seg):
            current = Chunk(index=len(chunks), start=seg.start, end=seg.end,
                            speaker=seg.speaker, segments=[seg])
            chunks.append(current)
        else:
            current.end = seg.end
            current.segments.append(seg)

    return chunks


def _must_break(chunk: Chunk, seg: Segment) -> bool:
    if seg.speaker != chunk.speaker:
        return True
    if seg.start - chunk.end > GAP:
        return True
    if seg.end - chunk.start > MAX:
        return True
    return chunk.duration >= TARGET


def _absorb_short(segments: list[Segment]) -> list[Segment]:
    """Реплики короче секунды приклеиваем к соседнему сегменту.

    На таких кусках эмбеддинг спикера ненадёжен, а отдельный вызов ASR
    ради полусекунды — чистые накладные расходы.
    """
    out: list[Segment] = []
    for seg in segments:
        if seg.duration >= MIN_SPEECH or not out:
            out.append(seg)
            continue
        prev = out[-1]
        if prev.speaker == seg.speaker:
            prev.end = seg.end
        else:
            out.append(seg)
    return out


def stats(chunks: list[Chunk]) -> dict[str, float]:
    """Для отчёта куратору и для отладки нарезки."""
    if not chunks:
        return {}
    lengths = [c.duration for c in chunks]
    return {
        "chunks": len(chunks),
        "total_seconds": round(sum(lengths), 1),
        "avg_seconds": round(sum(lengths) / len(lengths), 1),
        "min_seconds": round(min(lengths), 1),
        "max_seconds": round(max(lengths), 1),
    }
