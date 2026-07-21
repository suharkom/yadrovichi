"""Конвейер целиком.

Ключевое решение: run() — генератор. Реплики отдаются по мере готовности,
а не копятся в список. Отсюда потоковая выдача в FastAPI и Gradio получается
бесплатно, без отдельного кода.

Постобработка уходит на CPU в отдельные процессы и догоняет параллельно,
пока GPU занят следующим куском. На RTF это влияет слабо (постобработка
правилами — единицы миллисекунд на чанк), но это задел под стриминг.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from . import audio, postprocess, stages
from .schema import Segment

POST_WORKERS = 3          # не больше половины ядер CPU — уточнить у куратора
CHECKPOINT = Path("data/checkpoint.jsonl")


def run(src: str | Path, checkpoint: bool = True) -> Iterator[Segment]:
    """Полный проход: подготовка -> диаризация -> ASR -> роли -> постобработка.

    Отдаёт реплики по одной, в порядке таймлайна.
    """
    prepared = audio.prepare(src)

    # Диаризация обязательно по всему файлу целиком, иначе спикеры
    # переименуются на середине записи.
    segments = stages.diarize(str(prepared))
    segments = stages.transcribe(str(prepared), segments)
    segments = stages.assign_roles(segments)
    segments.sort(key=lambda s: s.start)

    if checkpoint:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT.write_text("", encoding="utf-8")

    with ProcessPoolExecutor(max_workers=POST_WORKERS) as pool:
        # Futures складываются в список в порядке отправки — порядок на
        # выходе восстанавливается сам, буфер пересортировки не нужен.
        futures = []
        prev_text = ""
        for seg in segments:
            futures.append(pool.submit(postprocess.process, seg, prev_text))
            prev_text = seg.text

        for future in futures:
            done = future.result()
            if checkpoint:
                with CHECKPOINT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(done.to_dict(), ensure_ascii=False) + "\n")
            yield done


def run_with_rtf(src: str | Path) -> tuple[list[Segment], dict[str, float]]:
    """То же, но с замером. Прогрев считать отдельно и в RTF не включать."""
    prepared = audio.prepare(src)
    total = audio.duration(prepared)

    started = time.perf_counter()
    segments = list(run(src))
    elapsed = time.perf_counter() - started

    return segments, {
        "audio_seconds": round(total, 2),
        "wall_seconds": round(elapsed, 2),
        "rtf": round(elapsed / total, 4) if total else 0.0,
    }
