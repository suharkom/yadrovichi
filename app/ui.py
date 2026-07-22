"""Веб-интерфейс.

Монтируется внутрь того же FastAPI-приложения: /docs — API, /ui — интерфейс.
Один процесс, один порт, ничего не дублируется.

Потоковая выдача здесь бесплатна: pipeline.run — генератор, а Gradio умеет
принимать функцию-генератор и дорисовывать результат на каждый yield.
Никакого SSE руками писать не нужно.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import gradio as gr

from src import audio, pipeline
from src.schema import Segment

# Цвет по роли: преподаватель отдельно, ученики оттенками.
COLORS = ["#c2410c", "#1d4ed8", "#15803d", "#7e22ce", "#b91c1c", "#0e7490"]


def _color(role: int | None) -> str:
    if role is None:
        return "#6b7280"
    return COLORS[role % len(COLORS)]


def _render(segments: list[Segment]) -> str:
    """Таймлайн репликами. Формулы показываем отдельной строкой:
    человеку нужна речь, символьная запись — справочно."""
    rows = []
    for seg in segments:
        stamp = f"{int(seg.start) // 60:02d}:{int(seg.start) % 60:02d}"
        math = ""
        if seg.has_math and seg.math_text:
            math = (f"<div style='font-family:monospace;font-size:.85em;"
                    f"opacity:.75;margin-top:.25em'>{seg.math_text}</div>")
        rows.append(
            f"<div style='margin:.6em 0;padding-left:.8em;"
            f"border-left:3px solid {_color(seg.role)}'>"
            f"<span style='opacity:.6;font-family:monospace'>{stamp}</span> "
            f"<b style='color:{_color(seg.role)}'>{seg.role_name or seg.speaker}</b>"
            f"<div>{seg.text}</div>{math}</div>"
        )
    return "".join(rows) or "<i>пусто</i>"


def process(file) -> Iterator[tuple[str, str, str]]:
    """Генератор: отдаёт (таймлайн, статус, json) на каждой готовой реплике."""
    if file is None:
        yield "<i>Загрузите файл</i>", "", ""
        return

    src = Path(file.name if hasattr(file, "name") else file)
    started = time.perf_counter()

    try:
        total = audio.duration(audio.prepare(src))
    except Exception:
        total = 0.0

    collected: list[Segment] = []
    for segment in pipeline.run(src):
        collected.append(segment)
        elapsed = time.perf_counter() - started
        done = segment.end

        # Счётчик RTF считается на лету. На защите это работает лучше
        # любого слайда: куратор видит, что укладываемся, прямо в демо.
        rtf = elapsed / done if done else 0.0
        progress = f"{done / total:.0%}" if total else "—"
        status = (f"Обработано {progress} · реплик {len(collected)} · "
                  f"прошло {elapsed:.0f}с · **RTF {rtf:.3f}**")

        yield _render(collected), status, ""

    elapsed = time.perf_counter() - started
    rtf = elapsed / total if total else 0.0
    verdict = "укладываемся" if rtf <= 0.4 else "ПРЕВЫШЕН бюджет 0.4"
    summary = (f"Готово · {len(collected)} реплик · {elapsed:.0f}с "
               f"на {total / 60:.1f} мин аудио · **RTF {rtf:.3f}** — {verdict}")

    payload = json.dumps([s.to_dict() for s in collected], ensure_ascii=False, indent=2)
    out = Path("data/results/last.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")

    yield _render(collected), summary, str(out)


def build() -> gr.Blocks:
    with gr.Blocks(title="yadrovichi") as demo:
        gr.Markdown("## Расшифровка урока\nРечь → текст по ролям с таймкодами")

        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Запись урока", file_types=["audio", "video"])
                run = gr.Button("Обработать", variant="primary")
                status = gr.Markdown()
                download = gr.File(label="Скачать JSON")
            with gr.Column(scale=2):
                timeline = gr.HTML(label="Таймлайн")

        run.click(process, inputs=file, outputs=[timeline, status, download])
    return demo


demo = build()
