"""Веб-интерфейс поверх того же приложения: /docs — API, /ui — интерфейс.

Рендерит результат AudioProcessingPipeline: таймлайн реплик с ролью,
цветом по спикеру, счётчиком RTF и кнопкой скачать JSON.

Пайплайн батчевый, поэтому результат показывается после обработки целиком.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

# Цвет по роли: преподаватель отдельно, ученики оттенками.
COLORS = ["#c2410c", "#1d4ed8", "#15803d", "#7e22ce", "#b91c1c", "#0e7490"]


def _color(speaker_id) -> str:
    if speaker_id is None:
        return "#6b7280"
    return COLORS[int(speaker_id) % len(COLORS)]


def _render(timeline: list[dict]) -> str:
    rows = []
    for item in timeline:
        start = float(item.get("start", 0.0))
        stamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        name = item.get("display_name", item.get("source_speaker", "?"))
        color = _color(item.get("speaker_id"))
        text = item.get("text", "")
        rows.append(
            f"<div style='margin:.6em 0;padding-left:.8em;"
            f"border-left:3px solid {color}'>"
            f"<span style='opacity:.6;font-family:monospace'>{stamp}</span> "
            f"<b style='color:{color}'>{name}</b>"
            f"<div>{text}</div></div>"
        )
    return "".join(rows) or "<i>пусто</i>"


def process(file):
    """Прогон файла через пайплайн. Возвращает (таймлайн, статус, путь к JSON)."""
    if file is None:
        return "<i>Загрузите файл</i>", "", None

    from app.main import WORK_DIR, get_pipeline

    src = Path(file.name if hasattr(file, "name") else file)
    result = get_pipeline().process(src, work_dir=WORK_DIR)

    metrics = result.get("metrics", {})
    role = result.get("role_detection", {})
    rtf = metrics.get("pipeline_rtf", 0.0)
    verdict = "укладываемся" if rtf <= 0.4 else "ПРЕВЫШЕН бюджет 0.4"
    status = (
        f"Готово · спикеров {result.get('speaker_count', '?')} · "
        f"реплик {metrics.get('final_utterance_count', '?')} · "
        f"**RTF {rtf:.3f}** — {verdict}\n\n"
        f"ASR RTF {metrics.get('asr_rtf', 0):.3f} · "
        f"диаризация RTF {metrics.get('diarization_rtf', 0):.3f} · "
        f"преподаватель {role.get('teacher_speaker', '?')} "
        f"(уверенность {role.get('heuristic_confidence', 0):.2f}"
        f"{', низкая' if role.get('low_confidence') else ''})"
    )

    out = Path("data/results/last.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return _render(result.get("timeline", [])), status, str(out)


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
