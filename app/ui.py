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


def _ribbon_html(ribbon: list[dict]) -> str:
    """Цветная лента доминирования: одна полоса = корзина времени, цвет по
    роли/спикеру, кто в ней говорил дольше всех. Молчание — серым."""
    total = sum(item["end"] - item["start"] for item in ribbon) or 1.0
    cells = []
    for item in ribbon:
        width = (item["end"] - item["start"]) / total * 100
        if item.get("dominant_speaker_id") is None:
            color = "#e5e7eb"
            label = "тишина"
        else:
            color = _color(item["dominant_speaker_id"])
            label = item.get("dominant_display_name", "?")
        start = int(item["start"])
        tip = f"{start // 60:02d}:{start % 60:02d} · {label}"
        cells.append(
            f"<div title='{tip}' style='flex:{width:.4f} 0 0;height:26px;"
            f"background:{color}'></div>"
        )
    return (
        "<div style='display:flex;width:100%;border-radius:5px;"
        "overflow:hidden;margin:.3em 0'>" + "".join(cells) + "</div>"
    )


def _analytics_html(a: dict) -> str:
    """Сводка вовлечённости: доли речи по спикерам, интерактивность, флаги."""
    part = a["participation"]
    inter = a["interactivity"]
    head = (
        f"<div style='margin:.2em 0 .6em'>"
        f"Преподаватель <b>{part['teacher_talk_share'] * 100:.0f}%</b> · "
        f"ученики <b>{part['student_talk_share'] * 100:.0f}%</b> · "
        f"переключений/мин <b>{inter['switches_per_minute']}</b> · "
        f"реплик с формулой <b>{a['math']['utterances_with_formulas']}</b>"
        f"</div>"
    )
    bars = []
    for s in a["speakers"]:
        color = _color(s.get("speaker_id"))
        pct = s["talk_share"] * 100
        bars.append(
            f"<div style='margin:.25em 0'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:.9em'><span style='color:{color}'><b>"
            f"{s['display_name']}</b></span>"
            f"<span style='opacity:.7'>{pct:.1f}% · {s['utterances']} реплик "
            f"· {s['questions']} вопр.</span></div>"
            f"<div style='height:8px;background:#f1f5f9;border-radius:4px'>"
            f"<div style='width:{pct:.1f}%;height:8px;background:{color};"
            f"border-radius:4px'></div></div></div>"
        )
    flag_labels = {
        "student_led_session": "урок ведёт ученик",
        "single_student_dominates": "один ученик доминирует",
        "passive_student_present": "есть пассивный ученик",
        "sparse_speech": "много тишины",
    }
    flags = "".join(
        f"<span style='background:#fef3c7;border-radius:10px;padding:.1em .6em;"
        f"margin-right:.4em;font-size:.85em'>{flag_labels.get(f, f)}</span>"
        for f in a["flags"]
    )
    flags_row = f"<div style='margin-top:.5em'>{flags}</div>" if flags else ""
    return head + "".join(bars) + flags_row


def process(file):
    """Потоковый прогон: таймлайн заполняется по мере готовности реплик.

    Генератор — Gradio дорисовывает вывод на каждый yield.
    """
    if file is None:
        yield "<i>Загрузите файл</i>", "", None, ""
        return

    from app.main import WORK_DIR, get_pipeline
    from app.services.streaming import stream_pipeline

    src = Path(file.name if hasattr(file, "name") else file)
    collected: list[dict] = []
    meta: dict = {}

    yield "<i>Диаризация…</i>", "Загружаю и размечаю по голосам", None, ""

    for item in stream_pipeline(get_pipeline(), src, WORK_DIR):
        if item["type"] == "meta":
            meta = item
            yield (
                "<i>Распознаю речь…</i>",
                f"Спикеров {item['speaker_count']} · "
                f"преподаватель {item.get('teacher', '?')} · "
                f"диаризация RTF {item['diarization_rtf']:.3f}",
                None,
                "",
            )
        elif item["type"] == "utterance":
            collected.append(item)
            yield _render(collected), f"Реплик получено: {len(collected)}…", None, ""
        elif item["type"] == "done":
            rtf = item["pipeline_rtf"]
            verdict = "укладываемся" if rtf <= 0.4 else "ПРЕВЫШЕН бюджет 0.4"
            out = Path("data/results/last.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"meta": meta, "timeline": collected},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            status = (
                f"Готово · спикеров {meta.get('speaker_count', '?')} · "
                f"реплик {item['utterance_count']} · **RTF {rtf:.3f}** — {verdict}"
            )
            # Аналитика вовлечённости из собранного таймлайна.
            from app.services.analytics import compute_analytics

            a = compute_analytics(
                {
                    "audio_duration_seconds": meta.get("audio_seconds", 0.0),
                    "timeline": collected,
                }
            )
            analytics_html = _ribbon_html(a["ribbon"]) + _analytics_html(a)
            yield _render(collected), status, str(out), analytics_html


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
                gr.Markdown("**Лента вовлечённости**")
                analytics = gr.HTML()
                timeline = gr.HTML(label="Таймлайн")

        run.click(
            process,
            inputs=file,
            outputs=[timeline, status, download, analytics],
        )
    return demo


demo = build()
