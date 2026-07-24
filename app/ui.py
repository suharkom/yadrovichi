"""Веб-интерфейс поверх того же приложения: /docs — API, /ui — интерфейс.

Вкладки: «Расшифровка» — лента спикеров над текстом реплик по ролям;
«Вовлечённость» — график по времени и доли речи; «История» — прошлые прогоны
с диска. Лента и метрики обновляются по ходу распознавания, не дожидаясь конца.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import gradio as gr

from app.services.refine import refine_utterances

# Ниже этого порога уверенности ASR слово считаем сомнительным и подсвечиваем.
UNCERTAIN_THRESHOLD = 0.5

# Каждые столько выплюнутых реплик прогоняем батч-постобработку.
REFINE_EVERY = 10

# Цвет по спикеру: преподаватель отдельно, ученики оттенками.
COLORS = ["#c2410c", "#1d4ed8", "#15803d", "#7e22ce", "#b91c1c", "#0e7490"]
ROLE_COLORS = {"teacher": "#c2410c", "student": "#1d4ed8", "unknown": "#9ca3af"}

RESULTS_DIR = Path("data/results")

# Футер Gradio (стикеры внизу) убираем через <head> — его содержимое не
# санитайзится, в отличие от gr.HTML, а css-параметр в этой сборке не сработал.
HEAD = (
    "<style>"
    "footer{display:none !important;}"
    ".gradio-container{max-width:100% !important;}"
    # Лампочка-переключатель темы: иконка в правом верхнем углу, без фона/рамки.
    "#theme-toggle{position:fixed;top:8px;right:14px;z-index:1000;"
    "width:40px;min-width:40px !important;height:40px;padding:0 !important;"
    "font-size:22px;line-height:1;background:transparent !important;"
    "border:none !important;box-shadow:none !important;}"
    "</style>"
)
CSS = ""

# Переключатель темы без перезагрузки: Gradio вешает класс dark на <body>.
THEME_TOGGLE_JS = "() => { document.body.classList.toggle('dark'); }"


def _color(speaker_id) -> str:
    if speaker_id is None:
        return "#9ca3af"
    return COLORS[int(speaker_id) % len(COLORS)]


def _text_with_confidence(item: dict) -> str:
    """Текст реплики с подсветкой низкоуверенных слов (пунктир + подсказка).
    Если по-словных данных нет (старый прогон/батч) — обычный текст."""
    words = item.get("words")
    if not words:
        return item.get("text", "")
    parts: list[str] = []
    for word in words:
        token = str(word.get("text", "")).strip()
        if not token:
            continue
        prob = word.get("probability")
        if prob is not None and prob < UNCERTAIN_THRESHOLD:
            parts.append(
                f"<span style='border-bottom:2px dotted #f59e0b;cursor:help' "
                f"title='низкая уверенность {prob * 100:.0f}%'>{token}</span>"
            )
        else:
            parts.append(token)
    html = " ".join(parts)
    return re.sub(r"\s+([,.;:!?…])", r"\1", html)


def _render(timeline: list[dict]) -> str:
    rows = []
    for item in timeline:
        start = float(item.get("start", 0.0))
        stamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        name = item.get("display_name", item.get("source_speaker", "?"))
        color = _color(item.get("speaker_id"))
        text = _text_with_confidence(item)
        rows.append(
            f"<div style='margin:.6em 0;padding-left:.8em;"
            f"border-left:3px solid {color}'>"
            f"<span style='opacity:.6;font-family:monospace'>{stamp}</span> "
            f"<b style='color:{color}'>{name}</b>"
            f"<div>{text}</div></div>"
        )
    return "".join(rows) or "<i>пусто</i>"


def _ribbon_html(ribbon: list[dict]) -> str:
    """Цветная лента доминирования: полоса = корзина времени, цвет по спикеру,
    кто говорил дольше всех. Молчание — серым. Идёт НАД всей расшифровкой."""
    if not ribbon:
        return ""
    total = sum(item["end"] - item["start"] for item in ribbon) or 1.0
    cells = []
    for item in ribbon:
        width = (item["end"] - item["start"]) / total * 100
        if item.get("dominant_speaker_id") is None:
            color, label = "rgba(128,128,128,.25)", "тишина"
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
        "<div style='font-weight:600;margin:.2em 0'>Лента спикеров</div>"
        "<div style='display:flex;width:100%;border-radius:5px;"
        "overflow:hidden;margin:.2em 0 .8em'>" + "".join(cells) + "</div>"
    )


def _timechart_svg(ribbon: list[dict], bucket_seconds: float = 60.0) -> str:
    """Вовлечённость по времени: столбик на корзину, высота — доля речи в
    минуте, цвет стопкой по СПИКЕРАМ — теми же цветами, что и лента, чтобы
    график и лента совпадали (напр. Ученик 2 зелёный там и там)."""
    if not ribbon:
        return ""
    n = len(ribbon)
    W, H = 1000.0, 120.0
    bw = W / n
    bars = []
    for i, item in enumerate(ribbon):
        x = i * bw
        y = H
        for seg in item.get("speaker_seconds", []):
            h = seg["seconds"] / bucket_seconds * H
            if h <= 0:
                continue
            y -= h
            bars.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bw:.2f}' "
                f"height='{h:.2f}' fill='{_color(seg['speaker_id'])}'/>"
            )
    legend = (
        "<div style='font-size:.85em;opacity:.8;margin-top:.2em'>"
        "цвет по спикеру, как в ленте · ось X — минуты урока</div>"
    )
    return (
        f"<svg viewBox='0 0 {W:.0f} {H:.0f}' preserveAspectRatio='none' "
        f"style='width:100%;height:120px'>{''.join(bars)}</svg>" + legend
    )


def _bars_html(a: dict) -> str:
    """Проценты и саммари: доли речи по спикерам, интерактивность, флаги."""
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
            f"<div style='height:8px;background:rgba(128,128,128,.25);"
            f"border-radius:4px'>"
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
        f"<span style='background:#fde68a;color:#78350f;border-radius:10px;"
        f"padding:.1em .6em;margin-right:.4em;font-size:.85em'>"
        f"{flag_labels.get(f, f)}</span>"
        for f in a["flags"]
    )
    flags_row = f"<div style='margin-top:.5em'>{flags}</div>" if flags else ""
    return head + "".join(bars) + flags_row


def _engagement_html(a: dict) -> str:
    """Вкладка «Вовлечённость»: график по времени + проценты и саммари."""
    return (
        "<b>Вовлечённость по времени</b>"
        + _timechart_svg(a["ribbon"])
        + "<div style='margin-top:.8em'></div>"
        + _bars_html(a)
    )


def _analytics_of(collected: list[dict], meta: dict) -> dict:
    from app.services.analytics import compute_analytics

    return compute_analytics(
        {
            "audio_duration_seconds": meta.get("audio_seconds", 0.0),
            "timeline": collected,
        }
    )


def _history_choices() -> list[tuple[str, str]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    return [(f.stem.replace("run_", "").replace("_", " ", 1), str(f)) for f in files]


def process(file):
    """Потоковый прогон. Обновляет ленту, таймлайн и метрики по ходу.

    Выходы: лента (над текстом), таймлайн, статус, файл для скачивания,
    блок вовлечённости, список истории.
    """
    if file is None:
        yield "", "<i>Загрузите файл</i>", "", None, "", gr.update()
        return

    from app.main import WORK_DIR, get_pipeline
    from app.services.streaming import stream_pipeline

    src = Path(file.name if hasattr(file, "name") else file)
    collected: list[dict] = []
    meta: dict = {}

    # Старт нового прогона: чистим ленту и график от прошлого файла, как и текст.
    yield (
        "",
        "<i>Диаризация…</i>",
        "Загружаю и размечаю по голосам",
        None,
        "",
        gr.update(),
    )

    for item in stream_pipeline(get_pipeline(), src, WORK_DIR):
        if item["type"] == "meta":
            meta = item
            yield (
                gr.update(),
                "<i>Распознаю речь…</i>",
                f"Спикеров {item['speaker_count']} · "
                f"преподаватель {item.get('teacher', '?')} · "
                f"диаризация RTF {item['diarization_rtf']:.3f}",
                None,
                gr.update(),
                gr.update(),
            )
        elif item["type"] == "utterance":
            collected.append(item)
            # Батч-постобработка и обновление ленты/метрик раз в REFINE_EVERY реплик.
            if len(collected) % REFINE_EVERY == 0:
                refine_utterances(collected)
                a = _analytics_of(collected, meta)
                ribbon_val = _ribbon_html(a["ribbon"])
                eng_val = _engagement_html(a)
            else:
                ribbon_val = gr.update()
                eng_val = gr.update()
            yield (
                ribbon_val,
                _render(collected),
                f"Реплик получено: {len(collected)}…",
                gr.update(),
                eng_val,
                gr.update(),
            )
        elif item["type"] == "done":
            refine_utterances(collected)  # финальная батч-постобработка остатка
            rtf = item["pipeline_rtf"]
            verdict = "укладываемся" if rtf <= 0.4 else "ПРЕВЫШЕН бюджет 0.4"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H-%M")
            base = RESULTS_DIR / f"run_{stamp}_{src.stem[:40]}"
            json_path = base.with_suffix(".json")
            json_path.write_text(
                json.dumps({"meta": meta, "timeline": collected},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Экспорт субтитров рядом с JSON — сразу доступны для скачивания.
            from app.services.export_subs import to_srt, to_vtt

            srt_path = base.with_suffix(".srt")
            srt_path.write_text(to_srt(collected), encoding="utf-8")
            vtt_path = base.with_suffix(".vtt")
            vtt_path.write_text(to_vtt(collected), encoding="utf-8")

            status = (
                f"Готово · спикеров {meta.get('speaker_count', '?')} · "
                f"реплик {item['utterance_count']} · **RTF {rtf:.3f}** — {verdict}"
            )
            a = _analytics_of(collected, meta)
            yield (
                _ribbon_html(a["ribbon"]),
                _render(collected),
                status,
                [str(json_path), str(srt_path), str(vtt_path)],
                _engagement_html(a),
                gr.update(choices=_history_choices()),
            )


def load_history(path: str):
    """Показать сохранённый прогон: лента + график + текст (в этом порядке)."""
    if not path:
        return "", "", "<i>Выберите прогон</i>"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    timeline = data.get("timeline", [])
    a = _analytics_of(timeline, data.get("meta", {}))
    return _ribbon_html(a["ribbon"]), _engagement_html(a), _render(timeline)


def build() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")
    with gr.Blocks(title="yadrovichi", theme=theme, css=CSS, head=HEAD) as demo:
        # Лампочка в правом верхнем углу — переключение светлой/тёмной темы.
        theme_btn = gr.Button("💡", elem_id="theme-toggle")
        theme_btn.click(fn=None, inputs=None, outputs=None, js=THEME_TOGGLE_JS)

        gr.Markdown("## Расшифровка урока")

        with gr.Row():
            # Основная область слева: лента спикеров НАД вкладками (видна всегда),
            # под ней — вкладки.
            with gr.Column(scale=3):
                ribbon = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("Расшифровка"):
                        gr.Markdown(
                            "<span style='opacity:.6;font-size:.85em'>Пунктиром "
                            "подчёркнуты слова с низкой уверенностью "
                            "распознавания.</span>"
                        )
                        timeline = gr.HTML()
                    with gr.Tab("Вовлечённость"):
                        engagement = gr.HTML()
                    with gr.Tab("История"):
                        gr.Markdown("Прошлые прогоны:")
                        history_dd = gr.Dropdown(
                            label="Выберите прогон",
                            choices=_history_choices(),
                            interactive=True,
                        )
                        load = gr.Button("Показать")
                        hist_ribbon = gr.HTML()
                        hist_engagement = gr.HTML()
                        hist_timeline = gr.HTML()
            # Панель управления справа: загрузка, обработка, скачивание.
            with gr.Column(scale=1):
                file = gr.File(
                    label="Запись урока", file_types=["audio", "video"]
                )
                run = gr.Button("Обработать", variant="primary")
                status = gr.Markdown()
                download = gr.File(
                    label="Скачать: JSON / SRT / VTT", file_count="multiple"
                )

        run.click(
            process,
            inputs=file,
            outputs=[ribbon, timeline, status, download, engagement, history_dd],
        )
        load.click(
            load_history,
            inputs=history_dd,
            outputs=[hist_ribbon, hist_engagement, hist_timeline],
        )
    return demo


demo = build()
