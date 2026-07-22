"""Локальный бэкенд поверх ML-пайплайна из app/services.

Требование куратора: локальный сервис на FastAPI, приём файла до 200 МБ.

Решения, зафиксированные здесь:

1. Файл пишется на диск потоком по 1 МБ, а не читается в память целиком —
   200 МБ в RAM на каждый запрос кладут сервис при демо.
2. Одна задача на GPU за раз (Semaphore). Видеокарта одна, второй прогон
   не влезет рядом с загруженными моделями. Uvicorn — строго --workers 1.
3. Модели грузятся один раз (ленивая инициализация пайплайна), а не в
   каждом запросе, иначе каждый запрос заново поднимает веса на GPU.

Пайплайн (AudioProcessingPipeline) синхронный и батчевый: обрабатывает
файл целиком и возвращает JSON. Потоковая выдача по мере готовности —
следующий шаг (пайплайн надо превратить в генератор), пока не сделана.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

MAX_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
UPLOAD_DIR = Path("data/uploads")
RESULT_DIR = Path("data/results")
WORK_DIR = Path("data/work")

gpu_lock = asyncio.Semaphore(1)
jobs: dict[str, dict] = {}

_pipeline = None


def get_pipeline():
    """Ленивая инициализация: torch/pyannote/whisper поднимаются один раз
    при первом запросе. Импорт внутри функции, чтобы `import app.main`
    работал на машине без GPU (для тестов и монтирования Gradio)."""
    global _pipeline
    if _pipeline is None:
        from app.core.config import load_settings
        from app.services.pipeline import AudioProcessingPipeline

        _pipeline = AudioProcessingPipeline(load_settings())
    return _pipeline


app = FastAPI(title="yadrovichi")


async def save_upload(upload: UploadFile) -> Path:
    """Потоковая запись на диск с проверкой размера по ходу."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "audio").suffix or ".mp3"
    dst = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    written = 0
    with dst.open("wb") as fh:
        while chunk := await upload.read(CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_BYTES:
                fh.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"Файл больше {MAX_BYTES // 1024 // 1024} МБ")
            fh.write(chunk)
    return dst


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """Приём файла -> обработка -> итоговый JSON с таймлайном и ролями."""
    src = await save_upload(file)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "source": str(src)}

    async with gpu_lock:
        jobs[job_id]["status"] = "running"
        loop = asyncio.get_running_loop()

        # Пайплайн синхронный и грузит GPU — уводим в пул потоков, чтобы не
        # блокировать event loop и остальные запросы, пока идёт обработка.
        result = await loop.run_in_executor(
            None, lambda: get_pipeline().process(src, work_dir=WORK_DIR)
        )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULT_DIR / f"{job_id}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        jobs[job_id] = {"status": "done", "result": str(result_path)}

    return JSONResponse(result)


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return jobs[job_id]


@app.get("/result/{job_id}")
async def result(job_id: str):
    """Сохранённый JSON — для скачивания и для метрик."""
    path = RESULT_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "Результат не найден")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "queued": len(jobs)}


# Gradio монтируется в это же приложение: /docs — API, /ui — интерфейс.
# Импорт мягкий: без gradio API всё равно поднимается.
try:
    import gradio as gr

    from app.ui import demo

    app = gr.mount_gradio_app(app, demo, path="/ui")
except ImportError:
    pass
