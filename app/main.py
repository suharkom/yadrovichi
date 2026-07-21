"""Локальный бэкенд.

Архитектура, которую просил куратор:
    приём файла (до 200 МБ) -> нарезка на чанки -> очередь обработки

Три решения, которые здесь зафиксированы:

1. Файл пишется на диск потоком по 1 МБ, а не читается в память целиком.
   200 МБ в RAM на каждый запрос — это то, на чём падают при демо.

2. Очередь на один воркер. Видеокарта одна, второй процесс просто не
   влезет в 8 GB. Uvicorn запускать строго с --workers 1, параллелизм
   делается очередью внутри приложения, а не размножением процессов.

3. Результат отдаётся потоком NDJSON. На часовом файле обычный
   JSON-ответ не доживёт: обработка 15-25 минут, запрос оборвётся.
   Финальный цельный JSON остаётся отдельной ручкой для скачивания.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src import pipeline

MAX_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
UPLOAD_DIR = Path("data/uploads")
RESULT_DIR = Path("data/results")

# Одна задача на GPU за раз. Остальные ждут в очереди.
gpu_lock = asyncio.Semaphore(1)
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Модели грузятся один раз при старте, а не в обработчике запроса.

    Иначе каждый запрос заново поднимает веса на GPU — минуты на пустом месте.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # TODO(МЛщик): здесь прогрев — загрузить whisper и pyannote в память.
    # Прогрев в RTF не считается, он отдельно.
    yield


app = FastAPI(title="yadrovichi", lifespan=lifespan)


async def save_upload(upload: UploadFile) -> Path:
    """Потоковая запись на диск с проверкой размера по ходу."""
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
    """Основная ручка: файл на вход, поток реплик на выход.

    Каждая строка ответа — отдельный JSON-объект (NDJSON). Клиент может
    рисовать их по мере поступления, не дожидаясь конца обработки.
    """
    src = await save_upload(file)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "source": str(src)}

    async def stream():
        yield json.dumps({"job_id": job_id, "status": "queued"},
                         ensure_ascii=False) + "\n"

        async with gpu_lock:
            jobs[job_id]["status"] = "running"
            collected = []
            loop = asyncio.get_running_loop()

            # Пайплайн синхронный и грузит CPU/GPU — уводим в пул потоков,
            # чтобы не блокировать event loop и остальные запросы.
            gen = pipeline.run(src)
            while True:
                segment = await loop.run_in_executor(None, lambda: next(gen, None))
                if segment is None:
                    break
                collected.append(segment.to_dict())
                yield json.dumps(segment.to_dict(), ensure_ascii=False) + "\n"

            result_path = RESULT_DIR / f"{job_id}.json"
            result_path.write_text(
                json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            jobs[job_id] = {"status": "done", "result": str(result_path)}
            yield json.dumps({"job_id": job_id, "status": "done"},
                             ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return jobs[job_id]


@app.get("/result/{job_id}")
async def result(job_id: str):
    """Финальный цельный JSON — для скачивания и для метрик."""
    path = RESULT_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "Результат ещё не готов")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "queued": len(jobs)}
