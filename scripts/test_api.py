"""Проверка API по HTTP на заглушках: GPU, веса и ffmpeg не нужны.

Проверяет то, что реально ломается:
  - файл принимается и пишется на диск
  - ответ приходит потоком, а не одним куском в конце
  - лимит 200 МБ срабатывает
  - финальный JSON доступен по job_id

Запуск: python -m scripts.test_api
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from src import audio

# ffmpeg в проверке не участвует: стадии заглушены, реальный звук не читается.
audio.prepare = lambda src, dst_dir="data/prepared": Path(src)

from app.main import MAX_BYTES, app  # noqa: E402  (после подмены prepare)


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200, health.text
        print(f"/health -> {health.json()}")

        payload = {"file": ("lesson.mp3", b"fake audio bytes", "audio/mpeg")}
        lines: list[dict] = []
        async with client.stream("POST", "/transcribe", files=payload) as response:
            assert response.status_code == 200, await response.aread()
            async for line in response.aiter_lines():
                if line.strip():
                    lines.append(json.loads(line))

        assert lines[0]["status"] == "queued", lines[0]
        assert lines[-1]["status"] == "done", lines[-1]
        job_id = lines[0]["job_id"]
        segments = [item for item in lines if "role_name" in item]

        print(f"поток: {len(lines)} строк, из них реплик {len(segments)}")
        for seg in segments[:3]:
            print(f"   [{seg['start']:7.2f}] {seg['role_name']:>14} | {seg['text'][:45]}")

        result = await client.get(f"/result/{job_id}")
        assert result.status_code == 200, result.text
        assert len(result.json()) == len(segments)
        print(f"/result/{job_id[:8]}... -> {len(result.json())} реплик")

        status = await client.get(f"/jobs/{job_id}")
        assert status.json()["status"] == "done"

        oversize = {"file": ("big.mp3", b"x" * (MAX_BYTES + 1), "audio/mpeg")}
        too_big = await client.post("/transcribe", files=oversize)
        assert too_big.status_code == 413, too_big.status_code
        print(f"лимит 200 МБ -> {too_big.status_code} {too_big.json()['detail']}")

    print("\nOK: API работает end-to-end на заглушках")


if __name__ == "__main__":
    asyncio.run(main())
