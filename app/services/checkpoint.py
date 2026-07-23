"""Чекпоинты — чтобы на длинном файле не терять уже сделанную работу.

Две задачи:

1. Кеш тяжёлых стадий (диаризация, ASR). Если пайплайн упал на поздней
   стадии (склейка, роли) или его перезапустили — не гоняем GPU заново,
   а поднимаем готовый результат с диска. Час аудио — это ~10 минут GPU,
   терять их обидно.

2. Потоковая запись реплик. stream_pipeline дописывает каждую реплику в
   .jsonl по мере готовности. Оборвётся соединение — распознанное уже
   лежит на диске, а не пропало.

Кеш стадий по умолчанию ВЫКЛЮЧЕН (`USE_CHECKPOINTS`), чтобы не портить
замеры RTF: иначе второй прогон поднимет стадию с диска и покажет RTF≈0.
Потоковая запись всегда включена — она почти ничего не стоит.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

CKPT_DIR = Path("data/checkpoints")


def enabled() -> bool:
    """Кеш стадий — только если явно включили. Замеры RTF не трогаем."""
    return os.getenv("USE_CHECKPOINTS", "").strip().lower() in {"1", "true", "yes", "on"}


def cache_key(source_path: str | Path) -> str:
    """Ключ по имени, размеру и времени файла — без чтения содержимого
    (для видео в гигабайты хеш содержимого был бы дорогим). Меняется
    файл — меняется ключ."""
    path = Path(source_path)
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_stage(key: str, stage: str) -> dict[str, Any] | None:
    """Поднять готовый результат стадии, если он есть в кеше."""
    if not enabled():
        return None
    path = CKPT_DIR / key / f"{stage}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_stage(key: str, stage: str, data: dict[str, Any]) -> None:
    """Сохранить результат стадии в кеш."""
    if not enabled():
        return
    path = CKPT_DIR / key / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class StreamCheckpoint:
    """Потоковая запись реплик в .jsonl по мере готовности.

    Всегда включена: обрыв стриминга не должен терять распознанное.
    Пишет и сбрасывает на диск каждую реплику сразу (flush).
    """

    def __init__(self, key: str) -> None:
        self.path = CKPT_DIR / key / "stream.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Новый прогон — начинаем файл заново.
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, item: dict[str, Any]) -> None:
        self._fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "StreamCheckpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_stream(key: str) -> Iterator[dict[str, Any]]:
    """Прочитать сохранённые реплики (для восстановления после обрыва)."""
    path = CKPT_DIR / key / "stream.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
