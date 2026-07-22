"""Ленивая загрузка тяжёлых моделей — по одному экземпляру на процесс.

Модели поднимаются один раз при первом обращении и держатся в памяти.
В FastAPI это дёргается на старте через lifespan, а не в обработчике
запроса, иначе каждый запрос заново грузит веса на GPU.

Целевое железо — RTX A4000 (Ampere, 16 ГБ). Обе модели (~2 ГБ whisper +
~2 ГБ pyannote) помещаются на карте одновременно, поэтому держим их рядом,
а не выгружаем между стадиями.
"""

from __future__ import annotations

import functools
import threading

ASR_MODEL = "large-v3-turbo"
ASR_COMPUTE_TYPE = "float16"   # Ampere: fp16 быстрее fp32. int8 — только на Pascal
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

_lock = threading.Lock()
_asr = None
_diarizer = None


def get_asr():
    """faster-whisper turbo, float16, CUDA. Синглтон."""
    global _asr
    if _asr is None:
        with _lock:
            if _asr is None:
                from faster_whisper import WhisperModel
                _asr = WhisperModel(ASR_MODEL, device="cuda", compute_type=ASR_COMPUTE_TYPE)
    return _asr


def get_diarizer():
    """pyannote community-1 на CUDA. Синглтон.

    from_pretrained молча возвращает None, если не приняты условия модели
    или у токена нет права read — ловим это здесь, иначе ошибка вылезет
    ниже как AttributeError на None, и причину не найти.
    """
    global _diarizer
    if _diarizer is None:
        with _lock:
            if _diarizer is None:
                import torch
                from pyannote.audio import Pipeline
                pipe = Pipeline.from_pretrained(DIARIZATION_MODEL)
                if pipe is None:
                    raise RuntimeError(
                        f"pyannote вернул None для {DIARIZATION_MODEL}. Проверь под тем "
                        "же аккаунтом, чей токен: приняты ли условия на "
                        "speaker-diarization-community-1 и segmentation-3.0, "
                        "и что у токена право read."
                    )
                if torch.cuda.is_available():
                    pipe.to(torch.device("cuda"))
                _diarizer = pipe
    return _diarizer


@functools.lru_cache(maxsize=2)
def load_audio(path: str):
    """Прочитать 16 кГц моно wav в float32-массив. Кэш на пару файлов,
    чтобы не перечитывать один и тот же файл для каждого чанка."""
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sample_rate


def warmup() -> None:
    """Поднять обе модели заранее. Вызывать на старте FastAPI и перед
    замером RTF — первый прогон компилирует кернелы и в RTF не считается."""
    get_asr()
    get_diarizer()
