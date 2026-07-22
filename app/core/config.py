from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv


load_dotenv()


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _bool_from_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        "Ожидалось логическое значение "
        f"(true/false), получено: {value!r}"
    )


@dataclass(frozen=True)
class Settings:
    hf_token: str | None

    asr_model_name: str
    asr_device: str
    asr_compute_type: str
    asr_beam_size: int
    asr_language: str | None

    diarization_model_name: str
    min_speakers: int
    max_speakers: int | None
    parallel_gpu_stages: bool

    normalized_sample_rate: int = 16_000
    max_file_size_mb: int = 200


def load_settings() -> Settings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_compute_type = "float16" if device == "cuda" else "int8"

    language = os.getenv("ASR_LANGUAGE", "ru").strip()

    return Settings(
        hf_token=os.getenv("HF_TOKEN"),
        asr_model_name=os.getenv(
            "ASR_MODEL_NAME",
            "large-v3-turbo",
        ),
        asr_device=device,
        asr_compute_type=os.getenv(
            "ASR_COMPUTE_TYPE",
            default_compute_type,
        ),
        asr_beam_size=int(
            os.getenv("ASR_BEAM_SIZE", "2")
        ),
        asr_language=language or None,
        diarization_model_name=os.getenv(
            "DIARIZATION_MODEL_NAME",
            "pyannote/speaker-diarization-community-1",
        ),
        min_speakers=int(
            os.getenv("MIN_SPEAKERS", "2")
        ),
        max_speakers=_optional_int(
            os.getenv("MAX_SPEAKERS")
        ),
        parallel_gpu_stages=_bool_from_env(
            os.getenv("PARALLEL_GPU_STAGES"),
            default=True,
        ),
    )
