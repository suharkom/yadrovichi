from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline

from app.core.config import Settings


class DiarizationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        if not settings.hf_token:
            raise ValueError(
                "Не задан HF_TOKEN. Добавь настоящий токен в файл .env."
            )

        if settings.hf_token == "your_huggingface_token_here":
            raise ValueError(
                "В .env всё ещё стоит пример HF_TOKEN. "
                "Замени его на настоящий токен Hugging Face."
            )

        print(
            "Загрузка модели диаризации "
            f"{settings.diarization_model_name}..."
        )

        self.pipeline = Pipeline.from_pretrained(
            settings.diarization_model_name,
            token=settings.hf_token,
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Не удалось загрузить модель pyannote. "
                "Проверь HF_TOKEN и принятие условий модели."
            )

        self._on_gpu = False
        if settings.asr_device == "cuda":
            self.pipeline.to(torch.device("cuda"))
            self._on_gpu = True

        print("Модель диаризации загружена.")

    def offload(self) -> None:
        """Снять модель с GPU и освободить VRAM. Стадии последовательны:
        во время транскрибации диаризатор не нужен, а на общей карте эти
        ~2 ГБ решают, влезет ли whisper. Перед следующей диаризацией
        модель сама вернётся на GPU."""
        if self._on_gpu:
            self.pipeline.to(torch.device("cpu"))
            self._on_gpu = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def diarize(
        self,
        audio_path: str | Path,
        audio_duration: float,
    ) -> dict[str, Any]:
        if audio_duration <= 0:
            raise ValueError("Длительность аудио должна быть положительной.")

        # Вернуть модель на GPU, если её выгружали после прошлого прогона.
        if self.settings.asr_device == "cuda" and not self._on_gpu:
            self.pipeline.to(torch.device("cuda"))
            self._on_gpu = True

        parameters: dict[str, Any] = {
            "min_speakers": self.settings.min_speakers,
        }

        if self.settings.max_speakers is not None:
            parameters["max_speakers"] = (
                self.settings.max_speakers
            )

        # Кормим pyannote готовый waveform, а не путь к файлу: на сервере
        # pyannote 4 иначе читает аудио через torchcodec, которому не хватает
        # libnvrtc. soundfile читает тот же 16 кГц моно без этой зависимости.
        import soundfile as sf

        samples, sample_rate = sf.read(str(audio_path), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(samples).unsqueeze(0)  # (channel=1, time)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        started_at = time.perf_counter()

        output = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **parameters,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - started_at
        rtf = elapsed_seconds / audio_duration

        annotation = getattr(
            output,
            "speaker_diarization",
            output,
        )

        if not hasattr(annotation, "itertracks"):
            raise RuntimeError(
                "Неожиданный формат результата pyannote: "
                "нет метода itertracks."
            )

        turns: list[dict[str, Any]] = []

        for turn, _, speaker in annotation.itertracks(
            yield_label=True
        ):
            turns.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )

        turns.sort(
            key=lambda item: (
                item["start"],
                item["end"],
            )
        )

        speakers = sorted(
            {turn["speaker"] for turn in turns}
        )

        return {
            "turns": turns,
            "speakers": speakers,
            "speaker_count": len(speakers),
            "elapsed_seconds": elapsed_seconds,
            "rtf": rtf,
        }
