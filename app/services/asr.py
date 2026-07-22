from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from app.core.config import Settings


class ASRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        """Ленивая загрузка: CTranslate2 трогает CUDA только при первой
        транскрибации. Это важно на сервере — если faster-whisper
        инициализирует CUDA раньше pyannote, NVML внутри torch падает
        ассертом. Поэтому диаризация должна отработать первой."""
        if self._model is None:
            from faster_whisper import WhisperModel

            print(
                f"Загрузка ASR-модели {self.settings.asr_model_name} "
                f"на {self.settings.asr_device}..."
            )
            self._model = WhisperModel(
                self.settings.asr_model_name,
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
            )
            print("ASR-модель загружена.")
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        audio_duration: float,
    ) -> dict[str, Any]:
        if audio_duration <= 0:
            raise ValueError("Длительность аудио должна быть положительной.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        started_at = time.perf_counter()

        segments_generator, info = self.model.transcribe(
            str(audio_path),
            language=self.settings.asr_language,
            beam_size=self.settings.asr_beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        raw_segments = list(segments_generator)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - started_at
        rtf = elapsed_seconds / audio_duration

        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for segment in raw_segments:
            text = segment.text.strip()

            if text:
                segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                    }
                )

            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue

                clean_word = word.word.strip()

                if not clean_word:
                    continue

                probability = getattr(word, "probability", None)

                words.append(
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "text": clean_word,
                        "probability": (
                            float(probability)
                            if probability is not None
                            else None
                        ),
                    }
                )

        language_probability = getattr(
            info,
            "language_probability",
            None,
        )

        return {
            "language": info.language,
            "language_probability": (
                float(language_probability)
                if language_probability is not None
                else None
            ),
            "segments": segments,
            "words": words,
            "elapsed_seconds": elapsed_seconds,
            "rtf": rtf,
        }
