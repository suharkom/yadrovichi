from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services import checkpoint
from app.services.alignment import (
    assign_speakers_to_words,
    build_utterances,
    smooth_short_speaker_turns,
)
from app.services.asr import ASRService
from app.services.audio import (
    get_audio_duration,
    normalize_audio,
    validate_audio_file,
)
from app.services.diarization import DiarizationService
from app.services.mathnorm import annotate_item
from app.services.role_detection import (
    apply_roles,
    create_speaker_mapping,
    detect_teacher,
)
from app.services.text_postprocessing import (
    postprocess_asr_result,
)


class AudioProcessingPipeline:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        # Порядок важен: pyannote (torch) грузится ПЕРВЫМ и инициализирует
        # CUDA/NVML корректно. faster-whisper (CTranslate2) — ленивый, трогает
        # CUDA только при первой транскрибации, уже после диаризации. Если
        # CTranslate2 инициализирует CUDA раньше pyannote, NVML внутри torch
        # падает ассертом на forward pyannote.
        self.diarization_service = DiarizationService(settings)
        self.asr_service = ASRService(settings)

    def process(
        self,
        audio_path: str | Path,
        work_dir: str | Path = "artifacts",
    ) -> dict[str, Any]:
        pipeline_started_at = time.perf_counter()

        source_path = validate_audio_file(
            audio_path,
            max_file_size_mb=self.settings.max_file_size_mb,
        )

        work_directory = Path(work_dir)
        work_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalized_path = (
            work_directory
            / f"{source_path.stem}_16k_mono.wav"
        )

        print("Нормализация аудио...")

        normalize_audio(
            input_path=source_path,
            output_path=normalized_path,
            sample_rate=self.settings.normalized_sample_rate,
        )

        audio_duration = get_audio_duration(
            normalized_path
        )

        print(
            f"Аудио: {audio_duration:.1f} секунд, "
            f"файл: {normalized_path}"
        )

        run_in_parallel = (
            self.settings.asr_device == "cuda"
            and self.settings.parallel_gpu_stages
        )

        if run_in_parallel:
            print("Запускаю ASR и диаризацию параллельно...")
            stages_started_at = time.perf_counter()

            with ThreadPoolExecutor(max_workers=2) as executor:
                asr_future = executor.submit(
                    self.asr_service.transcribe,
                    normalized_path,
                    audio_duration,
                )
                diarization_future = executor.submit(
                    self.diarization_service.diarize,
                    normalized_path,
                    audio_duration,
                )
                asr_result = asr_future.result()
                diarization_result = diarization_future.result()

            stages_elapsed_seconds = (
                time.perf_counter() - stages_started_at
            )
        else:
            # Диаризация ПЕРВОЙ: pyannote (torch) инициализирует CUDA до
            # того, как faster-whisper (CTranslate2) её тронет, иначе NVML
            # внутри torch падает ассертом.
            # Каждую стадию кешируем на диск: перезапуск не гоняет GPU
            # заново (кеш включается USE_CHECKPOINTS, замеры RTF не трогает).
            key = checkpoint.cache_key(source_path)
            stages_started_at = time.perf_counter()

            diarization_result = checkpoint.load_stage(key, "diarization")
            if diarization_result is None:
                print("Запускаю диаризацию...")
                diarization_result = self.diarization_service.diarize(
                    audio_path=normalized_path,
                    audio_duration=audio_duration,
                )
                checkpoint.save_stage(key, "diarization", diarization_result)
            else:
                print("Диаризация — из чекпоинта.")

            asr_result = checkpoint.load_stage(key, "asr")
            if asr_result is None:
                print("Запускаю транскрибацию...")
                asr_result = self.asr_service.transcribe(
                    audio_path=normalized_path,
                    audio_duration=audio_duration,
                )
                checkpoint.save_stage(key, "asr", asr_result)
            else:
                print("Транскрибация — из чекпоинта.")

            stages_elapsed_seconds = (
                time.perf_counter() - stages_started_at
            )

        asr_result = postprocess_asr_result(asr_result)

        print(
            f"ASR завершён: "
            f"{asr_result['elapsed_seconds']:.1f} с, "
            f"RTF={asr_result['rtf']:.3f}"
        )

        print(
            f"Диаризация завершена: "
            f"{diarization_result['elapsed_seconds']:.1f} с, "
            f"RTF={diarization_result['rtf']:.3f}, "
            f"спикеров: "
            f"{diarization_result['speaker_count']}"
        )

        assigned_words = assign_speakers_to_words(
            words=asr_result["words"],
            speaker_turns=diarization_result["turns"],
        )

        raw_utterances = build_utterances(
            words=assigned_words,
            max_gap_seconds=1.0,
            max_utterance_duration=20.0,
        )

        utterances = smooth_short_speaker_turns(
            raw_utterances,
            max_duration=2.0,
            max_words=3,
            max_neighbor_gap=0.5,
        )

        if not utterances:
            raise RuntimeError(
                "После объединения ASR и диаризации "
                "не получилось ни одной реплики."
            )

        role_result = detect_teacher(
            utterances
        )

        speaker_mapping = create_speaker_mapping(
            utterances=utterances,
            teacher_speaker=role_result[
                "teacher_speaker"
            ],
        )

        timeline = apply_roles(
            utterances=utterances,
            speaker_mapping=speaker_mapping,
        )
        timeline = [
            annotate_item(item)
            for item in timeline
        ]

        pipeline_elapsed_seconds = (
            time.perf_counter() - pipeline_started_at
        )

        pipeline_rtf = (
            pipeline_elapsed_seconds / audio_duration
        )

        unknown_word_count = sum(
            word["speaker"] == "UNKNOWN"
            for word in assigned_words
        )

        return {
            "source_file": source_path.name,
            "normalized_file": normalized_path.name,
            "audio_duration_seconds": audio_duration,
            "language": asr_result["language"],
            "language_probability": asr_result[
                "language_probability"
            ],
            "processing_parameters": {
                "max_utterance_duration_seconds": 20.0,
                "max_gap_seconds": 1.0,
                "speaker_smoothing_enabled": True,
                "speaker_smoothing_max_duration_seconds": 2.0,
                "speaker_smoothing_max_words": 3,
                "speaker_smoothing_max_neighbor_gap_seconds": 0.5,
                "text_replacements_enabled": True,
                "stage_execution_mode": (
                    "parallel" if run_in_parallel else "sequential"
                ),
            },
            "metrics": {
                "asr_elapsed_seconds": asr_result[
                    "elapsed_seconds"
                ],
                "asr_rtf": asr_result["rtf"],
                "diarization_elapsed_seconds": (
                    diarization_result[
                        "elapsed_seconds"
                    ]
                ),
                "diarization_rtf": diarization_result[
                    "rtf"
                ],
                "stages_elapsed_seconds": stages_elapsed_seconds,
                "stages_rtf": (
                    stages_elapsed_seconds / audio_duration
                ),
                "pipeline_elapsed_seconds": (
                    pipeline_elapsed_seconds
                ),
                "pipeline_rtf": pipeline_rtf,
                "unknown_word_count": unknown_word_count,
                "total_word_count": len(assigned_words),
                "raw_utterance_count": len(raw_utterances),
                "final_utterance_count": len(
                    utterances
                ),
                "smoothed_fragment_count": sum(
                    len(item.get("smoothed_fragments", []))
                    + (1 if "smoothed_from_speaker" in item else 0)
                    for item in utterances
                ),
            },
            "speaker_count": diarization_result[
                "speaker_count"
            ],
            "speaker_mapping": speaker_mapping,
            "role_detection": role_result,
            "asr_segments": asr_result["segments"],
            "diarization_turns": diarization_result[
                "turns"
            ],
            "timeline": timeline,
        }
