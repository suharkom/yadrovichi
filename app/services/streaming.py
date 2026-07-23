"""Потоковый пайплайн: реплики отдаются по мере готовности.

Отличие от AudioProcessingPipeline.process (батч, отдаёт результат целиком):
здесь результат течёт наружу генератором.

Схема, снимающая кажущийся конфликт с глобальной диаризацией:

  1. Диаризацию НЕ чанкуем — гоним весь файл, кластеры глобальны, спикеры
     не разъезжаются. Она быстрая (RTF ~0.03), несколько секунд.
  2. Роль преподавателя берём из ГРАФА переходов — он считается по одной
     диаризации, без текста. Значит роли известны ещё до ASR.
  3. Стримим ASR: faster-whisper отдаёт слова по мере декодирования, мы
     собираем их в реплики, вешаем роль и отдаём наружу.
  4. Постобработка реплики идёт в отдельном потоке параллельно с
     распознаванием следующей — GPU не простаивает, порядок сохранён.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from app.services import checkpoint
from app.services.alignment import assign_speaker_to_word, join_word_tokens
from app.services.audio import (
    get_audio_duration,
    normalize_audio,
    validate_audio_file,
)
from app.services.text_postprocessing import apply_text_replacements

# Граф и маппинг ролей считаем здесь, а не импортируем из role_detection:
# тот модуль сейчас активно правит МЛщик (граф-первичный фикс), имена функций
# в движении. Стриминг не должен ломаться от её рефакторинга.

MAX_GAP = 1.0             # пауза длиннее — новая реплика
MAX_UTTERANCE = 20.0      # реплика не длиннее — чтобы не копить бесконечно
UNKNOWN_ROLE = {"speaker_id": None, "role": "unknown", "display_name": "Неизвестный спикер"}


def graph_teacher(turns: list[dict[str, Any]]) -> str | None:
    """Преподаватель = спикер с максимумом уникальных соседей по таймлайну.
    Считается по одной диаризации, без текста — поэтому доступно до ASR."""
    sequence: list[str] = []
    for turn in sorted(turns, key=lambda t: t["start"]):
        speaker = str(turn["speaker"])
        if speaker == "UNKNOWN":
            continue
        if not sequence or sequence[-1] != speaker:
            sequence.append(speaker)

    neighbours: dict[str, set[str]] = defaultdict(set)
    for first, second in zip(sequence, sequence[1:]):
        if first != second:
            neighbours[first].add(second)
            neighbours[second].add(first)

    if neighbours:
        return max(neighbours, key=lambda s: len(neighbours[s]))
    speakers = {str(t["speaker"]) for t in turns if str(t["speaker"]) != "UNKNOWN"}
    return next(iter(speakers)) if speakers else None


def speaker_mapping(turns: list[dict[str, Any]], teacher: str | None) -> dict[str, dict]:
    """Преподавателю 0, остальным 1..N по первому появлению на таймлайне."""
    mapping: dict[str, dict] = {}
    if teacher is not None:
        mapping[teacher] = {"speaker_id": 0, "role": "teacher", "display_name": "Преподаватель"}
    next_id = 1
    for turn in sorted(turns, key=lambda t: t["start"]):
        speaker = str(turn["speaker"])
        if speaker == "UNKNOWN" or speaker in mapping:
            continue
        mapping[speaker] = {"speaker_id": next_id, "role": "student",
                            "display_name": f"Ученик {next_id}"}
        next_id += 1
    return mapping


def _breaks(current: dict[str, Any], word: dict[str, Any], speaker: str) -> bool:
    if speaker != current["speaker"]:
        return True
    if float(word["start"]) - float(current["end"]) > MAX_GAP:
        return True
    if float(word["end"]) - float(current["start"]) >= MAX_UTTERANCE:
        return True
    return False


def _finish(current: dict[str, Any], mapping: dict[str, dict], index: int) -> dict[str, Any]:
    """Собрать реплику: склейка слов, чистка ошибок ASR, роль. Чистая
    функция — гоняется в пуле параллельно с распознаванием следующей."""
    text = apply_text_replacements(join_word_tokens(current["words"]))
    role = mapping.get(current["speaker"], UNKNOWN_ROLE)
    return {
        "type": "utterance",
        "index": index,
        "start": round(current["start"], 3),
        "end": round(current["end"], 3),
        "source_speaker": current["speaker"],
        "text": text,
        **role,
    }


def _stream_words(asr_service, path: Path, settings) -> Iterator[dict[str, Any]]:
    """Слова из faster-whisper по мере декодирования. Генератор segments
    у CTranslate2 ленивый — GPU считает по ходу итерации, а не разом."""
    model = asr_service.model  # ленивая загрузка при первом обращении
    segments, _info = model.transcribe(
        str(path),
        language=settings.asr_language,
        beam_size=settings.asr_beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            token = word.word.strip()
            if not token:
                continue
            yield {"start": float(word.start), "end": float(word.end), "text": token}


def stream_pipeline(pipeline, audio_path, work_dir="data/work") -> Iterator[dict[str, Any]]:
    """Потоковый прогон. Отдаёт: meta → utterance* → done."""
    settings = pipeline.settings
    source = validate_audio_file(audio_path, max_file_size_mb=settings.max_file_size_mb)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    normalized = work / f"{source.stem}_16k_mono.wav"
    normalize_audio(source, normalized, settings.normalized_sample_rate)
    duration = get_audio_duration(normalized)

    started = time.perf_counter()

    # Потоковый чекпоинт: каждая отданная строка сразу дублируется на диск.
    # Оборвётся соединение — распознанное не пропадёт, лежит в stream.jsonl.
    ckpt = checkpoint.StreamCheckpoint(checkpoint.cache_key(source))

    def emit(item: dict[str, Any]) -> dict[str, Any]:
        ckpt.write(item)
        return item

    try:
        # 1-2. Диаризация целиком → роли из графа (без текста)
        diar = pipeline.diarization_service.diarize(normalized, duration)
        turns = diar["turns"]
        teacher = graph_teacher(turns)
        mapping = speaker_mapping(turns, teacher)

        yield emit({
            "type": "meta",
            "audio_seconds": round(duration, 2),
            "speaker_count": diar["speaker_count"],
            "teacher": teacher,
            "diarization_rtf": round(diar["rtf"], 4),
            "speaker_mapping": mapping,
        })

        # 3-4. Потоковый ASR + параллельная постобработка предыдущей реплики
        current: dict[str, Any] | None = None
        index = 0
        pending: deque[Future] = deque()

        with ThreadPoolExecutor(max_workers=1) as pool:
            for word in _stream_words(pipeline.asr_service, normalized, settings):
                speaker = assign_speaker_to_word(word, turns)
                if current is None or _breaks(current, word, speaker):
                    if current is not None:
                        pending.append(pool.submit(_finish, current, mapping, index))
                        index += 1
                        while pending and pending[0].done():
                            yield emit(pending.popleft().result())
                    current = {"start": word["start"], "end": word["end"],
                               "speaker": speaker, "words": [word["text"]]}
                else:
                    current["end"] = word["end"]
                    current["words"].append(word["text"])

            if current is not None:
                pending.append(pool.submit(_finish, current, mapping, index))
                index += 1

            for future in pending:
                yield emit(future.result())

        elapsed = time.perf_counter() - started
        yield emit({
            "type": "done",
            "utterance_count": index,
            "pipeline_rtf": round(elapsed / duration, 4) if duration else 0.0,
            "diarization_rtf": round(diar["rtf"], 4),
        })
    finally:
        ckpt.close()
