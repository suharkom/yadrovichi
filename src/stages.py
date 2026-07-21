"""Заглушки стадий пайплайна.

Сигнатуры зафиксированы — их реализует МЛщик, не меняя интерфейс.
Пока внутри фейковые данные: это позволяет прогонять пайплайн целиком,
не дожидаясь, пока скачаются веса pyannote и whisper.

Порядок вызова задан в pipeline.py:
    diarize -> transcribe -> assign_roles
"""

from __future__ import annotations

import random

from .schema import Segment, Word


def diarize(audio_path: str, min_speakers: int = 2, max_speakers: int | None = None) -> list[Segment]:
    """pyannote по всему файлу целиком -> сегменты со спикерами.

    Кластеризация обязательно одна на всю запись, не почанково,
    иначе спикеры переименуются на середине.

    Возвращает сегменты, отсортированные по start.
    """
    return _fake_segments()


def transcribe(audio_path: str, segments: list[Segment]) -> list[Segment]:
    """faster-whisper turbo int8 по речевым участкам -> текст с таймкодами.

    Заполняет segment.text, segment.words и segment.lang. Границы речи
    берутся из диаризации, чтобы не гонять VAD дважды и не транскрибировать
    тишину.

    Обязательные параметры:
        compute_type="int8"              на Pascal float16 медленнее fp32
        language="ru"                    ⚠️ жёстко, без автоопределения:
                                         на тишине whisper переключает язык
        condition_on_previous_text=False на часовых файлах иначе зацикливается
        word_timestamps=True             нужны для привязки к спикерам
        beam_size=5                      у turbo декодер 4 слоя, beam дёшев
        initial_prompt=postprocess.initial_prompt()
    """
    for seg in segments:
        seg.text = f"[заглушка транскрибации {seg.start:.1f}-{seg.end:.1f}]"
        seg.lang = "ru"
        seg.words = [Word(seg.start, seg.end, seg.text)]
    return segments


def assign_roles(segments: list[Segment]) -> list[Segment]:
    """Определение ролей.

    Порядок, о котором договорились:
      1. граф переходов по таймлайну — у преподавателя больше всего
         разных собеседников (работает без текста, не зависит от предмета)
      2. словарь маркеров, нормированный на 1000 слов кластера
      3. LLM только если первые два разошлись или спикеров всего два

    Преподаватель получает role=0, остальные нумеруются по первому
    появлению на таймлайне. Отсчёт заново в каждой записи.
    """
    from .roles import number_speakers, find_teacher

    teacher = find_teacher(segments)
    return number_speakers(segments, teacher)


def _fake_segments() -> list[Segment]:
    """Правдоподобная последовательность: ведущий переключает остальных."""
    random.seed(0)
    pattern = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_02",
               "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_02"]
    segments, t = [], 0.0
    for speaker in pattern:
        dur = random.uniform(3.0, 25.0)
        segments.append(Segment(start=round(t, 2), end=round(t + dur, 2), speaker=speaker))
        t += dur + random.uniform(0.2, 1.5)
    return segments
