"""Заглушки стадий пайплайна — реализует МЛщик.

Сигнатуры зафиксированы, их менять нельзя: на них опирается pipeline.py,
API и тесты. МЛщик заменяет тела функций настоящими моделями, не трогая
интерфейс (вход/выход — list[Segment] из schema.py).

Пока внутри фейковые данные — это позволяет прогонять пайплайн, API и
интерфейс целиком, не дожидаясь, пока встанут pyannote и whisper.

Порядок вызова задан в pipeline.py:
    diarize -> transcribe -> assign_roles

Что должен сделать МЛщик в каждой функции — см. докстроки ниже.
"""

from __future__ import annotations

import random

from .schema import Segment, Word


def diarize(audio_path: str, min_speakers: int = 2, max_speakers: int | None = None) -> list[Segment]:
    """pyannote по всему файлу целиком -> сегменты со спикерами.

    Кластеризация обязательно одна на всю запись, не почанково, иначе
    спикеры переименуются на середине. max_speakers по умолчанию не
    задаём: при ролях "ученик 1, 2, 3" верхняя граница мешает.

    Возвращает Segment со заполненными start, end, speaker
    (метки вида SPEAKER_00), отсортированные по start.
    """
    return _fake_segments()


def transcribe(audio_path: str, segments: list[Segment]) -> list[Segment]:
    """faster-whisper -> текст с таймкодами.

    Заполняет segment.text, segment.words и segment.lang. Границы речи
    берутся из диаризации, чтобы не гонять VAD дважды и не транскрибировать
    тишину.

    Рекомендованные параметры:
        language="ru"                    жёстко, без автоопределения
        condition_on_previous_text=False на длинных файлах иначе зацикливается
        word_timestamps=True             нужны для привязки к спикерам
        beam_size=5                      у turbo декодер 4 слоя, beam дёшев
    """
    for seg in segments:
        seg.text = f"[заглушка транскрибации {seg.start:.1f}-{seg.end:.1f}]"
        seg.lang = "ru"
        seg.words = [Word(seg.start, seg.end, seg.text)]
    return segments


def assign_roles(segments: list[Segment]) -> list[Segment]:
    """Определение ролей — реализует МЛщик.

    Договорённый порядок:
      1. граф переходов по таймлайну — у преподавателя больше всего
         разных собеседников (работает без текста, не зависит от предмета)
      2. словарь маркеров, нормированный на 1000 слов кластера
      3. на двух спикерах или при расхождении — решают маркеры

    Преподаватель получает role=0, остальные нумеруются по первому
    появлению на таймлайне. Отсчёт заново в каждой записи.

    Пока заглушка: роль не определяется, в role_name кладём метку спикера,
    чтобы пайплайн, API и интерфейс проходили целиком. МЛщик заменяет тело.
    """
    for seg in segments:
        seg.role = None
        seg.role_name = seg.speaker
    return segments


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
