"""Стадии пайплайна: диаризация и транскрибация.

Реальные реализации на GPU. Заглушки остаются за флагом окружения
YADRO_FAKE=1 — на нём работают smoke.py и test_api.py без GPU и весов.

Порядок вызова задан в pipeline.py:
    diarize -> transcribe -> assign_roles
"""

from __future__ import annotations

import os
import random

from .schema import Segment, Word


def _fake() -> bool:
    return os.environ.get("YADRO_FAKE") == "1"


def diarize(audio_path: str, min_speakers: int = 2, max_speakers: int | None = None) -> list[Segment]:
    """pyannote по всему файлу целиком -> сегменты со спикерами.

    Кластеризация обязательно одна на всю запись, не почанково, иначе
    спикеры переименуются на середине. max_speakers по умолчанию не
    задаём: при ролях "ученик 1, 2, 3" верхняя граница мешает.

    Возвращает сегменты, отсортированные по start.
    """
    if _fake():
        return _fake_segments()

    import torch

    from .models import get_diarizer, load_audio

    pipe = get_diarizer()
    kwargs: dict[str, int] = {"min_speakers": min_speakers}
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    # Кормим готовый waveform, а не путь к файлу: pyannote 4 иначе читает
    # аудио через torchcodec, которому на этой машине не хватает libnvrtc.
    # soundfile уже загрузил тот же 16 кГц моно, лишнего чтения нет.
    data, sample_rate = load_audio(audio_path)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (channel=1, time)
    output = pipe({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    # В pyannote 4 exclusive-режим оставляет одного спикера в каждый момент —
    # это убирает конфликты на наложениях при привязке к словам.
    annotation = getattr(output, "exclusive_speaker_diarization",
                         getattr(output, "speaker_diarization", output))

    segments = [
        Segment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s.start)
    return segments


def transcribe(audio_path: str, segments: list[Segment]) -> list[Segment]:
    """faster-whisper turbo float16 -> текст с таймкодами.

    Транскрибирует речевой участок переданных сегментов (в pipeline это
    один чанк одного спикера) и раскладывает слова обратно по сегментам
    по перекрытию интервалов.

    Обязательные параметры:
        language="ru"                    жёстко: на тишине whisper иначе
                                         переключает язык
        condition_on_previous_text=False на длинных файлах иначе зацикливается
        word_timestamps=True             нужны для привязки к спикерам
        beam_size=5                      у turbo декодер 4 слоя, beam дёшев
    """
    if not segments:
        return segments
    if _fake():
        for seg in segments:
            seg.text = f"[заглушка транскрибации {seg.start:.1f}-{seg.end:.1f}]"
            seg.lang = "ru"
            seg.words = [Word(seg.start, seg.end, seg.text)]
        return segments

    from . import postprocess
    from .models import get_asr, load_audio

    audio_data, sample_rate = load_audio(audio_path)
    region_start = min(s.start for s in segments)
    region_end = max(s.end for s in segments)
    lo = max(0, int(region_start * sample_rate))
    hi = min(len(audio_data), int(region_end * sample_rate))
    clip = audio_data[lo:hi]

    model = get_asr()
    seg_gen, info = model.transcribe(
        clip,
        language="ru",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=postprocess.initial_prompt() or None,
    )

    # Таймкоды слов относительны началу clip — сдвигаем к абсолютным.
    words: list[Word] = []
    for s in seg_gen:
        for w in (s.words or []):
            if w.start is None or w.end is None:
                continue
            words.append(Word(start=w.start + region_start,
                              end=w.end + region_start,
                              text=w.word))

    _attach_words(segments, words)
    for seg in segments:
        seg.lang = info.language

    # Сегменты, которым не досталось ни слова (слова ушли в соседний по
    # перекрытию), — это пустые реплики в таймлайне. Убираем.
    return [seg for seg in segments if seg.text]


def assign_roles(segments: list[Segment]) -> list[Segment]:
    """Определение ролей.

    Порядок, о котором договорились:
      1. граф переходов по таймлайну — у преподавателя больше всего
         разных собеседников (работает без текста, не зависит от предмета)
      2. словарь маркеров, нормированный на 1000 слов кластера
      3. на двух спикерах или при расхождении — решают маркеры

    Преподаватель получает role=0, остальные нумеруются по первому
    появлению на таймлайне. Отсчёт заново в каждой записи.
    """
    from .roles import find_teacher, number_speakers

    teacher = find_teacher(segments)
    return number_speakers(segments, teacher)


def _attach_words(segments: list[Segment], words: list[Word]) -> None:
    """Каждое слово — к сегменту с максимальным перекрытием, иначе к
    ближайшему по середине слова. Заодно собирает текст сегмента."""
    if not segments:
        return
    for seg in segments:
        seg.words = []

    for word in words:
        best, best_overlap = None, 0.0
        for seg in segments:
            ov = min(word.end, seg.end) - max(word.start, seg.start)
            if ov > best_overlap:
                best, best_overlap = seg, ov
        if best is None:
            mid = (word.start + word.end) / 2
            best = min(segments, key=lambda s: min(abs(mid - s.start), abs(mid - s.end)))
        best.words.append(word)

    for seg in segments:
        seg.text = " ".join(w.text for w in seg.words).strip()


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
