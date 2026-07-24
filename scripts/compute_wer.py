"""WER — доля ошибок распознавания против эталонного транскрипта.

⚠️ Эталон должен быть ЧЕЛОВЕЧЕСКИМ транскриптом (выверенные субтитры).
Авто-субтитры YouTube — это тоже ASR, сравнение с ними меряет согласие
двух моделей, а не точность нашей. Такой WER на защите не заявляем.

Оба текста нормализуются одинаково (нижний регистр, ё→е, снятие пунктуации),
иначе получите завышенный WER на ровном месте.

Запуск:
    python -m scripts.compute_wer --audio data/lecture.m4a --reference data/lecture.ru.srt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def strip_subtitles(raw: str) -> str:
    """Из .srt/.vtt вытащить только текст: убрать номера, таймкоды, теги."""
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.isdigit() or "-->" in s or s.upper().startswith("WEBVTT"):
            continue
        s = re.sub(r"<[^>]+>", "", s)          # <c>, <00:00:01.000> и прочие теги
        s = re.sub(r"\{[^}]*\}", "", s)        # {\an8} и стили
        if s:
            lines.append(s)
    return " ".join(lines)


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="аудио/видео файл")
    parser.add_argument("--reference", required=True, help=".srt/.vtt/.txt эталон")
    args = parser.parse_args()

    import jiwer

    from app.core.config import load_settings
    from app.services.asr import ASRService
    from app.services.audio import get_audio_duration, normalize_audio

    ref_path = Path(args.reference)
    raw = ref_path.read_text(encoding="utf-8", errors="ignore")
    reference = normalize(
        strip_subtitles(raw) if ref_path.suffix.lower() in {".srt", ".vtt"} else raw
    )
    if not reference:
        raise SystemExit("Эталон пуст после разбора — проверь файл субтитров.")

    settings = load_settings()
    work = Path("data/work")
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{Path(args.audio).stem}_16k_mono.wav"
    normalize_audio(args.audio, wav, settings.normalized_sample_rate)
    duration = get_audio_duration(wav)

    print(f"Аудио: {duration / 60:.1f} мин, транскрибирую...")
    asr = ASRService(settings)
    result = asr.transcribe(wav, duration)
    hypothesis = normalize(" ".join(seg["text"] for seg in result["segments"]))

    wer = jiwer.wer(reference, hypothesis)
    print()
    print(f"Слов в эталоне:  {len(reference.split())}")
    print(f"Слов у нас:      {len(hypothesis.split())}")
    print(f"ASR RTF:         {result['rtf']:.3f}")
    print(f"WER:             {wer * 100:.1f}%")


if __name__ == "__main__":
    main()
