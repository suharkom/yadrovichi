"""Прогон пайплайна на заглушках — без GPU, без весов, без ffmpeg.

Проверяет, что контракт данных сходится и стадии стыкуются:
диаризация -> нарезка на чанки -> транскрибация -> роли.
Запуск: python -m scripts.smoke
"""

from src import chunking, stages


def main() -> None:
    segments = stages.diarize("fake.wav")

    chunks = chunking.build(segments)
    print(f"Нарезка: {chunking.stats(chunks)}")
    for chunk in chunks[:4]:
        print(f"  #{chunk.index} {chunk.start:7.2f}-{chunk.end:7.2f} "
              f"({chunk.duration:5.1f}s) {chunk.speaker}")

    segments = stages.transcribe("fake.wav", segments)
    segments = stages.assign_roles(segments)

    print("\nРезультат:")
    for seg in segments:
        print(f"  [{seg.start:7.2f} - {seg.end:7.2f}] {seg.role_name:>12} | {seg.text}")

    assert all(s.text for s in segments), "не у всех сегментов есть текст"
    print(f"\nOK: сегментов {len(segments)}, стадии стыкуются")


if __name__ == "__main__":
    main()
