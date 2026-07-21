"""Прогон пайплайна на заглушках — без GPU, без весов, без ffmpeg.

Проверяет, что контракт данных сходится и стадии стыкуются.
Запуск: python -m scripts.smoke
"""

from src import stages
from src.roles import neighbour_counts


def main() -> None:
    segments = stages.diarize("fake.wav")
    segments = stages.transcribe("fake.wav", segments)

    print("Соседи по таймлайну:")
    for speaker, neighbours in sorted(neighbour_counts(segments).items()):
        print(f"  {speaker}: {len(neighbours)} собеседник(ов) -> {sorted(neighbours)}")

    segments = stages.assign_roles(segments)

    print("\nРезультат:")
    for seg in segments:
        print(f"  [{seg.start:7.2f} - {seg.end:7.2f}] {seg.role_name:>14} | {seg.text}")

    teacher = [s for s in segments if s.role == 0]
    assert teacher, "преподаватель не определился"
    print(f"\nOK: преподаватель — {teacher[0].speaker}, "
          f"ролей всего {len({s.role for s in segments})}")


if __name__ == "__main__":
    main()
