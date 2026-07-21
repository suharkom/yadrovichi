"""Контракт данных проекта.

Всё, что летает между стадиями пайплайна, — это list[Segment].
Менять поля только по договорённости всей командой: на эту структуру
одновременно пишут диаризация, транскрибация, определение ролей и API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


TEACHER_ROLE = 0


@dataclass
class Word:
    """Слово с таймкодом — то, что отдаёт ASR до привязки к спикерам."""

    start: float
    end: float
    text: str


@dataclass
class Segment:
    """Одна реплика: кусок речи одного человека.

    Заполняется по стадиям:
      diarize()      -> start, end, speaker
      transcribe()   -> text, words
      assign_roles() -> role, role_name
    """

    start: float
    end: float
    speaker: str = ""          # метка pyannote: SPEAKER_00, SPEAKER_01, ...
    text: str = ""
    words: list[Word] = field(default_factory=list)
    role: int | None = None    # 0 — преподаватель, 1..N — ученики
    role_name: str = ""        # "Преподаватель", "Ученик 1", ...

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def role_name(role: int) -> str:
    return "Преподаватель" if role == TEACHER_ROLE else f"Ученик {role}"
