"""Аналитика вовлечённости урока по готовому таймлайну.

На вход — результат пайплайна (dict из `pipeline.process`), на выход — метрики:
кто сколько говорил, баланс участия учеников, интерактивность (переключения
между говорящими), плотность формул и «лента» — кто доминирует по минутам.

Чистый постпроцесс: без GPU, без LLM, следующих этапов не трогает. Считается
из `timeline`, поэтому одинаково работает и на батч-, и на стрим-результате.
"""

from __future__ import annotations

from typing import Any


def _word_count(text: str) -> int:
    return len(text.split())


def _questions(text: str) -> int:
    return text.count("?")


def _share(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _round(value: float, ndigits: int = 3) -> float:
    return round(value, ndigits)


def compute_analytics(
    result: dict[str, Any],
    ribbon_bucket_seconds: float = 60.0,
) -> dict[str, Any]:
    """Собрать метрики вовлечённости из результата пайплайна.

    `ribbon_bucket_seconds` — ширина корзины «ленты доминирования» (по умолчанию
    минута): для каждой корзины считаем, кто говорил дольше всех.
    """
    timeline: list[dict[str, Any]] = result.get("timeline", [])
    lesson_duration = float(
        result.get("audio_duration_seconds", 0.0)
    )

    # Агрегаты по каждому говорящему (ключ — исходный спикер диаризации).
    speakers: dict[str, dict[str, Any]] = {}
    total_speech = 0.0

    for utt in timeline:
        speaker = utt.get("source_speaker", "UNKNOWN")
        duration = max(0.0, float(utt["end"]) - float(utt["start"]))
        total_speech += duration

        agg = speakers.setdefault(
            speaker,
            {
                "speaker": speaker,
                "role": utt.get("role", "unknown"),
                "display_name": utt.get("display_name", speaker),
                "talk_seconds": 0.0,
                "utterances": 0,
                "words": 0,
                "questions": 0,
                "math_utterances": 0,
            },
        )
        agg["talk_seconds"] += duration
        agg["utterances"] += 1
        agg["words"] += _word_count(utt.get("text", ""))
        agg["questions"] += _questions(utt.get("text", ""))
        if utt.get("has_math"):
            agg["math_utterances"] += 1

    total_utterances = len(timeline)

    # Оформляем список спикеров: доли и средняя длина реплики.
    speaker_list: list[dict[str, Any]] = []
    for agg in speakers.values():
        talk = agg["talk_seconds"]
        speaker_list.append(
            {
                "speaker": agg["speaker"],
                "role": agg["role"],
                "display_name": agg["display_name"],
                "talk_seconds": _round(talk, 1),
                "talk_share": _round(_share(talk, total_speech)),
                "utterances": agg["utterances"],
                "utterance_share": _round(
                    _share(agg["utterances"], total_utterances)
                ),
                "words": agg["words"],
                "avg_utterance_seconds": _round(
                    _share(talk, agg["utterances"]), 1
                ),
                "questions": agg["questions"],
                "math_utterances": agg["math_utterances"],
            }
        )
    speaker_list.sort(
        key=lambda item: item["talk_seconds"], reverse=True
    )

    # Роли: преподаватель против учеников.
    def role_talk(role: str) -> float:
        return sum(
            s["talk_seconds"]
            for s in speaker_list
            if s["role"] == role
        )

    teacher_talk = role_talk("teacher")
    student_talk = role_talk("student")

    roles: dict[str, dict[str, Any]] = {}
    for role in ("teacher", "student", "unknown"):
        members = [s for s in speaker_list if s["role"] == role]
        talk = sum(s["talk_seconds"] for s in members)
        roles[role] = {
            "talk_seconds": _round(talk, 1),
            "talk_share": _round(_share(talk, total_speech)),
            "utterances": sum(s["utterances"] for s in members),
            "words": sum(s["words"] for s in members),
            "speaker_count": len(members),
        }

    # Интерактивность: сколько раз менялся говорящий и как часто.
    switches = 0
    previous: str | None = None
    for utt in timeline:
        current = utt.get("source_speaker")
        if previous is not None and current != previous:
            switches += 1
        previous = current

    lesson_minutes = lesson_duration / 60.0 if lesson_duration else 0.0

    # Баланс участия учеников: доля самого тихого к доле самого активного.
    students = [s for s in speaker_list if s["role"] == "student"]
    student_shares = [s["talk_share"] for s in students]
    if student_shares and max(student_shares) > 0:
        balance = _share(min(student_shares), max(student_shares))
    else:
        balance = 0.0

    dominant = speaker_list[0] if speaker_list else None

    # «Лента доминирования»: по корзинам времени — кто говорил дольше всех.
    ribbon = _build_ribbon(
        timeline, lesson_duration, ribbon_bucket_seconds
    )

    math_utterances = sum(
        s["math_utterances"] for s in speaker_list
    )

    analytics = {
        "lesson_duration_seconds": _round(lesson_duration, 1),
        "total_speech_seconds": _round(total_speech, 1),
        "speech_coverage": _round(
            _share(total_speech, lesson_duration)
        ),
        "speaker_count": len(speaker_list),
        "utterance_count": total_utterances,
        "roles": roles,
        "speakers": speaker_list,
        "interactivity": {
            "speaker_switches": switches,
            "switches_per_minute": _round(
                _share(switches, lesson_minutes), 2
            ),
            "avg_turn_seconds": _round(
                _share(total_speech, switches + 1), 1
            ),
        },
        "participation": {
            "teacher_talk_share": _round(
                _share(teacher_talk, total_speech)
            ),
            "student_talk_share": _round(
                _share(student_talk, total_speech)
            ),
            "dominant_speaker": dominant["speaker"] if dominant else None,
            "dominant_display_name": (
                dominant["display_name"] if dominant else None
            ),
            "dominant_talk_share": (
                dominant["talk_share"] if dominant else 0.0
            ),
            "student_balance": _round(balance),
        },
        "math": {
            "utterances_with_formulas": math_utterances,
            "share": _round(
                _share(math_utterances, total_utterances)
            ),
        },
        "flags": _flags(
            teacher_share=_share(teacher_talk, total_speech),
            dominant=dominant,
            students=students,
            coverage=_share(total_speech, lesson_duration),
        ),
        "ribbon": ribbon,
    }
    return analytics


def _build_ribbon(
    timeline: list[dict[str, Any]],
    lesson_duration: float,
    bucket_seconds: float,
) -> list[dict[str, Any]]:
    """Для каждой корзины времени — роль и спикер, говорившие в ней дольше всех.

    Годится под цветную ленту в интерфейсе: одна полоса = одна корзина, цвет по
    роли доминирующего.
    """
    if bucket_seconds <= 0 or lesson_duration <= 0:
        return []

    bucket_count = int(lesson_duration // bucket_seconds) + 1
    buckets: list[dict[str, float]] = [
        {} for _ in range(bucket_count)
    ]
    meta: dict[str, dict[str, str]] = {}

    for utt in timeline:
        speaker = utt.get("source_speaker", "UNKNOWN")
        meta.setdefault(
            speaker,
            {
                "role": utt.get("role", "unknown"),
                "display_name": utt.get("display_name", speaker),
            },
        )
        start = float(utt["start"])
        end = float(utt["end"])
        # Разносим реплику по корзинам, которые она пересекает.
        index = int(start // bucket_seconds)
        while index < bucket_count and index * bucket_seconds < end:
            bucket_start = index * bucket_seconds
            bucket_end = bucket_start + bucket_seconds
            overlap = min(end, bucket_end) - max(start, bucket_start)
            if overlap > 0:
                buckets[index][speaker] = (
                    buckets[index].get(speaker, 0.0) + overlap
                )
            index += 1

    ribbon: list[dict[str, Any]] = []
    for i, bucket in enumerate(buckets):
        start = i * bucket_seconds
        entry: dict[str, Any] = {
            "start": _round(start, 1),
            "end": _round(
                min(start + bucket_seconds, lesson_duration), 1
            ),
            "dominant_speaker": None,
            "dominant_role": "silence",
        }
        if bucket:
            speaker = max(bucket, key=bucket.get)
            entry["dominant_speaker"] = speaker
            entry["dominant_role"] = meta[speaker]["role"]
            entry["dominant_display_name"] = meta[speaker][
                "display_name"
            ]
        ribbon.append(entry)
    return ribbon


def _flags(
    *,
    teacher_share: float,
    dominant: dict[str, Any] | None,
    students: list[dict[str, Any]],
    coverage: float,
) -> list[str]:
    """Короткие ярлыки-наблюдения для быстрого прочтения аналитики."""
    flags: list[str] = []

    if teacher_share < 0.3:
        # Урок ведёт не преподаватель — например, доклад студента.
        flags.append("student_led_session")

    if dominant and dominant["role"] == "student":
        student_shares = [s["talk_share"] for s in students]
        total_student = sum(student_shares)
        if total_student and dominant["talk_share"] / total_student > 0.6:
            flags.append("single_student_dominates")

    if len(students) >= 2:
        shares = [s["talk_share"] for s in students]
        if shares and min(shares) < 0.05:
            flags.append("passive_student_present")

    if coverage < 0.5:
        flags.append("sparse_speech")

    return flags
