from app.services.role_detection import (
    create_speaker_mapping,
    detect_teacher,
)


def test_markers_can_outweigh_longer_student_speech() -> None:
    utterances = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "TEACHER",
            "text": "Ребята, откройте тетради и обратите внимание.",
        },
        {
            "start": 5.0,
            "end": 45.0,
            "speaker": "STUDENT",
            "text": "Я подготовил длинный ответ по теме занятия.",
        },
        {
            "start": 45.0,
            "end": 50.0,
            "speaker": "TEACHER",
            "text": "Верно. У кого есть вопросы?",
        },
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "TEACHER"
    assert result["low_confidence"] is False


def test_students_are_numbered_by_first_appearance() -> None:
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "B", "text": "ответ"},
        {"start": 1.0, "end": 2.0, "speaker": "T", "text": "верно"},
        {"start": 2.0, "end": 3.0, "speaker": "A", "text": "ответ"},
    ]

    mapping = create_speaker_mapping(utterances, teacher_speaker="T")

    assert mapping["T"]["speaker_id"] == 0
    assert mapping["B"]["speaker_id"] == 1
    assert mapping["A"]["speaker_id"] == 2


def test_speaker_order_and_speech_share_do_not_affect_role() -> None:
    utterances = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "TEACHER",
            "text": "Ребята, откройте тетради.",
        },
        {
            "start": 2.0,
            "end": 30.0,
            "speaker": "STUDENT",
            "text": "Подробный ответ ученика без маркеров роли.",
        },
        {
            "start": 31.0,
            "end": 60.0,
            "speaker": "STUDENT",
            "text": "Продолжение длинного ответа.",
        },
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "TEACHER"
    assert result["scoring_basis"] == "teacher_markers"


def test_markers_break_a_graph_tie() -> None:
    utterances = [
        {"start": 0, "end": 1, "speaker": "A", "text": "ответ"},
        {"start": 1, "end": 2, "speaker": "T", "text": "ребята"},
        {"start": 2, "end": 3, "speaker": "B", "text": "ответ"},
        {"start": 3, "end": 4, "speaker": "T", "text": "верно"},
        {"start": 4, "end": 5, "speaker": "A", "text": "ответ"},
        {"start": 5, "end": 6, "speaker": "B", "text": "ответ"},
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "T"
    assert result["scoring_basis"] == "teacher_markers"


def test_two_speakers_use_markers_because_graph_is_tied() -> None:
    utterances = [
        {"start": 0, "end": 1, "speaker": "S", "text": "длинный ответ"},
        {"start": 1, "end": 2, "speaker": "T", "text": "откройте тетради"},
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "T"
    assert result["scoring_basis"] == "teacher_markers"
    assert result["used_speech_share_tiebreaker"] is False


def test_speech_share_tiebreaker_is_always_low_confidence() -> None:
    utterances = [
        {
            "start": 0,
            "end": 20,
            "speaker": "A",
            "text": "Длинная реплика без маркеров.",
        },
        {
            "start": 20,
            "end": 21,
            "speaker": "B",
            "text": "Короткая реплика без маркеров.",
        },
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "A"
    assert result["used_speech_share_tiebreaker"] is True
    assert result["low_confidence"] is True


def test_decisive_graph_outweighs_student_markers_and_speech_share() -> None:
    utterances = [
        {"start": 0, "end": 1, "speaker": "T", "text": "Хорошо."},
        {
            "start": 1,
            "end": 40,
            "speaker": "A",
            "text": (
                "Ребята, давайте перейдем к следующему слайду. "
                "Обратите внимание и запишите определение."
            ),
        },
        {"start": 40, "end": 41, "speaker": "T", "text": "Спасибо."},
        {"start": 41, "end": 42, "speaker": "B", "text": "Ответ."},
        {"start": 42, "end": 43, "speaker": "T", "text": "Продолжим."},
        {"start": 43, "end": 44, "speaker": "C", "text": "Ответ."},
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "T"
    assert result["scoring_basis"] == "speaker_graph"
    assert result["scores"]["A"]["raw_marker_score"] > 0
    assert (
        result["scores"]["A"]["speech_share"]
        > result["scores"]["T"]["speech_share"]
    )


def test_decisive_graph_works_with_fewer_than_five_transitions() -> None:
    utterances = [
        {"start": 0, "end": 1, "speaker": "T", "text": "Начнем."},
        {"start": 1, "end": 2, "speaker": "A", "text": "Ответ."},
        {"start": 2, "end": 3, "speaker": "T", "text": "Спасибо."},
        {
            "start": 3,
            "end": 10,
            "speaker": "B",
            "text": "Давайте перейдем к следующему слайду.",
        },
    ]

    result = detect_teacher(utterances)

    assert result["teacher_speaker"] == "T"
    assert result["scoring_basis"] == "speaker_graph"
