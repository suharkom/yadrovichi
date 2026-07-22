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
