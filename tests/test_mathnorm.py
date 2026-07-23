from app.services.mathnorm import annotate, annotate_timeline


def test_spoken_equation_is_normalized() -> None:
    llm_text, has_math = annotate(
        "Икс в квадрате плюс два икс равно нулю."
    )

    assert llm_text == "x^2 + 2x = 0."
    assert has_math is True


def test_plain_text_is_not_changed() -> None:
    text = "В аудитории было двадцать студентов."

    assert annotate(text) == (text, False)


def test_compound_comparison_is_not_split() -> None:
    llm_text, has_math = annotate("Икс не равно нулю.")

    assert llm_text == "x != 0."
    assert has_math is True


def test_timeline_keeps_original_text_and_adds_llm_fields() -> None:
    timeline = [
        {
            "start": 1.0,
            "end": 2.0,
            "text": "Икс плюс два равно нулю.",
            "role": "teacher",
        }
    ]

    result = annotate_timeline(timeline)

    assert result[0]["text"] == "Икс плюс два равно нулю."
    assert result[0]["llm_text"] == "x + 2 = 0."
    assert result[0]["has_math"] is True
    assert "llm_text" not in timeline[0]
