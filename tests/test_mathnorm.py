from app.services.mathnorm import annotate, annotate_item, has_math


def test_normalizes_spoken_equation() -> None:
    text, contains_math = annotate(
        "Икс в квадрате плюс два икс равно нулю."
    )

    assert contains_math is True
    assert text == "x^2 + 2x = 0."


def test_normalizes_economics_equation() -> None:
    text, contains_math = annotate(
        "Уравнение эм равно пэ умножить на игрек."
    )

    assert contains_math is True
    assert text == "уравнение m = p * y."


def test_normalizes_spoken_arithmetic_without_inventing_equality() -> None:
    text, contains_math = annotate(
        "Минус восемьдесят разделить на двадцать, минус четыре."
    )

    assert contains_math is True
    assert text == "- 80 / 20, - 4."


def test_does_not_treat_everyday_equal_as_math() -> None:
    text = "Мне всё равно, давайте продолжим."

    assert has_math(text) is False
    assert annotate(text) == (text, False)


def test_does_not_treat_splitting_assignment_as_math() -> None:
    text = "Давайте разделим задание на пункт А и пункт Б."

    assert annotate(text) == (text, False)


def test_does_not_replace_plain_comparison_words() -> None:
    text = "Выпуск меньше потенциального на 20 процентов."

    assert annotate(text) == (text, False)


def test_preserves_original_text_in_annotated_item() -> None:
    item = {
        "start": 1.0,
        "end": 2.0,
        "text": "Икс плюс два.",
    }

    result = annotate_item(item)

    assert result["text"] == "Икс плюс два."
    assert result["llm_text"] == "x + 2."
    assert result["has_math"] is True
