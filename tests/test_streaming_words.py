from app.services.streaming import _finish

MAPPING = {
    "S0": {"speaker_id": 0, "role": "teacher", "display_name": "Преподаватель"},
}


def test_finish_keeps_word_probabilities():
    current = {
        "start": 0.0,
        "end": 1.0,
        "speaker": "S0",
        "words": [
            {"text": "привет", "probability": 0.95},
            {"text": "мир", "probability": 0.30},
        ],
    }
    result = _finish(current, MAPPING, 0)

    assert result["text"] == "привет мир"
    assert result["display_name"] == "Преподаватель"
    assert len(result["words"]) == 2
    assert result["words"][0]["probability"] == 0.95
    assert result["words"][1]["probability"] == 0.30
    assert result["words"][1]["text"] == "мир"


def test_finish_handles_missing_probability():
    current = {
        "start": 0.0,
        "end": 1.0,
        "speaker": "S0",
        "words": [{"text": "слово", "probability": None}],
    }
    result = _finish(current, MAPPING, 1)
    assert result["words"][0]["probability"] is None
    assert result["text"] == "слово"
