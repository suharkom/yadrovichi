from app.services.alignment import (
    assign_speaker_to_word,
    build_utterances,
    join_word_tokens,
    smooth_short_speaker_turns,
)


def test_assigns_speaker_with_largest_overlap() -> None:
    word = {"start": 1.0, "end": 2.0, "text": "слово"}
    turns = [
        {"start": 0.0, "end": 1.4, "speaker": "A"},
        {"start": 1.4, "end": 3.0, "speaker": "B"},
    ]

    assert assign_speaker_to_word(word, turns) == "B"


def test_returns_unknown_when_turn_is_too_far() -> None:
    word = {"start": 5.0, "end": 5.2, "text": "слово"}
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}]

    assert assign_speaker_to_word(word, turns) == "UNKNOWN"


def test_builds_new_utterance_after_speaker_change() -> None:
    words = [
        {"start": 0.0, "end": 0.2, "text": "Добрый", "speaker": "A"},
        {"start": 0.3, "end": 0.5, "text": "день", "speaker": "A"},
        {"start": 0.6, "end": 0.8, "text": "Здравствуйте", "speaker": "B"},
    ]

    utterances = build_utterances(words)

    assert [item["speaker"] for item in utterances] == ["A", "B"]
    assert utterances[0]["text"] == "Добрый день"


def test_joins_hyphens_and_punctuation_without_extra_spaces() -> None:
    words = ["Что", "-то", ",", "во", "-первых", ",", "работает", "."]

    assert join_word_tokens(words) == "Что-то, во-первых, работает."


def test_smooths_short_a_b_a_switch() -> None:
    utterances = [
        {"start": 0.0, "end": 3.0, "speaker": "A", "text": "ваши"},
        {"start": 3.0, "end": 4.9, "speaker": "B", "text": "вкусы и"},
        {"start": 5.0, "end": 6.0, "speaker": "A", "text": "предпочтения"},
    ]

    result = smooth_short_speaker_turns(
        utterances,
        max_duration=2.0,
        max_words=3,
    )

    assert len(result) == 1
    assert result[0]["speaker"] == "A"
    assert result[0]["text"] == "ваши вкусы и предпочтения"


def test_does_not_smooth_short_response() -> None:
    utterances = [
        {"start": 0.0, "end": 3.0, "speaker": "A", "text": "начало"},
        {"start": 3.0, "end": 3.4, "speaker": "B", "text": "нет, давай"},
        {"start": 3.5, "end": 5.0, "speaker": "A", "text": "продолжение"},
    ]

    result = smooth_short_speaker_turns(
        utterances,
        max_duration=2.0,
        max_words=3,
    )

    assert [item["speaker"] for item in result] == ["A", "B", "A"]


def test_does_not_smooth_turn_separated_by_long_pause() -> None:
    utterances = [
        {"start": 0.0, "end": 2.0, "speaker": "A", "text": "начало"},
        {"start": 3.0, "end": 4.0, "speaker": "B", "text": "короткий ответ"},
        {"start": 6.0, "end": 7.0, "speaker": "A", "text": "продолжение"},
    ]

    result = smooth_short_speaker_turns(utterances)

    assert [item["speaker"] for item in result] == ["A", "B", "A"]
