from app.services.refine import _refine_plain_text, refine_utterances


def _words(*pairs):
    return [{"text": t, "probability": p} for t, p in pairs]


def test_removes_fillers():
    utt = {"words": _words(("э", 0.4), ("мы", 0.9), ("хотим", 0.9))}
    refine_utterances([utt])
    assert [w["text"] for w in utt["words"]] == ["Мы", "хотим"]


def test_collapses_long_duplicate_but_keeps_short():
    utt = {"words": _words(("хотим", 0.9), ("хотим", 0.8), ("да", 0.9), ("да", 0.9))}
    refine_utterances([utt])
    texts = [w["text"].lower() for w in utt["words"]]
    assert texts.count("хотим") == 1  # длинный повтор схлопнут
    assert texts.count("да") == 2      # короткий повтор сохранён


def test_capitalizes_start_and_after_sentence():
    utt = {"words": _words(("привет", 0.9), ("мир.", 0.9), ("как", 0.9), ("дела", 0.9))}
    refine_utterances([utt])
    texts = [w["text"] for w in utt["words"]]
    assert texts[0] == "Привет"
    assert texts[2] == "Как"  # после точки


def test_keeps_probability():
    utt = {"words": _words(("мир", 0.31), ("да", 0.9))}
    refine_utterances([utt])
    assert utt["words"][0]["probability"] == 0.31


def test_rebuilds_text_field():
    utt = {"words": _words(("э", 0.3), ("привет", 0.9))}
    refine_utterances([utt])
    assert utt["text"] == "Привет"


def test_idempotent():
    utt = {"words": _words(("э", 0.3), ("хотим", 0.9), ("хотим", 0.8))}
    refine_utterances([utt])
    first = [w["text"] for w in utt["words"]]
    refine_utterances([utt])
    assert [w["text"] for w in utt["words"]] == first


def test_plain_text_fallback():
    assert _refine_plain_text("э мы хотим хотим") == "Мы хотим"
