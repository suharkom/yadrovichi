from app.services.export_subs import _timestamp, to_srt, to_vtt


def _tl():
    return [
        {"start": 0.0, "end": 2.5, "text": "привет", "display_name": "Преподаватель"},
        {"start": 3.2, "end": 5.0, "text": "да", "display_name": "Ученик 1"},
    ]


def test_timestamp_srt_and_vtt():
    assert _timestamp(0.0, ",") == "00:00:00,000"
    assert _timestamp(3.2, ".") == "00:00:03.200"
    assert _timestamp(3661.5, ",") == "01:01:01,500"


def test_srt_structure():
    srt = to_srt(_tl())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nПреподаватель: привет")
    assert "2\n00:00:03,200 --> 00:00:05,000\nУченик 1: да" in srt


def test_srt_without_speaker():
    srt = to_srt(_tl(), with_speaker=False)
    assert "Преподаватель:" not in srt
    assert "привет" in srt


def test_vtt_structure():
    vtt = to_vtt(_tl())
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "<v Преподаватель>привет</v>" in vtt


def test_empty_timeline():
    assert to_srt([]) == "\n"
    assert to_vtt([]).startswith("WEBVTT")
