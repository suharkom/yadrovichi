from app.services.analytics import compute_analytics


def _result(timeline, duration=100.0):
    return {
        "audio_duration_seconds": duration,
        "timeline": timeline,
    }


def _utt(start, end, speaker, role, text="слово", has_math=False):
    return {
        "start": start,
        "end": end,
        "source_speaker": speaker,
        "role": role,
        "display_name": f"{role}:{speaker}",
        "text": text,
        "has_math": has_math,
    }


def test_talk_shares_sum_to_one():
    timeline = [
        _utt(0, 10, "S0", "teacher"),
        _utt(10, 40, "S1", "student"),
        _utt(40, 50, "S2", "student"),
    ]
    a = compute_analytics(_result(timeline))
    shares = sum(s["talk_share"] for s in a["speakers"])
    assert abs(shares - 1.0) < 1e-6
    assert a["total_speech_seconds"] == 50.0
    assert a["speaker_count"] == 3


def test_dominant_and_roles():
    timeline = [
        _utt(0, 5, "S0", "teacher"),
        _utt(5, 45, "S1", "student"),  # доминирует
        _utt(45, 55, "S2", "student"),
    ]
    a = compute_analytics(_result(timeline))
    assert a["participation"]["dominant_speaker"] == "S1"
    assert a["speakers"][0]["speaker"] == "S1"
    assert a["roles"]["student"]["talk_seconds"] == 50.0
    assert a["roles"]["teacher"]["talk_seconds"] == 5.0


def test_interactivity_counts_switches():
    timeline = [
        _utt(0, 5, "S0", "teacher"),
        _utt(5, 10, "S1", "student"),
        _utt(10, 15, "S1", "student"),  # тот же — не переключение
        _utt(15, 20, "S0", "teacher"),
    ]
    a = compute_analytics(_result(timeline))
    assert a["interactivity"]["speaker_switches"] == 2


def test_questions_and_math_counted():
    timeline = [
        _utt(0, 5, "S0", "teacher", text="а что это? почему?"),
        _utt(5, 10, "S1", "student", text="формула", has_math=True),
    ]
    a = compute_analytics(_result(timeline))
    teacher = next(s for s in a["speakers"] if s["speaker"] == "S0")
    assert teacher["questions"] == 2
    assert a["math"]["utterances_with_formulas"] == 1


def test_student_led_flag():
    # Преподаватель говорит мало — доклад ведёт ученик.
    timeline = [
        _utt(0, 5, "S0", "teacher"),
        _utt(5, 95, "S1", "student"),
    ]
    a = compute_analytics(_result(timeline))
    assert "student_led_session" in a["flags"]
    assert "single_student_dominates" in a["flags"]


def test_ribbon_buckets_dominant_speaker():
    timeline = [
        _utt(0, 50, "S1", "student"),   # доминирует в 1-й минуте
        _utt(50, 60, "S0", "teacher"),
        _utt(60, 120, "S0", "teacher"),  # доминирует во 2-й минуте
    ]
    a = compute_analytics(_result(timeline, duration=120.0))
    ribbon = a["ribbon"]
    assert ribbon[0]["dominant_speaker"] == "S1"
    assert ribbon[1]["dominant_speaker"] == "S0"


def test_empty_timeline_is_safe():
    a = compute_analytics(_result([], duration=0.0))
    assert a["speaker_count"] == 0
    assert a["total_speech_seconds"] == 0.0
    assert a["ribbon"] == []
