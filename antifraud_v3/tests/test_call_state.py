"""pipeline/call_state.py's alert debounce state machine — see docs/DESIGN.md §2.4. These are
exactly the scenarios reasoned through there: a single noisy chunk must not fire a general
alert, sustained/rising risk across >=debounce_chunks consecutive chunks should, and hard
triggers bypass debouncing entirely but still don't re-fire while already showing.
"""

from antifraud_v3.pipeline.call_state import CallState
from antifraud_v3.reasoning.schemas import HardTriggerHit, SynthesizeResult


def make_result(risk_level="low", hard_trigger_fired=False, trigger_name="OTP要求", quote="給我驗證碼"):
    return SynthesizeResult(
        risk_level=risk_level,
        chunk_risk_score={"low": 5, "medium": 50, "high": 90}[risk_level],
        hard_triggers=[
            HardTriggerHit(name=trigger_name, fired=hard_trigger_fired, quote=quote if hard_trigger_fired else None)
        ],
        justification="test justification",
        case_memory_update="test memory",
    )


def test_single_elevated_chunk_does_not_alert():
    cs = CallState(debounce_chunks=2)
    alert = cs.record_chunk("t1", None, make_result(risk_level="medium"))
    assert alert is None
    assert cs.alert_state == "watching"


def test_sustained_elevated_chunks_trigger_alert():
    cs = CallState(debounce_chunks=2)
    assert cs.record_chunk("t1", None, make_result(risk_level="medium")) is None
    alert = cs.record_chunk("t2", None, make_result(risk_level="high"))
    assert alert is not None
    assert alert.reason == "sustained_risk"
    assert cs.alert_state == "alert_issued"


def test_low_risk_chunk_resets_the_streak():
    cs = CallState(debounce_chunks=2)
    cs.record_chunk("t1", None, make_result(risk_level="medium"))
    cs.record_chunk("t2", None, make_result(risk_level="low"))  # streak should reset to 0
    alert = cs.record_chunk("t3", None, make_result(risk_level="medium"))
    assert alert is None  # only 1 consecutive elevated chunk since the reset
    assert cs.alert_state == "watching"


def test_three_low_risk_chunks_never_alert():
    cs = CallState(debounce_chunks=2)
    for i in range(5):
        alert = cs.record_chunk(f"t{i}", None, make_result(risk_level="low"))
        assert alert is None
    assert cs.alert_state == "none"


def test_hard_trigger_fires_immediately_even_with_high_debounce():
    cs = CallState(debounce_chunks=5)  # would otherwise need 5 elevated chunks
    alert = cs.record_chunk("t1", None, make_result(risk_level="low", hard_trigger_fired=True))
    assert alert is not None
    assert alert.reason == "hard_trigger"
    assert alert.trigger_name == "OTP要求"
    assert alert.quote == "給我驗證碼"
    assert cs.alert_state == "alert_issued"


def test_hard_trigger_does_not_refire_while_alert_still_issued():
    cs = CallState(debounce_chunks=5)
    first = cs.record_chunk("t1", None, make_result(risk_level="low", hard_trigger_fired=True))
    assert first is not None
    second = cs.record_chunk("t2", None, make_result(risk_level="low", hard_trigger_fired=True))
    assert second is None  # same alert still showing — not re-fired every subsequent chunk


def test_sustained_alert_does_not_refire_while_issued():
    cs = CallState(debounce_chunks=1)
    first = cs.record_chunk("t1", None, make_result(risk_level="high"))
    assert first is not None
    second = cs.record_chunk("t2", None, make_result(risk_level="high"))
    assert second is None


def test_acknowledge_alert_rearms_sustained_risk_path():
    cs = CallState(debounce_chunks=1)
    first = cs.record_chunk("t1", None, make_result(risk_level="high"))
    assert first is not None
    cs.acknowledge_alert()
    assert cs.alert_state == "resolved"
    second = cs.record_chunk("t2", None, make_result(risk_level="high"))
    assert second is not None
    assert cs.alert_state == "alert_issued"


def test_acknowledge_alert_rearms_hard_trigger_path():
    cs = CallState(debounce_chunks=5)
    first = cs.record_chunk("t1", None, make_result(risk_level="low", hard_trigger_fired=True))
    assert first is not None
    cs.acknowledge_alert()
    second = cs.record_chunk("t2", None, make_result(risk_level="low", hard_trigger_fired=True))
    assert second is not None
    assert second.reason == "hard_trigger"


def test_acknowledge_alert_is_a_no_op_when_no_alert_issued():
    cs = CallState()
    cs.acknowledge_alert()  # should not raise
    assert cs.alert_state == "none"


def test_record_chunk_appends_transcript_and_risk_trajectory():
    cs = CallState()
    cs.record_chunk("hello world", "caller", make_result(risk_level="low"))
    assert len(cs.transcript) == 1
    assert cs.transcript[0].text == "hello world"
    assert cs.transcript[0].speaker == "caller"
    assert len(cs.risk_trajectory) == 1
    assert cs.risk_trajectory[0].risk_level == "low"
    assert cs.case_memory == "test memory"


def test_record_chunk_timestamp_override_used_for_batch_upload():
    """See CallState.record_chunk()'s docstring — server/upload.py passes an explicit
    timestamp (position within the uploaded audio) instead of wall-clock elapsed time."""
    cs = CallState()
    cs.record_chunk("t1", None, make_result(), timestamp=42.5)
    assert cs.transcript[-1].timestamp == 42.5
    assert cs.risk_trajectory[-1].timestamp == 42.5


def test_record_chunk_default_timestamp_is_wall_clock():
    cs = CallState()
    cs.record_chunk("t1", None, make_result())
    assert cs.risk_trajectory[-1].timestamp >= 0.0
    assert cs.risk_trajectory[-1].timestamp < 5.0  # should be near-instant in a test


def test_baseline_not_set_before_window_elapses():
    cs = CallState()
    features = {
        "pitch": {"mean_pitch": 200.0},
        "volume": {"mean_volume": 0.5},
        "speech_rate": {"speech_rate_variation": 0.1},
    }
    cs.maybe_set_baseline(features)
    assert cs.baseline is None


def test_baseline_set_after_window_elapses():
    cs = CallState()
    cs._started_at -= CallState.BASELINE_WINDOW_S + 1  # simulate time having passed
    features = {
        "pitch": {"mean_pitch": 210.0},
        "volume": {"mean_volume": 0.6},
        "speech_rate": {"speech_rate_variation": 0.12},
    }
    cs.maybe_set_baseline(features)
    assert cs.baseline is not None
    assert cs.baseline["mean_pitch"] == 210.0


def test_baseline_only_set_once():
    cs = CallState()
    cs._started_at -= CallState.BASELINE_WINDOW_S + 1
    cs.maybe_set_baseline({"pitch": {"mean_pitch": 100.0}, "volume": {"mean_volume": 0.1}, "speech_rate": {"speech_rate_variation": 0.0}})
    cs.maybe_set_baseline({"pitch": {"mean_pitch": 999.0}, "volume": {"mean_volume": 0.9}, "speech_rate": {"speech_rate_variation": 0.9}})
    assert cs.baseline["mean_pitch"] == 100.0  # second call should not overwrite
