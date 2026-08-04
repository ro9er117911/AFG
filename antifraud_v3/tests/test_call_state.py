"""pipeline/call_state.py — the live keyword hard-trigger check (check_live_hard_trigger,
no LLM, runs on every chunk) and the single end-of-call alert evaluation
(apply_final_result, runs once after the one-shot reasoning pipeline). See that module's
docstring for why there's no more per-chunk-debounce logic: that existed specifically to stop
one noisy per-chunk LLM verdict from firing an alert alone, and there's only one verdict per
call now.
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


# ---- apply_final_result: the single end-of-call verdict ----


def test_low_risk_final_result_does_not_alert():
    cs = CallState()
    alert = cs.apply_final_result(make_result(risk_level="low"))
    assert alert is None
    assert cs.alert_state == "none"


def test_medium_risk_final_result_does_not_alert():
    # A final analysis isn't live protection — only "high" or an explicit hard-trigger hit
    # is severe enough to surface as an alert after the call has already ended.
    cs = CallState()
    alert = cs.apply_final_result(make_result(risk_level="medium"))
    assert alert is None


def test_high_risk_final_result_alerts():
    cs = CallState()
    alert = cs.apply_final_result(make_result(risk_level="high"))
    assert alert is not None
    assert alert.reason == "final_analysis"
    assert cs.alert_state == "alert_issued"


def test_hard_trigger_in_final_result_alerts_even_at_low_risk_level():
    cs = CallState()
    alert = cs.apply_final_result(make_result(risk_level="low", hard_trigger_fired=True))
    assert alert is not None
    assert alert.reason == "hard_trigger"
    assert alert.trigger_name == "OTP要求"
    assert alert.quote == "給我驗證碼"
    assert cs.alert_state == "alert_issued"


def test_apply_final_result_records_transcript_free_risk_point():
    cs = CallState()
    cs.apply_final_result(make_result(risk_level="medium"))
    assert len(cs.risk_trajectory) == 1
    assert cs.risk_trajectory[0].risk_level == "medium"
    assert cs.case_memory == "test memory"


def test_apply_final_result_timestamp_override():
    cs = CallState()
    cs.apply_final_result(make_result(), timestamp=42.5)
    assert cs.risk_trajectory[-1].timestamp == 42.5


def test_apply_final_result_default_timestamp_is_wall_clock():
    cs = CallState()
    cs.apply_final_result(make_result())
    assert cs.risk_trajectory[-1].timestamp >= 0.0
    assert cs.risk_trajectory[-1].timestamp < 5.0  # should be near-instant in a test


def test_apply_final_result_does_not_refire_if_a_live_hard_trigger_already_alerted():
    cs = CallState()
    first = cs.check_live_hard_trigger("請提供驗證碼給我")
    assert first is not None
    second = cs.apply_final_result(make_result(risk_level="high"))
    assert second is None  # already showing — the call already got its one alert


# ---- check_live_hard_trigger: the cheap no-LLM keyword check ----


def test_live_hard_trigger_matches_a_keyword():
    cs = CallState()
    alert = cs.check_live_hard_trigger("我們需要幫你圈存帳戶，請提供驗證碼")
    assert alert is not None
    assert alert.reason == "hard_trigger"
    assert cs.alert_state == "alert_issued"


def test_live_hard_trigger_does_not_match_ordinary_speech():
    cs = CallState()
    alert = cs.check_live_hard_trigger("我們晚上要不要一起吃飯")
    assert alert is None
    assert cs.alert_state == "none"


def test_live_hard_trigger_does_not_refire_while_alert_still_issued():
    cs = CallState()
    first = cs.check_live_hard_trigger("請提供驗證碼")
    assert first is not None
    second = cs.check_live_hard_trigger("再說一次驗證碼")
    assert second is None  # same alert still showing — not re-fired every subsequent chunk


def test_acknowledge_alert_rearms_live_hard_trigger_path():
    cs = CallState()
    first = cs.check_live_hard_trigger("請提供驗證碼")
    assert first is not None
    cs.acknowledge_alert()
    assert cs.alert_state == "resolved"
    second = cs.check_live_hard_trigger("再一次要求驗證碼")
    assert second is not None


def test_acknowledge_alert_is_a_no_op_when_no_alert_issued():
    cs = CallState()
    cs.acknowledge_alert()  # should not raise
    assert cs.alert_state == "none"


# ---- record_chunk_signals: per-chunk transcript/acoustic/emotion, no LLM ----


def test_record_chunk_signals_appends_transcript_and_chunk_signals():
    cs = CallState()
    features = {"pitch": {"mean_pitch": 200.0}, "volume": {"mean_volume": 0.5}, "speech_rate": {"speech_rate_variation": 0.1}}
    cs.record_chunk_signals("hello world", "caller", features, {"neutral": 0.9})
    assert len(cs.transcript) == 1
    assert cs.transcript[0].text == "hello world"
    assert cs.transcript[0].speaker == "caller"
    assert len(cs.chunk_signals) == 1
    assert cs.chunk_signals[0].acoustic_features == features
    assert cs.chunk_signals[0].emotion_probs == {"neutral": 0.9}


def test_record_chunk_signals_timestamp_override_used_for_batch_upload():
    """See CallState.record_chunk_signals()'s docstring — server/upload.py passes an explicit
    timestamp (position within the uploaded audio) instead of wall-clock elapsed time."""
    cs = CallState()
    features = {"pitch": {}, "volume": {}, "speech_rate": {}}
    cs.record_chunk_signals("t1", None, features, {}, timestamp=42.5)
    assert cs.transcript[-1].timestamp == 42.5
    assert cs.chunk_signals[-1].timestamp == 42.5


# ---- baseline (unchanged behavior, still exercised through the new entry points) ----


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
