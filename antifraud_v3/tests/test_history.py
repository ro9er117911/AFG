"""storage/history.py — always run against temp_history_db (tests/conftest.py), never the
real antifraud_v3/data/history.sqlite3.
"""

import sqlite3

from antifraud_v3.pipeline.call_state import CallState
from antifraud_v3.reasoning.schemas import SynthesizeResult
from antifraud_v3.storage import history


def make_result(risk_level="low"):
    return SynthesizeResult(
        risk_level=risk_level,
        chunk_risk_score={"low": 5, "medium": 50, "high": 90}[risk_level],
        hard_triggers=[],
        justification="j",
        case_memory_update="",
    )


def test_create_call_defaults_to_live_source(temp_history_db):
    call_id = history.create_call()
    cs = CallState()
    cs.apply_final_result(make_result("low"))
    history.finish_call(call_id, cs, ended_reason="stopped")

    calls = history.list_calls()
    assert len(calls) == 1
    assert calls[0]["source"] == "live"
    assert calls[0]["ended_reason"] == "stopped"


def test_create_call_upload_source(temp_history_db):
    call_id = history.create_call(source="upload")
    cs = CallState()
    history.finish_call(call_id, cs, ended_reason="uploaded", duration_seconds=12.5)

    detail = history.get_call_detail(call_id)
    assert detail is not None
    assert detail["source"] == "upload"
    assert detail["duration_seconds"] == 12.5


def test_finish_call_duration_override_used_over_wall_clock(temp_history_db):
    """See history.finish_call()'s docstring: a batch/upload call's real duration should be
    the audio's length, not how long ASR/LLM processing happened to take."""
    call_id = history.create_call(source="upload")
    cs = CallState()  # elapsed_seconds() would be ~0 here since no time has passed
    history.finish_call(call_id, cs, ended_reason="uploaded", duration_seconds=97.3)
    detail = history.get_call_detail(call_id)
    assert detail["duration_seconds"] == 97.3


def test_finish_call_final_risk_level_reflects_the_one_final_result(temp_history_db):
    """The reasoning pipeline runs once per call now (see pipeline/chunk_worker.py's
    run_final_analysis), so risk_trajectory holds exactly one point — no more "highest across
    chunks" to pick. Transcript entries come from record_chunk_signals independently of that
    one final verdict."""
    call_id = history.create_call()
    cs = CallState()
    features = {"pitch": {"mean_pitch": 0.0, "valid_samples": 0}, "volume": {}, "speech_rate": {}}
    cs.record_chunk_signals("t1", None, features, {})
    cs.record_chunk_signals("t2", None, features, {})
    cs.record_chunk_signals("t3", None, features, {})
    cs.apply_final_result(make_result("high"))
    history.finish_call(call_id, cs, ended_reason="stopped")

    detail = history.get_call_detail(call_id)
    assert detail["final_risk_level"] == "high"
    assert len(detail["risk_trajectory"]) == 1
    assert len(detail["transcript"]) == 3


def test_in_progress_calls_excluded_from_list(temp_history_db):
    history.create_call()  # never finished — ended_reason stays 'in_progress'
    assert history.list_calls() == []


def test_get_call_detail_returns_none_for_missing_call(temp_history_db):
    assert history.get_call_detail(99999) is None


def test_list_calls_orders_most_recent_first(temp_history_db):
    id1 = history.create_call()
    history.finish_call(id1, CallState(), ended_reason="stopped")
    id2 = history.create_call()
    history.finish_call(id2, CallState(), ended_reason="stopped")

    calls = history.list_calls()
    assert [c["id"] for c in calls] == [id2, id1]


def test_migration_adds_source_column_to_pre_existing_db(tmp_path, monkeypatch):
    """Simulates a DB created before the upload-analysis feature existed (no `source` column)
    — init_db()'s ALTER TABLE migration should add it without erroring or losing rows. See
    storage/history.py's init_db() docstring."""
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at REAL NOT NULL,
            duration_seconds REAL,
            final_risk_level TEXT,
            transcript_json TEXT,
            risk_trajectory_json TEXT,
            ended_reason TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO calls (started_at, duration_seconds, final_risk_level, ended_reason) "
        "VALUES (1000.0, 30.0, 'low', 'stopped')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(history, "DB_PATH", db_path)
    history.init_db()  # must not raise

    calls = history.list_calls()
    assert len(calls) == 1
    assert calls[0]["source"] == "live"  # DEFAULT 'live' applied retroactively
    assert calls[0]["final_risk_level"] == "low"


def test_init_db_is_idempotent(temp_history_db):
    history.init_db()
    history.init_db()  # should not raise on a DB that already has the source column
    assert history.list_calls() == []
