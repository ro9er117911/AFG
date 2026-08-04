"""server/api.py's REST endpoints (History/Settings screens, docs/DESIGN.md §9), via a
minimal FastAPI TestClient app that includes only api_router — NOT the real
server/main.py:app, which would run init_db() against the real production DB and mount the
real frontend directory at import time. reset_provider() is monkeypatched to a no-op since a
settings POST would otherwise try to rebuild a real ClaudeProvider on the next call.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from antifraud_v3.pipeline.call_state import CallState
from antifraud_v3.reasoning.schemas import SynthesizeResult
from antifraud_v3.server.api import router as api_router
from antifraud_v3.storage import history
from antifraud_v3.tests.conftest import DEFAULT_FRAUD_TYPE


@pytest.fixture
def client(temp_history_db, temp_settings, monkeypatch):
    monkeypatch.setattr("antifraud_v3.server.api.reset_provider", lambda: None)
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def test_get_settings_returns_defaults(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "llm_model" in body
    assert "hard_triggers" in body


def test_post_settings_updates_and_persists(client):
    r = client.post("/api/settings", json={"llm_provider": "claude"})
    assert r.status_code == 200
    assert r.json()["llm_provider"] == "claude"

    r2 = client.get("/api/settings")
    assert r2.json()["llm_provider"] == "claude"


def test_post_settings_partial_update_ignores_unset_fields(client):
    client.post("/api/settings", json={"llm_model": "claude-sonnet-5"})
    r = client.post("/api/settings", json={"llm_effort": "high"})
    body = r.json()
    assert body["llm_model"] == "claude-sonnet-5"
    assert body["llm_effort"] == "high"


def test_post_settings_hard_triggers_update(client):
    r = client.post("/api/settings", json={"hard_triggers": ["新的關鍵字"]})
    assert r.json()["hard_triggers"] == ["新的關鍵字"]


def test_list_calls_empty(client):
    r = client.get("/api/calls")
    assert r.status_code == 200
    assert r.json() == []


def test_list_calls_after_finishing_one(client):
    call_id = history.create_call()
    cs = CallState()
    cs.apply_final_result(
        SynthesizeResult(risk_level="high", chunk_risk_score=90, hard_triggers=[], justification="j", case_memory_update="", fraud_type=DEFAULT_FRAUD_TYPE),
    )
    history.finish_call(call_id, cs, ended_reason="stopped")

    r = client.get("/api/calls")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["id"] == call_id
    assert body[0]["final_risk_level"] == "high"


def test_get_call_detail_not_found_returns_404(client):
    r = client.get("/api/calls/999999")
    assert r.status_code == 404


def test_get_call_detail_returns_transcript_and_trajectory(client):
    call_id = history.create_call()
    cs = CallState()
    features = {"pitch": {"mean_pitch": 0.0, "valid_samples": 0}, "volume": {}, "speech_rate": {}}
    cs.record_chunk_signals("詐騙測試逐字稿", None, features, {})
    cs.apply_final_result(
        SynthesizeResult(risk_level="medium", chunk_risk_score=50, hard_triggers=[], justification="j", case_memory_update="", fraud_type=DEFAULT_FRAUD_TYPE),
    )
    history.finish_call(call_id, cs, ended_reason="stopped")

    r = client.get(f"/api/calls/{call_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == call_id
    assert len(body["transcript"]) == 1
    assert body["transcript"][0]["text"] == "詐騙測試逐字稿"
    assert len(body["risk_trajectory"]) == 1
