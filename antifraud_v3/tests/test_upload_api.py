"""server/upload.py's POST /api/upload-call (docs task: "audio file upload — analyze a
pre-recorded call"). VADChunker and process_chunk are mocked here — this test is checking the
endpoint's own contract (request/response shape, history persistence, error handling), not
VAD/ASR accuracy, which is exercised separately by real audio through the real pipeline (see
the manual end-to-end test with eval/test_clips/*.wav and a real browser, described in the
session report — deliberately not re-mocked into a "fast unit test" here, per this project's
repeated lesson that mocked-only testing misses real bugs the real pipeline would catch).
"""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

import antifraud_v3.server.upload as upload_mod
from antifraud_v3.llm import LLMProviderError
from antifraud_v3.pipeline.chunk_worker import ChunkResult
from antifraud_v3.reasoning.schemas import HardTriggerHit, SynthesizeResult
from antifraud_v3.storage import history


def make_fake_process_chunk(scripted):
    """Real process_chunk's contract includes a side effect — it calls
    call_state.record_chunk(), which is what actually populates CallState.transcript /
    risk_trajectory and runs the real alert debounce/hard-trigger logic
    (pipeline/call_state.py). A mock that only returns a canned ChunkResult without
    replicating that side effect would silently leave call_state empty, since
    server/upload.py's response is built from call_state, not from the raw per-chunk return
    values — this fake calls the real record_chunk() so the alert/state-machine logic under
    test is the real one, only ASR/acoustic/emotion/LLM inference is stubbed out.
    """
    results = iter(scripted)

    def fake(y, sr, provider, call_state, speaker_guess=None, hard_triggers=None, timestamp=None):
        transcript_text, synth_result = next(results)
        alert = call_state.record_chunk(transcript_text, speaker_guess, synth_result, timestamp=timestamp)
        return ChunkResult(transcript_text=transcript_text, synthesize_result=synth_result, alert=alert)

    return fake


def make_wav_bytes(seconds=1.0, sr=16000):
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class FakeChunker:
    """Stand-in for audio.vad.VADChunker — deterministic chunk count/content, independent of
    real VAD speech-detection behavior."""

    def __init__(self, *args, **kwargs):
        pass

    def push_audio(self, y):
        return [np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32)]

    def flush(self):
        return None


@pytest.fixture
def client(temp_history_db, temp_settings, monkeypatch):
    monkeypatch.setattr(upload_mod, "VADChunker", FakeChunker)
    monkeypatch.setattr(upload_mod, "get_llm_provider", lambda: object())
    app = FastAPI()
    app.include_router(upload_mod.router)
    return TestClient(app)


def test_upload_call_returns_transcript_trajectory_and_alerts(client, monkeypatch):
    scripted = [
        (
            "我們晚上要不要一起吃飯",
            SynthesizeResult(risk_level="low", chunk_risk_score=1, hard_triggers=[], justification="正常對話", case_memory_update=""),
        ),
        (
            "請提供簡訊驗證碼給我核對身分",
            SynthesizeResult(
                risk_level="high",
                chunk_risk_score=95,
                hard_triggers=[HardTriggerHit(name="OTP要求", fired=True, quote="請提供簡訊驗證碼")],
                justification="要求提供驗證碼",
                case_memory_update="已要求驗證碼",
            ),
        ),
    ]
    monkeypatch.setattr(upload_mod, "process_chunk", make_fake_process_chunk(scripted))

    r = client.post("/api/upload-call", files={"file": ("test.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()

    assert body["final_risk_level"] == "high"
    assert len(body["transcript"]) == 2
    assert len(body["risk_trajectory"]) == 2
    assert len(body["alerts"]) == 1
    assert body["alerts"][0]["trigger_name"] == "OTP要求"
    assert body["duration_seconds"] == pytest.approx(1.0, abs=0.01)

    # Persisted to history, distinguishable as "upload" not "live" — see the docs task's ask
    # that uploaded calls not be confused with live-monitored ones in the History screen.
    stored = history.get_call_detail(body["call_id"])
    assert stored is not None
    assert stored["source"] == "upload"
    assert stored["final_risk_level"] == "high"


def test_upload_call_skips_chunks_that_transcribe_to_nothing(client, monkeypatch):
    # process_chunk returns None for a VAD false-positive on non-speech noise (see its own
    # docstring in pipeline/chunk_worker.py) — the endpoint should just omit it, not error.
    monkeypatch.setattr(upload_mod, "process_chunk", lambda *a, **kw: None)

    r = client.post("/api/upload-call", files={"file": ("silence.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == []
    assert body["alerts"] == []
    assert body["final_risk_level"] == "low"


def test_upload_call_rejects_empty_file(client):
    r = client.post("/api/upload-call", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_upload_call_rejects_undecodable_audio(client):
    r = client.post("/api/upload-call", files={"file": ("bad.wav", b"this is not a wav file", "audio/wav")})
    assert r.status_code == 400
    assert "無法解析音訊檔案" in r.json()["detail"]


def test_upload_call_llm_failure_returns_502_not_500(client, monkeypatch):
    def raise_llm_error(*args, **kwargs):
        raise LLMProviderError("simulated provider outage")

    monkeypatch.setattr(upload_mod, "process_chunk", raise_llm_error)

    r = client.post("/api/upload-call", files={"file": ("test.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 502
    assert "simulated provider outage" in r.json()["detail"]

    # The call should still show up in history as a failed/errored entry, not vanish silently.
    calls = history.list_calls()
    assert any(c["ended_reason"] == "error" for c in calls)
