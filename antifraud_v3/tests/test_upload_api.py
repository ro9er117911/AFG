"""server/upload.py's POST /api/upload-call (docs task: "audio file upload — analyze a
pre-recorded call"). VADChunker, process_chunk_signals and run_final_analysis are mocked here
— this test is checking the endpoint's own streaming contract (event shapes/order, history
persistence, error handling), not VAD/ASR/LLM accuracy, which is exercised separately by real
audio through the real pipeline (see the manual end-to-end test with eval/test_clips/*.wav and
a real browser, described in the session report — deliberately not re-mocked into a "fast unit
test" here, per this project's repeated lesson that mocked-only testing misses real bugs the
real pipeline would catch).

The endpoint returns newline-delimited JSON (server/upload.py's stream_pipeline_over_audio),
not one JSON blob — FastAPI's TestClient buffers the whole streamed body synchronously, so
tests just split the response text on newlines and parse each line, same as a real client's
final result would look once fully read.
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

import antifraud_v3.server.upload as upload_mod
from antifraud_v3.llm import LLMProviderError
from antifraud_v3.pipeline.chunk_worker import ChunkSignalsResult
from antifraud_v3.reasoning.schemas import ChunkEvidence, HardTriggerHit, SynthesizeResult
from antifraud_v3.storage import history

EMPTY_ACOUSTIC_FEATURES = {
    "pitch": {"mean_pitch": 0.0, "valid_samples": 0},
    "volume": {"mean_volume": 0.0},
    "speech_rate": {"speech_rate_variation": 0.0},
}
FAKE_COMPACT_ACOUSTIC = {"mean_pitch": None, "mean_volume": 0.0, "jitter_local": 0.0, "shimmer_local": 0.0, "hnr": 0.0, "pause_ratio": 0.0, "speech_rate_variation": 0.0}
FAKE_COMPACT_EMOTION = {"label": "neutral", "top_prob": 0.9}


def parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def make_fake_process_chunk_signals(scripted_texts):
    """Real process_chunk_signals's contract includes a side effect — it calls
    call_state.record_chunk_signals() and call_state.check_live_hard_trigger(), which is what
    actually populates CallState.transcript and runs the real live keyword-alert logic
    (pipeline/call_state.py). A mock that only returns a canned result without replicating
    that side effect would silently leave call_state empty, since the final "done" event is
    built from call_state, not from the raw per-chunk return values — this fake calls the
    real record_chunk_signals()/check_live_hard_trigger() so the alert logic under test is the
    real one, only ASR/acoustic/emotion inference is stubbed out.
    """
    texts = iter(scripted_texts)

    def fake(y, sr, call_state, speaker_guess=None, timestamp=None):
        text = next(texts, None)
        if text is None:
            return None
        call_state.record_chunk_signals(text, speaker_guess, EMPTY_ACOUSTIC_FEATURES, {}, timestamp=timestamp)
        alert = call_state.check_live_hard_trigger(text)
        return ChunkSignalsResult(
            transcript_text=text, acoustic=FAKE_COMPACT_ACOUSTIC, emotion=FAKE_COMPACT_EMOTION, alert=alert
        )

    return fake


def make_fake_run_final_analysis(result: SynthesizeResult):
    """Same idea as make_fake_process_chunk_signals: calls the real
    call_state.apply_final_result() so the alert-firing logic under test is real, only the
    LLM reasoning pipeline itself is stubbed out."""

    def fake(provider, call_state, hard_triggers=None, timestamp=None):
        if not call_state.transcript:
            return None
        alert = call_state.apply_final_result(result, timestamp=timestamp)
        evidence = ChunkEvidence(
            transcript_segment="fake transcript",
            acoustic_summary="fake acoustic summary",
            emotion_summary="fake emotion summary",
            call_state_summary="",
        )
        return result, alert, evidence

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


def test_upload_call_streams_chunk_updates_alerts_and_final_analysis(client, monkeypatch):
    monkeypatch.setattr(
        upload_mod,
        "process_chunk_signals",
        make_fake_process_chunk_signals(["我們晚上要不要一起吃飯", "請提供簡訊驗證碼給我核對身分"]),
    )
    monkeypatch.setattr(
        upload_mod,
        "run_final_analysis",
        make_fake_run_final_analysis(
            SynthesizeResult(
                risk_level="high",
                chunk_risk_score=95,
                hard_triggers=[HardTriggerHit(name="OTP要求", fired=True, quote="請提供簡訊驗證碼")],
                justification="要求提供驗證碼",
                case_memory_update="已要求驗證碼",
            )
        ),
    )

    r = client.post("/api/upload-call", files={"file": ("test.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    events = parse_ndjson(r.text)
    types = [e["type"] for e in events]

    # Two chunk_updates (each carrying acoustic/emotion, not an LLM verdict), one live-keyword
    # alert (fires on the second chunk's "驗證碼"), then the one-shot final_analysis — the
    # LLM-based hard trigger in the scripted result doesn't re-fire on top of it (see
    # pipeline/call_state.py's apply_final_result docstring) — and a closing "done" summary.
    assert types == ["chunk_update", "chunk_update", "alert", "final_analysis", "done"]
    assert "acoustic" in events[0] and "emotion" in events[0]
    assert events[2]["reason"] == "hard_trigger"
    assert events[3]["risk_level"] == "high"
    assert "evidence" in events[3]

    done = events[-1]
    assert done["final_risk_level"] == "high"
    assert done["duration_seconds"] == pytest.approx(1.0, abs=0.01)

    # Persisted to history, distinguishable as "upload" not "live" — see the docs task's ask
    # that uploaded calls not be confused with live-monitored ones in the History screen.
    stored = history.get_call_detail(done["call_id"])
    assert stored is not None
    assert stored["source"] == "upload"
    assert stored["final_risk_level"] == "high"
    assert len(stored["transcript"]) == 2
    assert len(stored["alerts"]) == 1


def test_upload_call_skips_chunks_that_transcribe_to_nothing(client, monkeypatch):
    # process_chunk_signals returns None for a VAD false-positive on non-speech noise (see its
    # own docstring in pipeline/chunk_worker.py) — the endpoint should just omit it, not error.
    # run_final_analysis is left as the real implementation here: with an empty transcript it
    # returns None before ever touching the (fake, arbitrary-object) provider.
    monkeypatch.setattr(upload_mod, "process_chunk_signals", lambda *a, **kw: None)

    r = client.post("/api/upload-call", files={"file": ("silence.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    events = parse_ndjson(r.text)
    # No chunk_update/alert/final_analysis at all — just the closing summary.
    assert [e["type"] for e in events] == ["done"]
    assert events[0]["final_risk_level"] == "low"

    stored = history.get_call_detail(events[0]["call_id"])
    assert stored["transcript"] == []
    assert stored["alerts"] == []


def test_upload_call_rejects_empty_file(client):
    r = client.post("/api/upload-call", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_upload_call_rejects_undecodable_audio(client):
    r = client.post("/api/upload-call", files={"file": ("bad.wav", b"this is not a wav file", "audio/wav")})
    assert r.status_code == 400
    assert "無法解析音訊檔案" in r.json()["detail"]


def test_upload_call_llm_failure_streams_error_event_not_502(client, monkeypatch):
    # Once streaming has started, the HTTP status is already committed to 200 (chunk_update
    # events may already be on the wire) — an LLM failure partway through can no longer become
    # a 502 the way the old single-JSON-response version did. It's signaled in-band as an
    # "error" event instead, same pattern as server/ws.py's final-analysis error handling.
    monkeypatch.setattr(upload_mod, "process_chunk_signals", make_fake_process_chunk_signals(["測試逐字稿"]))

    def raise_llm_error(*args, **kwargs):
        raise LLMProviderError("simulated provider outage")

    monkeypatch.setattr(upload_mod, "run_final_analysis", raise_llm_error)

    r = client.post("/api/upload-call", files={"file": ("test.wav", make_wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    events = parse_ndjson(r.text)
    assert events[0]["type"] == "chunk_update"
    assert events[-1] == {"type": "error", "message": "simulated provider outage"}

    # The call should still show up in history as a failed/errored entry, not vanish silently.
    calls = history.list_calls()
    assert any(c["ended_reason"] == "error" for c in calls)
