"""pipeline/chunk_worker.py's process_chunk_signals() — previously only exercised indirectly
(server-level tests monkeypatch process_chunk_signals itself away entirely, see
test_upload_api.py), so its own wiring (ASR -> acoustic -> baseline -> emotion -> Line 1
deepfake scoring -> call_state recording -> live hard-trigger) was never directly unit tested.
Stubs every per-chunk predictor (transcribe_chunk/extract_acoustic_features/predict_emotion/
predict_deepfake) so this never loads a real model — see detectors/deepfake_voice.py's module
docstring for why the real XLS-R model shouldn't be part of the normal fast test loop.
"""

import numpy as np
import pytest

from antifraud_v3.detectors.deepfake_voice import DeepfakeChunkResult
from antifraud_v3.pipeline import chunk_worker
from antifraud_v3.pipeline.call_state import CallState

SAMPLE_RATE = 16000


def _fake_acoustic_features():
    return {
        "pitch": {"mean_pitch": 200.0, "std_pitch": 5.0, "pitch_instability": 1.0, "valid_samples": 10},
        "volume": {"mean_volume": 0.5, "std_volume": 0.05},
        "tremor": {"jitter_local": 0.8, "shimmer_local": 2.5, "hnr": 18.0},
        "speech_rate": {"speech_rate_variation": 0.1, "pause_ratio": 5.0},
    }


@pytest.fixture(autouse=True)
def stub_predictors(monkeypatch):
    monkeypatch.setattr(chunk_worker, "transcribe_chunk", lambda y, sr: "測試逐字稿")
    monkeypatch.setattr(chunk_worker, "extract_acoustic_features", lambda y, sr: _fake_acoustic_features())
    monkeypatch.setattr(chunk_worker, "predict_emotion", lambda y, sr: ("neutral", {"neutral": 0.9}))
    monkeypatch.setattr(
        chunk_worker, "predict_deepfake", lambda y, sr: DeepfakeChunkResult(fake_score=0.73, label="fake")
    )


def test_process_chunk_signals_includes_deepfake_field():
    call_state = CallState()
    y = np.zeros(SAMPLE_RATE, dtype=np.float32)
    result = chunk_worker.process_chunk_signals(y, SAMPLE_RATE, call_state)
    assert result is not None
    assert result.deepfake == {"fake_score": 0.73, "label": "fake"}


def test_process_chunk_signals_records_fake_score_and_raw_audio_into_call_state():
    call_state = CallState()
    y = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1
    chunk_worker.process_chunk_signals(y, SAMPLE_RATE, call_state)
    assert call_state.chunk_signals[-1].fake_score == 0.73
    assert len(call_state.chunk_audio) == 1
    assert np.array_equal(call_state.chunk_audio[0], y)


def test_process_chunk_signals_returns_none_on_empty_transcript(monkeypatch):
    monkeypatch.setattr(chunk_worker, "transcribe_chunk", lambda y, sr: "")
    call_state = CallState()
    y = np.zeros(SAMPLE_RATE, dtype=np.float32)
    result = chunk_worker.process_chunk_signals(y, SAMPLE_RATE, call_state)
    assert result is None
    # A VAD false-positive on non-speech noise shouldn't record anything, deepfake included.
    assert call_state.chunk_signals == []
    assert call_state.chunk_audio == []
