"""Central device/model-repo/quantization config for the local ML detector models
(antifraud_v3/detectors/) — deployment-level infra config, distinct from
storage/settings_store.py's user-editable runtime settings (which the Settings screen can
change without a process restart). Everything here implies loading/reloading a multi-GB model,
so it's read once from process environment at import time, not hot-reloadable.

Does NOT replace TIMNet's (audio/emotion.py) or faster-whisper's (asr/transcribe.py) own
independent device checks — those are pre-existing, working, and tied to their own runtimes
(plain torch vs. ctranslate2). resolve_device() is the one new detectors/ code should call, so
there's a single source of truth for anything added going forward instead of a third ad hoc
check pattern.
"""

import os

import torch

_device: torch.device | None = None


def resolve_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ---- Line 1: nii-yamagishilab/xls-r-2b-anti-deepfake (detectors/deepfake_voice.py) ----
# Real load takes ~8.65GB fp32 on disk; cast to fp16 on load halves resident VRAM to ~4.3GB,
# confirmed by direct measurement. Peak VRAM during inference on a realistic utterance-sized
# chunk (3-15s, matching VADChunker's actual output) stays close to resident weight size
# (~4.3-4.7GB measured directly) — do not extrapolate from a much longer clip fed as one chunk,
# which spikes activation memory far higher (measured ~8.7GB for an 18s single input) and never
# happens in the real per-VAD-chunk pipeline.
DEEPFAKE_MODEL_ID = os.getenv("DEEPFAKE_MODEL_ID", "nii-yamagishilab/xls-r-2b-anti-deepfake")
# UNCALIBRATED — no real (non-TTS) fake/real voice clips exist in this project yet to tune
# against; see detectors/deepfake_voice.py's module docstring for why the existing eval/
# test_clips corpus can't be used to calibrate this (it's entirely TTS-synthesized, so it
# reads as "fake" almost by definition regardless of scam/benign content).
DEEPFAKE_FAKE_SCORE_THRESHOLD = float(os.getenv("DEEPFAKE_FAKE_SCORE_THRESHOLD", "0.85"))

# ---- Line 2: JimmyMa99/AntiFraud-SFT (detectors/scam_semantic.py) ----
# The TeleAntiFraud-28k paper's fine-tuned Qwen2-Audio-7B-Instruct checkpoint (references/
# 05-teleantifraud.md) — confirmed via the model's own HF repo (Apache 2.0, ungated, ~16.8GB
# bf16 on disk). Its Whisper-derived audio feature extractor's own preprocessor_config.json
# declares chunk_length=30 (seconds) / n_samples=480000 — the architecture is built around
# 30s audio windows, which is why SCAM_SEMANTIC_MAX_AUDIO_SECONDS defaults to 30, not a guess.
SCAM_SEMANTIC_MODEL_ID = os.getenv("SCAM_SEMANTIC_MODEL_ID", "JimmyMa99/AntiFraud-SFT")
SCAM_SEMANTIC_LOAD_IN_4BIT = os.getenv("SCAM_SEMANTIC_LOAD_IN_4BIT", "true").lower() == "true"
SCAM_SEMANTIC_MAX_AUDIO_SECONDS = int(os.getenv("SCAM_SEMANTIC_MAX_AUDIO_SECONDS", "30"))
SCAM_SEMANTIC_MAX_NEW_TOKENS = int(os.getenv("SCAM_SEMANTIC_MAX_NEW_TOKENS", "512"))

# ---- Emotion, optional WavLM backend: tiantiaf/wavlm-large-categorical-emotion
# (detectors/emotion_wavlm.py) — an alternative to audio/emotion.py's TIMNet, selected via
# storage/settings_store.py's "emotion_backend" setting. WavLM-large + downstream heads,
# ~317M params (~1.2GB fp32 on disk) — confirmed ungated, Open RAIL (non-commercial) license,
# via the model's own HF repo. See models/wavlm_emotion.py for the ported architecture.
EMOTION_WAVLM_MODEL_ID = os.getenv("EMOTION_WAVLM_MODEL_ID", "tiantiaf/wavlm-large-categorical-emotion")
