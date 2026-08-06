"""Runtime settings, persisted as a small JSON file rather than a DB table — it's a handful
of scalar/list values, not relational data. Backs the Settings screen (docs/DESIGN.md §9).
"""

import json
import os
from pathlib import Path

from ..reasoning.rubric import DEFAULT_HARD_TRIGGERS

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

# .env values seed these defaults; once settings.json exists (i.e. the Settings screen has
# saved at least once), it takes precedence — settings_store is the single source of truth
# that everything else (llm/__init__.py, server/ws.py) reads from, not raw os.getenv() calls
# scattered around.
DEFAULTS = {
    "llm_provider": os.getenv("LLM_PROVIDER", "claude_code"),
    "llm_model": os.getenv("LLM_MODEL", "claude-opus-5"),
    "llm_effort": os.getenv("LLM_EFFORT", "medium"),
    "hard_triggers": list(DEFAULT_HARD_TRIGGERS),
    # Line 2 (scam/fraud judgment) backend: "claude" reuses llm_provider/llm_model above (CLI or
    # API) and skips loading the local Qwen2-Audio-7B model entirely — default, since it's
    # meaningfully faster than the ~6.6GB 4-bit local model's load+3-round-generate cost.
    # "qwen2audio" runs detectors/scam_semantic.py's classify_call() as before.
    "line2_backend": os.getenv("LINE2_BACKEND", "claude"),
    # ASR backend for asr/transcribe.py's transcribe_chunk() — independent of line2_backend.
    "asr_backend": os.getenv("ASR_BACKEND", "whisper"),
    # Emotion backend for audio/emotion.py's predict_emotion() — "wavlm" (default,
    # detectors/emotion_wavlm.py, tiantiaf/wavlm-large-categorical-emotion) or "timnet" (the
    # original model, kept as a fallback/comparison option — user's own call to make optional
    # rather than a hard cutover).
    "emotion_backend": os.getenv("EMOTION_BACKEND", "wavlm"),
    # Whether run_final_analysis() calls Claude at all for the end-of-call justification prose.
    # Default off (the user isn't currently trying to evaluate this): the verdict itself
    # (risk_level/chunk_risk_score/fraud_type) never depended on Claude anyway (see
    # reasoning/fusion.py) — build_synthesize_result's template-fallback path already produces a
    # full result with provider=None, so turning this off costs nothing but the prose quality.
    "llm_final_summary_enabled": os.getenv("LLM_FINAL_SUMMARY_ENABLED", "false").lower() == "true",
}

_cache: dict | None = None


def load_settings() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            stored = json.load(f)
        _cache = {**DEFAULTS, **stored}
    else:
        _cache = dict(DEFAULTS)
    return _cache


def save_settings(updates: dict) -> dict:
    global _cache
    current = {**load_settings(), **updates}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    _cache = current
    return current
