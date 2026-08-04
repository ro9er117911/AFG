"""Two entry points, deliberately split:

- process_chunk_signals(): runs on every completed VAD chunk, in real time. ASR + acoustic
  features + emotion + a cheap keyword-based hard-trigger check — no LLM call. This is what
  keeps the live/streaming experience actually real-time and free, per the project's current
  budget constraint (no paid ANTHROPIC_API_KEY — see llm/claude_code_provider.py). The result
  carries a compact numeric acoustic/emotion snapshot so callers (server/ws.py,
  server/upload.py, server/testdata.py) can push it to the client immediately for a live
  chart/badge, not just log it silently for later.
- run_final_analysis(): runs once, when the call ends. Aggregates every chunk's signals into
  one call-level summary and runs the discriminate/reflect/synthesize reasoning pipeline
  (reasoning/__init__.py) exactly once over it — replacing the old design where that pipeline
  ran 2-3 times per chunk (a real cost multiplier that was only affordable with per-token API
  billing, see docs/DESIGN.md §8). Also returns the exact ChunkEvidence that was sent to the
  LLM, so callers can show the user what the AI actually saw, not just its verdict.
"""

from dataclasses import dataclass

import numpy as np

from ..asr.transcribe import transcribe_chunk
from ..audio.emotion import aggregate_emotion, predict_emotion, summarize_emotion
from ..audio.features import aggregate_acoustic_features, extract_acoustic_features, summarize_acoustic_features
from ..llm.base import LLMProvider
from ..reasoning import run_reasoning_pipeline
from ..reasoning.schemas import ChunkEvidence, SynthesizeResult
from .call_state import Alert, CallState


def _compact_acoustic(features: dict) -> dict:
    """Pulls out just the numbers worth charting/reading live, from extract_acoustic_features()'s
    full nested dict — the rest (pitch_trend, sudden_speed_changes, etc.) still lives in
    call_state.chunk_signals for the final aggregate, but isn't interesting to show per-chunk."""
    p, v, t, sr = features["pitch"], features["volume"], features["tremor"], features["speech_rate"]
    return {
        "mean_pitch": round(p["mean_pitch"], 1) if p["valid_samples"] > 0 else None,
        "mean_volume": round(v["mean_volume"], 4),
        "jitter_local": round(t["jitter_local"], 2),
        "shimmer_local": round(t["shimmer_local"], 2),
        "hnr": round(t["hnr"], 1),
        "pause_ratio": round(sr["pause_ratio"], 1),
        "speech_rate_variation": round(sr["speech_rate_variation"], 2),
    }


def _compact_emotion(label: str, probs: dict[str, float]) -> dict:
    return {"label": label, "top_prob": round(probs[label], 2)}


@dataclass
class ChunkSignalsResult:
    transcript_text: str
    acoustic: dict
    emotion: dict
    alert: Alert | None  # from the live keyword hard-trigger check, not the LLM


def process_chunk_signals(
    y: np.ndarray,
    sr: int,
    call_state: CallState,
    speaker_guess: str | None = None,
    timestamp: float | None = None,
) -> ChunkSignalsResult | None:
    """Run ASR + acoustic + emotion for one completed chunk and record it into call_state.
    Returns None if the chunk transcribed to nothing (e.g. a false-positive VAD trigger on
    non-speech noise) — the caller (server/ws.py, server/upload.py) should skip pushing an
    update in that case.

    timestamp is forwarded to CallState.record_chunk_signals() — see that method's docstring.
    """
    transcript_text = transcribe_chunk(y, sr)
    if not transcript_text:
        return None

    acoustic_features = extract_acoustic_features(y, sr)
    call_state.maybe_set_baseline(acoustic_features)
    emotion_label, emotion_probs = predict_emotion(y, sr)

    call_state.record_chunk_signals(
        transcript_text, speaker_guess, acoustic_features, emotion_probs, timestamp=timestamp
    )
    alert = call_state.check_live_hard_trigger(transcript_text)
    return ChunkSignalsResult(
        transcript_text=transcript_text,
        acoustic=_compact_acoustic(acoustic_features),
        emotion=_compact_emotion(emotion_label, emotion_probs),
        alert=alert,
    )


def run_final_analysis(
    provider: LLMProvider,
    call_state: CallState,
    hard_triggers: list[str] | None = None,
    timestamp: float | None = None,
) -> tuple[SynthesizeResult, Alert | None, ChunkEvidence] | None:
    """Run discriminate -> reflect -> synthesize once over the whole accumulated call. Returns
    None if nothing was ever transcribed (e.g. a silent/empty call) — there's nothing to
    reason about. Call once, at call end (server/ws.py's call_socket finally-block,
    server/upload.py's stream_pipeline_over_audio).

    The returned ChunkEvidence is exactly what was sent to the LLM (discriminate.py builds its
    prompt straight from these fields) — callers surface it to the user so "what did you tell
    the AI" isn't a black box.
    """
    if not call_state.transcript:
        return None

    full_transcript = "\n".join(
        f"[{t.timestamp:.1f}s] {t.text}" for t in call_state.transcript
    )
    acoustic_summary = summarize_acoustic_features(
        aggregate_acoustic_features([cs.acoustic_features for cs in call_state.chunk_signals])
    )
    emotion_summary = summarize_emotion(
        aggregate_emotion([cs.emotion_probs for cs in call_state.chunk_signals])
    )

    evidence = ChunkEvidence(
        transcript_segment=full_transcript,
        speaker_guess=None,
        acoustic_summary=acoustic_summary,
        emotion_summary=emotion_summary,
        call_state_summary="",
    )
    result = run_reasoning_pipeline(provider, evidence, hard_triggers=hard_triggers)
    alert = call_state.apply_final_result(result, timestamp=timestamp)
    return result, alert, evidence
