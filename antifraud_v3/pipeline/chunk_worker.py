"""Two entry points, deliberately split:

- process_chunk_signals(): runs on every completed VAD chunk, in real time. ASR + acoustic
  features + emotion + Line 1 AI/cloned-voice scoring (detectors/deepfake_voice.py) + a cheap
  keyword-based hard-trigger check — no LLM call. This is what
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

import logging
from dataclasses import dataclass

import numpy as np

from .. import config
from ..asr.transcribe import transcribe_chunk
from ..audio.egemaps import aggregate_egemaps, extract_egemaps, summarize_egemaps_highlight
from ..audio.emotion import aggregate_emotion, predict_emotion, summarize_emotion
from ..audio.features import aggregate_acoustic_features, extract_acoustic_features, summarize_acoustic_features
from ..detectors.deepfake_voice import (
    DeepfakeChunkResult,
    aggregate_deepfake,
    predict_deepfake,
    unload_model as unload_deepfake_model,
)
from ..detectors.scam_semantic import ScamSemanticError, classify_call
from ..detectors.scam_semantic import unload_model as unload_scam_semantic_model
from ..detectors.scam_semantic_llm import classify_call_via_llm
from ..llm.base import LLMProvider, LLMProviderError
from ..reasoning import run_reasoning_pipeline
from ..reasoning.discriminate import discriminate
from ..reasoning.fusion import build_synthesize_result, fuse
from ..reasoning.pattern_matcher import match_patterns
from ..reasoning.schemas import ChunkEvidence, MatchedPattern, SynthesizeResult
from ..storage.settings_store import load_settings
from .call_state import Alert, CallState

logger = logging.getLogger(__name__)


def _compact_acoustic(features: dict) -> dict:
    """Pulls out just the numbers worth charting/reading live, from extract_acoustic_features()'s
    full nested dict — the rest (pitch_trend, sudden_speed_changes, etc.) still lives in
    call_state.chunk_signals for the final aggregate, but isn't interesting to show per-chunk."""
    p, v, t, sr = features["pitch"], features["volume"], features["tremor"], features["speech_rate"]
    return {
        "mean_pitch": round(p["mean_pitch"], 1) if p["valid_samples"] > 0 else None,
        "pitch_instability": round(p["pitch_instability"], 2) if p["valid_samples"] > 0 else None,
        "mean_volume": round(v["mean_volume"], 4),
        "jitter_local": round(t["jitter_local"], 2),
        "shimmer_local": round(t["shimmer_local"], 2),
        "hnr": round(t["hnr"], 1),
        "pause_ratio": round(sr["pause_ratio"], 1),
        "speech_rate_variation": round(sr["speech_rate_variation"], 2),
    }


def _compact_emotion(label: str, probs: dict[str, float]) -> dict:
    return {"label": label, "top_prob": round(probs[label], 2)}


def _compact_deepfake(result) -> dict:
    return {"fake_score": round(result.fake_score, 3), "label": result.label}


def _compact_egemaps(features: dict[str, float]) -> dict[str, float]:
    """Unlike _compact_acoustic, this keeps all 88 dims, not a curated subset — the frontend's
    baseline/z-score computation for the eGeMAPS detail panel follows the same "computed
    client-side from data already pushed to this view" pattern as the existing 8-axis radar
    (see frontend/app.js's RADAR_AXES comment), which needs the full per-key history, not just
    audio/egemaps.py's EGEMAPS_HIGHLIGHT_KEYS subset. Rounded to 3dp — plenty of precision for
    display, keeps the per-chunk WS/NDJSON payload reasonable (88 floats either way)."""
    return {k: round(v, 3) for k, v in features.items()}


@dataclass
class ChunkSignalsResult:
    transcript_text: str
    acoustic: dict
    emotion: dict
    deepfake: dict  # Line 1 (detectors/deepfake_voice.py) — {"fake_score": float, "label": str}
    alert: Alert | None  # from the live keyword hard-trigger check, not the LLM
    # Only non-None on the one chunk where CallState.maybe_set_baseline() just established it —
    # every other chunk carries None so callers don't resend/rechart an unchanged baseline.
    baseline: dict | None = None
    egemaps: dict[str, float] | None = None  # all 88 eGeMAPSv02 functionals, see _compact_egemaps
    # Companion to `baseline` above, same "only set on the establishing chunk" rule — kept
    # separate rather than nested inside `baseline` since audio/egemaps.py's baseline is a
    # distinct 88-key dict from audio/features.py's 10-key one (see CallState.egemaps_baseline).
    egemaps_baseline: dict[str, float] | None = None


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

    timestamp is forwarded to CallState.record_chunk_signals() — see that method's docstring for
    why batch/upload callers need the override. (maybe_set_baseline() no longer takes a
    timestamp — it gates on chunk count now, see its docstring.)
    """
    transcript_text = transcribe_chunk(y, sr)
    if not transcript_text:
        return None

    acoustic_features = extract_acoustic_features(y, sr)
    egemaps_features = extract_egemaps(y, sr)
    baseline_just_set = call_state.maybe_set_baseline(acoustic_features, egemaps_features)
    emotion_label, emotion_probs = predict_emotion(y, sr)
    # Line 1 is best-effort: it needs a CUDA GPU and a ~4.3GB model, and neither is guaranteed
    # (no-GPU machines, model download failures, OOM). An unguarded failure here propagated out
    # through upload.py's asyncio.to_thread and killed the whole streaming response mid-call —
    # the client got HTTP 200 with a truncated body and no verdict at all. Degrade instead:
    # fake_score 0.0 means "Line 1 contributed nothing", which fuse() already handles by falling
    # through to Line 2's branch. The "unavailable" label rides out on the existing deepfake
    # field (_compact_deepfake) so the UI can distinguish "not checked" from a real 0.0 score —
    # reporting the former as "confirmed not a deepfake" would be a different, false claim.
    try:
        deepfake_result = predict_deepfake(y, sr)
    except Exception:
        logger.warning("Line 1 (deepfake) unavailable for this chunk; continuing without it", exc_info=True)
        deepfake_result = DeepfakeChunkResult(fake_score=0.0, label="unavailable")

    call_state.record_chunk_signals(
        transcript_text,
        speaker_guess,
        acoustic_features,
        emotion_probs,
        deepfake_result.fake_score,
        y,
        timestamp=timestamp,
        egemaps_features=egemaps_features,
    )
    alert = call_state.check_live_hard_trigger(transcript_text)
    return ChunkSignalsResult(
        transcript_text=transcript_text,
        acoustic=_compact_acoustic(acoustic_features),
        emotion=_compact_emotion(emotion_label, emotion_probs),
        deepfake=_compact_deepfake(deepfake_result),
        alert=alert,
        baseline=call_state.baseline if baseline_just_set else None,
        egemaps=_compact_egemaps(egemaps_features),
        egemaps_baseline=call_state.egemaps_baseline if baseline_just_set else None,
    )


def run_final_analysis(
    provider: LLMProvider | None,
    call_state: CallState,
    hard_triggers: list[str] | None = None,
    timestamp: float | None = None,
) -> tuple[SynthesizeResult, Alert | None, ChunkEvidence] | None:
    """Runs once, at call end (server/ws.py's call_socket finally-block, server/upload.py's
    stream_pipeline_over_audio). Returns None if nothing was ever transcribed (e.g. a silent/
    empty call) — there's nothing to reason about.

    Two-line fusion architecture (see reasoning/fusion.py's module docstring): aggregates Line
    1's (detectors/deepfake_voice.py) per-chunk fake_score, runs Line 2
    (detectors/scam_semantic.py) once over the whole call's retained audio, fuses the two into
    the authoritative verdict (fuse()), then optionally asks Claude (discriminate/reflect/
    synthesize) only for hard-trigger evaluation + prose consistent with that verdict —
    `provider` is now optional; a missing/failed LLM degrades to a template-fallback
    justification (build_synthesize_result) rather than failing the whole analysis, since the
    verdict itself no longer depends on the LLM at all.

    The returned ChunkEvidence is exactly what was sent to the LLM (discriminate.py builds its
    prompt straight from these fields) — callers surface it to the user so "what did you tell
    the AI" isn't a black box. When provider is None or fails, evidence is still built (Line
    1/2's fusion doesn't need it) so callers keep a consistent return shape.
    """
    if not call_state.transcript:
        return None

    full_transcript = "\n".join(
        f"[{t.timestamp:.1f}s] {t.text}" for t in call_state.transcript
    )
    acoustic_summary = summarize_acoustic_features(
        aggregate_acoustic_features([cs.acoustic_features for cs in call_state.chunk_signals])
    )
    egemaps_list = [cs.egemaps_features for cs in call_state.chunk_signals if cs.egemaps_features]
    if egemaps_list:
        # Curated 10-key highlight only — the full 88 dims are for the UI's detail panel, not
        # worth spending LLM prompt tokens on (see audio/egemaps.py's summarize_egemaps_highlight).
        acoustic_summary += "；eGeMAPS：" + summarize_egemaps_highlight(aggregate_egemaps(egemaps_list))
    if call_state.audio_quality and call_state.audio_quality.get("narrowband"):
        # See audio/quality.py's detect_bandwidth — set by server/upload.py for pre-recorded
        # telephone-quality uploads (never for live mic calls). Appended as plain text into the
        # same prompt field the LLM already reads, consistent with the project's existing
        # "acoustic evidence is weak/contextual, not a hard signal" philosophy (rubric.py) —
        # not a new numeric threshold, just a caveat on how much to trust it.
        acoustic_summary += "；（此錄音為電話頻寬音訊，聲學細節可信度較低，應視為弱證據）"
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

    fake_agg = aggregate_deepfake([cs.fake_score for cs in call_state.chunk_signals])
    full_audio = (
        np.concatenate(call_state.chunk_audio) if call_state.chunk_audio else np.zeros(0, dtype=np.float32)
    )
    line2_result = None
    line2_backend = load_settings()["line2_backend"]
    if line2_backend == "claude" and provider is not None:
        # Text-only path (detectors/scam_semantic_llm.py) — reuses the same provider already
        # resolved for the reasoning pipeline below, no GPU model load/unload at all. Default
        # backend: meaningfully faster than Qwen2Audio's load+3-round-generate cost.
        try:
            line2_result = classify_call_via_llm(provider, full_transcript)
        except ScamSemanticError:
            logger.exception("Line 2 (claude backend) classification failed; falling back to Qwen2Audio")

    if line2_result is None and len(full_audio) > 0:
        # Reached when line2_backend=="qwen2audio", or as a fallback when the claude backend
        # was selected but unavailable/failed (provider is None or classify_call_via_llm raised)
        # — never silently give up on Line 2 just because the fast path wasn't usable this call.
        #
        # Frees Line 1's resident VRAM before Line 2 loads, and Line 2's after it's done —
        # measured directly, both models resident together OOM on this project's GPU, in either
        # loading order. See detectors/deepfake_voice.py's and detectors/scam_semantic.py's
        # unload_model() docstrings for the full measurement and trade-off. try/finally so a
        # failed classify_call() still frees Line 2's VRAM, not just a successful one.
        unload_deepfake_model()
        try:
            line2_result = classify_call(full_audio, 16000, full_transcript)
        except Exception:
            # Deliberately broader than ScamSemanticError: load_model() reaches
            # Qwen2AudioForConditionalGeneration.from_pretrained, which raises transformers'
            # and bitsandbytes' own exception types (e.g. ImportError when bitsandbytes is
            # missing for the 4-bit load). Those escaped the old ScamSemanticError-only clause
            # and killed the whole streaming response — the same failure mode Line 1 had.
            # fuse() already accepts line2=None, so degrading here is the intended path.
            logger.exception("Line 2 (scam-semantic) classification failed; proceeding without it")
        finally:
            unload_scam_semantic_model()

    fusion_result = fuse(fake_agg["max"], line2_result, config.DEEPFAKE_FAKE_SCORE_THRESHOLD)

    claude_result = None
    if provider is not None and load_settings().get("llm_final_summary_enabled", False):
        try:
            claude_result = run_reasoning_pipeline(provider, evidence, fusion_result, hard_triggers=hard_triggers)
        except LLMProviderError:
            logger.warning("Claude reasoning pipeline failed; falling back to template justification", exc_info=True)

    # 專利 TW I904863 步驟 S312：判定成立詐騙行為後，才依文字情緒 + 語意不合理特徵標記十種詐欺
    # 模式。Gated on fusion_result.is_fraud (fuse()'s verdict), NOT on llm_final_summary_enabled
    # — that setting governs whether Claude writes *prose*, and defaults to False, so hanging the
    # patent path off it would silently produce zero marks on every call. discriminate() is
    # called directly here rather than via run_reasoning_pipeline() because only its two evidence
    # blocks are needed; reflect/synthesize add nothing this lookup reads.
    matched_patterns: list[MatchedPattern] = []
    if fusion_result.is_fraud and provider is not None:
        try:
            discriminated = discriminate(provider, evidence)
            matched_patterns = match_patterns(discriminated.text_emotions, discriminated.semantic_features)
        except LLMProviderError:
            logger.warning("S312 pattern marking skipped: discriminate() failed", exc_info=True)

    result = build_synthesize_result(fusion_result, claude_result, matched_patterns)
    alert = call_state.apply_final_result(result, timestamp=timestamp)
    return result, alert, evidence
