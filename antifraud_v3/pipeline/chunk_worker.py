"""Orchestrates the per-chunk pipeline: ASR + acoustic features + emotion -> reasoning engine
-> call state update. See docs/DESIGN.md §2.2.
"""

from dataclasses import dataclass

import numpy as np

from ..asr.transcribe import transcribe_chunk
from ..audio.emotion import predict_emotion, summarize_emotion
from ..audio.features import extract_acoustic_features, summarize_acoustic_features
from ..llm.base import LLMProvider
from ..reasoning import run_reasoning_pipeline
from ..reasoning.schemas import ChunkEvidence, SynthesizeResult
from .call_state import Alert, CallState


@dataclass
class ChunkResult:
    transcript_text: str
    synthesize_result: SynthesizeResult
    alert: Alert | None


def process_chunk(
    y: np.ndarray,
    sr: int,
    provider: LLMProvider,
    call_state: CallState,
    speaker_guess: str | None = None,
    hard_triggers: list[str] | None = None,
    timestamp: float | None = None,
) -> ChunkResult | None:
    """Run the full per-chunk pipeline. Returns None if the chunk transcribed to nothing
    (e.g. a false-positive VAD trigger on non-speech noise) — the caller (server/ws.py)
    should skip pushing an update in that case.

    timestamp is forwarded to CallState.record_chunk() — see that method's docstring. The
    live WebSocket path (server/ws.py) leaves it None (wall-clock elapsed time is correct
    there); the upload path (server/upload.py) passes the chunk's position within the
    uploaded audio instead.
    """
    transcript_text = transcribe_chunk(y, sr)
    if not transcript_text:
        return None

    acoustic_features = extract_acoustic_features(y, sr)
    call_state.maybe_set_baseline(acoustic_features)

    _emotion_label, emotion_probs = predict_emotion(y, sr)

    evidence = ChunkEvidence(
        transcript_segment=transcript_text,
        speaker_guess=speaker_guess,
        acoustic_summary=summarize_acoustic_features(acoustic_features),
        emotion_summary=summarize_emotion(emotion_probs),
        call_state_summary=call_state.call_state_summary(),
    )

    result = run_reasoning_pipeline(provider, evidence, hard_triggers=hard_triggers)
    alert = call_state.record_chunk(transcript_text, speaker_guess, result, timestamp=timestamp)

    return ChunkResult(transcript_text=transcript_text, synthesize_result=result, alert=alert)
