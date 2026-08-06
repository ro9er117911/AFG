"""POST /api/upload-call — analyze a pre-recorded call audio file through the same VAD
chunking + per-chunk pipeline the live WebSocket path uses (audio/vad.py,
pipeline/chunk_worker.py), for a user checking a recording after the fact rather than during
a live call.

Streamed as newline-delimited JSON (NDJSON), not one big JSON response at the end: per-chunk
ASR/acoustic/emotion results (and any live keyword hard-trigger alert) are pushed to the
client as soon as each utterance is processed, and the one-shot LLM final analysis — which can
take tens of seconds — arrives last as its own event. Mirrors server/ws.py's message
vocabulary (chunk_update/alert/final_analysis) so the frontend renders both the same way. A
plain single-JSON response would mean showing the user nothing at all until the slowest part
(the LLM call) finishes, which defeats the point of computing the fast local signals first.
"""

import asyncio
import dataclasses
import io
import json
import logging
from collections.abc import AsyncGenerator

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..audio.quality import detect_bandwidth
from ..audio.vad import VADChunker
from ..llm import LLMProvider, LLMProviderError, get_llm_provider
from ..pipeline.call_state import CallState
from ..pipeline.chunk_worker import process_chunk_signals, run_final_analysis
from ..storage import history
from ..storage.settings_store import load_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

SAMPLE_RATE = 16000


def _decode_audio(raw: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """Decode to float32 mono at SAMPLE_RATE, and detect whether the source is telephone-quality
    (narrowband) audio — e.g. a real call recorded via a PSTN/VoIP conferencing bridge, as
    opposed to wideband mic audio — see audio/quality.py's detect_bandwidth and
    docs/TEST_DATA_ACQUISITION.md for how to source real telephone-quality test recordings.

    soundfile (libsndfile) handles wav/flac/ogg natively, which covers the eval/test_clips
    convention this project already uses (16kHz mono wav, see eval/run_test_set.py) and the
    formats free conferencing bridges typically export (wav/mp3 — mp3 needs a libsndfile build
    with mp3 support, otherwise convert first). Browser recordings are often webm/mp4, which
    libsndfile can't read — rather than silently failing deep in sf.read with a cryptic error,
    or adding an ffmpeg subprocess dependency that may not exist on every deployment target,
    this raises a clear 400 telling the user what formats are supported; converting client-side
    or with ffmpeg before upload is on the user, documented in README.md's upload section.

    Returns (y, quality) where quality is detect_bandwidth()'s dict, computed on the audio
    *before* resampling so the spectral check sees the real recorded bandwidth.
    """
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"無法解析音訊檔案「{filename}」：{e}。"
                "請上傳 wav/flac/ogg 格式（電話品質的窄頻錄音，例如免費會議橋接電話錄音，也支援），"
                "其他格式（如手機錄音常見的 m4a/mp3）請先用 ffmpeg 轉檔，例如："
                "ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav"
            ),
        ) from e
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono, same as eval/run_test_set.py
    quality = detect_bandwidth(y, sr, native_sr=sr)
    if sr != SAMPLE_RATE:
        import librosa  # lazy import — only needed off the common 16kHz-wav path

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32), quality


async def stream_pipeline_over_audio(
    y: np.ndarray, provider: LLMProvider, call_state: CallState, hard_triggers: list[str]
) -> AsyncGenerator[dict, None]:
    """VAD-chunk a whole pre-recorded/local audio array, run process_chunk_signals over each
    utterance in order (ASR + acoustic + emotion + live keyword hard-trigger check — no LLM),
    yielding one event dict per result, then run_final_analysis once over the whole thing.
    Shared by /api/upload-call and the Test Data screen's /api/test-clips/.../analyze
    (server/testdata.py) — same pipeline, the only difference is where the audio bytes came
    from and what happens with the result afterward (history persistence vs. not, see each
    router function).

    Each process_chunk_signals/run_final_analysis call is blocking CPU work (whisper,
    parselmouth, TIMNet, the `claude` CLI subprocess) — run off the event loop via
    asyncio.to_thread so a slow upload doesn't stall the whole server, same reasoning as
    server/ws.py's _handle_chunk.

    VADChunker strips silence between utterances, so the sum of chunk lengths processed so far
    isn't exactly "position in the original file" (it omits silence-gap duration) — but it's a
    reasonable monotonically-increasing approximation of "roughly when in the recording this
    was said," which is all the transcript timestamps need to be useful for a post-hoc review.
    See CallState.record_chunk_signals()'s docstring for why an explicit timestamp is passed
    here instead of relying on its wall-clock default.
    """
    chunker = VADChunker(sample_rate=SAMPLE_RATE)
    chunks = chunker.push_audio(y)
    tail = chunker.flush()
    if tail is not None and len(tail) > 0:
        chunks.append(tail)

    audio_position_s = 0.0
    for chunk in chunks:
        result = await asyncio.to_thread(
            process_chunk_signals, chunk, SAMPLE_RATE, call_state, None, audio_position_s
        )
        audio_position_s += len(chunk) / SAMPLE_RATE
        if result is None:
            continue
        yield {
            "type": "chunk_update",
            "timestamp": call_state.transcript[-1].timestamp,
            "transcript_text": result.transcript_text,
            "acoustic": result.acoustic,
            "emotion": result.emotion,
            "deepfake": result.deepfake,
            "baseline": result.baseline,
            "egemaps": result.egemaps,
            "egemaps_baseline": result.egemaps_baseline,
        }
        if result.alert is not None:
            yield {"type": "alert", **dataclasses.asdict(result.alert)}

    outcome = await asyncio.to_thread(
        run_final_analysis, provider, call_state, hard_triggers, audio_position_s
    )
    if outcome is None:
        return  # nothing was ever transcribed — nothing to analyze

    result, alert, evidence = outcome
    yield {
        "type": "final_analysis",
        "risk_level": result.risk_level,
        "chunk_risk_score": result.chunk_risk_score,
        "justification": result.justification,
        "fraud_type": result.fraud_type.model_dump(),
        "fake_score": result.fake_score,
        "ai_voice_flag": result.ai_voice_flag,
        "fusion_source": result.fusion_source,
        "audio_quality": call_state.audio_quality,
        "evidence": {
            "transcript_segment": evidence.transcript_segment,
            "acoustic_summary": evidence.acoustic_summary,
            "emotion_summary": evidence.emotion_summary,
        },
    }
    if alert is not None:
        yield {"type": "alert", **dataclasses.asdict(alert)}


def final_risk_level(call_state: CallState) -> str:
    levels_by_severity = {"low": 0, "medium": 1, "high": 2}
    level = "low"
    for point in call_state.risk_trajectory:
        if levels_by_severity.get(point.risk_level, 0) > levels_by_severity.get(level, 0):
            level = point.risk_level
    return level


def _ndjson_line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@router.post("/upload-call")
async def upload_call(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上傳的檔案是空的")
    y, quality = _decode_audio(raw, file.filename or "upload")

    settings = load_settings()
    # A missing/misconfigured Claude provider is no longer fatal (two-line fusion architecture,
    # see reasoning/fusion.py) — proceed with provider=None, run_final_analysis's fallback
    # template still produces a verdict from Line 1/2's fusion result alone.
    try:
        provider = get_llm_provider()
    except LLMProviderError as e:
        logger.warning("could not get an LLM provider for upload analysis, proceeding without it: %s", e)
        provider = None

    call_state = CallState()
    call_state.audio_quality = quality
    call_id = history.create_call(source="upload")
    duration_seconds = len(y) / SAMPLE_RATE

    async def event_stream() -> AsyncGenerator[bytes, None]:
        # Once streaming starts, the HTTP status is already committed (200) — an error partway
        # through can no longer become a 502 the way the old single-JSON-response version did.
        # It's signaled in-band as a "error" event instead, same pattern as server/ws.py.
        #
        # Catches Exception broadly, not just LLMProviderError — confirmed directly (2026-08-06):
        # a CUDA OOM inside process_chunk_signals (or any other bug partway through a chunk) used
        # to propagate out of this generator uncaught. Starlette can't turn that into a clean HTTP
        # error once headers/chunks are already on the wire, so it just drops the connection —
        # which the browser reports as an opaque "network error"/ERR_INCOMPLETE_CHUNKED_ENCODING
        # with no indication of what actually went wrong. Catching it here turns that into a
        # normal in-band "error" event the frontend already knows how to show.
        alerts: list[dict] = []
        try:
            async for event in stream_pipeline_over_audio(y, provider, call_state, settings["hard_triggers"]):
                if event["type"] == "alert":
                    alerts.append({k: v for k, v in event.items() if k != "type"})
                yield _ndjson_line(event)
        except Exception as e:
            logger.exception("pipeline failed while analyzing uploaded call")
            history.finish_call(
                call_id, call_state, ended_reason="error", duration_seconds=duration_seconds, alerts=alerts
            )
            yield _ndjson_line({"type": "error", "message": str(e) or e.__class__.__name__})
            return

        history.finish_call(
            call_id, call_state, ended_reason="uploaded", duration_seconds=duration_seconds, alerts=alerts
        )
        yield _ndjson_line(
            {
                "type": "done",
                "call_id": call_id,
                "duration_seconds": duration_seconds,
                "final_risk_level": final_risk_level(call_state),
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
