"""POST /api/upload-call — analyze a pre-recorded call audio file through the same VAD
chunking + per-chunk pipeline the live WebSocket path uses (audio/vad.py,
pipeline/chunk_worker.py), for a user checking a recording after the fact rather than during
a live call.

Deliberately synchronous, not streamed over a WebSocket like server/ws.py: batch processing a
single pre-recorded call isn't latency-sensitive the way live capture is (docs/DESIGN.md §5's
whole reason for the WebSocket path is "warn *during* the call"), so reusing that incremental
push machinery here would just be a second state machine to maintain for no real benefit. The
client gets one response with the full transcript + risk trajectory + alerts once processing
finishes.
"""

import dataclasses
import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..audio.vad import VADChunker
from ..llm import LLMProviderError, get_llm_provider
from ..pipeline.call_state import CallState
from ..pipeline.chunk_worker import process_chunk
from ..storage import history
from ..storage.settings_store import load_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

SAMPLE_RATE = 16000


def _decode_audio(raw: bytes, filename: str) -> np.ndarray:
    """Decode to float32 mono at SAMPLE_RATE.

    soundfile (libsndfile) handles wav/flac/ogg natively, which covers the eval/test_clips
    convention this project already uses (16kHz mono wav, see eval/run_test_set.py). Browser
    recordings are often webm/mp4/mp3, which libsndfile can't read — rather than silently
    failing deep in sf.read with a cryptic error, or adding an ffmpeg subprocess dependency
    that may not exist on every deployment target, this raises a clear 400 telling the user
    what formats are supported; converting client-side or with ffmpeg before upload is on the
    user, documented in README.md's upload section.
    """
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"無法解析音訊檔案「{filename}」：{e}。"
                "請上傳 wav/flac/ogg 格式（16kHz 單聲道效果最好），"
                "其他格式（如手機錄音常見的 m4a/mp3）請先用 ffmpeg 轉檔，例如："
                "ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav"
            ),
        ) from e
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono, same as eval/run_test_set.py
    if sr != SAMPLE_RATE:
        import librosa  # lazy import — only needed off the common 16kHz-wav path

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32)


@router.post("/upload-call")
async def upload_call(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上傳的檔案是空的")
    y = _decode_audio(raw, file.filename or "upload")

    settings = load_settings()
    try:
        provider = get_llm_provider()
    except LLMProviderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    call_state = CallState(debounce_chunks=settings["debounce_chunks"])
    chunker = VADChunker(sample_rate=SAMPLE_RATE)

    call_id = history.create_call(source="upload")

    chunks = chunker.push_audio(y)
    tail = chunker.flush()
    if tail is not None and len(tail) > 0:
        chunks.append(tail)

    # VADChunker strips silence between utterances, so the sum of chunk lengths processed so
    # far isn't exactly "position in the original file" (it omits silence-gap duration) — but
    # it's a reasonable monotonically-increasing approximation of "roughly when in the
    # recording this was said," which is all the transcript timestamps need to be useful for
    # a post-hoc review. See CallState.record_chunk()'s docstring for why an explicit
    # timestamp is passed here instead of relying on its wall-clock default.
    audio_position_s = 0.0
    alerts: list[dict] = []

    try:
        for chunk in chunks:
            result = process_chunk(
                chunk,
                SAMPLE_RATE,
                provider,
                call_state,
                None,
                settings["hard_triggers"],
                timestamp=audio_position_s,
            )
            audio_position_s += len(chunk) / SAMPLE_RATE
            if result is None:
                continue
            if result.alert is not None:
                alerts.append(dataclasses.asdict(result.alert))
    except LLMProviderError as e:
        logger.warning("reasoning pipeline failed while analyzing uploaded call: %s", e)
        history.finish_call(
            call_id, call_state, ended_reason="error", duration_seconds=len(y) / SAMPLE_RATE
        )
        raise HTTPException(status_code=502, detail=str(e)) from e

    history.finish_call(
        call_id, call_state, ended_reason="uploaded", duration_seconds=len(y) / SAMPLE_RATE
    )

    levels_by_severity = {"low": 0, "medium": 1, "high": 2}
    final_risk_level = "low"
    for point in call_state.risk_trajectory:
        if levels_by_severity.get(point.risk_level, 0) > levels_by_severity.get(final_risk_level, 0):
            final_risk_level = point.risk_level

    return {
        "call_id": call_id,
        "duration_seconds": len(y) / SAMPLE_RATE,
        "final_risk_level": final_risk_level,
        "transcript": [dataclasses.asdict(t) for t in call_state.transcript],
        "risk_trajectory": [dataclasses.asdict(r) for r in call_state.risk_trajectory],
        "alerts": alerts,
    }
