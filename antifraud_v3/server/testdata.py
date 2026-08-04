"""Test Data screen (frontend "測試資料" tab) — lets the user replay the eval/test_clips/
audio used for regression testing (eval/run_test_set.py) and see it run through the exact same
streaming pipeline as a live call or upload, with playback, timing, transcript, acoustic/
emotion signals, and the final LLM analysis.

Deliberately NOT persisted to storage/history.py: these are reference/demo clips, not real
calls — writing them into the call history would recreate the exact clutter the history screen
was just cleaned up from. Results only live in the HTTP response; the frontend holds them for
the duration of the tab visit.
"""

import time
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..llm import LLMProviderError, get_llm_provider
from ..pipeline.call_state import CallState
from ..storage.settings_store import load_settings
from .upload import SAMPLE_RATE, _decode_audio, _ndjson_line, final_risk_level, stream_pipeline_over_audio

router = APIRouter(prefix="/api/test-clips")

CLIPS_DIR = Path(__file__).resolve().parent.parent / "eval" / "test_clips"
CATEGORIES = ("benign", "scam")


def _safe_clip_path(category: str, filename: str) -> Path:
    if category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"未知的分類：{category}")
    base = (CLIPS_DIR / category).resolve()
    path = (base / filename).resolve()
    # filename comes straight from the URL path — reject anything that would escape the
    # category directory (e.g. "../../secrets") rather than trusting it's a bare filename.
    if path != base / filename or base not in path.parents:
        raise HTTPException(status_code=400, detail="無效的檔名")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="找不到這個測試音檔")
    return path


@router.get("")
def list_test_clips():
    return {category: sorted(p.name for p in (CLIPS_DIR / category).glob("*.wav")) for category in CATEGORIES}


@router.get("/{category}/{filename}/audio")
def get_test_clip_audio(category: str, filename: str):
    path = _safe_clip_path(category, filename)
    return FileResponse(path, media_type="audio/wav")


@router.post("/{category}/{filename}/analyze")
def analyze_test_clip(category: str, filename: str):
    path = _safe_clip_path(category, filename)
    y = _decode_audio(path.read_bytes(), filename)

    settings = load_settings()
    try:
        provider = get_llm_provider()
    except LLMProviderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    call_state = CallState()
    duration_seconds = len(y) / SAMPLE_RATE
    start = time.monotonic()

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for event in stream_pipeline_over_audio(y, provider, call_state, settings["hard_triggers"]):
                yield _ndjson_line(event)
        except LLMProviderError as e:
            yield _ndjson_line({"type": "error", "message": str(e)})
            return

        yield _ndjson_line(
            {
                "type": "done",
                "category": category,
                "filename": filename,
                "duration_seconds": duration_seconds,
                "processing_seconds": time.monotonic() - start,
                "final_risk_level": final_risk_level(call_state),
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
