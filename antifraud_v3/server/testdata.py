"""Test Data screen (frontend "測試資料" tab) — lets the user replay the eval/test_clips/
audio used for regression testing (eval/run_test_set.py) and see it run through the exact same
streaming pipeline as a live call or upload, with playback, timing, transcript, acoustic/
emotion signals, and the final LLM analysis.

Deliberately NOT persisted to storage/history.py: these are reference/demo clips, not real
calls — writing them into the call history would recreate the exact clutter the history screen
was just cleaned up from. Results only live in the HTTP response; the frontend holds them for
the duration of the tab visit.
"""

import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..llm import LLMProviderError, get_llm_provider
from ..pipeline.call_state import CallState
from ..storage.settings_store import load_settings
from .upload import SAMPLE_RATE, _decode_audio, _ndjson_line, final_risk_level, stream_pipeline_over_audio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-clips")

CLIPS_DIR = Path(__file__).resolve().parent.parent / "eval" / "test_clips"


def _discover_categories() -> tuple[str, ...]:
    """Any subdirectory of eval/test_clips/ containing at least one .wav is a category —
    not hardcoded to ("benign", "scam") so dropping in a new folder (e.g. real human-voice
    clips, or clips grouped by emotion) is enough to add a new section on the Test Data
    screen, no code change needed. Sorted for a stable, predictable tab/section order."""
    if not CLIPS_DIR.is_dir():
        return ()
    return tuple(sorted(p.name for p in CLIPS_DIR.iterdir() if p.is_dir() and any(p.glob("*.wav"))))


CATEGORIES = _discover_categories()


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
    y, quality = _decode_audio(path.read_bytes(), filename)

    settings = load_settings()
    # A missing/misconfigured Claude provider is no longer fatal (two-line fusion architecture,
    # see reasoning/fusion.py) — proceed with provider=None, run_final_analysis's fallback
    # template still produces a verdict from Line 1/2's fusion result alone.
    try:
        provider = get_llm_provider()
    except LLMProviderError as e:
        logger.warning("could not get an LLM provider for test-clip analysis, proceeding without it: %s", e)
        provider = None

    call_state = CallState()
    call_state.audio_quality = quality
    duration_seconds = len(y) / SAMPLE_RATE
    start = time.monotonic()

    async def event_stream() -> AsyncGenerator[bytes, None]:
        # Broad except, not just LLMProviderError — see server/upload.py's event_stream for why
        # (an uncaught exception partway through streaming otherwise just drops the connection,
        # which surfaces to the browser as an opaque "network error" instead of a real message).
        try:
            async for event in stream_pipeline_over_audio(y, provider, call_state, settings["hard_triggers"]):
                yield _ndjson_line(event)
        except Exception as e:
            logger.exception("pipeline failed while analyzing test clip %s/%s", category, filename)
            yield _ndjson_line({"type": "error", "message": str(e) or e.__class__.__name__})
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
