"""WebSocket endpoint: one persistent connection per live call. Accepts binary float32 PCM
frames (16kHz mono) from the browser, chunks them via VAD, runs the per-chunk pipeline off
the event loop thread, and pushes chunk/alert updates back as JSON — this is the part
Streamlit's rerun model can't do natively, see REWRITE_PLAN.md §5.
"""

import asyncio
import dataclasses
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..audio.vad import VADChunker
from ..llm import LLMProviderError, get_llm_provider
from ..pipeline.call_state import Alert, CallState
from ..pipeline.chunk_worker import process_chunk
from ..storage import history
from ..storage.settings_store import load_settings

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16000


def _alert_to_dict(alert: Alert) -> dict:
    d = dataclasses.asdict(alert)
    d["type"] = "alert"
    return d


async def _handle_chunk(
    ws: WebSocket, y: np.ndarray, call_state: CallState, hard_triggers: list[str]
) -> None:
    provider = get_llm_provider()
    # process_chunk does blocking I/O (whisper, LLM calls) — keep it off the event loop.
    try:
        result = await asyncio.to_thread(
            process_chunk, y, SAMPLE_RATE, provider, call_state, None, hard_triggers
        )
    except LLMProviderError as e:
        # Confirmed by real browser testing: a missing/invalid ANTHROPIC_API_KEY otherwise
        # kills the whole WebSocket connection on the very first chunk with an unhandled
        # exception, and the client never learns why. Surface it as a message instead of
        # dying — this is the single most likely first-run failure mode (see .env.example).
        logger.warning("reasoning pipeline failed for this chunk: %s", e)
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
        return
    if result is None:
        return  # VAD false-positive on non-speech noise — nothing to report

    await ws.send_text(
        json.dumps(
            {
                "type": "chunk_update",
                "timestamp": call_state.risk_trajectory[-1].timestamp,
                "transcript_text": result.transcript_text,
                "risk_level": result.synthesize_result.risk_level,
                "chunk_risk_score": result.synthesize_result.chunk_risk_score,
                "justification": result.synthesize_result.justification,
            },
            ensure_ascii=False,
        )
    )
    if result.alert is not None:
        await ws.send_text(json.dumps(_alert_to_dict(result.alert), ensure_ascii=False))


@router.websocket("/ws/call")
async def call_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    settings = load_settings()
    call_state = CallState(debounce_chunks=settings["debounce_chunks"])
    chunker = VADChunker(sample_rate=SAMPLE_RATE)
    call_id = history.create_call()
    client_connected = True  # false only on an actual disconnect, not an explicit "stop"
    ended_reason = "stopped"

    try:
        while True:
            message = await websocket.receive()
            # Confirmed by real testing (closing the browser tab): the raw receive() ASGI
            # message dict for a disconnect is {"type": "websocket.disconnect", ...} — it does
            # NOT raise WebSocketDisconnect the way the receive_bytes()/receive_text() helpers
            # do. Calling receive() again after this arrives is a Starlette RuntimeError, so
            # this has to be checked explicitly rather than relying on the except clause below.
            if message["type"] == "websocket.disconnect":
                client_connected = False
                ended_reason = "disconnected"
                break
            if message.get("bytes") is not None:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                for chunk in chunker.push_audio(audio):
                    await _handle_chunk(websocket, chunk, call_state, settings["hard_triggers"])
            elif message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "ack_alert":
                    call_state.acknowledge_alert()
                elif control.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        client_connected = False
        ended_reason = "disconnected"
    finally:
        final_chunk = chunker.flush()
        # Also confirmed by real testing: attempting ws.send_text() after the client is gone
        # raises WebSocketDisconnect/ClientDisconnected from inside send(). The
        # client_connected check above rules out the common case (tab closed, no "stop" sent
        # at all) but not a second race also confirmed by testing: the browser's stop button
        # sends {"type":"stop"} and calls ws.close() back-to-back without waiting for a
        # response (see frontend/app.js stopListening()), so the client can genuinely be gone
        # by the time this finally-block's own send finishes processing the final chunk (ASR +
        # acoustic + emotion + LLM call all take real time). Both are harmless/expected — no
        # one's listening for the result either way — so they're logged at debug level, not as
        # a scary unhandled-looking traceback; only a genuinely unexpected error gets the full
        # logger.exception() treatment.
        if client_connected and final_chunk is not None and len(final_chunk) > 0:
            try:
                await _handle_chunk(websocket, final_chunk, call_state, settings["hard_triggers"])
            except WebSocketDisconnect:
                logger.debug("client disconnected before the final chunk's result could be sent")
            except Exception:
                logger.exception("failed to process final chunk before clean stop")

        history.finish_call(call_id, call_state, ended_reason=ended_reason)
