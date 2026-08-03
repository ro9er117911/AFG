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
from ..llm import get_llm_provider
from ..pipeline.call_state import Alert, CallState
from ..pipeline.chunk_worker import process_chunk

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16000


def _alert_to_dict(alert: Alert) -> dict:
    d = dataclasses.asdict(alert)
    d["type"] = "alert"
    return d


async def _handle_chunk(ws: WebSocket, y: np.ndarray, call_state: CallState) -> None:
    provider = get_llm_provider()
    # process_chunk does blocking I/O (whisper, LLM calls) — keep it off the event loop.
    result = await asyncio.to_thread(process_chunk, y, SAMPLE_RATE, provider, call_state)
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
    call_state = CallState()
    chunker = VADChunker(sample_rate=SAMPLE_RATE)

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                for chunk in chunker.push_audio(audio):
                    await _handle_chunk(websocket, chunk, call_state)
            elif message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "ack_alert":
                    call_state.acknowledge_alert()
                elif control.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        final_chunk = chunker.flush()
        if final_chunk is not None and len(final_chunk) > 0:
            try:
                await _handle_chunk(websocket, final_chunk, call_state)
            except Exception:
                logger.exception("failed to process final chunk on disconnect")
