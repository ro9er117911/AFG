"""WebSocket endpoint: one persistent connection per live call. Accepts binary float32 PCM
frames (16kHz mono) from the browser, chunks them via VAD, and for each completed chunk runs
ASR + acoustic + emotion (no LLM — see pipeline/chunk_worker.py's module docstring) off the
event loop thread, pushing transcript/keyword-alert updates back as JSON in real time. The
LLM reasoning pipeline runs exactly once, after the call ends, and its verdict is pushed as a
single "final_analysis" message — this is the part Streamlit's rerun model can't do natively,
see docs/DESIGN.md §5.
"""

import asyncio
import contextlib
import dataclasses
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..asr.transcribe import transcribe_chunk
from ..audio.vad import VADChunker
from ..llm import LLMProviderError, get_llm_provider
from ..pipeline.call_state import Alert, CallState
from ..pipeline.chunk_worker import process_chunk_signals, run_final_analysis
from ..storage import history
from ..storage.settings_store import load_settings

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16000
INTERIM_TRANSCRIPT_INTERVAL_S = 1.8
# Below this, the in-progress buffer is almost certainly just the start of a word/breath —
# transcribing it would mostly flicker garbage rather than show useful partial text.
INTERIM_MIN_SAMPLES = int(0.6 * SAMPLE_RATE)


async def _send_interim(ws: WebSocket, text: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "interim_transcript", "text": text}, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass  # same send-after-close race as _handle_chunk — the main loop's own error
        # handling deals with the connection; this loop just quietly skips the update.


async def _interim_transcript_loop(ws: WebSocket, chunker: VADChunker) -> None:
    """Runs alongside the main receive loop for the whole call: every ~1.8s, if there's
    speech in progress that hasn't hit a silence gap yet, re-transcribes it (ASR only) and
    pushes it as a live preview line, so the user sees text appear while they're still talking
    instead of waiting for the full turn's chunk_update. See frontend/app.js's handling of
    "interim_transcript" for how this gets replaced once the utterance actually ends.
    """
    last_sent_text: str | None = None
    while True:
        await asyncio.sleep(INTERIM_TRANSCRIPT_INTERVAL_S)
        try:
            audio = chunker.peek_in_progress()
            if audio is None or len(audio) < INTERIM_MIN_SAMPLES:
                if last_sent_text:
                    last_sent_text = None
                    await _send_interim(ws, "")
                continue
            # Re-transcribing a growing buffer every tick is real CPU work (whisper 'base' on
            # the full in-progress utterance so far, not just the newest slice) — deliberately
            # not overlapped: if a previous transcription is still running when the next tick
            # fires, asyncio.to_thread here just hasn't returned yet, so this loop naturally
            # waits for it rather than piling up concurrent whisper calls competing for CPU.
            text = await asyncio.to_thread(transcribe_chunk, audio, SAMPLE_RATE)
            if text and text != last_sent_text:
                last_sent_text = text
                await _send_interim(ws, text)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An interim preview failing should never take down the call — the real per-chunk
            # pipeline (_handle_chunk) is what actually matters and has its own error handling.
            logger.exception("interim transcript tick failed; will retry next tick")


def _alert_to_dict(alert: Alert) -> dict:
    d = dataclasses.asdict(alert)
    d["type"] = "alert"
    return d


async def _chunk_worker(
    ws: WebSocket,
    queue: "asyncio.Queue[np.ndarray | None]",
    call_state: CallState,
    alerts: list[dict],
) -> None:
    """Consumes completed utterance chunks one at a time, in arrival order, off a queue —
    deliberately a separate task from the main receive loop.

    Confirmed by real testing: awaiting _handle_chunk() directly inside the main `while True:
    message = await websocket.receive()` loop means the server stops reading new audio for as
    long as one chunk's ASR + acoustic + emotion take — incoming audio just piles up unread in
    the ASGI receive queue for that whole time. That starves VADChunker.push_audio() of fresh
    frames, which in turn starves peek_in_progress() (the interim-transcript preview) and
    delays VAD ever detecting the *next* utterance's boundaries. Decoupling "receive audio and
    feed the VAD" from "run one chunk's pipeline" onto separate tasks — connected by this
    queue — keeps audio ingestion live the whole call. A `None` item is the drain sentinel (see
    call_socket's finally).
    """
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            try:
                await _handle_chunk(ws, item, call_state, alerts)
            except (WebSocketDisconnect, RuntimeError):
                # Confirmed by real testing: the client can be gone (tab closed, network
                # dropped, or the clean-stop race in frontend/app.js's stopListening()) by the
                # time this chunk's ws.send_text() runs — raised as WebSocketDisconnect in some
                # cases, a plain RuntimeError ("... after sending 'websocket.close'") in others,
                # depending on exactly when the ASGI server noticed. Both harmless: the chunk's
                # transcript was already recorded into call_state before the send was
                # attempted, so it's still captured in history even though no one was
                # listening live for it.
                logger.debug("client disconnected while processing a queued chunk")
            except Exception:
                logger.exception("failed to process a queued chunk")
        finally:
            queue.task_done()


async def _handle_chunk(
    ws: WebSocket, y: np.ndarray, call_state: CallState, alerts: list[dict]
) -> None:
    # process_chunk_signals does blocking I/O (whisper, parselmouth, TIMNet) — keep it off the
    # event loop. No LLM call here — see pipeline/chunk_worker.py's module docstring.
    result = await asyncio.to_thread(process_chunk_signals, y, SAMPLE_RATE, call_state)
    if result is None:
        return  # VAD false-positive on non-speech noise — nothing to report

    await ws.send_text(
        json.dumps(
            {
                "type": "chunk_update",
                "timestamp": call_state.transcript[-1].timestamp,
                "transcript_text": result.transcript_text,
                "acoustic": result.acoustic,
                "emotion": result.emotion,
                "baseline": result.baseline,
            },
            ensure_ascii=False,
        )
    )
    if result.alert is not None:
        alerts.append(dataclasses.asdict(result.alert))
        await ws.send_text(json.dumps(_alert_to_dict(result.alert), ensure_ascii=False))


async def _run_and_send_final_analysis(
    ws: WebSocket, call_state: CallState, hard_triggers: list[str], alerts: list[dict]
) -> None:
    """Called once, after the call ends — runs the discriminate/reflect/synthesize reasoning
    pipeline over the whole accumulated call and pushes the verdict as one "final_analysis"
    message. See pipeline/chunk_worker.py's run_final_analysis.
    """
    try:
        provider = get_llm_provider()
    except LLMProviderError as e:
        logger.warning("could not get an LLM provider for final analysis: %s", e)
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
        return

    try:
        outcome = await asyncio.to_thread(run_final_analysis, provider, call_state, hard_triggers)
    except LLMProviderError as e:
        # Confirmed by real browser testing: a missing/invalid ANTHROPIC_API_KEY (or, now, a
        # missing `claude` CLI / expired subscription session) otherwise kills the whole
        # WebSocket connection with an unhandled exception, and the client never learns why.
        logger.warning("final analysis failed: %s", e)
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
        return

    if outcome is None:
        return  # nothing was ever transcribed — a silent/empty call, nothing to analyze

    result, alert, evidence = outcome
    await ws.send_text(
        json.dumps(
            {
                "type": "final_analysis",
                "risk_level": result.risk_level,
                "chunk_risk_score": result.chunk_risk_score,
                "justification": result.justification,
                "fraud_type": result.fraud_type.model_dump(),
                "audio_quality": call_state.audio_quality,  # always None for live mic calls
                "evidence": {
                    "transcript_segment": evidence.transcript_segment,
                    "acoustic_summary": evidence.acoustic_summary,
                    "emotion_summary": evidence.emotion_summary,
                },
            },
            ensure_ascii=False,
        )
    )
    if alert is not None:
        alerts.append(dataclasses.asdict(alert))
        await ws.send_text(json.dumps(_alert_to_dict(alert), ensure_ascii=False))


@router.websocket("/ws/call")
async def call_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    settings = load_settings()
    call_state = CallState()
    chunker = VADChunker(sample_rate=SAMPLE_RATE)
    call_id = history.create_call()
    ended_reason = "stopped"
    alerts: list[dict] = []
    interim_task = asyncio.create_task(_interim_transcript_loop(websocket, chunker))
    chunk_queue: "asyncio.Queue[np.ndarray | None]" = asyncio.Queue()
    worker_task = asyncio.create_task(_chunk_worker(websocket, chunk_queue, call_state, alerts))

    try:
        while True:
            message = await websocket.receive()
            # Confirmed by real testing (closing the browser tab): the raw receive() ASGI
            # message dict for a disconnect is {"type": "websocket.disconnect", ...} — it does
            # NOT raise WebSocketDisconnect the way the receive_bytes()/receive_text() helpers
            # do. Calling receive() again after this arrives is a Starlette RuntimeError, so
            # this has to be checked explicitly rather than relying on the except clause below.
            if message["type"] == "websocket.disconnect":
                ended_reason = "disconnected"
                break
            if message.get("bytes") is not None:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                # Just hand completed chunks to the worker and immediately loop back to
                # receive() — see _chunk_worker's docstring for why this must NOT await chunk
                # processing here directly.
                for chunk in chunker.push_audio(audio):
                    chunk_queue.put_nowait(chunk)
            elif message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "ack_alert":
                    call_state.acknowledge_alert()
                elif control.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        ended_reason = "disconnected"
    finally:
        interim_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await interim_task

        final_chunk = chunker.flush()
        if final_chunk is not None and len(final_chunk) > 0:
            chunk_queue.put_nowait(final_chunk)
        chunk_queue.put_nowait(None)  # drain sentinel — worker returns once it reaches this
        await worker_task

        try:
            await _run_and_send_final_analysis(websocket, call_state, settings["hard_triggers"], alerts)
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("client disconnected before the final analysis result could be sent")

        history.finish_call(call_id, call_state, ended_reason=ended_reason, alerts=alerts)

        # Confirmed by real testing: Starlette does NOT send a WebSocket close frame just
        # because this handler coroutine returns — if we don't call close() ourselves, uvicorn
        # drops the raw connection instead of performing a proper closing handshake, which the
        # browser reports as an abnormal disconnect (close code 1006, event.wasClean=false —
        # frontend/app.js's ws.onclose then shows "與伺服器的連線已中斷" even though the call
        # actually finished successfully and every result was already sent). Only attempted if
        # the client hasn't already disconnected first.
        if ended_reason != "disconnected":
            with contextlib.suppress(RuntimeError):
                await websocket.close()
