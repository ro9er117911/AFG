"""VAD-triggered utterance chunking. See docs/DESIGN.md §2.1 for why chunks are
utterance-sized (VAD-detected speech/silence transitions) rather than fixed time windows.

Uses silero-vad's streaming VADIterator. NOTE: this is a first-pass implementation against
the silero-vad package API as documented/recalled — verify frame-size and constructor
argument names against your installed `silero-vad` version's actual signature (`pip show
silero-vad`, or read the package source) if this raises on first run; the streaming shape
(feed fixed-size frames, get start/end events back) is stable across recent versions even if
exact argument names drift.
"""

import numpy as np
from silero_vad import VADIterator, load_silero_vad

# silero-vad's JIT model requires exactly this frame size for 16kHz input.
FRAME_SAMPLES_16K = 512


class VADChunker:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_silence_ms: int = 600,
        max_chunk_s: float = 15.0,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = FRAME_SAMPLES_16K if sample_rate == 16000 else FRAME_SAMPLES_16K // 2
        self.max_chunk_samples = int(max_chunk_s * sample_rate)

        model = load_silero_vad()
        self.vad_iterator = VADIterator(
            model, sampling_rate=sample_rate, min_silence_duration_ms=min_silence_ms
        )

        self._frame_buf = np.array([], dtype=np.float32)
        self._speech_buf: list[np.ndarray] = []
        self._in_speech = False

    def push_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Feed newly-arrived float32 mono audio at self.sample_rate. Returns zero or more
        completed utterance chunks (silence-gap end, or forced split at max_chunk_s)."""
        self._frame_buf = np.concatenate([self._frame_buf, audio.astype(np.float32)])
        completed: list[np.ndarray] = []

        while len(self._frame_buf) >= self.frame_samples:
            frame = self._frame_buf[: self.frame_samples]
            self._frame_buf = self._frame_buf[self.frame_samples :]

            if self._in_speech:
                self._speech_buf.append(frame)

            event = self.vad_iterator(frame, return_seconds=False)
            if event is not None:
                if "start" in event and not self._in_speech:
                    self._in_speech = True
                    self._speech_buf = [frame]
                elif "end" in event and self._in_speech:
                    completed.append(np.concatenate(self._speech_buf))
                    self._speech_buf = []
                    self._in_speech = False

            if self._in_speech and sum(len(b) for b in self._speech_buf) >= self.max_chunk_samples:
                # Hard cap hit mid-utterance (e.g. a scripted monologue) — force-close this
                # chunk but stay "in speech" so the next frame starts a continuation chunk
                # rather than waiting for real silence. See docs/DESIGN.md §2.1.
                completed.append(np.concatenate(self._speech_buf))
                self._speech_buf = []

        return completed

    def peek_in_progress(self) -> np.ndarray | None:
        """Non-destructive snapshot of the current in-progress utterance (speech that has
        started but not yet hit a silence gap / the max_chunk_s cap) — used to show a live
        "still speaking" transcript preview without disturbing push_audio()'s own accumulation
        state. Safe to call concurrently with push_audio() from another asyncio task on the
        same event loop: push_audio() never awaits mid-mutation, so there's no point where a
        concurrent caller could observe a torn/partial _speech_buf."""
        if not self._speech_buf:
            return None
        return np.concatenate(self._speech_buf)

    def flush(self) -> np.ndarray | None:
        """Call when the call ends to emit any still-in-progress speech as a final chunk."""
        if not self._speech_buf:
            return None
        chunk = np.concatenate(self._speech_buf)
        self._speech_buf = []
        self._in_speech = False
        self.vad_iterator.reset_states()
        return chunk
