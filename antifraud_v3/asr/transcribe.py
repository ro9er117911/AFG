"""Per-chunk ASR. Ported config from antifraud_v2/language_processing.py (same model size/
device/compute_type) but transcribing an in-memory chunk array instead of a saved file, and
called once per completed VAD chunk instead of once per whole call — "transcribe on turn
completion," not literal word-level streaming ASR (see docs/DESIGN.md §5 for why word-level
streaming ASR is out of scope without a paid API).
"""

import numpy as np
from faster_whisper import WhisperModel

_model_cache: WhisperModel | None = None


def load_whisper_model() -> WhisperModel:
    global _model_cache
    if _model_cache is None:
        _model_cache = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)
    return _model_cache


def transcribe_chunk(y: np.ndarray, sr: int, language: str = "zh") -> str:
    model = load_whisper_model()
    # faster-whisper resamples internally if needed, but expects float32 in [-1, 1].
    audio = y.astype(np.float32) if y.dtype != np.float32 else y
    segments, _info = model.transcribe(audio, language=language)
    return "".join(segment.text for segment in segments).strip()
