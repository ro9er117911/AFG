"""Per-chunk ASR. Ported config from antifraud_v2/language_processing.py (same model size/
device/compute_type) but transcribing an in-memory chunk array instead of a saved file, and
called once per completed VAD chunk instead of once per whole call — "transcribe on turn
completion," not literal word-level streaming ASR (see docs/DESIGN.md §5 for why word-level
streaming ASR is out of scope without a paid API).
"""

import logging

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_model_cache: WhisperModel | None = None


def load_whisper_model() -> WhisperModel:
    global _model_cache
    if _model_cache is None:
        # faster-whisper runs on ctranslate2, which has its own CUDA runtime detection
        # independent of torch's — checked directly here rather than importing torch just for
        # torch.cuda.is_available(). float16 needs a real GPU; int8 is the CPU-appropriate
        # quantization (matches the original antifraud_v2 CPU-only config for the fallback
        # case, kept for machines/environments without a usable CUDA GPU).
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        device, compute_type = ("cuda", "float16") if has_cuda else ("cpu", "int8")
        try:
            _model_cache = WhisperModel(
                "base", device=device, compute_type=compute_type, local_files_only=False
            )
        except Exception:
            if device == "cpu":
                raise
            logger.exception("failed to load faster-whisper on cuda, falling back to cpu")
            _model_cache = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)
    return _model_cache


def transcribe_chunk(y: np.ndarray, sr: int, language: str = "zh") -> str:
    model = load_whisper_model()
    # faster-whisper resamples internally if needed, but expects float32 in [-1, 1].
    audio = y.astype(np.float32) if y.dtype != np.float32 else y
    segments, _info = model.transcribe(audio, language=language)
    return "".join(segment.text for segment in segments).strip()
