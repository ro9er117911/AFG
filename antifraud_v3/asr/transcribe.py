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

from ..storage.settings_store import load_settings

logger = logging.getLogger(__name__)

_model_cache: WhisperModel | None = None
_sensevoice_cache = None


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


def load_sensevoice_model():
    """Lazily imports funasr — a new, optional dependency (see requirements.txt) — so
    environments that only ever select "whisper" (the default) never pay its import cost or
    need it installed at all."""
    global _sensevoice_cache
    if _sensevoice_cache is None:
        import torch
        from funasr import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _sensevoice_cache = AutoModel(
            model="iic/SenseVoiceSmall", trust_remote_code=True, device=device, disable_update=True
        )
    return _sensevoice_cache


def _transcribe_whisper(audio: np.ndarray, language: str) -> str:
    model = load_whisper_model()
    segments, _info = model.transcribe(audio, language=language)
    return "".join(segment.text for segment in segments).strip()


def _transcribe_sensevoice(audio: np.ndarray, language: str) -> str:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = load_sensevoice_model()
    result = model.generate(input=audio, cache={}, language=language, use_itn=True, batch_size_s=60)
    return rich_transcription_postprocess(result[0]["text"]).strip()


def transcribe_chunk(y: np.ndarray, sr: int, language: str = "zh") -> str:
    # Both backends expect float32 in [-1, 1] and resample/assume 16kHz internally — this app's
    # pipeline already guarantees 16kHz mono for every chunk reaching here (see audio/vad.py).
    audio = y.astype(np.float32) if y.dtype != np.float32 else y
    backend = load_settings()["asr_backend"]
    if backend == "sensevoice":
        return _transcribe_sensevoice(audio, language)
    return _transcribe_whisper(audio, language)
