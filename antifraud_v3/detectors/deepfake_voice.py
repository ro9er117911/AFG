"""Per-chunk AI/cloned-voice (deepfake speech) detection — Line 1 of the two-line detection
spec: nii-yamagishilab/xls-r-2b-anti-deepfake (models/xlsr_deepfake.py), a Wav2Vec2 XLS-R-2B
frontend + FC classification head, taking raw 16kHz waveform directly (no manual feature
extraction). CC BY-NC-SA 4.0 licensed — non-commercial, fine for this personal tool.

Follows audio/emotion.py's module-global lazy-cache convention exactly (this file is to
models/xlsr_deepfake.py what audio/emotion.py is to models/TIM.py).

Loaded in fp16 — halves resident VRAM from the on-disk fp32 checkpoint's ~8.65GB to ~4.3GB,
confirmed by direct measurement on this project's actual GPU (RTX 3080 Ti, 12GB). Peak VRAM
during inference on a realistic utterance-sized chunk (3-15s, matching VADChunker's actual
output) stays close to that resident figure (~4.3-4.7GB measured directly) — do not extrapolate
from a much longer single input, which spikes activation memory far higher and never happens in
the real per-VAD-chunk pipeline (VAD caps individual chunks at 15s, see audio/vad.py).

IMPORTANT TESTING CAVEAT, confirmed by direct measurement: this project's existing
eval/test_clips/ corpus is entirely edge-tts-synthesized speech (see
feedback_antifraud_testing_standard in project memory) — every clip in it is, quite correctly,
detected as "fake" by this model regardless of whether its *content* is a scam script or benign
conversation, because it genuinely is synthesized audio, not a recording of a real human voice.
That corpus can validate that fake_score correctly flows through the pipeline end-to-end, but it
cannot validate the model's real/fake discrimination quality (which is externally validated by
the model's own published benchmarks, not re-validated here) — that needs real human-voice
recordings this project doesn't yet have, same gap already noted for telephone-quality testing.
"""

import gc
from dataclasses import dataclass

import numpy as np
import torch

from .. import config
from ..models.xlsr_deepfake import DeepfakeDetector

_model_cache: DeepfakeDetector | None = None


@dataclass
class DeepfakeChunkResult:
    fake_score: float  # 0..1, softmax probability of the "fake" class
    label: str  # "fake" | "real"


def load_model(model_id: str = config.DEEPFAKE_MODEL_ID) -> DeepfakeDetector:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = DeepfakeDetector.from_pretrained(model_id)
    model.to(config.resolve_device())
    model.half()
    model.eval()
    _model_cache = model
    return model


def unload_model() -> None:
    """Frees Line 1's ~4.3GB resident VRAM. Deliberately NOT the TIMNet/faster-whisper
    load-once-cache-forever convention — measured directly (2026-08-05, RTX 3080 Ti, 12GB): with
    both Line 1 (XLS-R, resident during live per-chunk processing) and Line 2 (AntiFraud-SFT,
    detectors/scam_semantic.py, loaded once per call at call end) resident simultaneously,
    generation OOMs (~11GB combined weights against 11.66GiB actually available, leaving <1GB
    for generation activations — round 1 fit, round 2's larger context did not).
    pipeline/chunk_worker.py's run_final_analysis calls this right before Line 2 loads, since
    Line 1 is never needed again for a call that's already ending — the cost is the *next*
    call's first chunk paying Line 1's ~15-20s reload-from-disk-cache latency, a deliberate,
    bounded trade-off (this project's own plan flagged exactly this as the fallback if resident-
    simultaneously didn't fit, which it measurably doesn't). gc.collect() before empty_cache()
    is load-bearing, not defensive boilerplate — confirmed directly: without it, ~7GB stayed
    resident after "unloading" both models (a reference-cycle-holds-CUDA-tensors-alive gotcha),
    which then OOM'd the *next* call's faster-whisper ASR before either detector even ran.
    """
    global _model_cache
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict_deepfake(y: np.ndarray, sr: int) -> DeepfakeChunkResult:
    """y must already be 16kHz mono float32 (this app's pipeline guarantees that for every
    chunk reaching here — see audio/vad.py's SAMPLE_RATE constant). sr is accepted for
    signature symmetry with the other per-chunk predictors (extract_acoustic_features,
    predict_emotion) but the model has no resampling path of its own; a mismatched sr would be
    a caller bug, not something to silently correct here.
    """
    model = load_model()
    device = config.resolve_device()

    wav = torch.from_numpy(y).half()
    wav = torch.nn.functional.layer_norm(wav, wav.shape)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(wav)
        probs = torch.nn.functional.softmax(logits, dim=1)

    fake_score = float(probs[0][0].item())
    label = "fake" if fake_score >= 0.5 else "real"
    return DeepfakeChunkResult(fake_score=fake_score, label=label)


def score_chunk(y: np.ndarray, sr: int) -> float:
    return predict_deepfake(y, sr).fake_score


def aggregate_deepfake(scores: list[float]) -> dict:
    """max, not mean: a strong single-chunk hit shouldn't get diluted by surrounding
    normal-sounding audio — mirrors the existing hard-trigger philosophy (one clear strong
    instance is enough), not TIMNet's whole-call-average philosophy (audio/emotion.py's
    aggregate_emotion), since a synthetic-voice detector firing strongly even once is a hard
    technical signal, not soft contextual evidence."""
    if not scores:
        return {"max": 0.0, "mean": 0.0}
    return {"max": float(np.max(scores)), "mean": float(np.mean(scores))}
