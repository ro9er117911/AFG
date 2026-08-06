"""Optional WavLM-based emotion backend — an alternative to audio/emotion.py's TIMNet, selected
via storage/settings_store.py's "emotion_backend" setting ("wavlm" default, "timnet" fallback).
Follows detectors/deepfake_voice.py's module-global lazy-cache convention (this file is to
models/wavlm_emotion.py what detectors/deepfake_voice.py is to models/xlsr_deepfake.py).

Kept in fp32 (not the fp16-on-load convention detectors/deepfake_voice.py uses for XLS-R):
this model's forward() round-trips the input through Wav2Vec2FeatureExtractor as plain numpy
internally, which always yields float32 tensors regardless of input dtype — casting the model
itself to fp16 would just produce a dtype mismatch against that float32 signal on the very
first conv layer. At ~317M params this is already far lighter than Line 1's 2B-param XLS-R, so
the fp32 VRAM cost (~1.2GB) isn't the same pressure point fp16-halving solved there.
"""

import numpy as np
import torch

from .. import config
from ..models.wavlm_emotion import WavLMWrapper

_model_cache: WavLMWrapper | None = None

# Exact order this checkpoint's emotion_layer was trained to output — confirmed against the
# upstream repo's own src/example/categorized_emotion_wavlm.py (not alphabetical, not guessed).
_RAW_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
# Normalized to this app's existing lowercase label style (audio/emotion.py's TIMNet emits
# anger/boredom/disgust/fear/happy/neutral/sad) so the frontend's EMOTION_LABELS_ZH lookup
# doesn't need a second, differently-cased vocabulary for the overlapping concepts — "Happiness"
# -> "happy" and "Sadness" -> "sad" match TIMNet's existing short forms. WavLM has no "boredom"
# equivalent; TIMNet has no contempt/surprise/other equivalent — both sets stay in
# EMOTION_LABELS_ZH so the UI renders correctly regardless of which backend is active.
_LABEL_MAP = {
    "Anger": "anger", "Contempt": "contempt", "Disgust": "disgust", "Fear": "fear",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise", "Other": "other",
}
EMOTION_LABELS = [_LABEL_MAP[label] for label in _RAW_LABELS]

# The upstream training set excluded clips shorter than 3s / longer than 15s as "unreliable" /
# too costly respectively (see their own example_emotion.py comment) — informational only, not
# enforced here: this app's VAD chunks are already capped at 15s (audio/vad.py), and a sub-3s
# chunk still produces *a* prediction, just a documented-less-reliable one, same "soft evidence,
# not a hard gate" spirit as TIMNet's own docstring in audio/emotion.py.
MIN_RELIABLE_SECONDS = 3.0


def load_model(model_id: str = config.EMOTION_WAVLM_MODEL_ID) -> WavLMWrapper:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = WavLMWrapper.from_pretrained(model_id)
    model.to(config.resolve_device())
    model.eval()
    _model_cache = model
    return model


def predict_emotion_wavlm(y: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
    """Returns (predicted_label, {label: probability}) — same shape as audio/emotion.py's
    TIMNet predict_emotion(), using this module's lowercase label vocabulary (see
    _LABEL_MAP), so chunk_worker.py's caller doesn't need to know which backend ran.
    y must already be 16kHz mono float32 (this app's pipeline guarantees that for every chunk
    reaching here — see audio/vad.py's SAMPLE_RATE constant); sr is accepted for signature
    symmetry with the other per-chunk predictors, same convention as
    detectors/deepfake_voice.py's predict_deepfake().
    """
    model = load_model()
    device = config.resolve_device()
    x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _features, _detailed, _arousal, _valence, _dominance = model(x, return_feature=True)
        probs = torch.softmax(logits, dim=1)[0]

    probabilities = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
    predicted = EMOTION_LABELS[int(torch.argmax(probs).item())]
    return predicted, probabilities


def unload_model() -> None:
    """Mirrors detectors/deepfake_voice.py's unload_model() — not currently called anywhere
    (this backend is meant to be a TIMNet-equivalent "load once, cache forever" resident model,
    not per-call loaded/unloaded like Line 1/Line 2), kept for symmetry and for a future caller
    if VRAM pressure ever requires it."""
    global _model_cache
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
