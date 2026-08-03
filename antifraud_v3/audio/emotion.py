"""Per-chunk TIMNet emotion inference. Ported from antifraud_v2/model_handling.py.

Kept but demoted to soft/low-confidence evidence per docs/DESIGN.md §4 — TIMNet's training
corpora (CASIA/EMO-DB/EMOVO/IEMOCAP/RAVDESS/SAVEE, see ../references/01-timnet.md and
02-cross-lingual-ser.md) are acted, non-telephone, and not Mandarin-conversational, so its
output is fed to the reasoning engine as one signal among several, not a hard gate the way
antifraud_v2/fraud_detection.py used it.

The sliding-window-average logic (window_size/stride) is unchanged from the already-fixed
antifraud_v2 version — for a chunk shorter than window_size (the common case, since chunks
are utterance-sized) this collapses to a single window, identical to pre-fix behavior on
short clips.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from ..models.TIM import TIM_Net, TIMNet, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer
from .mfcc import get_mfcc_from_array

EMOTION_LABELS = ["anger", "boredom", "disgust", "fear", "happy", "neutral", "sad"]
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "TIM-7_False_drop25_mfcc_smoothTrue_epoch500_l2re1_lr005_best.pt"
)

_model_cache: torch.nn.Module | None = None


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> torch.nn.Module:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not model_path.exists():
        raise FileNotFoundError(f"TIMNet weights not found: {model_path}")

    torch.serialization.add_safe_globals(
        [TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer]
    )
    # The checkpoint was pickled with these classes bound to whatever module was __main__
    # at save time (a training script, not model_handling.py) — pickle stores that as
    # "__main__.TIMNet" etc. antifraud_v2's fix (globals().update() inside model_handling.py)
    # only worked if model_handling.py itself happened to BE __main__, which it wasn't in the
    # real app (main.py was). Injecting straight into sys.modules['__main__'] is the actually
    # portable fix, regardless of which module calls load_model().
    main_module = sys.modules["__main__"]
    for cls in (TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer):
        setattr(main_module, cls.__name__, cls)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        new_model = TIMNet(feature_dim=39, drop_rate=0.1, num_class=7, filters=128, dilation=8, kernel_size=2)
        new_model.load_state_dict(model)
        model = new_model
    model.eval()
    _model_cache = model
    return model


def predict_emotion(
    y: np.ndarray, sr: int, window_size: float = 4.0, stride: float = 2.0, max_windows: int = 8
) -> tuple[str, dict[str, float]]:
    """Returns (predicted_label, {label: probability}). max_windows is lower than the old
    whole-call default (30) — chunks are utterance-sized (<=15s per docs/DESIGN.md §2.1),
    not whole calls, so far fewer windows are ever actually needed.
    """
    model = load_model()
    total_duration = len(y) / sr

    if total_duration <= window_size:
        offsets = [0.0]
    else:
        offsets = list(np.arange(0.0, total_duration - window_size + 1e-9, stride))
        if len(offsets) > max_windows:
            offsets = list(np.linspace(0.0, total_duration - window_size, max_windows))

    window_probabilities = []
    for offset in offsets:
        x = get_mfcc_from_array(y, sr, offset=offset, duration=window_size)
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, (0, 2, 1))
        with torch.no_grad():
            predictions = model(torch.tensor(x, dtype=torch.float32))
            probabilities = torch.softmax(predictions, dim=1)
        window_probabilities.append(probabilities.squeeze(0).numpy())

    avg_probabilities = np.mean(window_probabilities, axis=0)
    predicted_emotion = EMOTION_LABELS[int(np.argmax(avg_probabilities))]
    return predicted_emotion, {label: float(p) for label, p in zip(EMOTION_LABELS, avg_probabilities)}


def summarize_emotion(probabilities: dict[str, float]) -> str:
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return "、".join(f"{label} {prob:.2f}" for label, prob in ranked)
