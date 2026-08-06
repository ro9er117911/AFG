"""eGeMAPSv02 (extended Geneva Minimalistic Acoustic Parameter Set) via openSMILE — 88
standardized functionals per chunk, distinct from audio/features.py's hand-rolled pitch/
jitter/shimmer/pause features. Where features.py answers "what does this app's existing 3a/3b/
3c panels need," this answers "what does published paralinguistics research already validate
as informative for voice-quality/emotion/deception cues" — formants, MFCCs, spectral slope/
flux, Hammarberg index, none of which features.py computes.

Loaded once (module-global lazy cache), mirrors detectors/*.py's load-once convention —
opensmile.Smile() parses its config on construction (not free), but it's a signal-processing
pipeline, not a GPU model: extracting all 88 functionals over one VAD-chunk-sized clip measures
~30ms, negligible next to the ASR/acoustic/emotion/deepfake work already happening per chunk.
"""

import numpy as np
import opensmile

_smile_cache: opensmile.Smile | None = None


def _get_smile() -> opensmile.Smile:
    global _smile_cache
    if _smile_cache is None:
        _smile_cache = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile_cache


# Curated subset the frontend calls out by name (labelled tiles, bold rows in the full-88
# detail view) — the rest of the 88 dims are still extracted and shipped every chunk (cheap,
# see module docstring), just not worth a dedicated label each. Picked for (a) not duplicating
# what audio/features.py's existing 3a panel already shows (pitch/jitter/shimmer/HNR/pause/
# speech-rate) and (b) established relevance to vocal-tension/voice-quality cues in
# paralinguistics research: loudness, spectral tilt/flux (vocal effort), formants
# (articulation), MFCC1 (timbre), voicing rhythm.
EGEMAPS_HIGHLIGHT_KEYS = [
    "loudness_sma3_amean",
    "spectralFlux_sma3_amean",
    "alphaRatioV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_amean",
    "F1frequency_sma3nz_amean",
    "F2frequency_sma3nz_amean",
    "mfcc1_sma3_amean",
    "slopeV0-500_sma3nz_amean",
    "VoicedSegmentsPerSec",
    "equivalentSoundLevel_dBp",
]

EGEMAPS_LABELS_ZH = {
    "loudness_sma3_amean": "響度",
    "spectralFlux_sma3_amean": "頻譜變化率",
    "alphaRatioV_sma3nz_amean": "Alpha 比率",
    "hammarbergIndexV_sma3nz_amean": "Hammarberg 指數",
    "F1frequency_sma3nz_amean": "共振峰 F1",
    "F2frequency_sma3nz_amean": "共振峰 F2",
    "mfcc1_sma3_amean": "MFCC1",
    "slopeV0-500_sma3nz_amean": "頻譜斜率（低頻）",
    "VoicedSegmentsPerSec": "有聲段/秒",
    "equivalentSoundLevel_dBp": "等效音量",
}


def extract_egemaps(y: np.ndarray, sr: int) -> dict[str, float]:
    """Returns all 88 eGeMAPSv02 functionals as a flat {name: float} dict, NaN-safe (openSMILE
    can return NaN for a chunk with essentially no voiced content — e.g. a near-silent VAD
    chunk — cleaned to 0.0 so callers never special-case NaN downstream, same convention as
    audio/features.py's analyze_tremor._clean()).
    """
    df = _get_smile().process_signal(y.astype(np.float32), sr)
    row = df.iloc[0].to_dict()
    return {k: (0.0 if (v is None or np.isnan(v)) else float(v)) for k, v in row.items()}


def aggregate_egemaps(features_list: list[dict[str, float]]) -> dict[str, float]:
    """Call-level mean of every chunk's 88-dim vector — used once at call end
    (run_final_analysis) to fold into the LLM evidence text, mirroring
    audio/features.py's aggregate_acoustic_features."""
    if not features_list:
        return {}
    keys = features_list[0].keys()
    return {k: float(np.mean([f[k] for f in features_list])) for k in keys}


def summarize_egemaps_highlight(features: dict[str, float]) -> str:
    """Short, curated-subset-only text for ChunkEvidence.acoustic_summary — the full 88 dims
    are for the UI's detail panel, not worth spending LLM prompt tokens on."""
    parts = [f"{EGEMAPS_LABELS_ZH[k]} {features[k]:.2f}" for k in EGEMAPS_HIGHLIGHT_KEYS if k in features]
    return "、".join(parts)
