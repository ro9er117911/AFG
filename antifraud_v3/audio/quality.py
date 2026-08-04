"""Detects whether decoded audio is telephone-bandwidth (narrowband, ~300-3400Hz) rather than
full mic-quality wideband audio — used by server/upload.py to flag pre-recorded call
recordings (from PSTN/VoIP conferencing bridges) so the reasoning engine can treat their
acoustic details as lower-confidence evidence (see pipeline/chunk_worker.py's
run_final_analysis). Live mic audio (server/ws.py) never needs this — the browser path is
always 16kHz wideband by construction.
"""

import numpy as np

NARROWBAND_NATIVE_SR_HZ = 8500
ROLLOFF_HZ = 3400.0
# Calibrated against real audio, not guessed: genuine wideband speech (even synthetic TTS, which
# skews low-frequency-heavy) still carries ~1-2% of total energy above 3400Hz from
# fricatives/sibilants — confirmed against antifraud_v3/eval/test_clips/ (edge-tts, 16kHz
# wideband), which measured 1.2-2.0%. A real telephone-bandwidth signal, resampled to 16kHz for
# storage, has only numerical-noise-level energy left above the original bandpass cutoff —
# confirmed by downsampling one of those same clips to 8kHz and back up, which measured 0.29%.
# 0.5% sits between the two with real margin on both sides.
HIGH_ENERGY_RATIO_THRESHOLD = 0.005


def detect_bandwidth(
    y: np.ndarray,
    sr: int,
    native_sr: int,
    rolloff_hz: float = ROLLOFF_HZ,
    high_energy_ratio_threshold: float = HIGH_ENERGY_RATIO_THRESHOLD,
) -> dict:
    """Flags narrowband (telephone-quality) audio via two independent checks — either one
    firing is enough to flag `narrowband: True`:

    (a) native_sr itself is already telephone-grade (<= ~8.5kHz, e.g. a raw 8kHz wav).
    (b) spectral energy above `rolloff_hz` is a small fraction of total energy in `y` (sampled
    at `sr`). This is the check that actually matters for real-world bridge/PSTN recordings:
    conferencing services often upsample to 16kHz/44.1kHz before saving the file, so `native_sr`
    alone would miss a file that's still telephone-bandwidth in its actual acoustic content —
    telephone bandpass filtering leaves almost no energy above ~3400Hz regardless of the
    container's nominal sample rate.

    Call with `y`/`sr` as the audio *before* any resampling to the pipeline's working rate, so
    the spectral check sees the real recorded bandwidth rather than a resampler's interpolation.
    """
    if native_sr <= NARROWBAND_NATIVE_SR_HZ:
        return {"narrowband": True, "high_freq_energy_ratio": 0.0, "reason": "native_sample_rate"}

    if len(y) < 2:
        return {"narrowband": False, "high_freq_energy_ratio": 1.0, "reason": "wideband"}

    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    total_energy = float(np.sum(spectrum**2))
    if total_energy <= 0:
        return {"narrowband": False, "high_freq_energy_ratio": 0.0, "reason": "silent"}

    high_energy = float(np.sum(spectrum[freqs > rolloff_hz] ** 2))
    ratio = high_energy / total_energy

    if ratio < high_energy_ratio_threshold:
        return {"narrowband": True, "high_freq_energy_ratio": ratio, "reason": "spectral_rolloff"}
    return {"narrowband": False, "high_freq_energy_ratio": ratio, "reason": "wideband"}
