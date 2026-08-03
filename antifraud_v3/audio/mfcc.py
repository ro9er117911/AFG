import librosa
import numpy as np


def get_mfcc_from_array(
    y: np.ndarray, sr: int, offset: float = 0.0, duration: float = 4.0, framelength: float = 0.05
) -> np.ndarray:
    """MFCC + delta + delta-delta for TIMNet input. Ported from antifraud_v2/audio_processing.py's
    get_mfcc(), operating on an in-memory chunk array instead of re-reading a file per window.

    TIMNet was trained at 22050 Hz — resample if the chunk arrived at a different rate
    (browser audio is typically 48kHz).
    """
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    start_sample = int(offset * sr)
    data = y[start_sample : start_sample + int(duration * sr)]

    target_len = int(sr * duration)
    if len(data) > target_len:
        data = data[:target_len]
    else:
        data = np.hstack([data, np.zeros(target_len - len(data))])

    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize).T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    return np.hstack([mfcc, mfcc_delta, mfcc_acc])
