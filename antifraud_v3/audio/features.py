"""Per-chunk acoustic feature extraction.

Ported from antifraud_v2/audio_processing.py (git commit 0c66823, pre-deletion), operating on
an in-memory (y, sr) array instead of a file path, and adapted per docs/DESIGN.md §1/§2.2:

- Pitch tracking consolidated onto parselmouth's `to_pitch()` — CREPE (`model_capacity="full"`,
  Viterbi decoding) is dropped entirely. It was a second, heavier pitch tracker running
  alongside the one already needed for tremor analysis; not viable per-chunk in a real-time
  budget, and redundant even in the old whole-file design.
- The jitter/shimmer/HNR tremor analysis keeps the already-fixed `parselmouth.praat.call()`
  interface (the old code's `pitch.to_point_process()` call didn't exist on the installed
  parselmouth version and always raised, silently falling back to frozen constants — see
  antifraud_v2/FALSE_POSITIVE_REVIEW.md finding #1, fixed there, ported as-fixed here).
- Speech-rate/pause analysis is otherwise unchanged logic, just running on a chunk-sized
  array instead of a whole call.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy import stats


def analyze_speech_rate_and_pauses(y: np.ndarray, sr: int) -> dict:
    y_normalized = librosa.util.normalize(y)
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)

    energy = librosa.feature.rms(y=y_normalized, frame_length=frame_length, hop_length=hop_length)[0]

    # Dynamic per-window silence threshold rather than one global cutoff.
    window_size = 200
    dynamic_threshold = np.zeros_like(energy)
    for i in range(len(energy)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(energy), i + window_size // 2)
        window_energy = energy[start_idx:end_idx]
        if len(window_energy) > 0:
            local_threshold = np.percentile(window_energy, 20)
            min_threshold = np.percentile(energy, 10)
            dynamic_threshold[i] = max(local_threshold, min_threshold)

    is_silence = energy < dynamic_threshold
    min_silence_frames = int(0.15 / (hop_length / sr))

    silence_regions = []
    in_silence = False
    silence_start = 0
    for i in range(len(is_silence)):
        if is_silence[i] and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silence[i] and in_silence:
            in_silence = False
            if i - silence_start >= min_silence_frames:
                silence_regions.append((silence_start, i))
    if in_silence and len(is_silence) - silence_start >= min_silence_frames:
        silence_regions.append((silence_start, len(is_silence)))

    clean_is_silence = np.zeros_like(is_silence, dtype=bool)
    for start, end in silence_regions:
        clean_is_silence[start:end] = True

    silence_starts = np.where(np.diff(np.concatenate([[0], clean_is_silence.astype(int)])) == 1)[0]
    silence_ends = np.where(np.diff(np.concatenate([clean_is_silence.astype(int), [0]])) == -1)[0]

    pause_durations = []
    for i in range(min(len(silence_starts), len(silence_ends))):
        if silence_ends[i] > silence_starts[i]:
            duration = (silence_ends[i] - silence_starts[i]) * hop_length / sr
            if duration >= 0.15:
                pause_durations.append(duration)

    # Local speech-rate estimate via onset density + zero-crossing/energy voiced-ratio,
    # in sliding 2s windows (falls back to empty for chunks shorter than the window).
    window_size_samples = int(sr * 2)
    step_size = int(sr * 0.5)
    local_speech_rates = []
    for i in range(0, len(y_normalized) - window_size_samples, step_size):
        window = y_normalized[i : i + window_size_samples]
        zcr_window = librosa.feature.zero_crossing_rate(window, frame_length=frame_length, hop_length=hop_length)[0]
        energy_window = librosa.feature.rms(y=window, frame_length=frame_length, hop_length=hop_length)[0]
        onset_env = librosa.onset.onset_strength(y=window, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onsets) > 0:
            onset_rate = len(onsets) / (len(window) / sr)
            voiced_frames = np.sum((zcr_window > np.mean(zcr_window)) & (energy_window > np.mean(energy_window)))
            speech_ratio = voiced_frames / len(zcr_window) if len(zcr_window) > 0 else 0
            local_speech_rates.append(onset_rate * 0.7 + speech_ratio * 5.0 * 0.3)
        else:
            local_speech_rates.append(0)

    if len(local_speech_rates) > 1:
        abs_changes = np.abs(np.diff(local_speech_rates))
        speech_rate_std = np.std(local_speech_rates)
        speech_rate_range = np.max(local_speech_rates) - np.min(local_speech_rates)
        change_threshold = np.mean(abs_changes) + 2.0 * np.std(abs_changes) if len(abs_changes) > 0 else 0
        sudden_changes = int(np.sum(abs_changes > change_threshold))
    else:
        speech_rate_std = speech_rate_range = 0
        sudden_changes = 0

    total_duration = len(y_normalized) / sr
    if pause_durations:
        pause_mean = float(np.mean(pause_durations))
        pause_std = float(np.std(pause_durations)) if len(pause_durations) > 1 else 0.0
        pause_count = len(pause_durations)
        pause_rate = pause_count / total_duration if total_duration > 0 else 0
        pause_ratio = sum(pause_durations) / total_duration * 100 if total_duration > 0 else 0
    else:
        pause_mean = pause_std = pause_rate = pause_ratio = 0.0
        pause_count = 0

    return {
        "speech_duration": total_duration,
        "pause_count": pause_count,
        "pause_rate": min(max(pause_rate, 0), 5),
        "pause_mean_duration": min(max(pause_mean, 0), 5),
        "pause_std_duration": pause_std,
        "pause_ratio": min(max(pause_ratio, 0), 90),
        "speech_rate_variation": min(speech_rate_std, 10),
        "speech_rate_range": min(speech_rate_range, 20),
        "sudden_speed_changes": sudden_changes,
        "local_speech_rates": local_speech_rates,
    }


def analyze_pitch(sound: parselmouth.Sound) -> dict:
    """Parselmouth-based pitch features — replaces the old CREPE pass (see module docstring)."""
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies != 0]

    if len(voiced) < 5:
        return {
            "mean_pitch": 0.0, "std_pitch": 0.0, "pitch_range": 0.0,
            "pitch_change_rate": 0.0, "pitch_instability": 0.0, "pitch_trend": 0.0,
            "voiced_ratio": 0.0, "valid_samples": 0,
        }

    pitch_changes = np.sum(np.abs(np.diff(voiced)) > 20)
    pitch_instability = float(np.mean(np.abs(np.diff(np.diff(voiced))))) if len(voiced) > 2 else 0.0
    slope, *_ = stats.linregress(np.arange(len(voiced)), voiced)

    return {
        "mean_pitch": float(np.mean(voiced)),
        "std_pitch": float(np.std(voiced)),
        "pitch_range": float(np.ptp(voiced)),
        "pitch_change_rate": float(pitch_changes / len(voiced)),
        "pitch_instability": pitch_instability,
        "pitch_trend": float(slope),
        "voiced_ratio": float(len(voiced) / len(frequencies)),
        "valid_samples": len(voiced),
    }


def analyze_volume(y: np.ndarray) -> dict:
    rms = librosa.feature.rms(y=y)[0]
    return {
        "mean_volume": float(np.mean(rms)),
        "std_volume": float(np.std(rms)),
        "volume_range": float(np.ptp(rms)),
        "volume_changes": int(np.sum(np.abs(np.diff(rms)) > np.mean(rms) * 0.1)),
    }


def analyze_tremor(sound: parselmouth.Sound) -> dict:
    """Jitter/shimmer/HNR via the correct parselmouth.praat.call() interface — see module
    docstring. Unlike the old code, no silent except-and-fake-constants fallback: a genuine
    failure here should surface, not disappear into a frozen 1.00%/3.00%/15.00dB result.
    """
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = sound.to_harmonicity_cc(0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    def _clean(x):
        return 0.0 if (x is None or np.isnan(x)) else float(x)

    return {
        "jitter_local": _clean(jitter_local) * 100,
        "jitter_ppq5": _clean(jitter_ppq5) * 100,
        "shimmer_local": _clean(shimmer_local) * 100,
        "shimmer_apq5": _clean(shimmer_apq5) * 100,
        "hnr": _clean(hnr),
    }


def extract_acoustic_features(y: np.ndarray, sr: int) -> dict:
    """Top-level entry point for chunk_worker.py — one Sound object shared across pitch and
    tremor analysis so parselmouth only parses the chunk once."""
    sound = parselmouth.Sound(librosa.util.normalize(y), sampling_frequency=sr)
    return {
        "speech_rate": analyze_speech_rate_and_pauses(y, sr),
        "pitch": analyze_pitch(sound),
        "volume": analyze_volume(y),
        "tremor": analyze_tremor(sound),
    }


def aggregate_acoustic_features(features_list: list[dict]) -> dict:
    """Combine one call's worth of per-chunk extract_acoustic_features() dicts into a single
    call-level summary, in the same nested shape summarize_acoustic_features() expects — used
    once at call end (pipeline/chunk_worker.py's run_final_analysis) now that the reasoning
    engine runs once per call instead of once per chunk (see docs/DESIGN.md §8's original
    per-chunk-LLM-cost concern, and pipeline/call_state.py's module docstring).

    Counts (pauses, sudden changes) are summed across chunks — they're meaningful as call
    totals. Continuous descriptive stats (pitch, jitter/shimmer/HNR, pause ratio, speech-rate
    variation) are averaged. Chunks with no voiced pitch (valid_samples == 0, e.g. a very
    short/quiet utterance) are excluded from the pitch average rather than dragging it toward
    zero, mirroring summarize_acoustic_features()'s own "no voiced segment" special-case.
    """
    if not features_list:
        return {
            "speech_rate": {
                "speech_duration": 0.0, "pause_count": 0, "pause_rate": 0.0, "pause_mean_duration": 0.0,
                "pause_std_duration": 0.0, "pause_ratio": 0.0, "speech_rate_variation": 0.0,
                "speech_rate_range": 0.0, "sudden_speed_changes": 0,
            },
            "pitch": {
                "mean_pitch": 0.0, "std_pitch": 0.0, "pitch_range": 0.0, "pitch_change_rate": 0.0,
                "pitch_instability": 0.0, "pitch_trend": 0.0, "voiced_ratio": 0.0, "valid_samples": 0,
            },
            "volume": {"mean_volume": 0.0, "std_volume": 0.0, "volume_range": 0.0, "volume_changes": 0},
            "tremor": {"jitter_local": 0.0, "jitter_ppq5": 0.0, "shimmer_local": 0.0, "shimmer_apq5": 0.0, "hnr": 0.0},
        }

    def avg(dicts: list[dict], key: str) -> float:
        vals = [d[key] for d in dicts]
        return float(np.mean(vals)) if vals else 0.0

    def total(dicts: list[dict], key: str):
        return sum(d[key] for d in dicts)

    sr_list = [f["speech_rate"] for f in features_list]
    pitch_list = [f["pitch"] for f in features_list]
    voiced_pitch_list = [p for p in pitch_list if p["valid_samples"] > 0]
    volume_list = [f["volume"] for f in features_list]
    tremor_list = [f["tremor"] for f in features_list]

    return {
        "speech_rate": {
            "speech_duration": total(sr_list, "speech_duration"),
            "pause_count": total(sr_list, "pause_count"),
            "pause_rate": avg(sr_list, "pause_rate"),
            "pause_mean_duration": avg(sr_list, "pause_mean_duration"),
            "pause_std_duration": avg(sr_list, "pause_std_duration"),
            "pause_ratio": avg(sr_list, "pause_ratio"),
            "speech_rate_variation": avg(sr_list, "speech_rate_variation"),
            "speech_rate_range": avg(sr_list, "speech_rate_range"),
            "sudden_speed_changes": total(sr_list, "sudden_speed_changes"),
        },
        "pitch": {
            "mean_pitch": avg(voiced_pitch_list, "mean_pitch"),
            "std_pitch": avg(voiced_pitch_list, "std_pitch"),
            "pitch_range": avg(voiced_pitch_list, "pitch_range"),
            "pitch_change_rate": avg(voiced_pitch_list, "pitch_change_rate"),
            "pitch_instability": avg(voiced_pitch_list, "pitch_instability"),
            "pitch_trend": avg(voiced_pitch_list, "pitch_trend"),
            "voiced_ratio": avg(voiced_pitch_list, "voiced_ratio"),
            "valid_samples": total(pitch_list, "valid_samples"),
        },
        "volume": {
            "mean_volume": avg(volume_list, "mean_volume"),
            "std_volume": avg(volume_list, "std_volume"),
            "volume_range": avg(volume_list, "volume_range"),
            "volume_changes": total(volume_list, "volume_changes"),
        },
        "tremor": {
            "jitter_local": avg(tremor_list, "jitter_local"),
            "jitter_ppq5": avg(tremor_list, "jitter_ppq5"),
            "shimmer_local": avg(tremor_list, "shimmer_local"),
            "shimmer_apq5": avg(tremor_list, "shimmer_apq5"),
            "hnr": avg(tremor_list, "hnr"),
        },
    }


def summarize_acoustic_features(features: dict) -> str:
    """Human-readable summary for ChunkEvidence.acoustic_summary — this is what the LLM
    reasoning engine actually reads, not the raw dict."""
    p, v, t, sr = features["pitch"], features["volume"], features["tremor"], features["speech_rate"]
    parts = []
    if p["valid_samples"] > 0:
        parts.append(
            f"平均音高 {p['mean_pitch']:.0f}Hz（範圍 {p['pitch_range']:.0f}Hz，不穩定度 {p['pitch_instability']:.1f}）"
        )
    else:
        parts.append("音高：無足夠有聲段可供分析")
    parts.append(f"音量標準差 {v['std_volume']:.4f}")
    parts.append(f"jitter {t['jitter_local']:.2f}%、shimmer {t['shimmer_local']:.2f}%、HNR {t['hnr']:.1f}dB")
    parts.append(
        f"語速變化標準差 {sr['speech_rate_variation']:.2f}、停頓佔比 {sr['pause_ratio']:.1f}%、"
        f"停頓次數 {sr['pause_count']}"
    )
    return "；".join(parts)
