import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from faster_whisper import WhisperModel

# 初始化 Whisper 模型
whisper_model = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)

def get_mfcc(filename, sr=22050, duration=4, framelength=0.05):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件: {filename}")
    data, sr = librosa.load(filename, sr=sr)
    time = librosa.get_duration(y=data, sr=sr)
    if time > duration:
        data = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        data = np.hstack([data, np.zeros(padding_len)])
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    return mfcc

def create_audio_waveform(audio_file):
    y, sr = librosa.load(audio_file)
    fig, ax = plt.subplots(figsize=(10, 1))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    return fig

def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > 0]) if np.any(magnitudes > 0) else 0.0
    volume = np.mean(librosa.feature.rms(y=y)[0])
    spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
    
    return {
        "語速": tempo,
        "平均音高": pitch_mean,
        "平均音量": volume,
        "頻譜重心": spectral_centroids,
        "頻譜衰減": spectral_rolloff
    }

def transcribe_audio(audio_path):
    segments, info = whisper_model.transcribe(audio_path, language="zh")
    return " ".join([segment.text for segment in segments])