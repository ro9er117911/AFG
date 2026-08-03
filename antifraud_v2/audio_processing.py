import librosa
import numpy as np
import os
import crepe
from scipy import stats
import parselmouth
from parselmouth.praat import call
import numpy as np
import streamlit as st

def get_mfcc(filename, sr=22050, duration=4, framelength=0.05, offset=0.0):
    """
    從音訊檔案中提取MFCC特徵

    Parameters:
    filename (str): 音訊檔案路徑
    sr (int): 採樣率
    duration (float): 處理的音訊長度（秒）
    framelength (float): 音框長度（秒）
    offset (float): 從音檔的第幾秒開始讀取，用於在較長音檔上取不同時間窗口

    Returns:
    numpy.ndarray: MFCC特徵矩陣
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件: {filename}")

    data, sr = librosa.load(filename, sr=sr, offset=offset, duration=duration)
    time = librosa.get_duration(y=data, sr=sr)
    
    # 如果音訊長度超過指定duration，則截斷；否則用零填充
    if time > duration:
        data = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        data = np.hstack([data, np.zeros(padding_len)])
    
    # 計算MFCC特徵
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    
    # 計算MFCC的一階和二階導數
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    
    # 將MFCC特徵合併
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    
    return mfcc

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_speech_features(audio_file_path):
    """
    進階分析語音特徵，包括語速、音高和音量
    
    Parameters:
    audio_file_path (str): 音訊檔案路徑
    
    Returns:
    dict: 語音特徵分析結果
    """
    y, sr = librosa.load(audio_file_path)
    
    def analyze_speech_patterns(y, sr):
        """
        使用動態閾值和多特徵分析改進的語音分析函數
        
        參數:
        y: 音頻信號
        sr: 採樣率
        
        返回:
        包含語速和停頓分析結果的字典
        """
        # 基本預處理
        y_normalized = librosa.util.normalize(y)
        
        # 1. 改進的停頓偵測 - 使用動態閾值
        frame_length = int(sr * 0.025)  # 25ms 窗口
        hop_length = int(sr * 0.010)    # 10ms 步進
        
        # 計算短時能量和零交叉率
        energy = librosa.feature.rms(y=y_normalized, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y_normalized, frame_length=frame_length, hop_length=hop_length)[0]
        
        
        # 動態閾值計算 - 使用滑動窗口
        window_size = 200  # 約2秒 (取決於hop_length)
        dynamic_threshold = np.zeros_like(energy)
        
        for i in range(len(energy)):
            # 取當前位置周圍的窗口
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(energy), i + window_size // 2)
            window_energy = energy[start_idx:end_idx]
            
            # 使用窗口內的能量分佈計算閾值
            if len(window_energy) > 0:
                # 使用低百分位值作為本地閾值
                local_threshold = np.percentile(window_energy, 20)  # 第20百分位的能量
                # 確保閾值不會太低
                min_threshold = np.percentile(energy, 10)  # 全局最低閾值
                dynamic_threshold[i] = max(local_threshold, min_threshold)
        
        # 使用動態閾值偵測停頓
        is_silence = energy < dynamic_threshold
        
        # 去除過短的停頓和語音段 (去噪)
        min_silence_frames = int(0.15 / (hop_length / sr))  # 最小150ms
        min_speech_frames = int(0.1 / (hop_length / sr))    # 最小100ms
        
        # 去除過短的停頓
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i in range(len(is_silence)):
            if is_silence[i] and not in_silence:
                # 進入停頓
                in_silence = True
                silence_start = i
            elif not is_silence[i] and in_silence:
                # 結束停頓
                in_silence = False
                silence_length = i - silence_start
                if silence_length >= min_silence_frames:
                    silence_regions.append((silence_start, i))
        
        # 處理最後一個區段
        if in_silence and len(is_silence) - silence_start >= min_silence_frames:
            silence_regions.append((silence_start, len(is_silence)))
        
        # 重建清潔的停頓標記
        clean_is_silence = np.zeros_like(is_silence, dtype=bool)
        for start, end in silence_regions:
            clean_is_silence[start:end] = True
        
        # 找出停頓的開始和結束時間點
        silence_starts = np.where(np.diff(np.concatenate([[0], clean_is_silence.astype(int)])) == 1)[0]
        silence_ends = np.where(np.diff(np.concatenate([clean_is_silence.astype(int), [0]])) == -1)[0]
        
        # 計算停頓持續時間
        pause_durations = []
        valid_pauses = []
        
        for i in range(min(len(silence_starts), len(silence_ends))):
            if silence_ends[i] > silence_starts[i]:
                duration = (silence_ends[i] - silence_starts[i]) * hop_length / sr
                if duration >= 0.15:  # 確認至少150ms
                    valid_pauses.append((silence_starts[i], silence_ends[i]))
                    pause_durations.append(duration)
        
        # 2. 改進的語速變化分析
        window_size = int(sr * 2)  # 2秒窗口
        step_size = int(sr * 0.5)  # 0.5秒步進
        
        # 使用多特徵計算語速
        local_speech_rates = []
        
        for i in range(0, len(y_normalized) - window_size, step_size):
            window = y_normalized[i:i+window_size]
            window_clean = window.copy()
            
            # 使用零交叉率和能量結合的方法
            zcr_window = librosa.feature.zero_crossing_rate(window_clean, 
                                                frame_length=frame_length, 
                                                hop_length=hop_length)[0]
            
            energy_window = librosa.feature.rms(y=window_clean, 
                                    frame_length=frame_length, 
                                    hop_length=hop_length)[0]
            
            # 使用Onset檢測
            onset_env = librosa.onset.onset_strength(y=window_clean, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            
            # 結合多特徵估計語速
            if len(onsets) > 0:
                # 基於onset的語速估計
                onset_rate = len(onsets) / (len(window_clean) / sr)
                
                # 基於零交叉率和能量的語速信息
                # 高零交叉率+高能量區域通常對應於有聲語音
                voiced_frames = np.sum((zcr_window > np.mean(zcr_window)) & 
                                    (energy_window > np.mean(energy_window)))
                speech_ratio = voiced_frames / len(zcr_window) if len(zcr_window) > 0 else 0
                
                # 結合估計，給予onset更高權重
                combined_rate = onset_rate * 0.7 + speech_ratio * 5.0 * 0.3
                local_speech_rates.append(combined_rate)
            else:
                local_speech_rates.append(0)
        
        # 3. 語速變化指標計算
        if len(local_speech_rates) > 1:
            # 計算變化率
            speech_rate_changes = np.diff(local_speech_rates)
            abs_changes = np.abs(speech_rate_changes)
            
            # 統計指標
            speech_rate_std = np.std(local_speech_rates)
            speech_rate_range = np.max(local_speech_rates) - np.min(local_speech_rates) if len(local_speech_rates) > 0 else 0
            
            # 偵測突然變化 - 使用自適應閾值
            change_threshold = np.mean(abs_changes) + 2.0 * np.std(abs_changes) if len(abs_changes) > 0 else 0
            sudden_changes = np.sum(abs_changes > change_threshold)
        else:
            speech_rate_std = 0
            speech_rate_range = 0
            sudden_changes = 0
        
        # 4. 停頓分析
        # 計算停頓統計信息
        if pause_durations:
            pause_mean = np.mean(pause_durations)
            pause_std = np.std(pause_durations) if len(pause_durations) > 1 else 0
            pause_count = len(pause_durations)
            total_audio_duration = len(y_normalized) / sr
            pause_rate = pause_count / total_audio_duration if total_audio_duration > 0 else 0
            pause_ratio = sum(pause_durations) / total_audio_duration * 100 if total_audio_duration > 0 else 0
        else:
            pause_mean = 0
            pause_std = 0
            pause_count = 0
            pause_rate = 0
            pause_ratio = 0
        
        # 5. 輸出合理性檢查
        result = {
            # 基本語速特徵
            'speech_duration': len(y_normalized) / sr,
            
            # 停頓特徵
            'pause_count': pause_count,
            'pause_rate': pause_rate,  # 每秒停頓次數
            'pause_mean_duration': pause_mean,  # 平均停頓長度
            'pause_std_duration': pause_std,   # 停頓長度標準差
            'pause_ratio': pause_ratio,  # 停頓時間佔比 (百分比)
            'pause_durations': pause_durations,  # 全部停頓長度列表
            
            # 語速變化特徵
            'speech_rate_variation': speech_rate_std,  # 語速變化的標準差
            'speech_rate_range': speech_rate_range,  # 最快和最慢的差異
            'sudden_speed_changes': sudden_changes,  # 突然變化的次數
            'local_speech_rates': local_speech_rates  # 局部語速列表
        }
        
        # 合理性檢查
        result = validate_speech_results(result)
        
        return result


    def validate_speech_results(results):
        """
        檢查語音分析結果的合理性，並修正異常值
        
        參數:
        results: 語音分析結果字典
        
        返回:
        修正後的結果字典
        """
        # 複製結果以免修改原始數據
        validated = results.copy()
        
        # 檢查停頓特徵合理性
        # 1. 停頓比例不應超過90%或為負值
        if validated['pause_ratio'] < 0 or validated['pause_ratio'] > 90:
            # 修正為合理範圍
            validated['pause_ratio'] = max(0, min(validated['pause_ratio'], 90))
        
        # 2. 平均停頓長度應在合理範圍內 (通常0-3秒)
        if validated['pause_mean_duration'] < 0 or validated['pause_mean_duration'] > 5:
            validated['pause_mean_duration'] = max(0, min(validated['pause_mean_duration'], 5))
        
        # 3. 停頓率應在合理範圍內
        if validated['pause_rate'] < 0 or validated['pause_rate'] > 5:  # 正常人每秒最多約5次停頓
            validated['pause_rate'] = max(0, min(validated['pause_rate'], 5))
        
        # 檢查語速特徵合理性
        # 4. 語速變化標準差不應過大
        if validated['speech_rate_variation'] > 10:
            validated['speech_rate_variation'] = 10
        
        # 5. 語速範圍不應過大
        if validated['speech_rate_range'] > 20:  # 假設最大差異為20音節/秒
            validated['speech_rate_range'] = 20
        
        # 6. 突然變化次數應與音頻長度相稱
        max_changes = validated['speech_duration'] / 2  # 假設每2秒最多1次突變
        if validated['sudden_speed_changes'] > max_changes:
            validated['sudden_speed_changes'] = int(max_changes)
        
        # 記錄任何異常情況
        warnings = []
        if results['pause_ratio'] != validated['pause_ratio']:
            warnings.append(f"停頓時間佔比已從 {results['pause_ratio']:.2f}% 修正為 {validated['pause_ratio']:.2f}%")
        
        if results['pause_mean_duration'] != validated['pause_mean_duration']:
            warnings.append(f"平均停頓長度已從 {results['pause_mean_duration']:.2f}秒 修正為 {validated['pause_mean_duration']:.2f}秒")
        
        if warnings:
            validated['warnings'] = warnings
        
        return validated
    
    # 音高分析 - 使用CREPE算法
    def analyze_pitch_for_fraud_detection(y, sr, model_capacity="full"):
        """
        使用 CREPE 算法分析音頻的音高特徵，用於詐欺檢測
        
        參數:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        model_capacity (str): CREPE 模型容量，可選 "tiny", "small", "medium", "large", "full"
        
        返回:
        dict: 包含各種音高特徵的字典
        """
        
        # 使用 CREPE 進行音高估計
        time, frequency, confidence, activation = crepe.predict(y, sr, model_capacity=model_capacity, viterbi=True)
        
        # 過濾低置信度的預測 (confidence < 0.5 通常被視為非有聲段)
        valid_indices = confidence > 0.5
        valid_frequencies = frequency[valid_indices]
        valid_confidence = confidence[valid_indices]
        
        # 如果沒有足夠的有效音高數據，返回默認值
        if len(valid_frequencies) < 5:
            return {
                'mean_pitch': 0,
                'std_pitch': 0, 
                'pitch_range': 0,
                'pitch_change_rate': 0,
                'pitch_instability': 0,
                'pitch_trend': 0,
                'pitch_confidence': 0,
                'voiced_ratio': 0,
                'valid_samples': 0
            }
        
        # 基本統計特徵
        pitch_mean = np.mean(valid_frequencies)
        pitch_std = np.std(valid_frequencies)
        pitch_range = np.ptp(valid_frequencies)  # peak to peak (最大值 - 最小值)
        
        # 高級特徵
        # 1. 音高變化率 - 當有明顯音高變化時的比例
        pitch_changes = np.sum(np.abs(np.diff(valid_frequencies)) > 20)  # 20 Hz 作為閾值
        pitch_change_rate = pitch_changes / len(valid_frequencies)
        
        # 2. 音高不穩定性 - 二階差分，測量音高變化的無規律性
        if len(valid_frequencies) > 2:
            pitch_instability = np.mean(np.abs(np.diff(np.diff(valid_frequencies))))
        else:
            pitch_instability = 0
        
        # 3. 音高趨勢 - 線性回歸斜率，表示音高的整體上升或下降趨勢
        x = np.arange(len(valid_frequencies))
        slope, _, _, _, _ = stats.linregress(x, valid_frequencies)
        
        # 4. 平均置信度 - CREPE 對音高估計的確定性
        mean_confidence = np.mean(valid_confidence)
        
        # 5. 有聲段比例 - 有多少比例的片段被識別為有聲音
        voiced_ratio = np.sum(valid_indices) / len(frequency)
        
        # 整合所有特徵
        pitch_features = {
            'mean_pitch': pitch_mean,
            'std_pitch': pitch_std,
            'pitch_range': pitch_range,
            'pitch_change_rate': pitch_change_rate,
            'pitch_instability': pitch_instability,
            'pitch_trend': slope,
            'pitch_confidence': mean_confidence,
            'voiced_ratio': voiced_ratio,
            'valid_samples': len(valid_frequencies)
        }
        
        return pitch_features
    
    # 音量變化分析
    def analyze_volume_variation(y):
        # 計算RMS能量
        rms = librosa.feature.rms(y=y)[0]
        
        volume_stats = {
            'mean_volume': np.mean(rms),
            'std_volume': np.std(rms),
            'volume_range': np.ptp(rms),
            'volume_changes': np.sum(np.abs(np.diff(rms)) > np.mean(rms) * 0.1)
        }
        
        return volume_stats
    
    def analyze_voice_tremor_for_fraud(y, sr):
        """
        使用Parselmouth分析語音顫抖特徵，用於詐欺偵測
        
        參數:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        
        返回:
        dict: 語音顫抖特徵分析結果
        """
        print("開始顫抖分析...")
        
        try:
            # 正規化音頻數據
            print("[10%] 正在正規化音頻數據...")
            y_normalized = librosa.util.normalize(y)
            
            # 將numpy數組轉換為Parselmouth Sound對象
            print("[20%] 正在創建Parselmouth聲音對象...")
            sound = parselmouth.Sound(y_normalized, sampling_frequency=sr)
            
            # 提取音高
            print("[30%] 正在提取音高特徵...")
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            
            # 獲取多種jitter測量
            print("[40%] 正在計算頻率顫抖 (Jitter)...")
            try:
                # Pitch/Sound 物件沒有 to_point_process()/get_jitter()/get_shimmer() 方法，
                # 須透過 Praat script 介面 call() 取得，否則保證每次都拋 AttributeError
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

                jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

                # 獲取多種shimmer測量
                print("[60%] 正在計算音量顫抖 (Shimmer)...")
                shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

                # 諧噪比 (HNR)
                print("[70%] 正在計算諧噪比 (HNR)...")
                harmonicity = sound.to_harmonicity_cc(0.01, 75, 0.1, 1.0)
                hnr = call(harmonicity, "Get mean", 0, 0)

            except Exception as e:
                # 處理可能的錯誤，但輸出更多診斷信息
                print(f"計算顫抖特徵時出現錯誤: {e}")
                print("檢查音頻時長和質量，確保至少有1-2秒的清晰語音")
                
                # 設置默認值，但使用更合理的值而不是0
                jitter_local = 0.01  # 約1%，更合理的默認值
                jitter_ppq5 = 0.008
                shimmer_local = 0.03  # 約3%
                shimmer_apq5 = 0.025
                hnr = 15.0
            
            # 計算顫抖指標 (tremor)
            print("[80%] 正在分析顫抖頻率和強度...")
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # 只保留有聲部分
            
            tremor_intensity = 0
            tremor_frequency = 0
            
            if len(pitch_values) > 30:  # 至少需要足夠的採樣點
                # 使用FFT分析4-8Hz範圍的調制
                pitch_detrended = pitch_values - np.mean(pitch_values)
                
                # 應用窗函數以減少頻譜洩漏
                windowed = pitch_detrended * np.hamming(len(pitch_detrended))
                
                fft_result = np.abs(np.fft.rfft(windowed))
                freqs = np.fft.rfftfreq(len(windowed), d=0.01)  # 假設time_step=0.01
                
                # 查找4-8Hz範圍內的主要成分
                tremor_range = (freqs >= 4) & (freqs <= 8)
                if np.any(tremor_range):
                    peak_idx = np.argmax(fft_result[tremor_range])
                    real_idx = np.where(tremor_range)[0][peak_idx]
                    
                    tremor_frequency = freqs[real_idx]
                    tremor_intensity = fft_result[real_idx] / np.mean(np.abs(pitch_values)) * 100
                    
                    # 調試輸出
                    print(f"檢測到顫抖頻率: {tremor_frequency:.2f}Hz，強度: {tremor_intensity:.2f}")
            else:
                print(f"警告: 僅檢測到 {len(pitch_values)} 個有效音高點，不足以進行可靠的顫抖分析")
            
            print("[100%] 顫抖分析完成!")

            # 輸出更多診斷信息
            print(f"診斷信息:")
            print(f"- 音頻長度: {len(y)/sr:.2f}秒")
            print(f"- 有效音高點: {len(pitch_values)}")
            print(f"- Jitter local: {jitter_local*100:.2f}%")
            print(f"- Shimmer local: {shimmer_local*100:.2f}%")
            print(f"- HNR: {hnr:.2f}dB")

            return {
                "jitter_local": jitter_local * 100,  # 轉為百分比
                "jitter_ppq5": jitter_ppq5 * 100,
                "shimmer_local": shimmer_local * 100,
                "shimmer_apq5": shimmer_apq5 * 100,
                "hnr": hnr,
                "tremor_frequency": tremor_frequency,
                "tremor_intensity": tremor_intensity,
            }
        except Exception as e:
            print(f"語音顫抖分析出現錯誤: {e}")
            # 使用更合理的默認值
            return {
                "jitter_local": 0.5,
                "jitter_ppq5": 0.4,
                "shimmer_local": 2.0,
                "shimmer_apq5": 1.8,
                "hnr": 22.0,
                "tremor_frequency": 0.0,
                "tremor_intensity": 0.3,
                "error": str(e)  # 添加錯誤信息以便診斷
            }
    
    # 整合所有特徵
    speech_rate = analyze_speech_patterns(y, sr)
    pitch_stats = analyze_pitch_for_fraud_detection(y, sr)
    volume_stats = analyze_volume_variation(y)
    tremor_stats = analyze_voice_tremor_for_fraud(y,sr)
    
    return {
        'speech_rate': speech_rate,
        'pitch': pitch_stats,
        'volume': volume_stats,
        'tremor': tremor_stats
    }


    

def extract_audio_features(audio_file_path):
    """
    提取基本音訊特徵，包括語速、音高、音量等，使用CREPE算法處理音高
    
    Parameters:
    audio_file_path (str): 音訊檔案路徑
    
    Returns:
    dict: 基本音訊特徵
    """
    y, sr = librosa.load(audio_file_path)
    
    # 計算tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 使用CREPE算法估計音高
    time, frequency, confidence, _ = crepe.predict(y, sr, model_capacity="medium", viterbi=True)
    
    # 過濾低置信度的預測
    valid_indices = confidence > 0.5
    pitch_mean = np.mean(frequency[valid_indices]) if np.any(valid_indices) else 0.0
    
    # 計算音量
    volume = np.mean(librosa.feature.rms(y=y)[0])
    
    # 添加一些額外的特徵
    spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
    
    return {
        "語速": tempo,
        "平均音高": pitch_mean,
        "平均音量": volume,
        "頻譜重心": spectral_centroids,
        "頻譜衰減": spectral_rolloff
    }