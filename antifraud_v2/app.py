import streamlit as st
st.set_page_config(layout="wide")
import torch
import uuid
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import librosa
import os
import sys
from models.TIM import TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer
import torch
import os
import openai
from dotenv import load_dotenv
import models.TIM as TIM
import librosa.display
import numpy as np
from faster_whisper import WhisperModel
from models.TIM import TIM_Net, TIMNet
import crepe
from scipy import stats
#from faster_whisper_youtube_local_corrected import transcribe_audio
import plotly.graph_objects as go

load_dotenv()  # 載入 .env 文件中的環境變量
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化 Whisper 模型
whisper_model = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)

# 添加項目根目錄到 Python 路徑
sys.path.append('/Users/caozhiyu/Desktop/antifruad-gpt')

# 添加 TIMNet 到安全全局列表
torch.serialization.add_safe_globals([TIMNet])

# 將這些類添加到全局命名空間
globals().update({
    'TIMNet': TIMNet,
    'TIM_Net': TIM_Net,
    'Temporal_Aware_Block': Temporal_Aware_Block,
    'Chomp1d': Chomp1d,
    'SpatialDropout': SpatialDropout,
    'WeightLayer': WeightLayer
}) 
import torch.serialization

torch.serialization.add_safe_globals([TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer])
# Load model
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None
    
    try:
        # 加載模型
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 如果加載的是狀態字典，則創建新模型並加載狀態
        if isinstance(model, dict):
            new_model = TIMNet(feature_dim=39, drop_rate=0.1, num_class=7, filters=128, dilation=8, kernel_size=2)
            new_model.load_state_dict(model)
            model = new_model
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"加載模型時發生錯誤: {str(e)}")
        return None
# MFCC extraction
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


def analyze_speech_features(audio_file_path):
    """分析語音特徵的進階函數"""
    y, sr = librosa.load(audio_file_path)
    
    # 語速分析
    def analyze_speech_rate(y, sr):
        # 使用 librosa 的 onset detection 來檢測音節
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # 計算音節數和持續時間
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        syllables_per_second = len(onsets) / (len(y) / sr)
        
        return {
            'tempo': tempo,
            'syllables_per_second': syllables_per_second,
            'speech_duration': len(y) / sr
        }
    
    # 音高變化分析
    def analyze_pitch_variation(y, sr):
        # 使用 PYIN 算法進行音高追蹤
        pitches, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                         fmin=librosa.note_to_hz('C2'),
                                                         fmax=librosa.note_to_hz('C7'),
                                                         sr=sr)
        
        # 只考慮有效的音高值
        valid_pitches = pitches[voiced_flag]
        
        if len(valid_pitches) > 0:
            pitch_stats = {
                'mean_pitch': np.mean(valid_pitches),
                'std_pitch': np.std(valid_pitches),
                'pitch_range': np.ptp(valid_pitches),
                'pitch_changes': np.sum(np.abs(np.diff(valid_pitches)) > 1)
            }
        else:
            pitch_stats = {
                'mean_pitch': 0,
                'std_pitch': 0,
                'pitch_range': 0,
                'pitch_changes': 0
            }
            
        return pitch_stats
    
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
    
    # 整合所有特徵
    speech_rate = analyze_speech_rate(y, sr)
    pitch_stats = analyze_pitch_variation(y, sr)
    volume_stats = analyze_volume_variation(y)
    
    return {
        'speech_rate': speech_rate,
        'pitch': pitch_stats,
        'volume': volume_stats
    }


# Emotion prediction
def predict_emotion(model, audio_file_path):
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    x = get_mfcc(audio_file_path)
    x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (0, 2, 1))
    
    with torch.no_grad():
        predictions = model(torch.tensor(x, dtype=torch.float32))
        probabilities = torch.softmax(predictions, dim=1)
        _, predicted_label_index = torch.max(probabilities, 1)
        predicted_emotion = emotion_labels[predicted_label_index.item()]
    
    return predicted_emotion, probabilities.squeeze().tolist()

def get_emotion_parameters(model, audio_file_path):
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    _, probabilities = predict_emotion(model, audio_file_path)
    
    return {emotion: float(prob) for emotion, prob in zip(emotion_labels, probabilities)}
    
    parameters = []
    for emotion, prob in zip(emotion_labels, probabilities):
        # 確保 prob 是一個 Python 浮點數
        prob_float = float(prob)
        parameters.append(f"{emotion}: {prob_float:.4f}")
    
    return ", ".join(parameters)


def create_audio_waveform(audio_file):
    # 加载音频文件
    y, sr = librosa.load(audio_file)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 1))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    
    # 设置背景色为透明
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # 设置刻度标签颜色为白色
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    return fig

def transcribe_audio(audio_path):
    segments, info = whisper_model.transcribe(audio_path, language="zh")
    return " ".join([segment.text for segment in segments])

def get_emotion_explanation(emotion, parameters, audio_file_path):
    transcription = transcribe_audio(audio_file_path)
    audio_features = extract_audio_features(audio_file_path)
    
    # 安全地格式化音頻特徵
    audio_features_str = ", ".join([f"{key}: {float(value):.4f}" for key, value in audio_features.items()])
    
    # 確保 parameters 是字符串
    if not isinstance(parameters, str):
        parameters = str(parameters)
    
    prompt = f"""音頻逐字稿："{transcription}"
    音頻特徵：{audio_features_str}
    情緒預測結果為 {emotion}，參數為 {parameters}。
    請根據音頻逐字稿的內容、語氣變化以及音頻特徵，分析為什麼會得出這個情緒預測結果。特別注意說話者使用的詞語、語氣轉變、情緒強烈或不一致的地方，並根據欺騙模式的分類，具體指出該音頻中可能反映出欺騙行為的跡象。請列出 3-5 點具體分析，聚焦於以下方面：
    語音情緒與語速：該音頻中是否出現了高壓、憤怒、或極度緊張的情緒？這些情緒是否表現出攻擊性（欺騙模式#1 - 攻擊性謊言）或語調上的矛盾（欺騙模式#2 - 矛盾衝突）？
    絕對化的表述：語音或逐字稿中是否使用了「絕對」、「從未」等強烈否定的詞彙（欺騙模式#3 - 明確否認），這是否表示說話者在試圖否認某些真相？
    情緒掩飾或猶豫：是否出現猶豫不決、語音顫抖或表現出極度不自在（欺騙模式#4 - 尷尬掩蓋；欺騙模式#6 - 猶豫不決）？這些情緒波動是否反映出說話者在隱藏某些事實或努力掩飾？
    避談細節或過度誇張：該音頻中是否存在迴避具體問題、或使用模糊、空洞的表達（欺騙模式#5 - 警覺避談），亦或是過度誇張（欺騙模式#7 - 異常興奮）？這是否表現出說話者對某些事實有所隱瞞？
    邏輯漏洞：逐字稿中是否出現前後矛盾或語意上存在邏輯缺陷（欺騙模式#8 - 邏輯漏洞）？這是否可能表明說話者在故意混淆事實？
    請在你的回答中包含一個明確的詐欺可能性百分比，格式為 'X%'。輸出一定要用繁體中文
    語音特徵分析：
    - 語速: {advanced_features['speech_rate']['syllables_per_second']:.2f} 音節/秒
    - 平均音高: {advanced_features['pitch']['mean_pitch']:.2f} Hz
    - 音高變化: {advanced_features['pitch']['std_pitch']:.2f} Hz
    - 音量變化: {advanced_features['volume']['std_volume']:.2f}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一個專業的情緒分析助手，專門分析中文語音內容和語氣，並判斷他是否有騙人、詐欺的可能性，請在你的回答中包含一個明確的詐欺可能性百分比，格式為 'X%'。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content']

def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    
    # 計算tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 使用 piptrack 來估計音高
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > 0]) if np.any(magnitudes > 0) else 0.0
    
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

# Emotion translation
def translate_emotion(english_emotion):
    translation_dict = {
        'anger': '憤怒',
        'disgust': '厭惡',
        'fear': '恐懼',
        'happy': '快樂',
        'neutral': '中立',
        'boredom': '無聊',
        'sad': '悲傷'
    }
    return translation_dict.get(english_emotion, '未知情緒')

# Emotion visualization
def visualize_emotion(emotion):
    emotion_xyz = {
        '憤怒': {'x': -1.5, 'y': 1.5, 'z': 1.0},
        '厭惡': {'x': -1.2, 'y': -0.5, 'z': 0.8},
        '恐懼': {'x': -1.3, 'y': 1.0, 'z': 0.9},
        '快樂': {'x': 1.5, 'y': 1.0, 'z': 1.0},
        '中立': {'x': 0, 'y': 0, 'z': 0.5},
        '無聊': {'x': -0.5, 'y': -1.5, 'z': 0.4},
        '悲傷': {'x': -1.0, 'y': -1.0, 'z': 0.8}
    }
    
    xyz = emotion_xyz[emotion]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=[xyz['x']],
        y=[xyz['y']],
        z=[xyz['z']],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        ),
        text=[emotion],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-2,2]),
            yaxis = dict(range=[-2,2]),
            zaxis = dict(range=[0,1.5]),
            aspectmode='cube'
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    return fig


def create_fraud_probability_gauge(fraud_probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", 
        value = fraud_probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.5},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0,100,0,0.7)'},
                {'range': [20, 40], 'color': 'rgba(50,100,0,0.7)'},
                {'range': [40, 60], 'color': 'rgba(100,100,0,0.7)'},
                {'range': [60, 80], 'color': 'rgba(100,50,0,0.7)'},
                {'range': [80, 100], 'color': 'rgba(100,0,0,0.7)'}],
            }))

    fig.update_layout(
        title=dict(text="詐欺可能性", font=dict(size=20, color="white"), x=0),
        height=350,
        width=350,
        margin=dict(t=30,b=30),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Arial"},
        coloraxis_colorbar=dict(tickfont=dict(color="white"))
    )

    return fig


import numpy as np

# 在 main() 函數中添加視覺化
def create_speech_features_plots(audio_features):
    # Create three separate figures instead of subplots
    # Speech Rate Gauge (as a separate figure)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=audio_features['speech_rate']['syllables_per_second'],
        gauge={
            'axis': {'range': [0, 8]},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 6], 'color': "gray"},
                {'range': [6, 8], 'color': "darkgray"}
            ]
        },
        title={'text': "語速 (音節/秒)"}
    ))
    
    gauge_fig.update_layout(
        height=300,
        margin=dict(t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    # Pitch Changes Bar Chart
    pitch_fig = go.Figure()
    pitch_fig.add_trace(
        go.Bar(
            x=['平均音高', '音高標準差', '音高範圍'],
            y=[audio_features['pitch']['mean_pitch'],
               audio_features['pitch']['std_pitch'],
               audio_features['pitch']['pitch_range']],
            marker_color=['blue', 'red', 'green']
        )
    )
    
    pitch_fig.update_layout(
        title="音高變化",
        height=300,
        margin=dict(t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    # Volume Changes Bar Chart
    volume_fig = go.Figure()
    volume_fig.add_trace(
        go.Bar(
            x=['平均音量', '音量標準差', '音量範圍'],
            y=[audio_features['volume']['mean_volume'],
               audio_features['volume']['std_volume'],
               audio_features['volume']['volume_range']],
            marker_color=['blue', 'red', 'green']
        )
    )
    
    volume_fig.update_layout(
        title="音量變化",
        height=300,
        margin=dict(t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    return gauge_fig, pitch_fig, volume_fig

def create_emotion_radar_chart(emotion_parameters):
    # 排序情緒
    sorted_emotions = sorted(emotion_parameters.items(), key=lambda x: x[1], reverse=True)
    emotions, values = zip(*sorted_emotions)

    # 使用對數尺度調整值，同時設置最小可見值
    log_values = np.log10([max(v, 0.001) for v in values])  # 設置最小值為0.001
    scaled_values = (log_values - min(log_values)) / (max(log_values) - min(log_values))

    fig = go.Figure()
    
    # 添加雷達圖
    fig.add_trace(go.Scatterpolar(
        r=scaled_values,
        theta=emotions,
        fill='toself',
        line=dict(color='rgb(31, 119, 180)', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)',
    ))
    
    # 更新雷達圖布局
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".2f",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["0.001", "0.01", "0.1", "0.5", "1"],
                tickfont=dict(size=12, color="white"),
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color="white"),
            ),
            bgcolor="rgba(0, 0, 0, 0)"
        ),
        showlegend=False,
        title=dict(
            text="情緒雷達圖",
            font=dict(size=20, color="white"),
            x=0
        ),
        height=350,
        width=350,
        margin=dict(t=30,b=30),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white")
    )
    
    return fig

def create_xyz_chart(emotion):
    emotion_xyz = {
        '憤怒': {'x': -1.5, 'y': 1.5, 'z': 1.0},
        '厭惡': {'x': -1.2, 'y': -0.5, 'z': 0.8},
        '恐懼': {'x': -1.3, 'y': 1.0, 'z': 0.9},
        '快樂': {'x': 1.5, 'y': 1.0, 'z': 1.0},
        '中立': {'x': 0, 'y': 0, 'z': 0.5},
        '無聊': {'x': -0.5, 'y': -1.5, 'z': 0.4},
        '悲傷': {'x': -1.0, 'y': -1.0, 'z': 0.8}
    }
    
    xyz = emotion_xyz[emotion]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=[xyz['x']],
        y=[xyz['y']],
        z=[xyz['z']],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        ),
        text=[emotion],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=dict(text="情緒三軸圖", font=dict(size=20, color="white"), x=0),
        height=350,
        width=350,
        scene = dict(
            xaxis = dict(range=[-2,2], title='X'),
            yaxis = dict(range=[-2,2], title='Y'),
            zaxis = dict(range=[0,1.5], title='Z'),
            aspectmode='cube'
        ),
        # title=dict(
        #     text="情緒三軸圖",
        #     font=dict(size=24, color="white")
        # ),
        # height=500,
        margin=dict(r=20, l=10, b=30, t=30)
    )
    
    return fig

def create_risk_guidance_section(fraud_probability):
    """創建風險導流建議區塊"""
    if fraud_probability is None:
        return None
        
    guidance = ""
    risk_level = ""
    color = ""
    
    if fraud_probability <= 20:  # 極低風險
        risk_level = "極低風險 (Ultra-Low Risk)"
        guidance = """
        • 可進行基本過濾
        • 建議處理方式：
          - 依照標準作業流程處理
          - 確認基本資料完整性
          - 可由客服人員直接處理
        • 無需特殊審核程序
        """
        color = "rgba(0, 255, 0, 0.1)"  # 淺綠色
        
    elif fraud_probability <= 40:  # 低風險
        risk_level = "低風險 (Low Risk)"
        guidance = """
        • 需要基礎審核
        • 建議處理方式：
          - 核對申請者身份與聯絡資訊
          - 確認用戶使用設備資訊
          - 由中級風險評估專員進行評估
        • 建立基礎風險記錄
        """
        color = "rgba(144, 238, 144, 0.1)"  # 淺綠色
        
    elif fraud_probability <= 60:  # 中等風險
        risk_level = "中等風險 (Moderate Risk)"
        guidance = """
        • 需要進階審核
        • 建議處理方式：
          - 進行完整的財務背景審查
          - 要求額外的身份證明文件
          - 諮詢法律顧問意見
          - 記錄完整審核過程
        • 通知部門主管關注此案件
        """
        color = "rgba(255, 255, 0, 0.1)"  # 淺黃色
        
    elif fraud_probability <= 80:  # 高風險
        risk_level = "高風險 (High Risk)"
        guidance = """
        • 需要特殊處理
        • 建議處理方式：
          - 立即通知資深風險管理人員
          - 進行深入的背景調查
          - 要求完整的財務證明文件
          - 諮詢外部專業法律意見
        • 需要主管層級審批
        """
        color = "rgba(255, 165, 0, 0.1)"  # 淺橙色
        
    else:  # 極高風險
        risk_level = "極高風險 (Ultra-High Risk)"
        guidance = """
        • 需要立即處理
        • 建議處理方式：
          - 立即通報相關安全單位
          - 凍結相關帳戶或交易
          - 啟動緊急應變程序
          - 準備完整報告給高層主管
        • 進行案件全程錄音錄影
        """
        color = "rgba(255, 0, 0, 0.1)"  # 淺紅色

    return risk_level, guidance, color

def create_mock_speaker_display():
    """創建模擬的說話者顯示區域"""
    # 使用 plotly 創建簡單的時間軸視覺化
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    # 模擬數據
    times = np.linspace(0, 25, 100)  # 25秒的對話
    speaker1_data = [1 if (5 < t < 8) or (12 < t < 15) or (20 < t < 23) else 0 for t in times]
    speaker2_data = [1 if (2 < t < 5) or (9 < t < 12) or (16 < t < 19) else 0 for t in times]

    # 創建圖表
    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7])

    # 添加說話者時間軸
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=speaker1_data,
            name="客服人員",
            fill='tozeroy',
            line=dict(color='rgba(100, 149, 237, 0.8)'),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=times, 
            y=[-v for v in speaker2_data],  # 反轉值以在下方顯示
            name="客戶",
            fill='tozeroy',
            line=dict(color='rgba(144, 238, 144, 0.8)'),
        ),
        row=1, col=1
    )

    # 更新布局
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white")
        ),
        font=dict(color="white")
    )

    # 更新軸線設置
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    return fig

def main():
    st.title('AntiFraudGPT - 音訊情緒分析')
    fraud_probability = None

        # 側邊欄
    with st.sidebar:
        st.title("AntiFraudGPT")
        st.subheader("音訊情緒分析")
        uploaded_file = st.file_uploader("📂 上傳音訊檔案", type=['mp3', 'wav', 'ogg'])

    if uploaded_file is not None:
        # 顯示通話資訊
        display_call_info()
        
        # 顯示說話者時間軸
        speaker_fig = create_mock_speaker_display()
        st.plotly_chart(speaker_fig, use_container_width=True, config={'displayModeBar': False})
        
        # 顯示音訊波形圖
        waveform_fig = create_audio_waveform(uploaded_file)
        st.pyplot(waveform_fig, use_container_width=True)

        st.markdown("<div style='margin: 2em 0;'></div>", unsafe_allow_html=True)

        model_path = 'models/TIM-7_False_drop25_mfcc_smoothTrue_epoch500_l2re1_lr005_best.pt'
        model = load_model(model_path)

        if model is None:
            st.error("無法載入模型，請檢查模型檔案路徑是否正確。")
            return

        try:
            with st.spinner('分析中...'):
                temp_file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                advanced_features = analyze_speech_features(temp_file_path)
                #features_plot = create_speech_features_plots(advanced_features)
                #st.plotly_chart(features_plot, use_container_width=True)
                gauge_fig, pitch_fig, volume_fig = create_speech_features_plots(advanced_features)
                st.markdown("### 語音特徵分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
                with col2:
                    st.plotly_chart(pitch_fig, use_container_width=True, config={'displayModeBar': False})
                st.plotly_chart(volume_fig, use_container_width=True, config={'displayModeBar': False})

                # 預測和分析
                predicted_emotion, probabilities = predict_emotion(model, temp_file_path)
                translated_emotion = translate_emotion(predicted_emotion)
                
                emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
                emotion_parameters = dict(zip(emotion_labels, probabilities))
                audio_features = extract_audio_features(temp_file_path)
                processed_audio_features = {
                    key: float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
                    for key, value in audio_features.items()
                }

                # 基本指標顯示
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**預測情緒**")
                    st.markdown(f"<div style='background-color: rgba(70, 130, 180, 0.1); padding: 10px; border-radius: 5px; text-align: center;'><h4 style='margin: 0;'>{translated_emotion}</h4></div>", unsafe_allow_html=True)
                with col2:
                    st.metric(label="語速", value=f"{processed_audio_features['語速']:.2f}")
                with col3:
                    st.metric(label="平均音高", value=f"{processed_audio_features['平均音高']:.2f}")
                with col4:
                    st.metric(label="平均音量", value=f"{processed_audio_features['平均音量']:.4f}")
                
                st.markdown("<div style='margin: 2em 0;'></div>", unsafe_allow_html=True)
                

                
                # 視覺化區域
                col1, col2, col3 = st.columns(3)
                with col1:
                    emotion_radar = create_emotion_radar_chart(emotion_parameters)
                    st.plotly_chart(emotion_radar, use_container_width=True, config={'displayModeBar': False})
                with col2:
                    xyz_chart = create_xyz_chart(translated_emotion)
                    st.plotly_chart(xyz_chart, use_container_width=True, config={'displayModeBar': False})

                # 情緒概率分佈表格
                st.markdown("#### 情緒概率分佈")
                # 創建包含原始數值的DataFrame
                df = pd.DataFrame({
                    "情緒": [translate_emotion(k) for k in emotion_parameters.keys()],
                    "概率值": list(emotion_parameters.values()),  # 保存原始數值用於排序
                    "概率": [f"{v:.4f}" for v in emotion_parameters.values()]  # 格式化顯示
                })

                # 根據概率值排序
                df = df.sort_values(by='概率值', ascending=False)

                # 重置索引並隱藏
                df = df.reset_index(drop=True)

                # 移除用於排序的列
                df = df.drop('概率值', axis=1)

                # 顯示表格
                st.table(df)

                # 詳細分析
                with st.expander("🔍 詳細分析", expanded=False):
                    emotion_params_str = ", ".join([f"{translate_emotion(k)}: {v:.4f}" for k, v in emotion_parameters.items()])
                    explanation = get_emotion_explanation(translated_emotion, emotion_params_str, temp_file_path)
                    
                    # 從解釋中提取詐欺可能性
                    import re
                    fraud_probability_match = re.search(r'(\d+(?:\.\d+)?)%', explanation)
                    if fraud_probability_match:
                        fraud_probability = float(fraud_probability_match.group(1))
                    
                    st.write(explanation)

                # 只在最後顯示一次詐欺可能性圖表
                with col3:
                    if fraud_probability is not None:
                        fraud_chart = create_fraud_probability_gauge(fraud_probability)
                        st.plotly_chart(fraud_chart, use_container_width=True, config={'displayModeBar': False})
                
                if fraud_probability is not None:
                    st.markdown("<div style='margin: 2em 0;'></div>", unsafe_allow_html=True)
                    st.markdown("### 🔄 導流專區")
                    
                    risk_level, guidance, color = create_risk_guidance_section(fraud_probability)
                    
                    # 使用 container 來添加背景色
                    with st.container():
                        st.markdown(
                            f"""
                            <div style='background-color: {color}; padding: 1em; border-radius: 10px;'>
                                <h4 style='color: white; margin-bottom: 1em;'>{risk_level}</h4>
                                <pre style='color: white; white-space: pre-wrap;'>{guidance}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        except Exception as e:
            st.error(f"分析過程中發生錯誤: {str(e)}")
            st.error(f"錯誤類型: {type(e)}")
            st.error(f"錯誤發生位置: {e.__traceback__.tb_frame.f_code.co_filename}, 行 {e.__traceback__.tb_lineno}")
        
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if __name__ == "__main__":
    main()