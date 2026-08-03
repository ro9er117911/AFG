# 在 model_handling.py 中
import sys
import torch
import os
import streamlit as st
import numpy as np
import librosa
from audio_processing import get_mfcc

# 添加模型目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 從正確的路徑導入模型類
from models.TIM import TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer

# 將這些類添加到全局命名空間
globals().update({
    'TIMNet': TIMNet,
    'TIM_Net': TIM_Net,
    'Temporal_Aware_Block': Temporal_Aware_Block,
    'Chomp1d': Chomp1d,
    'SpatialDropout': SpatialDropout,
    'WeightLayer': WeightLayer
})

# 添加到安全反序列化列表
torch.serialization.add_safe_globals([TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer])
@st.cache_resource
def load_model(model_path):
    """
    載入預訓練的情緒分析模型
    
    Parameters:
    model_path (str): 模型檔案路徑
    
    Returns:
    torch.nn.Module: 載入的模型，若載入失敗則返回None
    """
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

def predict_emotion(model, audio_file_path, window_size=4.0, stride=2.0, max_windows=30):
    """
    預測音訊的情緒

    模型（TIMNet）是用固定長度（window_size 秒）的語句片段訓練的，一次只能吃一個固定長度的
    窗口。對於超過 window_size 秒的通話，改用滑動窗口在整段音訊上取多個片段分別預測、再平均
    各窗口的機率分佈，而不是只用整通電話最前面 window_size 秒（往往只是開場白）代表整通電話。
    對於長度不超過 window_size 秒的短音檔，行為與過去單窗口預測完全相同。

    Parameters:
    model (torch.nn.Module): 已載入的情緒分析模型
    audio_file_path (str): 音訊檔案路徑
    window_size (float): 每個窗口的長度（秒），需與模型訓練時使用的固定長度一致
    stride (float): 窗口滑動步長（秒）
    max_windows (int): 最多取樣窗口數，避免超長音檔運算量過大（超過時改用等距取樣）

    Returns:
    tuple: (預測的情緒標籤, 各情緒的平均機率)
    """
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    total_duration = librosa.get_duration(path=audio_file_path)

    if total_duration <= window_size:
        offsets = [0.0]
    else:
        offsets = list(np.arange(0.0, total_duration - window_size + 1e-9, stride))
        if len(offsets) > max_windows:
            offsets = list(np.linspace(0.0, total_duration - window_size, max_windows))

    window_probabilities = []
    for offset in offsets:
        x = get_mfcc(audio_file_path, offset=offset)
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, (0, 2, 1))

        with torch.no_grad():
            predictions = model(torch.tensor(x, dtype=torch.float32))
            probabilities = torch.softmax(predictions, dim=1)
        window_probabilities.append(probabilities.squeeze(0).numpy())

    avg_probabilities = np.mean(window_probabilities, axis=0)
    predicted_emotion = emotion_labels[int(np.argmax(avg_probabilities))]

    return predicted_emotion, avg_probabilities.tolist()

def get_emotion_parameters(model, audio_file_path):
    """
    獲取各情緒的機率參數
    
    Parameters:
    model (torch.nn.Module): 已載入的情緒分析模型
    audio_file_path (str): 音訊檔案路徑
    
    Returns:
    dict: 各情緒的機率參數
    """
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    _, probabilities = predict_emotion(model, audio_file_path)
    
    return {emotion: float(prob) for emotion, prob in zip(emotion_labels, probabilities)}