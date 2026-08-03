import torch
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
import sys
from pathlib import Path
load_dotenv()

# 確保可以引入 TIM 模型
# 獲取專案根目錄
ROOT_DIR = Path(__file__).parent.parent

try:
    sys.path.append(str(ROOT_DIR))
    from models.TIM import TIMNet
except ImportError as e:
    st.error(f"無法引入 TIM 模型: {e}")
    raise

load_dotenv()

@st.cache_resource
def load_model(model_path):
    """載入模型函數"""
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None
    
    try:
        # 確保 TIMNet 在全局命名空間
        globals()['TIMNet'] = TIMNet
        
        # 載入模型
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 創建新模型實例
        model = TIMNet(
            feature_dim=39,
            drop_rate=0.1,
            num_class=7,
            filters=128,
            dilation=8,
            kernel_size=2
        )
        
        # 如果載入的是狀態字典，直接加載
        if isinstance(model_data, dict):
            model.load_state_dict(model_data)
        else:
            # 如果載入的是完整模型，獲取其狀態字典
            model.load_state_dict(model_data.state_dict())
            
        model.eval()  # 設置為評估模式
        return model
        
    except Exception as e:
        st.error(f"載入模型時發生錯誤: {str(e)}")
        import traceback
        st.error(f"詳細錯誤信息: {traceback.format_exc()}")
        return None
    
# 獲取專案根目錄
ROOT_DIR = Path(__file__).parent.parent

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None
    
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(model, dict):
            new_model = TIMNet(feature_dim=39, drop_rate=0.1, num_class=7, filters=128, dilation=8, kernel_size=2)
            new_model.load_state_dict(model)
            model = new_model
        model.eval()
        return model
    except Exception as e:
        st.error(f"加載模型時發生錯誤: {str(e)}")
        return None

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

def get_emotion_explanation(emotion, parameters, audio_file_path):
    transcription = transcribe_audio(audio_file_path)
    audio_features = extract_audio_features(audio_file_path)
    audio_features_str = ", ".join([f"{key}: {float(value):.4f}" for key, value in audio_features.items()])
    
    if not isinstance(parameters, str):
        parameters = str(parameters)
    
    prompt = f"""音頻逐字稿："{transcription}"
    音頻特徵：{audio_features_str}
    情緒預測結果為 {emotion}，參數為 {parameters}。
    請根據音頻逐字稿的內容、語氣變化以及音頻特徵，分析為什麼會得出這個情緒預測結果..."""  # 此處省略了完整prompt以節省空間
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一個專業的情緒分析助手，專門分析中文語音內容和語氣，並判斷他是否有騙人、詐欺的可能性，請在你的回答中包含一個明確的詐欺可能性百分比，格式為 'X%'。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content']