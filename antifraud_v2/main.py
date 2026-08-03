import streamlit as st
import openai
import os
import time
import hashlib
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import sys
import re
from fraud_detection import detect_fraud_patterns, visualize_fraud_patterns, FraudPatternClassifier
from multimodal_fusion import integrate_audio_text_analysis
from language_processing import get_structured_fraud_analysis

# 確保當前目錄在導入路徑中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入 TIM 模型相關類別，確保模型能夠正確載入
from models.TIM import TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer

# 將這些類添加到全局命名空間，解決模型載入問題
globals().update({
    'TIMNet': TIMNet,
    'TIM_Net': TIM_Net,
    'Temporal_Aware_Block': Temporal_Aware_Block,
    'Chomp1d': Chomp1d,
    'SpatialDropout': SpatialDropout,
    'WeightLayer': WeightLayer
})

# 添加安全反序列化的類別
torch.serialization.add_safe_globals([TIMNet, TIM_Net, Temporal_Aware_Block, Chomp1d, SpatialDropout, WeightLayer])

# 導入其他模組
from audio_processing import get_mfcc, analyze_speech_features
from model_handling import load_model, predict_emotion
from visualization import create_audio_waveform
from language_processing import transcribe_audio
from utils import translate_emotion, create_risk_guidance_section, generate_call_id, evaluate_feature_status

# 定義情緒標籤，將其移到全局範圍
EMOTION_LABELS = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

load_dotenv()  # 載入 .env 文件中的環境變量
openai.api_key = os.getenv("OPENAI_API_KEY")

# 設定頁面
st.set_page_config(layout="wide")

def display_call_info(call_id, call_date, duration_str):
    """顯示通話基本資訊"""
    st.markdown("### 通話資訊")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**通話 ID**: {call_id}")
    with col2:
        st.markdown(f"**通話日期**: {call_date}")
    with col3:
        st.markdown(f"**通話時長**: {duration_str}")

def create_pause_pattern_chart(speech_rate_data):
    """
    創建停頓模式視覺化圖表
    
    Parameters:
    speech_rate_data (dict): 語速分析數據
    
    Returns:
    plotly.graph_objects.Figure: 停頓模式圖表
    """
    # 提取停頓持續時間
    pause_durations = speech_rate_data.get('pause_durations', [])
    
    # 確保停頓持續時間是有效的列表
    if not isinstance(pause_durations, list) or not pause_durations:
        # 返回一個空的圖表
        fig = go.Figure()
        fig.update_layout(
            title="無可用的停頓數據",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black")
        )
        return fig
    
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("停頓分佈", "停頓持續時間"),
                       vertical_spacing=0.2,
                       row_heights=[0.6, 0.4])
    
    # 停頓分佈圖 (直方圖)
    fig.add_trace(
        go.Histogram(
            x=pause_durations,
            nbinsx=20,
            marker_color='rgba(70, 130, 180, 0.6)',
            name="停頓分佈"
        ),
        row=1, col=1
    )
    
    # 停頓時間序列圖
    pause_indices = np.arange(len(pause_durations))
    fig.add_trace(
        go.Scatter(
            x=pause_indices, 
            y=pause_durations,
            mode='lines+markers',
            marker=dict(size=8, color='rgba(70, 130, 180, 0.8)'),
            line=dict(width=2, color='rgba(70, 130, 180, 0.5)'),
            name="停頓持續時間"
        ),
        row=2, col=1
    )
    
    # 更新佈局
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        showlegend=False
    )
    
    fig.update_xaxes(title_text="停頓持續時間 (秒)", row=1, col=1)
    fig.update_yaxes(title_text="頻率", row=1, col=1)
    
    fig.update_xaxes(title_text="停頓索引", row=2, col=1)
    fig.update_yaxes(title_text="持續時間 (秒)", row=2, col=1)
    
    return fig

def create_speech_rate_chart(speech_rate_data):
    """
    創建語速變化視覺化圖表
    
    Parameters:
    speech_rate_data (dict): 語速分析數據
    
    Returns:
    plotly.graph_objects.Figure: 語速變化圖表
    """
    # 提取局部語速數據
    local_speech_rates = speech_rate_data.get('local_speech_rates', [])
    
    # 確保局部語速是有效的列表
    if not isinstance(local_speech_rates, list) or not local_speech_rates:
        # 返回一個空的圖表
        fig = go.Figure()
        fig.update_layout(
            title="無可用的語速變化數據",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black")
        )
        return fig
    
    # 創建語速變化圖
    fig = go.Figure()
    
    # 添加時間序列
    time_indices = np.arange(len(local_speech_rates)) * 0.5  # 0.5秒步進
    fig.add_trace(
        go.Scatter(
            x=time_indices, 
            y=local_speech_rates,
            mode='lines',
            line=dict(width=2, color='rgba(70, 130, 180, 0.8)'),
            name="局部語速"
        )
    )
    
    # 添加均值線
    mean_rate = np.mean(local_speech_rates)
    fig.add_trace(
        go.Scatter(
            x=[time_indices[0], time_indices[-1]],
            y=[mean_rate, mean_rate],
            mode='lines',
            line=dict(width=1, color='rgba(255, 0, 0, 0.5)', dash='dash'),
            name="平均語速"
        )
    )
    
    # 標記突然變化點
    if len(local_speech_rates) > 1:
        changes = np.diff(local_speech_rates)
        std_change = np.std(changes)
        significant_changes = np.where(np.abs(changes) > 2 * std_change)[0]
        
        if len(significant_changes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_indices[significant_changes + 1],
                    y=[local_speech_rates[i+1] for i in significant_changes],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='x'),
                    name="顯著變化點"
                )
            )
    
    # 更新佈局
    fig.update_layout(
        title="語速變化曲線",
        height=300,
        margin=dict(l=0, r=0, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        xaxis=dict(title="時間 (秒)"),
        yaxis=dict(title="語速 (音節/秒)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    st.title('AntiFraudGPT - 音訊情緒分析')

    # 側邊欄
    with st.sidebar:
        st.title("AntiFraudGPT")
        st.subheader("音訊情緒分析")
        input_mode = st.radio("音訊來源", ["📂 上傳音檔（非即時）", "🎙️ 即時錄音"])
        if input_mode == "📂 上傳音檔（非即時）":
            input_audio = st.file_uploader("📂 上傳音訊檔案", type=['mp3', 'wav', 'ogg'])
        else:
            input_audio = st.audio_input("🎙️ 點擊開始錄音")

    if input_audio is not None:
        # 錄音檔沒有副檔名，補上 .wav 以便 librosa/soundfile 正確讀取
        audio_name = getattr(input_audio, "name", None) or "recording.wav"
        if not os.path.splitext(audio_name)[1]:
            audio_name += ".wav"

        audio_bytes = input_audio.getbuffer()

        # 每份音訊（不論上傳或錄音）只在第一次出現時產生通話 ID，避免 Streamlit rerun 造成流水號重複遞增
        file_id = getattr(input_audio, "file_id", None) or audio_name
        call_info_cache = st.session_state.setdefault("call_info_cache", {})
        if file_id not in call_info_cache:
            call_info_cache[file_id] = generate_call_id()
        call_id, call_date = call_info_cache[file_id]

        # 暫存檔名用音檔「內容」的雜湊組成，而不是原始檔名——即時錄音多半沒有 .name，
        # 會塌縮成固定的 "recording.wav"，若直接拿來當路徑，不同錄音會共用同一條路徑，
        # 導致 analyze_speech_features 的 st.cache_data（依路徑字串快取）吃到別次錄音的舊結果
        content_hash = hashlib.sha1(audio_bytes).hexdigest()[:16]
        audio_ext = os.path.splitext(audio_name)[1]
        temp_file_path = os.path.join(os.getcwd(), f"{content_hash}{audio_ext}")
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)

        duration_seconds = librosa.get_duration(path=temp_file_path)
        minutes, seconds = divmod(int(round(duration_seconds)), 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"

        # 顯示通話資訊
        display_call_info(call_id, call_date, duration_str)

        # 顯示音訊波形圖
        waveform_fig = create_audio_waveform(temp_file_path)
        st.pyplot(waveform_fig, use_container_width=True)

        st.markdown("<div style='margin: 2em 0;'></div>", unsafe_allow_html=True)

        model_path = 'models/TIM-7_False_drop25_mfcc_smoothTrue_epoch500_l2re1_lr005_best.pt'
        model = load_model(model_path)

        if model is None:
            st.error("無法載入模型，請檢查模型檔案路徑是否正確。")
            return

        try:
            analysis_start_time = time.time()
            with st.spinner('分析中...'):
                # 1. 音訊特徵分析
                advanced_features = analyze_speech_features(temp_file_path)

                # 2. 情緒預測
                predicted_emotion, probabilities = predict_emotion(model, temp_file_path)
                translated_emotion = translate_emotion(predicted_emotion)
                emotion_parameters = dict(zip(EMOTION_LABELS, probabilities))

                # 3. 文字轉錄
                transcript = transcribe_audio(temp_file_path)
                transcription_ok = bool(transcript) and transcript not in ["音訊檔案不存在", "無法轉錄音訊"]
                if not transcription_ok:
                    st.error(f"音訊轉錄失敗：{transcript}")
                    st.warning("將繼續進行音訊特徵分析，但無法進行文字欺騙分析")
                elif isinstance(transcript, list):
                    valid_texts = [t for t in transcript if isinstance(t, str) and t.strip()]
                    transcript = " ".join(valid_texts) if valid_texts else ""
                    transcription_ok = bool(transcript)

                # 4. 音訊欺騙模式分析
                audio_fraud_report = detect_fraud_patterns(advanced_features, emotion_parameters)

                # 5. 文字欺騙模式分析（僅在轉錄成功時進行；失敗時給空字串，讓融合分析走既有的「無文字訊號」降級邏輯）
                text_fraud_analysis = get_structured_fraud_analysis(transcript, translated_emotion) if transcription_ok else ""

                # 6. 融合分析
                integrated_fraud_report = integrate_audio_text_analysis(audio_fraud_report, text_fraud_analysis)

                # 7. 關鍵指標狀態判讀（給前端卡片用，取代原始數字表格）
                feature_status = evaluate_feature_status(advanced_features)

            analysis_latency = time.time() - analysis_start_time

            # ========== 以下為結果呈現：結論優先，細節收合 ==========

            st.caption(f"⏱️ 分析耗時：{analysis_latency:.1f} 秒")

            # 🎯 風險判讀結果（最上方，第一線人員第一眼要看到的結論）
            st.markdown("## 🎯 風險判讀結果")
            risk_score = integrated_fraud_report['risk_score']
            risk_level = integrated_fraud_report['risk_level']
            risk_color = {
                "低風險": "green",
                "中風險": "orange",
                "高風險": "red"
            }.get(risk_level, "gray")
            fusion_confidence = integrated_fraud_report.get('fusion_confidence', 0.0) * 100
            primary_patterns_text = "、".join(
                f"{pattern}（{score*100:.0f}%）" for pattern, score in integrated_fraud_report["primary_patterns"]
            ) or "無明顯模式"

            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown(f"""
                <div style='background-color: {risk_color}25; padding: 16px; border-radius: 8px; height: 100%;'>
                <h3 style='color: {risk_color}; margin: 0 0 8px 0;'>風險評分 {risk_score:.0f}/100（{risk_level}）</h3>
                <p style='margin: 4px 0;'>融合分析置信度：{fusion_confidence:.0f}%</p>
                <p style='margin: 4px 0;'>主要欺騙模式：{primary_patterns_text}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                guidance_level, guidance_text, guidance_color = create_risk_guidance_section(risk_score)
                st.markdown(f"""
                <div style='background-color: {guidance_color}; padding: 16px; border-radius: 8px; height: 100%;'>
                <h4 style='color: white; margin: 0 0 8px 0;'>{guidance_level}</h4>
                <pre style='color: white; white-space: pre-wrap; margin: 0; font-size: 0.85em;'>{guidance_text}</pre>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 1.5em 0;'></div>", unsafe_allow_html=True)

            # 📊 關鍵指標卡片（取代語速/停頓/音高/音量的原始數字表格）
            st.markdown("## 📊 關鍵指標")
            status_color = {"normal": "green", "warning": "orange", "alert": "red"}
            confidence = max(probabilities) * 100
            conf_status, conf_icon = ("normal", "🟢") if confidence >= 60 else ("warning", "🟡") if confidence >= 40 else ("alert", "🔴")

            cards = [
                (conf_icon, "情緒", translated_emotion, f"信心度 {confidence:.0f}%", conf_status),
                (feature_status["speech_rate"]["icon"], feature_status["speech_rate"]["label"], "", feature_status["speech_rate"]["message"], feature_status["speech_rate"]["status"]),
                (feature_status["pause"]["icon"], feature_status["pause"]["label"], "", feature_status["pause"]["message"], feature_status["pause"]["status"]),
                (feature_status["pitch"]["icon"], feature_status["pitch"]["label"], "", feature_status["pitch"]["message"], feature_status["pitch"]["status"]),
                (feature_status["volume"]["icon"], feature_status["volume"]["label"], "", feature_status["volume"]["message"], feature_status["volume"]["status"]),
            ]
            card_cols = st.columns(5)
            for card_col, (icon, label, value, message, status) in zip(card_cols, cards):
                color = status_color.get(status, "gray")
                value_html = f"<div style='margin-top: 4px; font-weight: 600;'>{value}</div>" if value else ""
                with card_col:
                    st.markdown(f"""
                    <div style='background-color: {color}18; border: 1px solid {color}55; padding: 12px; border-radius: 8px; text-align: center; height: 100%;'>
                    <div style='font-size: 1.3em;'>{icon} {label}</div>
                    {value_html}
                    <div style='margin-top: 6px; font-size: 0.85em; color: {color};'>{message}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 1.5em 0;'></div>", unsafe_allow_html=True)

            # 視覺化欺騙模式圖（縮小並置中，避免佔滿整個版面寬度）
            fig, _ = visualize_fraud_patterns(integrated_fraud_report)
            chart_left, chart_mid, chart_right = st.columns([1, 1, 1])
            with chart_mid:
                st.pyplot(fig, use_container_width=True)

            # 🔍 多模態欺騙分析詳情（原始評分表，供風控/稽核查閱）
            with st.expander("🔍 多模態欺騙分析詳情", expanded=False):
                st.markdown("### 音訊分析結果")
                audio_df = pd.DataFrame({
                   "模式": list(audio_fraud_report["all_pattern_scores"].keys()),
                   "評分": [f"{score*100:.1f}%" for score in audio_fraud_report["all_pattern_scores"].values()]
                })
                st.table(audio_df)

                st.markdown("### 融合分析")
                integrated_df = pd.DataFrame({
                    "模式": list(integrated_fraud_report["all_pattern_scores"].keys()),
                    "評分": [f"{score*100:.1f}%" for score in integrated_fraud_report["all_pattern_scores"].values()]
                })
                st.table(integrated_df)

            # 🔬 完整技術數據（原始特徵數字與圖表，預設收合，供風控/稽核/事後複查用）
            with st.expander("🔬 完整技術數據", expanded=False):
                st.markdown("#### 逐字稿")
                st.write(transcript if transcription_ok else "（轉錄失敗，無逐字稿）")

                st.markdown("#### 語速與停頓分析")
                speech_pattern_df = pd.DataFrame({
                    "特徵": [
                        "語音長度 (秒)",
                        "停頓次數",
                        "每秒停頓次數",
                        "平均停頓長度 (秒)",
                        "停頓長度標準差",
                        "停頓時間佔比",
                        "語速變化標準差",
                        "語速範圍",
                        "突然語速變化次數"
                    ],
                    "數值": [
                        f"{float(advanced_features['speech_rate']['speech_duration']):.2f}",
                        f"{int(advanced_features['speech_rate']['pause_count'])}",
                        f"{float(advanced_features['speech_rate']['pause_rate']):.2f}",
                        f"{float(advanced_features['speech_rate']['pause_mean_duration']):.2f}",
                        f"{float(advanced_features['speech_rate']['pause_std_duration']):.2f}",
                        f"{float(advanced_features['speech_rate']['pause_ratio']):.2f}%",
                        f"{float(advanced_features['speech_rate']['speech_rate_variation']):.2f}",
                        f"{float(advanced_features['speech_rate']['speech_rate_range']):.2f}",
                        f"{int(advanced_features['speech_rate']['sudden_speed_changes'])}"
                    ]
                })
                st.table(speech_pattern_df)

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    pause_pattern_chart = create_pause_pattern_chart(advanced_features['speech_rate'])
                    st.plotly_chart(pause_pattern_chart, use_container_width=True)
                with chart_col2:
                    speech_rate_chart = create_speech_rate_chart(advanced_features['speech_rate'])
                    st.plotly_chart(speech_rate_chart, use_container_width=True)

                st.markdown("#### 音高特徵 (CREPE 算法)")
                pitch_df = pd.DataFrame({
                    "特徵": [
                        "平均音高 (Hz)",
                        "音高標準差",
                        "音高範圍",
                        "音高變化率",
                        "音高不穩定性",
                        "音高趨勢",
                        "音高置信度",
                        "有聲段比例"
                    ],
                    "數值": [
                        f"{float(advanced_features['pitch']['mean_pitch']):.2f}",
                        f"{float(advanced_features['pitch']['std_pitch']):.2f}",
                        f"{float(advanced_features['pitch']['pitch_range']):.2f}",
                        f"{float(advanced_features['pitch']['pitch_change_rate']):.2f}",
                        f"{float(advanced_features['pitch']['pitch_instability']):.2f}",
                        f"{float(advanced_features['pitch']['pitch_trend']):.2f}",
                        f"{float(advanced_features['pitch']['pitch_confidence']):.2f}",
                        f"{float(advanced_features['pitch']['voiced_ratio']):.2f}"
                    ]
                })
                st.table(pitch_df)

                st.markdown("#### 音量特徵")
                volume_df = pd.DataFrame({
                    "特徵": ["平均音量", "音量標準差", "音量範圍", "音量變化次數"],
                    "數值": [
                        f"{float(advanced_features['volume']['mean_volume']):.4f}",
                        f"{float(advanced_features['volume']['std_volume']):.4f}",
                        f"{float(advanced_features['volume']['volume_range']):.4f}",
                        f"{int(advanced_features['volume']['volume_changes'])}"
                    ]
                })
                st.table(volume_df)

                st.markdown("#### 情緒概率分佈")
                emotion_df = pd.DataFrame({
                    "情緒": [translate_emotion(k) for k in emotion_parameters.keys()],
                    "概率值": list(emotion_parameters.values()),
                    "概率": [f"{v:.4f}" for v in emotion_parameters.values()]
                })
                emotion_df = emotion_df.sort_values(by='概率值', ascending=False).reset_index(drop=True).drop('概率值', axis=1)
                st.table(emotion_df)

        except Exception as e:
            st.error(f"分析過程中發生錯誤: {str(e)}")
            st.error(f"錯誤類型: {type(e)}")
            st.error(f"錯誤發生位置: {e.__traceback__.tb_frame.f_code.co_filename}, 行 {e.__traceback__.tb_lineno}")
        
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if __name__ == "__main__":
    main()