def analyze_pitch_for_fraud_detection(audio_file_path, model_capacity="full"):
    """
    使用 CREPE 算法分析音頻的音高特徵，用於詐欺檢測
    
    參數:
    audio_file_path (str): 音頻文件路徑
    model_capacity (str): CREPE 模型容量，可選 "tiny", "small", "medium", "large", "full"
    
    返回:
    dict: 包含各種音高特徵的字典
    """
    import librosa
    import numpy as np
    import crepe
    from scipy import stats
    
    # 加載音頻
    y, sr = librosa.load(audio_file_path, sr=None)
    
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

def visualize_pitch_fraud_indicators(pitch_features):
    """
    將音高特徵視覺化為詐欺指標
    
    參數:
    pitch_features (dict): analyze_pitch_for_fraud_detection 函數返回的特徵
    
    返回:
    plotly.graph_objects.Figure: 詐欺指標儀表板
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 創建包含4個子圖的儀表板
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("音高不穩定性", "音高變化率", "音高標準差", "音高趨勢")
    )
    
    # 1. 音高不穩定性指標 (越高越可能是欺騙)
    # 將數值標準化到0-100範圍，以便於解釋
    normalized_instability = min(100, pitch_features['pitch_instability'] * 10)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=normalized_instability,
            gauge={"axis": {"range": [0, 100]},
                  "steps": [
                      {"range": [0, 30], "color": "green"},
                      {"range": [30, 70], "color": "yellow"},
                      {"range": [70, 100], "color": "red"}
                  ],
                  "threshold": {
                      "line": {"color": "red", "width": 4},
                      "thickness": 0.75,
                      "value": 70
                  }},
            domain={"row": 0, "column": 0}
        )
    )
    
    # 2. 音高變化率指標
    normalized_change_rate = min(100, pitch_features['pitch_change_rate'] * 200)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=normalized_change_rate,
            gauge={"axis": {"range": [0, 100]},
                  "steps": [
                      {"range": [0, 30], "color": "green"},
                      {"range": [30, 70], "color": "yellow"},
                      {"range": [70, 100], "color": "red"}
                  ],
                  "threshold": {
                      "line": {"color": "red", "width": 4},
                      "thickness": 0.75,
                      "value": 70
                  }},
            domain={"row": 0, "column": 1}
        )
    )
    
    # 3. 音高標準差指標
    # 研究表明，說謊時語音的音高變化通常更大
    normalized_std = min(100, (pitch_features['std_pitch'] / pitch_features['mean_pitch']) * 500)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=normalized_std,
            gauge={"axis": {"range": [0, 100]},
                  "steps": [
                      {"range": [0, 30], "color": "green"},
                      {"range": [30, 70], "color": "yellow"},
                      {"range": [70, 100], "color": "red"}
                  ],
                  "threshold": {
                      "line": {"color": "red", "width": 4},
                      "thickness": 0.75,
                      "value": 70
                  }},
            domain={"row": 1, "column": 0}
        )
    )
    
    # 4. 音高趨勢指標 (絕對值，趨勢顯著性)
    # 強烈的上升或下降趨勢可能表明壓力或情緒變化
    normalized_trend = min(100, abs(pitch_features['pitch_trend']) * 5)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=normalized_trend,
            gauge={"axis": {"range": [0, 100]},
                  "steps": [
                      {"range": [0, 30], "color": "green"},
                      {"range": [30, 70], "color": "yellow"},
                      {"range": [70, 100], "color": "red"}
                  ],
                  "threshold": {
                      "line": {"color": "red", "width": 4},
                      "thickness": 0.75,
                      "value": 70
                  }},
            domain={"row": 1, "column": 1}
        )
    )
    
    # 更新布局
    fig.update_layout(
        height=600,
        width=800,
        title_text="詐欺語音指標分析",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    return fig

# 在app.py中整合此函數
def integrate_crepe_fraud_detection(main_function):
    """
    修改main函數，添加CREPE音高分析和詐欺檢測功能
    """
    # 在import區域添加
    # import crepe
    
    # 在main函數中適當位置添加
    """
    # 使用CREPE進行音高分析
    with st.spinner('使用CREPE分析音高特徵...'):
        pitch_features = analyze_pitch_for_fraud_detection(temp_file_path)
        st.write("### 音高特徵分析")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("平均音高 (Hz)", f"{pitch_features['mean_pitch']:.2f}")
            st.metric("音高標準差", f"{pitch_features['std_pitch']:.2f}")
        
        with col2:
            st.metric("有效樣本數", f"{pitch_features['valid_samples']}")
            st.metric("有聲段比例", f"{pitch_features['voiced_ratio']:.2%}")
            
        # 視覺化詐欺指標
        fraud_indicators_fig = visualize_pitch_fraud_indicators(pitch_features)
        st.plotly_chart(fraud_indicators_fig, use_container_width=True)
    """