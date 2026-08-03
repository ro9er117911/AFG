# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np

# def create_audio_waveform(audio_file):
#     """
#     創建音訊波形圖
    
#     Parameters:
#     audio_file: 音訊檔案路徑或上傳的檔案物件
    
#     Returns:
#     matplotlib.figure.Figure: 波形圖圖表物件
#     """
#     # 加载音频文件
#     y, sr = librosa.load(audio_file)
    
#     # 创建图形
#     fig, ax = plt.subplots(figsize=(10, 1))
#     librosa.display.waveshow(y, sr=sr, ax=ax)
#     ax.set_title('Audio Waveform')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Amplitude')
    
#     # 设置背景色为透明
#     fig.patch.set_alpha(0)
#     ax.patch.set_alpha(0)
    
#     # 设置刻度标签颜色为白色
#     ax.tick_params(colors='white')
#     ax.xaxis.label.set_color('white')
#     ax.yaxis.label.set_color('white')
#     ax.title.set_color('white')
    
#     return fig

# def create_speech_features_plots(audio_features):
#     """
#     創建語音特徵視覺化圖表
    
#     Parameters:
#     audio_features (dict): 語音特徵數據
    
#     Returns:
#     tuple: (語速儀表圖, 音高柱狀圖, 音量柱狀圖)
#     """
#     # Speech Rate Gauge
#     gauge_fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=audio_features['speech_rate']['syllables_per_second'],
#         gauge={
#             'axis': {'range': [0, 8]},
#             'steps': [
#                 {'range': [0, 3], 'color': "lightgray"},
#                 {'range': [3, 6], 'color': "gray"},
#                 {'range': [6, 8], 'color': "darkgray"}
#             ]
#         },
#         title={'text': "語速 (音節/秒)"}
#     ))
    
#     gauge_fig.update_layout(
#         height=300,
#         margin=dict(t=50, b=20),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(color="white")
#     )
    
#     # Pitch Changes Bar Chart
#     pitch_fig = go.Figure()
#     pitch_fig.add_trace(
#         go.Bar(
#             x=['平均音高', '音高標準差', '音高範圍'],
#             y=[audio_features['pitch']['mean_pitch'],
#                audio_features['pitch']['std_pitch'],
#                audio_features['pitch']['pitch_range']],
#             marker_color=['blue', 'red', 'green']
#         )
#     )
    
#     pitch_fig.update_layout(
#         title="音高變化",
#         height=300,
#         margin=dict(t=50, b=20),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(color="white")
#     )
    
#     # Volume Changes Bar Chart
#     volume_fig = go.Figure()
#     volume_fig.add_trace(
#         go.Bar(
#             x=['平均音量', '音量標準差', '音量範圍'],
#             y=[audio_features['volume']['mean_volume'],
#                audio_features['volume']['std_volume'],
#                audio_features['volume']['volume_range']],
#             marker_color=['blue', 'red', 'green']
#         )
#     )
    
#     volume_fig.update_layout(
#         title="音量變化",
#         height=300,
#         margin=dict(t=50, b=20),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(color="white")
#     )
    
#     return gauge_fig, pitch_fig, volume_fig

# def create_emotion_radar_chart(emotion_parameters):
#     """
#     創建情緒雷達圖
    
#     Parameters:
#     emotion_parameters (dict): 各情緒的機率參數
    
#     Returns:
#     plotly.graph_objects.Figure: 雷達圖物件
#     """
#     # 排序情緒
#     sorted_emotions = sorted(emotion_parameters.items(), key=lambda x: x[1], reverse=True)
#     emotions, values = zip(*sorted_emotions)

#     # 使用對數尺度調整值，同時設置最小可見值
#     log_values = np.log10([max(v, 0.001) for v in values])  # 設置最小值為0.001
#     scaled_values = (log_values - min(log_values)) / (max(log_values) - min(log_values))

#     fig = go.Figure()
    
#     # 添加雷達圖
#     fig.add_trace(go.Scatterpolar(
#         r=scaled_values,
#         theta=emotions,
#         fill='toself',
#         line=dict(color='rgb(31, 119, 180)', width=2),
#         fillcolor='rgba(31, 119, 180, 0.3)',
#     ))
    
#     # 更新雷達圖布局
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 tickformat=".2f",
#                 tickvals=[0, 0.25, 0.5, 0.75, 1],
#                 ticktext=["0.001", "0.01", "0.1", "0.5", "1"],
#                 tickfont=dict(size=12, color="white"),
#             ),
#             angularaxis=dict(
#                 tickfont=dict(size=14, color="white"),
#             ),
#             bgcolor="rgba(0, 0, 0, 0)"
#         ),
#         showlegend=False,
#         title=dict(
#             text="情緒雷達圖",
#             font=dict(size=20, color="white"),
#             x=0
#         ),
#         height=350,
#         width=350,
#         margin=dict(t=30,b=30),
#         paper_bgcolor="rgba(0, 0, 0, 0)",
#         plot_bgcolor="rgba(0, 0, 0, 0)",
#         font=dict(color="white")
#     )
    
#     return fig

# def create_xyz_chart(emotion):
#     """
#     創建情緒三維座標圖
    
#     Parameters:
#     emotion (str): 情緒名稱
    
#     Returns:
#     plotly.graph_objects.Figure: 3D散點圖物件
#     """
#     emotion_xyz = {
#         '憤怒': {'x': -1.5, 'y': 1.5, 'z': 1.0},
#         '厭惡': {'x': -1.2, 'y': -0.5, 'z': 0.8},
#         '恐懼': {'x': -1.3, 'y': 1.0, 'z': 0.9},
#         '快樂': {'x': 1.5, 'y': 1.0, 'z': 1.0},
#         '中立': {'x': 0, 'y': 0, 'z': 0.5},
#         '無聊': {'x': -0.5, 'y': -1.5, 'z': 0.4},
#         '悲傷': {'x': -1.0, 'y': -1.0, 'z': 0.8}
#     }
    
#     xyz = emotion_xyz[emotion]
    
#     fig = go.Figure(data=[go.Scatter3d(
#         x=[xyz['x']],
#         y=[xyz['y']],
#         z=[xyz['z']],
#         mode='markers',
#         marker=dict(
#             size=10,
#             color='red',
#             symbol='circle'
#         ),
#         text=[emotion],
#         hoverinfo='text'
#     )])
    
#     fig.update_layout(
#         title=dict(text="情緒三軸圖", font=dict(size=20, color="white"), x=0),
#         height=350,
#         width=350,
#         scene = dict(
#             xaxis = dict(range=[-2,2], title='X'),
#             yaxis = dict(range=[-2,2], title='Y'),
#             zaxis = dict(range=[0,1.5], title='Z'),
#             aspectmode='cube'
#         ),
#         margin=dict(r=20, l=10, b=30, t=30),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(color="white")
#     )
    
#     return fig

# def create_fraud_probability_gauge(fraud_probability):
#     """
#     創建詐欺可能性儀表圖
    
#     Parameters:
#     fraud_probability (float): 詐欺可能性百分比
    
#     Returns:
#     plotly.graph_objects.Figure: 儀表圖物件
#     """
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number", 
#         value = fraud_probability,
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         gauge = {
#             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
#             'bar': {'color': "white", 'thickness': 0.5},
#             'bgcolor': "rgba(0,0,0,0)",
#             'borderwidth': 2,
#             'bordercolor': "white",
#             'steps': [
#                 {'range': [0, 20], 'color': 'rgba(0,100,0,0.7)'},
#                 {'range': [20, 40], 'color': 'rgba(50,100,0,0.7)'},
#                 {'range': [40, 60], 'color': 'rgba(100,100,0,0.7)'},
#                 {'range': [60, 80], 'color': 'rgba(100,50,0,0.7)'},
#                 {'range': [80, 100], 'color': 'rgba(100,0,0,0.7)'}],
#             }))

#     fig.update_layout(
#         title=dict(text="詐欺可能性", font=dict(size=20, color="white"), x=0),
#         height=350,
#         width=350,
#         margin=dict(t=30,b=30),
#         paper_bgcolor = "rgba(0,0,0,0)",
#         plot_bgcolor = "rgba(0,0,0,0)",
#         font = {'color': "white", 'family': "Arial"},
#         coloraxis_colorbar=dict(tickfont=dict(color="white"))
#     )

#     return fig

# def create_mock_speaker_display():
#     """
#     創建模擬的說話者時間軸顯示
    
#     Returns:
#     plotly.graph_objects.Figure: 時間軸圖表物件
#     """
#     # 模擬數據
#     times = np.linspace(0, 25, 100)  # 25秒的對話
#     speaker1_data = [1 if (5 < t < 8) or (12 < t < 15) or (20 < t < 23) else 0 for t in times]
#     speaker2_data = [1 if (2 < t < 5) or (9 < t < 12) or (16 < t < 19) else 0 for t in times]

#     # 創建圖表
#     fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7])

#     # 添加說話者時間軸
#     fig.add_trace(
#         go.Scatter(
#             x=times, 
#             y=speaker1_data,
#             name="客服人員",
#             fill='tozeroy',
#             line=dict(color='rgba(100, 149, 237, 0.8)'),
#         ),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Scatter(
#             x=times, 
#             y=[-v for v in speaker2_data],  # 反轉值以在下方顯示
#             name="客戶",
#             fill='tozeroy',
#             line=dict(color='rgba(144, 238, 144, 0.8)'),
#         ),
#         row=1, col=1
#     )

#     # 更新布局
#     fig.update_layout(
#         height=200,
#         margin=dict(l=0, r=0, t=30, b=0),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         showlegend=True,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1,
#             font=dict(color="white")
#         ),
#         font=dict(color="white")
#     )

#     # 更新軸線設置
#     fig.update_xaxes(showgrid=False, showticklabels=False)
#     fig.update_yaxes(showgrid=False, showticklabels=False)

#     return fig

import matplotlib.pyplot as plt
import librosa
import librosa.display

def create_audio_waveform(audio_file):
    """
    創建音訊波形圖
    
    Parameters:
    audio_file: 音訊檔案路徑或上傳的檔案物件
    
    Returns:
    matplotlib.figure.Figure: 波形圖圖表物件
    """
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