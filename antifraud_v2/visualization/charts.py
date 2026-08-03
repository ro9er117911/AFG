import plotly.graph_objects as go
import numpy as np

# 對應情緒翻譯
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

#情緒雷達圖
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
            font=dict(size=24, color="white"),
            x=0
        ),
        height=400,
        width=400,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white")
    )
    
    return fig

#情緒三軸圖
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
        title=dict(text="情緒三軸圖", font=dict(size=24, color="white"), x=0),
        height=400,
        width=400,
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
        margin=dict(r=20, l=10, b=10, t=40)
    )
    
    return fig

#風險圖
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
        title=dict(text="詐欺可能性", font=dict(size=24, color="white"), x=0),
        height=400,
        width=400,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Arial"},
        coloraxis_colorbar=dict(tickfont=dict(color="white"))
    )

    return fig