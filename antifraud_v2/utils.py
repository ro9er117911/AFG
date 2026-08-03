import json
import os
from datetime import date


def generate_call_id(counter_file="call_counter.json"):
    """
    依今日日期產生通話 ID，流水號依日期持久化於本地 JSON 檔案

    Parameters:
    counter_file (str): 流水號計數檔案路徑

    Returns:
    tuple: (call_id, call_date_str)，例如 ("CALL-2025-0226-004", "2025-02-26")
    """
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    count = 1
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today_str:
                count = int(data.get("count", 0)) + 1
        except (json.JSONDecodeError, ValueError, OSError):
            count = 1

    with open(counter_file, "w", encoding="utf-8") as f:
        json.dump({"date": today_str, "count": count}, f)

    call_id = f"CALL-{today.strftime('%Y')}-{today.strftime('%m%d')}-{count:03d}"
    return call_id, today_str


def _status_tier(value, warn, alert):
    """數值 -> (status, icon) 三段式判讀：正常 / 略高 / 異常"""
    if value >= alert:
        return "alert", "🔴"
    if value >= warn:
        return "warning", "🟡"
    return "normal", "🟢"


def evaluate_feature_status(advanced_features):
    """
    將語速/停頓/音高/音量的原始特徵，轉換成第一線人員可以直接判讀的「正常/略高/異常」狀態卡片內容，
    取代直接攤開一大堆原始數字。

    門檻盡量沿用 fraud_detection.py 的 FraudPatternClassifier 裡各欺騙模式已經在用的參考值
    （停頓時間佔比 15%/30%、音高變化率 0.5、突然語速變化次數 3~5 次/分鐘），避免另外發明一套標準。
    音量例外：fraud_detection.py 的「音量標準差」門檻（6~7）跟實際 RMS 數值量級（std_volume 常態是
    0 點多）明顯不同尺度，直接沿用會永遠判定為正常、卡片失去意義，因此音量改用「變異係數」
    (std/mean，無量綱) 來判斷穩定度。

    Parameters:
    advanced_features (dict): analyze_speech_features() 的輸出

    Returns:
    dict: {"speech_rate": {...}, "pause": {...}, "pitch": {...}, "volume": {...}}
          每個子項目為 {"icon", "status", "label", "message"}
    """
    speech_rate = advanced_features.get('speech_rate', {})
    pitch = advanced_features.get('pitch', {})
    volume = advanced_features.get('volume', {})

    duration_min = max(speech_rate.get('speech_duration', 0) / 60, 1e-6)
    sudden_per_min = speech_rate.get('sudden_speed_changes', 0) / duration_min
    sr_status, sr_icon = _status_tier(sudden_per_min, warn=3, alert=5)

    pause_ratio = speech_rate.get('pause_ratio', 0)
    pa_status, pa_icon = _status_tier(pause_ratio, warn=15, alert=30)

    pitch_change_rate = pitch.get('pitch_change_rate', 0)
    pi_status, pi_icon = _status_tier(pitch_change_rate, warn=0.35, alert=0.5)

    mean_volume = volume.get('mean_volume', 0)
    std_volume = volume.get('std_volume', 0)
    volume_cv = std_volume / mean_volume if mean_volume > 0 else 0
    vo_status, vo_icon = _status_tier(volume_cv, warn=0.4, alert=0.7)

    return {
        "speech_rate": {
            "icon": sr_icon, "status": sr_status, "label": "語速",
            "message": "穩定" if sr_status == "normal" else f"每分鐘出現 {sudden_per_min:.1f} 次明顯變化",
        },
        "pause": {
            "icon": pa_icon, "status": pa_status, "label": "停頓",
            "message": f"佔比 {pause_ratio:.0f}%" + ("，正常" if pa_status == "normal" else "，偏高"),
        },
        "pitch": {
            "icon": pi_icon, "status": pi_status, "label": "音高",
            "message": "穩定，無異常波動" if pi_status == "normal" else "波動明顯",
        },
        "volume": {
            "icon": vo_icon, "status": vo_status, "label": "音量",
            "message": "穩定" if vo_status == "normal" else "起伏較大",
        },
    }


def translate_emotion(english_emotion):
    """
    將英文情緒名稱翻譯為中文
    
    Parameters:
    english_emotion (str): 英文情緒名稱
    
    Returns:
    str: 中文情緒名稱
    """
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

def create_risk_guidance_section(fraud_probability):
    """
    創建風險導流建議區塊
    
    Parameters:
    fraud_probability (float): 詐欺可能性百分比
    
    Returns:
    tuple: (風險等級, 建議處理方式, 背景顏色)
    """
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