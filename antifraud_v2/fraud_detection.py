import numpy as np
import os
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# 預設英文字型不含中文字形，圖表上的中文標籤/標題會變成空白方框，改用系統內建的中文字型
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'PingFang HK', 'Songti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class FraudPatternClassifier:
    """
    欺騙模式分類器，基於音訊特徵和情緒預測來分類不同類型的欺騙模式
    專為臺灣法人說明會環境設計
    """
    def __init__(self, config_path=None):
        """
        初始化分類器
        
        Parameters:
        config_path (str, optional): 閾值配置檔案路徑，若不指定則使用默認閾值
        """
        self.thresholds = self._load_thresholds(config_path)
        self.weights = self._init_weights()
        self.baseline = None  # 用於存儲基線發言特徵
        self.last_results = {}  # 存儲最後的分析結果
        
    def _load_thresholds(self, config_path):
        """
        載入閾值配置，如果提供了配置檔案則使用，否則使用默認值
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"載入配置檔案時出錯：{e}，使用默認閾值代替")
        
        # 基於研究文獻的默認閾值設定
        return {
            "攻擊性謊言": {
                "憤怒": 0.4,  # Pérez-Rosas et al. (2015) 研究支持
                "音量標準差": 6.0,  # 提高閾值，避免過度敏感
                "中立情緒下降": 0.25,  # 略微降低閾值，更符合實際觀察
                "語速變化標準差": 0.3,  # DePaulo et al. (2003) 研究支持
                "突然語速變化次數": 3,  # DePaulo et al. (2003) 研究支持
                "音高變化率": 0.5,  # Levitan et al. (2015) 研究支持
                "音高不穩定性": 0.4,  # Levitan et al. (2015) 研究支持
                "頻率顫抖": 1.04  # Hirschberg et al. (2005) 研究支持，從1.1調整為1.04
            },
            "矛盾衝突": {
                "停頓增加百分比": 35,  # Benus et al. (2006) 研究支持，略微提高
                "停頓時間標準差": 0.4,  # 保持原值
                "有聲段比例": 0.65,  # Hirschberg et al. (2005) 研究支持
                "音高置信度": 0.7,  # 保持原值
                "快樂": 0.4,  # 保持原值
                "恐懼": 0.25,  # 保持原值
                "音量顫抖": 3.81  # Graciarena et al. (2006) 研究支持，從5.5調整為3.81
            },
            "明確否認": {
                "音高增加百分比": 15,  # Burgoon et al. (2016) 研究支持
                "音高範圍增加": 30,  # Burgoon et al. (2016) 研究支持
                "語速增加百分比": 20,  # 保持原值
                "停頓時間佔比": 15,  # 保持原值
                "中立": 0.8,  # Pérez-Rosas et al. (2015) 研究支持
                "憤怒": 0.35,  # 保持原值
                "顫抖強度": 3.0  # 保持原值
            },
            "尷尬掩蓋": {
                "頻率顫抖": 1.3,  # 保持原值
                "音量顫抖": 5.0,  # 根據Enos et al. (2007)的觀察調整
                "恐懼": 0.3,  # 保持原值
                "厭惡": 0.25,  # 保持原值
                "平均停頓長度": 0.7,  # 保持原值
                "每秒停頓次數": 1.0,  # 保持原值
                "諧噪比": 20.0  # Harnsberger et al. (2009) 研究支持，從17.0調整為20.0
            },
            "警覺避談": {
                "語速增加百分比": 25,  # 保持原值
                "語速範圍": 0.5,  # 保持原值
                "主題轉換頻率": 2,  # Frank & Ekman (2004) 研究支持
                "中立情緒": 0.75,  # 保持原值
                "無聊": 0.4,  # 保持原值
                "音量下降百分比": 15  # 保持原值
            },
            "猶豫不決": {
                "停頓次數": 12,  # 保持原值
                "停頓時間佔比": 30,  # Vrij et al. (2008) 研究支持
                "停頓時間標準差": 0.45,  # 保持原值
                "諧噪比": 20.0,  # Harnsberger et al. (2009) 研究支持，調整
                "有聲段比例": 0.6,  # 保持原值
                "顫抖頻率下限": 4.0,  # 保持原值
                "顫抖頻率上限": 7.0,  # 保持原值
                "顫抖強度": 3.0,  # 略微調低閾值，避免過度敏感
                "恐懼加悲傷": 0.45  # 保持原值
            },
            "異常興奮": {
                "快樂": 0.65,  # 保持原值
                "中立": 0.25,  # 保持原值
                "語速增加百分比": 25,  # 保持原值
                "突然語速變化次數": 5,  # 保持原值
                "音量增加百分比": 20,  # 保持原值
                "音量標準差": 7.0,  # 保持原值
                "音高範圍增加": 40,  # 保持原值，符合Burgoon et al. (2016)觀察
                "音高變化率": 0.55  # 保持原值
            },
            "邏輯漏洞": {
                "語速變化標準差": 0.4,  # DePaulo et al. (2003) 研究支持
                "語速不一致閾值": 25,  # 保持原值
                "音高不穩定性": 0.5,  # Levitan et al. (2015) 研究支持
                "情緒轉換頻率": 2,  # Frank & Ekman (2004) 研究支持
                "關鍵停頓長度": 0.8,  # 保持原值
                "停頓分布標準差": 0.5  # 保持原值
            }
        }
        
    def _init_weights(self):
        """
        初始化各欺騙模式的特徵權重，基於相關研究調整
        """
        return {
            "攻擊性謊言": {
                "憤怒": 2.5,  # 提高權重，根據Pérez-Rosas et al. (2015)
                "音量標準差": 1.5,  # 維持不變
                "中立情緒下降": 1.2,  # 略微增加
                "語速變化標準差": 1.3,  # 根據DePaulo et al. (2003)增加
                "突然語速變化次數": 1.2,  # 維持不變
                "音高變化率": 1.7,  # 根據Levitan et al. (2015)增加
                "音高不穩定性": 1.4,  # 根據Levitan et al. (2015)增加
                "頻率顫抖": 2.0  # 根據Hirschberg et al. (2005)增加
            },
            "矛盾衝突": {
                "停頓增加": 1.5,  # 根據Benus et al. (2006)增加
                "停頓時間標準差": 1.2,  # 略微增加
                "有聲段比例": 1.8,  # 根據Hirschberg et al. (2005)增加
                "音高置信度": 1.2,  # 維持不變
                "情緒矛盾": 2.2,  # 增加，根據研究表明情緒矛盾是重要指標
                "音量顫抖": 1.7  # 根據Graciarena et al. (2006)增加
            },
            "明確否認": {
                "音高增加": 1.8,  # 根據Burgoon et al. (2016)增加
                "音高範圍增加": 1.5,  # 增加
                "語速增加": 1.2,  # 略微增加
                "停頓時間佔比": 1.3,  # 略微降低，因為停頓在否認中不如其他模式重要
                "中立": 2.0,  # 根據Pérez-Rosas et al. (2015)增加
                "憤怒": 1.2,  # 維持不變
                "顫抖強度": 1.5  # 維持不變
            },
            "尷尬掩蓋": {
                "頻率顫抖": 2.2,  # 增加，根據Wu et al. (2018)關於尷尬掩蓋的研究
                "音量顫抖": 1.7,  # 增加
                "恐懼": 2.0,  # 增加，恐懼在尷尬掩蓋中更為重要
                "厭惡": 1.4,  # 略微增加
                "平均停頓長度": 1.2,  # 略微增加
                "每秒停頓次數": 1.0,  # 維持不變
                "諧噪比": 1.8  # 根據Harnsberger et al. (2009)增加
            },
            "警覺避談": {
                "語速增加": 1.2,  # 略微增加
                "語速範圍": 1.3,  # 略微增加
                "主題轉換頻率": 1.8,  # 根據Frank & Ekman (2004)增加
                "中立情緒": 1.9,  # 增加，中立情緒在警覺避談中更為明顯
                "無聊": 1.4,  # 略微增加
                "音量下降": 1.6  # 略微增加
            },
            "猶豫不決": {
                "停頓次數": 1.8,  # 增加，根據Vrij et al. (2008)
                "停頓時間佔比": 2.0,  # 增加，這是猶豫不決的關鍵指標
                "停頓時間標準差": 1.4,  # 略微增加
                "諧噪比": 1.2,  # 略微增加
                "有聲段比例": 1.5,  # 維持不變
                "顫抖頻率範圍": 1.4,  # 略微增加
                "顫抖強度": 1.9,  # 增加
                "恐懼加悲傷": 1.7  # 增加
            },
            "異常興奮": {
                "快樂": 2.2,  # 增加，這是異常興奮的核心指標
                "中立": 1.2,  # 略微增加
                "語速增加": 1.5,  # 增加
                "突然語速變化次數": 1.7,  # 增加
                "音量增加": 1.5,  # 增加
                "音量標準差": 1.2,  # 略微增加
                "音高範圍增加": 1.4,  # 略微增加
                "音高變化率": 1.7  # 增加
            },
            "邏輯漏洞": {
                "語速變化標準差": 1.4,  # 略微增加
                "語速不一致": 1.8,  # 增加，這是邏輯漏洞的重要指標
                "音高不穩定性": 2.0,  # 增加，根據Levitan et al. (2015)
                "情緒轉換頻率": 1.7,  # 增加，根據Frank & Ekman (2004)
                "關鍵停頓長度": 1.2,  # 略微增加
                "停頓分布標準差": 1.4  # 略微增加
            }
        }
        
    def set_baseline(self, audio_features, emotion_probs, smoothing_factor=0.3):
        """
        設定基線特徵，用於根據說話者音高/語速調整閾值

        Parameters:
        audio_features (dict): 音訊特徵
        emotion_probs (dict): 情緒預測概率
        smoothing_factor (float): 平滑因子 (0-1)，保留參數以維持介面相容，目前未使用
            （classify_patterns() 每次都是全新的 FraudPatternClassifier 實例，self.baseline
            一定是 None，因此不會有「舊基線」可以平滑更新——這裡不假裝有跨通話的歷史基線）
        """
        # 提取當前特徵，僅用於 adjust_thresholds_based_on_baseline() 的音高/語速正規化，
        # 不用於「與基線比較差異」——這通電話自己的特徵不能拿來當自己的比較基準
        self.baseline = {
            "平均音高": audio_features['pitch']['mean_pitch'],
            "語速": audio_features['speech_rate'].get('local_speech_rates', [0])[0] if len(audio_features['speech_rate'].get('local_speech_rates', [])) > 0 else 0,
            "平均音量": audio_features['volume']['mean_volume']
        }

        # 根據基線特徵調整閾值
        self.thresholds = self.adjust_thresholds_based_on_baseline(self.baseline)

        
    def classify_patterns(self, audio_features, emotion_probs):
        """
        根據音訊特徵和情緒概率分類欺騙模式
        
        Parameters:
        audio_features (dict): 音訊特徵
        emotion_probs (dict): 情緒預測概率
        
        Returns:
        dict: 各欺騙模式的可能性評分 (0-1)
        """
        # 如果沒有基線，使用當前數據作為基線
        if not self.baseline:
            self.set_baseline(audio_features, emotion_probs)
        
        # 初始化結果
        pattern_scores = {}
        detailed_scores = {}
        
        # 評估各欺騙模式
        pattern_scores["攻擊性謊言"], detailed_scores["攻擊性謊言"] = self._evaluate_aggressive_lies(audio_features, emotion_probs)
        pattern_scores["矛盾衝突"], detailed_scores["矛盾衝突"] = self._evaluate_contradictions(audio_features, emotion_probs)
        pattern_scores["明確否認"], detailed_scores["明確否認"] = self._evaluate_explicit_denial(audio_features, emotion_probs)
        pattern_scores["尷尬掩蓋"], detailed_scores["尷尬掩蓋"] = self._evaluate_embarrassment_cover(audio_features, emotion_probs)
        pattern_scores["警覺避談"], detailed_scores["警覺避談"] = self._evaluate_alert_avoidance(audio_features, emotion_probs)
        pattern_scores["猶豫不決"], detailed_scores["猶豫不決"] = self._evaluate_hesitation(audio_features, emotion_probs)
        pattern_scores["異常興奮"], detailed_scores["異常興奮"] = self._evaluate_abnormal_excitement(audio_features, emotion_probs)
        pattern_scores["邏輯漏洞"], detailed_scores["邏輯漏洞"] = self._evaluate_logical_gaps(audio_features, emotion_probs)
        
        # 保存詳細結果
        self.last_results = {
            "pattern_scores": pattern_scores,
            "detailed_scores": detailed_scores,
            "audio_features": audio_features,
            "emotion_probs": emotion_probs
        }
        
        return pattern_scores, detailed_scores
    
    def adjust_thresholds_based_on_baseline(self, baseline_features):
        """
        根據說話者基線特徵調整閾值
        
        Parameters:
        baseline_features (dict): 說話者的基線音訊特徵
        
        Returns:
        dict: 調整後的閾值
        """
        # 獲取當前閾值
        thresholds = self.thresholds.copy()
        
        # 根據說話者特徵調整閾值
        # 例如，調整與音高相關的閾值
        if "平均音高" in baseline_features and baseline_features["平均音高"] > 0:
            # 若說話者基本音高較高，調整音高相關閾值
            pitch_factor = min(1.2, max(0.8, baseline_features["平均音高"] / 120.0))
            
            # 調整音高增加百分比閾值
            thresholds["明確否認"]["音高增加百分比"] *= pitch_factor
            thresholds["異常興奮"]["音高範圍增加"] *= pitch_factor
        
        # 類似地，調整語速相關閾值
        if "語速" in baseline_features and baseline_features["語速"] > 0:
            speed_factor = min(1.2, max(0.8, baseline_features["語速"] / 4.0))
            
            thresholds["明確否認"]["語速增加百分比"] *= speed_factor
            thresholds["警覺避談"]["語速增加百分比"] *= speed_factor
            thresholds["異常興奮"]["語速增加百分比"] *= speed_factor
        
        return thresholds

    def _evaluate_aggressive_lies(self, audio_features, emotion_probs):
        """評估是否存在攻擊性謊言模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查情緒指標
        anger_score = emotion_probs.get("anger", 0)
        if anger_score > self.thresholds["攻擊性謊言"]["憤怒"]:
            s = min((anger_score / self.thresholds["攻擊性謊言"]["憤怒"]) * self.weights["攻擊性謊言"]["憤怒"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["憤怒"]
            detailed_scores["憤怒"] = s / self.weights["攻擊性謊言"]["憤怒"]
        else:
            detailed_scores["憤怒"] = 0
        
        # 檢查音量標準差
        volume_std = audio_features.get("volume", {}).get("std_volume", 0)
        if volume_std > self.thresholds["攻擊性謊言"]["音量標準差"]:
            s = min((volume_std / self.thresholds["攻擊性謊言"]["音量標準差"]) * self.weights["攻擊性謊言"]["音量標準差"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["音量標準差"]
            detailed_scores["音量標準差"] = s / self.weights["攻擊性謊言"]["音量標準差"]
        else:
            detailed_scores["音量標準差"] = 0
            
        # 檢查中立情緒突然下降
        neutral_score = emotion_probs.get("neutral", 0)
        if neutral_score < self.thresholds["攻擊性謊言"]["中立情緒下降"]:
            s = min(((self.thresholds["攻擊性謊言"]["中立情緒下降"] - neutral_score) / self.thresholds["攻擊性謊言"]["中立情緒下降"]) * self.weights["攻擊性謊言"]["中立情緒下降"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["中立情緒下降"]
            detailed_scores["中立情緒下降"] = s / self.weights["攻擊性謊言"]["中立情緒下降"]
        else:
            detailed_scores["中立情緒下降"] = 0
        
        # 檢查語速變化標準差
        speech_rate_std = audio_features.get("speech_rate", {}).get("speech_rate_variation", 0)
        if speech_rate_std > self.thresholds["攻擊性謊言"]["語速變化標準差"]:
            s = min((speech_rate_std / self.thresholds["攻擊性謊言"]["語速變化標準差"]) * self.weights["攻擊性謊言"]["語速變化標準差"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["語速變化標準差"]
            detailed_scores["語速變化標準差"] = s / self.weights["攻擊性謊言"]["語速變化標準差"]
        else:
            detailed_scores["語速變化標準差"] = 0
        
        # 檢查突然語速變化次數
        sudden_changes = audio_features.get("speech_rate", {}).get("sudden_speed_changes", 0)
        speech_duration = audio_features.get("speech_rate", {}).get("speech_duration", 60)
        speech_minutes = max(speech_duration / 60, 1)
        sudden_changes_per_minute = sudden_changes / speech_minutes
        
        if sudden_changes_per_minute > self.thresholds["攻擊性謊言"]["突然語速變化次數"]:
            s = min((sudden_changes_per_minute / self.thresholds["攻擊性謊言"]["突然語速變化次數"]) * self.weights["攻擊性謊言"]["突然語速變化次數"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["突然語速變化次數"]
            detailed_scores["突然語速變化次數"] = s / self.weights["攻擊性謊言"]["突然語速變化次數"]
        else:
            detailed_scores["突然語速變化次數"] = 0
        
        # 檢查音高變化率
        pitch_change_rate = audio_features.get("pitch", {}).get("pitch_change_rate", 0)
        if pitch_change_rate > self.thresholds["攻擊性謊言"]["音高變化率"]:
            s = min((pitch_change_rate / self.thresholds["攻擊性謊言"]["音高變化率"]) * self.weights["攻擊性謊言"]["音高變化率"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["音高變化率"]
            detailed_scores["音高變化率"] = s / self.weights["攻擊性謊言"]["音高變化率"]
        else:
            detailed_scores["音高變化率"] = 0
        
        # 檢查音高不穩定性
        pitch_instability = audio_features.get("pitch", {}).get("pitch_instability", 0)
        if pitch_instability > self.thresholds["攻擊性謊言"]["音高不穩定性"]:
            s = min((pitch_instability / self.thresholds["攻擊性謊言"]["音高不穩定性"]) * self.weights["攻擊性謊言"]["音高不穩定性"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["音高不穩定性"]
            detailed_scores["音高不穩定性"] = s / self.weights["攻擊性謊言"]["音高不穩定性"]
        else:
            detailed_scores["音高不穩定性"] = 0
        
        # 檢查頻率顫抖
        jitter_local = audio_features.get("tremor", {}).get("jitter_local", 0)
        if jitter_local > self.thresholds["攻擊性謊言"]["頻率顫抖"]:
            s = min((jitter_local / self.thresholds["攻擊性謊言"]["頻率顫抖"]) * self.weights["攻擊性謊言"]["頻率顫抖"], 3.0)
            score += s
            max_score += self.weights["攻擊性謊言"]["頻率顫抖"]
            detailed_scores["頻率顫抖"] = s / self.weights["攻擊性謊言"]["頻率顫抖"]
        else:
            detailed_scores["頻率顫抖"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores  # 上限設為0.95，避免絕對肯定
    
    def _evaluate_contradictions(self, audio_features, emotion_probs):
        """評估是否存在矛盾衝突模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查停頓增加
        baseline_pause_rate = 0.8  # 假設基線值
        current_pause_rate = audio_features.get("speech_rate", {}).get("pause_rate", 0)
        pause_increase_percent = ((current_pause_rate - baseline_pause_rate) / baseline_pause_rate * 100) if baseline_pause_rate > 0 else 0
        
        if pause_increase_percent > self.thresholds["矛盾衝突"]["停頓增加百分比"]:
            s = min((pause_increase_percent / self.thresholds["矛盾衝突"]["停頓增加百分比"]) * self.weights["矛盾衝突"]["停頓增加"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["停頓增加"]
            detailed_scores["停頓增加"] = s / self.weights["矛盾衝突"]["停頓增加"]
        else:
            detailed_scores["停頓增加"] = 0
        
        # 檢查停頓時間標準差
        pause_std = audio_features.get("speech_rate", {}).get("pause_std_duration", 0)
        if pause_std > self.thresholds["矛盾衝突"]["停頓時間標準差"]:
            s = min((pause_std / self.thresholds["矛盾衝突"]["停頓時間標準差"]) * self.weights["矛盾衝突"]["停頓時間標準差"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["停頓時間標準差"]
            detailed_scores["停頓時間標準差"] = s / self.weights["矛盾衝突"]["停頓時間標準差"]
        else:
            detailed_scores["停頓時間標準差"] = 0
        
        # 檢查有聲段比例
        voiced_ratio = audio_features.get("pitch", {}).get("voiced_ratio", 1)
        if voiced_ratio < self.thresholds["矛盾衝突"]["有聲段比例"]:
            s = min(((self.thresholds["矛盾衝突"]["有聲段比例"] - voiced_ratio) / self.thresholds["矛盾衝突"]["有聲段比例"]) * self.weights["矛盾衝突"]["有聲段比例"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["有聲段比例"]
            detailed_scores["有聲段比例"] = s / self.weights["矛盾衝突"]["有聲段比例"]
        else:
            detailed_scores["有聲段比例"] = 0
        
        # 檢查音高置信度
        pitch_confidence = audio_features.get("pitch", {}).get("pitch_confidence", 1)
        if pitch_confidence < self.thresholds["矛盾衝突"]["音高置信度"]:
            s = min(((self.thresholds["矛盾衝突"]["音高置信度"] - pitch_confidence) / self.thresholds["矛盾衝突"]["音高置信度"]) * self.weights["矛盾衝突"]["音高置信度"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["音高置信度"]
            detailed_scores["音高置信度"] = s / self.weights["矛盾衝突"]["音高置信度"]
        else:
            detailed_scores["音高置信度"] = 0
        
        # 檢查情緒矛盾 (快樂 + 恐懼同時高)
        happy_score = emotion_probs.get("happy", 0)
        fear_score = emotion_probs.get("fear", 0)
        
        if happy_score > self.thresholds["矛盾衝突"]["快樂"] and fear_score > self.thresholds["矛盾衝突"]["恐懼"]:
            emotion_contradiction = min(happy_score, fear_score) * 2  # 取較小的情緒值並加權
            s = min(emotion_contradiction * self.weights["矛盾衝突"]["情緒矛盾"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["情緒矛盾"]
            detailed_scores["情緒矛盾"] = s / self.weights["矛盾衝突"]["情緒矛盾"]
        else:
            detailed_scores["情緒矛盾"] = 0
        
        # 檢查音量顫抖
        shimmer_local = audio_features.get("tremor", {}).get("shimmer_local", 0)
        if shimmer_local > self.thresholds["矛盾衝突"]["音量顫抖"]:
            s = min((shimmer_local / self.thresholds["矛盾衝突"]["音量顫抖"]) * self.weights["矛盾衝突"]["音量顫抖"], 3.0)
            score += s
            max_score += self.weights["矛盾衝突"]["音量顫抖"]
            detailed_scores["音量顫抖"] = s / self.weights["矛盾衝突"]["音量顫抖"]
        else:
            detailed_scores["音量顫抖"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores
    
        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    
    def _evaluate_explicit_denial(self, audio_features, emotion_probs):
        """評估是否存在明確否認模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 音高增加（相對基準）：self.baseline 是用「這通電話自己的特徵」設定的（見 set_baseline），
        # 對這裡用到的音高欄位而言 baseline 恆等於 current，差異永遠是 0%，無法真正偵測音高變化。
        # 明確跳過而非算出一個看似有意義、實則恆為 0 的假差異；不影響現有分數（該分支本來就從未觸發）。
        detailed_scores["音高增加"] = 0
        
        # 檢查音高範圍擴大
        pitch_range = audio_features.get("pitch", {}).get("pitch_range", 0)
        baseline_pitch_range = 100  # 假設基線值
        pitch_range_increase = ((pitch_range - baseline_pitch_range) / baseline_pitch_range * 100) if baseline_pitch_range > 0 else 0
        
        if pitch_range_increase > self.thresholds["明確否認"]["音高範圍增加"]:
            s = min((pitch_range_increase / self.thresholds["明確否認"]["音高範圍增加"]) * self.weights["明確否認"]["音高範圍增加"], 3.0)
            score += s
            max_score += self.weights["明確否認"]["音高範圍增加"]
            detailed_scores["音高範圍增加"] = s / self.weights["明確否認"]["音高範圍增加"]
        else:
            detailed_scores["音高範圍增加"] = 0
        
        # 檢查語速增加
        current_speech_rate = 0
        if len(audio_features.get("speech_rate", {}).get("local_speech_rates", [])) > 0:
            current_speech_rate = np.mean(audio_features["speech_rate"]["local_speech_rates"])
            
        baseline_speech_rate = self.baseline.get("語速", current_speech_rate)
        speech_rate_increase = ((current_speech_rate - baseline_speech_rate) / baseline_speech_rate * 100) if baseline_speech_rate > 0 else 0
        
        if speech_rate_increase > self.thresholds["明確否認"]["語速增加百分比"]:
            s = min((speech_rate_increase / self.thresholds["明確否認"]["語速增加百分比"]) * self.weights["明確否認"]["語速增加"], 3.0)
            score += s
            max_score += self.weights["明確否認"]["語速增加"]
            detailed_scores["語速增加"] = s / self.weights["明確否認"]["語速增加"]
        else:
            detailed_scores["語速增加"] = 0
        
        # 檢查停頓時間佔比低
        pause_ratio = audio_features.get("speech_rate", {}).get("pause_ratio", 0)
        if pause_ratio < self.thresholds["明確否認"]["停頓時間佔比"]:
            s = min(((self.thresholds["明確否認"]["停頓時間佔比"] - pause_ratio) / self.thresholds["明確否認"]["停頓時間佔比"]) * self.weights["明確否認"]["停頓時間佔比"], 3.0)
            score += s
            max_score += self.weights["明確否認"]["停頓時間佔比"]
            detailed_scores["停頓時間佔比"] = s / self.weights["明確否認"]["停頓時間佔比"]
        else:
            detailed_scores["停頓時間佔比"] = 0
        
        # 檢查中立情緒過高
        neutral_score = emotion_probs.get("neutral", 0)
        if neutral_score > self.thresholds["明確否認"]["中立"]:
            s = min((neutral_score / self.thresholds["明確否認"]["中立"]) * self.weights["明確否認"]["中立"], 3.0)
            score += s
            max_score += self.weights["明確否認"]["中立"]
            detailed_scores["中立"] = s / self.weights["明確否認"]["中立"]
        else:
            detailed_scores["中立"] = 0
        
        # 檢查憤怒短暫尖峰
        anger_score = emotion_probs.get("anger", 0)
        if anger_score > self.thresholds["明確否認"]["憤怒"]:
            s = min((anger_score / self.thresholds["明確否認"]["憤怒"]) * self.weights["明確否認"]["憤怒"],3.0)
            score += s
            max_score += self.weights["明確否認"]["憤怒"]
            detailed_scores["憤怒"] = s / self.weights["明確否認"]["憤怒"] 

        else:
             detailed_scores["憤怒"] = 0
        

        # 檢查顫抖強度
        tremor_intensity = audio_features.get("tremor", {}).get("tremor_intensity", 0)
        if tremor_intensity > self.thresholds["明確否認"]["顫抖強度"]:
            s = min((tremor_intensity / self.thresholds["明確否認"]["顫抖強度"]) * self.weights["明確否認"]["顫抖強度"], 3.0)
            score += s
            max_score += self.weights["明確否認"]["顫抖強度"]
            detailed_scores["顫抖強度"] = s / self.weights["明確否認"]["顫抖強度"]
        else:
            detailed_scores["顫抖強度"] = 0
        
        # 歸一化分數 (0-1)
        #normalized_score = score / max_score if max_score > 0 else 0
        #return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    

    def _evaluate_embarrassment_cover(self, audio_features, emotion_probs):
        """評估是否存在尷尬掩蓋模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查頻率顫抖
        jitter_local = audio_features.get("tremor", {}).get("jitter_local", 0)
        if jitter_local > self.thresholds["尷尬掩蓋"]["頻率顫抖"]:
            s = min((jitter_local / self.thresholds["尷尬掩蓋"]["頻率顫抖"]) * self.weights["尷尬掩蓋"]["頻率顫抖"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["頻率顫抖"]
            detailed_scores["頻率顫抖"] = s / self.weights["尷尬掩蓋"]["頻率顫抖"]
        else:
            detailed_scores["頻率顫抖"] = 0
        
        # 檢查音量顫抖
        shimmer_local = audio_features.get("tremor", {}).get("shimmer_local", 0)
        if shimmer_local > self.thresholds["尷尬掩蓋"]["音量顫抖"]:
            s = min((shimmer_local / self.thresholds["尷尬掩蓋"]["音量顫抖"]) * self.weights["尷尬掩蓋"]["音量顫抖"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["音量顫抖"]
            detailed_scores["音量顫抖"] = s / self.weights["尷尬掩蓋"]["音量顫抖"]
        else:
            detailed_scores["音量顫抖"] = 0
        
        # 檢查恐懼情緒
        fear_score = emotion_probs.get("fear", 0)
        if fear_score > self.thresholds["尷尬掩蓋"]["恐懼"]:
            s = min((fear_score / self.thresholds["尷尬掩蓋"]["恐懼"]) * self.weights["尷尬掩蓋"]["恐懼"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["恐懼"]
            detailed_scores["恐懼"] = s / self.weights["尷尬掩蓋"]["恐懼"]
        else:
            detailed_scores["恐懼"] = 0
        
        # 檢查厭惡情緒
        disgust_score = emotion_probs.get("disgust", 0)
        if disgust_score > self.thresholds["尷尬掩蓋"]["厭惡"]:
            s = min((disgust_score / self.thresholds["尷尬掩蓋"]["厭惡"]) * self.weights["尷尬掩蓋"]["厭惡"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["厭惡"]
            detailed_scores["厭惡"] = s / self.weights["尷尬掩蓋"]["厭惡"]
        else:
            detailed_scores["厭惡"] = 0
        
        # 檢查平均停頓長度
        pause_mean_duration = audio_features.get("speech_rate", {}).get("pause_mean_duration", 0)
        if pause_mean_duration > self.thresholds["尷尬掩蓋"]["平均停頓長度"]:
            s = min((pause_mean_duration / self.thresholds["尷尬掩蓋"]["平均停頓長度"]) * self.weights["尷尬掩蓋"]["平均停頓長度"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["平均停頓長度"]
            detailed_scores["平均停頓長度"] = s / self.weights["尷尬掩蓋"]["平均停頓長度"]
        else:
            detailed_scores["平均停頓長度"] = 0
        
        # 檢查每秒停頓次數
        pause_rate = audio_features.get("speech_rate", {}).get("pause_rate", 0)
        if pause_rate > self.thresholds["尷尬掩蓋"]["每秒停頓次數"]:
            s = min((pause_rate / self.thresholds["尷尬掩蓋"]["每秒停頓次數"]) * self.weights["尷尬掩蓋"]["每秒停頓次數"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["每秒停頓次數"]
            detailed_scores["每秒停頓次數"] = s / self.weights["尷尬掩蓋"]["每秒停頓次數"]
        else:
            detailed_scores["每秒停頓次數"] = 0
        
        # 檢查諧噪比低
        hnr = audio_features.get("tremor", {}).get("hnr", 30)
        if hnr < self.thresholds["尷尬掩蓋"]["諧噪比"]:
            s = min(((self.thresholds["尷尬掩蓋"]["諧噪比"] - hnr) / self.thresholds["尷尬掩蓋"]["諧噪比"]) * self.weights["尷尬掩蓋"]["諧噪比"], 3.0)
            score += s
            max_score += self.weights["尷尬掩蓋"]["諧噪比"]
            detailed_scores["諧噪比"] = s / self.weights["尷尬掩蓋"]["諧噪比"]
        else:
            detailed_scores["諧噪比"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    
    def _evaluate_alert_avoidance(self, audio_features, emotion_probs):
        """評估是否存在警覺避談模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查提問後語速增加
        current_speech_rate = 0
        if len(audio_features.get("speech_rate", {}).get("local_speech_rates", [])) > 0:
            current_speech_rate = np.mean(audio_features["speech_rate"]["local_speech_rates"])
            
        baseline_speech_rate = self.baseline.get("語速", current_speech_rate)
        speech_rate_increase = ((current_speech_rate - baseline_speech_rate) / baseline_speech_rate * 100) if baseline_speech_rate > 0 else 0
        
        if speech_rate_increase > self.thresholds["警覺避談"]["語速增加百分比"]:
            s = min((speech_rate_increase / self.thresholds["警覺避談"]["語速增加百分比"]) * self.weights["警覺避談"]["語速增加"], 3.0)
            score += s
            max_score += self.weights["警覺避談"]["語速增加"]
            detailed_scores["語速增加"] = s / self.weights["警覺避談"]["語速增加"]
        else:
            detailed_scores["語速增加"] = 0
        
        # 檢查語速範圍變窄
        speech_rate_range = audio_features.get("speech_rate", {}).get("speech_rate_range", 1.0)
        if speech_rate_range < self.thresholds["警覺避談"]["語速範圍"]:
            s = min(((self.thresholds["警覺避談"]["語速範圍"] - speech_rate_range) / self.thresholds["警覺避談"]["語速範圍"]) * self.weights["警覺避談"]["語速範圍"], 3.0)
            score += s
            max_score += self.weights["警覺避談"]["語速範圍"]
            detailed_scores["語速範圍"] = s / self.weights["警覺避談"]["語速範圍"]
        else:
            detailed_scores["語速範圍"] = 0
        
        # 檢查主題轉換頻率 (這個可能需要文本分析，此處為簡化實現)
        # 假設主題轉換與停頓次數及語速變化相關
        pause_count = audio_features.get("speech_rate", {}).get("pause_count", 0)
        speech_duration = audio_features.get("speech_rate", {}).get("speech_duration", 60)
        speech_minutes = max(speech_duration / 60, 1)
        
        # 簡單估算主題轉換頻率
        topic_changes_per_minute = (pause_count / speech_minutes) * (speech_rate_range / 1.0)
        
        if topic_changes_per_minute > self.thresholds["警覺避談"]["主題轉換頻率"]:
            s = min((topic_changes_per_minute / self.thresholds["警覺避談"]["主題轉換頻率"]) * self.weights["警覺避談"]["主題轉換頻率"], 3.0)
            score += s
            max_score += self.weights["警覺避談"]["主題轉換頻率"]
            detailed_scores["主題轉換頻率"] = s / self.weights["警覺避談"]["主題轉換頻率"]
        else:
            detailed_scores["主題轉換頻率"] = 0
        
        # 檢查中立情緒過高
        neutral_score = emotion_probs.get("neutral", 0)
        if neutral_score > self.thresholds["警覺避談"]["中立情緒"]:
            s = min((neutral_score / self.thresholds["警覺避談"]["中立情緒"]) * self.weights["警覺避談"]["中立情緒"], 3.0)
            score += s
            max_score += self.weights["警覺避談"]["中立情緒"]
            detailed_scores["中立情緒"] = s / self.weights["警覺避談"]["中立情緒"]
        else:
            detailed_scores["中立情緒"] = 0
        
        # 檢查無聊情緒
        boredom_score = emotion_probs.get("boredom", 0)
        if boredom_score > self.thresholds["警覺避談"]["無聊"]:
            s = min((boredom_score / self.thresholds["警覺避談"]["無聊"]) * self.weights["警覺避談"]["無聊"], 3.0)
            score += s
            max_score += self.weights["警覺避談"]["無聊"]
            detailed_scores["無聊"] = s / self.weights["警覺避談"]["無聊"]
        else:
            detailed_scores["無聊"] = 0
        
        # 音量下降（相對基準）：self.baseline 的「平均音量」同樣是用這通電話自己的特徵設定的，
        # baseline 恆等於 current，差異永遠是 0%。明確跳過而非算出恆為 0 的假差異；
        # 不影響現有分數（該分支本來就從未觸發）。
        detailed_scores["音量下降"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    
    def _evaluate_hesitation(self, audio_features, emotion_probs):
        """評估是否存在猶豫不決模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查停頓次數
        pause_count = audio_features.get("speech_rate", {}).get("pause_count", 0)
        speech_duration = audio_features.get("speech_rate", {}).get("speech_duration", 60)
        speech_minutes = max(speech_duration / 60, 1)
        pauses_per_minute = pause_count / speech_minutes
        
        if pauses_per_minute > self.thresholds["猶豫不決"]["停頓次數"]:
            s = min((pauses_per_minute / self.thresholds["猶豫不決"]["停頓次數"]) * self.weights["猶豫不決"]["停頓次數"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["停頓次數"]
            detailed_scores["停頓次數"] = s / self.weights["猶豫不決"]["停頓次數"]
        else:
            detailed_scores["停頓次數"] = 0
        
        # 檢查停頓時間佔比
        pause_ratio = audio_features.get("speech_rate", {}).get("pause_ratio", 0)
        if pause_ratio > self.thresholds["猶豫不決"]["停頓時間佔比"]:
            s = min((pause_ratio / self.thresholds["猶豫不決"]["停頓時間佔比"]) * self.weights["猶豫不決"]["停頓時間佔比"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["停頓時間佔比"]
            detailed_scores["停頓時間佔比"] = s / self.weights["猶豫不決"]["停頓時間佔比"]
        else:
            detailed_scores["停頓時間佔比"] = 0
        
        # 檢查停頓時間標準差
        pause_std = audio_features.get("speech_rate", {}).get("pause_std_duration", 0)
        if pause_std > self.thresholds["猶豫不決"]["停頓時間標準差"]:
            s = min((pause_std / self.thresholds["猶豫不決"]["停頓時間標準差"]) * self.weights["猶豫不決"]["停頓時間標準差"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["停頓時間標準差"]
            detailed_scores["停頓時間標準差"] = s / self.weights["猶豫不決"]["停頓時間標準差"]
        else:
            detailed_scores["停頓時間標準差"] = 0
        
        # 檢查諧噪比低
        hnr = audio_features.get("tremor", {}).get("hnr", 30)
        if hnr < self.thresholds["猶豫不決"]["諧噪比"]:
            s = min(((self.thresholds["猶豫不決"]["諧噪比"] - hnr) / self.thresholds["猶豫不決"]["諧噪比"]) * self.weights["猶豫不決"]["諧噪比"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["諧噪比"]
            detailed_scores["諧噪比"] = s / self.weights["猶豫不決"]["諧噪比"]
        else:
            detailed_scores["諧噪比"] = 0
        
        # 檢查有聲段比例低
        voiced_ratio = audio_features.get("pitch", {}).get("voiced_ratio", 1)
        if voiced_ratio < self.thresholds["猶豫不決"]["有聲段比例"]:
            s = min(((self.thresholds["猶豫不決"]["有聲段比例"] - voiced_ratio) / self.thresholds["猶豫不決"]["有聲段比例"]) * self.weights["猶豫不決"]["有聲段比例"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["有聲段比例"]
            detailed_scores["有聲段比例"] = s / self.weights["猶豫不決"]["有聲段比例"]
        else:
            detailed_scores["有聲段比例"] = 0
        
        # 檢查顫抖頻率範圍
        tremor_frequency = audio_features.get("tremor", {}).get("tremor_frequency", 0)
        in_tremor_range = (tremor_frequency >= self.thresholds["猶豫不決"]["顫抖頻率下限"] and 
                          tremor_frequency <= self.thresholds["猶豫不決"]["顫抖頻率上限"])
        
        if in_tremor_range and tremor_frequency > 0:
            s = self.weights["猶豫不決"]["顫抖頻率範圍"]
            score += s
            max_score += self.weights["猶豫不決"]["顫抖頻率範圍"]
            detailed_scores["顫抖頻率範圍"] = 1.0
        else:
            detailed_scores["顫抖頻率範圍"] = 0
        
        # 檢查顫抖強度
        tremor_intensity = audio_features.get("tremor", {}).get("tremor_intensity", 0)
        if tremor_intensity > self.thresholds["猶豫不決"]["顫抖強度"]:
            s = min((tremor_intensity / self.thresholds["猶豫不決"]["顫抖強度"]) * self.weights["猶豫不決"]["顫抖強度"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["顫抖強度"]
            detailed_scores["顫抖強度"] = s / self.weights["猶豫不決"]["顫抖強度"]
        else:
            detailed_scores["顫抖強度"] = 0
        
        # 檢查恐懼加悲傷情緒
        fear_score = emotion_probs.get("fear", 0)
        sad_score = emotion_probs.get("sad", 0)
        emotion_sum = fear_score + sad_score
        
        if emotion_sum > self.thresholds["猶豫不決"]["恐懼加悲傷"]:
            s = min((emotion_sum / self.thresholds["猶豫不決"]["恐懼加悲傷"]) * self.weights["猶豫不決"]["恐懼加悲傷"], 3.0)
            score += s
            max_score += self.weights["猶豫不決"]["恐懼加悲傷"]
            detailed_scores["恐懼加悲傷"] = s / self.weights["猶豫不決"]["恐懼加悲傷"]
        else:
            detailed_scores["恐懼加悲傷"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    
    def _evaluate_abnormal_excitement(self, audio_features, emotion_probs):
        """評估是否存在異常興奮模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查快樂情緒過高
        happy_score = emotion_probs.get("happy", 0)
        if happy_score > self.thresholds["異常興奮"]["快樂"]:
            s = min((happy_score / self.thresholds["異常興奮"]["快樂"]) * self.weights["異常興奮"]["快樂"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["快樂"]
            detailed_scores["快樂"] = s / self.weights["異常興奮"]["快樂"]
        else:
            detailed_scores["快樂"] = 0
        
        # 檢查中立情緒過低
        neutral_score = emotion_probs.get("neutral", 0)
        if neutral_score < self.thresholds["異常興奮"]["中立"]:
            s = min(((self.thresholds["異常興奮"]["中立"] - neutral_score) / self.thresholds["異常興奮"]["中立"]) * self.weights["異常興奮"]["中立"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["中立"]
            detailed_scores["中立"] = s / self.weights["異常興奮"]["中立"]
        else:
            detailed_scores["中立"] = 0
        
        # 檢查語速增加
        current_speech_rate = 0
        if len(audio_features.get("speech_rate", {}).get("local_speech_rates", [])) > 0:
            current_speech_rate = np.mean(audio_features["speech_rate"]["local_speech_rates"])
            
        baseline_speech_rate = self.baseline.get("語速", current_speech_rate)
        speech_rate_increase = ((current_speech_rate - baseline_speech_rate) / baseline_speech_rate * 100) if baseline_speech_rate > 0 else 0
        
        if speech_rate_increase > self.thresholds["異常興奮"]["語速增加百分比"]:
            s = min((speech_rate_increase / self.thresholds["異常興奮"]["語速增加百分比"]) * self.weights["異常興奮"]["語速增加"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["語速增加"]
            detailed_scores["語速增加"] = s / self.weights["異常興奮"]["語速增加"]
        else:
            detailed_scores["語速增加"] = 0
        
        # 檢查突然語速變化次數
        sudden_changes = audio_features.get("speech_rate", {}).get("sudden_speed_changes", 0)
        speech_duration = audio_features.get("speech_rate", {}).get("speech_duration", 60)
        speech_minutes = max(speech_duration / 60, 1)
        sudden_changes_per_minute = sudden_changes / speech_minutes
        
        if sudden_changes_per_minute > self.thresholds["異常興奮"]["突然語速變化次數"]:
            s = min((sudden_changes_per_minute / self.thresholds["異常興奮"]["突然語速變化次數"]) * self.weights["異常興奮"]["突然語速變化次數"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["突然語速變化次數"]
            detailed_scores["突然語速變化次數"] = s / self.weights["異常興奮"]["突然語速變化次數"]
        else:
            detailed_scores["突然語速變化次數"] = 0
        
        # 音量增加（相對基準）：self.baseline 的「平均音量」同樣是用這通電話自己的特徵設定的，
        # baseline 恆等於 current，差異永遠是 0%。明確跳過而非算出恆為 0 的假差異；
        # 不影響現有分數（該分支本來就從未觸發）。
        detailed_scores["音量增加"] = 0
        
        # 檢查音量標準差
        volume_std = audio_features.get("volume", {}).get("std_volume", 0)
        if volume_std > self.thresholds["異常興奮"]["音量標準差"]:
            s = min((volume_std / self.thresholds["異常興奮"]["音量標準差"]) * self.weights["異常興奮"]["音量標準差"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["音量標準差"]
            detailed_scores["音量標準差"] = s / self.weights["異常興奮"]["音量標準差"]
        else:
            detailed_scores["音量標準差"] = 0
        
        # 檢查音高範圍擴大
        pitch_range = audio_features.get("pitch", {}).get("pitch_range", 0)
        baseline_pitch_range = 100  # 假設基線值
        pitch_range_increase = ((pitch_range - baseline_pitch_range) / baseline_pitch_range * 100) if baseline_pitch_range > 0 else 0
        
        if pitch_range_increase > self.thresholds["異常興奮"]["音高範圍增加"]:
            s = min((pitch_range_increase / self.thresholds["異常興奮"]["音高範圍增加"]) * self.weights["異常興奮"]["音高範圍增加"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["音高範圍增加"]
            detailed_scores["音高範圍增加"] = s / self.weights["異常興奮"]["音高範圍增加"]
        else:
            detailed_scores["音高範圍增加"] = 0
        
        # 檢查音高變化率
        pitch_change_rate = audio_features.get("pitch", {}).get("pitch_change_rate", 0)
        if pitch_change_rate > self.thresholds["異常興奮"]["音高變化率"]:
            s = min((pitch_change_rate / self.thresholds["異常興奮"]["音高變化率"]) * self.weights["異常興奮"]["音高變化率"], 3.0)
            score += s
            max_score += self.weights["異常興奮"]["音高變化率"]
            detailed_scores["音高變化率"] = s / self.weights["異常興奮"]["音高變化率"]
        else:
            detailed_scores["音高變化率"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores
    
        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores
    
    def _evaluate_logical_gaps(self, audio_features, emotion_probs):
        """評估是否存在邏輯漏洞模式"""
        score = 0
        max_score = 0
        detailed_scores = {}
        
        # 檢查語速變化標準差
        speech_rate_std = audio_features.get("speech_rate", {}).get("speech_rate_variation", 0)
        if speech_rate_std > self.thresholds["邏輯漏洞"]["語速變化標準差"]:
            s = min((speech_rate_std / self.thresholds["邏輯漏洞"]["語速變化標準差"]) * self.weights["邏輯漏洞"]["語速變化標準差"], 3.0)
            score += s
            max_score += self.weights["邏輯漏洞"]["語速變化標準差"]
            detailed_scores["語速變化標準差"] = s / self.weights["邏輯漏洞"]["語速變化標準差"]
        else:
            detailed_scores["語速變化標準差"] = 0
        
        # 語速不一致 (需要時間序列分析，這裡用簡化方法)
        local_speech_rates = audio_features.get("speech_rate", {}).get("local_speech_rates", [])
        if len(local_speech_rates) > 1:
            speech_rate_diffs = np.abs(np.diff(local_speech_rates))
            max_diff_percent = (np.max(speech_rate_diffs) / np.mean(local_speech_rates) * 100) if np.mean(local_speech_rates) > 0 else 0
            
            if max_diff_percent > self.thresholds["邏輯漏洞"]["語速不一致閾值"]:
                s = min((max_diff_percent / self.thresholds["邏輯漏洞"]["語速不一致閾值"]) * self.weights["邏輯漏洞"]["語速不一致"], 3.0)
                score += s
                max_score += self.weights["邏輯漏洞"]["語速不一致"]
                detailed_scores["語速不一致"] = s / self.weights["邏輯漏洞"]["語速不一致"]
            else:
                detailed_scores["語速不一致"] = 0
        else:
            detailed_scores["語速不一致"] = 0
        
        # 檢查音高不穩定性
        pitch_instability = audio_features.get("pitch", {}).get("pitch_instability", 0)
        if pitch_instability > self.thresholds["邏輯漏洞"]["音高不穩定性"]:
            s = min((pitch_instability / self.thresholds["邏輯漏洞"]["音高不穩定性"]) * self.weights["邏輯漏洞"]["音高不穩定性"], 3.0)
            score += s
            max_score += self.weights["邏輯漏洞"]["音高不穩定性"]
            detailed_scores["音高不穩定性"] = s / self.weights["邏輯漏洞"]["音高不穩定性"]
        else:
            detailed_scores["音高不穩定性"] = 0
        
        # 檢查情緒轉換頻率 (簡化實現)
        # 用情緒波動來估計
        emotions = ["anger", "boredom", "disgust", "fear", "happy", "neutral", "sad"]
        emotion_values = [emotion_probs.get(e, 0) for e in emotions]
        emotion_std = np.std(emotion_values)
        
        # 高情緒標準差表示情緒多變
        emotion_changes = emotion_std * 10  # 簡單估算
        
        if emotion_changes > self.thresholds["邏輯漏洞"]["情緒轉換頻率"]:
            s = min((emotion_changes / self.thresholds["邏輯漏洞"]["情緒轉換頻率"]) * self.weights["邏輯漏洞"]["情緒轉換頻率"], 3.0)
            score += s
            max_score += self.weights["邏輯漏洞"]["情緒轉換頻率"]
            detailed_scores["情緒轉換頻率"] = s / self.weights["邏輯漏洞"]["情緒轉換頻率"]
        else:
            detailed_scores["情緒轉換頻率"] = 0
        
        # 檢查關鍵停頓長度
        pause_durations = audio_features.get("speech_rate", {}).get("pause_durations", [])
        if pause_durations:
            longest_pause = max(pause_durations)
            if longest_pause > self.thresholds["邏輯漏洞"]["關鍵停頓長度"]:
                s = min((longest_pause / self.thresholds["邏輯漏洞"]["關鍵停頓長度"]) * self.weights["邏輯漏洞"]["關鍵停頓長度"], 3.0)
                score += s
                max_score += self.weights["邏輯漏洞"]["關鍵停頓長度"]
                detailed_scores["關鍵停頓長度"] = s / self.weights["邏輯漏洞"]["關鍵停頓長度"]
            else:
                detailed_scores["關鍵停頓長度"] = 0
        else:
            detailed_scores["關鍵停頓長度"] = 0
        
        # 檢查停頓分布標準差
        pause_std = audio_features.get("speech_rate", {}).get("pause_std_duration", 0)
        if pause_std > self.thresholds["邏輯漏洞"]["停頓分布標準差"]:
            s = min((pause_std / self.thresholds["邏輯漏洞"]["停頓分布標準差"]) * self.weights["邏輯漏洞"]["停頓分布標準差"], 3.0)
            score += s
            max_score += self.weights["邏輯漏洞"]["停頓分布標準差"]
            detailed_scores["停頓分布標準差"] = s / self.weights["邏輯漏洞"]["停頓分布標準差"]
        else:
            detailed_scores["停頓分布標準差"] = 0
        
        # 歸一化分數 (0-1)
        # normalized_score = score / max_score if max_score > 0 else 0
        # return min(normalized_score, 1.0), detailed_scores

        # 歸一化分數 (0-1)，添加平滑因子
        normalized_score = score / (max_score + 0.1) if max_score > 0 else 0
        # 使用sigmoid函數讓極端值變得更平滑
        normalized_score = 1 / (1 + np.exp(-5 * (normalized_score - 0.5)))
        return min(normalized_score, 0.95), detailed_scores

class FraudRiskAssessor:
    """
    欺騙風險評估器，根據欺騙模式評分計算總體風險
    """
    def __init__(self, industry_type=None, company_size=None):
        """
        初始化風險評估器
        
        Parameters:
        industry_type (str, optional): 產業類型，用於調整評分
        company_size (str, optional): 公司規模，用於調整評分
        """
        self.industry_adjustments = self._load_industry_adjustments(industry_type)
        self.size_adjustments = self._load_size_adjustments(company_size)
        self.pattern_weights = self._init_pattern_weights()
        
    def _load_industry_adjustments(self, industry_type):
        """載入產業特定調整參數"""
        if not industry_type:
            return {}
            
        adjustments = {
            "科技業": {
                "異常興奮": 1.2,  # 科技業常有過度樂觀描述
                "警覺避談": 1.1
            },
            "金融業": {
                "明確否認": 1.2,  # 金融業更敏感於明確否認
                "邏輯漏洞": 1.3
            },
            "傳統製造": {
                "矛盾衝突": 1.1,
                "尷尬掩蓋": 1.1
            },
            "生技醫療": {
                "矛盾衝突": 1.2,  # 生技業常有研發風險與財務表現的矛盾
                "異常興奮": 1.1
            }
        }
        
        return adjustments.get(industry_type, {})
        
    def _load_size_adjustments(self, company_size):
        """載入公司規模特定調整參數"""
        if not company_size:
            return {}
            
        adjustments = {
            "大型上市": {
                "攻擊性謊言": 0.9,  # 大型公司較少出現激烈反應
                "明確否認": 1.1
            },
            "中型上市": {
                "矛盾衝突": 1.1,
                "異常興奮": 1.1
            },
            "小型上市": {
                "猶豫不決": 1.2,  # 小型公司較易出現不確定性
                "警覺避談": 1.1
            }
        }
        
        return adjustments.get(company_size, {})
    
    def _init_pattern_weights(self):
        """初始化各欺騙模式的權重"""
        return {
            "攻擊性謊言": 1.2,
            "矛盾衝突": 1.5,
            "明確否認": 1.4,
            "尷尬掩蓋": 1.3,
            "警覺避談": 1.2,
            "猶豫不決": 1.0,
            "異常興奮": 1.1,
            "邏輯漏洞": 1.3
        }
    
    def calculate_risk(self, pattern_scores):
        """
        計算總體風險分數
        
        Parameters:
        pattern_scores (dict): 各欺騙模式的評分
        
        Returns:
        tuple: (總體風險分數, 風險級別)
        """
        weighted_sum = 0
        total_weight = 0
        
        for pattern, score in pattern_scores.items():
            # 獲取基本權重
            weight = self.pattern_weights.get(pattern, 1.0)
            
            # 應用產業調整
            industry_factor = self.industry_adjustments.get(pattern, 1.0)
            weight *= industry_factor
            
            # 應用公司規模調整
            size_factor = self.size_adjustments.get(pattern, 1.0)
            weight *= size_factor
            
            weighted_sum += score * weight
            total_weight += weight
        
        # 計算加權平均
        risk_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 0
        risk_score = min(risk_score, 100)  # 確保最高為100
        
        # 確定風險級別
        if risk_score >= 70:
            risk_level = "高風險"
        elif risk_score >= 40:
            risk_level = "中風險"
        else:
            risk_level = "低風險"
        
        return risk_score, risk_level
    
    def generate_risk_report(self, pattern_scores, detailed_scores):
        """
        生成詳細的風險報告
        
        Parameters:
        pattern_scores (dict): 各欺騙模式的評分
        detailed_scores (dict): 各欺騙模式的詳細得分
        
        Returns:
        dict: 風險報告
        """
        risk_score, risk_level = self.calculate_risk(pattern_scores)
        
        # 找出主要欺騙模式 (評分最高的兩個)
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        primary_patterns = sorted_patterns[:2] if len(sorted_patterns) >= 2 else sorted_patterns
        
        # 找出關鍵指標 (每個主要模式中分數最高的指標)
        key_indicators = {}
        for pattern, _ in primary_patterns:
            if pattern in detailed_scores:
                pattern_details = detailed_scores[pattern]
                sorted_indicators = sorted(pattern_details.items(), key=lambda x: x[1], reverse=True)
                key_indicators[pattern] = sorted_indicators[:3] if len(sorted_indicators) >= 3 else sorted_indicators
        
        report = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "primary_patterns": primary_patterns,
            "key_indicators": key_indicators,
            "all_pattern_scores": pattern_scores,
            "detailed_scores": detailed_scores
        }
        
        return report


def detect_fraud_patterns(audio_features, emotion_probs, config_path=None, industry_type=None, company_size=None):
    """
    檢測音訊中的欺騙模式並評估風險
    
    Parameters:
    audio_features (dict): 音訊特徵
    emotion_probs (dict): 情緒預測概率
    config_path (str, optional): 閾值配置檔案路徑
    industry_type (str, optional): 產業類型
    company_size (str, optional): 公司規模
    
    Returns:
    dict: 風險評估報告
    """
    # 初始化欺騙模式分類器
    classifier = FraudPatternClassifier(config_path)
    
    # 分類欺騙模式
    pattern_scores, detailed_scores = classifier.classify_patterns(audio_features, emotion_probs)
    
    # 初始化風險評估器
    assessor = FraudRiskAssessor(industry_type, company_size)
    
    # 生成風險報告
    report = assessor.generate_risk_report(pattern_scores, detailed_scores)
    
    return report


def save_thresholds_to_file(thresholds, file_path):
    """
    將閾值配置保存到文件
    
    Parameters:
    thresholds (dict): 閾值配置
    file_path (str): 保存路徑
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存閾值配置時出錯：{e}")
        return False


def visualize_fraud_patterns(report, save_path=None):
    """
    視覺化欺騙模式分析結果
    
    Parameters:
    report (dict): 風險評估報告
    save_path (str, optional): 圖表保存路徑
    
    Returns:
    tuple: (fig, ax) matplotlib圖表物件
    """
    pattern_scores = report["all_pattern_scores"]
    
    # 創建雷達圖
    labels = list(pattern_scores.keys())
    values = list(pattern_scores.values())
    
    # 計算角度
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    # 閉合雷達圖
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(4.5, 4.2), subplot_kw=dict(polar=True))

    # 繪製雷達圖
    ax.plot(angles, values, 'o-', linewidth=1.5, markersize=4)
    ax.fill(angles, values, alpha=0.25)

    # 設置標籤
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1], fontsize=8)

    # 設置y軸範圍
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=7)

    # 添加標題和風險分數
    plt.title(f'欺騙模式分析\n風險分數: {report["risk_score"]:.1f} ({report["risk_level"]})', size=10, color='black', y=1.15)
    
    # 如果需要保存圖表
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return fig, ax