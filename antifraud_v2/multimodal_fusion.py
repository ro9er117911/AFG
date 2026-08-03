import re
import numpy as np

class MultimodalFraudDetector:
    """
    多模態欺騙檢測器，整合音訊和文字分析
    """
    def __init__(self):
        self.fraud_patterns = [
            "攻擊性謊言", "矛盾衝突", "明確否認", "尷尬掩蓋", 
            "警覺避談", "猶豫不決", "異常興奮", "邏輯漏洞"
        ]
        self.mode_weights = {
        "攻擊性謊言": {"audio": 0.65, "text": 0.35},  # 適度調整，避免過於依賴音訊
        "矛盾衝突": {"audio": 0.55, "text": 0.45},  # 適度調整
        "明確否認": {"audio": 0.35, "text": 0.65},  # 更依賴文字特徵
        "尷尬掩蓋": {"audio": 0.75, "text": 0.25},  # 主要依賴音訊特徵
        "警覺避談": {"audio": 0.3, "text": 0.7},    # 主要依賴文字特徵
        "猶豫不決": {"audio": 0.65, "text": 0.35},  # 更依賴音訊特徵
        "異常興奮": {"audio": 0.65, "text": 0.35},  # 適度調整
        "邏輯漏洞": {"audio": 0.2, "text": 0.8}     
}
        
    def parse_text_analysis(self, text_analysis):
        """
        從GPT-4文字分析中提取欺騙模式評分
        
        Parameters:
        text_analysis (str): GPT-4生成的分析文本
        
        Returns:
        dict: 各欺騙模式的評分 (0-1)
        """
        pattern_scores = {}
        
        # 使用正則表達式提取評分
        for pattern in self.fraud_patterns:
            regex = f"{pattern}[：:]\s*(\d+(?:\.\d+)?)%"
            match = re.search(regex, text_analysis)
            if match:
                score = float(match.group(1)) / 100.0
                pattern_scores[pattern] = score
            else:
                pattern_scores[pattern] = 0.0
                
        return pattern_scores
    
    def extract_fraud_probability(self, text_analysis):
        """
        從文本中提取總體詐欺可能性
        
        Parameters:
        text_analysis (str): GPT-4生成的分析文本
        
        Returns:
        float: 詐欺可能性 (0-100)
        """
        match = re.search(r"詐欺可能性[：:]\s*(\d+(?:\.\d+)?)%", text_analysis)
        if match:
            return float(match.group(1))
        
        # 嘗試其他可能的格式
        match = re.search(r"(\d+(?:\.\d+)?)%的詐欺可能性", text_analysis)
        if match:
            return float(match.group(1))
            
        # 如果無法提取，返回None
        return None
    
    def integrate_analysis(self, audio_report, text_analysis):
        """
        整合音訊和文字分析結果
        
        Parameters:
        audio_report (dict): 音訊分析報告
        text_analysis (str): GPT-4生成的文字分析
        
        Returns:
        dict: 整合後的分析報告
        """
        # 提取音訊分析結果
        audio_scores = audio_report["all_pattern_scores"]
        
        # 解析文字分析結果
        text_scores = self.parse_text_analysis(text_analysis)
        
        # 整合兩種分析結果
        integrated_scores = {}
        for pattern in self.fraud_patterns:
            if pattern in audio_scores and pattern in text_scores:
                # 獲取兩種分析的權重
                audio_weight = self.mode_weights[pattern]["audio"]
                text_weight = self.mode_weights[pattern]["text"]
                
                # 計算加權評分
                audio_score = audio_scores[pattern]
                text_score = text_scores[pattern]
                weighted_score = (audio_score * audio_weight) + (text_score * text_weight)
                
                # 檢查互補和衝突情況
                if audio_score > 0.7 and text_score > 0.7:
                    # 兩種分析都高度確認，增強可信度
                    weighted_score = min(1.0, weighted_score * 1.2)
                elif abs(audio_score - text_score) > 0.5:
                    # 兩種分析存在較大差異，降低可信度
                    weighted_score = weighted_score * 0.8
                
                integrated_scores[pattern] = weighted_score
        
        # 確保所有模式都有評分
        for pattern in self.fraud_patterns:
            if pattern not in integrated_scores:
                integrated_scores[pattern] = 0.0
        
        # 找出主要欺騙模式
        sorted_patterns = sorted(integrated_scores.items(), key=lambda x: x[1], reverse=True)
        primary_patterns = sorted_patterns[:2] if len(sorted_patterns) >= 2 else sorted_patterns
        
        # 計算整合風險分數
        # audio_risk = audio_report["risk_score"]
        # text_risk = self.extract_fraud_probability(text_analysis)
        
        # if text_risk is not None:
        #     # 如果能提取出文字分析的風險評分，進行加權平均
        #     integrated_risk = audio_risk * 0.6 + text_risk * 0.4
        # else:
        #     # 如果無法提取，則使用音訊風險評分
        #     integrated_risk = audio_risk

        # 計算整合風險分數
        audio_risk = audio_report["risk_score"]
        text_risk = self.extract_fraud_probability(text_analysis)

        if text_risk is not None:
            # 使用更平衡的加權方式，避免一方過度主導
            if abs(audio_risk - text_risk) > 40:  # 若差異過大
                # 給予較低置信度一方較低權重
                if audio_risk > text_risk:
                    integrated_risk = audio_risk * 0.55 + text_risk * 0.45
                else:
                    integrated_risk = audio_risk * 0.45 + text_risk * 0.55
            else:
                # 若差異不大，保持原有權重
                integrated_risk = audio_risk * 0.6 + text_risk * 0.4
        else:
            # 若無法提取文字風險評分，則使用音訊風險評分但降低置信度
            integrated_risk = audio_risk * 0.9  # 降低10%以反映不完整分析
        
        # 確定風險級別
        if integrated_risk >= 70:
            risk_level = "高風險"
        elif integrated_risk >= 40:
            risk_level = "中風險"
        else:
            risk_level = "低風險"
        
        # 生成整合報告
        integrated_report = {
            "risk_score": integrated_risk,
            "risk_level": risk_level,
            "primary_patterns": primary_patterns,
            "all_pattern_scores": integrated_scores,
            "audio_analysis": audio_report,
            "text_scores": text_scores,
            "fusion_confidence": self.calculate_fusion_confidence(audio_scores, text_scores)
        }
        
        return integrated_report
    
    def calculate_fusion_confidence(self, audio_scores, text_scores):
        """
        計算融合分析的置信度
        
        Parameters:
        audio_scores (dict): 音訊分析評分
        text_scores (dict): 文字分析評分
        
        Returns:
        float: 融合置信度 (0-1)
        """
        # 計算音訊和文字分析結果的一致性
        pattern_diffs = []
        for pattern in self.fraud_patterns:
            if pattern in audio_scores and pattern in text_scores:
                diff = abs(audio_scores[pattern] - text_scores[pattern])
                pattern_diffs.append(diff)
        
        if not pattern_diffs:
            return 0.5  # 默認中等置信度
        
        # 平均差異越小，置信度越高
        avg_diff = sum(pattern_diffs) / len(pattern_diffs)
        confidence = 1.0 - min(avg_diff, 1.0)
        
        return confidence

# 提供一個簡單的接口函數
def integrate_audio_text_analysis(audio_report, text_analysis):
    """
    整合音訊和文字欺騙分析結果
    
    Parameters:
    audio_report (dict): 從fraud_detection.py生成的音訊分析報告
    text_analysis (str): 從language_processing.py生成的GPT-4文字分析
    
    Returns:
    dict: 整合後的報告
    """
    detector = MultimodalFraudDetector()
    return detector.integrate_analysis(audio_report, text_analysis)