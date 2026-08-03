import logging
import tiktoken
import time
import re
import os
import json
from concurrent.futures import ThreadPoolExecutor
import openai
from faster_whisper import WhisperModel

# 初始化 Whisper 模型
whisper_model = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)


def transcribe_audio(audio_path):
    """
    轉錄音訊為文字
    
    Parameters:
    audio_path (str): 音訊檔案路徑
    
    Returns:
    str: 轉錄的文字內容
    """
    try:
        if not os.path.exists(audio_path):
            logging.error(f"音訊檔案不存在: {audio_path}")
            return "音訊檔案不存在"
            
        segments, info = whisper_model.transcribe(audio_path, language="zh")
        transcript_segments = [segment.text for segment in segments]
        transcript = " ".join(transcript_segments)
        
        logging.info(f"音訊轉錄成功，文本長度: {len(transcript)}")
        logging.info(f"轉錄結果類型: {type(transcript)}")
        return transcript
        
    except Exception as e:
        logging.error(f"音訊轉錄錯誤: {str(e)}")
        return "無法轉錄音訊"


def get_structured_fraud_analysis(transcript, emotion=None, audio_features=None):
    """
    獲取結構化的文字欺騙分析
    
    Parameters:
    transcript (str or list): 音訊轉錄文字，可能是字符串或列表
    emotion (str, optional): 預測的情緒
    audio_features (dict, optional): 音訊特徵
    
    Returns:
    str: 結構化的文字分析結果
    """
    try:
        # 檢查並轉換 transcript 類型
        if isinstance(transcript, list):
            # 如果是列表，找出非空元素並合併
            valid_texts = [t for t in transcript if isinstance(t, str) and t.strip()]
            if not valid_texts:
                return "欺騙分析無法完成: 轉錄文本為空"
            # 合併所有有效文本
            transcript = " ".join(valid_texts)
            logging.info(f"將轉錄文本從列表轉換為字符串，長度: {len(transcript)}")
        
        # 確保 transcript 是字符串
        if not isinstance(transcript, str):
            return f"欺騙分析無法完成: 轉錄文本類型非字符串 ({type(transcript)})"
        
        # 檢查文本是否為空
        if not transcript.strip():
            return "欺騙分析無法完成: 轉錄文本為空"
        # 步驟1: 將轉錄文本動態分段，使用較小的最大token數
        text_segments = dynamic_segment_transcript(transcript, max_tokens=4000)
        logging.info(f"轉錄文本已分割為 {len(text_segments)} 個段落")
        
        # 步驟2: 使用迭代方式分析所有段落
        segment_analyses = analyze_segments_iterative(text_segments, emotion)
        
        # 步驟3: 整合所有段落分析結果
        integrated_analysis = integrate_analyses(segment_analyses)
        
        # 步驟4: 將結果格式化為完整報告
        final_report = format_final_report(integrated_analysis)
        
        return final_report
        
    except Exception as e:
        logging.error(f"API調用錯誤: {str(e)}")
        return f"欺騙分析暫時無法完成: {str(e)}。請稍後再試。"

def count_tokens(text, model="gpt-4"):
    """精確計算token數"""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def dynamic_segment_transcript(transcript, max_tokens=4000):
    """動態適應性分段，基於token計數而非字符數"""
    print(f"傳入的transcript類型: {type(transcript)}, 長度: {len(transcript)}")
    print(f"前100個字符: {transcript[:100]}")
    
    # 如果文本太短，不需要分段
    if len(transcript) < 500:  # 假設500字符以下不需要分段
        return [transcript]
    
    encoder = tiktoken.encoding_for_model("gpt-4")
    
    # 嘗試使用多種方式分句
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', transcript)
    
    # 如果分句結果不理想（只有一個句子），嘗試中文分句
    if len(sentences) <= 1:
        sentences = re.split(r'([。！？])', transcript)
        # 重組分句，確保標點符號跟著前面的文字
        if len(sentences) > 1:
            tmp = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    tmp.append(sentences[i] + sentences[i+1])
                else:
                    tmp.append(sentences[i])
            sentences = tmp
    
    print(f"分句結果: {len(sentences)} 個句子")
    if len(sentences) <= 3:  # 只有少量句子，直接打印
        print(f"前{len(sentences)}個句子: {sentences}")
    else:
        print(f"前3個句子: {sentences[:3]}")
    
    # 如果還是只有一個句子，直接按字符長度強制分段
    if len(sentences) <= 1 and len(transcript) > 1000:  # 大文本但只有一個句子
        char_per_segment = 1000  # 每段約1000字符
        num_segments = (len(transcript) + char_per_segment - 1) // char_per_segment
        segments = []
        for i in range(num_segments):
            start = i * char_per_segment
            end = min((i + 1) * char_per_segment, len(transcript))
            segments.append(transcript[start:end])
        
        print(f"強制按字符長度分段，得到 {len(segments)} 個段落")
        return [seg for seg in segments if seg.strip()]  # 過濾空段落
    
    # 正常的token分段邏輯
    segments = []
    current_segment = []
    current_tokens = 0
    
    for sentence in sentences:
        if not sentence.strip():  # 跳過空句子
            continue
            
        sentence_tokens = len(encoder.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            if current_segment:  # 只有當有內容時才添加段落
                segments.append(" ".join(current_segment))
            current_segment = [sentence]
            current_tokens = sentence_tokens
        else:
            current_segment.append(sentence)
            current_tokens += sentence_tokens
    
    # 不要忘記最後一個段落
    if current_segment:  # 只有當有內容時才添加段落
        segments.append(" ".join(current_segment))
    
    # 過濾掉空段落（以防萬一）
    segments = [seg for seg in segments if seg.strip()]
    
    print(f"最終分段結果: {len(segments)} 個段落")
    for i, seg in enumerate(segments):
        print(f"段落 {i+1} 類型: {type(seg)}, 長度: {len(seg)}")
        print(f"段落 {i+1} 前50個字符: {seg[:50]}")
    
    return segments

def optimize_analysis_prompt(segment, index, total, emotion=None):
    """
    優化後的Prompt結構，精簡指令，減少token消耗
    
    Parameters:
    segment (str): 文本段落
    index (int): 段落索引
    total (int): 總段落數
    emotion (str, optional): 預測情緒
    
    Returns:
    str: 優化後的提示
    """
    base_prompt = f"""音頻逐字稿（第{index+1}/{total}段）：
    {segment}

    請按以下格式回應：
    【評分】
    1. 攻擊性謊言: X%
    2. 矛盾衝突: X%
    3. 明確否認: X%
    4. 尷尬掩蓋: X%
    5. 警覺避談: X%
    6. 猶豫不決: X%
    7. 異常興奮: X%
    8. 邏輯漏洞: X%

    【分析】
    - 關鍵欺騙跡象：(最多3項)
    - 主要情緒特徵：(50字)
    - 邏輯一致性：(是/否)"""
    
    if emotion:
        base_prompt += f"\n- 預測情緒: {emotion}"
    
    return base_prompt

def analyze_single_segment(segment, emotion=None, index=0, total=1):
    """
    分析單個文本段落，包含重試機制
    
    Parameters:
    segment (str): 待分析的文本段落
    emotion (str, optional): 預測情緒
    index (int): 段落索引
    total (int): 總段落數
    
    Returns:
    str: 分析結果
    """
    MAX_ATTEMPTS = 3
    prompt = optimize_analysis_prompt(segment, index, total, emotion)
    print(f"發送的Prompt内容：\n{prompt[:500]}...")  # 截取前500字符观察
    print(f"段落 {index+1} 實際內容: {segment[:100]}...")
    print(f"生成的prompt前100個字符: {prompt[:100]}...")
    
    # 檢查segment是否為空或非字符串 
    if not segment or not isinstance(segment, str) or len(segment.strip()) == 0:
        return f"段落{index+1}分析失敗: 空白或無效段落"
    
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "你是專業的欺騙模式分析助手，你的任務是分析文本中可能存在的欺騙模式。你必須先提供結構化的評分，然後再給出詳細分析。請確保評分部分格式正確，以便程式能夠準確解析。"},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1024
            )
            return response.choices[0].message['content']
        except Exception as e:
            logging.warning(f"段落{index+1}分析嘗試{attempt+1}失敗: {str(e)}")
            if attempt == MAX_ATTEMPTS - 1:
                return f"段落{index+1}分析失敗: {str(e)}"
            time.sleep(2)  # 添加重試間隔

def analyze_segments_iterative(segments, emotion=None):
    """
    使用迭代方式分析所有文本段落，避免遞迴深度問題
    
    Parameters:
    segments (list): 文本段落列表
    emotion (str, optional): 預測情緒
    
    Returns:
    list: 分析結果列表
    """

    for i, segment in enumerate(segments):
        print(f"段落 {i+1} 類型: {type(segment)}, 長度: {len(segment) if isinstance(segment, str) else 'N/A'}")
        if isinstance(segment, str):
            print(f"段落 {i+1} 前50個字符: {segment[:50]}")

    # 初始化必要的變數
    SAFETY_MARGIN = 512
    segments_to_process = [(i, segment) for i, segment in enumerate(segments)]
    total_segments = len(segments)
    results = [None] * total_segments  # 預先分配結果列表空間
    
    # 使用線程池進行並行處理
    with ThreadPoolExecutor(max_workers=3) as executor:  # 降低並行數量以避免API限制
        # 提交初始批次任務
        batch_size = 3  # 使用較小的批次大小
        futures = {}
        
        # 處理所有段落直到完成
        while segments_to_process:
            # 取出一批待處理段落
            current_batch = segments_to_process[:batch_size]
            segments_to_process = segments_to_process[batch_size:]
            
            # 檢查每個段落的token數，必要時進一步分割
            for idx, segment in current_batch:
                prompt = optimize_analysis_prompt(segment, idx, total_segments, emotion)
                total_tokens = count_tokens(prompt) + 500  # 預估回覆長度
                
                if total_tokens > 8192 - SAFETY_MARGIN:
                    # 段落過大，需要再分割
                    logging.info(f"段落{idx+1}token數過多({total_tokens})，進行再次分割")
                    sub_segments = dynamic_segment_transcript(segment, max_tokens=2000)
                    
                    # 將子段落的結果合併為一個
                    sub_results = []
                    for sub_idx, sub_segment in enumerate(sub_segments):
                        sub_prompt = optimize_analysis_prompt(
                            sub_segment, 
                            sub_idx, 
                            len(sub_segments), 
                            emotion
                        )
                        # 檢查子段落大小
                        sub_tokens = count_tokens(sub_prompt) + 500
                        if sub_tokens > 8192 - SAFETY_MARGIN:
                            sub_results.append(f"子段落{sub_idx+1}過大，無法處理")
                            continue
                            
                        # 直接處理子段落
                        future = executor.submit(
                            analyze_single_segment, 
                            sub_segment, 
                            emotion, 
                            sub_idx, 
                            len(sub_segments)
                        )
                        futures[future] = (idx, f"sub_{sub_idx}")
                else:
                    # 段落大小適中，直接提交處理
                    future = executor.submit(
                        analyze_single_segment, 
                        segment, 
                        emotion, 
                        idx, 
                        total_segments
                    )
                    futures[future] = (idx, "main")
            
            # 等待當前批次完成
            # 使用as_completed避免一個慢任務阻塞整個處理流程
            completed_futures = []
            sub_results_by_idx = {}
            
            for future in futures:
                try:
                    result = future.result(timeout=120)  # 設定超時以避免卡住
                    idx, result_type = futures[future]
                    
                    if result_type == "main":
                        # 主段落結果直接保存
                        results[idx] = result
                        completed_futures.append(future)
                    else:
                        # 子段落結果先收集，稍後合併
                        if idx not in sub_results_by_idx:
                            sub_results_by_idx[idx] = []
                        sub_results_by_idx[idx].append(result)
                        completed_futures.append(future)
                except Exception as e:
                    idx, result_type = futures[future]
                    logging.error(f"處理段落{idx+1}時出錯: {str(e)}")
                    if result_type == "main":
                        results[idx] = f"分析錯誤: {str(e)}"
                    completed_futures.append(future)
            
            # 清理已完成的future
            for future in completed_futures:
                futures.pop(future, None)
            
            # 合併子段落結果
            for idx, sub_results in sub_results_by_idx.items():
                combined = f"===== 段落 {idx+1} (分割為{len(sub_results)}個子段落) 分析 =====\n\n"
                for i, result in enumerate(sub_results):
                    combined += f"--- 子段落 {i+1} ---\n{result}\n\n"
                results[idx] = combined
    
    # 檢查是否有未完成的段落
    for i, result in enumerate(results):
        if result is None:
            results[i] = f"段落{i+1}分析未完成"
    
    return results

def parallel_analyze_segments(segments, emotion=None):
    """
    此函數保留但已不使用，由analyze_segments_iterative取代
    僅為向後兼容保留
    """
    return analyze_segments_iterative(segments, emotion)

def integrate_analyses(segment_analyses):
    """
    整合多個段落分析結果
    
    Parameters:
    segment_analyses (list): 段落分析結果列表
    
    Returns:
    str: 整合後的分析結果
    """
    if not segment_analyses:
        return "沒有有效的段落分析結果"
    
    # 從各個段落中提取評分數據
    all_scores = {}
    pattern_names = [
        "攻擊性謊言", "矛盾衝突", "明確否認", "尷尬掩蓋", 
        "警覺避談", "猶豫不決", "異常興奮", "邏輯漏洞"
    ]
    
    # 從每個段落分析中提取評分
    for i, analysis in enumerate(segment_analyses):
        # 跳過錯誤的分析結果
        if isinstance(analysis, str) and analysis.startswith("分析錯誤"):
            continue
            
        for j, pattern in enumerate(pattern_names):
            # 使用正則表達式提取評分
            pattern_regex = f"{j+1}\\. {pattern}: (\\d+)%"
            match = re.search(pattern_regex, analysis)
            if match:
                score = int(match.group(1))
                if pattern in all_scores:
                    all_scores[pattern].append(score)
                else:
                    all_scores[pattern] = [score]
    
    # 計算平均評分
    average_scores = {}
    for pattern, scores in all_scores.items():
        if scores:
            average_scores[pattern] = sum(scores) / len(scores)
    
    # 創建整合後的分析文本
    integrated_text = "段落分析整合結果：\n\n"
    
    # 添加平均評分
    for i, pattern in enumerate(pattern_names):
        score = average_scores.get(pattern, 0)
        integrated_text += f"{i+1}. {pattern}: {score:.1f}%\n"
    
    # 計算詐欺可能性 (所有評分的平均值)
    if average_scores:
        fraud_probability = sum(average_scores.values()) / len(average_scores)
        integrated_text += f"9. 詐欺可能性: {fraud_probability:.1f}%\n\n"
    else:
        integrated_text += "9. 詐欺可能性: 無法計算\n\n"
    
    # 添加每個段落的分析
    integrated_text += "各段落詳細分析：\n\n"
    for i, analysis in enumerate(segment_analyses):
        integrated_text += f"===== 段落 {i+1} 分析 =====\n{analysis}\n\n"
    
    return integrated_text

def format_final_report(integrated_analysis):
    """
    將整合後的分析格式化為最終報告
    
    Parameters:
    integrated_analysis (str): 整合後的分析結果
    
    Returns:
    str: 格式化後的最終報告
    """
    # 提取詐欺可能性
    fraud_prob_match = re.search(r"9\. 詐欺可能性: (\d+\.\d+)%", integrated_analysis)
    fraud_probability = float(fraud_prob_match.group(1)) if fraud_prob_match else 0
    
    # 提取各項欺騙模式評分
    pattern_scores = {}
    pattern_names = [
        "攻擊性謊言", "矛盾衝突", "明確否認", "尷尬掩蓋", 
        "警覺避談", "猶豫不決", "異常興奮", "邏輯漏洞"
    ]
    
    for i, pattern in enumerate(pattern_names):
        score_match = re.search(f"{i+1}\\. {pattern}: (\\d+\\.\\d+)%", integrated_analysis)
        if score_match:
            pattern_scores[pattern] = float(score_match.group(1))
    
    # 确定欺骗風險級別
    risk_level = "低風險"
    if fraud_probability >= 75:
        risk_level = "極高風險"
    elif fraud_probability >= 50:
        risk_level = "高風險"
    elif fraud_probability >= 30:
        risk_level = "中等風險"
    
    # 主要欺騙模式（評分最高的前3個）
    sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    primary_patterns = sorted_patterns[:3]
    
    # 格式化報告
    report = f"""# 欺騙分析總結報告

## 風險評估
- **風險分數**: {fraud_probability:.1f}/100
- **風險級別**: {risk_level}
- **主要欺騙模式**: {', '.join([f"{p[0]} ({p[1]:.1f}%)" for p in primary_patterns])}

## 詳細評分
"""
    
    for i, pattern in enumerate(pattern_names):
        score = pattern_scores.get(pattern, 0)
        report += f"- {pattern}: {score:.1f}%\n"
    
    report += f"\n## 段落分析整合結果\n{integrated_analysis}"
    
    return report

# 以下函數保持不變
def get_emotion_explanation(emotion, parameters, audio_file_path, advanced_features, fraud_report=None):
    """
    獲取情緒分析的解釋，通過分批處理解決token限制問題
    
    Parameters:
    emotion (str): 預測的情緒
    parameters (str): 情緒參數字符串
    audio_file_path (str): 音訊檔案路徑
    advanced_features (dict): 進階音訊特徵
    fraud_report (dict, optional): 欺騙檢測報告
    
    Returns:
    str: 情緒分析的解釋文字
    """
    try:
        # 獲取文字轉錄
        transcription = transcribe_audio(audio_file_path)
        
        # 檢查transcription的token數量
        # trans_tokens = count_tokens(transcription)
        
        # 將分析分成多個部分，降低每次請求的token數
        # 1. 基本情緒分析
        basic_emotion_prompt = f"""
        音頻逐字稿："{transcription}"
        
        情緒預測結果為 {emotion}，參數為 {parameters}。
        
        請根據音頻逐字稿的內容和語氣變化，簡要分析為什麼會得出這個情緒預測結果。
        特別關注說話者使用的詞語和語氣轉變。請用繁體中文回答。
        """
        
        # 2. 語音特徵分析
        voice_features_prompt = f"""
        基於以下語音特徵，分析說話者的情緒狀態：
        
        語音特徵分析（使用CREPE算法）：
        - 語速: {float(advanced_features['speech_rate'].get('mean_speech_rate', 0.0)):.2f} 音節/秒
        - 平均音高: {float(advanced_features['pitch']['mean_pitch']):.2f} Hz
        - 音高標準差: {float(advanced_features['pitch']['std_pitch']):.2f} Hz
        - 音高變化率: {float(advanced_features['pitch']['pitch_change_rate']):.2f}
        - 音高不穩定性: {float(advanced_features['pitch']['pitch_instability']):.2f}
        - 音高趨勢: {float(advanced_features['pitch']['pitch_trend']):.2f}
        - 有聲段比例: {float(advanced_features['pitch']['voiced_ratio']):.2f}
        - 音量變化: {float(advanced_features['volume']['std_volume']):.4f}
        
        前面的分析指出說話者情緒為"{emotion}"。這些語音特徵如何支持或反駁這一結論？請用繁體中文回答。
        """
        
        # 3. 欺騙風險分析（如果有fraud_report）
        fraud_analysis_prompt = ""
        if fraud_report:
            fraud_analysis_prompt = f"""
            根據前面的分析和以下欺騙風險評估結果：
            - 風險分數：{fraud_report['risk_score']:.1f}/100
            - 風險級別：{fraud_report['risk_level']}
            - 主要欺騙模式：{', '.join([p[0] for p in fraud_report['primary_patterns']])}
            
            分析該音頻中可能反映出欺騙行為的跡象。請聚焦於以下五個方面，並在回答中包含一個明確的詐欺可能性百分比，格式為'X%'。
            """
        
        # 初始化完整分析結果
        full_analysis = ""
        
        # 依次發送API請求並整合結果
        # 1. 基本情緒分析
        response1 = openai.ChatCompletion.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": "你是一個專業的情緒分析助手，專門分析中文語音內容和語氣。"},
                {"role": "user", "content": basic_emotion_prompt}
            ]
        )
        
        basic_analysis = response1.choices[0].message['content']
        full_analysis += basic_analysis + "\n\n"
        
        # 短暫延遲避免API限制
        time.sleep(1)
        
        # 2. 語音特徵分析
        response2 = openai.ChatCompletion.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": "你是一個專業的語音分析專家，能夠從語音特徵中分析情緒。"},
                {"role": "user", "content": voice_features_prompt}
            ]
        )
        
        features_analysis = response2.choices[0].message['content']
        full_analysis += features_analysis + "\n\n"
        
        # 3. 欺騙風險分析（如果有）
        if fraud_report and fraud_analysis_prompt:
            # 再次短暫延遲
            time.sleep(1)
            
            response3 = openai.ChatCompletion.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "你是一個專門分析欺騙行為的專家，能從語音和文字中識別欺騙跡象，並判斷詐欺可能性。"},
                    {"role": "user", "content": fraud_analysis_prompt}
                ]
            )
            
            fraud_analysis = response3.choices[0].message['content']
            full_analysis += fraud_analysis
        
        return full_analysis
        
    except Exception as e:
        logging.error(f"API調用錯誤: {str(e)}")
        # 提供基本回應而不是完全失敗
        return f"情緒分析暫時無法完成: {str(e)}。請稍後再試。預測情緒為: {emotion}"