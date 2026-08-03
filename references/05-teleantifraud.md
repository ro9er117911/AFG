# TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection

**Citation**: Ma, Z., Wang, P., Huang, M., Wang, J., Wu, K., Lv, X., Pang, Y., Yang, Y., Tang, W., & Kang, Y. (2025). TeleAntiFraud-28k: An audio-text slow-thinking dataset for telecom fraud detection. *Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025)*.

**連結**: https://doi.org/10.1145/3746027.3755835　｜　arXiv：https://arxiv.org/abs/2503.24115

## 重點摘要

發表於 2025 年 ACM Multimedia（多媒體領域頂尖會議之一）的最新研究，建立了第一個開源的「音訊+文字」
雙模態電信詐騙偵測資料集，包含 28,511 筆經過嚴格處理、附有詳細詐騙推理標註的語音-文字配對資料。
資料集用三種方式建構：(1) 用 ASR 轉錄真實通話錄音、以 TTS 重新生成語音以保護隱私但維持真實性，
(2) 用 LLM 做語意擴增以涵蓋更多場景，(3) 用多代理對抗式合成模擬新興詐騙手法。同時涵蓋三個分類任務
（場景辨識、詐騙偵測、詐騙類型分類），並釋出一個在真實/合成混合資料上微調過的生產可用模型，以及
完整的資料處理框架供社群擴充。

## 對本專案的啟示

這篇論文的系統架構——**音訊 + 文字雙模態融合做電信詐騙偵測**——跟 `antifraud_v2` 的整體設計理念
（`audio_processing.py` 抽聲學特徵 + GPT-4 做文字分析 + `multimodal_fusion.py` 融合）幾乎是同一個
問題的正規學術版本，是目前找到與本專案應用場景最貼近的高品質、可對照範例。差異在於方法論的嚴謹度：
TeleAntiFraud-28k 有 28,511 筆經過標註驗證的真實/合成混合資料、明確定義的分類任務與評測基準，
並公開了訓練資料與框架供他人重現；而 `antifraud_v2` 目前的 `fraud_detection.py` 是一組沒有經過
資料集驗證的手動門檻規則。若本專案未來想認真降低誤報率，這是一個具體可參考的方向：先建立（或採用）
一個像 TeleAntiFraud-28k 這樣有標註的中文電信詐騙語料，才有辦法真正校準/訓練/評測現有的規則式
判斷邏輯，而不是繼續依賴手動調整的魔術數字門檻。
