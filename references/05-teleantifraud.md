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

## 資料集細節（供實際採用時參考）

- **語言**：原生中文（Mandarin），不需要跨語言翻譯這一關——這點跟本專案介面/語料都是繁體中文一致，
  是目前 `references/` 裡少數原生中文的資料集參考。
- **授權**：程式碼倉庫（`github.com/JimmyMa99/TeleAntiFraud`）為 Apache License 2.0，允許商用、
  需保留授權聲明；資料集本身另外託管於 Hugging Face（`JimmyMa99/TeleAntiFraud`）與 ModelScope，
  實際採用前應個別確認資料集頁面上的授權條款，不能只看程式碼倉庫的 LICENSE。
- **音訊品質——重要限制**：音訊是用 ChatTTS 從匿名化後的 ASR 逐字稿重新合成，屬於**乾淨、寬頻
  （clean/wideband）合成語音**，並未模擬真實電話頻寬（300–3400Hz band-limited）或電信編碼失真。
  **不適合用來驗證/校準本專案窄頻電話錄音的聲學處理**（見 `antifraud_v3/audio/quality.py` 的
  `detect_bandwidth`）；它的價值在文字/情境內容與分類任務設計，不在聲學真實度。
- **雙軌設計**：caller／callee 分為兩條獨立音軌，這點是本專案目前完全沒有的能力（無語者分離／
  diarization，見 `docs/DESIGN.md` 已知缺口），可作為未來若要做語者分離時的架構參考，但本身不是
  可以直接搬過來用的元件。
- **詐騙類型分類法（7 類）**：Investment Fraud（投資詐騙）、Phishing Fraud（網路釣魚詐騙）、
  Identity Theft（身分冒用）、Lottery Fraud（中獎摸彩詐騙）、Banking Fraud（銀行詐騙）、
  Extortion Fraud（勒索詐騙）、Customer Service Fraud（客服詐騙）。這是一套「詐騙話術屬於哪種
  類型」的分類，跟 `antifraud_v3` 現有的 4 階段 kill-chain（接觸→催促→隔絕→取款，判斷「詐騙進行
  到哪個階段」）是兩個不同維度，互補不衝突——已採用為 `reasoning/schemas.py` 的
  `FraudTypeClassification` 分類法依據。
