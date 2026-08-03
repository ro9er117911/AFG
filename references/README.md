# 語音詐欺偵測 / 語音分析文獻回顧（2023–2026）

為 `antifraud_v2`（通話音訊詐欺偵測系統）整理的精選文獻，**全部限定在 2023–2026 年發表**，
聚焦頂尖會議/期刊，數量精不求多。每篇都已逐一查證真實出處（DOI/IEEE Xplore/ACM DL/arXiv 連結），
不含虛構條目、不含過時文獻。

前 6 篇（1–6）是為了**理解、診斷現有系統**而選的：現有 TIMNet 模型的出處與適用性侷限、聲學線索的
科學證據強度、現有系統完全沒涵蓋的風險缺口。後 4 篇（7–10）是使用者決定**完全重寫**整個
`antifraud_v2` 之後新增的，聚焦「重寫時具體要照哪個架構/方法做」，而非診斷現有 bug。

## 1–6：診斷現有系統

| # | 文獻 | Venue / 年 | 一句話關聯性 |
|---|------|-----------|-------------|
| [1. TIM-Net 原始論文](01-timnet.md) | Ye et al. | ICASSP 2023 | 本專案 TIMNet 情緒模型的出處，訓練語料不含電話通話、不含中文對話 |
| [2. 跨語言 SER 領域適應](02-cross-lingual-ser.md) | Upadhyay, Martinez-Lucas, Katz, Busso, Lee | *IEEE Trans. Affective Computing* 16(3), 2025 | 2025 年最新研究證實情緒辨識模型跨語言套用會有明顯效能落差，需要專門的領域適應技術 |
| [3. F0/音高壓力生物標記統合分析](03-f0-stress-biomarker.md) | Veiga, Almeida, Uchida, Cordeiro | *Stress and Health*, 2025 | 2025 年統合分析：音高確與壓力顯著相關，但效應異質性高、有出版偏誤，不支持單一固定門檻判斷 |
| [4. ASVspoof 5](04-asvspoof5.md) | Wang, Delgado, Tak, et al. | *Computer Speech & Language* 95, 2025 | 最新一代語音防偽/深偽偵測基準——本專案目前完全未涵蓋的 AI 合成語音詐騙風險缺口 |
| [5. TeleAntiFraud-28k](05-teleantifraud.md) | Ma, Wang, Huang, et al. | ACM Multimedia 2025 | 音訊+文字雙模態電信詐騙偵測資料集，架構與本專案最貼近的高品質對照範例 |
| [6. 多模態欺騙偵測綜述](06-multimodal-deception-survey.md) | Zhang, Lin, Huang, et al. | *Machine Intelligence Research* 23, 2026 | 2026 年最新完整回顧，涵蓋公開資料集、評測指標與架構演變，宏觀視角 |

## 7–10：為「完全重寫」打底

| # | 文獻 | Venue / 年 | 一句話關聯性 |
|---|------|-----------|-------------|
| [7. SAFE-QAQ](07-safe-qaq.md) | Wang, Ma, Dai, et al. | arXiv:2601.01392, 2026 | TeleAntiFraud 團隊後續生產系統論文（每天處理 7 萬通電話），`multimodal_fusion.py` 該參考的端對端音訊+文字融合＋RL 推理架構，而非固定權重加權平均 |
| [8. 演化式詐騙電話偵測（LLM 輔助專家規則）](08-evolving-scam-calls-llm-rules.md) | Ma, Su, Huang, Kai | EMNLP 2025 Findings | 直接處理「沒有標註資料集」這個核心限制，示範用 LLM 輔助專家規則取代手動門檻，是 `fraud_detection.py` 重寫方向最直接的參考 |
| [9. PreScam 基準](09-prescam-benchmark.md) | Sun, Ma, Li, et al. | arXiv:2605.12243, 2026 | 「通話講到一半該不該示警」的即時判斷基準，對應重寫若要做到通話中即時預警而非事後報告 |
| [10. 即時詐騙示警使用者研究](10-chi-realtime-scam-warning.md) | Shen, Yan, Zhang, et al. | CHI EA 2025 | 唯一有真人受試者測試「即時示警」實際體驗的研究，補上示警時機/UX 設計這塊 |

## 閱讀建議順序

**診斷現有系統**：想先理解「本專案的情緒模型出處與跨語言適用性問題」：讀 1 → 2。
想理解「音高/壓力這類聲學線索目前科學證據到什麼程度、能不能用固定門檻」：讀 3。
想理解「這系統目前沒防到的另一種詐騙（AI 換聲）」：讀 4。
想理解「同樣問題，正規、有資料集驗證過的做法長怎樣」：讀 5 → 6。

**規劃重寫**：想理解「融合架構該怎麼重寫」：讀 7。想理解「沒有標註資料集時偵測邏輯該怎麼做」：
讀 8。想理解「要不要做到通話中即時示警」：讀 9 → 10。

音訊分析誤報率的根因調查與具體修復，見 `../antifraud_v2/FALSE_POSITIVE_REVIEW.md`。
