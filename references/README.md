# 語音詐欺偵測 / 語音分析文獻回顧（2023–2026）

為 `antifraud_v2`（通話音訊詐欺偵測系統）整理的精選文獻。每篇都已逐一查證真實出處
（DOI/IEEE Xplore/ACM DL/arXiv 連結），不含虛構條目。

- **1–10**：單篇文獻摘要，**限定 2023–2026 年發表**，聚焦頂尖會議/期刊，數量精不求多。
- **11–13**：2026-08-11 新增的跨領域綜合報告。為涵蓋反面證據，**刻意納入較早的奠基文獻**
  （如 DePaulo 2003、NRC 2003、Damphousse 2007）——否定聲學測謊的關鍵證據多在 2003–2007 年
  產生，這些是該領域至今未被推翻的定論，不適用 2023–2026 的時間限制。

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

## 11–13：綜合調研報告（2026-08-11）

與 1–10 的性質不同：前十篇是**單篇文獻摘要**，這三份是**跨領域綜合報告**，
補的是既有十篇沒涵蓋的三塊——反面證據（否定聲學測謊的文獻）、
競品與專利地景、以及授權與校準的工程阻斷項。

| # | 文件 | 用途 | 核心結論 |
|---|------|------|---------|
| [11. TapPay 技術說明](11-tappay-technical-defense.md) | 買方盡調預備 | 四條獨立證據否定聲學測謊；本專案「聲學不進判決」是可辯護性賣點；Nemesysco/LVA 事件須主動切割 |
| [12. Prior art 與 novelty](12-prior-art-novelty.md) | 專利／論文寫作 | 聲學→情緒→詐欺路徑已被 US6638217B1、US8965770B2 佔據；可主張的縫隙在電話頻寬、門檻校準、台灣在地語料 |
| [13. 工程待辦](13-engineering-actions.md) | 實作依據 | P0：Line 1 模型 CC BY-NC-SA 不可商用；P1：0.85 門檻未校準、裸機率呈現不合理；P2：雷達圖缺 urgency/isolation |

**三份文件共同的關鍵引用**（不在 1–10 之列）：
DePaulo et al. (2003) DOI 10.1037/0033-2909.129.1.74（欺騙線索統合分析，d≈0.25）、
NRC (2003)《The Polygraph and Lie Detection》、
Damphousse et al. (2007) DOI 10.3886/ICPSR20625.v1（VSA 實地驗證約等同擲硬幣）、
Interspeech 2016 ComParE Deception DOI 10.21437/interspeech.2016-129（Deception baseline UAR **68.3%**，隨機 50%；先前誤植的 45.1% 屬 Native Language 子挑戰）、
SVC 2025 Multimodal Deception Challenge arXiv:2508.04129（跨域多模態冠軍 62.44%，未報告純聲學數字）、
Schewski et al. (2025) PLOS One DOI 10.1371/journal.pone.0328833（jitter/shimmer/HNR 無一致方向）、
arXiv:2606.21584（EER 掩蓋部署失敗：78.7% 真人語音被誤殺）、
arXiv:2507.16291（LLM 改寫話術使分類器準確率降 30.96%）。

⚠️ 三份文件中標註「未驗證」的識別碼，投稿或送件前須人工核對原始來源。

## 閱讀建議順序

**診斷現有系統**：想先理解「本專案的情緒模型出處與跨語言適用性問題」：讀 1 → 2。
想理解「音高/壓力這類聲學線索目前科學證據到什麼程度、能不能用固定門檻」：讀 3。
想理解「這系統目前沒防到的另一種詐騙（AI 換聲）」：讀 4。
想理解「同樣問題，正規、有資料集驗證過的做法長怎樣」：讀 5 → 6。

**規劃重寫**：想理解「融合架構該怎麼重寫」：讀 7。想理解「沒有標註資料集時偵測邏輯該怎麼做」：
讀 8。想理解「要不要做到通話中即時示警」：讀 9 → 10。

音訊分析誤報率的根因調查與具體修復，見 `../antifraud_v2/FALSE_POSITIVE_REVIEW.md`。
