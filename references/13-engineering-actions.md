# 工程實作依據與待辦

調研日期：2026-08-11
用途：把文獻結論對應到具體檔案與修改。依阻斷性排序。

---

## P0：商用阻斷項

### 1. Line 1 模型授權不可商用

**現況**：`antifraud_v3/detectors/deepfake_voice.py` 使用
`nii-yamagishilab/xls-r-2b-anti-deepfake`，**CC BY-NC-SA 4.0**。
docstring 自述「non-commercial, fine for this personal tool」。
它是 `reasoning/fusion.py` 中優先序最高的判決來源。

**為何是 P0**：買方為 TapPay。商用即觸法，且擋在最關鍵那條線上。

**可商用替代（文獻調研結果）**：

| 選項 | 授權 | 8kHz 支援 | 備註 |
|---|---|---|---|
| `clovaai/aasist` | **MIT**（已確認 LICENSE） | 否，原生 16kHz | 唯一確認可商用者，需自行 resample／fine-tune |
| XLS-R + SLS | **未確認**，商用前須查 LICENSE | 於 ASVspoof 2021 DF track 驗證過（EER 1.92%），最貼近壓縮語音場景 | 授權未明前不可用 |
| RawNet2 baseline | **未確認** | 原生 16kHz | 授權未明前不可用 |
| AASIST3 | CC BY-NC-ND 4.0 | — | **明確禁止商用，排除** |

**關鍵事實**：**沒有任何公開 checkpoint 原生針對電話頻寬訓練。**
無論選哪個都需要自有語料 fine-tune。

---

## P1：正確性與可信度

### 2. `fake_score` 門檻未校準

**現況**：`antifraud_v3/config.py`

```python
DEEPFAKE_FAKE_SCORE_THRESHOLD = float(os.getenv("DEEPFAKE_FAKE_SCORE_THRESHOLD", "0.85"))
```

程式碼已自標 `UNCALIBRATED` 並說明原因——誠實，但問題仍在。

**文獻警訊**（arXiv:2606.21584「When EER Hides Deployment Failure」）：
把來源域門檻直接套到新場景，**78.7% 的真人語音被誤判為假音，HTER 39.5%**，
即使該場景 EER 讀數僅 11.2%。

**待辦**：
- 建立電話頻寬驗證集（8kHz + G.711/Opus + 封包損失 + **真人語音**）
- 以 Platt scaling 或 isotonic regression 做 post-hoc calibration
- 監控 **HTER 與 FRR/FAR pair**，不可只看 EER
- 門檻取保守側：寧可漏放，不要大量誤殺真人來電
  （誤報會讓使用者養成忽略警示的習慣，傷害大於漏放）

### 3. eval 語料無法驗證鑑別力

**現況**：`eval/test_clips/` 全為 edge-tts 合成語音，
每支皆被正確判為 fake——可驗證 pipeline 接通，**不能驗證 real/fake 鑑別品質**。
（`detectors/deepfake_voice.py` docstring 已載明此限制。）

**待辦**：補真人語音錄音 + 電話頻寬語料。
**在此之前，對外不得引用現有 eval 結果作為準確率數字。**

### 4. UI 裸機率呈現不合理

**現況**：UI 顯示「AI 合成／複製語音機率 0%」這類連續百分比，
即模型原始 softmax 輸出。

**問題**：在 8kHz + 未見攻擊情境下，合理預期 EER 為 15-35%（外推值）。
這個精度撐不起「87% 機率是假音」的呈現方式，等於把未校準的 raw score
包裝成看似精確的數字。

**建議**：
- 改用**分級標籤**（低／中／高三檔），加註「輔助參考，非鑑識結論」
- 加**前置可靠度指標**——音訊時長是否足夠、訊噪比、是否為窄頻通話。
  比統計信賴區間對使用者更有用（商用系統對 <1.5 秒音檔常直接拒判）

---

## P2：偵測品質

### 5. 雷達圖八維：補強證據最硬的缺口

**現況**：八維為工程師自訂，證據強度差異極大。

| 評級 | 維度 | 處置 |
|---|---|---|
| 相對紮實 | 警覺避談、矛盾衝突 | 保留 |
| 間接／弱 | 猶豫不決、明確否認、攻擊性語氣 | 保留但降權，標註效果量偏小 |
| **近乎憑空發明** | **尷尬掩蓋** | 查無對應構念，建議併入警覺避談 |
| **定義含糊** | **邏輯漏洞** | 須先釐清量的是詐騙者話術還是受害者陳述——兩者理論基礎完全不同 |
| 對象不明 | 異常興奮 | 原文獻談**受害者**情緒喚起，非詐騙者語氣。須釐清量測對象 |

**最高性價比的補強**——文獻支持最強但目前完全缺席：

| 應補維度 | 文獻依據 | 實作成本 |
|---|---|---|
| **Urgency / Scarcity** | 釣魚信出現率 43% vs 正常信 5% | 低——規則式偵測「限時」「馬上」「最後機會」即可先跑基線 |
| **Isolation（孤立受害者）** | PsyScam / PreScam 常見類別 | 低——「不要跟家人說」「不要掛電話」等句型。**假檢警核心話術** |
| **Authority（權威冒用）** | Modic & Lea 四因子之一 | 已部分存在於「身分冒用」欄位，雷達圖獨立列出即可 |
| **Consistency pressure** | Modic & Lea 四因子之一 | 中——「你剛剛已經同意了」 |

理論校準建議走 Modic & Lea (2013) 與 Modic, Anderson & Palomäki (2018,
PLOS ONE, DOI 10.1371/journal.pone.0194119) 的四因子：
authority influence、social influence、self-control、need for consistency。
比 Cialdini 六原則更貼近「哪些話術最能預測受害」。

### 6. 對抗性風險：話術可被 LLM 改寫繞過

**文獻**：Li et al. (2025, arXiv:2507.16291) 證明用 GPT-4o 改寫詐騙逐字稿
（保留語意、換句話說），分類器準確率**下降 30.96%**。

詐騙集團同樣在用 LLM。這是持續性風險，不是一次性修補。
Line 2 若偏向關鍵詞或固定句式比對，半年內會被追上。

**待辦**：建立改寫攻擊的回歸測試——用 LLM 改寫既有測試語料，
測量準確率衰減，作為長期監控指標。

---

## P3：資料與長期投資

### 7. 台灣在地語料是缺口，也是護城河

文獻調研確認：
- **查無台灣華語／台語深偽偵測資料集**（ADD 系列聚焦中國大陸普通話；
  Taiwan Tongues 是 ASR/合成語料，非深偽標註）
- **查無公開的中文詐騙通話逐字稿語料**。可用者為韓語 KorCCVi
  （2,927 筆）或來源不透明的中國大陸公安平台資料

現有可用公開資料集：

| 資料集 | 語言 | 規模 | 用途 |
|---|---|---|---|
| ASVspoof 2021 DF | 多語 | — | codec 壓縮情境，最接近電話場景 |
| In-the-Wild | 英 | 20.8h 真 / 17.2h 假 | 跨域泛化驗證 |
| MLAAD | 38-51 語 | 687.4h | 跨語言遷移 |
| KorCCVi v2 | 韓 | 2,927 筆逐字稿 | 方法論參考（非中文） |
| TeleAntiFraud-28k | 中（需確認） | 28,511 筆 | Line 2 模型的訓練來源 |
| SUSAS | 英 | 32 講者 / 16,000+ utterances | 壓力語音特徵管線校準 |

---

## 附錄：聲學層的正確定位（不需修改，但要能解釋）

`reasoning/fusion.py` 目前**不讓聲學特徵進入判決**——只有 Line 1 的
`fake_score` 與 Line 2 的 `AntiFraudQwenResult` 決定 `is_fraud` / `risk_level`。

**這是對的，不要改。** 四條獨立證據否定聲學測謊
（DePaulo 2003 d≈0.25；NRC 2003；Damphousse 2007 約等同擲硬幣；
ComParE 2016 baseline UAR 45.1% 低於隨機）。

各特徵證據強度（**不可等權**）：
- **F0**：SMD=0.55，但校正發表偏誤後 0.17 **不顯著**
- **jitter / shimmer / HNR**：38 篇系統性回顧顯示**跨研究無一致方向**
- **停頓 / 語速**：僅「認知負荷」情境方向一致

**within-call baseline 設計方向有文獻支持**（speaker normalization 優於絕對閾值，
單一研究相對提升達 20%），但「通話開頭數秒」的窗口穩定性文獻未驗證，
須自行實測——這是目前 `audio/features.py` 基準值做法的已知未驗證假設。

工具選擇無需更動：Praat／parselmouth 是業界標準，eGeMAPS 是已驗證的標準參數集。
