# 音訊分析誤報（False Positive）根因調查與修復

本文件記錄一次針對「誤報率偏高」疑慮的根因調查，範圍限定在**音訊分析**這端
（`audio_processing.py`、`fraud_detection.py`、`multimodal_fusion.py`、`model_handling.py`、
`main.py`），不含 GPT-4 文字分析那端。方法論背景與延伸閱讀見 `../references/README.md`。

**修復原則**：只修「確定的程式錯誤」（API 呼叫錯誤、死代碼、自我參照邏輯、暫存檔命名碰撞），
**不調整** `fraud_detection.py` 裡任何手動調過的門檻/權重魔術數字（如 jitter 1.04%、pitch baseline
100Hz、pause baseline 0.8）——沒有真實標註的通話資料集可以拿來校準，這次調整風險大於幫助。

---

## 發現 1：顫抖/jitter/shimmer/HNR 分析是死的，每次都回傳同一組假數字

**位置**：`audio_processing.py:429`（修復前）

`pitch.to_point_process()` 這個方法在目前安裝的 parselmouth 版本裡根本不存在——已用
`hasattr()` 實際檢查過 `Pitch.to_point_process`、`Sound.to_point_process`、`Sound.get_jitter`、
`Sound.get_shimmer` 全部回傳 `False`。所以這行程式碼**每次都必定拋出 `AttributeError`**，被
`audio_processing.py:444-454`（修復前）的 `except` 接住後回傳固定值：jitter=1.00%、shimmer=3.00%、
HNR=15.00dB，**跟音檔內容完全無關**。

其中 HNR=15（< 20 的門檻）每次都會讓 `"尷尬掩蓋"`（`fraud_detection.py:669-676`）與
`"猶豫不決"`（`fraud_detection.py:820-828`）兩個詐欺模式各拿到固定的 +0.45 / +0.30 加分——不管
說話者實際狀態如何。這是目前找到**最直接、最確定**的誤報來源，因為它是 100% 必發生的系統性偏移，
不是機率性的。

**修復**：改用檔案本來就已 import 的 `parselmouth.praat.call()` 介面（Praat script command
語法）取得 PointProcess、Jitter、Shimmer，HNR 沿用原本就正確、但因為前面先炸掉而從沒被執行到的
`sound.to_harmonicity_cc()`。順手移除同一函式內、確認全專案沒有任何地方讀取的死代碼
`fraud_risk_score`/`risk_level` 欄位。

**修復前後對比**（合成音檔，非真實通話錄音）：

| 音檔 | jitter_local | shimmer_local | HNR |
|------|-------------|---------------|-----|
| 修復前（任何音檔） | 1.00%（固定） | 3.00%（固定） | 15.00dB（固定） |
| clipA（低頻、平穩） | 0.46% | 2.41% | 20.55dB |
| clipB（高頻、有雜訊調變） | 1.71% | 7.88% | 8.24dB |

兩個內容明顯不同的音檔現在會算出明顯不同的數字，且都不等於修復前那組寫死的常數。

---

## 發現 2：`self.baseline` 是自己跟自己比較，3 個「相對基準」指標永遠算出 0% 差異

**位置**：`fraud_detection.py:259-260`（修復前，呼叫處）、`fraud_detection.py:212-244`（修復前，
`set_baseline` 定義）

`classify_patterns()` 在沒有基線時，直接拿「這通電話自己的特徵」當「基準線」
（`self.set_baseline(audio_features, emotion_probs)`，用的正是同一個 `audio_features`）。
確認影響到 3 個指標：

- `_evaluate_explicit_denial` 的「音高增加」（`fraud_detection.py:508-510` 修復前）
- `_evaluate_alert_avoidance` 的「音量下降」（`fraud_detection.py:758-760` 修復前）
- `_evaluate_abnormal_excitement` 的「音量增加」（`fraud_detection.py:943-945` 修復前）

這三個都是拿 `audio_features['pitch']['mean_pitch']` / `audio_features['volume']['mean_volume']`
同時當「基準」也當「當前值」，差異永遠是精確的 0%。

**這個 bug 本身不會造成誤報**（0% 差異永遠不會超過任何正門檻，`if` 分支從未觸發，對分數的實際貢獻
本來就一直是 0）——但它是死代碼、且看起來像是「有在比較行為變化」的功能，容易誤導維護者。已明確標記
「單次分析沒有獨立於本次通話之外的基準可比較」並直接跳過這 3 個指標，**不影響現有分數輸出**（純
correctness/可讀性修復，經確認為分數中性變更）。

**保留不動的部分**：`set_baseline()` 裡「平均音高」「語速」欄位仍用來調整
`明確否認`/`異常興奮`/`警覺避談` 的門檻（`adjust_thresholds_based_on_baseline`，
`fraud_detection.py:286-317`），以及「語速增加百分比」這 3 個指標（比較整通電話平均語速
vs. 第一個約 2 秒窗口的語速）——這兩塊**不是死代碼**，會實際影響分數，屬於「設計選擇的品質好壞」
而非「明確錯誤」，這次不動。同時移除了 `set_baseline` 裡永遠不會被執行到的指數平滑分支（因為
`classify_patterns()` 每次呼叫都會拿到一個全新的 `FraudPatternClassifier` 實例，`self.baseline`
一定從 `None` 開始）。

---

## 發現 3：情緒分析只看整通電話的前 4 秒

**位置**：`audio_processing.py:11,30-35`（`get_mfcc` 預設 `duration=4` 且直接截斷）、
`model_handling.py:72`（修復前，`predict_emotion` 未覆寫這個參數）

任何超過 4 秒的通話，其情緒特徵只反映開場白（通常是打招呼、自我介紹），卻被當成**整通電話唯一**
的情緒依據，套進全部 8 個詐欺模式裡跟 emotion 相關的門檻比對（`main.py:294,308`）。

**修復**：`predict_emotion()` 改成在整段音訊上以固定步長（預設每 2 秒）滑動取多個 4 秒窗口分別
預測，再平均各窗口的 softmax 機率。對 4 秒以內的短音檔，行為與修復前完全相同（單一窗口）。
`get_mfcc()` 新增 `offset` 參數，改用 `librosa.load(..., offset=offset, duration=duration)`
直接讀取音檔中對應時間段，而不是整段讀入後再截斷。

**修復前後對比**（合成 12 秒音檔，前 6 秒為低頻平穩訊號，後 6 秒為高頻、有調變雜訊的訊號，
模擬「開場平靜、後段情緒明顯不同」的通話）：

| 情境 | 主要預測情緒 | 機率分佈（節錄） |
|------|-------------|-----------------|
| 修復前（只看前 4 秒） | sad | sad=98.7%, happy=1.2%，幾乎忽略後 8 秒 |
| 修復後（滑動窗口平均） | fear | fear=56.3%, sad=40.4%，反映前後兩段內容都有貢獻 |

這是這次調查中**唯一會實際改變分析輸出**的修復（其餘都是分數中性的正確性修復），因為它改變了
情緒特徵的計算方式本身，而非單純修掉一個必然出錯的 bug。

---

## 發現 4：即時錄音的暫存檔路徑會互相碰撞，快取可能吃到別通電話的結果

**位置**：`main.py:249,261`（修復前）、`audio_processing.py:51`（`@st.cache_data(ttl=3600)`）

`st.audio_input`（即時錄音模式）回傳的音檔物件通常沒有 `.name` 屬性，`audio_name` 會固定塌縮成
常數字串 `"recording.wav"`，`temp_file_path` 因此在同一台機器上對不同錄音是**同一條路徑**。而
`analyze_speech_features` 是照**路徑字串**（不是音檔內容）快取，`ttl=3600` 期間，第二通完全不同
的即時錄音有機會直接吃到前一通的分析結果，導致分析結果跟實際講話內容脫鉤（誤報或漏報都有可能，
取決於誰先被快取）。

**修復**：暫存檔名改用音檔**內容**的 SHA-1 雜湊（取前 16 碼）組成，而不是可能塌縮成固定值的
`audio_name`。不同內容的錄音必定產生不同的雜湊值、不同的路徑，`st.cache_data` 才不會誤用到不相干
的舊結果；相同內容（例如重複上傳同一個檔案）仍會命中快取，這是預期且正確的行為。

**驗證**：兩段不同 byte 內容的測試資料，SHA-1 雜湊前 16 碼確認不同（`c160540530305eb1` vs.
`c13417ea447e5dc8`），路徑不會碰撞。

---

## 綜合驗證

用先前建立的 synthetic-audio smoke test（直接呼叫 `analyze_speech_features` → `predict_emotion`
→ `detect_fraud_patterns` → `integrate_audio_text_analysis`，不經 Streamlit UI）重新跑過一次
`test.wav`（4 秒純音調合成音檔）：

| | risk_score | risk_level |
|---|-----------|-----------|
| 修復前 | 46.16 | 中風險 |
| 修復後 | 57.98 | 中風險 |

**分數上升不代表修復無效或方向錯誤**：這段測試音檔是一個近乎理想的純週期性正弦波，修復後量測到
真實的 HNR≈53dB（訊噪比極高、幾乎沒有雜訊，符合合成純音調的物理特性），因此原本必發的
「尷尬掩蓋/猶豫不決」HNR 加分（HNR=15 恆小於門檻 20）消失了；但同時真實量測到的 shimmer_local
≈3.87%（前為固定 3.00%）剛好超過 `"矛盾衝突"` 的 3.81% 門檻，貢獻了新的一小筆分數。這正是修復的
意義所在——**分數現在真的反映音檔的實際聲學特性，而不是每次都輸出同一組人造的假訊號**；對這個
特定的合成測試音檔而言恰好淨增加，但對真實、正常語速語調的說話者而言，拿掉「每通電話必觸發」的
系統性偏移，長期應該會讓整體誤報率下降，只是無法用單一測試案例證明這件事——這需要真實標註過的通話
資料集才能量化，超出這次調查範圍。

重新以 `streamlit run main.py`（沿用已建好的 `AFG` conda 環境，`CUDA_VISIBLE_DEVICES=-1` 強制
CPU）啟動整個系統，`curl` 確認回應 `HTTP 200`，啟動日誌無新增例外。

---

## 這次沒有動的部分（供未來規劃參考）

- `fraud_detection.py` 裡所有手動調整過的門檻/權重魔術數字——需要真實標註過的通話資料集才能做有
  意義的校準，見 `../references/05-teleantifraud.md` 的正規做法對照。
- `baseline_pitch_range = 100`（`fraud_detection.py`）、`baseline_pause_rate = 0.8` 這類獨立於
  `self.baseline` 之外、單獨寫死的假設常數——屬於門檻校準問題，不是本次「明確程式錯誤」的範圍。
- 「語速增加百分比」相關的 3 個指標，用整通電話平均語速跟開頭第一個約 2 秒窗口比較——概念上是一個
  很粗糙的基準，但它是真的在運作、有實際影響分數的機制，不是死代碼，這次不動。
- 所有分析都是整通電話的聚合統計量（whole-call aggregate），沒有把「明確否認」「矛盾衝突」這類
  概念上應該對應到特定時間點/語句的指標，對齊到實際發生的時間——這是比較大的架構問題，不在本次
  bug 修復範圍。
- 完全沒有偵測 AI 合成語音/換聲詐騙的能力，見 `../references/04-asvspoof5.md`。
