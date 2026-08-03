# Detecting Continuously Evolving Scam Calls under Limited Annotation: A LLM-Augmented Expert Rule Framework

**Citation**: Ma, H., Su, Q., Huang, M., & Kai, W. (2025). Detecting continuously evolving scam calls under limited annotation: A LLM-augmented expert rule framework. In *Findings of the Association for Computational Linguistics: EMNLP 2025* (pp. 5047–5068).

**連結**: https://aclanthology.org/2025.findings-emnlp.270/

## 重點摘要

發表於 2025 年 EMNLP Findings（自然語言處理領域頂尖會議之一）。這篇直接處理一個很實際的限制：真實
世界的詐騙電話偵測系統通常**沒有大量標註資料**，而且詐騙手法會持續演化，用靜態標註資料訓練出來的
分類器很快就過時。作者提出一個層級式的少樣本（few-shot）提示框架，用 LLM 輔助專家規則，包含三個模組：
判別模組（識別詐騙特徵）、反思模組（拿正常通話特徵做比對以降低誤報）、摘要模組（整合成最終判斷）。
在真實與合成資料集上驗證，證明能用極少標註資料就有效偵測持續演化的詐騙手法，且系統只需要調整專家
規則本身，就能快速適應新型態的詐騙場景，不必重新訓練整套模型。

## 對本專案的啟示（對「完全重寫」的啟示）

這篇論文的問題設定跟 `antifraud_v2` 目前的處境幾乎一模一樣：`fraud_detection.py` 裡的 8 種詐欺模式、
所有門檻/權重數字，全部是手動調出來的規則，**沒有任何標註資料集可以驗證或校準**——這正是這次誤報率
調查中明確標記「不動」的部分（見 `../antifraud_v2/FALSE_POSITIVE_REVIEW.md`「這次沒有動的部分」）。
這篇論文示範的做法，是在同樣「沒有大量標註資料」的前提下，用 LLM 輔助專家規則（判別＋反思降低誤報＋
綜合判斷），取代純手動調整的門檻邏輯——反思模組專門對比正常通話特徵來壓低誤報，這正好對應到本專案
目前最大的痛點。若重寫 `fraud_detection.py`，這是目前找到最直接可參考、且不需要先擁有大型標註資料集
就能落地的具體替代架構，比繼續手動調魔術數字門檻更有原則、也更容易隨新型詐騙手法快速調整。
