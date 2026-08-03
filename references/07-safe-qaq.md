# SAFE-QAQ: End-to-End Slow-Thinking Audio-Text Fraud Detection via Reinforcement Learning

**Citation**: Wang, P., Ma, Z., Dai, X., Liu, Y., Feng, S., Yang, X., Hu, W., Wang, Z., Pan, M., Yuan, L., & Wang, D. (2026). SAFE-QAQ: End-to-end slow-thinking audio-text fraud detection via reinforcement learning. *arXiv preprint arXiv:2601.01392*.

**連結**: https://arxiv.org/abs/2601.01392

## 重點摘要

2026 年 1 月發表的 arXiv 論文，是 `05-teleantifraud.md`（TeleAntiFraud-28k）同一批作者群的後續生產
系統論文。既有做法多半只用轉錄文字做詐騙偵測，會被 ASR 錯誤拖累、也遺失語調等聲學線索；SAFE-QAQ
提出一個**端對端**的音訊+文字融合框架，用規則式的「慢思考」獎勵機制（rule-based slow-thinking reward
mechanisms）引導模型透過層級化推理去抓細粒度的音訊線索，而不是先轉文字再分析。用強化學習訓練，在
TeleAntiFraud-Bench 上於準確率、推論效率、即時處理能力等多個面向都大幅超越既有方法，且論文明確指出
系統**已經部署、每天實際分析超過 7 萬通電話**，是少數有生產規模驗證數據的同類研究。

## 對本專案的啟示（對「完全重寫」的啟示）

`antifraud_v2` 目前的 `multimodal_fusion.py` 是音訊風險分數與文字風險分數各自獨立算完後，用固定權重
做加權平均——音訊端（`fraud_detection.py`）跟文字端（GPT-4）之間完全沒有互動，也沒有經過任何資料驅動
的訓練或校準。SAFE-QAQ 展示的是同一個應用場景（電信詐騙、音訊+文字雙模態）**具體要怎麼把融合這件事
做對**：不是兩個獨立分數加權平均，而是端對端訓練、讓模型自己學會該怎麼結合聲學線索與語意線索做推理，
且訓練訊號直接針對「慢思考」式的可解釋推理過程做獎勵，而非黑箱分類。若重寫要把 `multimodal_fusion.py`
從規則式加權平均升級成真正的融合模型，這篇是目前找到最直接對應本專案架構、且有生產規模驗證的參考
範例。**注意**：這是 arXiv 預印本，尚未見到正式同行評審會議/期刊發表紀錄，引用時應留意其結論尚未經過
獨立的同行評審把關。
