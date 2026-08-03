# Multimodal Deception Detection: A Survey

**Citation**: Zhang, J., Lin, X., Huang, J., et al. (2026). Multimodal deception detection: A survey. *Machine Intelligence Research*, 23, 284–307.

**連結**: https://www.mi-research.net/en/article/doi/10.1007/s11633-025-1625-x　｜　DOI: 10.1007/s11633-025-1625-x

## 重點摘要

2026 年發表於 Springer 旗下期刊 *Machine Intelligence Research* 的最新完整回顧型論文（survey），
系統性整理多模態欺騙偵測（Multimodal Deception Detection, MMDD）領域的研究背景、公開基準資料集、
評測指標、特徵融合方法，以及從傳統機器學習演進到深度學習的偵測架構演變，並隨論文釋出一份持續更新的
GitHub 資源清單（精選資料集與相關研究）。涵蓋的模態包含語音/聲學、文字、臉部表情、生理訊號等，是
目前對「多模態怎麼做欺騙偵測」這個問題最新、最完整的中立性文獻整理。

## 對本專案的啟示

這篇提供一個宏觀視角，可以拿來檢視 `antifraud_v2` 的整體架構設計是否跟得上這個領域目前公認的做法：
本專案用的兩個模態（聲學特徵、GPT-4 文字分析）落在這篇 survey 涵蓋的範圍內，但融合方式
（`multimodal_fusion.py` 用固定權重加權平均）是相對傳統、簡化的做法，而非 survey 中提到的、
較新的深度學習端對端融合架構。這篇也是一個很好的入口，供未來想系統性升級本專案時，快速掌握這個
領域目前有哪些公開資料集、評測指標與架構選項可以參考，而不必從零開始逐篇搜尋。搭配
`05-teleantifraud.md`（電信詐騙專用資料集）閱讀，可以從「整體方法論全景」與「最貼近本專案場景的
具體範例」兩個角度理解目前研究的實際水準。
