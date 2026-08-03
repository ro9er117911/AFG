# The Fundamental Frequency of Voice as a Potential Stress Biomarker: A Systematic Review and Meta-Analysis

**Citation**: Veiga, D. de L., Almeida, T. M., Uchida, R. R., & Cordeiro, Q. (2025). The fundamental frequency of voice as a potential stress biomarker: A systematic review and meta-analysis. *Stress and Health*. https://doi.org/10.1002/smi.70112

**連結**: https://doi.org/10.1002/smi.70112　｜　PMC 全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12531429/

## 重點摘要

2025 年發表於 *Stress and Health*（Wiley）期刊的系統性回顧與統合分析（meta-analysis），系統性檢索
PubMed 與 Scopus 上 2010 年 1 月至 2024 年 9 月間關於「壓力對聲音基頻（F0/音高）影響」的實驗研究，
納入 10 篇研究、共 1148 筆觀測。結果：急性壓力確實會讓聲音基頻顯著升高（SMD = 0.55，
95% CI [0.30, 0.80]，p < 0.001），在女性受試者與自發語音中效果更明顯——**但**研究間異質性
（heterogeneity）很高，且偵測到出版偏誤（publication bias）。作者結論是：F0 做為壓力生物標記
「有潛力」，但需要在更大規模、前瞻性的研究中進一步驗證，目前只能謹慎解讀。

## 對本專案的啟示

這是一篇方法論嚴謹、時間跨度到 2024 年 9 月的最新統合分析，取代舊版文獻回顧中引用的 2003/2009 年
文獻，提供更即時、更量化的證據基礎。它同時支持與提醒了兩件事：(1) 音高變化跟壓力狀態確實有統計上
顯著的關聯（這給了 `fraud_detection.py` 用音高相關特徵做為訊號的一部分合理性），但 (2) 效應量存在
明顯異質性、且已知有出版偏誤——用單一固定門檻（如本專案 `明確否認` 模式的「音高增加百分比: 15」）
去判斷「是否有壓力/是否在說謊」，缺乏跨研究一致驗證過的最佳切點，也没有把「哪些受試者/情境下效應
更強（女性、自發語音）」這類調節變數納入考慮。換句話說：音高是一個**方向正確但門檻不確定**的訊號，
不是本專案目前這種「單一固定百分比門檻」可以可靠捕捉的。
