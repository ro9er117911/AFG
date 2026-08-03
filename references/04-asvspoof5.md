# ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech

**Citation**: Wang, X., Delgado, H., Tak, H., Jung, J., Shim, H., Todisco, M., Kukanov, I., Liu, X., Sahidullah, M., Kinnunen, T. H., Evans, N., Lee, K. A., & Yamagishi, J. (2025). ASVspoof 5: Design, collection and validation of resources for spoofing, deepfake, and adversarial attack detection using crowdsourced speech. *Computer Speech & Language*, 95, 101825.

**連結**: https://doi.org/10.1016/j.csl.2025.101825　｜　arXiv 版本（ASVspoof 5 Workshop, INTERSPEECH 2024）：https://arxiv.org/abs/2408.08739

## 重點摘要

ASVspoof 是語音防偽（anti-spoofing）/深偽語音偵測領域最具指標性的系列挑戰賽，這篇是 2025 年發表於
語音處理領域老牌期刊 *Computer Speech & Language*（Elsevier）的第五代完整論文（先前有 2024 年
INTERSPEECH 衛星工作坊版本）。ASVspoof 5 資料庫以群眾外包方式，從約 2000 名語者、多元錄音條件下
收集資料，涵蓋 32 種不同的攻擊演算法，包含傳統與最新的語音合成（TTS）、語音轉換（voice
conversion），並首次納入對抗式攻擊（adversarial attack）。這是與本專案「聲學行為特徵→詐欺意圖」
完全不同的另一種「語音詐欺」：**偵測的不是「這個人在說謊」，而是「這段聲音根本不是真人即時說的
話」**（AI 語音複製、換聲詐騙電話等）。

## 對本專案的啟示

`antifraud_v2` 目前的偵測邏輯完全建立在「假設通話另一端是真人」的前提上——分析的是聲學/情緒/語言
行為模式，**沒有任何機制偵測合成語音或聲音複製**。隨著 2024-2025 年生成式語音技術快速普及（ASVspoof
5 特別納入最新神經聲碼器與對抗攻擊即反映這個趨勢），「假冒特定人物聲音」的電話詐騙是實務上快速增長
的風險型態，且用本專案現有的行為特徵分析邏輯很可能完全偵測不到——因為合成的聲音可以被調整成語速
平穩、情緒中性、毫無「顫抖」或「猶豫」的樣子，反而更容易被誤判為低風險。列入這篇最新版本是提醒這個
目前完全空白、且隨技術演進正在快速惡化的風險缺口，供未來若要擴充系統涵蓋範圍時參考。
