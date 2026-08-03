# Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition (TIM-Net)

**Citation**: Ye, J., Wen, X.-C., Wei, Y., Xu, Y., Liu, K., & Shan, H. (2023). Temporal modeling matters: A novel temporal emotional modeling approach for speech emotion recognition. *ICASSP 2023 – IEEE International Conference on Acoustics, Speech and Signal Processing*. arXiv:2211.08233.

**連結**: https://arxiv.org/abs/2211.08233　｜　官方程式碼：https://github.com/Jiaxin-Ye/TIM-Net_SER

## 重點摘要

本專案 `antifraud_v2/models/TIM.py` 實作的 TIMNet（Temporal-aware bI-direction Multi-scale
Network）**就是這篇論文的架構**，發表於 ICASSP 2023（訊號處理領域頂尖會議之一）。TIM-Net 用
「時間感知區塊（Temporal-Aware Block）」在多個時間尺度上抽取語音的情緒表徵，再做雙向（過去/未來）
融合。論文在六個公開情緒語料庫上驗證：CASIA（中文，演員）、EMO-DB（德語，演員）、EMOVO（義大利語，
演員）、IEMOCAP（英語，即興演出對話）、RAVDESS（英語，演員）、SAVEE（英語，演員）——全部都是研究用
的情緒語料庫，不是電話客服/詐騙通話錄音，且驗證方式是逐語料庫內部的交叉驗證（in-corpus），沒有做
跨語料庫（cross-corpus）泛化測試。

## 對本專案的啟示

這篇是本專案情緒辨識模組的原始出處，用來確認兩件事：(1) `models/TIM.py` 的架構與這篇論文一致，
模型本身的設計是合理、有審稿驗證過的；(2) 但論文從未宣稱這個模型能用在電話詐騙偵測、中文自發對話
或長時通話上——這些都是本專案自行外推的應用場景。`main.py` 的
`EMOTION_LABELS = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']` 剛好完全
對應 EMO-DB（德語、7 類演出情緒資料庫）的標籤集合，強烈暗示實際載入的預訓練權重是針對 EMO-DB
fine-tune 的——這個外推是否成立，見 `02-cross-lingual-ser.md` 對跨語言泛化落差的最新實證。
