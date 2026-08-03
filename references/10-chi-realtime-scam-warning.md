# "It Warned Me Just at the Right Moment": Exploring LLM-based Real-time Detection of Phone Scams

**Citation**: Shen, Z., Yan, S., Zhang, Y., Luo, X., Ngai, G., & Fu, E. Y. (2025). "It warned me just at the right moment": Exploring LLM-based real-time detection of phone scams. In *Extended Abstracts of the 2025 CHI Conference on Human Factors in Computing Systems (CHI EA '25)*, Article 18.

**連結**: https://doi.org/10.1145/3706599.3720263

## 重點摘要

發表於 2025 年 CHI（人機互動領域頂尖會議之一）Extended Abstracts 場次。這篇建立一個即時將通話轉文字、
再用 LLM（GPT-4）比對已知詐騙話術模式、施壓手法、可疑要求的系統，在偵測到疑似詐騙時**即時**對使用者
發出警示，而非事後產出報告。研究特別找了 20 位 25–65 歲的受試者，由專業演員扮演技術支援詐騙、銀行
冒名、政府機關冒名等多種詐騙情境做受控實驗，重點檢驗系統在「多快發出警示」與「偵測準確率/召回率」
之間的取捨（trade-off between recall and timeliness）——警示太早容易誤報、太晚就失去保護意義。

## 對本專案的啟示（對「完全重寫」的啟示）

現有 6 篇文獻（以及本次新增的其他 3 篇）多半聚焦在偵測準確率或架構設計本身，唯獨這篇是目前找到
**唯一**用真人受試者測試「即時示警」實際使用體驗的研究，補上一個容易被純技術視角忽略的面向：一個
偵測系統再準，如果示警的時機（太早／太晚）或呈現方式讓使用者無法有效反應，實務上還是沒有用。這篇
的受控實驗設計（找演員扮演多種詐騙情境、量化 recall 與即時性的取捨）也是一個具體可參考的評測方法論，
若重寫時想把「通話中即時預警」（見 `09-prescam-benchmark.md`）落地成實際功能，這篇提醒了警示時機/UX
本身需要被獨立設計與測試，而不只是把偵測模型準確率做高就好。
