"""Static prompt content for the discriminate/reflect/synthesize reasoning engine.

The stage taxonomy is deliberately NOT the old 8-pattern earnings-call rubric from
antifraud_v2/fraud_detection.py (which was, per its own docstring, "專為臺灣法人說明會環境設計" —
built to catch an executive lying about financials, not a scammer manipulating a phone-call
victim). See docs/DESIGN.md §1/§4. It's rewritten around scam-kill-chain stages, informed by
the PreScam benchmark (../references/09-prescam-benchmark.md).

The underlying acoustic/prosodic research citations from the old threshold dict are kept as
prompt grounding — they're real domain research, only the per-indicator arithmetic built on
top of them (the normalization bug, see docs/DESIGN.md §1) is what's being discarded.
"""

SCAM_KILL_CHAIN_RUBRIC = """\
評估這通電話目前所在的詐騙手法階段，不必假設每通電話都會走完全部階段，也不必假設階段一定按順序發生。
針對每個階段，用整體、全面的方式判斷證據強度（none / weak / moderate / strong），
而不是逐一指標加總計分——證據強度應該反映「這個階段真的存在」的整體可信度。

## 階段一：接觸/建立身分（contact_pretext）
對方自稱銀行、政府機關、公司客服、親友等身分，作為後續要求的基礎。
單純的身分自稱不構成強證據；身分與後續要求不合理地連動（例如「銀行」卻要求提供密碼），才是強證據。

## 階段二：製造急迫感/引用權威（urgency_authority）
對方強調「馬上」「立即」「否則會怎樣」等急迫用語，或引用法律後果、帳戶凍結、罰款等權威性威脅。
聲學參考（非固定門檻，作為整體判斷的輔助線索）：
語速突然改變、音高不穩定性（Levitan et al. 2015）、音量標準差偏高、憤怒情緒短暫尖峰（Rosas et al. 2015）。

## 階段三：要求隔絕（isolation）
對方要求「不要告訴家人」「不要跟銀行確認」「這是機密，不能讓別人知道」。
這個階段通常有明確的語句可以引用，是四階段裡最容易從文字直接判斷的一個。

## 階段四：要求付款/憑證（payment_credential_extraction）
要求提供驗證碼、密碼、轉帳到指定帳戶、購買虛擬貨幣/點數卡等。
這是後果最嚴重的階段——只要有明確語句要求驗證碼或轉帳，即使是單一句話，也可能構成 strong 證據
（不需要等其他階段都出現）。

## 聲學/情緒證據的使用方式
以下研究支持聲學/韻律線索與說謊/壓力有一定關聯，但效應量普遍不大、且因人而異
（Burgoon et al. 2016；DePaulo et al. 2003；Hirschberg et al. 2005；Vrij et al. 2008；
Benus et al. 2006；Graciarena et al. 2006；Harnsberger et al. 2009；Enos et al. 2007；
Wu et al. 2018；Ekman 2004）。**平靜、中性的語氣本身不是證據**——正常說話最常見的狀態就是平靜，
把「冷靜」當成可疑訊號是舊系統誤報率過高的直接原因，不要重蹈覆轍。聲學證據只在跟文字內容的具體
主張互相印證時才有意義（例如：語氣異常急促 + 文字明確要求立即轉帳，兩者一起看才是證據，各自單獨看都不是）。
"""

# TeleAntiFraud-28k's 7-category fraud-type taxonomy (Ma et al., ACM MM 2025 — see
# ../../references/05-teleantifraud.md). Independent of SCAM_KILL_CHAIN_RUBRIC above: that
# rubric asks "which stage of the scam process is this call in", this taxonomy asks "what kind
# of scam narrative does it resemble".
#
# No longer fed into any live LLM prompt (reference/documentation only) — since the two-line
# fusion architecture (reasoning/fusion.py), fraud_type classification comes from
# detectors/scam_semantic.py's Line 2 output via map_fraud_type(), not from Claude's own
# judgment, so this text has no synthesize()-prompt consumer anymore. Kept as the canonical
# Chinese description of each FraudType (schemas.py) category for anyone reading the taxonomy.
FRAUD_TYPE_TAXONOMY = """\
## 詐騙類型分類（與上面的階段判斷是不同維度，彼此獨立）

- investment_fraud（投資詐騙）：報明牌、保證獲利、虛假投資平台、加密貨幣投資群組等。
- phishing_fraud（網路釣魚詐騙）：假冒官方連結/簡訊要求點擊、輸入帳密或個資。
- identity_theft（身分冒用）：冒充公務員、警察、檢察官、親友等身分本身就是核心手法（猜猜我是誰、假冒公務員）。
- lottery_fraud（中獎摸彩詐騙）：宣稱中獎、抽中贈品，要求先付手續費/稅金才能領取。
- banking_fraud（銀行詐騙）：冒充銀行客服，聲稱帳戶異常、盜刷、需要「保護帳戶」或「圈存」。
- extortion_fraud（勒索詐騙）：威脅、假綁架、聲稱涉案要求付款了事。
- customer_service_fraud（客服詐騙）：冒充網購/物流客服，聲稱訂單有誤、需要退款或重新設定付款方式。
- unclassified：內容不明顯符合以上任一類型，或資訊不足以判斷。

分類依據**只看逐字稿內容的敘事類型**，跟 risk_level／階段證據強度無關——即使 risk_level 是
low，只要對話明顯屬於某個類型的敘事就可以標記；反之 risk_level 是 high 也可能因為敘事不清楚
而歸類為 unclassified。這是描述性的分類，不是風險判斷的一部分，不要讓兩者互相影響彼此的判斷。
"""

DEFAULT_HARD_TRIGGERS = [
    "要求提供簡訊/OTP 驗證碼",
    "要求「圈存」或「保護帳戶」（常見假保護話術）",
    "要求轉帳到指定帳戶或購買虛擬貨幣/點數卡",
    "要求不要告訴家人或不要跟銀行/警方求證",
]

# Literal substrings for the *live*, no-LLM hard-trigger check (pipeline/call_state.py's
# check_live_hard_trigger). DEFAULT_HARD_TRIGGERS above are descriptive rule names fed to the
# final one-shot LLM analysis (reasoning/synthesize.py) — those can't be substring-matched
# against a transcript directly. This is a separate, deliberately not user-editable (unlike
# DEFAULT_HARD_TRIGGERS, see storage/settings_store.py's hard_triggers setting) list, because
# the whole point is near-zero latency/cost during a live call — a fast path that still
# catches the handful of phrases severe enough to warn on immediately, while the nuanced
# reasoning happens once at call end (docs/DESIGN.md §2.2's original per-chunk LLM hard-trigger
# check no longer runs per chunk at all, see pipeline/chunk_worker.py).
HARD_TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "要求提供簡訊/OTP 驗證碼": ["驗證碼", "認證碼", "OTP", "one-time password"],
    "要求「圈存」或「保護帳戶」（常見假保護話術）": ["圈存", "保護帳戶", "帳戶保護", "資金保護", "凍結您的帳戶"],
    "要求轉帳到指定帳戶或購買虛擬貨幣/點數卡": [
        "轉帳", "匯款", "虛擬貨幣", "比特幣", "點數卡", "儲值卡", "遊戲點數", "購買禮品卡",
    ],
    "要求不要告訴家人或不要跟銀行/警方求證": [
        "不要告訴", "不要跟家人", "不要跟銀行", "不要報警", "不能讓別人知道", "不要讓別人知道", "這是機密",
    ],
}

# 專利十種詐欺模式標記（Phase 1）——13 類文字情緒（S309）與表一 9 項不合理語意特徵（S310）的
# 原文對照，組進 DISCRIMINATE_SYSTEM_PROMPT，讓 discriminate 步驟同一次呼叫額外輸出這兩個
# 區塊（schemas.TextEmotionScores / SemanticFeatureFinding）。key 是 schema 欄位名，value 是
# 中文名稱，供 reasoning/pattern_matcher.py 與 prompt 共用同一份對照，不重複定義。
TEXT_EMOTION_LABELS: dict[str, str] = {
    "anger": "生氣",
    "disgust": "厭惡",
    "fear": "害怕",
    "sadness": "悲傷",
    "surprise": "驚訝",
    "happy": "開心",
    "neutral": "中立",
    "stressful": "壓力",
    "extreme": "激動",
    "focus": "專注",
    "contradiction": "矛盾",
    "embarrassing": "尷尬",
    "vigilance": "警覺",
}

# 表一（docs/patent-ten-patterns-extract.md）「特徵—代表意義」對照，原文照抄，未改寫。key 是
# schemas.SemanticFeatureFinding 的欄位名，value 是 (中文特徵名, 代表意義原文)。
SEMANTIC_FEATURE_MEANINGS: dict[str, tuple[str, str]] = {
    "improper_pronoun_use": ("不當使用代詞", "避免使用「我」，減少對事件的個人承擔"),
    "lack_of_denial": ("缺乏否認", "避免直接否認指控"),
    "subjective_objective_time_mismatch": ("時間的主觀與客觀不一致", "文字描述的時間長短與實際時間不一致"),
    "language_change": ("語言改變", "不同段落中使用不同的詞語描述相同事件"),
    "incoherent_message": ("不連貫的訊息", "使用不必要的連接詞或句子"),
    "non_sequential_message": ("非順序訊息", "提供的細節不按邏輯順序排列"),
    "spontaneous_correction": ("自發修正", "不斷改正自己的話"),
    "unnecessary_connection": ("不必要的連接", "使用「然後」、「之後」來迴避問題"),
    "lack_of_commitment": ("缺乏承諾", "使用「我想」、「也許」來表示不確定性"),
}

TEXT_EMOTION_AND_SEMANTIC_FEATURE_RUBRIC = f"""\
除了以上四個階段的證據強度，請在同一次回覆中，額外針對這通電話的逐字稿內容輸出：

## 13 類文字情緒評分（text_emotions，每類 0-100 分，分數越高代表該情緒在文字語氣中的強度越明顯）
{'、'.join(f"{zh}（{en}）" for en, zh in TEXT_EMOTION_LABELS.items())}

## 9 項不合理語意特徵（semantic_features，每項判斷是否出現；出現的話附上逐字稿原句作為佐證引句）
{chr(10).join(f"- {zh}：{meaning}" for zh, meaning in SEMANTIC_FEATURE_MEANINGS.values())}
沒有出現的特徵，present 填 false、quote 留空；不要為了填滿而勉強引用不相關的句子。"""


DISCRIMINATE_SYSTEM_PROMPT = f"""你是一個電話詐騙偵測助理，正在分析一通電話結束後的完整逐字稿（含每句話的時間戳記）、
整通電話彙總後的聲學特徵、以及整通電話彙總後的情緒機率分布。這是整通電話結束後唯一一次的完整分析，
不是逐句即時判斷——請通盤考慮整通對話的脈絡（例如同一個要求在通話中反覆出現、或情勢隨時間升高），
不要只看單一句話。

{SCAM_KILL_CHAIN_RUBRIC}

針對這通電話，逐一評估四個階段目前的證據強度，並用一兩句話引用具體的逐字稿內容或聲學特徵作為理由。
沒有證據支持的階段，強度就是 none，不要為了填滿而勉強找理由。

{TEXT_EMOTION_AND_SEMANTIC_FEATURE_RUBRIC}"""

REFLECT_SYSTEM_PROMPT = """你是同一個詐騙偵測系統裡的「反思」步驟，任務是主動挑戰前一步（判別步驟）的結論，
找出每一項被標記證據是否有合理、無辜的解釋。

這是舊版規則引擎完全沒有的機制——舊系統只會累加分數，從來不會反過來問「這個證據會不會只是正常說話」。
你的存在就是為了防止這件事再發生一次。

對每個被標記為 weak/moderate/strong 的階段，認真想一個最合理的無辜解釋（例如：語氣急促可能只是趕時間、
要求提供資訊可能是正常客服流程、情緒起伏可能單純是講話習慣）。如果無辜解釋合理，就調降證據強度；
如果經過反思後這個證據還是站得住腳（尤其是階段三、四這種有明確語句可引用的），維持原本的強度，不用為了
「顯得有在反思」而硬降。"""

LINE2_LLM_SYSTEM_PROMPT = f"""你是「話術詐騙偵測」模型（Line 2），只根據電話逐字稿的文字內容判斷這通電話是不是詐騙——
你看不到音檔，只有文字，不要臆測聲音特徵。這是 detectors/scam_semantic.py 原本用本機 Qwen2Audio
模型做的同一件事，現在改由你純文字判斷，輸出格式必須相容。

{FRAUD_TYPE_TAXONOMY}

請輸出：
- scenario：這通電話大致屬於什麼日常情境（例如客服來電、外送、銀行來電等），一兩個詞即可。
- is_fraud：這通電話的話術內容是否構成詐騙。
- confidence：對 is_fraud 判斷的信心程度，0 到 1 之間的數字。
- fraud_type：is_fraud 為 true 時，從上面 7 類詐騙類型中選最符合的一個；is_fraud 為 false 時填
  unclassified。"""


def build_synthesize_system_prompt(hard_triggers: list[str] | None = None) -> str:
    """hard_triggers is a parameter, not a module constant, so it can come from the live
    settings store (storage/settings_store.py, wired via the Settings screen) rather than
    being frozen at import time — see docs/DESIGN.md §9's settings chip list.

    NOTE (two-line fusion architecture, see reasoning/fusion.py): risk_level/chunk_risk_score/
    fraud_type are no longer decided here — reasoning/fusion.py's fuse() already decided them
    from the two audio-native detectors' output before this prompt ever runs, and the caller
    (reasoning/synthesize.py) tells you that decision in the user_content. Your job shrank to
    exactly two things: evaluate the hard-trigger checklist, and write justification/
    case-memory prose *consistent with* the decision you're given, not a fresh one.
    """
    triggers = hard_triggers if hard_triggers is not None else DEFAULT_HARD_TRIGGERS
    trigger_lines = "\n".join(f"- {t}" for t in triggers)
    return f"""你是詐騙偵測系統的「綜合」步驟。這通電話的風險等級與詐騙類型已經由另外兩個專用的音訊模型
（AI合成語音偵測、話術詐騙語意偵測）判斷完畢，會在使用者訊息裡告訴你這個結論——你的工作**不是**重新
判斷風險等級或詐騙類型，而是：
1. 根據反思後的階段證據，評估立即示警關鍵字清單是否命中。
2. 寫出跟系統結論一致、具體可讀的說明文字與案件摘要。

立即示警關鍵字清單：
{trigger_lines}

justification 要具體、可以直接顯示給使用者看，不要只寫「風險偏高」這種空話——
要講清楚「為什麼」，並且要跟使用者訊息裡給你的系統判斷結論一致，不能自相矛盾
（例如系統結論是高風險，你的說明文字不能寫「這通電話看起來正常」）。

case_memory_update 是這通電話的簡短案件摘要，記錄目前為止已經確立的事實
（例如「對方自稱銀行，已要求提供帳號一次」），不用重複整段逐字稿。"""
