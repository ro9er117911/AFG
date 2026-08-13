"""替代特徵集實驗：把「審訊破綻」換成「話術結構」，測其可及性與鑑別方向。

**動機**（見 docs/eval/pattern_reachability.md）：專利表一的九項語意特徵源自測謊訪談
情境，其中 `lack_of_denial`（缺乏否認）與 `spontaneous_correction`（自發修正）在 23 筆
語料上 present 次數皆為 0——前者預設「有人在指控你」，後者預設「審訊壓力下的口誤」，
而詐騙犯是主動來電、照稿念的一方。這使 global/p3/p8/p9 四個模式結構性不可觸發。

本實驗換一組**以詐騙話術結構為對象**的特徵，問同樣的問題：它們可及嗎？有鑑別方向嗎？

**與 pattern_behavior.py 的關係**：同樣刻意不宣稱準確率——這組特徵同樣沒有 ground truth
（是本專案自訂，不是公開 benchmark 的標註任務）。可問且值得問的仍然是行為：

  A1 可及性：每項特徵在 scam 上被回報幾次？**有沒有從未出現的？**
     這是本實驗的主要問題——原九項的死因就是可及性。

  A2 鑑別方向：scam 與 benign 的出現率是否分離？
     benign 若同樣高，該特徵就沒有資訊量。

  A3 每筆命中數分布：scam 平均命中幾項、benign 幾項？

**不回答**：這組特徵是否比原九項「更準」。沒有 ground truth 就不可能比較準確率，
只能比較可及性與鑑別方向。

用法：
    python -m antifraud_v3.eval.script_features            # 23 筆自建語料
    python -m antifraud_v3.eval.script_features --corpus teleantifraud --limit 400
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from ..llm import get_llm_provider
from ..llm.base import LLMProvider
from .provenance import stamp

EVAL_DIR = Path(__file__).parent
RESULT_PATH = EVAL_DIR / "script_features_result.json"


# 每項都是「詐騙話術為達成目的必須做的事」，而非「說謊者不小心露出的破綻」。
# 前者是加害者的**主動行為**，會留在逐字稿裡；後者預設一個審訊情境，本語料沒有。
SCRIPT_FEATURES: dict[str, tuple[str, str]] = {
    "authority_impersonation": (
        "冒充權威身份",
        "自稱銀行/警政/檢調/電信/平台客服等具權威或可信度的身份",
    ),
    "urgency_pressure": (
        "製造時間壓力",
        "強調期限、立即處理、否則後果嚴重，壓縮受害者思考時間",
    ),
    "isolation_attempt": (
        "阻止求證",
        "要求不要告訴家人/同事，或不要向銀行、警方查證",
    ),
    "credential_request": (
        "索取憑證",
        "要求提供簡訊驗證碼、OTP、密碼、卡號、身分證字號等",
    ),
    "irregular_payment": (
        "要求非常規金流",
        "要求轉帳到指定帳戶、購買點數卡/虛擬貨幣、或所謂「圈存」「保護帳戶」",
    ),
    "remote_control_request": (
        "要求遠端操作",
        "要求安裝程式、點擊連結、或配合操作網路銀行/ATM",
    ),
    "pretext_of_problem": (
        "虛構待解決問題",
        "宣稱帳戶異常、重複扣款、包裹卡關、涉案、欠費等需要處理的事由",
    ),
    "reward_baiting": (
        "利益誘餌",
        "承諾退款、中獎、高報酬投資、優惠或補助",
    ),
}


class ScriptFeature(BaseModel):
    present: bool = Field(default=False)
    quote: str | None = Field(default=None)


class ScriptFeatureFinding(BaseModel):
    authority_impersonation: ScriptFeature = Field(default_factory=ScriptFeature)
    urgency_pressure: ScriptFeature = Field(default_factory=ScriptFeature)
    isolation_attempt: ScriptFeature = Field(default_factory=ScriptFeature)
    credential_request: ScriptFeature = Field(default_factory=ScriptFeature)
    irregular_payment: ScriptFeature = Field(default_factory=ScriptFeature)
    remote_control_request: ScriptFeature = Field(default_factory=ScriptFeature)
    pretext_of_problem: ScriptFeature = Field(default_factory=ScriptFeature)
    reward_baiting: ScriptFeature = Field(default_factory=ScriptFeature)


SYSTEM_PROMPT = f"""你正在分析一通電話的逐字稿，判斷其中是否出現下列詐騙話術結構特徵。

{chr(10).join(f"- {en}（{zh}）：{meaning}" for en, (zh, meaning) in SCRIPT_FEATURES.items())}

每項獨立判斷是否出現；出現的話附上逐字稿原句作為佐證引句（quote），沒有出現就 present=false、
quote 留空。**不要為了填滿而勉強引用不相關的句子**——正常通話本來就大多數特徵都不會出現。

只描述逐字稿裡實際說了什麼，不要推測說話者的動機或情緒。"""


def extract(provider: LLMProvider, transcript: str) -> ScriptFeatureFinding:
    return provider.structured_complete(
        system=SYSTEM_PROMPT,
        user_content=f"逐字稿：\n{transcript}",
        schema=ScriptFeatureFinding,
    )


def load_local_corpus() -> list[tuple[str, str, str]]:
    """回傳 (category, clip, transcript)。用已快取的逐字稿，不重跑 ASR——
    重跑會讓本實驗與 pattern_behavior 的輸入不一致，測到的就變成 ASR 抖動。"""
    out = []
    for cat, fname in (("scam", "adversarial_transcripts.json"),
                       ("benign", "benign_transcripts.json")):
        path = EVAL_DIR / fname
        if not path.exists():
            continue
        for clip, text in json.loads(path.read_text(encoding="utf-8")).items():
            out.append((cat, clip, text))
    return out


def load_teleantifraud(limit: int) -> list[tuple[str, str, str]]:
    """TeleAntiFraud 的逐字稿來自 Q1 已跑完的 base 臂 JSONL——該檔存的是我方 ASR 的輸出，
    正是 Line 2 實際看到的文字。資料集本身不含逐字稿（PROTOCOL.md §7.2）。"""
    from .teleantifraud import RESULT_PATH as TA_PATH

    path = TA_PATH.with_suffix(".base.jsonl")
    if not path.exists():
        raise SystemExit(f"找不到 {path}——請先跑 teleantifraud 的 base 臂。")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        text = d.get("transcript")
        if not text:
            continue
        out.append(("scam" if d["label"] else "benign", str(d["index"]), text))
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["local", "teleantifraud"], default="local")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = (load_local_corpus() if args.corpus == "local"
               else load_teleantifraud(args.limit))
    if not records:
        raise SystemExit("語料為空。")

    provider = get_llm_provider()
    keys = list(SCRIPT_FEATURES)
    rows = []
    for i, (cat, clip, text) in enumerate(records, 1):
        try:
            f = extract(provider, text)
            present = {k: getattr(f, k).present for k in keys}
            quotes = {k: getattr(f, k).quote for k in keys if getattr(f, k).present}
            rows.append({"category": cat, "clip": clip, "present": present, "quotes": quotes})
        except Exception as e:  # 單筆失敗不該讓整批重跑——記下來，統計時排除
            rows.append({"category": cat, "clip": clip, "error": f"{type(e).__name__}: {e}"})
        print(f"  [{i}/{len(records)}] {cat:6} {clip}")

    ok = [r for r in rows if "present" in r]
    scam = [r for r in ok if r["category"] == "scam"]
    ben = [r for r in ok if r["category"] == "benign"]

    a1 = {k: {"scam": sum(1 for r in scam if r["present"][k]),
              "benign": sum(1 for r in ben if r["present"][k])} for k in keys}
    never = [k for k in keys if a1[k]["scam"] == 0]
    hits_scam = [sum(r["present"].values()) for r in scam]
    hits_ben = [sum(r["present"].values()) for r in ben]

    out = {
        "provenance": stamp(),
        "scope_note": (
            "本實驗測替代特徵集的『可及性與鑑別方向』，不測判定準確率——這組特徵同樣"
            "沒有任何資料集提供 ground truth（本專案自訂），準確率不可計算，亦不宣稱。"
        ),
        "corpus": args.corpus,
        "n_records": len(rows),
        "n_scored": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_scam": len(scam),
        "n_benign": len(ben),
        "A1_present_counts": a1,
        "A1_never_present_in_scam": never,
        "A2_rates": {k: {"scam": round(a1[k]["scam"] / len(scam), 3) if scam else None,
                         "benign": round(a1[k]["benign"] / len(ben), 3) if ben else None}
                     for k in keys},
        "A3_hits_per_call": {
            "scam_mean": round(sum(hits_scam) / len(hits_scam), 2) if hits_scam else None,
            "benign_mean": round(sum(hits_ben) / len(hits_ben), 2) if hits_ben else None,
            "scam_dist": dict(sorted(Counter(hits_scam).items())),
            "benign_dist": dict(sorted(Counter(hits_ben).items())),
        },
        "records": rows,
    }
    path = Path(args.out) if args.out else RESULT_PATH
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 可及性（A1）scam n={len(scam)} / benign n={len(ben)} ===")
    for k in keys:
        z = "  ← 從未出現" if a1[k]["scam"] == 0 else ""
        print(f"  {k:26} scam {a1[k]['scam']:3}  benign {a1[k]['benign']:3}{z}")
    print(f"\n每通命中數：scam 平均 {out['A3_hits_per_call']['scam_mean']}，"
          f"benign 平均 {out['A3_hits_per_call']['benign_mean']}")
    print(f"從未出現於 scam 的特徵：{never or '（無）'}")
    print(f"\n報表：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
