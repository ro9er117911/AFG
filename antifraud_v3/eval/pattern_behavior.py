"""專利十模式規則表在真實語料上的**行為分析**——刻意不宣稱判定準確率。

為什麼不測準確率：`pattern_matcher.py` 實作的是專利 TW I904863 S312 定義的十種詐騙模式。
**世界上沒有任何資料集標註了這十種模式**——它是專利自訂的分類法，不是公開 benchmark 的
任務。沒有 ground truth 就不可能算準確率，硬報一個數字會是虛假精確。

能問而且值得問的是規則表的**行為**（都不需要 ground truth）：

  B1 觸發率分布：哪些模式常觸發？**哪些從未觸發？**
     從未觸發代表門檻訂太嚴、或該組合在真實話術中不存在——這是對專利實施品質的
     實質檢驗，不是對專利本身的檢驗。

  B2 鑑別方向：詐騙通話觸發的模式數，是否顯著多於正常通話？
     若沒有差異，規則表就沒有在做事。這是規則表有無資訊量的最低檢驗。

  B3 門檻敏感度：DEFAULT_EMOTION_THRESHOLD=60 是我方的實作選擇，不是專利內容
     （pattern_matcher.py 的 docstring 自己載明）。掃描不同門檻，看觸發率如何變動——
     若結果對門檻極度敏感，那 60 這個數字就承載了過多未經驗證的重量。

  B4 情緒/語意特徵的邊際分布：LLM 實際上多常回報每一項特徵？
     若某個語意特徵從來不被回報，依賴它的模式就永遠不可能觸發。

用法：
    python -m antifraud_v3.eval.pattern_behavior --limit 8
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .provenance import stamp
from ..reasoning.pattern_matcher import DEFAULT_EMOTION_THRESHOLD, PATTERN_DEFINITIONS, match_patterns
from ..reasoning.schemas import ChunkEvidence, SemanticFeatureFinding, TextEmotionScores

RESULT_PATH = Path(__file__).parent / "pattern_behavior_result.json"
EVAL_DIR = Path(__file__).parent

# 掃描範圍涵蓋 60 兩側，看判定對這個「我方自訂」的數字有多敏感。
THRESHOLD_SWEEP = [40, 50, 60, 70, 80]


def load_transcripts() -> dict[str, dict[str, str]]:
    """scam 與 benign 都要——B2 需要對照組才能問「有沒有鑑別方向」。"""
    out = {}
    for cat, fn in [("scam", "adversarial_transcripts.json"), ("benign", "benign_transcripts.json")]:
        p = EVAL_DIR / fn
        if p.exists():
            out[cat] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"  [警告] 缺少 {fn}——{cat} 組將被跳過"
                  f"（產生方式：python -m antifraud_v3.eval.make_transcripts --category {cat}）")
    return out


def extract_evidence(provider, transcript: str) -> tuple[TextEmotionScores, SemanticFeatureFinding]:
    """走與正式系統相同的 discriminate() 路徑，而不是另外寫一個 prompt——
    測的必須是系統實際的行為，不是一個為了評測而造的近似品。"""
    from ..reasoning.discriminate import discriminate

    evidence = ChunkEvidence(
        transcript_segment=transcript,
        speaker_guess=None,
        acoustic_summary="（本分析僅測文字層，未提供聲學摘要）",
        emotion_summary="（同上）",
        call_state_summary="",
    )
    r = discriminate(provider, evidence)
    return r.text_emotions, r.semantic_features


def analyze(records: list[dict]) -> dict:
    """records: [{category, clip, emotions: dict, features: dict}]"""
    scam = [r for r in records if r["category"] == "scam"]
    benign = [r for r in records if r["category"] == "benign"]

    # ---- B1 觸發率分布（含從未觸發者）----
    trigger_counts: Counter = Counter()
    for r in records:
        for p in r["matched"]:
            trigger_counts[p] += 1
    never = [d.pattern_id for d in PATTERN_DEFINITIONS if trigger_counts[d.pattern_id] == 0]

    # ---- B2 鑑別方向 ----
    def counts(rows):
        return [len(r["matched"]) for r in rows]

    scam_n, benign_n = counts(scam), counts(benign)
    discrimination = {
        "scam_mean_patterns": round(statistics.fmean(scam_n), 3) if scam_n else None,
        "benign_mean_patterns": round(statistics.fmean(benign_n), 3) if benign_n else None,
        "scam_any_rate": round(sum(1 for c in scam_n if c) / len(scam_n), 3) if scam_n else None,
        "benign_any_rate": round(sum(1 for c in benign_n if c) / len(benign_n), 3) if benign_n else None,
        "n_scam": len(scam_n),
        "n_benign": len(benign_n),
    }

    # ---- B3 門檻敏感度 ----
    sweep = {}
    for th in THRESHOLD_SWEEP:
        s_hits = b_hits = 0
        for r in records:
            em = TextEmotionScores(**r["emotions"])
            sf = SemanticFeatureFinding(**r["features"])
            m = match_patterns(em, sf, threshold=th)
            if r["category"] == "scam":
                s_hits += bool(m)
            else:
                b_hits += bool(m)
        sweep[th] = {
            "scam_any_rate": round(s_hits / len(scam), 3) if scam else None,
            "benign_any_rate": round(b_hits / len(benign), 3) if benign else None,
        }

    # ---- B4 特徵邊際分布 ----
    feat_present: Counter = Counter()
    emo_over: Counter = Counter()
    for r in records:
        for k, v in r["features"].items():
            if isinstance(v, dict) and v.get("present"):
                feat_present[k] += 1
        for k, v in r["emotions"].items():
            if isinstance(v, (int, float)) and v >= DEFAULT_EMOTION_THRESHOLD:
                emo_over[k] += 1

    return {
        "provenance": stamp(),
        "scope_note": (
            "本分析測的是規則表的『行為』，不是判定準確率。十模式無任何公開資料集提供 "
            "ground truth（專利自訂分類法），故準確率不可計算，亦不宣稱。"
        ),
        "n_records": len(records),
        "default_threshold": DEFAULT_EMOTION_THRESHOLD,
        "B1_trigger_counts": dict(trigger_counts.most_common()),
        "B1_never_triggered": never,
        "B2_discrimination": discrimination,
        "B3_threshold_sweep": sweep,
        "B4_feature_present_counts": dict(feat_present.most_common()),
        "B4_emotion_over_threshold_counts": dict(emo_over.most_common()),
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="每組限制筆數（每筆一次 LLM 呼叫）")
    args = ap.parse_args()

    from ..llm import get_llm_provider

    transcripts = load_transcripts()
    if not transcripts:
        print("沒有任何逐字稿可用。")
        return 1

    provider = get_llm_provider()
    records = []
    for category, items in transcripts.items():
        rows = list(items.items())[: args.limit]
        print(f"\n[{category}] {len(rows)} 筆")
        for name, text in rows:
            try:
                em, sf = extract_evidence(provider, text)
            except Exception as e:
                print(f"  {name}: 抽取失敗（{type(e).__name__}），跳過")
                continue
            matched = [p.pattern_id for p in match_patterns(em, sf)]
            records.append({
                "category": category,
                "clip": name,
                "emotions": em.model_dump(),
                "features": sf.model_dump(),
                "matched": matched,
            })
            print(f"  {name:44s} 命中 {len(matched)}: {matched or '—'}")
            time.sleep(0.4)

    result = analyze(records)

    print(f"\n{'='*58}\nB1 觸發率分布（n={result['n_records']}）")
    for pid, c in result["B1_trigger_counts"].items():
        print(f"   {pid:8s} {c:3d} 次")
    if result["B1_never_triggered"]:
        print(f"   從未觸發：{result['B1_never_triggered']}  <- 門檻過嚴，或該組合不存在於真實話術")

    d = result["B2_discrimination"]
    print(f"\nB2 鑑別方向")
    print(f"   詐騙 (n={d['n_scam']})：平均命中 {d['scam_mean_patterns']} 個，"
          f"至少命中一個的比例 {d['scam_any_rate']}")
    print(f"   正常 (n={d['n_benign']})：平均命中 {d['benign_mean_patterns']} 個，"
          f"至少命中一個的比例 {d['benign_any_rate']}")

    print(f"\nB3 門檻敏感度（預設 {result['default_threshold']} 為我方實作選擇，非專利內容）")
    print(f"   {'門檻':>6s} {'詐騙觸發率':>10s} {'正常觸發率':>10s}")
    for th, s in result["B3_threshold_sweep"].items():
        print(f"   {th:6d} {str(s['scam_any_rate']):>10s} {str(s['benign_any_rate']):>10s}")

    print(f"\nB4 語意特徵回報次數")
    for k, c in result["B4_feature_present_counts"].items():
        print(f"   {k:38s} {c:3d}")
    missing = [f for f in SemanticFeatureFinding.model_fields if f not in result["B4_feature_present_counts"]]
    if missing:
        print(f"   從未被回報：{missing}  <- 依賴這些特徵的模式永遠不可能觸發")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
