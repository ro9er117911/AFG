"""量測同一輸入重複呼叫的判定變異——LLM 判定本身的雜訊底線。

**這個測試源自一個意外發現**：Q3 執行兩次，light 條件從 9/13 變成 12/13，
同一模型、同一資料、同一段程式碼。既然變異這麼大，就必須把它當成一個
需要量化的量，而不是一句「有隨機性」的免責聲明。

為什麼這是最基礎的測試：**任何效應若小於重複執行的變異，就無法被偵測。**
若判定的重跑變異是 ±20 個百分點，那麼所有小於 20pp 的比較（包括 Q2 的 15pp 門檻、
Q3 的 light 條件）都不具意義。這個數字應該在其他所有實驗**之前**就量出來——
本文沒有做到，這是設計上的缺陷，在此補上。

作法：對同一份逐字稿重複呼叫 N 次，量測 is_fraud 的翻轉率與 confidence 的離散度。

用法：
    python -m antifraud_v3.eval.run_variance --repeats 5 --clips 6
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .provenance import stamp

EVAL_DIR = Path(__file__).parent
RESULT_PATH = EVAL_DIR / "run_variance_result.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5, help="每份逐字稿重複幾次")
    ap.add_argument("--clips", type=int, default=6, help="用幾份逐字稿")
    args = ap.parse_args()

    from ..detectors.scam_semantic_llm import classify_call_via_llm
    from ..llm import get_llm_provider

    scam = json.loads((EVAL_DIR / "adversarial_transcripts.json").read_text(encoding="utf-8"))
    benign = json.loads((EVAL_DIR / "benign_transcripts.json").read_text(encoding="utf-8"))

    # scam 與 benign 各取一半，因為變異可能不對稱——
    # 模型對「明顯詐騙」可能很穩定，對邊界案例才搖擺。
    half = max(args.clips // 2, 1)
    items = ([(n, t, True) for n, t in list(scam.items())[:half]] +
             [(n, t, False) for n, t in list(benign.items())[:half]])

    provider = get_llm_provider()
    rows = []
    print(f"重複判定測試：{len(items)} 份 x {args.repeats} 次\n")

    for name, text, is_scam in items:
        verdicts, confs = [], []
        for _ in range(args.repeats):
            try:
                r = classify_call_via_llm(provider, text)
                verdicts.append(r.is_fraud)
                confs.append(r.confidence)
            except Exception as e:
                print(f"  {name}: 失敗（{type(e).__name__}）")
            time.sleep(0.3)
        if not verdicts:
            continue
        majority = Counter(verdicts).most_common(1)[0]
        rows.append({
            "clip": name,
            "label_is_scam": is_scam,
            "n": len(verdicts),
            "verdicts": verdicts,
            "unstable": len(set(verdicts)) > 1,
            "majority_verdict": majority[0],
            "majority_share": round(majority[1] / len(verdicts), 3),
            "conf_mean": round(statistics.fmean(confs), 3),
            "conf_stdev": round(statistics.stdev(confs), 3) if len(confs) > 1 else 0.0,
            "conf_range": [min(confs), max(confs)],
        })
        flag = "  <- 判定不穩定" if rows[-1]["unstable"] else ""
        print(f"  {name:44s} {['正常','詐騙'][is_scam]} "
              f"{sum(verdicts)}/{len(verdicts)} 判為詐騙  conf {rows[-1]['conf_mean']:.2f}"
              f"±{rows[-1]['conf_stdev']:.2f}{flag}")

    unstable = [r for r in rows if r["unstable"]]
    result = {
        "provenance": stamp(),
        "purpose": (
            "量測 LLM 判定的重跑變異。任何小於此變異的效應都無法被偵測——"
            "這個數字界定了本專案所有比較的解析度下限。"
        ),
        "repeats": args.repeats,
        "n_clips": len(rows),
        "n_unstable": len(unstable),
        "unstable_rate": round(len(unstable) / len(rows), 3) if rows else None,
        "unstable_clips": [r["clip"] for r in unstable],
        "mean_conf_stdev": round(statistics.fmean([r["conf_stdev"] for r in rows]), 4) if rows else None,
        "per_clip": rows,
    }

    print(f"\n{'='*56}")
    print(f"判定不穩定的逐字稿：{len(unstable)}/{len(rows)}"
          f"（{result['unstable_rate']:.0%}）")
    if unstable:
        print(f"  {[r['clip'] for r in unstable]}")
    print(f"信心值平均標準差：{result['mean_conf_stdev']}")
    print("\n意義：本專案任何小於此變異的比較都不具意義，"
          "包括 Q2 的 15pp 門檻與 Q3 的 light 條件。")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
