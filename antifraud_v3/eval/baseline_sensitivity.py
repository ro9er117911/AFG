"""Q4 重寫版：量測**基準選擇**的敏感度，而非通話內的自然波動。

為什麼需要這個檔案：`eval/baseline_stability.py`（第一版）計算的是整通電話所有 chunk 的
變異係數，那是「特徵在通話中波動多大」。獨立審查指出**那不是 §5.4 要問的問題**，
兩者不等價——一個特徵可以在通話中大幅波動，但只要開頭幾個 chunk 彼此接近，
基準選擇仍然穩定；反之亦然。

本檔測的才是實際宣稱：`call_state.maybe_set_baseline()` 把整通電話的聲學基準
釘在**第二個語音 chunk** 上，之後 UI 所有「相對基準上升 X%」都以它為準。
**如果基準改取第 3、第 4 個 chunk，那個 X% 會差多少？**

作法：對每通電話，依序用第 k 個 chunk（k=1..K）當基準，計算下游相對量
`(value_i - baseline_k) / baseline_k`，再看這組相對量隨 k 的離散程度。
離散大 = 使用者看到的百分比高度取決於「系統剛好挑了第幾個 chunk」，那個數字就不可靠。

判定門檻沿用 `PROTOCOL.md` §6 預先登錄的 20%，但**套用對象改為正確的量**
（此更動已記入協定修訂記錄）。

用法：
    python -m antifraud_v3.eval.baseline_sensitivity
"""

import json
import statistics
from pathlib import Path

import soundfile as sf

from .baseline_stability import TRACKED, _flatten, chunk_features
from .provenance import stamp

CLIPS_DIR = Path(__file__).parent / "test_clips"
RESULT_PATH = Path(__file__).parent / "baseline_sensitivity_result.json"

# 只考慮前幾個 chunk 作為候選基準——真實系統就是在通話開頭挑，
# 拿通話中段當基準不是它會做的事，測了也不對應任何真實行為。
#
# 設為 2 是語料逼出來的，不是理想值：實測現有 23 個音檔的 chunk 數最多只有 4
# （分布 [3]*10 + [4]*4），要測 k 個候選基準需要至少 k+2 個 chunk。
# k=4 時符合的音檔為 0，k=3 時仍為 0，k=2 時只有 4/23。
# **這代表本測試在現有語料上先天無法得到有意義的結論**，見 PAPER.md §5.4。
MAX_BASELINE_CANDIDATES = 2
CV_UNSTABLE_THRESHOLD = 0.20
# 樣本數低於此值就不下判定——n=0 卻印「全部穩定」是假陽性的結論
MIN_CLIPS_FOR_VERDICT = 8


def analyze() -> dict:
    per_metric: dict[str, list[float]] = {m: [] for m in TRACKED}
    clips = []

    for path in sorted(CLIPS_DIR.rglob("*.wav")):
        feats = chunk_features(path)
        # 至少要有「幾個候選基準」+「幾個被比較的後續 chunk」才有得算
        if len(feats) < MAX_BASELINE_CANDIDATES + 2:
            continue

        row = {"clip": str(path.relative_to(CLIPS_DIR)), "n_chunks": len(feats)}
        for m in TRACKED:
            # 對每個候選基準 k，算出「後續 chunk 相對於它的平均變化率」
            rel_by_k = []
            for k in range(MAX_BASELINE_CANDIDATES):
                base = feats[k][m]
                if not base:
                    continue
                after = [f[m] for f in feats[k + 1:] if f[m]]
                if not after:
                    continue
                rel_by_k.append(statistics.fmean((v - base) / base for v in after))

            if len(rel_by_k) < 2:
                continue
            spread = max(rel_by_k) - min(rel_by_k)
            row[m] = {
                "relative_change_by_baseline_choice": [round(x, 4) for x in rel_by_k],
                # 這才是要問的量：換一個 chunk 當基準，使用者看到的百分比會差多少
                "spread": round(spread, 4),
            }
            per_metric[m].append(spread)
        clips.append(row)

    summary = {}
    for m, spreads in per_metric.items():
        if not spreads:
            continue
        med = statistics.median(spreads)
        summary[m] = {
            "n_clips": len(spreads),
            "median_spread": round(med, 4),
            "max_spread": round(max(spreads), 4),
            "unstable": med > CV_UNSTABLE_THRESHOLD,
        }
    return {
        "provenance": stamp(),
        "what_this_measures": (
            "基準選擇的敏感度：改用第 1..K 個 chunk 當基準時，下游『相對基準變化率』"
            "的最大差距。這與 baseline_stability.py 測的『通話內自然波動』不同——"
            "後者是該檔的已知缺陷（見 PAPER.md §5.4 限制 (i)）。"
        ),
        "max_baseline_candidates": MAX_BASELINE_CANDIDATES,
        "threshold": CV_UNSTABLE_THRESHOLD,
        "summary": summary,
        "per_clip": clips,
    }


def main() -> int:
    r = analyze()
    print(f"基準選擇敏感度（候選：前 {r['max_baseline_candidates']} 個 chunk，"
          f"門檻 {r['threshold']:.0%}）\n")
    print(f"{'特徵':16s} {'n':>4s} {'中位數 spread':>13s} {'最大 spread':>12s}  判定")
    for m, s in r["summary"].items():
        print(f"{m:16s} {s['n_clips']:4d} {s['median_spread']:13.1%} "
              f"{s['max_spread']:12.1%}  {'不穩定' if s['unstable'] else '穩定'}")

    bad = [m for m, s in r["summary"].items() if s["unstable"]]
    n = max((s["n_clips"] for s in r["summary"].values()), default=0)
    print()
    if n < MIN_CLIPS_FOR_VERDICT:
        # n=0 或極小時印「全部穩定」是假陽性結論——這裡明確拒絕下判定
        print(f"**不下判定**：符合條件的音檔僅 {n} 個（需要 >= {MIN_CLIPS_FOR_VERDICT}）。")
        print(f"現有語料每個音檔最多只有 4 個 chunk，測 {MAX_BASELINE_CANDIDATES} 個候選基準"
              f"需要至少 {MAX_BASELINE_CANDIDATES + 2} 個。")
        print("這是**語料的先天限制**，不是「基準穩定」的證據——"
              "要回答 Q4 需要更長的通話錄音。")
    elif bad:
        print(f"結論：{len(bad)} 項對基準選擇敏感（{', '.join(bad)}）——"
              f"這些欄位的『相對基準』百分比高度取決於系統挑了第幾個 chunk。")
    else:
        print(f"結論：所有追蹤特徵對基準選擇不敏感（n={n}）。")

    RESULT_PATH.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
