"""Q1 雙臂的配對檢定（McNemar）。

**為什麼需要這個檔**：兩個 ASR 臂跑的是**同一批 400 筆音檔**，這是配對設計。
`PROTOCOL_v2.md` §4 明訂「配對設計一律用 McNemar，不得比較獨立信賴區間」——
這條規則來自 v1 的實際失誤（`REVIEW_LOG.md` 第 4 項）：當時用獨立 CI 比較配對資料，
重算後**結論改變**，原本撐起「衰減曲線」敘事的對比其實不顯著。

`q1_report.py` 只報兩臂的點估計之差。**點估計之差不是證據**——
400 筆裡差 3 個判定和差 30 個判定，HTER 差值可能相近，但一個是噪音一個是真的。
本檔回答的是「這個差是不是雜訊」。

本檔只讀 JSONL，不呼叫任何模型，因此可重複執行且結果決定性。

用法：
    python -m antifraud_v3.eval.q1_mcnemar
    python -m antifraud_v3.eval.q1_mcnemar --arms base medium
"""

import argparse
import json
from math import comb
from pathlib import Path

from .provenance import stamp
from .q1_report import discover_arms
from .teleantifraud import RESULT_PATH

REPORT_PATH = Path(__file__).parent / "q1_mcnemar.json"


def load_correctness(asr_model: str) -> dict[int, bool]:
    """回傳 {index: 該筆判定是否正確}。錯誤筆直接跳過——無法判定對錯。"""
    p = RESULT_PATH.with_suffix(f".{asr_model}.jsonl")
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("error") or d.get("predicted") is None:
            continue
        out[d["index"]] = bool(d["predicted"]) == bool(d["label"])
    return out


def exact_mcnemar(b: int, c: int) -> float:
    """精確 McNemar 的雙尾 p 值（二項檢定，n=b+c, p=0.5）。

    刻意用精確法而非卡方近似：b+c 小的時候（本專案常見）卡方近似會高估顯著性，
    而高估顯著性正是本專案在方法上最不能犯的錯。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(arm_a: str, arm_b: str) -> dict | None:
    a, b = load_correctness(arm_a), load_correctness(arm_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None

    # b_cnt: A 對 B 錯；c_cnt: A 錯 B 對。只有這兩格進入檢定——
    # 兩臂都對或都錯的樣本不含「哪一臂較好」的資訊。
    b_cnt = sum(1 for i in shared if a[i] and not b[i])
    c_cnt = sum(1 for i in shared if not a[i] and b[i])
    both_right = sum(1 for i in shared if a[i] and b[i])
    both_wrong = sum(1 for i in shared if not a[i] and not b[i])
    p = exact_mcnemar(b_cnt, c_cnt)

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_paired": len(shared),
        "n_a_only_in_a": len(set(a) - set(b)),
        "n_a_only_in_b": len(set(b) - set(a)),
        "table": {
            "both_correct": both_right,
            "a_correct_b_wrong": b_cnt,
            "a_wrong_b_correct": c_cnt,
            "both_wrong": both_wrong,
        },
        "discordant": b_cnt + c_cnt,
        "p_value": round(p, 5),
        "significant_at_0.05": p < 0.05,
        "verdict": _verdict(arm_a, arm_b, b_cnt, c_cnt, p),
    }


def _verdict(arm_a: str, arm_b: str, b_cnt: int, c_cnt: int, p: float) -> str:
    if b_cnt + c_cnt == 0:
        return "兩臂在每一筆上的對錯完全相同——無任何差異可檢定。"
    if p >= 0.05:
        return (
            f"不顯著（p={p:.3f}）。**不得宣稱兩臂有差異**，也不得用點估計之差"
            f"敘述「衰減」。不一致筆數僅 {b_cnt + c_cnt}，這個解析度看不出差異。"
        )
    better, worse = (arm_a, arm_b) if b_cnt > c_cnt else (arm_b, arm_a)
    return (
        f"顯著（p={p:.3f}）：{better} 優於 {worse}。"
        f"可陳述方向；效果量請另行以不一致筆數與 CI 表達，不要只報 p。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=None,
                    help="要比較的臂；預設自動偵測磁碟上所有臂並兩兩比較")
    args = ap.parse_args()

    arms = args.arms or discover_arms()
    arms = [a for a in arms if load_correctness(a)]
    if len(arms) < 2:
        print(f"需要至少兩個有結果的臂才能做配對檢定（目前：{arms or '無'}）。")
        return 1

    results = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            r = compare(arms[i], arms[j])
            if r:
                results.append(r)

    for r in results:
        t = r["table"]
        print(f"\n=== {r['arm_a']} vs {r['arm_b']} ===")
        if r["n_a_only_in_a"] or r["n_a_only_in_b"]:
            print(f"  註：配對檢定只用兩臂都有的 {r['n_paired']} 筆。這是**唯一**"
                  f"能比較部分執行的臂的正確方式——直接比兩臂各自的 HTER 會混入"
                  f"語料順序效應（語料未打亂，前段難度與標籤比例都和整體不同）。")
        print(f"  配對樣本 n={r['n_paired']}"
              + (f"（{r['arm_a']} 多出 {r['n_a_only_in_a']} 筆、"
                 f"{r['arm_b']} 多出 {r['n_a_only_in_b']} 筆，未納入）"
                 if r["n_a_only_in_a"] or r["n_a_only_in_b"] else ""))
        print(f"  兩臂皆對 {t['both_correct']}  兩臂皆錯 {t['both_wrong']}")
        print(f"  {r['arm_a']} 對／{r['arm_b']} 錯：{t['a_correct_b_wrong']}"
              f"    {r['arm_a']} 錯／{r['arm_b']} 對：{t['a_wrong_b_correct']}")
        print(f"  p = {r['p_value']}（精確 McNemar，雙尾）")
        print(f"  → {r['verdict']}")

    incomplete = [a for a in arms if len(load_correctness(a)) < 400]
    out = {
        "provenance": stamp(),
        "method": (
            "精確 McNemar（二項檢定，雙尾）。用精確法而非卡方近似，"
            "因為不一致筆數小時卡方會高估顯著性。"
        ),
        "protocol_basis": "PROTOCOL_v2.md §4：配對設計一律用 McNemar，不得比較獨立 CI。",
        "incomplete_arms": incomplete,
        "incomplete_caveat": (
            f"下列臂尚未完成 400 筆：{incomplete}。配對檢定只用兩臂**都有**的樣本，"
            "故結果有效，但屬於部分執行的中途快照，n 較小、檢定力較低。"
            "最終論文數字須待該臂跑完後重跑本檔。"
        ) if incomplete else None,
        "comparisons": results,
    }
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n報表：{REPORT_PATH}")
    return 0


def _self_check() -> None:
    """對照 REVIEW_LOG.md 第 4 項已獨立算出的三個 p 值。

    那三個數字是本專案歷史上唯一一次「重算後結論改變」的檢定，
    拿它們當迴歸基準，比自己編一組期望值有意義。
    """
    for b, c, want in [(4, 0, 0.125), (8, 0, 0.0078), (5, 1, 0.2188), (0, 0, 1.0)]:
        got = exact_mcnemar(b, c)
        assert abs(got - want) < 5e-4, f"McNemar({b},{c})={got}, 期望 {want}"
    # 對稱性：交換兩臂不應改變 p
    assert exact_mcnemar(3, 9) == exact_mcnemar(9, 3)
    # 卡方近似會在小樣本高估顯著性，精確法不該把 b=5,c=0 判為顯著
    assert exact_mcnemar(5, 0) > 0.05
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    raise SystemExit(main())
