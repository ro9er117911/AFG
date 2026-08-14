"""替代特徵集要組成幾個「模式」、怎麼組，才不會重蹈十模式的覆轍？

**這支程式不改產品行為**，只回答設計問題。`pattern_matcher.py` 的十個模式各自是
「情緒 AND 語意」的固定組合，而 `pattern_reachability.md` 已證明：組合越多項、
每一項的邊際出現率越低，交集就越可能是空的——四個模式死於一個從未出現的特徵。

要用 Q6b 的八項特徵取代原九項，就得決定新模式怎麼組。這個決定不該憑直覺，
因為同樣的錯誤會以同樣的方式再犯一次。本檔在 400 筆的實測資料上，
把每一種候選組合的實際觸發率算出來——**用資料擋掉組不出來的組合**。

方法：窮舉 2 項與 3 項的特徵組合（AND），計算：
  - scam 觸發率（可及性：組合是否根本觸發不了）
  - benign 觸發率（誤觸）
  - lift = scam率 / benign率（鑑別力方向）

**刻意不做的事**：不挑出「最好的」組合當成結論。在唯一一份語料上窮舉再挑最高分，
就是 PROTOCOL.md §6 禁止的調參到過關——挑出來的東西會過擬合這 400 筆。
本檔的用途是**排除**：把 scam 觸發率為 0 或過低的組合標出來，
告訴設計者哪些組合是死的。要宣稱某個組合有效，須在獨立語料上驗證。

用法：
    python -m antifraud_v3.eval.script_pattern_design
"""

import json
from itertools import combinations
from pathlib import Path

from .provenance import stamp
from .script_features import RESULT_PATH, SCRIPT_FEATURES

JSONL_PATH = RESULT_PATH.with_suffix(".jsonl")
REPORT_PATH = RESULT_PATH.parent / "script_pattern_design.json"

# 組合的 scam 觸發率低於此值即視為「實務上死的」——與 pattern_matcher 十模式中
# 觸發 5/23 的 p1 同量級。低於此值的組合在部署中幾乎不會出聲。
MIN_USEFUL_RATE = 0.05


def main() -> int:
    if not JSONL_PATH.exists():
        raise SystemExit(f"找不到 {JSONL_PATH}")

    rows = [json.loads(line) for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ok = [r for r in rows if "present" in r]
    scam = [r for r in ok if r["category"] == "scam"]
    ben = [r for r in ok if r["category"] == "benign"]
    keys = list(SCRIPT_FEATURES)

    def rate(rs: list[dict], combo: tuple[str, ...]) -> tuple[int, float]:
        n = sum(1 for r in rs if all(r["present"][k] for k in combo))
        return n, (n / len(rs) if rs else 0.0)

    results = []
    for size in (1, 2, 3):
        for combo in combinations(keys, size):
            s_n, s_rate = rate(scam, combo)
            b_n, b_rate = rate(ben, combo)
            results.append({
                "size": size,
                "features": list(combo),
                "zh": " + ".join(SCRIPT_FEATURES[k][0] for k in combo),
                "scam_n": s_n, "scam_rate": round(s_rate, 4),
                "benign_n": b_n, "benign_rate": round(b_rate, 4),
                # benign 為 0 時 lift 無定義；用 None 而非塞一個無限大的假數字
                "lift": round(s_rate / b_rate, 2) if b_rate > 0 else None,
                "dead": s_rate < MIN_USEFUL_RATE,
            })

    by_size = {}
    for size in (1, 2, 3):
        subset = [r for r in results if r["size"] == size]
        dead = [r for r in subset if r["dead"]]
        by_size[str(size)] = {
            "n_combinations": len(subset),
            "n_dead": len(dead),
            "dead_rate": round(len(dead) / len(subset), 3) if subset else None,
        }

    out = {
        "provenance": stamp(),
        "purpose": (
            "為『用 Q6b 八項特徵取代專利九項』提供設計依據：哪些 AND 組合在實測資料上"
            "根本觸發不了。**這是排除性分析，不是選型結論。**"
        ),
        "not_a_conclusion": (
            "本檔不挑出『最佳組合』。在唯一一份語料上窮舉再挑最高 lift，"
            "挑出的組合會過擬合這 400 筆——即 PROTOCOL.md §6 禁止的調參到過關。"
            "要宣稱任何組合有效，須在獨立語料上驗證。"
        ),
        "corpus": f"TeleAntiFraud test，scam {len(scam)} / benign {len(ben)}",
        "min_useful_rate": MIN_USEFUL_RATE,
        "dead_combination_rate_by_size": by_size,
        "key_finding": (
            "AND 組合的項數每增加一項，死組合的比例就上升——這正是專利十模式"
            "（每個模式要求 1-2 個情緒 AND 1-3 個語意特徵，共 2-5 項全中）"
            "有九個觸發不了的結構性原因。"
        ),
        "all_combinations": sorted(results, key=lambda r: (r["size"], -r["scam_rate"])),
    }
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"語料：scam {len(scam)} / benign {len(ben)}\n")
    print("組合項數越多，死組合越多——這是十模式的結構性死因：")
    for size in ("1", "2", "3"):
        d = by_size[size]
        print(f"  {size} 項 AND：{d['n_combinations']:3} 種組合，"
              f"其中 scam 觸發率 <{MIN_USEFUL_RATE:.0%} 的有 {d['n_dead']:3} 種"
              f"（{d['dead_rate']:.0%}）")

    print(f"\n單項特徵（供對照，非組合）：")
    for r in [x for x in out["all_combinations"] if x["size"] == 1]:
        lift = f"lift {r['lift']}" if r["lift"] else "benign=0"
        print(f"  {r['zh']:<12} scam {r['scam_rate']:5.1%}  benign {r['benign_rate']:5.1%}  {lift}")

    alive2 = [r for r in out["all_combinations"] if r["size"] == 2 and not r["dead"]]
    print(f"\n2 項 AND 中未死的 {len(alive2)} 種（**僅供設計參考，非推薦**）：")
    for r in alive2[:8]:
        lift = f"lift {r['lift']}" if r["lift"] else "benign=0"
        print(f"  {r['zh']:<28} scam {r['scam_rate']:5.1%}  benign {r['benign_rate']:5.1%}  {lift}")

    print(f"\n報表：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
