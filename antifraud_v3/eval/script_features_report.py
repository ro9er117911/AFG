"""從 `script_features_result.jsonl` 產生可貼進論文的數字。

與 `q1_report.py` 同樣的分工理由：把「算數字」與「跑實驗」分開，可以在不重跑
400 次 LLM 呼叫的前提下修正報表邏輯。本檔只讀 JSONL、不呼叫任何模型，
因此可重複執行且結果決定性。

本檔回答三個問題，全部不需要 ground truth（沿用 `pattern_behavior.py` 的立場）：

  A1 可及性：每項特徵在 scam 上被回報幾次？有沒有從未出現的？
     原九項的死因就是可及性（見 docs/eval/pattern_reachability.md），
     這是替代特徵集要通過的第一關。

  A2 鑑別方向：scam 與 benign 的出現率是否分離？附 Wilson 95% CI——
     依 PROTOCOL.md §4.3「沒有 n 的數字不得出現在論文中」，
     每個比率都要能看到它的不確定性。

  A3 每通命中數分布：整體上兩類是否可分。

**刻意不做的事**：
  - 不報準確率。這組特徵無 ground truth，硬報是虛假精確。
  - 不做特徵選擇或門檻調校。那需要獨立的驗證集，這裡只有一份語料，
    在同一份資料上挑特徵再宣稱效果，就是 PROTOCOL.md §6 禁止的調參到過關。

用法：
    python -m antifraud_v3.eval.script_features_report
"""

import json
from math import sqrt
from pathlib import Path

from .provenance import stamp
from .script_features import RESULT_PATH, SCRIPT_FEATURES

JSONL_PATH = RESULT_PATH.with_suffix(".jsonl")
REPORT_PATH = RESULT_PATH.parent / "script_features_report.json"
EXPECTED_N = 400


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval。n 小時比常態近似可靠，且不會給出負的下界——
    本專案多處在 n=13 這種規模上報比率，常態近似會產生無意義的區間。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (centre - half) / d), 4),
            round(min(1.0, (centre + half) / d), 4))


def main() -> int:
    if not JSONL_PATH.exists():
        raise SystemExit(f"找不到 {JSONL_PATH}")

    rows = [json.loads(line) for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ok = [r for r in rows if "present" in r]
    scam = [r for r in ok if r["category"] == "scam"]
    ben = [r for r in ok if r["category"] == "benign"]
    keys = list(SCRIPT_FEATURES)
    complete = len(ok) >= EXPECTED_N

    a1 = {}
    for k in keys:
        s = sum(1 for r in scam if r["present"][k])
        b = sum(1 for r in ben if r["present"][k])
        a1[k] = {
            "zh": SCRIPT_FEATURES[k][0],
            "scam_n": s, "scam_total": len(scam),
            "scam_rate": round(s / len(scam), 4) if scam else None,
            "scam_ci95": wilson(s, len(scam)),
            "benign_n": b, "benign_total": len(ben),
            "benign_rate": round(b / len(ben), 4) if ben else None,
            "benign_ci95": wilson(b, len(ben)),
        }

    hs = [sum(r["present"].values()) for r in scam]
    hb = [sum(r["present"].values()) for r in ben]
    dist = lambda xs: {str(i): xs.count(i) for i in range(len(keys) + 1) if xs.count(i)}

    never = [k for k in keys if a1[k]["scam_n"] == 0]
    # CI 不重疊只是方向性的粗略判準，不是顯著性檢定——標示出來供人眼掃描，
    # 不在論文中當作統計結論使用。
    separated = [k for k in keys
                 if a1[k]["scam_ci95"][0] > a1[k]["benign_ci95"][1]]

    out = {
        "provenance": stamp(),
        "scope_note": (
            "本報表測替代特徵集的『可及性與鑑別方向』，不測判定準確率——"
            "這組特徵沒有任何資料集提供 ground truth（本專案自訂），"
            "準確率不可計算，亦不宣稱。沿用 pattern_behavior.py 的立場。"
        ),
        "completeness": ("完整" if complete
                         else f"**部分執行（{len(ok)}/{EXPECTED_N}）**"),
        "partial_run_caveat": None if complete else (
            f"僅完成 {len(ok)}/{EXPECTED_N} 筆，且語料未打亂——前段樣本的難度與"
            "標籤比例都和整體不同（見 memory: 語料順序陷阱）。"
            "本數字**不可與完整執行的結果比較**，也不得對外引用。"
        ),
        "n_total_rows": len(rows),
        "n_scored": len(ok),
        "n_scam": len(scam),
        "n_benign": len(ben),
        "A1_A2_per_feature": a1,
        "A1_never_present_in_scam": never,
        "A2_ci_separated_features": separated,
        "A3_hits_per_call": {
            "scam_mean": round(sum(hs) / len(hs), 2) if hs else None,
            "benign_mean": round(sum(hb) / len(hb), 2) if hb else None,
            "scam_dist": dist(hs),
            "benign_dist": dist(hb),
        },
        "comparison_to_patent_features": (
            "原專利表一九項中，lack_of_denial 與 spontaneous_correction 在 23 筆語料上"
            "present 次數皆為 0，使 global/p3/p8/p9 四個模式結構性不可觸發"
            "（docs/eval/pattern_reachability.md）。本報表的 A1_never_present_in_scam "
            "即對應指標：清單為空代表替代特徵集不存在同型的死角。"
        ),
        "not_claimed": [
            "不宣稱本特徵集比專利九項『更準』——無 ground truth，準確率不可比較。",
            "不宣稱 benign 低出現率等於精準：若語料為乾淨合成語音，可能是天花板效應。",
            "CI 不重疊僅為方向性判準，非顯著性檢定，不得作為統計結論。",
        ],
    }
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 替代特徵集 —— {out['completeness']} ===")
    if not complete:
        print(f"  ⚠ {out['partial_run_caveat']}")
    print(f"  scam n={len(scam)}  benign n={len(ben)}\n")
    print(f"  {'特徵':28}{'scam':>18}  {'benign':>18}")
    for k in keys:
        d = a1[k]
        s = f"{d['scam_n']:3}/{d['scam_total']:<3} {d['scam_rate']:5.1%}"
        b = f"{d['benign_n']:3}/{d['benign_total']:<3} {d['benign_rate']:5.1%}"
        mark = "  ←未出現" if d["scam_n"] == 0 else ("  ✓分離" if k in separated else "")
        print(f"  {d['zh']:<12}{k:20}{s:>14}  {b:>14}{mark}")
    print(f"\n  每通命中數：scam {out['A3_hits_per_call']['scam_mean']}  "
          f"benign {out['A3_hits_per_call']['benign_mean']}")
    print(f"  scam 分布 {out['A3_hits_per_call']['scam_dist']}")
    print(f"  benign 分布 {out['A3_hits_per_call']['benign_dist']}")
    print(f"\n  從未出現於 scam：{never or '（無——替代特徵集無死角）'}")
    print(f"\n報表：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
