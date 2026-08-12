"""Q5：在給定的營運成本結構下，本系統淨效益為正的操作點區間是什麼？

**為什麼需要這一題**：PROTOCOL.md §8 引用了英國 DWP 的 Voice Risk Analysis 試辦，
但只學到一半的教訓。DWP 2010 年終止的官方理由是**「不具成本效益」，不是「準確率不足」**——
那是一個部署與營運失敗，不是分類器失敗。而本協定的 Q1-Q4 全部是分類器問題。
一套 HTER 漂亮的系統，仍然可以因為成本結構不成立而在部署三年後被終止。

**這一題也修正了主要指標的一個預設**：§4.2 選 HTER 作為主要指標，而 HTER 是
(FRR + FAR) / 2——它**預設兩種錯誤等重**。但一次誤報的成本（客服中斷一通正常通話）
與一次漏報的成本（一筆詐騙交易完成）幾乎確定差好幾個數量級。§4.2 說「分開報，
讓讀者自行判斷哪種錯誤更嚴重」聽起來嚴謹，實際上是把最重要的問題外包出去。

**輸出可能是空集合**——也就是「在任何門檻下這套系統都不划算」。
那是一個極有價值的負面結果，而且比再測十個語料都便宜。

**本檔用的是我方產業估計值，不是 TapPay 的實際數字。** 所有參數集中在 SCENARIOS，
拿到真實參數後只需替換該常數即可重跑。估計值的來源與不確定性見各欄註解。

用法：
    python -m antifraud_v3.eval.cost_benefit
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .provenance import stamp

RESULT_PATH = Path(__file__).parent / "cost_benefit_result.json"


@dataclass
class CostStructure:
    """一組營運成本參數。四個數字，全部不需要任何通話錄音。"""

    name: str
    daily_calls: int          # 每日通過本系統的話務量
    fraud_base_rate: float    # 話務中實際為詐騙的比例（先驗機率）
    cost_false_alarm: float   # 一次誤報的成本（TWD）：客服工時 + 客戶體驗損失
    cost_missed_fraud: float  # 一次漏報的平均損失（TWD）
    max_daily_alerts: int     # 可承受的每日告警數上限（人力天花板）
    # 告警成功阻止損失的比例。**這個參數不能省略**：偵測到不等於阻止得了。
    # 告警出現時受害者可能已經轉帳，或不理會警示（自動化決策支援系統的已知現象）。
    # 先前版本隱含假設此值為 1.0，導致「每日淨效益一千萬」這種明顯失真的輸出。
    intervention_efficacy: float
    source: str


# 我方估計值。**每一個都標明來源與不確定性**——這些不是量測值。
SCENARIOS = [
    CostStructure(
        name="baseline_estimate",
        daily_calls=10_000,
        # 台灣金融詐騙的通話層盛行率沒有公開統計。0.5% 是一個中性假設，
        # 且下方會做敏感度分析——這個數字對結論的影響極大，不可當成已知量。
        fraud_base_rate=0.005,
        # 客服處理一次誤報：約 5 分鐘工時 + 客戶不滿。以每小時 300 元估。
        cost_false_alarm=50,
        # 一筆完成的詐騙損失。內政部警政署歷年統計的單案平均損失落在數十萬，
        # 此處保守取 200,000。
        cost_missed_fraud=200_000,
        max_daily_alerts=200,
        # 告警後真正阻止損失的比例。無公開實證，0.5 是刻意保守的中性假設，
        # 且下方單獨做敏感度分析。
        intervention_efficacy=0.5,
        source="我方估計值，非 TapPay 提供；fraud_base_rate 與 intervention_efficacy 不確定性最大",
    ),
]

# 分類器的操作點沿著一條取捨曲線移動，不是 (FRR, FAR) 的自由組合——
# 沒有任何分類器能同時做到 FRR=0 且 FAR=0。先前版本把兩者當獨立網格掃，
# 於是「最佳點」必然落在 (0, 0)，那不是一個可達的操作點，是網格的角落。
#
# 這裡用單參數族近似：FAR = (1 - FRR^(1/beta))^beta 形式過於武斷，
# 改用更透明的做法——直接由「總錯誤預算」參數化：
#   給定系統整體能力 c（越小越強），操作點滿足 FRR + FAR >= c，
#   在該約束上掃過各種偏好（偏保守 vs 偏寬鬆）。
# c 由 Q1 的實測 HTER 決定：c = 2 * HTER。
CAPABILITY_LEVELS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
TRADEOFF_GRID = [round(x / 100, 2) for x in range(0, 101, 2)]


def evaluate(cs: CostStructure, frr: float, far: float) -> dict:
    """給定一組 (FRR, FAR)，算出每日淨效益。

    FRR = 誤報率（正常被判詐騙的比例）
    FAR = 漏抓率（詐騙未被判出的比例）
    """
    n_fraud = cs.daily_calls * cs.fraud_base_rate
    n_normal = cs.daily_calls - n_fraud

    false_alarms = n_normal * frr
    missed = n_fraud * far
    caught = n_fraud * (1 - far)

    # 偵測到 != 阻止得了。乘上 intervention_efficacy 才是真正避免的損失。
    benefit = caught * cs.cost_missed_fraud * cs.intervention_efficacy
    cost = false_alarms * cs.cost_false_alarm        # 誤報的代價
    # 漏報不算「新增成本」——沒有系統時本來就會損失，這裡只算系統帶來的淨變化。

    alerts = false_alarms + caught
    return {
        "FRR": round(frr, 4),
        "FAR": round(far, 4),
        "HTER": round((frr + far) / 2, 4),
        "daily_false_alarms": round(false_alarms, 1),
        "daily_caught": round(caught, 1),
        "daily_missed": round(missed, 1),
        "daily_alerts": round(alerts, 1),
        "within_staffing": alerts <= cs.max_daily_alerts,
        "daily_net_benefit": round(benefit - cost, 0),
        "net_positive": benefit > cost,
    }


def sweep(cs: CostStructure, capability: float) -> dict:
    """在給定的系統能力下，沿著取捨曲線找可行操作點。

    capability c = FRR + FAR 的下界。c 越小代表分類器越強。
    在 c 固定的前提下，把錯誤預算在 FRR 與 FAR 之間分配——這才對應
    「調門檻」這個真實可做的動作。
    """
    points = []
    for share in TRADEOFF_GRID:
        # 錯誤預算的分配：share 為分給 FAR 的比例
        far = capability * share
        frr = capability * (1 - share)
        if far > 1 or frr > 1:
            continue
        points.append(evaluate(cs, frr, far))

    feasible = [p for p in points if p["net_positive"] and p["within_staffing"]]
    best = max(feasible, key=lambda p: p["daily_net_benefit"], default=None)

    return {
        "capability_c": capability,
        "implied_HTER": round(capability / 2, 4),
        "n_operating_points": len(points),
        "n_feasible": len(feasible),
        "is_empty_set": not feasible,
        "best_operating_point": best,
        "max_tolerable_FRR": max((p["FRR"] for p in feasible), default=None),
    }


def sensitivity(cs: CostStructure, field: str, values: list[float],
                capability: float) -> list[dict]:
    """對單一參數做敏感度分析。

    若結論在合理範圍內翻轉，那麼「這套系統划不划算」在拿到真實參數前
    **無法回答**，而那本身就是要對買方講的結論。
    """
    out = []
    for v in values:
        r = sweep(CostStructure(**{**asdict(cs), field: v}), capability)
        out.append({
            field: v,
            "is_empty_set": r["is_empty_set"],
            "max_tolerable_FRR": r["max_tolerable_FRR"],
            "best_net_benefit": (r["best_operating_point"] or {}).get("daily_net_benefit"),
        })
    return out


def main() -> int:
    results = []
    for cs in SCENARIOS:
        print(f"=== {cs.name} ===")
        print(f"  參數來源：{cs.source}")
        print(f"  每日話務 {cs.daily_calls:,}，盛行率 {cs.fraud_base_rate:.1%}，"
              f"人力上限 {cs.max_daily_alerts} 則/日")
        print(f"  誤報成本 {cs.cost_false_alarm:,} / 漏報損失 {cs.cost_missed_fraud:,} TWD"
              f"（比值 1:{cs.cost_missed_fraud / cs.cost_false_alarm:.0f}）")
        print(f"  告警阻止成功率 {cs.intervention_efficacy:.0%}\n")

        by_capability = []
        print(f"  {'系統能力 c':>10s} {'≈HTER':>7s} {'空集合?':>8s} "
              f"{'最佳FRR':>8s} {'最佳FAR':>8s} {'每日淨效益':>14s}")
        for c in CAPABILITY_LEVELS:
            r = sweep(cs, c)
            by_capability.append(r)
            b = r["best_operating_point"]
            s_frr = f"{b['FRR']:.1%}" if b else "—"
            s_far = f"{b['FAR']:.1%}" if b else "—"
            s_nb = f"{b['daily_net_benefit']:+,.0f}" if b else "—"
            print(f"  {c:10.0%} {r['implied_HTER']:7.0%} "
                  f"{('是' if r['is_empty_set'] else '否'):>8s} "
                  f"{s_frr:>8s} {s_far:>8s} {s_nb:>14s}")

        # 敏感度分析固定在一個中等能力點上做，避免混淆兩個變因
        ref_c = 0.20
        sens = {
            "fraud_base_rate": sensitivity(
                cs, "fraud_base_rate", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05], ref_c),
            "intervention_efficacy": sensitivity(
                cs, "intervention_efficacy", [0.1, 0.25, 0.5, 0.75, 1.0], ref_c),
            "cost_missed_fraud": sensitivity(
                cs, "cost_missed_fraud", [20_000, 50_000, 100_000, 200_000, 500_000], ref_c),
        }
        for field, rows in sens.items():
            print(f"\n  敏感度：{field}（固定 c={ref_c:.0%}）")
            for s in rows:
                frr, nb = s["max_tolerable_FRR"], s["best_net_benefit"]
                print(f"    {s[field]:>10} {('空集合' if s['is_empty_set'] else '可行'):>8s} "
                      f"  可容忍FRR {(f'{frr:.1%}' if frr is not None else '—'):>7s}"
                      f"  淨效益 {(f'{nb:+,.0f}' if nb is not None else '—'):>14s}")

        results.append({
            "scenario": asdict(cs),
            "by_capability": by_capability,
            "sensitivity": sens,
        })

    out = {
        "provenance": stamp(),
        "what_this_is_not": (
            "本分析使用我方產業估計值，**不是 TapPay 提供的實際參數**。"
            "結論的效力完全取決於這四個數字，其中 fraud_base_rate 的不確定性最大。"
            "本檔的用途是展示分析框架並找出結論對哪個參數最敏感，"
            "不是宣稱本系統在 TapPay 的營運環境中划算。"
        ),
        "why_not_HTER": (
            "HTER = (FRR + FAR) / 2 預設兩種錯誤等重。本分析顯示在漏報成本遠高於"
            "誤報成本時，最佳操作點的 FRR 與 FAR 差距極大，HTER 相同的兩個系統"
            "可以有完全不同的營運價值——故 HTER 適合作為技術指標，不適合作為決策指標。"
        ),
        "scenarios": results,
    }
    RESULT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
