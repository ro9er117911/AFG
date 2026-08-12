"""從 Q1 的續跑 JSONL 產生可貼進論文的數字。

**為什麼獨立成一個檔**：`teleantifraud.py` 的彙整是在 process 啟動時載入的，
執行中對該檔的修改不會生效於當次執行。把「算數字」與「跑實驗」分開，
可以在不重跑 400 筆的前提下修正報表邏輯——這在單次執行要數小時時是必要的。

本檔只讀 JSONL，不呼叫任何模型，因此可重複執行且結果決定性。

用法：
    python -m antifraud_v3.eval.q1_report
"""

import json
from pathlib import Path

from .provenance import stamp
from .teleantifraud import RESULT_PATH, Trial, compute_metrics

REPORT_PATH = Path(__file__).parent / "q1_report.json"


def load_arm(asr_model: str) -> list[Trial]:
    p = RESULT_PATH.with_suffix(f".{asr_model}.jsonl")
    if not p.exists():
        return []
    trials = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        trials.append(Trial(d["index"], d["label"], d["predicted"],
                            d["confidence"], d.get("fraud_type"), d.get("error")))
    return trials


def asr_health(asr_model: str) -> dict:
    """轉寫品質的健康檢查。

    空白或極短的轉寫會讓分類器在「沒有內容」上做判定，那種 100% 或 0%
    都不是能力的證據。這一欄必須跟指標一起報，否則指標可能是假的。
    """
    p = RESULT_PATH.with_suffix(f".{asr_model}.jsonl")
    if not p.exists():
        return {}
    chars = [json.loads(l)["asr_chars"] for l in p.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not chars:
        return {}
    s = sorted(chars)
    return {
        "n": len(s),
        "median_chars": s[len(s) // 2],
        "min_chars": s[0],
        "n_empty": sum(1 for c in s if c == 0),
        "n_under_20_chars": sum(1 for c in s if c < 20),
    }


def main() -> int:
    arms = {}
    for asr_model in ["base", "large-v3"]:
        trials = load_arm(asr_model)
        if not trials:
            continue
        arms[asr_model] = {
            "system_under_test": f"Whisper {asr_model} (ASR) + Line 2 (zero-shot)",
            "asr_health": asr_health(asr_model),
            "metrics": compute_metrics(trials),
        }

    if not arms:
        print("尚無任何結果檔。")
        return 1

    for name, a in arms.items():
        m, h = a["metrics"], a["asr_health"]
        complete = "完整" if m["n_total"] >= 400 else f"**部分執行（{m['n_total']}/400）**"
        print(f"\n=== Whisper {name} —— {complete} ===")
        print(f"  已評分 {m['n_scored']}/{m['n_total']}（錯誤 {m['n_errors']}）"
              f"  詐騙 {m['n_fraud']} / 正常 {m['n_benign']}")
        print(f"  ASR 健康：中位數 {h['median_chars']} 字，最短 {h['min_chars']}，"
              f"空白 {h['n_empty']}，<20 字 {h['n_under_20_chars']}")
        print(f"  漏抓率 FAR : {m['FAR_missed_fraud']:.1%}  CI95 {m['FAR_ci95']}")
        print(f"  誤報率 FRR : {m['FRR_false_alarm']:.1%}  CI95 {m['FRR_ci95']}")
        print(f"  HTER       : {m['HTER']:.1%}")
        print(f"  混淆矩陣   : {m['confusion']}")

    delta = None
    if len(arms) == 2 and all(a["metrics"]["n_total"] >= 400 for a in arms.values()):
        b, l = arms["base"]["metrics"], arms["large-v3"]["metrics"]
        delta = {
            "HTER_base_minus_large": round(b["HTER"] - l["HTER"], 4),
            "FRR_base_minus_large": round(b["FRR_false_alarm"] - l["FRR_false_alarm"], 4),
            "FAR_base_minus_large": round(b["FAR_missed_fraud"] - l["FAR_missed_fraud"], 4),
            "interpretation": (
                "為正代表較弱的 ASR 拖累表現，其量級即 ASR 品質對本系統的敏感度。"
                "這是唯一一條不需要真人語料就能對電話頻寬說出點什麼的路徑："
                "電話頻寬會使 ASR 更差，而此處已量到系統對 ASR 品質的敏感度。"
            ),
        }
        print(f"\nASR 衰減（base - large-v3）："
              f"HTER {delta['HTER_base_minus_large']:+.1%}，"
              f"FRR {delta['FRR_base_minus_large']:+.1%}，"
              f"FAR {delta['FAR_base_minus_large']:+.1%}")
    elif len(arms) == 2:
        print("\n（雙臂尚未皆完成 400 筆，不計算 ASR 衰減——"
              "部分執行的兩臂樣本不同，相減無意義。）")

    out = {
        "provenance": stamp(),
        "no_baseline_comparison": (
            "不與該資料集已發表的 baseline 比較：對方為在該資料集上微調過的端到端"
            "音訊模型，我方為 zero-shot 且自行 ASR——受測系統形態不同。"
        ),
        "external_validity_limits": (
            "全部為 ChatTTS 合成寬頻音訊、中國大陸詐騙情境、zero-shot。"
            "**不預測** 8kHz 電話頻寬表現，也**不預測**台灣場景表現。"
            "條件在多數維度上皆比實際部署差，故若表現可用，是一個下限估計；反之不成立。"
        ),
        "arms": arms,
        "asr_degradation": delta,
    }
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n報表：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
