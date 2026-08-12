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


# Whisper 模型由弱到強。ASR 衰減 = 較弱臂 - 最強的已完成臂。
ASR_STRENGTH = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def discover_arms() -> list[str]:
    """找出磁碟上實際存在的臂，而不是寫死一組。

    `teleantifraud.py` 的 `--asr` 接受任意模型名，寫死 ["base", "large-v3"]
    會讓其他臂（例如 medium）**靜默地不出現在報表中**——跑了幾小時的結果
    看起來像沒跑過。依 PROTOCOL.md §4.3「沒有 n 的數字不得出現」的同一精神，
    跑過的臂也不該從報表裡消失。
    """
    found = {p.suffixes[-2].lstrip(".") for p in RESULT_PATH.parent.glob(
        f"{RESULT_PATH.stem}.*.jsonl") if len(p.suffixes) >= 2}
    known = [m for m in ASR_STRENGTH if m in found]
    # 不在已知強度表中的臂仍要報，只是無法排序強弱
    return known + sorted(found - set(known))


def main() -> int:
    arms = {}
    for asr_model in discover_arms():
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
        if m["n_total"] < 400:
            print(f"  ⚠ 本臂的數字**不可與完整臂比較**——語料未打亂，前段樣本"
                  f"難度與標籤比例都和整體不同（詐騙 {m['n_fraud']} / 正常 {m['n_benign']}，"
                  f"整體為 200/200）。要比較請跑 q1_mcnemar.py 的配對檢定。")
        print(f"  已評分 {m['n_scored']}/{m['n_total']}（錯誤 {m['n_errors']}）"
              f"  詐騙 {m['n_fraud']} / 正常 {m['n_benign']}")
        print(f"  ASR 健康：中位數 {h['median_chars']} 字，最短 {h['min_chars']}，"
              f"空白 {h['n_empty']}，<20 字 {h['n_under_20_chars']}")
        print(f"  漏抓率 FAR : {m['FAR_missed_fraud']:.1%}  CI95 {m['FAR_ci95']}")
        print(f"  誤報率 FRR : {m['FRR_false_alarm']:.1%}  CI95 {m['FRR_ci95']}")
        print(f"  HTER       : {m['HTER']:.1%}")
        print(f"  混淆矩陣   : {m['confusion']}")

    # ASR 衰減只在「皆完成 400 筆」的臂之間計算。部分執行的兩臂樣本不同，相減無意義。
    delta = None
    complete = [n for n, a in arms.items() if a["metrics"]["n_total"] >= 400]
    ranked = [m for m in ASR_STRENGTH if m in complete]
    if len(ranked) >= 2:
        strong, weak = ranked[-1], ranked[0]
        s, w = arms[strong]["metrics"], arms[weak]["metrics"]
        delta = {
            "weak_arm": weak,
            "strong_arm": strong,
            f"HTER_{weak}_minus_{strong}": round(w["HTER"] - s["HTER"], 4),
            f"FRR_{weak}_minus_{strong}": round(w["FRR_false_alarm"] - s["FRR_false_alarm"], 4),
            f"FAR_{weak}_minus_{strong}": round(w["FAR_missed_fraud"] - s["FAR_missed_fraud"], 4),
            "interpretation": (
                "為正代表較弱的 ASR 拖累表現，其量級即 ASR 品質對本系統的敏感度。"
                "這是唯一一條不需要真人語料就能對電話頻寬說出點什麼的路徑："
                "電話頻寬會使 ASR 更差，而此處已量到系統對 ASR 品質的敏感度。"
            ),
            "paired_test_required": (
                "本表僅為點估計之差。兩臂為同一批 400 筆音檔的配對設計，"
                "依 PROTOCOL_v2.md §4，宣稱差異顯著前必須跑 McNemar，"
                "不得比較獨立 CI。見 q1_mcnemar.py。"
            ),
        }
        print(f"\nASR 衰減（{weak} - {strong}）："
              f"HTER {delta[f'HTER_{weak}_minus_{strong}']:+.1%}，"
              f"FRR {delta[f'FRR_{weak}_minus_{strong}']:+.1%}，"
              f"FAR {delta[f'FAR_{weak}_minus_{strong}']:+.1%}")
        if len(ranked) > 2:
            print(f"（另有中間臂 {', '.join(ranked[1:-1])} 已完成，"
                  f"衰減取最弱與最強之差）")
    elif len(arms) >= 2:
        print("\n（尚未有兩個完成 400 筆的臂，不計算 ASR 衰減——"
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
