"""對抗性改寫測試：詐騙集團也在用 LLM，話術會被改寫來規避偵測。

文獻依據：Li et al. (2025, arXiv:2507.16291) 證明用 GPT-4o 改寫詐騙逐字稿（保留語意、
換句話說），可讓分類器準確率下降 **30.96%**。

這不是可以修好的 bug，是需要持續監控的風險——所以本測試建立的是**基線與監控指標**，
不是解法。評測協定 Q3。

作法：對每份詐騙逐字稿，用 LLM 以三種強度改寫，再送進 Line 2 的純文字路徑
（`classify_call_via_llm`），量測 is_fraud 命中率隨改寫強度的衰減曲線。

為什麼測純文字路徑而非音訊路徑：改寫的是**話術內容**，這是語意層的攻擊。用純文字路徑
可以把語意層單獨隔離出來，不被 ASR 誤差混淆。若要測完整音訊鏈，需要把改寫後的文字重新
合成語音，那會引入 TTS 音色這個額外變因（且我方語料本來就已經是合成語音，見
PROTOCOL.md §5.1 的限制說明）。

**倫理與用途界線**：本模組的用途是量測我方系統的脆弱性，屬防禦性安全測試。改寫產生的
文字僅用於離線評測，不對外散布。

用法：
    python -m antifraud_v3.eval.adversarial --clips 5
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .provenance import stamp
from ..detectors.scam_semantic_llm import classify_call_via_llm
from ..llm import LLMProviderError, get_llm_provider

TRANSCRIPTS_PATH = Path(__file__).parent / "adversarial_transcripts.json"
BENIGN_PATH = Path(__file__).parent / "benign_transcripts.json"
RESULT_PATH = Path(__file__).parent / "adversarial_result.json"

# 三種強度，對應不同的對抗成本。詐騙集團越願意投入，改寫越徹底。
STRENGTHS = {
    "none": None,  # 對照組：原文，不改寫
    "light": "改寫下面這段電話逐字稿，換掉明顯的關鍵詞（例如「驗證碼」「圈存」「安全帳戶」），"
             "但保留完全相同的意圖與情節。維持口語、自然，長度相近。只輸出改寫後的逐字稿。",
    "heavy": "改寫下面這段電話逐字稿，完全換一套說法：不同的用詞、不同的句型、不同的鋪陳順序，"
             "但保留完全相同的意圖與情節，讓聽的人受到一樣的影響。維持口語、自然。"
             "只輸出改寫後的逐字稿。",
}


class Rewritten(BaseModel):
    transcript: str


@dataclass
class Trial:
    clip: str
    strength: str
    is_fraud: bool
    confidence: float
    fraud_type: str | None
    label_is_scam: bool = True  # benign 對照組為 False


def rewrite(provider, transcript: str, instruction: str) -> str:
    result = provider.structured_complete(
        system="你是語料處理工具，負責改寫電話逐字稿以測試偵測系統的強健性。"
               "忠實保留原文的意圖與情節，只改變表達方式。",
        user_content=f"{instruction}\n\n---\n{transcript}",
        schema=Rewritten,
    )
    return result.transcript


def run(scam: dict[str, str], benign: dict[str, str], limit: int | None) -> dict:
    """scam 與 benign 都跑同樣的改寫流程。

    benign 對照組是必要的，不是加分項：只測 scam 的話，「偵測率下降」有一個
    無法排除的替代解釋——**改寫可能只是讓分類器整體變保守**。若 benign 同時被
    大量改判為詐騙，衰減的成因就完全不同。一個把所有東西都判成詐騙的分類器，
    可以在只測 scam 的設計下輕鬆拿到 100%。
    """
    provider = get_llm_provider()
    trials: list[Trial] = []

    for label_is_scam, items in ((True, scam), (False, benign)):
        tag = "scam" if label_is_scam else "benign"
        rows = list(items.items())[:limit]
        print(f"\n[{tag}] {len(rows)} 份")
        for name, original in rows:
            for strength, instruction in STRENGTHS.items():
                text = original
                if instruction is not None:
                    try:
                        text = rewrite(provider, original, instruction)
                    except LLMProviderError as e:
                        print(f"  [{name} / {strength}] 改寫失敗，跳過：{e}")
                        continue
                try:
                    r = classify_call_via_llm(provider, text)
                except Exception as e:
                    print(f"  [{name} / {strength}] 分類失敗，跳過：{e}")
                    continue
                trials.append(Trial(name, strength, r.is_fraud, r.confidence,
                                    r.fraud_type_raw, label_is_scam))
                print(f"  {name:42s} {strength:6s} is_fraud={r.is_fraud} conf={r.confidence:.2f}")
                time.sleep(0.5)  # 對 CLI provider 客氣一點，避免連續打爆

    by_strength: dict[str, dict] = {}
    for s in STRENGTHS:
        sc = [t for t in trials if t.strength == s and t.label_is_scam]
        bn = [t for t in trials if t.strength == s and not t.label_is_scam]
        if not sc:
            continue
        hits = sum(1 for t in sc if t.is_fraud)
        false_alarms = sum(1 for t in bn if t.is_fraud)
        by_strength[s] = {
            "n_scam": len(sc),
            "detected": hits,
            "detection_rate": round(hits / len(sc), 4),
            "n_benign": len(bn),
            "false_alarms": false_alarms,
            # 這一欄是關鍵：若它隨改寫強度上升，代表分類器變寬鬆而非變準；
            # 若它維持在 0，「改寫讓分類器整體變保守」的替代解釋就被排除。
            "false_alarm_rate": round(false_alarms / len(bn), 4) if bn else None,
            "mean_conf_correct": round(
                sum(t.confidence for t in sc if t.is_fraud) / max(hits, 1), 4),
            "mean_conf_wrong": round(
                sum(t.confidence for t in sc if not t.is_fraud) / max(len(sc) - hits, 1), 4),
        }

    base = by_strength.get("none", {}).get("detection_rate")
    for s, agg in by_strength.items():
        if base:
            agg["relative_drop"] = round((base - agg["detection_rate"]) / base, 4)

    return {
        "provenance": stamp(),
        "reference": "Li et al. 2025, arXiv:2507.16291 — GPT-4o 改寫致準確率下降 30.96%",
        "protocol_threshold": 0.30,
        "known_limitation": (
            "改寫者與分類器共用同一個 provider 單例（同模型），攻防同源使數字方向與"
            "量級皆未知。此為已揭露的方法論限制，見 PAPER.md §5.3。"
        ),
        "by_strength": by_strength,
        "trials": [t.__dict__ for t in trials],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=None, help="限制測試的逐字稿數（省時用）")
    args = ap.parse_args()

    for p, what in ((TRANSCRIPTS_PATH, "scam"), (BENIGN_PATH, "benign")):
        if not p.exists():
            print(f"缺少逐字稿：{p}")
            print(f"請先執行 python -m antifraud_v3.eval.make_transcripts --category {what}")
            return 1

    scam = json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
    benign = json.loads(BENIGN_PATH.read_text(encoding="utf-8"))
    print(f"對抗性改寫測試：詐騙 {len(scam)} + 正常 {len(benign)} 份 x {len(STRENGTHS)} 種強度")
    result = run(scam, benign, args.clips)

    print(f"\n{'強度':8s} {'偵測率':>9s} {'相對衰減':>9s} {'誤報率':>9s} {'信心(對)':>9s} {'信心(錯)':>9s}")
    for s, agg in result["by_strength"].items():
        drop = agg.get("relative_drop")
        far = agg.get("false_alarm_rate")
        print(f"{s:8s} {agg['detection_rate']:9.1%} "
              f"{(f'{drop:.1%}' if drop is not None else '—'):>9s} "
              f"{(f'{far:.1%}' if far is not None else '—'):>9s} "
              f"{agg['mean_conf_correct']:9.2f} {agg['mean_conf_wrong']:9.2f}")

    # 替代解釋的檢定：誤報率若隨強度上升，衰減就不是「被規避」而是「變寬鬆」
    fars = [a.get("false_alarm_rate") for a in result["by_strength"].values()]
    if all(f is not None for f in fars):
        if max(fars) > min(fars) + 0.1:
            print("\n[注意] 誤報率隨改寫強度明顯變動——"
                  "「改寫使分類器整體變保守」的替代解釋無法排除。")
        else:
            print("\n誤報率未隨改寫強度顯著變動——"
                  "「改寫只是讓分類器整體變寬鬆」的替代解釋可排除。")

    worst = max((a.get("relative_drop", 0) for a in result["by_strength"].values()), default=0)
    print()
    if worst > result["protocol_threshold"]:
        print(f"結論：最大相對衰減 {worst:.1%} 超過協定門檻 30%——"
              f"列為持續性風險，需附監控計畫。")
    else:
        print(f"結論：最大相對衰減 {worst:.1%}，未超過協定門檻 30%。"
              f"注意樣本數小，見 PROTOCOL.md §4.3。")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
