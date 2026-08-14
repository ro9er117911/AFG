"""Q7：Line 2 判錯的案例，錯在哪一層？逐字稿丟失了什麼？

**這一題要回答的產品問題**：聲學層目前不參與判定（設計立場見 PROTOCOL.md §3）。
若要主張「聲學層有獨立價值」，必須先證明**逐字稿丟失了判定所需的資訊**——
否則聲學層就只是輔助顯示，不該進入判決。

反過來說，若 Line 2 的錯誤全都源自「文字裡本來就沒有足夠證據」（例如通話在
索取階段之前就結束），那再多的聲學特徵也救不回來，因為那不是模態問題，
是**這通電話還沒發生足以判定的事**。

**為什麼這比「再訓練一個 detector」有價值**：它回答「該不該做」，不是「做得多好」。

## 方法

對 Q1 base 臂判錯的每一筆，做三件事，全部不需要新語料：

  L1 **錯誤歸因**：讓 LLM 讀逐字稿並被告知真實標籤，判斷錯誤屬於哪一類：
     - `asr_garbled`：ASR 錯字使關鍵詞無法辨識（文字裡有東西，但被轉錯）
     - `insufficient_content`：通話內容本身不含足以判定的證據（話術未進行到關鍵階段）
     - `semantic_misjudgement`：文字證據充分，是判斷失誤
     - `label_questionable`：資料集標籤本身可疑
     這四類的**處置完全不同**，混在一起看就只會得到「錯了 11 筆」這種無用資訊。

  L2 **可救性**：若該筆為 `asr_garbled`，用較強的 ASR（Q1 的 medium 臂）重看，
     錯誤是否消失？兩臂的判定已經在 JSONL 裡，直接查表即可，不需重跑。

  L3 **聲學可及性**：判定所需的證據，是否**原則上**存在於音訊而不存在於文字？
     這是本題的核心。刻意用「原則上」——本檔不聽音訊、不抽聲學特徵，
     只由 LLM 依逐字稿內容推論「若要判對，需要什麼資訊」，並分類該資訊的性質：
     - `in_text_already`：文字裡就有，不需音訊
     - `needs_audio_signal`：需要音訊才有（如合成語音痕跡、說話者身分、通道特性）
     - `needs_neither`：兩者都沒有（通話內容不足）

## 這一題不宣稱什麼

- **n=12，不做統計推論。** 這是**假設產生**用的個案分析，不是效能評測。
  任何「X% 的錯誤源自 Y」的句子都不得寫進論文。
- **不宣稱聲學層有效。** 本題最好的結果是「指出聲學層可能有價值的具體案例」，
  距離「聲學層有效」還隔著一個需要真人語料的實驗。
- **L3 為 LLM 的推論，非實測。** 沒有聽音訊，沒有抽特徵。它產生的是待驗證的
  假設清單，不是證據。

用法：
    python -m antifraud_v3.eval.error_provenance
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import get_llm_provider
from .provenance import stamp
from .teleantifraud import RESULT_PATH as TA_RESULT
from .teleantifraud_transcripts import OUT_PATH as TRANSCRIPTS

REPORT_PATH = Path(__file__).parent / "error_provenance.json"

ERROR_CLASSES = {
    "asr_garbled": "ASR 錯字使關鍵詞無法辨識——文字裡本來有，但被轉錯",
    "insufficient_content": "通話內容本身不含足以判定的證據，例如話術尚未進行到索取階段",
    "semantic_misjudgement": "文字證據充分，屬判斷失誤",
    "label_questionable": "資料集的標籤本身可疑",
}

EVIDENCE_LOCATIONS = {
    "in_text_already": "判對所需的資訊，逐字稿裡就有",
    "needs_audio_signal": "需要音訊才取得（合成語音痕跡、說話者身分、通道特性、語者重疊等）",
    "needs_neither": "音訊與文字都沒有——這通電話尚未發生足以判定的事",
}


class ErrorAnalysis(BaseModel):
    # 用 Literal 而非 str：自由字串會讓 LLM 回傳清單外的類別（例如自創一個
    # "partially_garbled"），統計時就得事後歸併，而歸併規則本身會變成一個
    # 沒人稽核的判斷。讓 schema 在生成階段就擋掉。
    error_class: Literal["asr_garbled", "insufficient_content",
                         "semantic_misjudgement", "label_questionable"] = Field(
        description="錯誤歸因，必須是四類之一")
    error_class_reason: str = Field(description="一到兩句，引用逐字稿具體內容")
    evidence_location: Literal["in_text_already", "needs_audio_signal",
                               "needs_neither"] = Field(
        description="判對所需資訊的所在，必須是三類之一")
    what_would_be_needed: str = Field(description="要判對，需要什麼資訊？具體描述")
    garbled_terms: list[str] = Field(
        default_factory=list,
        description="若為 asr_garbled，列出疑似被轉錯的關鍵詞（原文錯字形式）")


SYSTEM_PROMPT = f"""你在分析一個詐騙偵測系統的**錯誤案例**。系統只讀逐字稿判斷是否為詐騙，
現在給你逐字稿、系統的判斷、以及**真實標籤**，請診斷錯誤的來源。

## 錯誤歸因（error_class，四選一）
{chr(10).join(f"- {k}：{v}" for k, v in ERROR_CLASSES.items())}

## 判對所需資訊的所在（evidence_location，三選一）
{chr(10).join(f"- {k}：{v}" for k, v in EVIDENCE_LOCATIONS.items())}

注意：本語料的逐字稿由 ASR 產生，**含大量同音錯字**（例如「客服」被轉成「克服」、
「積分」被轉成「機分」）。判斷 asr_garbled 時，要區分「錯字但語意仍可還原」
與「錯字使關鍵資訊真的無法辨識」——前者不算 asr_garbled。

誠實作答。若這通電話的內容本身就不足以判定，就選 insufficient_content 或
needs_neither，不要為了給出有用的答案而硬說需要音訊。"""


def main() -> int:
    transcripts = {}
    for line in TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            transcripts[d["index"]] = d

    arms = {}
    for arm in ("base", "medium"):
        p = TA_RESULT.with_suffix(f".{arm}.jsonl")
        if not p.exists():
            continue
        arms[arm] = {json.loads(l)["index"]: json.loads(l)
                     for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    if "base" not in arms:
        raise SystemExit("找不到 Q1 base 臂結果。")

    errors = [r for r in arms["base"].values()
              if r.get("error") is None and bool(r["predicted"]) != bool(r["label"])]
    errors.sort(key=lambda r: r["index"])

    provider = get_llm_provider()
    rows = []
    for i, e in enumerate(errors, 1):
        idx = e["index"]
        t = transcripts.get(idx, {})
        text = t.get("transcript", "")
        truth = "詐騙" if e["label"] else "正常"
        pred = "詐騙" if e["predicted"] else "正常"
        user = (f"逐字稿：\n{text}\n\n"
                f"系統判斷：{pred}\n真實標籤：{truth}\n"
                f"（此為 {'誤報' if not e['label'] else '漏抓'}）")
        try:
            a = provider.structured_complete(system=SYSTEM_PROMPT, user_content=user,
                                             schema=ErrorAnalysis)
            rec = {
                "index": idx,
                "label": "scam" if e["label"] else "benign",
                "predicted": "scam" if e["predicted"] else "benign",
                "error_type": "false_negative" if e["label"] else "false_positive",
                "asr_chars": t.get("asr_chars"),
                "error_class": a.error_class,
                "error_class_reason": a.error_class_reason,
                "evidence_location": a.evidence_location,
                "what_would_be_needed": a.what_would_be_needed,
                "garbled_terms": a.garbled_terms,
                # L2：較強的 ASR 是否修正了這一筆？兩臂結果已存在，查表即可
                "fixed_by_stronger_asr": (
                    None if "medium" not in arms or idx not in arms["medium"]
                    else bool(arms["medium"][idx]["predicted"]) == bool(e["label"])),
                "transcript": text,
            }
        except Exception as exc:
            rec = {"index": idx, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(f"  [{i}/{len(errors)}] idx {idx} "
              f"{rec.get('error_class', rec.get('error', '?'))}")

    ok = [r for r in rows if "error_class" in r]
    tally = lambda field: {k: sum(1 for r in ok if r[field] == k)
                           for k in sorted({r[field] for r in ok})}

    out = {
        "provenance": stamp(),
        "scope_note": (
            "n=12 的個案分析，用於**產生假設**，非效能評測。"
            "不得書寫任何『X% 的錯誤源自 Y』式的統計陳述。"
        ),
        "method_limits": [
            "L3（證據所在）為 LLM 依逐字稿的推論，**未聽音訊、未抽任何聲學特徵**——"
            "產出的是待驗證假設，不是證據。",
            "分析時已告知 LLM 真實標籤，故存在事後合理化的風險："
            "知道答案再解釋為何錯，比事前判斷容易。",
            "本語料為 ChatTTS 合成之中國大陸情境音訊，"
            "asr_garbled 的比例不可外推至真人電話語料。",
        ],
        "n_errors_analysed": len(ok),
        "n_failed": len(rows) - len(ok),
        "error_class_tally": tally("error_class"),
        "evidence_location_tally": tally("evidence_location"),
        "fixed_by_stronger_asr": {
            "yes": sum(1 for r in ok if r["fixed_by_stronger_asr"] is True),
            "no": sum(1 for r in ok if r["fixed_by_stronger_asr"] is False),
            "unknown": sum(1 for r in ok if r["fixed_by_stronger_asr"] is None),
        },
        "acoustic_layer_candidates": [
            {"index": r["index"], "error_type": r["error_type"],
             "what_would_be_needed": r["what_would_be_needed"]}
            for r in ok if r["evidence_location"] == "needs_audio_signal"
        ],
        "class_definitions": ERROR_CLASSES,
        "location_definitions": EVIDENCE_LOCATIONS,
        "records": rows,
    }
    REPORT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 錯誤歸因（n={len(ok)}）===")
    for k, v in out["error_class_tally"].items():
        print(f"  {v:2}  {k:24} {ERROR_CLASSES.get(k, '')}")
    print(f"\n=== 判對所需資訊的所在 ===")
    for k, v in out["evidence_location_tally"].items():
        print(f"  {v:2}  {k:24} {EVIDENCE_LOCATIONS.get(k, '')}")
    f = out["fixed_by_stronger_asr"]
    print(f"\n較強 ASR（medium）修正了：{f['yes']} 筆，未修正 {f['no']} 筆")
    print(f"\n聲學層的候選案例：{len(out['acoustic_layer_candidates'])} 筆")
    for c in out["acoustic_layer_candidates"]:
        print(f"  idx {c['index']} ({c['error_type']}): {c['what_would_be_needed'][:80]}")
    print(f"\n報表：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
