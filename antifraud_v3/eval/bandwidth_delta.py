"""Q2：從乾淨寬頻降到電話頻寬，效能衰減多少？

這是部署現實的核心問題——系統要跑在 8kHz PSTN 上，但所有可得的語料（自建的 23 檔、
以及 TeleAntiFraud-28k）都是乾淨寬頻合成語音。**目前沒有任何公開資料集能回答這個問題。**

作法：配對設計（paired design）。同一批音檔的寬頻原版與電話頻寬版，內容完全相同、
只有通道條件不同，因此兩者的差就是通道的效果，不混入內容差異。這比另外錄一批電話音檔
再跟原本的比乾淨得多——後者會把「內容不同」跟「通道不同」綁在一起無法分離。

量測鏈路：音檔 -> VAD -> ASR -> Line 2 判定。這樣量到的是**通道劣化對整條鏈路的
累積影響**，而不只是 ASR 的字錯率。對部署決策來說，後者才是要問的問題——ASR 掉幾個字
不重要，判定翻掉才重要。

同時輸出 ASR 逐字稿差異，因為若判定有變，要能分辨是「ASR 聽錯」還是「Line 2 判斷變了」。

前置：先產生電話頻寬語料
    python -m antifraud_v3.eval.telephony --in antifraud_v3/eval/test_clips \\
        --out antifraud_v3/eval/test_clips_phone --profile g711_alaw

用法：
    python -m antifraud_v3.eval.bandwidth_delta
"""

import argparse
import difflib
import json
import time
from pathlib import Path

from .provenance import stamp
from .teleantifraud import compute_metrics, Trial

EVAL_DIR = Path(__file__).parent
RESULT_PATH = EVAL_DIR / "bandwidth_delta_result.json"


def transcribe(path: Path) -> str:
    import soundfile as sf

    from ..asr.transcribe import transcribe_chunk
    from ..audio.vad import VADChunker

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    chunker = VADChunker()
    chunks = list(chunker.push_audio(y) or [])
    tail = chunker.flush()
    if tail is not None:
        chunks.append(tail)
    parts = []
    for c in chunks:
        if len(c) < 8000:
            continue
        t = transcribe_chunk(c, sr)
        if t:
            parts.append(t)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wideband", type=Path, default=EVAL_DIR / "test_clips")
    ap.add_argument("--narrowband", type=Path, default=EVAL_DIR / "test_clips_phone")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.narrowband.exists():
        print(f"找不到電話頻寬語料 {args.narrowband}——請先執行 eval.telephony 產生。")
        return 1

    from ..detectors.scam_semantic_llm import classify_call_via_llm
    from ..llm import get_llm_provider

    provider = get_llm_provider()
    pairs = []
    for wb in sorted(args.wideband.rglob("*.wav")):
        rel = wb.relative_to(args.wideband)
        nb = args.narrowband / rel
        if nb.exists():
            pairs.append((rel, wb, nb))
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"配對比較 {len(pairs)} 組（同內容、不同通道）\n")
    rows = []
    wb_trials, nb_trials = [], []

    for i, (rel, wb, nb) in enumerate(pairs):
        label = rel.parts[0] == "scam"
        entry = {"clip": str(rel), "label_is_scam": label}
        for tag, path, bucket in (("wideband", wb, wb_trials), ("narrowband", nb, nb_trials)):
            try:
                text = transcribe(path)
                r = classify_call_via_llm(provider, text)
                entry[tag] = {
                    "transcript": text,
                    "is_fraud": r.is_fraud,
                    "confidence": round(r.confidence, 3),
                    "fraud_type": r.fraud_type_raw,
                }
                bucket.append(Trial(i, label, r.is_fraud, r.confidence, r.fraud_type_raw))
            except Exception as e:
                entry[tag] = {"error": type(e).__name__}
                bucket.append(Trial(i, label, False, 0.0, None, error=type(e).__name__))
            time.sleep(0.3)

        wbe, nbe = entry.get("wideband", {}), entry.get("narrowband", {})
        if "transcript" in wbe and "transcript" in nbe:
            entry["asr_similarity"] = round(
                difflib.SequenceMatcher(None, wbe["transcript"], nbe["transcript"]).ratio(), 3
            )
        flipped = (
            "is_fraud" in wbe and "is_fraud" in nbe and wbe["is_fraud"] != nbe["is_fraud"]
        )
        entry["verdict_flipped"] = flipped
        rows.append(entry)
        mark = "  <- 判定翻轉" if flipped else ""
        print(f"  {str(rel):48s} ASR相似度 {entry.get('asr_similarity', '—')}{mark}")

    wb_m, nb_m = compute_metrics(wb_trials), compute_metrics(nb_trials)
    flips = [r["clip"] for r in rows if r["verdict_flipped"]]

    result = {
        "provenance": stamp(),
        "scope_note": (
            "配對設計：同內容不同通道。語料為 edge-tts 合成語音經 G.711 編碼，"
            "非真人真實 PSTN 錄音——量到的是通道劣化的效果，不是真實電話環境的全部效果。"
        ),
        "n_pairs": len(pairs),
        "wideband_metrics": wb_m,
        "narrowband_metrics": nb_m,
        "hter_delta_pp": (
            round((nb_m["HTER"] - wb_m["HTER"]) * 100, 2)
            if wb_m["HTER"] == wb_m["HTER"] and nb_m["HTER"] == nb_m["HTER"] else None
        ),
        "protocol_threshold_pp": 15,
        "verdict_flips": flips,
        "mean_asr_similarity": round(
            sum(r["asr_similarity"] for r in rows if "asr_similarity" in r)
            / max(sum(1 for r in rows if "asr_similarity" in r), 1), 3),
        "pairs": rows,
    }

    print(f"\n{'='*56}")
    print(f"{'':12s} {'HTER':>8s} {'FAR(漏抓)':>10s} {'FRR(誤報)':>10s}")
    print(f"{'寬頻':12s} {wb_m['HTER']:8.1%} {wb_m['FAR_missed_fraud']:10.1%} {wb_m['FRR_false_alarm']:10.1%}")
    print(f"{'電話頻寬':12s} {nb_m['HTER']:8.1%} {nb_m['FAR_missed_fraud']:10.1%} {nb_m['FRR_false_alarm']:10.1%}")
    d = result["hter_delta_pp"]
    print(f"\nHTER 變化：{d:+.2f} 個百分點（協定門檻 §6：惡化 >15pp 即列為部署阻斷項）")
    print(f"判定翻轉：{len(flips)} 組 {flips or ''}")
    print(f"ASR 平均相似度：{result['mean_asr_similarity']}")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
