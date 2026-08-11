"""測試 within-call 聲學基準的穩定性——一個針對我方自己設計的證偽測試。

背景：`pipeline/call_state.py` 的 `maybe_set_baseline()` 用**第二個語音 chunk** 當作整通
電話的聲學基準，之後所有「相對基準上升/下降」的呈現都以它為準。

文獻支持的是「speaker normalization 優於絕對閾值」這個大方向，但**沒有任何文獻驗證過
「用單一個開頭 chunk 當基準」的穩定性**（見 references/13-engineering-actions.md 附錄，
我方自己已載明這是未驗證假設）。

如果基準本身隨機性很大，那麼「F0 較基準上升 12%」這句話就不是觀察，而是噪音——
UI 上所有相對數字的意義都要重新評估。

作法：對每通電話，把所有 chunk 的聲學特徵都算出來，然後問「如果基準取的是第 k 個 chunk
而不是第 2 個，數值會差多少」。用變異係數（CV = 標準差/平均）量化。

判定門檻在 docs/eval/PROTOCOL.md §6 事前訂定：**CV > 20% 即判定不穩定**，
需加長視窗或降低聲學面板的宣稱強度。

用法：
    python -m antifraud_v3.eval.baseline_stability
"""

import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.features import extract_acoustic_features
from ..audio.vad import VADChunker

SAMPLE_RATE = 16000
CLIPS_DIR = Path(__file__).parent / "test_clips"

# 只看有數值意義、且 UI 真的拿來跟基準比的欄位。
TRACKED = ["mean_pitch", "jitter_local", "shimmer_local", "hnr", "pause_ratio"]

# PROTOCOL.md §6 事前訂定的門檻，不在看到結果之後才決定。
CV_UNSTABLE_THRESHOLD = 0.20


def _flatten(feat: dict) -> dict[str, float]:
    """extract_acoustic_features 回傳巢狀結構（pitch/volume/tremor/speech_rate），
    這裡攤平成 UI 實際顯示的那幾個純量。"""
    out = {}
    p, t = feat.get("pitch", {}), feat.get("tremor", {})
    out["mean_pitch"] = p.get("mean_pitch", 0.0)
    out["jitter_local"] = t.get("jitter_local", 0.0)
    out["shimmer_local"] = t.get("shimmer_local", 0.0)
    out["hnr"] = t.get("hnr", 0.0)
    out["pause_ratio"] = feat.get("speech_rate", {}).get("pause_ratio", 0.0)
    return out


def chunk_features(path: Path) -> list[dict[str, float]]:
    """用跟正式 pipeline 同一個 VAD 切法，否則測的就不是真實情境。"""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    chunker = VADChunker()
    chunks = list(chunker.push_audio(y) or [])
    tail = chunker.flush()
    if tail is not None:
        chunks.append(tail)

    feats = []
    for chunk in chunks:
        # praat 的 pitch floor 75Hz 需要至少 ~6.4 個週期的音訊才跑得動；太短的 chunk 會
        # 直接拋 PraatError。跳過而非補零——補零會製造假的靜音特徵，汙染變異數統計。
        if len(chunk) < SAMPLE_RATE // 2:
            continue
        try:
            feats.append(_flatten(extract_acoustic_features(chunk, sr)))
        except Exception:
            continue
    return feats


def analyze() -> dict:
    per_metric: dict[str, list[float]] = {m: [] for m in TRACKED}
    clips = []

    for path in sorted(CLIPS_DIR.rglob("*.wav")):
        feats = chunk_features(path)
        if len(feats) < 3:
            # 少於 3 個 chunk 就無從比較「換一個 chunk 當基準」會怎樣。
            continue
        row = {"clip": str(path.relative_to(CLIPS_DIR)), "n_chunks": len(feats)}
        for m in TRACKED:
            vals = [f[m] for f in feats if f[m]]
            if len(vals) < 3:
                continue
            mean = statistics.fmean(vals)
            cv = statistics.stdev(vals) / mean if mean else float("nan")
            row[m] = {"cv": round(cv, 4), "min": round(min(vals), 2), "max": round(max(vals), 2)}
            if not np.isnan(cv):
                per_metric[m].append(cv)
        clips.append(row)

    summary = {}
    for m, cvs in per_metric.items():
        if not cvs:
            continue
        median_cv = statistics.median(cvs)
        summary[m] = {
            "median_cv": round(median_cv, 4),
            "max_cv": round(max(cvs), 4),
            "n_clips": len(cvs),
            "unstable": median_cv > CV_UNSTABLE_THRESHOLD,
        }
    return {"threshold": CV_UNSTABLE_THRESHOLD, "summary": summary, "per_clip": clips}


def main() -> int:
    result = analyze()
    print(f"判定門檻（PROTOCOL.md §6 事前訂定）：CV > {CV_UNSTABLE_THRESHOLD:.0%} 即判定不穩定\n")
    print(f"{'特徵':16s} {'中位數 CV':>10s} {'最大 CV':>10s} {'n':>4s}  判定")
    for m, s in result["summary"].items():
        verdict = "不穩定" if s["unstable"] else "穩定"
        print(f"{m:16s} {s['median_cv']:9.1%} {s['max_cv']:9.1%} {s['n_clips']:4d}  {verdict}")

    unstable = [m for m, s in result["summary"].items() if s["unstable"]]
    print()
    if unstable:
        print(f"結論：{len(unstable)} 項不穩定（{', '.join(unstable)}）——"
              f"這些欄位的「相對基準」呈現不可靠，須加長基準視窗或降低宣稱強度。")
    else:
        print("結論：所有追蹤特徵的通話內變異都在門檻內。")

    out = Path(__file__).parent / "baseline_stability_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
