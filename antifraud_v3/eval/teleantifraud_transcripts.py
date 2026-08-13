"""把 TeleAntiFraud 的 400 筆音檔轉成逐字稿並快取。

**為什麼需要獨立一步**：Q1 的 `teleantifraud.py` 只把 `len(text)` 存進 JSONL（`asr_chars`），
沒有存文字本身。任何後續的文字層實驗（例如 `script_features.py`）都得重跑一次 ASR，
而 400 筆在 CPU 上是以小時計的。

**為什麼不直接改 `teleantifraud.py`**：那支 harness 已經跑完並提交了結果（`q1_report.json`），
改它會讓已提交的數字與產生它的程式碼不再對應。新增一支只做轉寫的腳本，
不動已完成的實驗——沿用 `make_transcripts.py` 對本地語料的同一套做法。

轉寫方式與 Q1 的 base 臂一致（同一個 `_transcribe`、同一個模型），因此輸出的文字
就是 Line 2 在 Q1 中實際看到的文字，兩個實驗的輸入可對齊。

逐筆 append，可續跑——中途掛掉不必從頭來。

用法：
    python -m antifraud_v3.eval.teleantifraud_transcripts
    python -m antifraud_v3.eval.teleantifraud_transcripts --asr base --limit 400
"""

import argparse
import json
import time
from pathlib import Path

from .provenance import stamp
from .teleantifraud import AUDIO_ROOT, _transcribe, load_split

OUT_PATH = Path(__file__).parent / "teleantifraud_transcripts.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="base", help="與 Q1 base 臂一致，預設 base")
    ap.add_argument("--limit", type=int, default=None,
                    help="預設 None＝全部 400 筆。load_split 在給 limit 時會隨機取樣，"
                         "而本步驟要的是全量，故預設不給。")
    args = ap.parse_args()

    rows = load_split(args.limit)

    done: set[int] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["index"])
        print(f"續跑：已有 {len(done)} 筆，跳過")

    t0 = time.time()
    n_new = 0
    with OUT_PATH.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(rows, 1):
            if row["index"] in done:
                continue
            audio = AUDIO_ROOT / row["audio"]
            try:
                text = _transcribe(audio, args.asr)
                rec = {"index": row["index"], "label": row["label"],
                       "transcript": text, "asr_chars": len(text), "error": None}
            except Exception as e:
                rec = {"index": row["index"], "label": row["label"],
                       "transcript": "", "asr_chars": 0,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 10 == 0:
                rate = (time.time() - t0) / n_new
                left = (len(rows) - n) * rate
                print(f"  [{args.asr}] {n}/{len(rows)}  ({rate:.1f}s/筆，剩約 {left/60:.0f} 分)")

    meta = OUT_PATH.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "provenance": stamp(),
        "asr_model": args.asr,
        "n": sum(1 for line in OUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()),
        "note": ("轉寫方式與 Q1 base 臂一致（同一個 _transcribe），"
                 "故此處的文字即 Line 2 在 Q1 中實際看到的文字。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n逐字稿：{OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
