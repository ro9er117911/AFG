"""把測試音檔轉成逐字稿並快取，供對抗性測試等文字層評測重複使用。

分開成獨立步驟的理由：ASR 在 CPU 上很慢（實測整套語料數分鐘），而對抗性測試要對同一份
逐字稿反覆改寫、反覆分類。每次都重跑 Whisper 是浪費，而且會讓不同次實驗的輸入不一致
——快取下來才能保證「改寫前後餵給分類器的原文完全相同」，否則測到的是 ASR 抖動。

輸出的逐字稿本身也是評測產出物的一部分：論文要能讓第三方看到我們實際餵給模型的是什麼
文字，而不是只有最後的數字。

用法：
    python -m antifraud_v3.eval.make_transcripts --category scam
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from ..asr.transcribe import transcribe_chunk
from ..audio.vad import VADChunker

SAMPLE_RATE = 16000
CLIPS_DIR = Path(__file__).parent / "test_clips"


def transcribe_clip(path: Path) -> str:
    """用與正式 pipeline 相同的 VAD 切法逐段轉錄再接起來，而不是整檔一次餵給 ASR
    ——後者會得到跟系統實際處理不同的文字，測出來的東西就不是系統的行為。"""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    chunker = VADChunker()
    chunks = list(chunker.push_audio(y) or [])
    tail = chunker.flush()
    if tail is not None:
        chunks.append(tail)

    parts = []
    for chunk in chunks:
        if len(chunk) < SAMPLE_RATE // 2:
            continue
        text = transcribe_chunk(chunk, sr)
        if text:
            parts.append(text)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="scam", choices=["scam", "benign"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or Path(__file__).parent / (
        "adversarial_transcripts.json" if args.category == "scam" else f"{args.category}_transcripts.json"
    )

    src = CLIPS_DIR / args.category
    transcripts = {}
    for path in sorted(src.glob("*.wav")):
        text = transcribe_clip(path)
        if not text:
            print(f"  {path.name}: 沒有轉錄出內容，跳過")
            continue
        transcripts[path.name] = text
        print(f"  {path.name}: {len(text)} 字")

    out.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(transcripts)} 份逐字稿 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
