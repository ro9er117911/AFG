"""把寬頻錄音轉成電話頻寬，用來隔離「通道條件」這個變因。

現有 eval/test_clips/ 全部是 edge-tts 合成的乾淨寬頻語音（實測 23/23 皆為 wideband），
但部署目標是 8kHz PSTN 窄頻。兩者之間效能差多少，是評測協定的 Q2。

這個模組產生的是「同內容、不同通道」的配對語料：同一句話的寬頻版與電話頻寬版可以直接
比較，把頻寬的影響從內容的影響裡分離出來。這是 quasi-experimental 的配對設計，比另外
錄一批電話音檔再跟原本的比更乾淨——後者會把「內容不同」跟「通道不同」混在一起。

**這不能取代真人電話錄音。** 合成語音經過編碼器，仍然是合成語音經過編碼器；真實 PSTN
還有背景噪音、迴音消除、自動增益、封包抖動、以及真人講話本身的變異。這個模組的定位是
「在拿到真人語料之前，先把頻寬這一個變因量出來」，不是終點。見 docs/eval/PROTOCOL.md §5.2
語料 B 與語料 C 的分工。

用法：
    python -m antifraud_v3.eval.telephony --in eval/test_clips --out eval/test_clips_phone
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 8000

# 三種真實電信網路會用到的編碼，難度遞增。分開產生而不是疊在一起，因為要能回答
# 「是哪一種編碼造成衰減」，混在一起就只知道「有衰減」。
#   g711_alaw  — 歐洲/亞洲 PSTN 主流，8kHz，最基本的窄頻情境
#   g722       — 寬頻語音編碼（HD voice），16kHz，用來對照「窄頻本身」的影響
#   opus_nb    — VoIP 常見，可設低位元率，模擬網路電話品質
# 中繼副檔名要給 ffmpeg 一個它認得的容器，否則它無法決定 muxer 而直接失敗。
PROFILES = {
    "g711_alaw": (["-ar", "8000", "-acodec", "pcm_alaw"], ".wav"),
    "g722": (["-ar", "16000", "-acodec", "g722"], ".wav"),
    "opus_nb": (["-ar", "8000", "-acodec", "libopus", "-b:a", "16k", "-application", "voip"], ".ogg"),
}


def _sha256(path: Path) -> str:
    """語料的 checksum 要進論文——第三方要能確認拿到的是同一份檔案。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def degrade(src: Path, dst: Path, profile: str) -> None:
    """單向轉換：寬頻 -> 電話編碼 -> 轉回 16kHz wav。

    最後轉回 16kHz 是必要的，不是多此一舉：pipeline 下游（faster-whisper、
    parselmouth、eGeMAPS）都預期 16kHz 輸入。我們要模擬的是「這段音訊經歷過電話通道」，
    不是「pipeline 收到 8kHz 檔案」——真實系統收到的也是電信網路解碼後的訊號。
    重採樣回來並不會把編碼已經丟掉的高頻資訊變回來，所以頻寬限制的效果保留著。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec_args, container = PROFILES[profile]
    encoded = dst.with_suffix(".encoded" + container)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *codec_args, str(encoded)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(encoded),
             "-ar", str(16000), "-ac", "1", str(dst)],
            check=True,
        )
    finally:
        encoded.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--profile", default="g711_alaw", choices=sorted(PROFILES))
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("需要 ffmpeg，但 PATH 上找不到。", file=sys.stderr)
        return 1

    manifest = []
    for src in sorted(args.src.rglob("*.wav")):
        rel = src.relative_to(args.src)
        out = args.dst / rel
        degrade(src, out, args.profile)
        manifest.append({
            "source": str(rel),
            "profile": args.profile,
            "source_sha256": _sha256(src),
            "output_sha256": _sha256(out),
        })
        print(f"  {rel}")

    # manifest 跟語料一起進版本控制：論文說「我們用了這些檔案」要能被查證，
    # 而不是只有一句「我們轉換了測試集」。
    manifest_path = args.dst / "manifest.json"
    manifest_path.write_text(
        json.dumps({"profile": args.profile, "sample_rate": SAMPLE_RATE, "clips": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(manifest)} 個檔案 -> {args.dst}（manifest: {manifest_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
