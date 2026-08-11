"""在 TeleAntiFraud-28k 官方 test split 上評測 Line 2 與融合層。

這是評測協定的 Q1：在**有統計意義的樣本數**上量測鑑別力。現有自建語料只有 23 筆
（PROTOCOL.md §4.3 明訂 n<100 不得以準確率語氣陳述結論），這個資料集的官方 test split
有 7,021 筆，是目前唯一能讓 Q1 得到可信答案的來源。

**這個資料集不能回答 Q2（電話頻寬）。** 論文確認全部音訊經 ChatTTS 重新合成，
連源自真實通話的 DS1 部分也是匿名化後重新生成，且論文未載明取樣率或頻寬規格，
亦無任何電話頻寬模擬。它與我方現有語料有**相同的寬頻合成語音限制**。
Q2 仍須依賴 eval/telephony.py 產生的語料 B 與真人電話錄音（語料 C）。

與論文 baseline 的可比性——必須誠實標註：
  論文報 Fraud detection 任務 F1：base 58.51 -> fine-tuned 84.78
  但論文**未載明 F1 是 macro / micro / weighted**，因此我方計算的 F1 與其數字
  **非嚴格可比**。本模組同時輸出三種 F1，讓讀者自行判斷，並以 HTER/FRR/FAR
  作為主要指標（PROTOCOL.md §4.2）——那些定義無歧義。

前置作業：資料集為 gated，需先取得存取權。
    hf auth login
    python -m antifraud_v3.eval.teleantifraud --prepare

用法：
    python -m antifraud_v3.eval.teleantifraud --limit 200
"""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .provenance import stamp

REPO_ID = "JimmyMa99/TeleAntiFraud"
CACHE_DIR = Path(__file__).parent / "teleantifraud_cache"
RESULT_PATH = Path(__file__).parent / "teleantifraud_result.json"


@dataclass
class Trial:
    index: int
    label: bool          # ground truth：是否為詐騙
    predicted: bool
    confidence: float
    fraud_type: str | None
    error: str | None = None


# ---------- 指標 ----------

def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval。n 小的時候必須顯示不確定性——PROTOCOL.md §4.2 要求
    每個數字都附信賴區間，而 Wilson 在極端比例（接近 0 或 1）時比常態近似可靠。"""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_metrics(trials: list[Trial]) -> dict:
    """主要指標是 HTER 與 FRR/FAR pair，不是單一準確率——理由見 PROTOCOL.md §4.1。

    此處的定義（詐騙偵測情境）：
      FAR (false acceptance) = 詐騙通話被判為正常 = 漏抓
      FRR (false rejection)  = 正常通話被判為詐騙 = 誤報
    這兩種錯誤的代價完全不同，所以分開報，讓讀者自行權衡。
    """
    ok = [t for t in trials if t.error is None]
    fraud = [t for t in ok if t.label]
    benign = [t for t in ok if not t.label]

    missed = sum(1 for t in fraud if not t.predicted)      # 漏抓
    false_alarm = sum(1 for t in benign if t.predicted)    # 誤報

    far = missed / len(fraud) if fraud else float("nan")
    frr = false_alarm / len(benign) if benign else float("nan")
    hter = (far + frr) / 2 if fraud and benign else float("nan")

    tp = sum(1 for t in fraud if t.predicted)
    fp = false_alarm
    fn = missed
    tn = len(benign) - false_alarm

    def f1(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    f1_fraud = f1(tp, fp, fn)
    f1_benign = f1(tn, fn, fp)
    support_f = len(fraud)
    support_b = len(benign)
    total = support_f + support_b

    return {
        "n_total": len(trials),
        "n_scored": len(ok),
        "n_errors": len(trials) - len(ok),
        "n_fraud": len(fraud),
        "n_benign": len(benign),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        # 主要指標
        "FAR_missed_fraud": round(far, 4),
        "FRR_false_alarm": round(frr, 4),
        "HTER": round(hter, 4),
        "FAR_ci95": [round(x, 4) for x in wilson_ci(missed, len(fraud))] if fraud else None,
        "FRR_ci95": [round(x, 4) for x in wilson_ci(false_alarm, len(benign))] if benign else None,
        # 三種 F1 一起報：論文未載明用哪種，所以不能只挑一個說「我們跟它比」
        "f1_fraud_class": round(f1_fraud, 4),
        "f1_macro": round((f1_fraud + f1_benign) / 2, 4),
        "f1_weighted": round((f1_fraud * support_f + f1_benign * support_b) / total, 4) if total else None,
        "comparability_note": (
            "論文未載明其 F1 為 macro/micro/weighted，故上述任一數字與論文的 "
            "58.51 / 84.78 皆為『非嚴格可比』。主要指標請看 HTER 與 FRR/FAR。"
        ),
    }


# ---------- 資料載入 ----------

def prepare() -> int:
    """下載官方 test split。刻意只抓 viewer/test.parquet 而非 12.7GB 的 audio.zip：
    Line 2 的判定路徑吃的是逐字稿文字，音訊只在 qwen2audio 後端才需要。先用文字路徑
    把 Q1 跑出來，音訊留到確定要測 qwen2audio 後端時再抓。"""
    from huggingface_hub import hf_hub_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ["dataset_manifest.json", "viewer/test.parquet"]:
        try:
            p = hf_hub_download(REPO_ID, fn, repo_type="dataset", local_dir=str(CACHE_DIR))
            print(f"  OK  {fn} -> {Path(p).stat().st_size:,} bytes")
        except Exception as e:
            name = type(e).__name__
            if "Gated" in name or "401" in str(e):
                print(f"\n存取被拒（{name}）——這個資料集是 gated，需要先取得存取權：")
                print("  1. 登入 huggingface.co")
                print(f"  2. 前往 huggingface.co/datasets/{REPO_ID} 並同意條款（auto-approval）")
                print("  3. 建立 READ token：huggingface.co/settings/tokens")
                print("  4. 執行：hf auth login")
                return 1
            print(f"  FAIL {fn}: {name}: {str(e)[:200]}")
            return 1
    return 0


def load_split(limit: int | None) -> list[dict]:
    """從 parquet 讀出 (逐字稿, 標籤)。欄位名稱在實際拿到檔案前無法確定，
    所以這裡對幾種常見命名做偵測，並在找不到時明確報錯而不是猜——猜錯會讓整份
    評測的 ground truth 是錯的，那比跑不動嚴重得多。"""
    import pandas as pd

    path = CACHE_DIR / "viewer" / "test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 --prepare")

    df = pd.read_parquet(path)
    print(f"  欄位：{list(df.columns)}")
    print(f"  列數：{len(df):,}")

    text_col = next((c for c in ["transcript", "text", "content", "conversation", "asr_text", "prompt"]
                     if c in df.columns), None)
    label_col = next((c for c in ["label", "is_fraud", "fraud", "category", "answer", "target"]
                      if c in df.columns), None)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"無法自動辨識逐字稿/標籤欄位。實際欄位：{list(df.columns)}\n"
            "請檢查後在 load_split() 明確指定——不猜，因為猜錯會讓 ground truth 整份是錯的。"
        )
    print(f"  逐字稿欄位={text_col}  標籤欄位={label_col}")
    print(f"  標籤分布：{Counter(df[label_col]).most_common(6)}")

    rows = []
    for i, r in df.iterrows():
        raw = r[label_col]
        if isinstance(raw, str):
            is_fraud = raw.strip().lower() in {"fraud", "1", "true", "yes", "詐騙"}
        else:
            is_fraud = bool(raw)
        rows.append({"index": int(i), "text": str(r[text_col]), "label": is_fraud})
        if limit and len(rows) >= limit:
            break
    return rows


# ---------- 執行 ----------

def run(rows: list[dict]) -> dict:
    from ..detectors.scam_semantic_llm import classify_call_via_llm
    from ..llm import get_llm_provider

    provider = get_llm_provider()
    trials: list[Trial] = []
    t0 = time.time()

    for n, row in enumerate(rows, 1):
        try:
            r = classify_call_via_llm(provider, row["text"])
            trials.append(Trial(row["index"], row["label"], r.is_fraud, r.confidence, r.fraud_type_raw))
        except Exception as e:
            trials.append(Trial(row["index"], row["label"], False, 0.0, None, error=type(e).__name__))
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(rows)}  ({el/n:.1f}s/筆，預估剩餘 {(len(rows)-n)*el/n/60:.0f} 分鐘)")

    return {
        "provenance": stamp(),
        "dataset": REPO_ID,
        "split": "official test",
        "backend": "line2_claude_text",
        "metrics": compute_metrics(trials),
        "trials": [asdict(t) for t in trials],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="下載官方 test split")
    ap.add_argument("--limit", type=int, default=None, help="限制筆數（每筆都是一次 LLM 呼叫）")
    args = ap.parse_args()

    if args.prepare:
        return prepare()

    rows = load_split(args.limit)
    print(f"\n評測 {len(rows)} 筆（詐騙 {sum(r['label'] for r in rows)} / "
          f"正常 {sum(not r['label'] for r in rows)}）\n")

    result = run(rows)
    m = result["metrics"]

    print(f"\n{'='*54}")
    print(f"已評分 {m['n_scored']}/{m['n_total']}（錯誤 {m['n_errors']}）")
    print(f"  漏抓率 FAR : {m['FAR_missed_fraud']:.1%}  CI95 {m['FAR_ci95']}")
    print(f"  誤報率 FRR : {m['FRR_false_alarm']:.1%}  CI95 {m['FRR_ci95']}")
    print(f"  HTER       : {m['HTER']:.1%}")
    print(f"  混淆矩陣   : {m['confusion']}")
    print(f"\n  F1（fraud class） : {m['f1_fraud_class']:.4f}")
    print(f"  F1（macro）       : {m['f1_macro']:.4f}")
    print(f"  F1（weighted）    : {m['f1_weighted']:.4f}")
    print(f"\n  {m['comparability_note']}")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
