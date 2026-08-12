"""在 TeleAntiFraud-28k 官方 test split 上評測 Line 2 與融合層。

這是評測協定的 Q1：在**有統計意義的樣本數**上量測鑑別力。現有自建語料只有 23 筆
（PROTOCOL.md §4.3 明訂 n<100 不得以準確率語氣陳述結論），這個資料集的
binary_classification test split 有 **400 筆**（fraud 200 / normal 200，完全平衡），
是目前唯一能讓 Q1 得到可信答案的來源。

（7,021 是先前的誤記——那是 sft split 的規模，實測為 6,807，且非詐騙偵測任務。）

**受測對象是「ASR + Line 2」整條鏈路，不是 Line 2。** 該資料集三個 split 都
沒有逐字稿（欄位僅 id/task/audio_path/instruction/label），文字必須由我方 ASR 產生。

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
# audio.zip 解壓後的根目錄；parquet 的 audio_path 是相對於它的路徑
AUDIO_ROOT = CACHE_DIR / "audio_extracted"


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
    """下載官方 test split 的 metadata。

    先前這裡的註解說「刻意不抓 audio.zip，因為 Line 2 吃的是逐字稿文字」——
    **那個前提是錯的**：實測三個 split 都沒有逐字稿欄位（2026-08-12）。
    音訊是必要的，12.7GB 的 audio.zip 需另外下載（見 --audio-root）。"""
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


# 實測 schema（2026-08-12）：['id', 'task', 'audio_path', 'instruction', 'label']
# **沒有逐字稿欄位**，文字必須由我方 ASR 從音訊產生。
#
# 這裡刻意寫死欄位名而非啟發式偵測。原本的偵測清單含 "prompt"，而
# binary_classification 的 JSON 確實有 prompt 欄——但它是 nunique=1 的固定模板。
# 若被選中，400 筆會拿到同一段模板文字，跑完得到一份看起來正常但完全無意義的結果。
# 寫死是為了讓這種失敗不可能發生。
LABEL_COL = "label"
AUDIO_COL = "audio_path"
FRAUD_LABELS = {"fraud"}
BENIGN_LABELS = {"normal"}


def load_split(limit: int | None, seed: int = 0) -> list[dict]:
    """從 parquet 讀出 (音訊路徑, 標籤)。逐字稿由 ASR 在 run() 內產生。"""
    import pandas as pd

    path = CACHE_DIR / "viewer" / "test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 --prepare")

    df = pd.read_parquet(path)
    print(f"  欄位：{list(df.columns)}")
    print(f"  列數：{len(df):,}")

    missing = {LABEL_COL, AUDIO_COL} - set(df.columns)
    if missing:
        raise SystemExit(
            f"預期欄位不存在：{missing}。實際欄位：{list(df.columns)}\n"
            "資料集 schema 已變動，請重新檢查後修正——不猜，因為猜錯會讓 ground truth 整份是錯的。"
        )

    labels = set(df[LABEL_COL].unique())
    unknown = labels - FRAUD_LABELS - BENIGN_LABELS
    if unknown:
        raise SystemExit(f"出現未知標籤值 {unknown}，無法安全映射為二元 ground truth。")
    print(f"  標籤分布：{Counter(df[LABEL_COL]).most_common()}")

    # --limit 必須先打亂。parquet 的列序不保證與標籤獨立，依序取前 N 筆可能
    # 抽到高度偏斜的子集，得到一個「準確率很高」的假象而白跑一整晚。
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=seed).sort_index()
        print(f"  --limit {limit}：已用 seed={seed} 隨機取樣，"
              f"取樣後分布 {Counter(df[LABEL_COL]).most_common()}")

    return [
        {"index": int(i), "audio": str(r[AUDIO_COL]), "label": r[LABEL_COL] in FRAUD_LABELS}
        for i, r in df.iterrows()
    ]


# ---------- 執行 ----------

_asr_cache: dict[str, object] = {}


def _transcribe(path: Path, model_size: str) -> str:
    """評測專用轉寫器，不走 asr/transcribe.py。

    生產路徑 `load_whisper_model()` 把模型固定在單一全域 singleton（寫死 "base"），
    無法在同一個 process 內切換模型——而雙臂設計的**整個重點**就是切換模型。
    這裡自行持有 cache；生產程式碼不因評測需求而改動。
    """
    import ctranslate2
    from faster_whisper import WhisperModel

    if model_size not in _asr_cache:
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        device, ct = ("cuda", "float16") if has_cuda else ("cpu", "int8")
        print(f"  載入 Whisper {model_size}（{device}/{ct}）…", flush=True)
        _asr_cache[model_size] = WhisperModel(model_size, device=device, compute_type=ct)

    segments, _ = _asr_cache[model_size].transcribe(str(path), language="zh")
    return "".join(s.text for s in segments).strip()


def run(rows: list[dict], asr_model: str, resume_path: Path) -> dict:
    """單一臂：用指定的 ASR 模型轉寫，再送 Line 2 判定。

    逐筆 append 到 JSONL。400 筆 x 2 臂是數百次 LLM 呼叫加上 CPU 上的 ASR，
    中途掛掉全部重跑的代價太高——續跑不是優化，是這個規模下的必要條件。
    """
    from ..detectors.scam_semantic_llm import classify_call_via_llm
    from ..llm import get_llm_provider

    done: dict[int, dict] = {}
    if resume_path.exists():
        for line in resume_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["index"]] = d
        print(f"  續跑：已有 {len(done)} 筆，跳過")

    provider = get_llm_provider()
    trials: list[Trial] = []
    t0 = time.time()
    n_new = 0

    with resume_path.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(rows, 1):
            if row["index"] in done:
                d = done[row["index"]]
                trials.append(Trial(d["index"], d["label"], d["predicted"], d["confidence"],
                                    d.get("fraud_type"), error=d.get("error")))
                continue

            audio = AUDIO_ROOT / row["audio"]
            try:
                if not audio.exists():
                    raise FileNotFoundError(str(audio))
                text = _transcribe(audio, asr_model)
                r = classify_call_via_llm(provider, text)
                t = Trial(row["index"], row["label"], r.is_fraud, r.confidence, r.fraud_type_raw)
                rec = {"index": t.index, "label": t.label, "predicted": t.predicted,
                       "confidence": t.confidence, "fraud_type": t.fraud_type,
                       "asr_chars": len(text), "error": None}
            except Exception as e:
                t = Trial(row["index"], row["label"], False, 0.0, None, error=type(e).__name__)
                rec = {"index": t.index, "label": t.label, "predicted": False, "confidence": 0.0,
                       "fraud_type": None, "asr_chars": 0,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"}

            trials.append(t)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1

            if n_new % 10 == 0:
                el = time.time() - t0
                rate = el / n_new
                print(f"  [{asr_model}] {n}/{len(rows)}  ({rate:.1f}s/筆，"
                      f"預估剩餘 {(len(rows)-n)*rate/60:.0f} 分鐘)", flush=True)

    return {
        "dataset": REPO_ID,
        "split": "binary_classification test (n=400, fraud 200 / normal 200)",
        "system_under_test": f"Whisper {asr_model} (ASR) + Line 2 (zero-shot)",
        "asr_model": asr_model,
        "metrics": compute_metrics(trials),
        "trials": [asdict(t) for t in trials],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="下載官方 test split metadata")
    ap.add_argument("--limit", type=int, default=None, help="限制筆數（隨機取樣，見 --seed）")
    ap.add_argument("--seed", type=int, default=0, help="--limit 取樣用的亂數種子")
    ap.add_argument("--asr", nargs="+", default=["base", "large-v3"],
                    help="要跑的 ASR 臂。預設雙臂：base 與 large-v3，兩者之差即 ASR 造成的衰減")
    args = ap.parse_args()

    if args.prepare:
        return prepare()

    if not AUDIO_ROOT.exists():
        print(f"找不到音訊目錄：{AUDIO_ROOT}")
        print("該資料集沒有逐字稿，必須下載並解壓 audio.zip（12.7GB）後才能評測。")
        return 1

    rows = load_split(args.limit, args.seed)
    print(f"\n評測 {len(rows)} 筆（詐騙 {sum(r['label'] for r in rows)} / "
          f"正常 {sum(not r['label'] for r in rows)}），ASR 臂：{args.asr}\n")

    arms = {}
    for asr_model in args.asr:
        print(f"\n--- 臂：Whisper {asr_model} ---")
        arms[asr_model] = run(rows, asr_model, RESULT_PATH.with_suffix(f".{asr_model}.jsonl"))

    print(f"\n{'='*60}")
    for asr_model, result in arms.items():
        m = result["metrics"]
        print(f"\n[Whisper {asr_model}]  已評分 {m['n_scored']}/{m['n_total']}"
              f"（錯誤 {m['n_errors']}）")
        print(f"  漏抓率 FAR : {m['FAR_missed_fraud']:.1%}  CI95 {m['FAR_ci95']}")
        print(f"  誤報率 FRR : {m['FRR_false_alarm']:.1%}  CI95 {m['FRR_ci95']}")
        print(f"  HTER       : {m['HTER']:.1%}")
        print(f"  混淆矩陣   : {m['confusion']}")

    # 雙臂之差就是 ASR 造成的衰減——這是本測試最有價值的產出，
    # 因為它把「語意層失敗」與「ASR 失敗」分開了。
    asr_delta = None
    if len(arms) == 2:
        (a, ra), (b, rb) = arms.items()
        asr_delta = {
            "arms": [a, b],
            "HTER_delta": round(ra["metrics"]["HTER"] - rb["metrics"]["HTER"], 4),
            "interpretation": (
                f"HTER({a}) - HTER({b})。為正代表較弱的 ASR 拖累了整體表現，"
                f"其量級即為 ASR 品質對本系統的敏感度。"
            ),
        }
        print(f"\nASR 衰減：HTER({a}) - HTER({b}) = {asr_delta['HTER_delta']:+.1%}")

    out = {
        "provenance": stamp(),
        "no_baseline_comparison": (
            "本結果**不與該資料集已發表的 baseline（58.51 / 84.78）比較**。"
            "其 baseline 為在該資料集上微調過的端到端音訊模型，我方為 zero-shot "
            "且需自行 ASR 產生逐字稿——受測系統形態不同，並排數字會誤導。"
        ),
        "external_validity_limits": (
            "全部為 ChatTTS 合成寬頻音訊、中國大陸詐騙情境。"
            "本結果**不預測** 8kHz 電話頻寬表現，也不預測台灣場景表現。"
        ),
        "arms": arms,
        "asr_degradation": asr_delta,
    }
    RESULT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果：{RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
