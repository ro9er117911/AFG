"""清理 `script_features_result.jsonl` 的重複列，並順便量出 LLM 的自我一致性。

**為什麼會有重複**：2026-08-13 的執行中，一個舊 process 未完全結束時又啟動了新的，
兩者同時 append 到同一個 JSONL。這是操作失誤，不是程式邏輯錯誤——但它意外產生了
一批「同一段逐字稿被獨立判斷兩次」的資料。

**為什麼不直接刪掉重複列**：重複列的內容**不完全一致**。同一段文字、同一個模型、
同一個 prompt，兩次呼叫給出不同答案——這正是 `PROTOCOL_v2.md` §2.2 記載的
「provider 無 temperature/seed 控制，結果可歸因但不可精確重現」的直接證據。
把它當垃圾丟掉會浪費一個免費取得的測量值，所以先量再清。

**去重規則**：保留每個 clip 的**第一次**判斷（時序上先發生的），並記錄不一致率。
選第一次而非多數決，是因為多數決需要奇數次，而這裡只有兩次；
選「先發生」則是一個與結果無關的規則，不會偏向任何一邊。

錯誤列（LLM 逾時等）不納入統計，也不保留——重跑時 `script_features.py` 的續跑
邏輯會重新處理這些 clip。

用法：
    python -m antifraud_v3.eval.script_features_dedup            # report only
    python -m antifraud_v3.eval.script_features_dedup --write    # 實際覆寫
"""

import argparse
import json
from pathlib import Path

from .provenance import stamp
from .script_features import RESULT_PATH

JSONL_PATH = RESULT_PATH.with_suffix(".jsonl")
CONSISTENCY_PATH = RESULT_PATH.parent / "script_features_consistency.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="實際覆寫 JSONL；預設只報告")
    args = ap.parse_args()

    if not JSONL_PATH.exists():
        raise SystemExit(f"找不到 {JSONL_PATH}")

    rows = [json.loads(line) for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ok = [r for r in rows if "present" in r]
    errors = [r for r in rows if "present" not in r]

    by_clip: dict[str, list[dict]] = {}
    for r in ok:
        by_clip.setdefault(r["clip"], []).append(r)
    dup = {k: v for k, v in by_clip.items() if len(v) > 1}

    keys = list(ok[0]["present"]) if ok else []
    agree = disagree = 0
    flips = {k: 0 for k in keys}
    disagreeing_clips = []
    for clip, v in dup.items():
        a, b = v[0]["present"], v[1]["present"]
        clip_diff = [f for f in keys if a[f] != b[f]]
        for f in keys:
            if a[f] == b[f]:
                agree += 1
            else:
                disagree += 1
                flips[f] += 1
        if clip_diff:
            disagreeing_clips.append({"clip": clip, "features": clip_diff})

    total = agree + disagree
    consistency = {
        "provenance": stamp(),
        "how_this_arose": (
            "非刻意設計的重測——兩個 process 同時 append 造成同一 clip 被獨立判斷兩次。"
            "資料既然產生了就據實記錄，但**不得**當作預先登錄的信度實驗："
            "重複的 clip 不是隨機抽樣，而是兩個 process 執行區間恰好重疊的那一段。"
        ),
        "n_clips_judged_twice": len(dup),
        "n_feature_judgements_compared": total,
        "n_agree": agree,
        "n_disagree": disagree,
        "agreement_rate": round(agree / total, 4) if total else None,
        "flips_by_feature": {k: v for k, v in sorted(flips.items(), key=lambda x: -x[1]) if v},
        "disagreeing_clips": disagreeing_clips,
        "interpretation": (
            "同一段逐字稿、同一模型、同一 prompt，兩次呼叫的特徵判斷一致率。"
            "不一致的部分即 provider 無 temperature/seed 控制所致的抖動下限"
            "（PROTOCOL_v2.md §2.2）。這是**下限**不是完整估計：只測了兩次，"
            "且僅涵蓋重疊區間的 clip。"
        ),
    }
    CONSISTENCY_PATH.write_text(json.dumps(consistency, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    print(f"總列數 {len(rows)}（有效 {len(ok)}、錯誤 {len(errors)}）")
    print(f"unique clip {len(by_clip)}，其中被判斷兩次的 {len(dup)}")
    if total:
        print(f"自我一致性：{agree}/{total} = {agree/total:.1%}"
              f"（不一致 {disagree}）")
        print(f"翻轉分布：{consistency['flips_by_feature']}")
    print(f"一致性報表：{CONSISTENCY_PATH}")

    if args.write:
        # 保留每個 clip 的第一次判斷，維持原始出現順序
        kept, seen = [], set()
        for r in ok:
            if r["clip"] in seen:
                continue
            seen.add(r["clip"])
            kept.append(r)
        JSONL_PATH.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            encoding="utf-8")
        print(f"\n已覆寫：{len(rows)} 列 → {len(kept)} 列"
              f"（去重 {len(ok) - len(kept)}、丟棄錯誤列 {len(errors)}，"
              f"錯誤列的 clip 會由續跑重新處理）")
    else:
        print("\n（--write 未指定，未修改檔案）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
