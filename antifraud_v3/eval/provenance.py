"""記錄每次評測執行的環境指紋。

`PROTOCOL.md` §10 要求「模型版本、隨機種子全部納入版本控制」，
但獨立審查指出目前未做到——`ClaudeCodeProvider` 無 temperature/seed 參數，
模型別名 `claude-opus-5` 會隨時間漂移。

**這個模組不能讓結果變成可重現**（那需要改動 provider 或改用可 pin 版本的 API），
但它能讓結果**可歸因**：至少讓讀者知道某個數字是在什麼環境下產生的，
而不是一個來歷不明的數字。這是誠實揭露，不是修好問題。

用法：
    from .provenance import stamp
    result["provenance"] = stamp()
"""

import platform
import subprocess
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None
    except Exception:
        return None


def stamp() -> dict:
    """回傳可歸因所需的環境指紋。刻意不拋例外——記錄失敗不該讓評測整個掛掉。"""
    from ..storage.settings_store import load_settings

    settings = load_settings()
    git_sha = _run(["git", "rev-parse", "HEAD"])
    git_dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])

    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        # 工作區若不乾淨，這個數字就無法對應到任何一個 commit——必須標出來
        "git_worktree_clean": git_dirty == "" or git_dirty is None,
        "llm_provider": settings.get("llm_provider"),
        "llm_model_alias": settings.get("llm_model"),
        "llm_effort": settings.get("llm_effort"),
        "line2_backend": settings.get("line2_backend"),
        "asr_backend": settings.get("asr_backend"),
        "emotion_backend": settings.get("emotion_backend"),
        "claude_cli_version": _run(["claude", "--version"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reproducibility_caveat": (
            "模型別名（如 claude-opus-5）會隨時間指向不同的實際模型版本，"
            "且 ClaudeCodeProvider 未提供 temperature/seed 控制。"
            "因此本結果『可歸因』但**不可精確重現**——第三方重跑會得到不同數字。"
            "見 PROTOCOL.md §10 與 REVIEW_LOG.md 第 9 項。"
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(stamp(), ensure_ascii=False, indent=2))
