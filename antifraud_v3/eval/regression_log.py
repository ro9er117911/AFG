"""Regression logging per REWRITE_PLAN.md §6 — one JSONL line per test-clip run, so a
prompt/rubric change can be checked for false-positive/negative regressions without needing
a large labeled dataset (there isn't one).
"""

import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).parent / "regression_log.jsonl"


def log_result(
    clip_name: str,
    expected_category: str,  # "scam" | "benign"
    alert_fired: bool,
    alert_timestamp: float | None,
    cited_pattern: str | None,
    human_judgment: str | None = None,
) -> None:
    entry = {
        "timestamp": time.time(),
        "clip_name": clip_name,
        "expected_category": expected_category,
        "alert_fired": alert_fired,
        "alert_timestamp": alert_timestamp,
        "cited_pattern": cited_pattern,
        "human_judgment": human_judgment,  # fill in by hand after reviewing the run
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
