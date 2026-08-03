"""Runtime settings, persisted as a small JSON file rather than a DB table — it's a handful
of scalar/list values, not relational data. Backs the Settings screen (REWRITE_PLAN.md §9).
"""

import json
import os
from pathlib import Path

from ..reasoning.rubric import DEFAULT_HARD_TRIGGERS

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

# .env values seed these defaults; once settings.json exists (i.e. the Settings screen has
# saved at least once), it takes precedence — settings_store is the single source of truth
# that everything else (llm/__init__.py, server/ws.py) reads from, not raw os.getenv() calls
# scattered around.
DEFAULTS = {
    "llm_model": os.getenv("LLM_MODEL", "claude-opus-5"),
    "llm_effort": os.getenv("LLM_EFFORT", "medium"),
    "debounce_chunks": int(os.getenv("DEBOUNCE_CHUNKS", "2")),
    "hard_triggers": list(DEFAULT_HARD_TRIGGERS),
}

_cache: dict | None = None


def load_settings() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            stored = json.load(f)
        _cache = {**DEFAULTS, **stored}
    else:
        _cache = dict(DEFAULTS)
    return _cache


def save_settings(updates: dict) -> dict:
    global _cache
    current = {**load_settings(), **updates}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    _cache = current
    return current
