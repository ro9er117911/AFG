"""Shared fixtures for the antifraud_v3 test suite.

Design note: nearly every fixture here exists to keep tests from touching real state — the
production SQLite DB (storage/history.py), the production settings.json
(storage/settings_store.py), or a real LLM API. See docs/DESIGN.md §3.1 for why LLMProvider
is an interface in the first place: FakeLLMProvider below is exactly the kind of thing that
interface was built to make possible.
"""

import sys
from pathlib import Path

import pytest

# Make sure "antifraud_v3.x" imports resolve regardless of where pytest is invoked from —
# normally unnecessary (pytest's rootdir insertion handles it when run from the repo root,
# see pyproject.toml), but cheap insurance against a different invocation directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from antifraud_v3.llm.base import LLMProvider  # noqa: E402
from antifraud_v3.reasoning.schemas import (  # noqa: E402
    DiscriminateResult,
    FraudTypeClassification,
    ReflectResult,
    SynthesizeResult,
)

DEFAULT_FRAUD_TYPE = FraudTypeClassification(fraud_type="unclassified", confidence="none", justification="無明顯詐騙類型敘事。")


class FakeLLMProvider(LLMProvider):
    """Deterministic stand-in for a real LLM, matching llm/base.py's one-method interface.

    Returns a canned response per schema type by default (all "nothing interesting
    happened") so any code built on top of the reasoning engine can be tested without
    live API access; construct with `synthesize_result=` to control the risk verdict a
    test cares about. `self.calls` records every request for assertions about how many
    times / in what order the provider was actually invoked (e.g. reflect.py's
    short-circuit-when-nothing-flagged path should make zero calls).
    """

    def __init__(self, synthesize_result: SynthesizeResult | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self._synthesize_result = synthesize_result or SynthesizeResult(
            risk_level="low",
            chunk_risk_score=0,
            hard_triggers=[],
            justification="正常對話，沒有偵測到詐騙跡象。",
            case_memory_update="",
            fraud_type=DEFAULT_FRAUD_TYPE,
        )

    def structured_complete(self, system, user_content, schema):
        self.calls.append((schema.__name__, system, user_content))
        if schema is DiscriminateResult:
            return DiscriminateResult(stage_evidence=[], overall_note="無明顯證據。")
        if schema is ReflectResult:
            return ReflectResult(reflected_stages=[])
        if schema is SynthesizeResult:
            return self._synthesize_result
        raise AssertionError(f"FakeLLMProvider got an unexpected schema: {schema}")


@pytest.fixture
def fake_llm():
    return FakeLLMProvider()


@pytest.fixture
def temp_history_db(tmp_path, monkeypatch):
    """Point storage/history.py at a throwaway SQLite file instead of
    antifraud_v3/data/history.sqlite3, so tests never read or write real call history."""
    from antifraud_v3.storage import history

    db_path = tmp_path / "history_test.sqlite3"
    monkeypatch.setattr(history, "DB_PATH", db_path)
    history.init_db()
    return db_path


@pytest.fixture
def temp_settings(tmp_path, monkeypatch):
    """Point storage/settings_store.py at a throwaway JSON file instead of
    antifraud_v3/data/settings.json, and reset its module-level cache before and after so
    tests don't leak settings state into each other or into a real run."""
    from antifraud_v3.storage import settings_store

    settings_path = tmp_path / "settings_test.json"
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)
    settings_store._cache = None
    yield settings_path
    settings_store._cache = None
