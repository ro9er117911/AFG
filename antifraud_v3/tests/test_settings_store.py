"""storage/settings_store.py — always run against temp_settings (tests/conftest.py), never
the real antifraud_v3/data/settings.json.
"""

from antifraud_v3.reasoning.rubric import DEFAULT_HARD_TRIGGERS
from antifraud_v3.storage import settings_store


def test_defaults_returned_when_no_settings_file_exists(temp_settings):
    assert not temp_settings.exists()
    settings = settings_store.load_settings()
    assert settings["llm_provider"]
    assert settings["llm_model"]
    assert settings["llm_effort"]
    assert settings["hard_triggers"] == DEFAULT_HARD_TRIGGERS


def test_save_settings_persists_to_disk(temp_settings):
    settings_store.save_settings({"llm_effort": "high"})
    assert temp_settings.exists()

    settings_store._cache = None  # force a real reload from disk, not the in-memory cache
    reloaded = settings_store.load_settings()
    assert reloaded["llm_effort"] == "high"


def test_partial_update_preserves_other_keys(temp_settings):
    settings_store.save_settings({"llm_model": "claude-sonnet-5"})
    result = settings_store.save_settings({"llm_effort": "high"})
    assert result["llm_model"] == "claude-sonnet-5"  # not clobbered by the second partial update
    assert result["llm_effort"] == "high"


def test_hard_triggers_can_be_fully_replaced(temp_settings):
    custom = ["自訂關鍵字一", "自訂關鍵字二"]
    result = settings_store.save_settings({"hard_triggers": custom})
    assert result["hard_triggers"] == custom


def test_load_settings_uses_cache_across_calls(temp_settings):
    first = settings_store.load_settings()
    second = settings_store.load_settings()
    assert first is second  # same object — no redundant disk read


def test_save_settings_updates_the_cache_immediately(temp_settings):
    settings_store.load_settings()
    settings_store.save_settings({"llm_effort": "xhigh"})
    # No manual cache reset here — save_settings should update _cache itself so the very
    # next load_settings() call (e.g. from the next chunk in a live call) sees the change
    # without needing a server restart.
    assert settings_store.load_settings()["llm_effort"] == "xhigh"
