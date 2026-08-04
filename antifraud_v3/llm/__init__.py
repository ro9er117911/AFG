from .base import LLMProvider, LLMProviderError

_provider_instance: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Factory reading llm_provider/llm_model/llm_effort from storage/settings_store.py (which
    itself seeds from LLM_PROVIDER/LLM_MODEL/LLM_EFFORT env vars on first run — see
    .env.example). Cached; call reset_provider() after a settings change so the next call
    rebuilds with the new values. Adding a provider means one new class + one branch here;
    reasoning/ never changes. See docs/DESIGN.md §3.4.

    "claude_code" (the default) shells out to the `claude` CLI, billed against a Claude
    Pro/Max subscription rather than a metered ANTHROPIC_API_KEY — see
    claude_code_provider.py's module docstring for why. "claude" is the original direct
    `anthropic` SDK path, for anyone who does have an API key and prefers it.
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    # Imported lazily to avoid storage <-> llm <-> reasoning import cycles at module load time.
    from ..storage.settings_store import load_settings

    settings = load_settings()
    provider_name = settings.get("llm_provider", "claude_code")

    if provider_name == "claude_code":
        from .claude_code_provider import ClaudeCodeProvider

        _provider_instance = ClaudeCodeProvider(model=settings["llm_model"], effort=settings["llm_effort"])
    else:
        from .claude_provider import ClaudeProvider

        _provider_instance = ClaudeProvider(model=settings["llm_model"], effort=settings["llm_effort"])
    return _provider_instance


def reset_provider() -> None:
    """Call after storage.settings_store.save_settings() changes llm_model/llm_effort so the
    next get_llm_provider() call picks up the new values instead of serving the cached one.
    """
    global _provider_instance
    _provider_instance = None


__all__ = ["LLMProvider", "LLMProviderError", "get_llm_provider", "reset_provider"]
