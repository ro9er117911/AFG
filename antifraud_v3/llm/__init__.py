import os

from .base import LLMProvider, LLMProviderError

_provider_instance: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Factory reading LLM_PROVIDER/LLM_MODEL/LLM_EFFORT from env. Cached — one provider
    instance per process. Adding a second provider means one new class + one branch here;
    reasoning/ never changes. See REWRITE_PLAN.md §3.4.
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider = os.getenv("LLM_PROVIDER", "claude")
    if provider == "claude":
        from .claude_provider import ClaudeProvider

        _provider_instance = ClaudeProvider(
            model=os.getenv("LLM_MODEL", "claude-opus-5"),
            effort=os.getenv("LLM_EFFORT", "medium"),
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (only 'claude' is implemented)")

    return _provider_instance


__all__ = ["LLMProvider", "LLMProviderError", "get_llm_provider"]
