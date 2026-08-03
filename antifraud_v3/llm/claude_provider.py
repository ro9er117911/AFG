import anthropic
from pydantic import BaseModel

from .base import LLMProvider, LLMProviderError, SchemaT

_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class ClaudeProvider(LLMProvider):
    """Current default LLMProvider implementation. See docs/DESIGN.md §3.3 for the reasoning
    behind each parameter choice (model default, adaptive thinking, structured outputs).

    Auth resolves automatically via the anthropic SDK: ANTHROPIC_API_KEY env var, or an
    `ant auth login` profile if no key is set. Neither is configured in this dev environment
    yet — see .env.example.
    """

    def __init__(self, model: str = "claude-opus-5", effort: str = "medium"):
        if effort not in _VALID_EFFORTS:
            raise ValueError(f"effort must be one of {_VALID_EFFORTS}, got {effort!r}")
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def structured_complete(
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_format=schema,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
            )
        except anthropic.RateLimitError as e:
            raise LLMProviderError(f"Claude rate limited: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMProviderError(f"Claude connection error: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMProviderError(f"Claude API error ({e.status_code}): {e.message}") from e
        except Exception as e:
            # Confirmed by real testing (no ANTHROPIC_API_KEY / `ant auth login` profile
            # configured): the SDK raises a plain TypeError from _build_headers() when no
            # credentials resolve at all — before any HTTP request is even attempted, so it
            # isn't one of the anthropic.* response-error classes above. Catch broadly here
            # so a missing-credentials dev environment degrades to a clean, catchable
            # LLMProviderError instead of an unhandled exception killing the caller
            # (server/ws.py's WebSocket loop, in particular).
            raise LLMProviderError(f"Claude request failed: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMProviderError(
                "Claude declined this request (stop_reason=refusal) — "
                "see stop_details for the category."
            )

        parsed = response.parsed_output
        if not isinstance(parsed, schema):
            raise LLMProviderError(
                f"Expected {schema.__name__}, got {type(parsed).__name__} — "
                "structured output validation failed unexpectedly."
            )
        return parsed
