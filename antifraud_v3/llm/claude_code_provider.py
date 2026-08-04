"""LLMProvider backed by the `claude` CLI (Claude Code) instead of the `anthropic` API SDK.

Why this exists: the reasoning engine used to run 2-3 LLM calls per VAD chunk
(discriminate/reflect/synthesize), which is a real per-token API cost multiplier
(docs/DESIGN.md §8) — untenable without a paid ANTHROPIC_API_KEY. The architecture now runs
the reasoning pipeline once per call instead of once per chunk (see pipeline/chunk_worker.py's
run_final_analysis), which makes a slower-per-call but subscription-billed path viable: the
`claude` CLI in non-interactive print mode (`-p`) draws from a Claude Pro/Max subscription's
usage allowance rather than metered pay-per-token API billing, confirmed via `claude auth
status` (authMethod: "claude.ai", subscriptionType: "pro") in this environment.

`--json-schema` + `--output-format json` gives back a `structured_output` field already
parsed against the schema — no hand-rolled JSON extraction needed. `--tools ""` and
`--strict-mcp-config` (no --mcp-config given) keep this a single-turn completion with no tool
use, and `cwd` is a throwaway directory (not this repo) plus `--setting-sources ""` so this
doesn't pick up this project's or the user's personal CLAUDE.md/hooks/memory as unwanted
context — the system/user content passed in here is meant to be the complete input.
"""

import json
import logging
import os
import subprocess
import tempfile

from pydantic import BaseModel

from .base import LLMProvider, LLMProviderError, SchemaT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 180


class ClaudeCodeProvider(LLMProvider):
    """See module docstring. `model`/`effort` map directly onto the CLI's own `--model`/
    `--effort` flags — same config surface as ClaudeProvider, just a different transport.
    """

    def __init__(self, model: str = "claude-opus-5", effort: str = "medium", timeout_s: int = DEFAULT_TIMEOUT_S):
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        # A dedicated empty scratch dir, not this repo and not the user's home directory —
        # `claude` auto-discovers CLAUDE.md/project settings from cwd, and this call is meant
        # to be a self-contained structured-completion request, not a coding-agent session.
        self._cwd = tempfile.mkdtemp(prefix="antifraud_v3_claude_code_")

    def structured_complete(
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        cmd = [
            "claude",
            "-p",
            user_content,
            "--output-format",
            "json",
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(schema.model_json_schema()),
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
        ]
        # Blank ANTHROPIC_API_KEY (if set, e.g. inherited from a .env meant for
        # ClaudeProvider) would make the CLI try metered API billing instead of the
        # subscription/OAuth session this provider exists to use — same fix as
        # claude_provider.py's blank-key handling, applied here for the opposite reason.
        env = dict(os.environ)
        if not env.get("ANTHROPIC_API_KEY"):
            env.pop("ANTHROPIC_API_KEY", None)

        try:
            proc = subprocess.run(
                cmd,
                cwd=self._cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise LLMProviderError(f"claude CLI timed out after {self.timeout_s}s") from e
        except FileNotFoundError as e:
            raise LLMProviderError(
                "claude CLI not found on PATH — install Claude Code or switch LLM_PROVIDER "
                "back to 'claude' in settings"
            ) from e

        if proc.returncode != 0:
            raise LLMProviderError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"claude CLI returned non-JSON output: {proc.stdout[:500]!r}") from e

        if payload.get("is_error"):
            raise LLMProviderError(f"claude CLI reported an error: {payload.get('result')}")

        structured = payload.get("structured_output")
        if structured is None:
            raise LLMProviderError(
                f"claude CLI response had no structured_output (result={payload.get('result')!r})"
            )

        try:
            return schema.model_validate(structured)
        except Exception as e:
            raise LLMProviderError(f"structured_output failed schema validation: {e}") from e
