from ..llm.base import LLMProvider
from .discriminate import discriminate
from .reflect import reflect
from .schemas import ChunkEvidence, DiscriminateResult, ReflectResult, SynthesizeResult
from .synthesize import synthesize


def run_reasoning_pipeline(
    provider: LLMProvider, evidence: ChunkEvidence, hard_triggers: list[str] | None = None
) -> SynthesizeResult:
    """discriminate -> reflect -> synthesize, in order. See docs/DESIGN.md §4.

    hard_triggers defaults to rubric.DEFAULT_HARD_TRIGGERS when omitted; pass the live
    settings-store value (storage/settings_store.py) to respect user edits from the Settings
    screen.
    """
    discriminated = discriminate(provider, evidence)
    reflected = reflect(provider, evidence, discriminated)
    return synthesize(provider, evidence, reflected, hard_triggers=hard_triggers)


__all__ = [
    "ChunkEvidence",
    "DiscriminateResult",
    "ReflectResult",
    "SynthesizeResult",
    "run_reasoning_pipeline",
]
