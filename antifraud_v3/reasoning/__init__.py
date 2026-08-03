from ..llm.base import LLMProvider
from .discriminate import discriminate
from .reflect import reflect
from .schemas import ChunkEvidence, DiscriminateResult, ReflectResult, SynthesizeResult
from .synthesize import synthesize


def run_reasoning_pipeline(provider: LLMProvider, evidence: ChunkEvidence) -> SynthesizeResult:
    """discriminate -> reflect -> synthesize, in order. See REWRITE_PLAN.md §4."""
    discriminated = discriminate(provider, evidence)
    reflected = reflect(provider, evidence, discriminated)
    return synthesize(provider, evidence, reflected)


__all__ = [
    "ChunkEvidence",
    "DiscriminateResult",
    "ReflectResult",
    "SynthesizeResult",
    "run_reasoning_pipeline",
]
