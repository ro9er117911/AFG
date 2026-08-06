"""Text-only alternative to Line 2 (detectors/scam_semantic.py's local Qwen2Audio classifier),
selectable via storage.settings_store's "line2_backend" setting ("claude" vs "qwen2audio").
Routes through the existing LLMProvider abstraction (llm/base.py) instead of loading a local
~6.6GB 4-bit model — the point is speed: no GPU model load/unload, one structured-output call
instead of a 2-3 round audio generate() loop.

Returns the same AntiFraudQwenResult shape scam_semantic.py's classify_call() does, so
reasoning/fusion.py's fuse() needs no changes at all — it only ever reads is_fraud, confidence,
and fraud_type_raw off that dataclass, regardless of which backend produced it.
"""

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, LLMProviderError
from ..reasoning.rubric import LINE2_LLM_SYSTEM_PROMPT
from ..reasoning.schemas import FraudType
from .scam_semantic import AntiFraudQwenResult, ScamSemanticError


class Line2LLMResult(BaseModel):
    scenario: str
    is_fraud: bool
    confidence: float = Field(ge=0, le=1)
    fraud_type: FraudType


def classify_call_via_llm(provider: LLMProvider, transcript: str) -> AntiFraudQwenResult:
    """Text-only counterpart to scam_semantic.classify_call() — transcript is the only input
    (no audio), since the LLM providers here don't do native audio classification. fraud_type
    comes straight from the model's own FraudType pick, not map_fraud_type()'s keyword
    matching (that's only needed for Qwen2Audio's free-text Chinese label output).
    """
    try:
        result = provider.structured_complete(
            system=LINE2_LLM_SYSTEM_PROMPT,
            user_content=f"逐字稿：\n{transcript}",
            schema=Line2LLMResult,
        )
    except LLMProviderError as e:
        raise ScamSemanticError(f"Line 2 (claude backend) failed: {e}") from e

    return AntiFraudQwenResult(
        scenario=result.scenario,
        is_fraud=result.is_fraud,
        confidence=result.confidence,
        fraud_type_raw=result.fraud_type if result.is_fraud else None,
        reasoning_trace=[],
    )
