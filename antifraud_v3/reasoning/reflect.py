from ..llm.base import LLMProvider
from .rubric import REFLECT_SYSTEM_PROMPT
from .schemas import ChunkEvidence, DiscriminateResult, ReflectResult


def reflect(
    provider: LLMProvider, evidence: ChunkEvidence, discriminated: DiscriminateResult
) -> ReflectResult:
    flagged = [s for s in discriminated.stage_evidence if s.strength != "none"]
    if not flagged:
        # Nothing to reflect on — short-circuit rather than spending an LLM call arguing
        # against evidence that was already "none". Mirrors the flagged stages 1:1 so
        # synthesize() always has a ReflectResult to read regardless of this branch.
        return ReflectResult(
            reflected_stages=[
                {
                    "stage": s.stage,
                    "original_strength": s.strength,
                    "revised_strength": s.strength,
                    "innocent_explanation": None,
                    "reasoning": "無標記證據，跳過反思。",
                }
                for s in discriminated.stage_evidence
            ]
        )

    stage_lines = "\n".join(
        f"- {s.stage}: {s.strength} — {s.justification}" for s in flagged
    )
    user_content = (
        f"逐字稿片段：「{evidence.transcript_segment}」\n"
        f"聲學特徵：{evidence.acoustic_summary}\n"
        f"情緒機率：{evidence.emotion_summary}\n\n"
        f"判別步驟標記的證據：\n{stage_lines}\n\n"
        "針對以上每一項，判斷有沒有合理的無辜解釋，並給出修正後的證據強度。"
    )
    return provider.structured_complete(
        system=REFLECT_SYSTEM_PROMPT,
        user_content=user_content,
        schema=ReflectResult,
    )
