from ..llm.base import LLMProvider
from .rubric import build_synthesize_system_prompt
from .schemas import ChunkEvidence, ReflectResult, SynthesizeResult


def synthesize(
    provider: LLMProvider,
    evidence: ChunkEvidence,
    reflected: ReflectResult,
    hard_triggers: list[str] | None = None,
) -> SynthesizeResult:
    stage_lines = "\n".join(
        f"- {s.stage}: {s.revised_strength}"
        + (f"（原始 {s.original_strength}，理由：{s.reasoning}）" if s.revised_strength != s.original_strength else "")
        for s in reflected.reflected_stages
    )
    user_content = (
        f"逐字稿片段：「{evidence.transcript_segment}」\n\n"
        f"反思後的階段證據：\n{stage_lines}\n\n"
        "根據以上證據與立即示警關鍵字清單，產出這個 chunk 的風險評估。"
    )
    return provider.structured_complete(
        system=build_synthesize_system_prompt(hard_triggers),
        user_content=user_content,
        schema=SynthesizeResult,
    )
