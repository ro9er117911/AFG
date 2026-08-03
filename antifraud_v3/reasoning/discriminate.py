from ..llm.base import LLMProvider
from .rubric import DISCRIMINATE_SYSTEM_PROMPT
from .schemas import ChunkEvidence, DiscriminateResult


def discriminate(provider: LLMProvider, evidence: ChunkEvidence) -> DiscriminateResult:
    user_content = (
        f"逐字稿片段：「{evidence.transcript_segment}」\n"
        f"說話者：{evidence.speaker_guess or '未知'}\n"
        f"聲學特徵：{evidence.acoustic_summary}\n"
        f"情緒機率：{evidence.emotion_summary}\n"
        f"目前為止的通話狀態：{evidence.call_state_summary or '（通話剛開始，尚無先前狀態）'}"
    )
    return provider.structured_complete(
        system=DISCRIMINATE_SYSTEM_PROMPT,
        user_content=user_content,
        schema=DiscriminateResult,
    )
