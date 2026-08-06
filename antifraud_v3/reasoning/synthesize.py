from ..llm.base import LLMProvider
from .rubric import build_synthesize_system_prompt
from .schemas import ChunkEvidence, ClaudeJustification, FusionResult, ReflectResult


def synthesize(
    provider: LLMProvider,
    evidence: ChunkEvidence,
    reflected: ReflectResult,
    fusion: FusionResult,
    hard_triggers: list[str] | None = None,
) -> ClaudeJustification:
    """Shrunk job (see reasoning/fusion.py's module docstring): `fusion` already carries the
    decided risk_level/fraud_type — this call only evaluates the hard-trigger checklist and
    writes justification/case-memory prose *consistent with* fusion's verdict, not a fresh one.
    """
    stage_lines = "\n".join(
        f"- {s.stage}: {s.revised_strength}"
        + (f"（原始 {s.original_strength}，理由：{s.reasoning}）" if s.revised_strength != s.original_strength else "")
        for s in reflected.reflected_stages
    )
    fusion_line = (
        f"系統的自動判斷結果：{fusion.fusion_source}（風險等級：{fusion.risk_level}，"
        f"是否詐騙：{fusion.is_fraud}，詐騙類型：{fusion.fraud_type.fraud_type}）。"
        "請在你的說明文字中呼應這個結論，不要自己重新判斷風險等級或詐騙類型。"
    )
    user_content = (
        f"逐字稿片段：「{evidence.transcript_segment}」\n\n"
        f"反思後的階段證據：\n{stage_lines}\n\n"
        f"{fusion_line}\n\n"
        "根據以上證據、立即示警關鍵字清單、以及系統的自動判斷結果，產出這通電話的立即示警檢查與說明文字。"
    )
    return provider.structured_complete(
        system=build_synthesize_system_prompt(hard_triggers),
        user_content=user_content,
        schema=ClaudeJustification,
    )
