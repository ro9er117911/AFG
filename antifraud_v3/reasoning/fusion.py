"""The fraud verdict itself now lives here, not in synthesize()'s own LLM judgment — a
deliberate architecture change (the user's own decision, "hybrid" scope): the two audio-native
detectors (detectors/deepfake_voice.py's Line 1, detectors/scam_semantic.py's Line 2) produce
the authoritative is_fraud/risk_level/fraud_type, and Claude (if configured at all) only writes
justification prose *consistent with* that verdict — see synthesize.py's much-shrunk job.

fuse() implements the user's own 3-branch priority spec verbatim:
  1. Line 1 fake_score >= threshold -> "AI cloned/synthesized voice impersonation," reported
     with priority, regardless of what Line 2 says.
  2. Else, Line 2 is_fraud=true -> "scam-script fraud."
  3. Else -> normal call.

fuse() must run *before* synthesize()'s LLM call and be fed into it as input, not applied as a
post-hoc patch over Claude's own output — otherwise Claude could write prose contradicting the
fusion verdict (e.g. "this seems normal" next to a Line-1-triggered "AI cloned voice, high
risk"), a real inconsistency a silent overwrite would not catch. See
pipeline/chunk_worker.py:run_final_analysis for the actual call order.
"""

from ..detectors.scam_semantic import AntiFraudQwenResult, map_fraud_type
from .schemas import ClaudeJustification, FraudTypeClassification, FusionResult, MatchedPattern, SynthesizeResult

_NO_SIGNAL_FRAUD_TYPE = FraudTypeClassification(
    fraud_type="unclassified", confidence="none", justification="Line 1（合成語音偵測）與 Line 2（話術詐騙偵測）均未偵測到明顯訊號。"
)


def fuse(fake_score: float, line2: AntiFraudQwenResult | None, threshold: float) -> FusionResult:
    if fake_score >= threshold:
        return FusionResult(
            fusion_source="line1_ai_voice",
            fake_score=fake_score,
            ai_voice_flag=True,
            is_fraud=True,
            risk_level="high",
            chunk_risk_score=round(fake_score * 100),
            # Deliberately NOT suppressed even though Line 1 "won": a cloned voice can still be
            # running a banking-fraud script, and Line 2 already ran unconditionally once per
            # call regardless of Line 1's score — dropping that context would throw away real
            # information. fraud_type and ai_voice_flag/fusion_source are independent axes, same
            # principle as FraudType's existing stage-vs-type independence (see schemas.py).
            fraud_type=(
                FraudTypeClassification(
                    fraud_type=map_fraud_type(line2.fraud_type_raw),
                    confidence="moderate" if line2.confidence >= 0.5 else "weak",
                    justification=f"Line 2（話術詐騙偵測）判斷：{line2.fraud_type_raw or '未提供詐騙類型'}（信心 {line2.confidence:.2f}）。",
                )
                if line2 is not None and line2.is_fraud
                else _NO_SIGNAL_FRAUD_TYPE
            ),
        )

    if line2 is not None and line2.is_fraud:
        risk_level = "high" if line2.confidence >= 0.7 else "medium"
        return FusionResult(
            fusion_source="line2_semantic",
            fake_score=fake_score,
            ai_voice_flag=False,
            is_fraud=True,
            risk_level=risk_level,
            chunk_risk_score=max(1, round(line2.confidence * 100)),
            fraud_type=FraudTypeClassification(
                fraud_type=map_fraud_type(line2.fraud_type_raw),
                confidence="strong" if line2.confidence >= 0.7 else "moderate",
                justification=f"Line 2（話術詐騙偵測）判斷：{line2.fraud_type_raw or '未提供詐騙類型'}（信心 {line2.confidence:.2f}）。",
            ),
        )

    return FusionResult(
        fusion_source="none",
        fake_score=fake_score,
        ai_voice_flag=False,
        is_fraud=False,
        risk_level="low",
        chunk_risk_score=round(fake_score * 10),  # informational only — keep it near-zero, not literally 0
        fraud_type=_NO_SIGNAL_FRAUD_TYPE,
    )


_FALLBACK_TEMPLATES = {
    "line1_ai_voice": "系統偵測到本通話音訊有極高機率為 AI 合成或複製聲音（fake_score={fake_score:.2f}），建議提高警覺，不要僅憑聲音確認對方身分。",
    "line2_semantic": "系統依話術內容判斷本通話疑似詐騙（信心程度：{risk_level}），研判類型為「{fraud_type}」。",
    "none": "系統未偵測到明顯的合成語音或詐騙話術跡象，本通話研判為正常對話。",
}


def build_synthesize_result(
    fusion: FusionResult,
    claude: ClaudeJustification | None,
    matched_patterns: list[MatchedPattern] | None = None,
) -> SynthesizeResult:
    """Merges fusion's risk-defining fields with Claude's prose. claude=None (provider not
    configured, or the LLM call failed — see run_final_analysis's try/except) falls back to a
    template string built from fusion's own fields, so the app can still produce *a* verdict
    without a working LLM provider — a real robustness property unlocked by the verdict no
    longer depending on Claude's own judgment at all.

    matched_patterns carries 專利 TW I904863 S312's fraud-pattern marks. It's a separate
    parameter rather than being derived here because S312 gates on "詐騙行為判定成立" —
    fusion.is_fraud — while the marks themselves need text-emotion/semantic-feature evidence
    that only a discriminate() call produces; the caller owns that sequencing (see
    pipeline/chunk_worker.py's run_final_analysis). None and [] both mean "no marks", the
    former because the patent path didn't run, the latter because it ran and matched nothing."""
    if claude is not None:
        hard_triggers = claude.hard_triggers
        justification = claude.justification
        case_memory_update = claude.case_memory_update
    else:
        hard_triggers = []
        justification = _FALLBACK_TEMPLATES[fusion.fusion_source].format(
            fake_score=fusion.fake_score, risk_level=fusion.risk_level, fraud_type=fusion.fraud_type.fraud_type
        )
        case_memory_update = ""

    return SynthesizeResult(
        risk_level=fusion.risk_level,
        chunk_risk_score=fusion.chunk_risk_score,
        hard_triggers=hard_triggers,
        justification=justification,
        case_memory_update=case_memory_update,
        fraud_type=fusion.fraud_type,
        fake_score=fusion.fake_score,
        ai_voice_flag=fusion.ai_voice_flag,
        fusion_source=fusion.fusion_source,
        matched_patterns=matched_patterns or [],
    )
