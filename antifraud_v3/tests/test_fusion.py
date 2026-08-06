"""reasoning/fusion.py's fuse()/build_synthesize_result() — pure functions, no model needed
(same "test the wiring, not model quality" split as test_reasoning_pipeline.py vs
test_live_reasoning.py). Covers the user's exact 3-branch priority spec verbatim."""

from antifraud_v3.detectors.scam_semantic import AntiFraudQwenResult
from antifraud_v3.reasoning.fusion import build_synthesize_result, fuse
from antifraud_v3.reasoning.schemas import ClaudeJustification

THRESHOLD = 0.85


def test_high_fake_score_wins_regardless_of_line2():
    line2 = AntiFraudQwenResult(scenario="打车服务", is_fraud=False, confidence=0.9)
    result = fuse(0.95, line2, THRESHOLD)
    assert result.fusion_source == "line1_ai_voice"
    assert result.ai_voice_flag is True
    assert result.is_fraud is True
    assert result.risk_level == "high"


def test_high_fake_score_wins_even_when_line2_is_none():
    result = fuse(0.99, None, THRESHOLD)
    assert result.fusion_source == "line1_ai_voice"
    assert result.is_fraud is True


def test_low_fake_score_and_line2_fraud_uses_semantic_branch():
    line2 = AntiFraudQwenResult(scenario="银行客服", is_fraud=True, confidence=0.8, fraud_type_raw="银行诈骗")
    result = fuse(0.1, line2, THRESHOLD)
    assert result.fusion_source == "line2_semantic"
    assert result.ai_voice_flag is False
    assert result.is_fraud is True
    assert result.risk_level == "high"  # confidence >= 0.7
    assert result.fraud_type.fraud_type == "banking_fraud"


def test_low_fake_score_and_lower_confidence_line2_fraud_is_medium_risk():
    line2 = AntiFraudQwenResult(scenario="客服", is_fraud=True, confidence=0.55, fraud_type_raw="钓鱼诈骗")
    result = fuse(0.1, line2, THRESHOLD)
    assert result.fusion_source == "line2_semantic"
    assert result.risk_level == "medium"
    assert result.fraud_type.fraud_type == "phishing_fraud"


def test_both_low_is_normal_call():
    line2 = AntiFraudQwenResult(scenario="打车服务", is_fraud=False, confidence=0.9)
    result = fuse(0.1, line2, THRESHOLD)
    assert result.fusion_source == "none"
    assert result.is_fraud is False
    assert result.risk_level == "low"
    assert result.fraud_type.fraud_type == "unclassified"


def test_line1_branch_keeps_line2_fraud_type_not_suppressed():
    """A cloned voice can still be running a banking-fraud script — fraud_type shouldn't be
    thrown away just because Line 1 "won" the priority ordering (see fusion.py's module
    docstring: ai_voice_flag/fusion_source and fraud_type are independent axes)."""
    line2 = AntiFraudQwenResult(scenario="银行客服", is_fraud=True, confidence=0.9, fraud_type_raw="银行诈骗")
    result = fuse(0.95, line2, THRESHOLD)
    assert result.fusion_source == "line1_ai_voice"
    assert result.fraud_type.fraud_type == "banking_fraud"


def test_build_synthesize_result_uses_claude_prose_when_available():
    fusion = fuse(0.1, None, THRESHOLD)
    claude = ClaudeJustification(hard_triggers=[], justification="Claude 寫的說明", case_memory_update="摘要")
    result = build_synthesize_result(fusion, claude)
    assert result.justification == "Claude 寫的說明"
    assert result.case_memory_update == "摘要"
    assert result.risk_level == fusion.risk_level


def test_build_synthesize_result_falls_back_to_template_without_claude():
    fusion = fuse(0.95, None, THRESHOLD)
    result = build_synthesize_result(fusion, None)
    assert result.justification  # non-empty fallback text
    assert result.risk_level == "high"
    assert result.ai_voice_flag is True
