"""Live smoke tests — hit the real Claude API through llm/get_llm_provider() and the actual
reasoning engine (docs/DESIGN.md §4), not FakeLLMProvider. These are the only tests in the
suite that need ANTHROPIC_API_KEY / a valid `ant auth login` session and cost real API calls.

Run the whole suite including these with plain `pytest`; skip them with
`pytest -m "not live_api"` (e.g. no credential available, or in CI) — see pyproject.toml's
marker registration.

IMPORTANT — architecture note (two-line fusion, reasoning/fusion.py): these tests used to
assert `result.risk_level` directly, regression-testing this project's founding bug (the old
rule engine flagging calm speech as high-risk) against the *reasoning pipeline's own* judgment.
That's no longer what these tests can check — the reasoning pipeline no longer decides
risk_level/fraud_type at all (`run_reasoning_pipeline` now returns ClaudeJustification, not
SynthesizeResult); that verdict comes from Line 1/2 + fuse() instead. What's left for Claude to
get right is narrower but still worth checking live: given a verdict it's *handed*, does its
prose actually agree with it, not contradict it (the exact failure mode
build_synthesize_system_prompt now explicitly warns against). The scam-vs-benign classification
regression check itself has moved to Line 2 (detectors/scam_semantic.py) — that needs its own
live test against the real (large, slow) AntiFraud-Qwen2Audio model, not written here.
"""

import pytest

from antifraud_v3.llm import get_llm_provider
from antifraud_v3.reasoning import run_reasoning_pipeline
from antifraud_v3.reasoning.schemas import ChunkEvidence, FraudTypeClassification, FusionResult

pytestmark = pytest.mark.live_api


def test_live_justification_agrees_with_a_high_risk_fusion_verdict():
    """Given fuse() already decided this call is high-risk (as if Line 2 had fired on an OTP
    request), Claude's justification prose should reflect that, not write something that reads
    as reassuring/normal — the exact inconsistency build_synthesize_system_prompt now warns
    against."""
    provider = get_llm_provider()
    evidence = ChunkEvidence(
        transcript_segment="我們需要立即幫您做圈存保護，避免持續被扣款，請把簡訊收到的驗證碼提供給我核對身分",
        speaker_guess=None,
        acoustic_summary="語速正常，音量穩定，音高變化不大",
        emotion_summary="neutral 0.65, fear 0.15, happy 0.05",
        call_state_summary="",
    )
    fusion = FusionResult(
        fusion_source="line2_semantic",
        fake_score=0.0,
        ai_voice_flag=False,
        is_fraud=True,
        risk_level="high",
        chunk_risk_score=90,
        fraud_type=FraudTypeClassification(fraud_type="banking_fraud", confidence="strong", justification="要求提供驗證碼"),
    )
    result = run_reasoning_pipeline(provider, evidence, fusion)
    assert "正常" not in result.justification and "沒有" not in result.justification


def test_live_justification_agrees_with_a_low_risk_fusion_verdict():
    """The exact kind of ordinary conversation antifraud_v2's structural bug used to
    misclassify as high-risk purely from a calm tone (docs/DESIGN.md §1). Given fuse() already
    decided this call is low-risk/normal, Claude's justification prose shouldn't invent alarm
    that isn't there."""
    provider = get_llm_provider()
    evidence = ChunkEvidence(
        transcript_segment="我們晚上要不要一起吃飯，我看了家新開的餐廳評價還不錯，要不要試試看",
        speaker_guess=None,
        acoustic_summary="語速正常，音量穩定，音高變化不大",
        emotion_summary="neutral 0.8, happy 0.15, fear 0.02",
        call_state_summary="",
    )
    fusion = FusionResult(
        fusion_source="none",
        fake_score=0.0,
        ai_voice_flag=False,
        is_fraud=False,
        risk_level="low",
        chunk_risk_score=2,
        fraud_type=FraudTypeClassification(fraud_type="unclassified", confidence="none", justification="無明顯詐騙敘事"),
    )
    result = run_reasoning_pipeline(provider, evidence, fusion)
    assert "驗證碼" not in result.justification and "轉帳" not in result.justification
