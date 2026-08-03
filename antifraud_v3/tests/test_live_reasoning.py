"""Live smoke tests — hit the real Claude API through llm/get_llm_provider() and the actual
reasoning engine (docs/DESIGN.md §4), not FakeLLMProvider. These are the only tests in the
suite that need ANTHROPIC_API_KEY / a valid `ant auth login` session and cost real API calls.

Run the whole suite including these with plain `pytest`; skip them with
`pytest -m "not live_api"` (e.g. no credential available, or in CI) — see pyproject.toml's
marker registration.
"""

import pytest

from antifraud_v3.llm import get_llm_provider
from antifraud_v3.reasoning import run_reasoning_pipeline
from antifraud_v3.reasoning.schemas import ChunkEvidence

pytestmark = pytest.mark.live_api


def test_live_hard_trigger_phrase_is_flagged_high_risk():
    """Direct regression check for this project's founding bug (see docs/DESIGN.md §1): the
    old rule engine could flag calm/normal speech as high-risk. This test checks the opposite
    failure isn't happening either — an unambiguous scam phrase (OTP request under a fake
    "account protection" pretext) should reliably score medium/high."""
    provider = get_llm_provider()
    evidence = ChunkEvidence(
        transcript_segment="我們需要立即幫您做圈存保護，避免持續被扣款，請把簡訊收到的驗證碼提供給我核對身分",
        speaker_guess=None,
        acoustic_summary="語速正常，音量穩定，音高變化不大",
        emotion_summary="neutral 0.65, fear 0.15, happy 0.05",
        call_state_summary="",
    )
    result = run_reasoning_pipeline(provider, evidence)
    assert result.risk_level in ("medium", "high")


def test_live_benign_dinner_chat_stays_low_risk():
    """The exact kind of ordinary conversation antifraud_v2's structural bug used to
    misclassify as high-risk purely from a calm tone — see docs/DESIGN.md §1's root-cause
    writeup. This is the false-positive check that matters most for this project."""
    provider = get_llm_provider()
    evidence = ChunkEvidence(
        transcript_segment="我們晚上要不要一起吃飯，我看了家新開的餐廳評價還不錯，要不要試試看",
        speaker_guess=None,
        acoustic_summary="語速正常，音量穩定，音高變化不大",
        emotion_summary="neutral 0.8, happy 0.15, fear 0.02",
        call_state_summary="",
    )
    result = run_reasoning_pipeline(provider, evidence)
    assert result.risk_level == "low"
