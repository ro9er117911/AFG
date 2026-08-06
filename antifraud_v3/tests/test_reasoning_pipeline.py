"""reasoning/discriminate.py -> reflect.py -> synthesize.py chaining (docs/DESIGN.md §4),
tested against FakeLLMProvider (tests/conftest.py) rather than a real model — this checks the
*wiring* (each step gets called with the right schema, in the right order, reflect's
short-circuit optimization actually skips the LLM call when nothing was flagged), not
reasoning quality, which is what the tests/test_live_reasoning.py-marked live tests are for.
"""

from antifraud_v3.reasoning import run_reasoning_pipeline
from antifraud_v3.reasoning.reflect import reflect
from antifraud_v3.reasoning.rubric import DEFAULT_HARD_TRIGGERS
from antifraud_v3.reasoning.schemas import (
    ChunkEvidence,
    ClaudeJustification,
    DiscriminateResult,
    FusionResult,
    ReflectedStage,
    ReflectResult,
    StageEvidence,
)
from antifraud_v3.tests.conftest import DEFAULT_FRAUD_TYPE, FakeLLMProvider


def make_evidence(text="測試逐字稿"):
    return ChunkEvidence(
        transcript_segment=text,
        speaker_guess=None,
        acoustic_summary="語速正常，音量穩定",
        emotion_summary="neutral 0.8, happy 0.1",
        call_state_summary="",
    )


def make_fusion(risk_level="low", is_fraud=False, fusion_source="none"):
    """run_reasoning_pipeline no longer decides risk_level/fraud_type itself (see
    reasoning/fusion.py) — it's handed a FusionResult as context. These reasoning-pipeline
    wiring tests only care that the pipeline chains correctly and returns the LLM's prose, so a
    minimal fixture is enough; fusion.py's own decision logic has its own tests."""
    return FusionResult(
        fusion_source=fusion_source,
        fake_score=0.0,
        ai_voice_flag=False,
        is_fraud=is_fraud,
        risk_level=risk_level,
        chunk_risk_score=90 if risk_level == "high" else 2,
        fraud_type=DEFAULT_FRAUD_TYPE,
    )


def test_reflect_short_circuits_when_nothing_flagged():
    """See reasoning/reflect.py: if discriminate found nothing (all strength="none"), reflect
    shouldn't spend an LLM call arguing against evidence that doesn't exist."""
    fake = FakeLLMProvider()
    discriminated = DiscriminateResult(
        stage_evidence=[StageEvidence(stage="contact_pretext", strength="none", justification="無證據")],
        overall_note="無明顯證據",
    )
    result = reflect(fake, make_evidence(), discriminated)
    assert fake.calls == []
    assert result.reflected_stages[0].revised_strength == "none"
    assert result.reflected_stages[0].stage == "contact_pretext"


def test_reflect_calls_provider_when_something_flagged():
    fake = FakeLLMProvider()
    discriminated = DiscriminateResult(
        stage_evidence=[StageEvidence(stage="payment_credential_extraction", strength="strong", justification="要求驗證碼")],
        overall_note="要求驗證碼",
    )
    reflect(fake, make_evidence("請提供簡訊驗證碼"), discriminated)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "ReflectResult"


class ScamScriptedProvider(FakeLLMProvider):
    """Simulates a provider that consistently finds strong evidence, to exercise the full
    three-call chain (discriminate -> reflect -> synthesize) rather than reflect's
    short-circuit path."""

    def structured_complete(self, system, user_content, schema):
        self.calls.append((schema.__name__, system, user_content))
        if schema is DiscriminateResult:
            return DiscriminateResult(
                stage_evidence=[
                    StageEvidence(stage="payment_credential_extraction", strength="strong", justification="要求驗證碼")
                ],
                overall_note="高風險",
            )
        if schema is ReflectResult:
            return ReflectResult(
                reflected_stages=[
                    ReflectedStage(
                        stage="payment_credential_extraction",
                        original_strength="strong",
                        revised_strength="strong",
                        innocent_explanation=None,
                        reasoning="沒有合理的無辜解釋",
                    )
                ]
            )
        if schema is ClaudeJustification:
            return self._claude_justification
        raise AssertionError(f"unexpected schema {schema}")


def test_run_reasoning_pipeline_chains_all_three_steps_in_order():
    provider = ScamScriptedProvider(
        claude_justification=ClaudeJustification(
            hard_triggers=[],
            justification="要求提供驗證碼",
            case_memory_update="已要求驗證碼一次",
        )
    )
    result = run_reasoning_pipeline(
        provider, make_evidence("請提供簡訊驗證碼"), make_fusion(risk_level="high"), hard_triggers=DEFAULT_HARD_TRIGGERS
    )

    assert [c[0] for c in provider.calls] == ["DiscriminateResult", "ReflectResult", "ClaudeJustification"]
    assert result.case_memory_update == "已要求驗證碼一次"


def test_run_reasoning_pipeline_low_risk_path_still_calls_all_steps():
    # Even a low-risk chunk goes through discriminate; only reflect short-circuits when
    # nothing was flagged (tested above) — synthesize always runs since it also evaluates
    # hard triggers independently of stage evidence.
    fake = FakeLLMProvider(
        claude_justification=ClaudeJustification(
            hard_triggers=[],
            justification="正常對話",
            case_memory_update="",
        )
    )
    result = run_reasoning_pipeline(fake, make_evidence("我們晚上要不要一起吃飯"), make_fusion(risk_level="low"))
    assert result.justification == "正常對話"
    called_schemas = [c[0] for c in fake.calls]
    assert "DiscriminateResult" in called_schemas
    assert "ClaudeJustification" in called_schemas
