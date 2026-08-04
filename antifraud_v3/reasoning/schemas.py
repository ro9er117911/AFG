from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["contact_pretext", "urgency_authority", "isolation", "payment_credential_extraction"]
Strength = Literal["none", "weak", "moderate", "strong"]


class ChunkEvidence(BaseModel):
    """Everything the reasoning engine knows going into one discriminate/reflect/synthesize
    pass. Despite the name, this now carries the *whole call's* transcript + aggregated
    acoustic/emotion summary (pipeline/chunk_worker.py's run_final_analysis) rather than a
    single chunk — the reasoning pipeline itself is unchanged, only called once per call
    instead of once per chunk. This is the input side of the LLMProvider boundary.
    """

    transcript_segment: str
    speaker_guess: str | None = None
    acoustic_summary: str  # human-readable, e.g. "pitch range 210Hz (elevated vs baseline), jitter 1.8%, HNR 12dB, speech rate steady"
    emotion_summary: str  # human-readable, e.g. "neutral 0.61, fear 0.22, happy 0.09, ..."
    call_state_summary: str  # unused now that reasoning runs once per call, not per chunk; kept "" by callers


class StageEvidence(BaseModel):
    stage: Stage
    strength: Strength
    justification: str = Field(description="One or two sentences citing the specific transcript/acoustic evidence.")


class DiscriminateResult(BaseModel):
    stage_evidence: list[StageEvidence]
    overall_note: str = Field(description="Brief holistic read of this chunk, one to two sentences.")


class ReflectedStage(BaseModel):
    stage: Stage
    original_strength: Strength
    revised_strength: Strength
    innocent_explanation: str | None = Field(
        default=None, description="A plausible non-fraud explanation for the flagged evidence, if one exists."
    )
    reasoning: str


class ReflectResult(BaseModel):
    reflected_stages: list[ReflectedStage]


class HardTriggerHit(BaseModel):
    name: str
    fired: bool
    quote: str | None = Field(default=None, description="The exact phrase from the transcript that triggered this, if fired.")


class SynthesizeResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    chunk_risk_score: int = Field(ge=0, le=100, description="Informational only — not arithmetically summed across chunks.")
    hard_triggers: list[HardTriggerHit]
    justification: str = Field(description="Short natural-language justification, kept for audit/human review and shown in the alert banner UI.")
    case_memory_update: str = Field(description="Updated rolling case-memory summary, replaces CallState.case_memory.")
