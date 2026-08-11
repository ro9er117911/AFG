from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["contact_pretext", "urgency_authority", "isolation", "payment_credential_extraction"]
Strength = Literal["none", "weak", "moderate", "strong"]

# TeleAntiFraud-28k's 7-category fraud-TYPE taxonomy (see ../../references/05-teleantifraud.md) —
# a different axis from Stage above: Stage asks "where in the scam process is this call",
# FraudType asks "what kind of scam narrative does this call resemble". Independent, both can
# be reported for the same call (see rubric.py's FRAUD_TYPE_TAXONOMY and
# build_synthesize_system_prompt's instruction that this classification must not be derived
# from / gated by risk_level).
FraudType = Literal[
    "investment_fraud",
    "phishing_fraud",
    "identity_theft",
    "lottery_fraud",
    "banking_fraud",
    "extortion_fraud",
    "customer_service_fraud",
    "unclassified",
]


class TextEmotionScores(BaseModel):
    """13 類文字情緒分數（0-100），對應專利 S309「文字情緒分析結果」。每一欄位是 LLM 對逐字稿
    整體語氣的評分，分數越高代表該情緒在文字中的強度越明顯。所有欄位預設 0（中性/未評估），
    讓既有程式在未提供這個區塊時仍可建構出合法的 DiscriminateResult（見該 schema 的擴充註記）。
    """

    anger: int = Field(default=0, ge=0, le=100, description="生氣")
    disgust: int = Field(default=0, ge=0, le=100, description="厭惡")
    fear: int = Field(default=0, ge=0, le=100, description="害怕")
    sadness: int = Field(default=0, ge=0, le=100, description="悲傷")
    surprise: int = Field(default=0, ge=0, le=100, description="驚訝")
    happy: int = Field(default=0, ge=0, le=100, description="開心")
    neutral: int = Field(default=0, ge=0, le=100, description="中立")
    stressful: int = Field(default=0, ge=0, le=100, description="壓力")
    extreme: int = Field(default=0, ge=0, le=100, description="激動")
    focus: int = Field(default=0, ge=0, le=100, description="專注")
    contradiction: int = Field(default=0, ge=0, le=100, description="矛盾")
    embarrassing: int = Field(default=0, ge=0, le=100, description="尷尬")
    vigilance: int = Field(default=0, ge=0, le=100, description="警覺")

class SemanticFeatureItem(BaseModel):
    """表一單一項「不合理語意特徵」的判定結果。quote 是逐字稿中支持這個判定的原句佐證，
    未出現該特徵時 present=False、quote=None——不強迫模型硬找一句不相關的話來湊。
    """

    present: bool = False
    quote: str | None = None

class SemanticFeatureFinding(BaseModel):
    """表一（docs/patent-ten-patterns-extract.md）9 項「不合理語意特徵」的完整判定結果，對應
    專利 S310「語意不合理分析結果」。欄位名為特徵的英文代稱，逐一對應原文 9 項名稱（原順序）：
    不當使用代詞、缺乏否認、時間的主觀與客觀不一致、語言改變、不連貫的訊息、非順序訊息、
    自發修正、不必要的連接、缺乏承諾。每欄預設「未出現」，理由同 TextEmotionScores。
    """

    improper_pronoun_use: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    lack_of_denial: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    subjective_objective_time_mismatch: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    language_change: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    incoherent_message: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    non_sequential_message: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    spontaneous_correction: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    unnecessary_connection: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)
    lack_of_commitment: SemanticFeatureItem = Field(default_factory=SemanticFeatureItem)

class MatchedPattern(BaseModel):
    """規則表（reasoning/pattern_matcher.py）比對命中的一種詐欺模式——專利 S312「詐騙模式
    分析結果」的一筆記錄。pattern_id 是 "global" 或 "p1".."p9"；emotions/semantic_features
    是命中該模式所需的情緒/語意特徵欄位名（用於前端顯示「為什麼」）；quotes 收集命中的語意
    特徵中有提供逐字稿佐證引句的部分（可能為空，若命中的特徵都沒有 quote）。
    """

    pattern_id: str
    name: str
    emotions: list[str]
    semantic_features: list[str]
    quotes: list[str]


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
    text_emotions: TextEmotionScores = Field(default_factory=TextEmotionScores)
    semantic_features: SemanticFeatureFinding = Field(default_factory=SemanticFeatureFinding)


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


class FraudTypeClassification(BaseModel):
    """Which of TeleAntiFraud-28k's 7 fraud-type categories this call's narrative resembles —
    descriptive, independent of risk_level (see rubric.py's FRAUD_TYPE_TAXONOMY)."""

    fraud_type: FraudType
    confidence: Strength
    justification: str = Field(description="One or two sentences citing the specific transcript content this classification is based on.")


class ClaudeJustification(BaseModel):
    """What the (now much smaller) synthesize() step actually asks Claude for: the hard-trigger
    checklist evaluation plus prose consistent with a verdict it's *handed*, not one it invents.
    See reasoning/fusion.py's module docstring for why the risk-defining fields moved out of
    Claude's own judgment — this is the LLM-output-shaped subset of SynthesizeResult that's
    still worth an LLM call, exactly 3 of its 6 fields, copied verbatim."""

    hard_triggers: list[HardTriggerHit]
    justification: str = Field(description="Short natural-language justification, kept for audit/human review and shown in the alert banner UI.")
    case_memory_update: str = Field(description="Updated rolling case-memory summary, replaces CallState.case_memory.")


class FusionResult(BaseModel):
    """reasoning/fusion.py's decision object — never sent to or returned by an LLM. The
    authoritative fraud verdict for a call, per the user's own 3-branch priority spec: Line 1
    (AI/cloned-voice) fake_score high wins outright; else Line 2 (AntiFraud-Qwen2Audio)
    is_fraud=true; else normal. Fed into synthesize() as input (see fusion.py) rather than
    applied as a post-hoc patch over Claude's output, so justification prose is never generated
    from a different verdict than the one actually being reported."""

    fusion_source: Literal["line1_ai_voice", "line2_semantic", "none"]
    fake_score: float = Field(ge=0, le=1)
    ai_voice_flag: bool
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    chunk_risk_score: int = Field(ge=0, le=100)
    fraud_type: FraudTypeClassification


class SynthesizeResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    chunk_risk_score: int = Field(ge=0, le=100, description="Informational only — not arithmetically summed across chunks.")
    hard_triggers: list[HardTriggerHit]
    justification: str = Field(description="Short natural-language justification, kept for audit/human review and shown in the alert banner UI.")
    case_memory_update: str = Field(description="Updated rolling case-memory summary, replaces CallState.case_memory.")
    fraud_type: FraudTypeClassification
    # Added for the two-line (XLS-R deepfake + AntiFraud-Qwen2Audio) fusion — see
    # reasoning/fusion.py. risk_level/chunk_risk_score/fraud_type above are now sourced from
    # FusionResult (via build_synthesize_result), not derived by Claude itself.
    fake_score: float = Field(default=0.0, ge=0, le=1, description="Line 1's max per-chunk AI/cloned-voice score for this call.")
    ai_voice_flag: bool = Field(default=False, description="True if fake_score crossed the threshold — Line 1 fired, takes priority over Line 2.")
    fusion_source: Literal["line1_ai_voice", "line2_semantic", "none"] = "none"
    # 專利 TW I904863 步驟 S312「詐騙模式分析結果」——fuse() 判定 is_fraud 成立後，由
    # reasoning/pattern_matcher.py 查表產生（見 pipeline/chunk_worker.py 的呼叫點）。
    # 預設空清單：判定不成立、或未取得文字情緒/語意特徵時，就是沒有模式可標記。
    matched_patterns: list[MatchedPattern] = Field(default_factory=list)
