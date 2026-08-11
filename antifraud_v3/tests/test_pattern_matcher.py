"""reasoning/pattern_matcher.py — 專利十種詐欺模式的確定性規則表比對（Phase 1，S312）。

前十個測試（一種模式一個正例）逐一對照 docs/patent-ten-patterns-extract.md 表二~表十一的
情緒/語意組合；其餘測試涵蓋門檻邊界、AND 語意、可複選、risk_level gating、空輸入。最後一個
是 pipeline 整合案例，直接驗 pipeline/chunk_worker.py 的 run_final_analysis：S312 的
「判定成立後才標記」現在掛在 reasoning/fusion.py 的 fuse() is_fraud 上，不是 LLM 的 risk_level。
"""

import json

import numpy as np

from antifraud_v3.audio.features import extract_acoustic_features

from antifraud_v3.detectors.deepfake_voice import DeepfakeChunkResult
from antifraud_v3.detectors.scam_semantic import AntiFraudQwenResult, ScamSemanticError
from antifraud_v3.pipeline import chunk_worker
from antifraud_v3.pipeline.call_state import CallState
from antifraud_v3.reasoning.pattern_matcher import DEFAULT_EMOTION_THRESHOLD, match_patterns
from antifraud_v3.reasoning.schemas import (
    DiscriminateResult,
    SemanticFeatureFinding,
    SemanticFeatureItem,
    StageEvidence,
    TextEmotionScores,
)


def emotions(**kwargs) -> TextEmotionScores:
    """All fields default to 0 — pass only the ones a test cares about."""
    return TextEmotionScores(**kwargs)


def features(*present_fields: str, quotes: dict[str, str] | None = None) -> SemanticFeatureFinding:
    """All fields default to not-present — pass the field names that should be present=True,
    optionally with a quote for a subset of them."""
    quotes = quotes or {}
    return SemanticFeatureFinding(
        **{f: SemanticFeatureItem(present=True, quote=quotes.get(f)) for f in present_fields}
    )


# ---- 十種模式各一個正例（原文照抄自表二~表十一） ----


def test_global_pattern_matches_anger_stressful_with_three_semantic_features():
    result = match_patterns(
        emotions(anger=80, stressful=70),
        features("improper_pronoun_use", "lack_of_denial", "subjective_objective_time_mismatch"),
    )
    ids = [p.pattern_id for p in result]
    assert "global" in ids


def test_pattern_1_matches_stressful_focus_language_change():
    result = match_patterns(emotions(stressful=70, focus=65), features("language_change"))
    assert [p.pattern_id for p in result] == ["p1"]


def test_pattern_2_matches_extreme_contradiction_language_change_incoherent():
    result = match_patterns(
        emotions(extreme=90, contradiction=75),
        features("language_change", "incoherent_message"),
    )
    assert [p.pattern_id for p in result] == ["p2"]


def test_pattern_3_matches_stressful_contradiction_denial_non_sequential():
    result = match_patterns(
        emotions(stressful=61, contradiction=60),
        features("lack_of_denial", "non_sequential_message"),
    )
    assert [p.pattern_id for p in result] == ["p3"]


def test_pattern_4_matches_embarrassing_pronoun_spontaneous_correction():
    result = match_patterns(emotions(embarrassing=80), features("improper_pronoun_use", "spontaneous_correction"))
    assert [p.pattern_id for p in result] == ["p4"]


def test_pattern_5_matches_stressful_vigilance_unnecessary_connection_non_sequential():
    result = match_patterns(
        emotions(stressful=70, vigilance=70),
        features("unnecessary_connection", "non_sequential_message"),
    )
    assert [p.pattern_id for p in result] == ["p5"]


def test_pattern_6_matches_contradiction_commitment_unnecessary_connection():
    result = match_patterns(emotions(contradiction=99), features("lack_of_commitment", "unnecessary_connection"))
    assert [p.pattern_id for p in result] == ["p6"]


def test_pattern_7_matches_vigilance_happy_language_change():
    result = match_patterns(emotions(vigilance=60, happy=60), features("language_change"))
    assert [p.pattern_id for p in result] == ["p7"]


def test_pattern_8_matches_vigilance_contradiction_pronoun_non_sequential():
    result = match_patterns(
        emotions(vigilance=65, contradiction=65),
        features("improper_pronoun_use", "non_sequential_message"),
    )
    assert [p.pattern_id for p in result] == ["p8"]


def test_pattern_9_matches_extreme_pronoun_denial():
    result = match_patterns(emotions(extreme=100), features("improper_pronoun_use", "lack_of_denial"))
    assert [p.pattern_id for p in result] == ["p9"]


# ---- 門檻邊界 ----


def test_emotion_score_below_threshold_does_not_match():
    result = match_patterns(emotions(extreme=DEFAULT_EMOTION_THRESHOLD - 1), features("improper_pronoun_use", "lack_of_denial"))
    assert result == []


def test_emotion_score_at_threshold_matches():
    result = match_patterns(emotions(extreme=DEFAULT_EMOTION_THRESHOLD), features("improper_pronoun_use", "lack_of_denial"))
    assert [p.pattern_id for p in result] == ["p9"]


# ---- AND 語意：缺一個條件就不標記 ----


def test_missing_one_required_emotion_does_not_match():
    # p9 needs "extreme"; only stressful is high here (irrelevant to p9).
    result = match_patterns(emotions(stressful=90), features("improper_pronoun_use", "lack_of_denial"))
    assert result == []


def test_missing_one_required_semantic_feature_does_not_match():
    # p9 needs both improper_pronoun_use and lack_of_denial; only one is present.
    result = match_patterns(emotions(extreme=90), features("improper_pronoun_use"))
    assert result == []


# ---- 可複選：一通電話同時中兩種模式 ----


def test_multiple_patterns_can_match_simultaneously():
    # p6 (contradiction + commitment/unnecessary_connection) and p9 (extreme + pronoun/denial)
    # share no required fields, so satisfying both conditions sets should fire both.
    result = match_patterns(
        emotions(contradiction=90, extreme=90),
        features("lack_of_commitment", "unnecessary_connection", "improper_pronoun_use", "lack_of_denial"),
    )
    ids = {p.pattern_id for p in result}
    assert {"p6", "p9"}.issubset(ids)


# ---- 空輸入不炸 ----


def test_empty_input_returns_empty_list_without_raising():
    result = match_patterns(TextEmotionScores(), SemanticFeatureFinding())
    assert result == []


# ---- quotes 收集：只收有引句的命中特徵 ----


def test_matched_pattern_collects_only_non_none_quotes():
    result = match_patterns(
        emotions(extreme=90),
        features("improper_pronoun_use", "lack_of_denial", quotes={"improper_pronoun_use": "反正就是有人辦的"}),
    )
    assert len(result) == 1
    assert result[0].quotes == ["反正就是有人辦的"]


# ---- pipeline 整合案例：S312 gating 改掛在 fuse() 的 is_fraud 上 ----
#
# 這兩個測試在移植到 two-line fusion 架構時重寫過。原本斷言的是 run_reasoning_pipeline() 回傳的
# risk_level，但判定權威已經移到 reasoning/fusion.py 的 fuse()，且 run_reasoning_pipeline 只在
# llm_final_summary_enabled=True 時才跑（預設 False）。現在改為直接驗 run_final_analysis：
# S312「判定成立詐騙行為後才標記」= fusion.is_fraud 為真才標記。


def _call_state_with_transcript() -> CallState:
    """Uses the real extractor rather than a hand-built dict — aggregate_acoustic_features()
    reads a nested structure whose exact shape is that function's business, not this test's."""
    audio = np.zeros(16000, dtype=np.float32)
    cs = CallState()
    cs.record_chunk_signals(
        "測試逐字稿", None,
        extract_acoustic_features(audio, 16000),
        {"neutral": 1.0}, 0.0, audio, timestamp=0.0,
    )
    return cs


def test_patterns_are_marked_when_fusion_says_fraud(monkeypatch):
    """Line 2 判定詐騙 -> fuse() is_fraud=True -> 依 p9 的情緒/語意組合標記出 p9。"""
    discriminated = DiscriminateResult(
        stage_evidence=[StageEvidence(stage="payment_credential_extraction", strength="strong", justification="要求驗證碼")],
        overall_note="高風險",
        text_emotions=emotions(extreme=95),
        semantic_features=features("improper_pronoun_use", "lack_of_denial", quotes={"lack_of_denial": "我沒有說謊"}),
    )
    monkeypatch.setattr(chunk_worker, "discriminate", lambda provider, evidence: discriminated)
    monkeypatch.setattr(chunk_worker, "classify_call_via_llm",
                        lambda provider, transcript: AntiFraudQwenResult(
                            scenario="銀行來電", is_fraud=True, confidence=0.9,
                            fraud_type_raw="banking_fraud", reasoning_trace=[]))

    out = chunk_worker.run_final_analysis(object(), _call_state_with_transcript())
    assert out is not None
    result, _alert, _evidence = out
    assert result.risk_level == "high"
    assert [p.pattern_id for p in result.matched_patterns] == ["p9"]
    assert result.matched_patterns[0].quotes == ["我沒有說謊"]


def test_patterns_are_not_marked_when_fusion_says_no_fraud(monkeypatch):
    """同一組會命中 p9 的證據，但 fuse() 判定不成立 -> 依 S312 不得標記，且 discriminate 不應被呼叫。"""
    called = []
    monkeypatch.setattr(chunk_worker, "discriminate",
                        lambda provider, evidence: called.append(1) or DiscriminateResult(
                            stage_evidence=[], overall_note="",
                            text_emotions=emotions(extreme=95),
                            semantic_features=features("improper_pronoun_use", "lack_of_denial")))
    monkeypatch.setattr(chunk_worker, "classify_call_via_llm",
                        lambda provider, transcript: AntiFraudQwenResult(
                            scenario="日常閒聊", is_fraud=False, confidence=0.1,
                            fraud_type_raw=None, reasoning_trace=[]))

    out = chunk_worker.run_final_analysis(object(), _call_state_with_transcript())
    assert out is not None
    result, _alert, _evidence = out
    assert result.risk_level == "low"
    assert result.matched_patterns == []
    assert called == [], "S312 未判定成立時不應呼叫 discriminate"


# ---- 降級路徑：偵測器掛掉不得炸掉整條分析（demo 可靠性）----


def test_line1_failure_does_not_abort_the_analysis(monkeypatch):
    """Line 1 沒有 CUDA/模型時會丟例外。整條分析必須照樣產出判定，而不是讓串流中斷。"""
    monkeypatch.setattr(chunk_worker, "predict_deepfake",
                        lambda y, sr: (_ for _ in ()).throw(RuntimeError("no CUDA")))
    monkeypatch.setattr(chunk_worker, "transcribe_chunk", lambda y, sr: "測試逐字稿")
    monkeypatch.setattr(chunk_worker, "predict_emotion", lambda y, sr: ("neutral", {"neutral": 1.0}))

    out = chunk_worker.process_chunk_signals(np.zeros(16000, dtype=np.float32), 16000, CallState())
    assert out is not None, "Line 1 失敗不得讓整個 chunk 被丟掉"
    assert out.deepfake["label"] == "unavailable", "必須能區分「沒檢查」與「確定不是深偽」"


def test_line2_non_scamsemantic_error_does_not_abort_the_analysis(monkeypatch):
    """Line 2 載模型時丟的是 ImportError（缺 bitsandbytes），不是 ScamSemanticError。
    舊的 except 子句接不到，會炸掉整條串流；現在必須降級成沒有 Line 2 的判定。"""
    monkeypatch.setattr(chunk_worker, "classify_call_via_llm",
                        lambda provider, transcript: (_ for _ in ()).throw(
                            ScamSemanticError("claude backend unavailable")))
    monkeypatch.setattr(chunk_worker, "classify_call",
                        lambda *a, **kw: (_ for _ in ()).throw(ImportError("requires bitsandbytes")))

    out = chunk_worker.run_final_analysis(object(), _call_state_with_transcript())
    assert out is not None
    result, _alert, _evidence = out
    assert result.risk_level == "low"
    assert result.fusion_source == "none"


# ---- API 合約：模式標記必須真的送到前端 ----


def test_matched_patterns_reach_the_streaming_api(monkeypatch):
    """回歸測試：S312 的標記算得出來但沒被序列化進 final_analysis 事件，前端就永遠看不到。
    這正是移植後實際發生過的缺口——後端邏輯正確，但 upload.py/ws.py 的事件沒帶這個欄位。"""
    from fastapi.testclient import TestClient
    from antifraud_v3.server.main import app

    monkeypatch.setattr(chunk_worker, "predict_deepfake",
                        lambda y, sr: DeepfakeChunkResult(fake_score=0.0, label="bonafide"))
    monkeypatch.setattr(chunk_worker, "classify_call_via_llm",
                        lambda provider, transcript: AntiFraudQwenResult(
                            scenario="銀行來電", is_fraud=True, confidence=0.9,
                            fraud_type_raw="banking_fraud", reasoning_trace=[]))
    monkeypatch.setattr(chunk_worker, "discriminate", lambda provider, evidence: DiscriminateResult(
        stage_evidence=[StageEvidence(stage="payment_credential_extraction", strength="strong", justification="要求驗證碼")],
        overall_note="高風險",
        text_emotions=emotions(extreme=95),
        semantic_features=features("improper_pronoun_use", "lack_of_denial", quotes={"lack_of_denial": "我沒有說謊"}),
    ))

    final = None
    with TestClient(app).stream("POST", "/api/test-clips/scam/bank_otp_request.wav/analyze") as r:
        for line in r.iter_lines():
            if line and '"final_analysis"' in line:
                final = json.loads(line)

    assert final is not None, "沒有收到 final_analysis 事件"
    assert "matched_patterns" in final, "final_analysis 必須帶 matched_patterns 欄位"
    assert [p["pattern_id"] for p in final["matched_patterns"]] == ["p9"]
    assert final["matched_patterns"][0]["quotes"] == ["我沒有說謊"]
