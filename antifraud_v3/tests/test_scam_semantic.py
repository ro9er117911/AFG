"""detectors/scam_semantic.py's pure-function pieces — map_fraud_type() and _parse_response()
— no model needed. classify_call() itself needs the real ~7GB AntiFraud-SFT model and isn't
covered by the normal fast test loop; it was verified manually against real eval clips (see
project notes) rather than here.
"""

import pytest

from antifraud_v3.detectors.scam_semantic import ScamSemanticError, _parse_response, map_fraud_type


def test_map_fraud_type_handles_real_captured_compound_label():
    """Real generation captured 2026-08-05 against eval/test_clips/scam/bank_otp_request.wav:
    the model's actual output, "冒充公检法/金融机构类诈骗", combines two categories in one
    compound label — identity_theft should win over banking_fraud since the impersonation is
    the core mechanism (see this module's _FRAUD_TYPE_KEYWORDS comment)."""
    assert map_fraud_type("冒充公检法/金融机构类诈骗") == "identity_theft"


def test_map_fraud_type_banking():
    assert map_fraud_type("银行诈骗") == "banking_fraud"


def test_map_fraud_type_phishing():
    assert map_fraud_type("钓鱼诈骗") == "phishing_fraud"


def test_map_fraud_type_unrecognized_label_is_unclassified():
    assert map_fraud_type("完全沒見過的類別") == "unclassified"


def test_map_fraud_type_none_is_unclassified():
    assert map_fraud_type(None) == "unclassified"


def test_parse_response_extracts_answer_tag():
    raw = "<think>some reasoning</think><answer>{\"is_fraud\": true, \"confidence\": 0.9}</answer>"
    assert _parse_response(raw) == {"is_fraud": True, "confidence": 0.9}


def test_parse_response_falls_back_to_bare_json_in_code_fence():
    """Real captured generations (2026-08-05) don't actually use <answer> tags at all — they
    wrap JSON in a plain ```json code fence. _parse_response must handle this, not just the
    spec's documented <answer> format."""
    raw = '好的，我來分析。\n\n```json\n{\n  "scene": "咨询客服",\n  "confidence": 0.95\n}\n```'
    assert _parse_response(raw) == {"scene": "咨询客服", "confidence": 0.95}


def test_parse_response_raises_on_unparseable_text():
    with pytest.raises(ScamSemanticError):
        _parse_response("這裡完全沒有任何 JSON 內容")
