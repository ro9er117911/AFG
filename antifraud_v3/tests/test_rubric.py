"""reasoning/rubric.py's build_synthesize_system_prompt() — the hard-trigger list is a
Settings-screen-editable, per-call parameter (docs/DESIGN.md §9), not a frozen module
constant, so this checks that the prompt actually reflects whatever list it's given rather
than silently falling back to DEFAULT_HARD_TRIGGERS.
"""

from antifraud_v3.reasoning.rubric import DEFAULT_HARD_TRIGGERS, build_synthesize_system_prompt


def test_default_hard_triggers_used_when_none_passed():
    prompt = build_synthesize_system_prompt(None)
    for trigger in DEFAULT_HARD_TRIGGERS:
        assert trigger in prompt


def test_custom_hard_triggers_replace_defaults():
    custom = ["自訂測試關鍵字：要求提供銀行帳戶密碼", "自訂測試關鍵字：假冒法院傳票"]
    prompt = build_synthesize_system_prompt(custom)
    for trigger in custom:
        assert trigger in prompt
    # A user who edited the list on the Settings screen expects their edits to actually take
    # effect, not silently merge with the defaults underneath.
    for default_trigger in DEFAULT_HARD_TRIGGERS:
        assert default_trigger not in prompt


def test_empty_hard_trigger_list_is_respected_not_replaced_with_defaults():
    prompt = build_synthesize_system_prompt([])
    for default_trigger in DEFAULT_HARD_TRIGGERS:
        assert default_trigger not in prompt
    # The prompt structure itself should still be intact even with zero configured triggers.
    assert "立即示警關鍵字清單" in prompt


def test_prompt_tells_claude_not_to_redecide_the_verdict():
    """Two-line fusion architecture (reasoning/fusion.py): risk_level/fraud_type are now decided
    by fuse(), not by this prompt's own instructions — the prompt should say so explicitly,
    rather than still walking through high/medium/low classification rules that no longer apply
    to what this step actually does (hard-trigger checklist + prose only)."""
    prompt = build_synthesize_system_prompt(None)
    assert "不是" in prompt and "風險等級" in prompt
