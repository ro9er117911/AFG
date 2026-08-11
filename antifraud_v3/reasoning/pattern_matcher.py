"""十種詐欺模式的確定性規則表比對——專利 TW I904863 步驟 S312 的實施（Phase 1）。

專利段落【0048】-【0049】：判定成立詐騙行為後（S311，由 discriminate/reflect/synthesize 三步
完成），處理模組再根據語音情緒分析結果、文字情緒分析結果及語意不合理分析結果，將客戶的詐騙
行為標記為「全局詐騙模式」及「詐騙模式#1~#9」其中至少一者（可複選），作為詐騙模式分析結果。

表二~表十一（docs/patent-ten-patterns-extract.md）只給「情緒特徵 + 語意特徵」的組合條件
（邏輯上為 AND），未給量化門檻值——DEFAULT_EMOTION_THRESHOLD 是本專案的實作選擇，不是專利
內容（見 docs/ten-patterns-voice-implementation.md §2 附註「門檻是我們的實作自由」）。

呼叫點：reasoning/__init__.py 的 run_reasoning_pipeline，在 synthesize 判定 risk_level 為
medium/high 之後才呼叫（S312「判定成立後才標記」）——本模組本身是純函式，不做這個 gating。
"""

from dataclasses import dataclass

from .schemas import MatchedPattern, SemanticFeatureFinding, TextEmotionScores

DEFAULT_EMOTION_THRESHOLD = 60


@dataclass(frozen=True)
class PatternDefinition:
    pattern_id: str
    name: str
    required_emotions: tuple[str, ...]  # schemas.TextEmotionScores 的欄位名
    required_features: tuple[str, ...]  # schemas.SemanticFeatureFinding 的欄位名


# 全局模式 + 模式#1~#9，原文照抄自 docs/patent-ten-patterns-extract.md 表二~表十一（一字不差）。
PATTERN_DEFINITIONS: tuple[PatternDefinition, ...] = (
    PatternDefinition(
        pattern_id="global",
        name="全局詐騙模式",
        required_emotions=("anger", "stressful"),
        required_features=("improper_pronoun_use", "lack_of_denial", "subjective_objective_time_mismatch"),
    ),
    PatternDefinition(
        pattern_id="p1",
        name="詐騙模式 #1",
        required_emotions=("stressful", "focus"),
        required_features=("language_change",),
    ),
    PatternDefinition(
        pattern_id="p2",
        name="詐騙模式 #2",
        required_emotions=("extreme", "contradiction"),
        required_features=("language_change", "incoherent_message"),
    ),
    PatternDefinition(
        pattern_id="p3",
        name="詐騙模式 #3",
        required_emotions=("stressful", "contradiction"),
        required_features=("lack_of_denial", "non_sequential_message"),
    ),
    PatternDefinition(
        pattern_id="p4",
        name="詐騙模式 #4",
        required_emotions=("embarrassing",),
        required_features=("improper_pronoun_use", "spontaneous_correction"),
    ),
    PatternDefinition(
        pattern_id="p5",
        name="詐騙模式 #5",
        required_emotions=("stressful", "vigilance"),
        required_features=("unnecessary_connection", "non_sequential_message"),
    ),
    PatternDefinition(
        pattern_id="p6",
        name="詐騙模式 #6",
        required_emotions=("contradiction",),
        required_features=("lack_of_commitment", "unnecessary_connection"),
    ),
    PatternDefinition(
        pattern_id="p7",
        name="詐騙模式 #7",
        required_emotions=("vigilance", "happy"),
        required_features=("language_change",),
    ),
    PatternDefinition(
        pattern_id="p8",
        name="詐騙模式 #8",
        required_emotions=("vigilance", "contradiction"),
        required_features=("improper_pronoun_use", "non_sequential_message"),
    ),
    PatternDefinition(
        pattern_id="p9",
        name="詐騙模式 #9",
        required_emotions=("extreme",),
        required_features=("improper_pronoun_use", "lack_of_denial"),
    ),
)


def match_patterns(
    text_emotions: TextEmotionScores,
    semantic_features: SemanticFeatureFinding,
    threshold: int = DEFAULT_EMOTION_THRESHOLD,
) -> list[MatchedPattern]:
    """兩欄皆全數命中才算符合該模式（AND）：required_emotions 每一項分數都 >= threshold，
    且 required_features 每一項 present 都是 True。可複選——回傳所有符合的模式，不是單選；
    空輸入（全預設值）自然回傳空清單，不會拋例外。
    """
    matched: list[MatchedPattern] = []
    for definition in PATTERN_DEFINITIONS:
        emotions_hit = all(
            getattr(text_emotions, emotion) >= threshold for emotion in definition.required_emotions
        )
        if not emotions_hit:
            continue
        features_hit = all(
            getattr(semantic_features, feature).present for feature in definition.required_features
        )
        if not features_hit:
            continue

        quotes = [
            getattr(semantic_features, feature).quote
            for feature in definition.required_features
            if getattr(semantic_features, feature).quote
        ]
        matched.append(
            MatchedPattern(
                pattern_id=definition.pattern_id,
                name=definition.name,
                emotions=list(definition.required_emotions),
                semantic_features=list(definition.required_features),
                quotes=quotes,
            )
        )
    return matched
