"""Per-call running state: transcript, within-call baseline, and the one final risk verdict.

The reasoning engine (discriminate/reflect/synthesize) now runs once per call, at call end
(pipeline/chunk_worker.py's run_final_analysis), not once per chunk — see that module's
docstring for why. Per-chunk work here is limited to what doesn't need an LLM: recording
transcript/acoustic/emotion signals as they arrive, the within-call acoustic baseline, and a
cheap live keyword-based hard-trigger check (check_live_hard_trigger) that keeps some
real-time protection even though the nuanced risk judgment is deferred to call end.
"""

import time
from dataclasses import dataclass

from ..reasoning.rubric import HARD_TRIGGER_KEYWORDS
from ..reasoning.schemas import SynthesizeResult


@dataclass
class TranscriptEntry:
    timestamp: float
    speaker: str | None
    text: str


@dataclass
class ChunkSignals:
    """Raw per-chunk acoustic/emotion features, kept around only so run_final_analysis can
    aggregate them into a call-level summary (audio/features.py's aggregate_acoustic_features,
    audio/emotion.py's aggregate_emotion) — not fed to any LLM call per chunk."""

    timestamp: float
    acoustic_features: dict
    emotion_probs: dict[str, float]


@dataclass
class RiskPoint:
    timestamp: float
    risk_level: str
    chunk_risk_score: int
    justification: str


@dataclass
class Alert:
    reason: str  # "hard_trigger" | "final_analysis"
    justification: str
    trigger_name: str | None = None
    quote: str | None = None


class CallState:
    """One instance per live call. Not thread-safe beyond what a single asyncio task calling
    it sequentially needs — see server/ws.py for how one connection owns one CallState.
    """

    BASELINE_WINDOW_S = 15.0

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self.transcript: list[TranscriptEntry] = []
        self.chunk_signals: list[ChunkSignals] = []
        self.risk_trajectory: list[RiskPoint] = []
        self.case_memory: str = ""
        self.alert_state: str = "none"  # none -> alert_issued -> resolved (-> re-armable)
        self.baseline: dict | None = None

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def maybe_set_baseline(self, acoustic_features: dict) -> None:
        """Establish a real within-call baseline once ~15s of the call have elapsed — the one
        place this rewrite can legitimately fix the self-referential-baseline issue the
        antifraud_v2 bug-fix pass explicitly declined to touch (there was no earlier segment
        of the same call to compare against in a whole-call batch analysis; here there is).
        """
        if self.baseline is not None or self.elapsed_seconds() < self.BASELINE_WINDOW_S:
            return
        self.baseline = {
            "mean_pitch": acoustic_features["pitch"]["mean_pitch"],
            "mean_volume": acoustic_features["volume"]["mean_volume"],
            "speech_rate_variation": acoustic_features["speech_rate"]["speech_rate_variation"],
        }

    def record_chunk_signals(
        self,
        transcript_text: str,
        speaker: str | None,
        acoustic_features: dict,
        emotion_probs: dict[str, float],
        timestamp: float | None = None,
    ) -> None:
        """Called once per completed VAD chunk — no LLM involved, see module docstring.
        timestamp overrides the default wall-clock elapsed_seconds() reading. Needed for
        batch/upload analysis (server/upload.py): there, "when this happened" should mean
        position within the uploaded audio, not how long ASR/acoustic/emotion processing took.
        """
        ts = timestamp if timestamp is not None else self.elapsed_seconds()
        self.transcript.append(TranscriptEntry(timestamp=ts, speaker=speaker, text=transcript_text))
        self.chunk_signals.append(
            ChunkSignals(timestamp=ts, acoustic_features=acoustic_features, emotion_probs=emotion_probs)
        )

    def check_live_hard_trigger(self, transcript_text: str) -> Alert | None:
        """Cheap, no-LLM hard-trigger check run on every chunk as it arrives (see
        reasoning/rubric.py's HARD_TRIGGER_KEYWORDS docstring for why this exists separately
        from the LLM-based reasoning, which now only runs once at call end). Literal substring
        match — good enough for the handful of severe, unambiguous phrases this list covers;
        not meant to replace the nuanced final analysis, just to not lose live protection
        entirely while that final analysis is deferred to call end.
        """
        for trigger_name, keywords in HARD_TRIGGER_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in transcript_text.lower():
                    if self.alert_state == "alert_issued":
                        return None  # already showing — don't re-fire every subsequent chunk
                    self.alert_state = "alert_issued"
                    return Alert(
                        reason="hard_trigger", justification=f"偵測到關鍵字「{kw}」", trigger_name=trigger_name, quote=kw
                    )
        return None

    def apply_final_result(self, result: SynthesizeResult, timestamp: float | None = None) -> Alert | None:
        """Called exactly once per call, after run_final_analysis's discriminate/reflect/
        synthesize pass over the whole accumulated transcript. Records the single risk
        verdict and fires an alert if it's warranted — no debounce-across-chunks logic is
        needed here (that existed specifically to stop one noisy per-chunk LLM call from
        firing alone; there's only one call now, so it either warrants an alert or it doesn't).
        """
        ts = timestamp if timestamp is not None else self.elapsed_seconds()
        self.risk_trajectory.append(
            RiskPoint(
                timestamp=ts,
                risk_level=result.risk_level,
                chunk_risk_score=result.chunk_risk_score,
                justification=result.justification,
            )
        )
        self.case_memory = result.case_memory_update

        if self.alert_state == "alert_issued":
            return None  # a live keyword hard-trigger already fired for this call

        fired_trigger = next((t for t in result.hard_triggers if t.fired), None)
        if fired_trigger is not None:
            self.alert_state = "alert_issued"
            return Alert(
                reason="hard_trigger",
                justification=result.justification,
                trigger_name=fired_trigger.name,
                quote=fired_trigger.quote,
            )

        if result.risk_level == "high":
            self.alert_state = "alert_issued"
            return Alert(reason="final_analysis", justification=result.justification)

        return None

    def acknowledge_alert(self) -> None:
        if self.alert_state == "alert_issued":
            self.alert_state = "resolved"
