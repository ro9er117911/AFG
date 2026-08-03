"""Per-call running state: transcript, within-call baseline, risk trajectory, and the alert
debounce state machine. See docs/DESIGN.md §2.3/§2.4.
"""

import time
from dataclasses import dataclass, field

from ..reasoning.schemas import SynthesizeResult


@dataclass
class TranscriptEntry:
    timestamp: float
    speaker: str | None
    text: str


@dataclass
class RiskPoint:
    timestamp: float
    risk_level: str
    chunk_risk_score: int
    justification: str


@dataclass
class Alert:
    reason: str  # "hard_trigger" | "sustained_risk"
    justification: str
    trigger_name: str | None = None
    quote: str | None = None


class CallState:
    """One instance per live call. Not thread-safe beyond what a single asyncio task calling
    it sequentially needs — see server/ws.py for how one connection owns one CallState.
    """

    BASELINE_WINDOW_S = 15.0

    def __init__(self, debounce_chunks: int = 2):
        self._started_at = time.monotonic()
        self.transcript: list[TranscriptEntry] = []
        self.risk_trajectory: list[RiskPoint] = []
        self.case_memory: str = ""
        self.alert_state: str = "none"  # none -> watching -> alert_issued -> resolved (-> re-armable)
        self.debounce_chunks = debounce_chunks
        self.baseline: dict | None = None
        self._elevated_streak = 0

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

    def call_state_summary(self) -> str:
        """Fed into ChunkEvidence.call_state_summary for the next chunk's reasoning pass."""
        return self.case_memory

    def record_chunk(
        self,
        transcript_text: str,
        speaker: str | None,
        result: SynthesizeResult,
        timestamp: float | None = None,
    ) -> Alert | None:
        """timestamp overrides the default wall-clock elapsed_seconds() reading. Needed for
        batch/upload analysis (server/upload.py): there, "when this happened" should mean
        position within the uploaded audio, not how long the ASR/LLM pipeline took to process
        it (which is what elapsed_seconds() would otherwise measure, since CallState's clock
        starts at construction, not at audio-position zero)."""
        ts = timestamp if timestamp is not None else self.elapsed_seconds()
        self.transcript.append(TranscriptEntry(timestamp=ts, speaker=speaker, text=transcript_text))
        self.risk_trajectory.append(
            RiskPoint(
                timestamp=ts,
                risk_level=result.risk_level,
                chunk_risk_score=result.chunk_risk_score,
                justification=result.justification,
            )
        )
        self.case_memory = result.case_memory_update
        return self._evaluate_alert(result)

    def _evaluate_alert(self, result: SynthesizeResult) -> Alert | None:
        """Two independent trigger paths — see docs/DESIGN.md §2.4. A fired alert doesn't
        re-fire every subsequent chunk (alert_state gates it); acknowledge_alert() re-arms.
        """
        fired_trigger = next((t for t in result.hard_triggers if t.fired), None)
        if fired_trigger is not None and self.alert_state not in ("alert_issued",):
            self.alert_state = "alert_issued"
            self._elevated_streak = 0
            return Alert(
                reason="hard_trigger",
                justification=result.justification,
                trigger_name=fired_trigger.name,
                quote=fired_trigger.quote,
            )

        if result.risk_level in ("medium", "high"):
            self._elevated_streak += 1
        else:
            self._elevated_streak = 0

        if self._elevated_streak >= self.debounce_chunks:
            # Bug found by tests/test_call_state.py: this used to check
            # `alert_state in ("none", "resolved")`, which is nearly always false right here
            # for any debounce_chunks >= 2 — the elif branch below already moved alert_state
            # to "watching" on an earlier chunk in the same streak (the streak has to pass
            # through >0-but-below-threshold before it can reach the threshold), so by the
            # time the threshold is actually met, "watching" fails that guard and the alert
            # silently never fires. "watching" only means "still accumulating toward a
            # possible alert," not "an alert is currently showing" — only "alert_issued"
            # should block firing here.
            if self.alert_state != "alert_issued":
                self.alert_state = "alert_issued"
                return Alert(reason="sustained_risk", justification=result.justification)
        elif self._elevated_streak > 0 and self.alert_state == "none":
            self.alert_state = "watching"

        return None

    def acknowledge_alert(self) -> None:
        if self.alert_state == "alert_issued":
            self.alert_state = "resolved"
            self._elevated_streak = 0
