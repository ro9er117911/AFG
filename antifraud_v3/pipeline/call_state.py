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

import numpy as np

from ..reasoning.rubric import HARD_TRIGGER_KEYWORDS
from ..reasoning.schemas import SynthesizeResult


@dataclass
class TranscriptEntry:
    timestamp: float
    speaker: str | None
    text: str


@dataclass
class ChunkSignals:
    """Raw per-chunk acoustic/emotion/deepfake features, kept around only so run_final_analysis
    can aggregate them into a call-level summary (audio/features.py's
    aggregate_acoustic_features, audio/emotion.py's aggregate_emotion,
    detectors/deepfake_voice.py's aggregate_deepfake) — not fed to any LLM call per chunk."""

    timestamp: float
    acoustic_features: dict
    emotion_probs: dict[str, float]
    fake_score: float = 0.0
    # Full 88-dim eGeMAPSv02 functional vector (audio/egemaps.py) — kept alongside
    # acoustic_features (the hand-rolled pitch/jitter/shimmer set) rather than merged into it,
    # since the two are aggregated separately at call end (aggregate_acoustic_features vs.
    # aggregate_egemaps). None for callers/tests that don't pass it.
    egemaps_features: dict[str, float] | None = None


@dataclass
class RiskPoint:
    timestamp: float
    risk_level: str
    chunk_risk_score: int
    justification: str
    # Independent of risk_level/chunk_risk_score — "what kind of scam does this look like"
    # (TeleAntiFraud-28k's 7-category taxonomy, see reasoning/rubric.py's FRAUD_TYPE_TAXONOMY),
    # not a risk judgment. None for calls analyzed before this field existed.
    fraud_type: dict | None = None
    # Two-line fusion fields (reasoning/fusion.py) — defaulted so history rows/tests from before
    # this existed don't break, same pattern as fraud_type above.
    fake_score: float = 0.0
    ai_voice_flag: bool = False
    fusion_source: str = "none"


@dataclass
class Alert:
    reason: str  # "hard_trigger" | "deepfake_voice" | "final_analysis"
    justification: str
    trigger_name: str | None = None
    quote: str | None = None


class CallState:
    """One instance per live call. Not thread-safe beyond what a single asyncio task calling
    it sequentially needs — see server/ws.py for how one connection owns one CallState.
    """

    # Baseline establishes on the *second* completed utterance chunk (i.e. once 1 chunk already
    # exists), not after a fixed span of audio time — see maybe_set_baseline()'s docstring for
    # why the earlier time-based gate (BASELINE_WINDOW_S = 15.0) was replaced.
    MIN_PRIOR_CHUNKS_FOR_BASELINE = 1

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self.transcript: list[TranscriptEntry] = []
        self.chunk_signals: list[ChunkSignals] = []
        self.risk_trajectory: list[RiskPoint] = []
        self.case_memory: str = ""
        self.alert_state: str = "none"  # none -> alert_issued -> resolved (-> re-armable)
        self.baseline: dict | None = None
        # eGeMAPS's own within-call baseline — established on the same triggering chunk as
        # self.baseline (maybe_set_baseline sets both together), kept as a separate attribute
        # rather than nested inside self.baseline since the two feature sets are computed and
        # aggregated by entirely separate code (audio/features.py vs audio/egemaps.py).
        self.egemaps_baseline: dict[str, float] | None = None
        # Set once, immediately after construction, by upload/testdata callers that already know
        # it at decode time (see audio/quality.py) — never set for live mic calls (server/ws.py),
        # which are always 16kHz wideband by construction.
        self.audio_quality: dict | None = None
        # Raw 16kHz mono chunks, retained (unlike acoustic/emotion features, which are already
        # reduced to small dicts) so Line 2 (detectors/scam_semantic.py) can run once on the
        # whole call's actual audio at call end — previously `y` was computed per chunk and
        # discarded once process_chunk_signals() returned, with nothing keeping it around. A
        # multi-minute call is a few tens of MB at most (float32 16kHz mono, VAD already strips
        # inter-utterance silence) — not a real memory concern.
        self.chunk_audio: list[np.ndarray] = []

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def maybe_set_baseline(self, acoustic_features: dict, egemaps_features: dict[str, float] | None = None) -> bool:
        """Establish a real within-call baseline once the call's second utterance chunk has
        arrived — the one place this rewrite can legitimately fix the self-referential-baseline
        issue the antifraud_v2 bug-fix pass explicitly declined to touch (there was no earlier
        segment of the same call to compare against in a whole-call batch analysis; here there
        is).

        Previously gated on 15s of *audio position* having elapsed (not wall-clock — that was
        an earlier, already-fixed bug of its own). That still failed in practice for short
        pre-recorded clips: eval test clips run 17-49s, so a 15s gate left only the last few
        seconds — or nothing at all, confirmed directly for an 18s clip where the gate never
        opened before the clip ended — to show post-baseline movement in the frontend's radar/
        sparkline charts, which made the charts look static even though the pipeline was working
        correctly. Gating on chunk *count* instead of audio duration fixes this uniformly for
        both live and batch calls, since VAD already segments both into discrete utterances
        regardless of overall call/clip length — "wait for the second utterance" doesn't care
        whether that utterance arrives 2s or 20s into the recording.

        Returns True exactly on the one chunk where the baseline was just established, so
        callers (pipeline/chunk_worker.py) know which chunk_signals update to attach it to for
        the frontend, instead of silently recomputing/resending it every chunk thereafter.
        """
        if self.baseline is not None or len(self.chunk_signals) < self.MIN_PRIOR_CHUNKS_FOR_BASELINE:
            return False
        p, v, t, sr = (
            acoustic_features["pitch"],
            acoustic_features["volume"],
            acoustic_features["tremor"],
            acoustic_features["speech_rate"],
        )
        self.baseline = {
            "mean_pitch": p["mean_pitch"],
            "std_pitch": p["std_pitch"],
            "pitch_instability": p["pitch_instability"],
            "mean_volume": v["mean_volume"],
            "std_volume": v["std_volume"],
            "jitter_local": t["jitter_local"],
            "shimmer_local": t["shimmer_local"],
            "hnr": t["hnr"],
            "pause_ratio": sr["pause_ratio"],
            "speech_rate_variation": sr["speech_rate_variation"],
        }
        if egemaps_features is not None:
            self.egemaps_baseline = dict(egemaps_features)
        return True

    def record_chunk_signals(
        self,
        transcript_text: str,
        speaker: str | None,
        acoustic_features: dict,
        emotion_probs: dict[str, float],
        fake_score: float = 0.0,
        raw_audio: np.ndarray | None = None,
        timestamp: float | None = None,
        egemaps_features: dict[str, float] | None = None,
    ) -> None:
        """Called once per completed VAD chunk — no LLM involved, see module docstring.
        timestamp overrides the default wall-clock elapsed_seconds() reading. Needed for
        batch/upload analysis (server/upload.py): there, "when this happened" should mean
        position within the uploaded audio, not how long ASR/acoustic/emotion processing took.

        raw_audio (this chunk's own waveform, appended to self.chunk_audio) is optional so
        existing call sites/tests that only care about transcript/acoustic/emotion don't need
        to fabricate a dummy array — the real per-chunk pipeline (pipeline/chunk_worker.py)
        always passes it.
        """
        ts = timestamp if timestamp is not None else self.elapsed_seconds()
        self.transcript.append(TranscriptEntry(timestamp=ts, speaker=speaker, text=transcript_text))
        self.chunk_signals.append(
            ChunkSignals(
                timestamp=ts,
                acoustic_features=acoustic_features,
                emotion_probs=emotion_probs,
                fake_score=fake_score,
                egemaps_features=egemaps_features,
            )
        )
        if raw_audio is not None:
            self.chunk_audio.append(raw_audio)

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
                fraud_type=result.fraud_type.model_dump(),
                fake_score=result.fake_score,
                ai_voice_flag=result.ai_voice_flag,
                fusion_source=result.fusion_source,
            )
        )
        self.case_memory = result.case_memory_update

        if self.alert_state == "alert_issued":
            return None  # a live keyword hard-trigger already fired for this call

        # Checked first, ahead of hard_triggers/risk_level below: makes the user's own
        # "AI voice wins" fusion priority (reasoning/fusion.py's fuse()) literal and traceable
        # in the alert's own reason/trigger_name — fuse() already sets risk_level="high" on this
        # branch too, so this check is largely about giving the frontend a more specific reason
        # ("this sounds like a cloned voice" vs. a generic content-based verdict), not a
        # different alerting decision.
        if result.ai_voice_flag:
            self.alert_state = "alert_issued"
            return Alert(
                reason="deepfake_voice",
                justification=result.justification,
                trigger_name="ai_voice_clone",
            )

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
