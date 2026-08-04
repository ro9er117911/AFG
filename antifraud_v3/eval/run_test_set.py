"""Run the pipeline against hand-curated test clips and log results. See docs/DESIGN.md §6.

eval/test_clips/{scam,benign}/*.wav — NOT populated yet. scam/ should hold scripted scam-call
recordings (tech-support, bank/government impersonation, urgency+payment requests — see
../references/10-chi-realtime-scam-warning.md for the CHI paper's scripted-scenario precedent);
benign/ should specifically cover the conditions that false-positived before this rewrite:
calm/neutral tone, a naturally slow talker, mild unrelated annoyance, lots of natural pauses.

This treats each clip as a single chunk for simplicity — real calls go through VADChunker
first (audio/vad.py); for a short scripted test clip that's usually one utterance anyway.

Usage: python -m antifraud_v3.eval.run_test_set
"""

from pathlib import Path

import soundfile as sf

from ..llm import get_llm_provider
from ..pipeline.call_state import CallState
from ..pipeline.chunk_worker import process_chunk_signals, run_final_analysis
from .regression_log import log_result

TEST_CLIPS_DIR = Path(__file__).parent / "test_clips"


def run_clip(path: Path, expected_category: str, provider) -> None:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono

    call_state = CallState()
    signals = process_chunk_signals(y, sr, call_state)
    live_alert = signals.alert if signals is not None else None

    outcome = run_final_analysis(provider, call_state)
    final_alert = outcome[1] if outcome is not None else None

    alert = live_alert or final_alert
    alert_fired = alert is not None
    alert_timestamp = call_state.risk_trajectory[-1].timestamp if call_state.risk_trajectory else None
    cited_pattern = (alert.trigger_name or alert.reason) if alert_fired else None
    log_result(path.name, expected_category, alert_fired, alert_timestamp, cited_pattern)
    print(f"{path.name}: expected={expected_category} alert_fired={alert_fired} cited={cited_pattern}")


def main() -> None:
    provider = get_llm_provider()
    found_any = False
    for category in ("scam", "benign"):
        category_dir = TEST_CLIPS_DIR / category
        if not category_dir.exists():
            continue
        for clip_path in sorted(category_dir.glob("*.wav")):
            found_any = True
            run_clip(clip_path, category, provider)

    if not found_any:
        print(
            f"No test clips found under {TEST_CLIPS_DIR}/{{scam,benign}}/*.wav — "
            "populate them first (see module docstring)."
        )


if __name__ == "__main__":
    main()
