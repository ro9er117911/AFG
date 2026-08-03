# Design Rationale: Real-Time Rewrite of `antifraud_v2`

> This was the build plan for `antifraud_v3` — it's kept here (not deleted) as the design
> record, since most modules' docstrings cite specific sections of it (`docs/DESIGN.md §N`) to
> explain *why* a decision was made, not just what the code does. Originally `docs/DESIGN.md`
> at the repo root; the implementation described here is now complete — see `README.md` for
> current status, setup, and how to run it.

## 0. TL;DR

Build a new `antifraud_v3/` (clean rewrite, not in-place edits — reasoning in §7). A **FastAPI + WebSocket backend** ingests live call audio, segments it into utterance-sized chunks via VAD, and per chunk runs: faster-whisper ASR, the (already-fixed) parselmouth/librosa acoustic feature extraction, TIMNet emotion inference, and a three-step LLM reasoning pass (discriminate → reflect → synthesize) that replaces both `fraud_detection.py`'s broken threshold-sum scoring and `multimodal_fusion.py`'s fixed-weight averaging with one joint reasoning step over audio + text evidence together. Alerts fire on **debounced, sustained risk across chunks** plus a small set of high-precision **hard triggers** (OTP requests, payment-channel switches), not on any single chunk's score. TIMNet is kept but demoted to soft evidence. This is one integrated system to build directly — not staged as a Streamlit-first stepping stone, since the target is real-time WebSocket delivery from the start.

**LLM integration is provider-agnostic by design**: a small `LLMProvider` interface sits between the reasoning engine and whichever model actually answers it, with a **Claude-backed implementation as the default/current provider** (via the official `anthropic` Python SDK, model `claude-opus-5`, structured JSON outputs, adaptive thinking) replacing the old direct, version-broken OpenAI integration. Swapping providers later (GPT, a different Claude model tier, a self-hosted model) means writing one new class, not touching the reasoning logic.

---

## 1. What the current system actually does (grounding)

Confirmed by reading the code directly:

- **Real entry point is `main.py`**, importing the *root* modules: `audio_processing.py`, `model_handling.py`, `language_processing.py`, `fraud_detection.py`, `multimodal_fusion.py`, `utils.py`, `visualization.py`. **`app.py`, `speechFeature.py`, `utils/audio_processing.py`, `utils/model_utils.py`, `visualization/charts.py` are dead legacy code** — `app.py` even hardcodes `sys.path.append('/Users/caozhiyu/Desktop/antifruad-gpt')`, a path from the original developer's Mac. Not this rewrite's job to clean up, but worth flagging.
- **Pipeline** (`main.py:292–326`): upload/record whole call → save temp file → `analyze_speech_features()` (whole-file librosa + parselmouth + CREPE) → `predict_emotion()` (TIMNet, sliding 4s windows, averaged over the whole call) → `transcribe_audio()` (faster-whisper, whole file) → `detect_fraud_patterns()` (8-pattern rule engine) → `get_structured_fraud_analysis()` (GPT text analysis) → `integrate_audio_text_analysis()` (fixed-weight fusion) → render report. Entirely post-hoc, whole-call batch.
- **`fraud_detection.py`'s bug is structural, not cosmetic.** All 8 `_evaluate_*` methods (e.g. `_evaluate_explicit_denial` line 474, `_evaluate_alert_avoidance` line 653) share this shape: `score`/`max_score` accumulate only over indicators that *fired*, then `normalized_score = score/(max_score+0.1)`, pushed through a sigmoid, capped at 0.95. If exactly one weak, common indicator fires (e.g. `neutral > 0.75` for 警覺避談, line 704), `score == max_score` → `normalized_score ≈ 0.91` before the sigmoid even amplifies further — one common, weak signal (like a calm speaking tone) alone nearly maxes out a pattern regardless of the other 5 indicators. Confirmed root cause of "隨便講一句話就覺得我在詐騙." No amount of magic-number recalibration fixes a formula shaped like this — the mechanism itself has to go, which is exactly what the new reasoning engine (§4) replaces it with.
- **`multimodal_fusion.py`** computes an audio risk score and a text risk score fully independently, then does weighted averaging with several hand-added multipliers (`if audio_score>0.7 and text_score>0.7: *1.2`, etc.) — more unvalidated magic numbers, same disease.
- **Taxonomy mismatch**: `FraudRiskAssessor`'s industry/company-size adjustment logic (`fraud_detection.py:1056–1102`) and `FraudPatternClassifier`'s own docstring ("專為臺灣法人說明會環境設計") reveal the 8-pattern taxonomy was built to detect a company executive lying about financials in an earnings call — not a scammer manipulating a phone-call victim. Explains part of why the rubric misfires on ordinary conversation. The new rubric (§4) is written for phone-scam manipulation tactics instead.
- **No speaker diarization anywhere** — all acoustic/emotion features are computed over the full mixed single-channel audio, so "who's calm, who's anxious" is unrecoverable. Flagged as a known gap (§8), not solved by this rewrite.
- **GPT integration is likely already broken independent of this rewrite**: `language_processing.py` calls `openai.ChatCompletion.create(...)` (pre-1.0 SDK syntax) at lines 238, 575, 590, 606. The installed SDK in this environment is `openai==2.49.0`, which dropped that interface entirely — confirmed via `pip show openai` — so this code path almost certainly raises `AttributeError` right now, not just "at risk of breaking." Also confirmed: `.env` does not currently exist in `antifraud_v2/`, so `OPENAI_API_KEY` is unset regardless. This rewrite replaces the integration outright (§3), so the legacy bug is moot for `antifraud_v3` — noted here only so the user understands the old `antifraud_v2` text-analysis path is not just "at risk," it's already down.
- **Text scoring parsed via regex** (`multimodal_fusion.py:38`) against GPT's freeform text output — silently scores `0.0` on any format drift, with no error surfaced. The new reasoning engine uses structured JSON output instead (§3), which eliminates this failure mode by construction.
- **CREPE pitch tracking runs `model_capacity="full"` with Viterbi decoding over the whole call** (`audio_processing.py:324`) — a heavy, duplicate pitch-tracking path alongside the already-computed parselmouth pitch object used for tremor analysis (`audio_processing.py:424`). Consolidate onto parselmouth's pitch tracker; drop CREPE-full — full+Viterbi repeated per chunk is not viable within a real-time latency budget.

---

## 2. Target architecture

### 2.1 Chunking: VAD-triggered utterances, not fixed windows

- Audio arrives over a WebSocket connection (browser `MediaRecorder`/`AudioWorklet` streams PCM/Opus frames to the backend).
- A lightweight VAD (`silero-vad` — more accurate than `webrtcvad`, still CPU-cheap) runs over the incoming buffer to detect speech/silence transitions.
- **A chunk = one detected utterance**: starts when speech begins, ends after a configurable silence gap (500–700ms), or a hard cap (15s) to bound latency/cost on run-on monologues (a scammer reading a long scripted pitch shouldn't block analysis for a minute).
- Why utterance-sized, not fixed time windows: (a) jitter/shimmer/HNR need multiple voiced pitch periods to be meaningful — a fixed 1–2s slice can cut mid-word; (b) TIMNet's existing sliding-window logic (`model_handling.py:59`, `window_size=4.0, stride=2.0`) maps naturally onto utterance-sized chunks with minimal rework; (c) the reasoning engine cares about *what was just said*, which is inherently a turn-level concept; (d) fewer, larger chunks means fewer LLM calls, directly controlling per-call cost (§8).

### 2.2 Per-chunk pipeline (`antifraud_v3/pipeline/chunk_worker.py`)

For each completed chunk, in order:

1. **ASR** — faster-whisper transcribes just that chunk (`antifraud_v3/asr/transcribe.py`). Not literal word-level streaming ASR (see §5 for why that's out of scope without a paid streaming ASR API) — "transcribe on turn completion," the same pattern most practical whisper-based real-time systems use.
2. **Acoustic features** — the already-fixed parselmouth jitter/shimmer/HNR + librosa speech-rate/pause analysis (`antifraud_v3/audio/features.py`, ported from `audio_processing.py`), run on just this chunk. Pitch tracking consolidated onto parselmouth's `to_pitch()`; CREPE-full dropped (§1).
3. **Emotion** — TIMNet on the chunk (`antifraud_v3/audio/emotion.py`, reusing `model_handling.py:59`'s windowing logic, called once per live chunk instead of swept across a saved file).
4. **(New, scoped small) Speaker attribution** — cheap 2-speaker turn segmentation (same-vs-different-speaker embedding check between consecutive chunks, or separate audio legs if capturing both sides independently), not full diarization. Genuinely new capability, not present in `antifraud_v2` at all; see §8 for the scope call.
5. Push `{transcript_segment, acoustic_features, emotion_probs, speaker_guess, timestamp}` into the call state.

### 2.3 Call-level state (`antifraud_v3/pipeline/call_state.py`)

- Running transcript (ordered chunk segments).
- **A real within-call baseline**: average pitch/speech-rate/volume from the call's first ~10–15s, used for "relative change" indicators. This is the one place the rewrite can legitimately fix the self-referential-baseline issue the earlier bug-fix pass explicitly declined to touch (see `antifraud_v2/FALSE_POSITIVE_REVIEW.md`) — because now there's an earlier segment of the *same* call to compare against, which a whole-call batch analysis never had.
- Running risk trajectory (short history of per-chunk risk assessments, not just the latest).
- Rolling LLM-maintained case-memory summary, updated each turn by the synthesize step (§4) rather than re-derived from scratch every chunk.
- Alert state machine: `none → watching → alert_issued → resolved`, so a fired alert doesn't re-fire every subsequent chunk.

### 2.4 Alert triggering (two independent paths)

- **Sustained/rising risk**: requires the risk trajectory elevated across **≥2 consecutive chunks** (or a short recency-weighted rolling window) — a single noisy chunk (calm tone, one indicator crossing threshold) cannot fire an alert on its own. Direct fix for the exact failure mode the user described.
- **Hard triggers**: a small, explicit, high-precision checklist evaluated against each chunk's transcript — request for a one-time verification code, request to switch payment channel/app, legal-threat-plus-payment-demand, request to not tell family/bank. These bypass the sustained-risk requirement because they're rare and severe enough to justify immediate warning even from one chunk.

---

## 3. LLM provider interface (provider-agnostic, Claude as the current implementation)

### 3.1 Why an interface at all

The reasoning engine (§4) needs one capability from an LLM: given a system prompt, some structured evidence, and a JSON schema, return a validated structured answer. That's it — no tool use, no agentic loop, no multi-turn state beyond what the reasoning functions build explicitly. Isolating that behind a one-method interface means the reasoning code never imports `anthropic` or any other SDK directly, so switching models or providers later is a new file, not a rewrite.

### 3.2 The interface

`antifraud_v3/llm/base.py`:

```python
from abc import ABC, abstractmethod
from typing import TypeVar
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

class LLMProvider(ABC):
    @abstractmethod
    def structured_complete(
        self,
        system: str,
        user_content: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Send a single structured-output request; return a validated instance of `schema`."""
```

Every reasoning step (§4) calls `provider.structured_complete(system=..., user_content=..., schema=DiscriminateResult)` and gets back a validated Pydantic object — never raw JSON, never a provider-specific response type.

### 3.3 Claude implementation — the current default

`antifraud_v3/llm/claude_provider.py`, using the official `anthropic` Python SDK (already the right choice per Anthropic's own guidance: this is a Python project, so the SDK is the default surface, not raw HTTP):

```python
import anthropic
from pydantic import BaseModel

class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-5", effort: str = "medium"):
        self.client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / `ant auth login` profile
        self.model = model
        self.effort = effort

    def structured_complete(self, system: str, user_content: str, schema: type[BaseModel]) -> BaseModel:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_format=schema,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        return response.parsed_output
```

Notes tied to current Anthropic API behavior (verified against the live skill reference, not recalled from memory):

- **Model**: `claude-opus-5` is the correct default per Anthropic's current guidance ("always use claude-opus-5 unless the user explicitly names a different model — never downgrade for cost, that's the user's decision"). Given this pipeline calls the LLM 2–3 times *per chunk* (discriminate/reflect/synthesize) rather than once per call — a real cost multiplier (§8) — the model and `effort` are constructor parameters read from config/env (`LLM_MODEL`, `LLM_EFFORT`), not hardcoded, so the cost/quality tradeoff is an explicit choice the user makes later with real usage data in hand, not a default I quietly picked for them.
- **Structured output**: `client.messages.parse(..., output_format=SomePydanticModel)` is the recommended, SDK-validated way to get guaranteed-parseable JSON back — this is the direct fix for `multimodal_fusion.py`'s current regex-on-freeform-text parsing (§1), which silently zeroes scores on format drift. The Pydantic schema also doubles as the interface contract between the reasoning engine and whichever provider implements it.
- **Adaptive thinking**: left on (`{"type": "adaptive"}`) — Claude Opus 5 thinks by default, and reasoning about "is this evidence actually indicative of fraud, or is there an innocent explanation" (the reflect step, §4) is exactly the kind of judgment call adaptive thinking is for. `effort: "medium"` is a reasonable starting point per chunk (balances quality against the per-chunk cost multiplier) — flagged as a config to sweep against real data in the evaluation harness (§6), not asserted as correct.
- **Auth**: this environment currently has neither an `ANTHROPIC_API_KEY` env var nor an authenticated `ant` CLI profile (checked directly — `ant` isn't installed, `ANTHROPIC_API_KEY` is unset). Setup needs one of: (a) `.env` entry `ANTHROPIC_API_KEY=...` (parallel to the existing `OPENAI_API_KEY` pattern already in `antifraud_v2/main.py`), or (b) installing the `ant` CLI and running `ant auth login`, which the bare `anthropic.Anthropic()` constructor picks up automatically with no env var needed. Either is fine; `.env` is more consistent with how this codebase already handles the OpenAI key.

### 3.4 Provider selection

`antifraud_v3/llm/__init__.py` — a small factory reading `LLM_PROVIDER` (default `"claude"`) from environment, returning the configured `LLMProvider` instance. Adding a second provider later (e.g. GPT, a self-hosted model) means writing one new class implementing `structured_complete()` and adding one branch here — the reasoning engine in §4 never changes.

---

## 4. Reasoning engine: discriminate → reflect → synthesize

Directly modeled on paper 08 (`../references/08-evolving-scam-calls-llm-rules.md`, EMNLP 2025 Findings) because it's explicitly designed for "no labeled dataset, rules evolve" — this project's exact situation — and needs no training run. Each step is one `provider.structured_complete()` call.

1. **Discriminate** (`antifraud_v3/reasoning/discriminate.py`) — feed the chunk's transcript + acoustic feature summary + emotion distribution + running call-state summary, alongside a rubric derived from the *existing* 8-pattern taxonomy's documented evidence base (the inline research citations already in `fraud_detection.py`'s threshold dict — Burgoon, DePaulo, Levitan, etc. — are worth keeping as prompt content; they represent real domain research already collected, even though the arithmetic built on top of them is being discarded). Ask for **holistic** evidence strength per pattern (none/weak/moderate/strong + short justification), not indicator-by-indicator arithmetic.
2. **Reflect** (`antifraud_v3/reasoning/reflect.py`) — a second structured call, given the discriminate step's flagged patterns, explicitly prompted to argue the innocent-explanation counter-case and allowed to downgrade. This is paper 08's specific false-positive-reduction mechanism, and the most direct lever against the user's core complaint — a step structurally *absent* from the current rule engine, which has no mechanism to argue against its own flags at all.
3. **Synthesize** (`antifraud_v3/reasoning/synthesize.py`) — combines the reflected evidence + hard-trigger checks (§2.4) into the chunk-level risk assessment, updates the running call-state risk trajectory, and keeps a short natural-language justification for audit/human review.

**Replaces `multimodal_fusion.py` entirely** — audio and text evidence are fed into the *same* discriminate call together, so the model reasons about them jointly (e.g., "caller sounds unusually calm and rehearsed while making an urgent claim" is a genuinely cross-modal observation a fixed-weight post-hoc average can never produce). This is SAFE-QAQ's philosophy (`../references/07-safe-qaq.md`) — end-to-end reasoning over both modalities together — **without** its RL training loop, which needs labeled data and training infrastructure this project doesn't have. Be explicit: this is "SAFE-QAQ-inspired prompt design," not SAFE-QAQ itself.

**On the taxonomy mismatch (§1)**: the rubric fed into the discriminate step should be written around scam-kill-chain stages (contact/pretext → urgency/authority establishment → isolation ("don't tell anyone") → payment/credential extraction, informed by PreScam's framing, `../references/09-prescam-benchmark.md`) rather than the earnings-call-flavored 8 patterns — this is a prompt/rubric change, cheap to iterate once the reasoning engine exists.

---

## 5. Real-time delivery: FastAPI + WebSocket, built directly (no Streamlit stepping stone)

Since the target is genuinely mid-call alerting, build the WebSocket path from the start rather than prototyping in Streamlit first:

- **Backend** (`antifraud_v3/server/`): FastAPI app with a WebSocket endpoint that accepts a persistent per-call connection, buffers incoming audio frames, runs the VAD/chunking pipeline (§2), and pushes chunk-level risk updates and alerts back over the same socket as they're produced — genuinely mid-call, not gated behind a UI framework's rerun cycle.
- **Frontend** (`antifraud_v3/frontend/`): a minimal single-page client using the browser's `MediaRecorder`/`AudioWorklet` API to stream audio to the WebSocket and render the risk trajectory + alerts as they arrive. Doesn't need to be elaborate — this is a personal tool, not a product — but it needs to actually push audio continuously and receive pushed updates, which is the one thing Streamlit's rerun-based model cannot do natively.
- **Why not Streamlit at all**: `st.audio_input` only returns a completed recording on stop, with no mid-recording callback into the running script. `streamlit-webrtc` can bridge continuous audio in, but pushing an *unsolicited* mid-session alert back out to the browser fights Streamlit's request/response-ish rerun model — workarounds exist (autorefresh polling, `st.rerun()` loops) but they're hacks on a framework not built for this. Given the target behavior is specifically "warn during the call," building directly on WebSocket avoids building the same thing twice.
- **Not recommended for now**: a managed low-latency streaming ASR API (Deepgram, AssemblyAI, OpenAI Realtime) — would cut ASR latency below turn-based faster-whisper chunking, but adds a second paid, per-minute-billed dependency before the rest of the pipeline is validated. A future upgrade if turn-based chunking proves too slow in practice, not a starting assumption.

---

## 6. Evaluation without a labeled dataset

No labeled fraud-call dataset exists and won't soon — design evaluation around that constraint.

1. **Two small, hand-curated test sets** (`antifraud_v3/eval/`, 10–20 clips each is enough to start): *synthetic/scripted scam calls* covering common categories (tech-support scam, bank/government impersonation, urgency + payment request — the CHI paper's approach of scripted non-victim actors, `../references/10-chi-realtime-scam-warning.md`, is a legitimate precedent for this substituting for real victim data), and *ordinary non-fraudulent calls* specifically covering the conditions known to false-positive today (calm/neutral tone, a naturally slow talker, someone mildly annoyed for unrelated reasons, lots of natural pauses) — this second set exists specifically to measure the false-positive rate the user is complaining about, tracked as a first-class metric, not an afterthought.
2. **Log four things per test clip** on every reasoning/prompt change (`antifraud_v3/eval/regression_log.jsonl`): did an alert fire, at what point in the call, which pattern/hard-trigger it cites, and a 1-line human judgment of "reasonable or not." Cheap, gives a before/after regression check without needing statistical significance from a large dataset.
3. **Log every real call's full reasoning trace in production** (with user consent/opt-in) — the synthesize step's structured output already includes a justification string; keep it around as raw material for future calibration once real usage feedback exists.
4. **Validate the hard-trigger checklist separately and more tightly** than general risk-reasoning — PreScam's benchmark result (supervised approaches beat zero-shot LLM judgment specifically on termination/urgency prediction) suggests LLM judgment alone may be weaker here than intuition suggests. Run the adversarial test set specifically against the hard-trigger phrase list, looking for both misses and false fires.
5. **This is deliberately not a claim of statistical validation** — it proves "we didn't regress on the specific cases we know about, and we have a growing log to eventually calibrate against." Honest, achievable bar without a labeled dataset; still a large step up from the current "no evaluation at all."

---

## 7. Why a clean rewrite, not in-place edits to `antifraud_v2`

The unit of work changes from "whole call" to "chunk," which touches nearly every module's function signature (`analyze_speech_features(audio_file_path)` → something taking an in-memory chunk + running state; `detect_fraud_patterns` → an LLM reasoning call; `multimodal_fusion` → deleted, folded into synthesize). Maintaining `antifraud_v2/main.py`'s batch mode alongside a parallel streaming path in the same files means maintaining two execution models in one codebase — more work for a solo dev, not less. `antifraud_v2/` has been deleted from the working directory (recoverable via the `git` snapshot commit made immediately before deletion, `0c66823`, since this repo has no other history) — the project is now committing fully to `antifraud_v3/` rather than keeping the old batch tool around as a fallback.

---

## 8. Risks and tradeoffs

- **This is a genuinely large project relative to `antifraud_v2`.** Live audio ingestion, VAD-based chunking, running call-state, an alert-debouncing state machine, and a persistent WebSocket backend are all new infrastructure that doesn't exist today — a long-lived server process instead of a stateless Streamlit script per session, connection handling, more moving parts to keep running.
- **Cost**: running discriminate/reflect/synthesize *per chunk* instead of once per call multiplies LLM API cost roughly by the number of chunks per call — a 3-minute call segmented into ~10–15s utterances could easily be 10–20 chunks × 2–3 LLM calls each, i.e. 10–20x today's one-shot analysis cost. The provider config (`LLM_MODEL`, `LLM_EFFORT` in §3.4) is exposed specifically so this can be tuned against real measured cost rather than guessed at; also worth considering batching reflect/synthesize to run less often than every single chunk (e.g. only on meaningful signal changes) rather than assuming every chunk needs the full three-step pipeline.
- **Latency vs. accuracy**: turn-based (VAD-triggered) chunking means the system's reaction time is bounded below by "however long the current utterance is + the silence-gap detection delay" — a scammer talking in one long unbroken paragraph could mean 10–15+ seconds before that turn's analysis even starts. "Real-time" here means "within a turn or two," not sub-second.
- **The debounce logic (§2.4) is itself a new tuning surface** — "require 2+ consecutive elevated chunks" needs the same self-testing (§6) the old magic numbers needed. This plan removes one source of unvalidated magic numbers (the per-indicator rule engine) but introduces a smaller, more legible one (debounce window, hard-trigger phrase list) that still needs eyes on it over time.
- **Diarization/speaker attribution (§2.2) is flagged, not solved** — full-fidelity diarization is its own non-trivial ML problem; this plan scopes it to cheap turn-level attribution deliberately, not as an oversight.
- **SAFE-QAQ-inspired ≠ SAFE-QAQ**, worth repeating: the reasoning engine here is a well-designed prompt pipeline, not a trained, reward-shaped model verified at 70k-calls/day production scale like the paper it's inspired by. Expect meaningfully less accuracy and consistency than that reference point.
- **AI-synthesized/cloned voice detection** (`../references/04-asvspoof5.md`) is explicitly out of scope — a different detection problem (signal authenticity vs. behavior/content) needing its own model and likely its own labeled spoofing corpus.
- **Provider lock-in is deliberately minimized but not zero**: the `LLMProvider` interface (§3) means swapping models/providers is one new class, but prompt behavior (what "medium effort" or a given system prompt actually produces) is still provider-specific — moving providers later will need the evaluation harness (§6) re-run, not just a config flip.

---

## 9. User interface

Solo-dev personal tool, not a product — keep it to **one local web app**: the FastAPI backend (§5) serves both the WebSocket API and a static single-page frontend from the same process. No separate hosting, no app-store distribution, no native mobile app.

### Audio input — the real practical constraint

To protect the user during an actual phone call, audio has to reach the browser mic somehow. Three options, cheapest first:

1. **Speakerphone + laptop mic** — zero setup, works with any phone, but picks up room noise and re-records through open air (lossier than a direct feed). This is the right starting point: it needs no platform-specific code and exercises the entire pipeline end-to-end immediately.
2. **Virtual audio cable** (e.g. BlackHole on macOS, VB-Cable on Windows) routing a softphone or call-recording app's output into the browser as a virtual microphone — cleaner audio, one-time setup per machine.
3. **Browser extension / native VoIP integration** if calls already happen through a computer-based softphone (Zoom Phone, Google Voice) — best audio quality, most engineering.

Build for (1) now; note (2)/(3) as an upgrade path, not something to build up front — don't let audio-capture polish block validating the reasoning engine, which is the part actually being tested for false-positive reduction.

### Screens

1. **Live call view** (the core screen — everything else is secondary): risk trajectory chart (chunk-level risk over time, updating live over the WebSocket), running transcript with the chunk(s) driving the current risk assessment visually flagged, and a prominent alert banner + sound when the debounced/hard-trigger alert fires — showing the synthesize step's short justification text, not just a bare score (a number like "82%" isn't actionable; "對方要求提供銀行驗證碼" is). Manual start/stop-listening control, and a way to dismiss/acknowledge an alert.
2. **Call history / review** — past calls with final risk level, duration, timestamp; click into one to replay its risk trajectory, transcript, and reasoning trace. This reuses the same data the evaluation harness (§6) already logs — one storage path, two consumers (self-testing + UI review), not two separate systems to keep in sync.
3. **Settings** — the LLM provider/model/effort knobs from §3.4 (surfaced, not buried in an env file), and the alert-sensitivity knobs flagged in §8 as needing ongoing tuning (debounce window size, hard-trigger phrase list — editable here, not a code change).

### Alert UX specifically

Directly informed by the CHI paper's (`../references/10-chi-realtime-scam-warning.md`) recall-vs-timeliness finding: an alert should say *why*, not just flash a risk number, or the user has no way to judge whether to trust it. Dismissing an alert maps directly onto the `alert_issued → resolved` transition already in the §2.4 state machine, so acknowledgment isn't a UI-only concept bolted on top — it's wired into the same state the backend already tracks.

This keeps UI scoped proportionately: a real, load-bearing part of the system (chunk update → push → live view; call end → history log; settings change → provider re-init) rather than an afterthought, but still sized for what one person builds and maintains, not a multi-platform product.

### Design direction (for the demo)

A clickable HTML design mockup of all three screens (live monitoring with a calm→alert state toggle, history, settings) has been built and published: https://claude.ai/code/artifact/4b62043b-eaa0-43a9-acd1-f7701f92675f — token system: a "night operations console" concept (calm, low-chrome baseline that escalates visually only when risk actually rises), teal/harbor accent (`#2F8F86` light / `#4FC2B6` dark) kept separate from the semantic risk colors (green/amber/red), system Traditional-Chinese font stack for transcript/body content (deliberate choice — the interface is predominantly Chinese-language, so an embedded Latin webfont would only cover the English micro-labels), and a monospace face reserved for numeric telemetry (call duration, timestamps, risk score) for a console-readout feel. This is the reference to build `antifraud_v3/frontend/` against — implementation should follow its token system rather than restarting the design decisions from scratch.

---

### Critical files for implementation

- `/home/tommy/Project/AFG/antifraud_v2/fraud_detection.py` — source of the rule-engine taxonomy/rubric content to carry forward into the LLM prompt (§4), and the concrete normalization-bug example (§1) to avoid re-creating.
- `/home/tommy/Project/AFG/antifraud_v2/multimodal_fusion.py` — the fixed-weight fusion logic being replaced by the joint discriminate/reflect/synthesize reasoning step.
- `/home/tommy/Project/AFG/antifraud_v2/audio_processing.py` — the acoustic feature extraction (parselmouth/librosa/CREPE) to port from whole-file to per-chunk; where the CREPE/parselmouth pitch-tracker consolidation happens.
- `/home/tommy/Project/AFG/antifraud_v2/model_handling.py` — TIMNet's existing sliding-window inference logic (`predict_emotion`, `window_size`/`stride`), the direct building block for per-chunk emotion inference.
- `/home/tommy/Project/AFG/antifraud_v2/language_processing.py` — the broken OpenAI integration being replaced outright by `antifraud_v3/llm/` (§3); not ported, not fixed in place.
- `/home/tommy/Project/AFG/antifraud_v2/main.py` — orchestration and session-handling patterns (temp-file hashing fix from `FALSE_POSITIVE_REVIEW.md`) to reference when building the new pipeline orchestration, even though the execution model itself changes.
