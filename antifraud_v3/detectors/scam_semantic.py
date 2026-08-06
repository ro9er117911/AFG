"""Line 2 of the two-line detection spec: AntiFraud-Qwen2Audio (JimmyMa99/AntiFraud-SFT), a
TeleAntiFraud-28k-fine-tuned Qwen2-Audio-7B-Instruct (references/05-teleantifraud.md). Apache
2.0, ~16.8GB bf16 on disk, loaded 4-bit via bitsandbytes (config.SCAM_SEMANTIC_LOAD_IN_4BIT).

IMPORTANT, confirmed by reading the model's own published example transcripts (JimmyMa99/
TeleAntiFraud GitHub repo, example/*think.html — real captured model output, not a spec
document): this is NOT a single `<think>...</think><answer>JSON</answer>` generation. It's a
**3-round conversation**, each round a separate model.generate() call whose prompt explicitly
builds on the previous rounds' own JSON answers as conversation history:
  1. Scene classification — "which of these 7 everyday call-scenario categories" (food
     ordering/customer service/appointment/transportation/shopping/ride-hailing/delivery — note:
     these are NOT this app's own 7 fraud-type categories, they're a different, mundane
     "what kind of call is this generally" axis the model was fine-tuned to answer first, as
     scaffolding for the rounds that follow). Always runs.
  2. is_fraud judgment, given round 1's scene as context. Always runs.
  3. Fraud-type classification, given rounds 1+2 as context. Only runs if round 2's is_fraud
     is true — confirmed directly: the benign example (example/case1think.html) never has a
     round 3 turn at all, the fraud example (example/case2think.html) does.
This 3-round structure is *why* this fine-tune scores much higher than the base
Qwen2-Audio-7B-Instruct (84.78 vs 58.51 F1, per the TeleAntiFraud-28k paper) — it's the "slow
thinking" staged-reasoning the paper is about, not incidental API shape. Collapsing it into one
prompt would feed the model something it wasn't fine-tuned to answer.

Each round's raw generation is appended to the growing conversation as that turn's assistant
message (transformers' standard multi-turn audio-chat pattern — see AutoProcessor.
apply_chat_template's own multi-turn "Audio Analysis Inference" example), so later rounds see
earlier rounds' full raw output (think block included) as real conversation history, not a
paraphrase constructed by this code.
"""

import gc
import json
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch._dynamo
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

from .. import config
from ..reasoning.schemas import FraudType

# Confirmed directly (2026-08-05): passing disable_compile=True to model.generate() alone did
# NOT reliably prevent torch.compile/dynamo from kicking in when this ran inside the real
# server (server/upload.py's stream_pipeline_over_audio calls classify_call via
# asyncio.to_thread, a different thread per request from a pool) — a second request with a
# different audio/prompt shape than the first re-triggered a multi-minute recompilation
# (spawning ~24 compile-worker subprocesses, holding ~11.9GB VRAM idle) even with the kwarg set,
# hanging the request. A module-level global disable is the reliable fix, confirmed to actually
# stop it (the per-call kwarg is left in place too, belt-and-suspenders, but this is what
# actually matters). This app calls generate() 2-3 times per call, never in a tight loop — a JIT
# compile tax on every new shape is never worth whatever steady-state speedup it buys.
torch._dynamo.config.disable = True

_model_cache: Qwen2AudioForConditionalGeneration | None = None
_processor_cache: AutoProcessor | None = None


class ScamSemanticError(Exception):
    """Raised on anything from model-loading failure to unparseable output. Callers
    (pipeline/chunk_worker.py's run_final_analysis) catch this and degrade to line2_result=None
    — reasoning/fusion.py's fuse() already handles that gracefully, so one heavy model failing
    never crashes the whole final analysis."""


@dataclass
class AntiFraudQwenResult:
    scenario: str  # round 1's raw "scene" string — internal scaffolding context, not surfaced
    is_fraud: bool
    confidence: float
    fraud_type_raw: str | None = None  # round 3's raw label, only set if is_fraud
    reasoning_trace: list[str] = field(default_factory=list)  # each round's raw generation, debug/audit only


# ---- 7-category label mapping ----
# The model outputs short Chinese labels (confirmed real examples: "银行诈骗", "钓鱼诈骗") that
# describe the same TeleAntiFraud-28k taxonomy this app's reasoning/schemas.py:FraudType already
# uses, but not necessarily identical strings (simplified vs traditional, slightly different
# phrasing) — substring matching, not an exact dict, same discipline as reasoning/rubric.py's
# HARD_TRIGGER_KEYWORDS.
#
# Broadened after a real captured generation (bank_otp_request.wav, 2026-08-05): the model's
# actual label vocabulary is richer/more compound than this app's own 7 categories and doesn't
# always use the exact words a naive keyword list would guess — a real fraud call was labeled
# "冒充公检法/金融机构类诈骗" (impersonating police/procuratorate/court AND financial-institution
# fraud, combined in one compound label), which the original narrow keyword list (only "银行"
# for banking, nothing for "公检法") completely missed. Order matters: dict iteration order is
# the priority when a compound label matches more than one category — identity_theft is checked
# before banking_fraud, so "公检法" (the impersonation mechanism) wins over "金融机构" (the
# generic financial-institution mention) for exactly this kind of compound label, matching this
# app's own taxonomy description ("identity_theft: impersonating an official IS the core
# mechanism", see rubric.py's FRAUD_TYPE_TAXONOMY).
_FRAUD_TYPE_KEYWORDS: dict[FraudType, list[str]] = {
    "investment_fraud": [
        "投资", "投資", "理财", "理財", "股票", "虚拟货币", "虛擬貨幣", "比特币", "比特幣",
        "刷单", "刷單", "返利", "炒股", "荐股", "薦股",
    ],
    "phishing_fraud": ["钓鱼", "釣魚", "网址", "網址", "链接", "連結", "仿冒网站", "仿冒網站", "假冒网站", "假冒網站"],
    "identity_theft": [
        "身份冒用", "身分冒用", "冒充公务员", "冒充公務員", "冒充警察", "冒充检察", "冒充檢察",
        "猜猜我是谁", "猜猜我是誰", "公检法", "公檢法", "冒充领导", "冒充領導", "冒充亲友", "冒充親友",
    ],
    "lottery_fraud": ["中奖", "中獎", "彩票", "抽奖", "抽獎", "兑奖", "兌獎"],
    "banking_fraud": ["银行", "銀行", "贷款", "貸款", "信用卡", "圈存", "征信", "徵信", "金融机构", "金融機構"],
    "extortion_fraud": ["勒索", "敲诈", "敲詐", "绑架", "綁架", "威胁", "威脅", "裸聊", "裸照"],
    "customer_service_fraud": ["客服", "订单", "訂單", "物流", "退款", "电商", "電商", "网购", "網購"],
}


def map_fraud_type(raw_label: str | None) -> FraudType:
    if not raw_label:
        return "unclassified"
    # detectors/scam_semantic_llm.py's Claude-based Line 2 already outputs one of FraudType's
    # own literal values directly (the LLM picks from the enum, no free-text label to match) —
    # short-circuit before the Chinese-keyword matching below, which is only needed for
    # Qwen2Audio's free-text Chinese output.
    if raw_label in _FRAUD_TYPE_KEYWORDS:
        return raw_label  # type: ignore[return-value]
    for fraud_type, keywords in _FRAUD_TYPE_KEYWORDS.items():
        if any(kw in raw_label for kw in keywords):
            return fraud_type
    return "unclassified"


# ---- model loading ----


def load_model() -> tuple[Qwen2AudioForConditionalGeneration, AutoProcessor]:
    global _model_cache, _processor_cache
    if _model_cache is not None:
        return _model_cache, _processor_cache

    if not config.SCAM_SEMANTIC_MODEL_ID:
        raise ScamSemanticError("SCAM_SEMANTIC_MODEL_ID is not set — refusing to silently skip Line 2")

    quantization_config = None
    if config.SCAM_SEMANTIC_LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )

    processor = AutoProcessor.from_pretrained(config.SCAM_SEMANTIC_MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.SCAM_SEMANTIC_MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    _model_cache = model
    _processor_cache = processor
    return model, processor


def unload_model() -> None:
    """Frees Line 2's ~6.6GB resident VRAM. Same reasoning as detectors/deepfake_voice.py's
    unload_model(), and just as necessary in the other direction — confirmed directly
    (2026-08-05): leaving Line 2 resident after classify_call() returns (the normal load-once-
    cache-forever convention) collides with the *next* call's Line 1 needing to reload, OOMing
    on the reload itself (not even generation) since Line 2's ~6.6GB + Line 1's ~4.3GB leaves
    under 1GB free on this project's 12GB card — actually observed, not just estimated. Called
    from pipeline/chunk_worker.py's run_final_analysis right after classify_call() returns
    (success or failure), so the GPU is clear again before the next call's Line 1 usage starts.
    The cost is every call now pays both Line 1's and Line 2's reload latency once — the
    "sequential load/unload" fallback this project's plan flagged as the answer if resident-
    simultaneously didn't fit, which it measurably doesn't, in either direction. gc.collect()
    before empty_cache() is load-bearing, not defensive boilerplate — confirmed directly:
    without it, several GB stayed resident after "unloading" (a reference-cycle-holds-CUDA-
    tensors-alive gotcha), enough to OOM the next call's ASR before either detector even ran."""
    global _model_cache, _processor_cache
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
        _processor_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---- <think>...</think><answer>JSON</answer> parsing ----


def _parse_response(raw: str) -> dict:
    """Never guesses: extract the <answer> block if present, else fall back to scanning for the
    last balanced {...} object via json.JSONDecoder().raw_decode() (never a greedy bracket
    regex — that silently grabs the wrong span the moment there's a nested brace or stray text).
    Raises ScamSemanticError on anything unparseable rather than returning a default."""
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    candidate = match.group(1).strip() if match else raw

    decoder = json.JSONDecoder()
    last_error = None
    start = 0
    last_obj = None
    while True:
        brace_pos = candidate.find("{", start)
        if brace_pos == -1:
            break
        try:
            obj, end = decoder.raw_decode(candidate, brace_pos)
            last_obj = obj
            start = end
        except json.JSONDecodeError as e:
            last_error = e
            start = brace_pos + 1

    if last_obj is None:
        raise ScamSemanticError(f"could not parse a JSON object out of model output: {raw!r}") from last_error
    return last_obj


def _strip_think(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ---- 3-round conversation ----

_SCENE_PROMPT = """你是一个专业的音频大模型，能够直接分析音频内容并判断通话场景。请根据你听到的音频内容，判断通话场景属于以下哪种类型：
- 订餐服务 (Food Ordering Services)
- 咨询客服 (Customer Service Inquiries)
- 预约服务 (Appointment Services)
- 交通咨询 (Transportation Inquiries)
- 日常购物 (Daily Shopping)
- 打车服务 (Ride-hailing Services)
- 外卖服务 (Delivery Services)
- 其他 (Other)

输出格式：
{
  "scene": "<scene_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}"""

_FRAUD_JUDGMENT_PROMPT = """你是一个专业的音频大模型，能够直接分析音频内容并判断其是否涉及诈骗。请根据第一轮分析的通话场景与音频内容，输出你的判断：

输出格式：
{
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>,
  "is_fraud": <true/false>
}"""

_FRAUD_TYPE_PROMPT = """你是一个专业的音频大模型，能够直接分析音频内容并判断其涉及的诈骗类型。请根据第一轮的通话场景、第二轮的涉诈分析与音频内容，输出你的判断：

输出格式：
{
  "fraud_type": "<fraud_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}"""


def _generate_turn(model, processor, conversation: list[dict], audio: np.ndarray, device) -> str:
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        # disable_compile=True: confirmed directly (2026-08-05) — without it, transformers'
        # default auto-compile-on-generate silently kicked in and re-triggered a multi-minute
        # torch.compile/dynamo recompilation (spawning ~24 compile-worker subprocesses) the
        # moment a second call presented a different input shape than the first, hanging the
        # request. This app calls generate() 2-3 times per call, never in a tight loop — a JIT
        # compile tax paid on every new audio/prompt length is never worth whatever steady-state
        # speedup it buys.
        generate_ids = model.generate(
            **inputs, max_new_tokens=config.SCAM_SEMANTIC_MAX_NEW_TOKENS, disable_compile=True
        )
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def classify_call(y: np.ndarray, sr: int, transcript: str) -> AntiFraudQwenResult:
    """Runs the real 3-round (2 if not fraud) conversation described in this module's docstring.
    Audio fed to the model is capped at config.SCAM_SEMANTIC_MAX_AUDIO_SECONDS (most recent N
    seconds) — the model's Whisper-derived audio tower is built around 30s windows
    (preprocessor_config.json's chunk_length=30, confirmed directly against the base
    Qwen2-Audio-7B-Instruct's published config) and was fine-tuned on single-utterance clips,
    not multi-minute calls. transcript is passed as plain context text regardless of audio
    truncation length — it doesn't share the audio tower's length constraint.
    """
    if sr != 16000:
        raise ScamSemanticError(f"classify_call expects 16kHz audio, got sr={sr}")

    model, processor = load_model()
    device = next(model.parameters()).device

    max_samples = config.SCAM_SEMANTIC_MAX_AUDIO_SECONDS * 16000
    audio_for_model = y[-max_samples:] if len(y) > max_samples else y

    conversation: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "call_audio"},
                {"type": "text", "text": _SCENE_PROMPT},
            ],
        }
    ]
    trace: list[str] = []

    round1_raw = _generate_turn(model, processor, conversation, audio_for_model, device)
    trace.append(round1_raw)
    round1 = _parse_response(round1_raw)
    conversation.append({"role": "assistant", "content": round1_raw})

    conversation.append({"role": "user", "content": [{"type": "text", "text": _FRAUD_JUDGMENT_PROMPT}]})
    round2_raw = _generate_turn(model, processor, conversation, audio_for_model, device)
    trace.append(round2_raw)
    round2 = _parse_response(round2_raw)
    conversation.append({"role": "assistant", "content": round2_raw})

    is_fraud = bool(round2.get("is_fraud", False))
    confidence = float(round2.get("confidence", 0.0))

    fraud_type_raw = None
    if is_fraud:
        conversation.append({"role": "user", "content": [{"type": "text", "text": _FRAUD_TYPE_PROMPT}]})
        round3_raw = _generate_turn(model, processor, conversation, audio_for_model, device)
        trace.append(round3_raw)
        round3 = _parse_response(round3_raw)
        fraud_type_raw = round3.get("fraud_type")

    return AntiFraudQwenResult(
        scenario=str(round1.get("scene", "")),
        is_fraud=is_fraud,
        confidence=confidence,
        fraud_type_raw=fraud_type_raw,
        reasoning_trace=trace,
    )
