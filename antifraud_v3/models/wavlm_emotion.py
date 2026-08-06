#########################
##    wavlm_emotion.py
#########################
# Architecture classes for tiantiaf/wavlm-large-categorical-emotion (Vox-Profile project,
# https://github.com/tiantiaf0627/vox-profile-release) — a WavLM-large frontend with a small
# conv/linear downstream head, fine-tuned on MSP-Podcast for 9-way categorical emotion plus
# continuous arousal/valence/dominance. Ported near-verbatim from the upstream repo's own
# src/model/emotion/wavlm_emotion.py (confirmed via the published config.json that this specific
# checkpoint uses finetune_method="finetune", not "lora" — the LoRA branch below exists only to
# match the upstream class shape for state_dict compatibility; it never executes for this
# checkpoint). Kept in models/ (not detectors/) to match TIM.py/xlsr_deepfake.py's precedent:
# hand-written nn.Module architecture definitions live here; detectors/emotion_wavlm.py owns
# loading/caching/inference.
#
# One dependency deliberately NOT pulled in: the upstream file imports
# `speechbrain.lobes...make_padding_masks` for a single ~10-line utility (turn a wav-length
# fraction into a boolean padding mask) — reimplemented directly below instead of adding all of
# speechbrain as a dependency for that. Confirmed equivalent against speechbrain's own
# implementation (speechbrain/dataio/dataio.py's length_to_mask + huggingface.py's
# make_padding_masks) for the single-sample-batch case this app always calls it with (VAD
# already segments audio into individual chunks — never a padded multi-clip batch).
#
# Requires loralib (`pip install loralib`) — a small, standalone package, just for the
# `lora.Linear` reference inside WavLMEncoderLayerStableLayerNorm's __init__ (unused at runtime
# for this checkpoint, but the class must still define it identically to what the checkpoint's
# state_dict keys expect if a future checkpoint does use finetune_method="lora").
import loralib as lora
import torch
import transformers.models.wavlm.modeling_wavlm as wavlm
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


def _make_padding_masks(src: torch.Tensor, wav_len: torch.Tensor) -> torch.Tensor:
    """src: [B, T] waveform batch. wav_len: [B] fraction (0..1) of each row's real length vs
    T (1.0 for a single, unpadded chunk — the only case this app ever calls this with)."""
    abs_len = torch.round(wav_len * src.shape[1])
    max_len = src.shape[1]
    mask = torch.arange(max_len, device=abs_len.device, dtype=abs_len.dtype).unsqueeze(0) < abs_len.unsqueeze(1)
    return mask.bool()


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    """The "large" WavLM variant's encoder layer (stable/pre-layernorm ordering) — matches
    transformers' own WavLMEncoderLayerStableLayerNorm, plus an optional LoRA injection into the
    upper half of layers' feed-forward projections when finetune_method is "lora"/"combined"."""

    def __init__(self, layer_idx, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if layer_idx > config.num_hidden_layers // 2:
            if self.config.finetune_method in ("lora", "combined"):
                self.feed_forward.intermediate_dense = lora.Linear(
                    config.hidden_size, config.intermediate_size, r=config.lora_rank
                )
                self.feed_forward.output_dense = lora.Linear(
                    config.intermediate_size, config.hidden_size, r=config.lora_rank
                )

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMWrapper(nn.Module, PyTorchModelHubMixin, repo_url="https://github.com/tiantiaf0627/vox-profile-release"):
    """WavLM-large backbone + weighted-layer-sum + small conv stack + five downstream heads
    (categorical emotion, a more granular categorical set, and continuous arousal/valence/
    dominance). forward(..., return_feature=True) returns
    (emotion_logits, pooled_features, detailed_emotion_logits, arousal, valence, dominance) —
    this app only reads the first (categorical emotion) plus arousal/valence/dominance;
    detailed_emotion_logits and pooled_features are unused but kept in the return shape to match
    the upstream checkpoint's actual head structure (all five heads' weights are part of the
    same state_dict this class loads via from_pretrained)."""

    def __init__(
        self,
        pretrain_model="wavlm_large",
        hidden_dim=256,
        finetune_method="finetune",
        lora_rank=16,
        freeze_params=True,
        output_class_num=9,
        use_conv_output=True,
        detailed_class_num=17,
    ):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.finetune_method = finetune_method
        self.freeze_params = freeze_params
        self.use_conv_output = use_conv_output
        self.lora_rank = lora_rank

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.backbone_model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)
        state_dict = self.backbone_model.state_dict()
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = self.finetune_method
        self.model_config.lora_rank = self.lora_rank

        self.backbone_model.encoder.layers = nn.ModuleList(
            [
                WavLMEncoderLayerStableLayerNorm(i, self.model_config, has_relative_position_bias=(i == 0))
                for i in range(self.model_config.num_hidden_layers)
            ]
        )
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        if self.freeze_params and self.finetune_method != "lora":
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = False
        elif self.freeze_params and self.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                p.requires_grad = name in msg.missing_keys
        else:
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = True

        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )

        num_layers = self.model_config.num_hidden_layers + 1 if self.use_conv_output else self.model_config.num_hidden_layers
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers if self.use_conv_output else torch.zeros(num_layers))

        self.emotion_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_class_num))
        self.detailed_out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, detailed_class_num)
        )
        self.arousal_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.valence_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.dominance_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        """x: [1, T] float32 waveform, 16kHz mono, already guaranteed by this app's pipeline
        (audio/vad.py) — see detectors/emotion_wavlm.py for the length caveat (upstream's own
        training data excluded chunks shorter than 3s / longer than 15s)."""
        with torch.no_grad():
            attention_mask = _make_padding_masks(x, wav_len=torch.tensor([1.0], device=x.device))
            signal = torch.stack(
                [self.processor(row.float().cpu().numpy(), sampling_rate=16_000, return_tensors="pt")["input_values"][0].to(x.device) for row in x]
            )

        hidden_states = self.backbone_model(signal, attention_mask=attention_mask, output_hidden_states=True).hidden_states

        stacked_feature = torch.stack(hidden_states, dim=0)
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers + 1, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        features = torch.mean(features, dim=1)  # mean-pool over time — no padding to respect (single full-length chunk)

        predicted = self.emotion_layer(features)
        detailed_predicted = self.detailed_out_layer(features)
        arousal = self.arousal_layer(features)
        valence = self.valence_layer(features)
        dominance = self.dominance_layer(features)
        if return_feature:
            return predicted, features, detailed_predicted, arousal, valence, dominance
        return predicted, detailed_predicted, arousal, valence, dominance
