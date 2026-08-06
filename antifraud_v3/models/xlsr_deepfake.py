#########################
##    xlsr_deepfake.py
#########################
# Architecture classes for nii-yamagishilab/xls-r-2b-anti-deepfake — a Wav2Vec2 XLS-R-2B
# frontend + a small fully-connected classification head. Ported near-verbatim from the exact
# usage code published on the model's own Hugging Face card (no separate modeling.py ships in
# the model repo itself — the card's own inference example says to copy this class definition
# into your own code, which is what this file is). Kept in models/ (not detectors/) to match
# TIM.py's precedent: hand-written nn.Module architecture definitions live here; detectors/
# deepfake_voice.py owns loading/caching/inference, mirroring audio/emotion.py's relationship
# to this file's TIM.py sibling.
#
# Requires fairseq==0.12.2 for fairseq.models.wav2vec.Wav2Vec2Model — this is the model's own
# stated dependency, not a choice made here. fairseq's PyPI package ships broken dependency
# metadata (an `omegaconf<2.1` version specifier using an old `.* ` syntax pip>=24.1 refuses to
# parse) that makes `pip install fairseq==0.12.2` fail outright on any modern pip. Confirmed
# fix, matching the model card's own setup instructions: `pip install "pip==24.0"` first, then
# install fairseq normally. This is a one-time environment setup step, not something this
# module can work around at import time.
import torch
import torch.nn as nn
from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model
from huggingface_hub import PyTorchModelHubMixin


class SSLModel(nn.Module):
    """Wraps the XLS-R-2B self-supervised frontend. Config values are the model card's own —
    they must match what the published checkpoint was trained with, not tunable."""

    def __init__(self):
        super().__init__()
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=1024,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=48,
            encoder_embed_dim=1920,
            encoder_ffn_embed_dim=7680,
            encoder_attention_heads=16,
            feature_grad_mult=1.0,
        )
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        # Device is read from the module's own parameters at call time (not stored separately)
        # so a plain `.to(device)` on the outer DeepfakeDetector after construction — the normal
        # load -> .to(device) -> .eval() sequence — is enough to move inference here too.
        device = next(self.parameters()).device
        with torch.no_grad():
            features = self.model(input_data.to(device), mask=False, features_only=True)["x"]
        return features


class DeepfakeDetector(nn.Module, PyTorchModelHubMixin):
    """SSL frontend + adaptive-pool + linear classification head. forward() returns raw
    2-class logits (index 0 = fake, index 1 = real, per the model card's own softmax example)
    — softmax/score extraction happens in detectors/deepfake_voice.py, not here."""

    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1920
        self.num_classes = 2
        self.m_ssl = SSLModel()
        self.adap_pool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = nn.Linear(in_features=self.ssl_orig_output_dim, out_features=self.num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.m_ssl.extract_feat(wav)  # [B, T, D]
        emb = emb.transpose(1, 2)  # [B, D, T]
        pooled_emb = self.adap_pool1d(emb).squeeze(-1)  # [B, D]
        return self.proj_fc(pooled_emb)  # [B, 2]
