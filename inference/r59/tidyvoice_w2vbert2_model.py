"""Multilingual full-finetuned W2V-BERT2 Adapter-MFA speaker encoder.

This is a dependency-light inference implementation of the LI-MSV TidyVoice
2026 ``s2`` model.  It intentionally constructs the acoustic encoder from the
published configuration and then loads the full speaker checkpoint, so the
original Facebook base checkpoint is not required at runtime.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim=3200, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        weights = self.attention(x.transpose(1, 2)).transpose(1, 2)
        mean = torch.sum(x * weights, dim=1)
        std = torch.sqrt(
            (torch.sum(x.square() * weights, dim=1) - mean.square())
            .clamp(min=1e-5)
        )
        return torch.cat((mean, std), dim=1)


class ContextMHFAHead(nn.Module):
    """Pool all adapted Conformer levels with independent key/value paths."""

    def __init__(
        self, layers=25, input_dimension=128, heads=8,
        key_dimension=64, value_dimension=96, output_dimension=256,
    ):
        super().__init__()
        self.key_projection = nn.Linear(input_dimension, key_dimension, bias=False)
        self.value_projection = nn.Linear(input_dimension, value_dimension, bias=False)
        self.context_projection = nn.Linear(input_dimension, key_dimension, bias=False)
        self.key_layer_logits = nn.Parameter(torch.zeros(heads, layers))
        self.value_layer_logits = nn.Parameter(torch.zeros(heads, layers))
        self.queries = nn.Parameter(torch.empty(heads, key_dimension))
        self.output = nn.Sequential(
            nn.Linear(2 * heads * value_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.normal_(self.queries, std=key_dimension ** -0.5)
        nn.init.normal_(self.output[0].weight, std=1.0e-3)
        nn.init.zeros_(self.output[0].bias)

    def forward(self, adapted):
        adapted = adapted.to(self.key_projection.weight.dtype)
        key_weights = torch.softmax(
            self.key_layer_logits.float(), dim=1
        ).to(adapted.dtype)
        value_weights = torch.softmax(
            self.value_layer_logits.float(), dim=1
        ).to(adapted.dtype)
        keys = torch.einsum("btlc,hl->bhtc", adapted, key_weights)
        values = torch.einsum("btlc,hl->bhtc", adapted, value_weights)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        context = self.context_projection(adapted.mean(dim=(1, 2)))
        queries = self.queries.unsqueeze(0) + context.unsqueeze(1)
        logits = torch.einsum(
            "bhtd,bhd->bht", torch.tanh(keys), queries
        ) / math.sqrt(keys.shape[-1])
        weights = torch.softmax(logits.float(), dim=2).to(values.dtype)
        mean = torch.sum(weights.unsqueeze(-1) * values, dim=2)
        variance = torch.sum(
            weights.unsqueeze(-1)
            * (values - mean.unsqueeze(2)).float().square(),
            dim=2,
        ).clamp_min(1.0e-5)
        statistics = torch.cat(
            (mean, variance.sqrt().to(mean.dtype)), dim=2
        ).flatten(1)
        residual = self.output(statistics)
        return torch.sigmoid(self.residual_logit) * residual


class TidyVoiceW2VBert2Encoder(nn.Module):
    """Waveform interface for the 24-layer, multi-corpus 256-D SV model."""

    output_dimension = 256

    def __init__(
        self, checkpoint, config_directory, adapted_checkpoint=None,
        context_checkpoint=None,
    ):
        super().__init__()
        from transformers import AutoFeatureExtractor, Wav2Vec2BertConfig
        from transformers import Wav2Vec2BertModel

        config_directory = Path(config_directory)
        self.processor = AutoFeatureExtractor.from_pretrained(
            str(config_directory), local_files_only=True
        )
        with (config_directory / "config_tea.json").open() as handle:
            config = Wav2Vec2BertConfig(**json.load(handle))

        # The released checkpoint is loaded immediately below.  Constructing
        # in FP16 avoids a redundant 2.3 GB FP32 allocation and is exact for
        # the intended CUDA inference path.
        parameter_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(parameter_dtype)
            self.encoder = Wav2Vec2BertModel(config)
            self.adapter_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(True),
                    nn.Linear(128, 128),
                )
                for _ in range(config.num_hidden_layers + 1)
            ])
            self.pooling = AttentiveStatisticsPooling(
                128 * (config.num_hidden_layers + 1), 128
            )
            self.bottleneck = nn.Linear(
                2 * 128 * (config.num_hidden_layers + 1), 256
            )
        finally:
            torch.set_default_dtype(original_dtype)
        if hasattr(self.encoder, "masked_spec_embed"):
            delattr(self.encoder, "masked_spec_embed")

        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint_files = sorted(checkpoint.glob("shard_*.pt"))
            if not checkpoint_files:
                raise RuntimeError("TidyVoice checkpoint shard directory is empty")
        else:
            checkpoint_files = [checkpoint]

        expected = set(self.state_dict())
        loaded = set()
        unexpected = set()
        for checkpoint_file in checkpoint_files:
            try:
                package = torch.load(
                    str(checkpoint_file), map_location="cpu", mmap=True,
                    weights_only=True,
                )
            except TypeError:
                # Torch 2.0 does not yet expose mmap. Small deployment shards
                # keep fallback peak RAM bounded instead of materializing the
                # complete 1.17 GB checkpoint beside the model.
                package = torch.load(str(checkpoint_file), map_location="cpu")
            if "state_dict" in package:
                state = package["state_dict"]
            elif "modules" in package and "spk_model" in package["modules"]:
                state = package["modules"]["spk_model"]
            else:
                raise RuntimeError(
                    "TidyVoice checkpoint lacks state_dict or modules.spk_model"
                )
            converted = {}
            for name, value in state.items():
                # The language-adversarial TidyVoice release adds this training
                # head after the 256-D speaker embedding. It is not part of the
                # per-utterance speaker representation used at inference.
                if name.startswith("lang_head."):
                    continue
                if name.startswith("front.encoder."):
                    name = "encoder." + name[len("front.encoder."):]
                elif name.startswith("front."):
                    # Audio feature extractor owns no trainable tensors.
                    continue
                if name in loaded:
                    raise RuntimeError(
                        "duplicate TidyVoice checkpoint tensor: %s" % name
                    )
                converted[name] = value
            _, shard_unexpected = self.load_state_dict(converted, strict=False)
            loaded.update(converted)
            unexpected.update(shard_unexpected)
            del converted, state, package
        missing = sorted(expected - loaded)
        unexpected.update(loaded - expected)
        if missing or unexpected:
            raise RuntimeError(
                "TidyVoice W2V-BERT2 checkpoint mismatch: missing=%s "
                "unexpected=%s" % (missing, sorted(unexpected))
            )
        if adapted_checkpoint is not None:
            adapted = torch.load(str(adapted_checkpoint), map_location="cpu")
            adapted = adapted.get("state_dict", adapted)
            current = self.state_dict()
            invalid = {}
            for name, value in adapted.items():
                if name not in current:
                    invalid[name] = (tuple(value.shape), None)
                elif current[name].shape != value.shape:
                    invalid[name] = (
                        tuple(value.shape), tuple(current[name].shape)
                    )
            if invalid:
                raise RuntimeError(
                    "adapted W2V-BERT2 tensors mismatch: %s" % invalid
                )
            with torch.no_grad():
                for name, value in adapted.items():
                    current[name].copy_(value)
        self.context_mhfa = None
        if context_checkpoint is not None:
            package = torch.load(str(context_checkpoint), map_location="cpu")
            state = package.get("state_dict", package)
            prefix = "context_mhfa."
            state = {
                name[len(prefix):]: value for name, value in state.items()
                if name.startswith(prefix)
            }
            self.context_mhfa = ContextMHFAHead(
                layers=len(self.adapter_layers)
            )
            self.context_mhfa.load_state_dict(state, strict=True)

    def configure_upper_finetuning(self, upper_layers=6):
        """Train full upper Conformer blocks and the complete MFA SV head."""
        upper_layers = int(upper_layers)
        if not 1 <= upper_layers <= len(self.encoder.encoder.layers):
            raise ValueError("upper_layers is outside the encoder depth")
        self.requires_grad_(False)
        modules = list(self.encoder.encoder.layers[-upper_layers:])
        modules.extend((self.adapter_layers, self.pooling, self.bottleneck))
        self.eval()
        for module in modules:
            module.float().requires_grad_(True).train()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return [
            (name, parameter) for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def _extract_features(self, waveforms):
        values = waveforms.detach().float().cpu().numpy()
        features = self.processor(
            values, sampling_rate=16000, return_tensors="pt",
            padding=False, truncation=False,
            return_attention_mask=False,
        )
        reference = next(self.encoder.parameters())
        return features.input_features.to(
            device=reference.device, dtype=reference.dtype
        )

    def _forward_equal(self, waveforms):
        inputs = self._extract_features(waveforms)
        hidden = self.encoder.feature_projection(inputs)[0]
        hidden_states = [hidden]
        for layer in self.encoder.encoder.layers:
            hidden = layer(hidden)[0]
            hidden_states.append(hidden)
        adapted_layers = [
            adapter(value)
            for adapter, value in zip(self.adapter_layers, hidden_states)
        ]
        adapted = torch.cat(adapted_layers, dim=-1)
        base = self.bottleneck(self.pooling(adapted))
        if self.context_mhfa is None:
            return base
        correction = self.context_mhfa(torch.stack(adapted_layers, dim=2))
        return base.float() + correction.float()

    def forward(
        self, features=None, lengths=None, waveforms=None,
        waveform_lengths=None,
    ):
        if waveforms is None:
            waveforms = features
        if waveform_lengths is not None and not (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        ):
            return torch.cat([
                self._forward_equal(
                    waveforms[index:index + 1, :int(length.item())]
                )
                for index, length in enumerate(waveform_lengths)
            ], dim=0)
        return self._forward_equal(waveforms)


def load_tidyvoice_w2vbert2_encoder(
    checkpoint, config_directory, device="cuda", adapted_checkpoint=None,
    context_checkpoint=None,
):
    return TidyVoiceW2VBert2Encoder(
        checkpoint, config_directory, adapted_checkpoint, context_checkpoint
    ).eval().to(device)
