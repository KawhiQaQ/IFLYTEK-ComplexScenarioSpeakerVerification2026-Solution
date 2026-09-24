"""Context-aware multi-layer pooling for the released TidyVoice W2V-BERT2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from miwu.tidyvoice_w2vbert2_model import (
    load_tidyvoice_w2vbert2_encoder,
)


class ContextMHFAHead(nn.Module):
    """Pool the 25 adapted Conformer levels with independent key/value paths."""

    def __init__(
        self, layers=25, input_dimension=128, heads=8,
        key_dimension=64, value_dimension=96, output_dimension=256,
    ):
        super().__init__()
        self.layers = int(layers)
        self.heads = int(heads)
        self.key_projection = nn.Linear(
            input_dimension, key_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            input_dimension, value_dimension, bias=False
        )
        self.context_projection = nn.Linear(
            input_dimension, key_dimension, bias=False
        )
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
        # adapted: batch x time x layer x channel
        # The released encoder is FP16 on CUDA while this small trainable head
        # intentionally keeps FP32 parameters.  Training autocast reconciles
        # them automatically; explicit conversion also supports plain
        # inference_mode evaluation.
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
        return torch.sigmoid(self.residual_logit) * residual, residual


class TidyVoiceW2VBert2ContextMHFA(nn.Module):
    """Frozen published encoder plus a trainable content-aware pooling path."""

    output_dimension = 256

    def __init__(
        self, checkpoint, config_directory, trained_checkpoint=None,
    ):
        super().__init__()
        self.base = load_tidyvoice_w2vbert2_encoder(
            checkpoint, config_directory, device="cpu"
        )
        self.context_mhfa = ContextMHFAHead(
            layers=len(self.base.adapter_layers)
        )
        if trained_checkpoint is not None:
            package = torch.load(str(trained_checkpoint), map_location="cpu")
            state = package.get("state_dict", package)
            missing, unexpected = self.load_state_dict(state, strict=False)
            invalid_missing = [
                name for name in missing if not name.startswith("base.")
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "context-MHFA checkpoint mismatch: missing=%s unexpected=%s"
                    % (invalid_missing, unexpected)
                )

    def set_trainable(self):
        self.requires_grad_(False)
        self.context_mhfa.float().requires_grad_(True).train()
        self.base.eval()
        return self

    def _frozen_parts(self, waveforms):
        with torch.no_grad():
            inputs = self.base._extract_features(waveforms)
            hidden = self.base.encoder.feature_projection(inputs)[0]
            hidden_states = [hidden]
            for layer in self.base.encoder.encoder.layers:
                hidden = layer(hidden)[0]
                hidden_states.append(hidden)
            adapted = torch.stack([
                adapter(value)
                for adapter, value in zip(
                    self.base.adapter_layers, hidden_states
                )
            ], dim=2)
            concatenated = adapted.flatten(2)
            base = self.base.bottleneck(self.base.pooling(concatenated))
        return adapted, base

    def _forward_equal(self, waveforms, return_parts=False):
        adapted, base = self._frozen_parts(waveforms)
        correction, raw_residual = self.context_mhfa(adapted)
        embedding = base.float() + correction.float()
        if return_parts:
            return embedding, base.float(), raw_residual.float()
        return embedding

    def forward(
        self, features=None, lengths=None, waveforms=None,
        waveform_lengths=None, return_parts=False,
    ):
        del lengths
        if waveforms is None:
            waveforms = features
        if waveform_lengths is not None and not (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        ):
            parts = [self._forward_equal(
                waveforms[index:index + 1, :int(size.item())], return_parts
            ) for index, size in enumerate(waveform_lengths)]
            if return_parts:
                return tuple(torch.cat([
                    part[position] for part in parts
                ], dim=0) for position in range(3))
            return torch.cat(parts, dim=0)
        return self._forward_equal(waveforms, return_parts)


def load_tidyvoice_w2vbert2_context_mhfa(
    checkpoint, config_directory, trained_checkpoint=None, device="cuda",
):
    return TidyVoiceW2VBert2ContextMHFA(
        checkpoint, config_directory, trained_checkpoint
    ).to(device)
