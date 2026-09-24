"""Temporal-pyramid ReDimNet2 initialized from the released strong model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miwu.palabra_redimnet2_model import _import_redimnet2


class MultiScaleEvidencePool(nn.Module):
    """Pool short speaker evidence with three receptive fields and two queries."""

    def __init__(self, input_channels, hidden=256, output_dimension=192):
        super().__init__()
        self.input_projection = nn.Conv1d(
            input_channels, hidden, kernel_size=1, bias=False
        )
        self.input_norm = nn.GroupNorm(32, hidden)
        self.branches = nn.ModuleList([
            nn.Conv1d(
                hidden, hidden, kernel_size=kernel, padding=kernel // 2,
                groups=hidden, bias=False,
            )
            for kernel in (1, 3, 5)
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden),
            nn.SiLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(3 * hidden, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.Linear(4 * hidden, output_dimension),
            nn.LayerNorm(output_dimension),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.normal_(self.output[0].weight, std=1.0e-3)
        nn.init.zeros_(self.output[0].bias)

    def forward(self, frames):
        value = F.silu(self.input_norm(self.input_projection(frames)))
        value = self.fusion(torch.cat([
            branch(value) for branch in self.branches
        ], dim=1))
        global_mean = value.mean(dim=2, keepdim=True).expand_as(value)
        global_std = value.float().var(
            dim=2, keepdim=True, unbiased=False
        ).clamp_min(1.0e-5).sqrt().to(value.dtype).expand_as(value)
        context = torch.cat((value, global_mean, global_std), dim=1)
        weights = torch.softmax(self.attention(context).float(), dim=2).to(
            value.dtype
        )
        statistics = []
        for head in range(weights.shape[1]):
            weight = weights[:, head:head + 1]
            mean = torch.sum(weight * value, dim=2)
            variance = torch.sum(
                weight * (value - mean.unsqueeze(2)).float().square(), dim=2
            ).clamp_min(1.0e-5)
            statistics.extend((mean, variance.sqrt().to(mean.dtype)))
        residual = self.output(torch.cat(statistics, dim=1))
        return torch.sigmoid(self.residual_logit) * residual


class PalabraReDimNet2TemporalPyramid(nn.Module):
    """One ReDimNet2 with an integrated multi-scale evidence pooling path."""

    output_dimension = 192

    def __init__(self, backbone_checkpoint, trained_checkpoint=None):
        super().__init__()
        package = torch.load(str(backbone_checkpoint), map_location="cpu")
        if "model_config" not in package or "state_dict" not in package:
            raise RuntimeError("ReDimNet2 checkpoint lacks config or weights")
        config = dict(package["model_config"])
        config["return_all_outputs"] = True
        encoder_type = _import_redimnet2()
        self.encoder = encoder_type(**config)
        self.encoder.load_state_dict(package["state_dict"], strict=True)
        self.evidence_pool = MultiScaleEvidencePool(self.encoder.pool.in_dim)
        if trained_checkpoint is not None:
            state = torch.load(str(trained_checkpoint), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            contains_encoder = any(
                name.startswith("encoder.") for name in state
            )
            missing, unexpected = self.load_state_dict(
                state, strict=contains_encoder
            )
            invalid_missing = (
                list(missing) if contains_encoder else [
                    name for name in missing
                    if not name.startswith("encoder.")
                ]
            )
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "temporal-pyramid checkpoint mismatch: missing=%s unexpected=%s"
                    % (missing, unexpected)
                )

    def set_trainable(self, upper=False):
        self.requires_grad_(False)
        self.evidence_pool.requires_grad_(True)
        if upper:
            for name in ("stage4", "stage5", "fin_wght1d", "head"):
                getattr(self.encoder.backbone, name).requires_grad_(True)
            self.encoder.pool.requires_grad_(True)
            self.encoder.bn.requires_grad_(True)
            self.encoder.linear.requires_grad_(True)
        self.train()
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def _forward_equal(self, waveforms, return_base=False):
        features = self.encoder.spec(waveforms)
        if features.ndim == 3:
            features = features.unsqueeze(1)
        frames, _ = self.encoder.backbone(features)
        if frames.ndim == 4:
            frames = frames.reshape(
                frames.shape[0], frames.shape[1] * frames.shape[2],
                frames.shape[3],
            )
        base = self.encoder.linear(self.encoder.bn(self.encoder.pool(frames)))
        embedding = base + self.evidence_pool(frames)
        return (embedding, base) if return_base else embedding

    def forward(
        self, features=None, lengths=None, waveforms=None,
        waveform_lengths=None, return_base=False,
    ):
        del lengths
        if waveforms is None:
            waveforms = features
        if waveform_lengths is not None and not (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        ):
            parts = [self._forward_equal(
                waveforms[index:index + 1, :int(size.item())], return_base
            ) for index, size in enumerate(waveform_lengths)]
            if return_base:
                return tuple(torch.cat([
                    part[position] for part in parts
                ], dim=0) for position in (0, 1))
            return torch.cat(parts, dim=0)
        return self._forward_equal(waveforms, return_base)


def load_palabra_redimnet2_temporal_pyramid(
    backbone_checkpoint, trained_checkpoint=None, device="cuda",
):
    return PalabraReDimNet2TemporalPyramid(
        backbone_checkpoint, trained_checkpoint
    ).to(device)
