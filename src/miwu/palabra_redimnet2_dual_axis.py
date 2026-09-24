"""Dual-axis temporal and spectral evidence pooling for ReDimNet2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miwu.palabra_redimnet2_temporal_pyramid import (
    load_palabra_redimnet2_temporal_pyramid,
)


class SpectralEvidencePool(nn.Module):
    """Pool local frequency evidence before the final frequency flattening."""

    def __init__(self, input_channels=224, hidden=128, output_dimension=192):
        super().__init__()
        self.input_projection = nn.Conv2d(
            input_channels, hidden, kernel_size=1, bias=False
        )
        self.input_norm = nn.GroupNorm(16, hidden)
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hidden, hidden, kernel_size=kernel,
                padding=(kernel[0] // 2, kernel[1] // 2),
                groups=hidden, bias=False,
            )
            for kernel in ((1, 1), (3, 1), (5, 1), (1, 3))
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * hidden, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(16, hidden),
            nn.SiLU(),
        )
        self.frequency_attention = nn.Sequential(
            nn.Conv1d(6 * hidden, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 2, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.Linear(8 * hidden, output_dimension),
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
        time_mean = value.mean(dim=3)
        time_std = value.float().var(
            dim=3, unbiased=False
        ).clamp_min(1.0e-5).sqrt().to(value.dtype)
        sequence = torch.cat((time_mean, time_std), dim=1)
        frequency_mean = sequence.mean(dim=2, keepdim=True).expand_as(sequence)
        frequency_std = sequence.float().var(
            dim=2, keepdim=True, unbiased=False
        ).clamp_min(1.0e-5).sqrt().to(sequence.dtype).expand_as(sequence)
        context = torch.cat(
            (sequence, frequency_mean, frequency_std), dim=1
        )
        weights = torch.softmax(
            self.frequency_attention(context).float(), dim=2
        ).to(sequence.dtype)
        statistics = []
        for head in range(weights.shape[1]):
            weight = weights[:, head:head + 1]
            mean = torch.sum(weight * sequence, dim=2)
            variance = torch.sum(
                weight * (sequence - mean.unsqueeze(2)).float().square(),
                dim=2,
            ).clamp_min(1.0e-5)
            statistics.extend((mean, variance.sqrt().to(mean.dtype)))
        residual = self.output(torch.cat(statistics, dim=1))
        return torch.sigmoid(self.residual_logit) * residual


class PalabraReDimNet2DualAxis(nn.Module):
    """One encoder joining temporal and frequency evidence before projection."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, temporal_checkpoint,
        trained_checkpoint=None,
    ):
        super().__init__()
        self.temporal = load_palabra_redimnet2_temporal_pyramid(
            backbone_checkpoint, temporal_checkpoint, device="cpu"
        )
        self.spectral_pool = SpectralEvidencePool(
            self.temporal.encoder.backbone.head.out_channels
        )
        if trained_checkpoint is not None:
            state = torch.load(str(trained_checkpoint), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.load_state_dict(state, strict=True)

    @property
    def evidence_pool(self):
        return self.spectral_pool

    def set_trainable(self, upper=False):
        self.requires_grad_(False)
        self.spectral_pool.requires_grad_(True)
        if upper:
            self.temporal.evidence_pool.requires_grad_(True)
            for name in ("stage4", "stage5", "fin_wght1d", "head"):
                getattr(
                    self.temporal.encoder.backbone, name
                ).requires_grad_(True)
            self.temporal.encoder.pool.requires_grad_(True)
            self.temporal.encoder.bn.requires_grad_(True)
            self.temporal.encoder.linear.requires_grad_(True)
        self.train()
        for module in self.temporal.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def embed_axis_views_mel(self, features):
        encoder = self.temporal.encoder
        if features.ndim == 3:
            features = features.unsqueeze(1)
        frames_2d, _ = encoder.backbone(features)
        frames = frames_2d.reshape(
            frames_2d.shape[0], frames_2d.shape[1] * frames_2d.shape[2],
            frames_2d.shape[3],
        )
        base = encoder.linear(encoder.bn(encoder.pool(frames)))
        temporal_residual = self.temporal.evidence_pool(frames)
        spectral_residual = self.spectral_pool(frames_2d)
        temporal_view = base + temporal_residual
        spectral_view = base + spectral_residual
        return (
            base + temporal_residual + spectral_residual,
            temporal_view,
            spectral_view,
        )

    def embed_mel(self, features):
        return self.embed_axis_views_mel(features)[0]

    def _forward_equal(self, waveforms):
        return self.embed_mel(self.temporal.encoder.spec(waveforms))

    def _axis_views_equal(self, waveforms):
        return self.embed_axis_views_mel(self.temporal.encoder.spec(waveforms))

    def axis_views(self, waveforms, waveform_lengths=None):
        """Return the summed output and its temporal/spectral identity views."""
        if waveform_lengths is not None and not (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        ):
            rows = [
                self._axis_views_equal(
                    waveforms[index:index + 1, :int(size.item())]
                )
                for index, size in enumerate(waveform_lengths)
            ]
            return tuple(
                torch.cat([row[view] for row in rows], dim=0)
                for view in range(3)
            )
        return self._axis_views_equal(waveforms)

    def forward(
        self, features=None, lengths=None, waveforms=None,
        waveform_lengths=None,
    ):
        del lengths
        if waveforms is None:
            waveforms = features
        if waveform_lengths is not None and not (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        ):
            return torch.cat([
                self._forward_equal(
                    waveforms[index:index + 1, :int(size.item())]
                )
                for index, size in enumerate(waveform_lengths)
            ], dim=0)
        return self._forward_equal(waveforms)


def load_palabra_redimnet2_dual_axis(
    backbone_checkpoint, temporal_checkpoint, trained_checkpoint=None,
    device="cuda",
):
    return PalabraReDimNet2DualAxis(
        backbone_checkpoint, temporal_checkpoint, trained_checkpoint
    ).to(device)
