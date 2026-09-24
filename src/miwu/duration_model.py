"""Duration-aware multi-scale speaker encoder built on ERes2NetV2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miwu.encoder import build_encoder


def _masked_statistics(x, lengths):
    """Mean and sample standard deviation over valid temporal frames."""
    if lengths is None:
        mean = x.mean(dim=-1)
        std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        return torch.cat((mean.flatten(1), std.flatten(1)), dim=1)
    steps = torch.arange(x.shape[-1], device=x.device)[None, :]
    mask = (steps < lengths[:, None]).to(x.dtype)[:, None, None, :]
    count = lengths.to(x.dtype).clamp_min(1)[:, None, None]
    mean = (x * mask).sum(dim=-1) / count
    squared = ((x - mean[..., None]) ** 2 * mask).sum(dim=-1)
    std = torch.sqrt(squared / (count - 1).clamp_min(1) + 1e-8)
    return torch.cat((mean.flatten(1), std.flatten(1)), dim=1)


class TemporalScaleBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.output_norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm(self.depthwise(x)))
        x = self.output_norm(self.pointwise(x))
        return F.silu(x + residual)


class AttentiveScalePool(nn.Module):
    def __init__(self, channels, bottleneck=96):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    def forward(self, x, lengths):
        steps = torch.arange(x.shape[-1], device=x.device)[None, :]
        if lengths is None:
            valid = torch.ones(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
        else:
            valid = steps < lengths[:, None]
        weights = valid.to(x.dtype)[:, None, :]
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1)
        context_mean = (x * weights).sum(dim=-1, keepdim=True) / count
        context_var = ((x - context_mean) ** 2 * weights).sum(dim=-1, keepdim=True) / count
        context_std = torch.sqrt(context_var + 1e-8)
        context = torch.cat(
            (x, context_mean.expand_as(x), context_std.expand_as(x)), dim=1
        )
        logits = self.attention(context).masked_fill(~valid[:, None, :], -1e4)
        alpha = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        mean = (alpha * x).sum(dim=-1)
        var = (alpha * (x - mean[..., None]) ** 2).sum(dim=-1)
        return torch.cat((mean, torch.sqrt(var.clamp_min(1e-8))), dim=1)


class DurationAwareMultiScalePool(nn.Module):
    """Learn scale-specific speaker statistics and gate them by voiced duration."""

    def __init__(self, input_channels=1024, frequency_bins=10, channels=384):
        super().__init__()
        self.spectral_projection = nn.Sequential(
            nn.Conv2d(input_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(),
        )
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(192 * frequency_bins, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.scales = nn.ModuleList(
            [TemporalScaleBlock(channels, dilation) for dilation in (1, 2, 4)]
        )
        self.pools = nn.ModuleList([AttentiveScalePool(channels) for _ in self.scales])
        self.duration_gate = nn.Sequential(
            nn.Linear(2, 24),
            nn.SiLU(),
            nn.Linear(24, len(self.scales)),
        )
        self.output = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.Linear(channels, 192),
        )
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, feature_map, lengths):
        x = self.spectral_projection(feature_map).flatten(1, 2)
        x = self.temporal_projection(x)
        if lengths is None:
            valid_lengths = torch.full(
                (x.shape[0],), x.shape[-1], dtype=torch.long, device=x.device
            )
        else:
            valid_lengths = lengths.clamp(min=1, max=x.shape[-1])
        duration = valid_lengths.to(x.dtype)
        duration_features = torch.stack(
            (torch.log1p(duration) / 5.0, torch.rsqrt(duration)), dim=1
        )
        gates = torch.softmax(self.duration_gate(duration_features).float(), dim=1).to(x.dtype)
        statistics = []
        for block, pool in zip(self.scales, self.pools):
            statistics.append(pool(block(x), valid_lengths))
        stacked = torch.stack(statistics, dim=1)
        fused = (stacked * gates[:, :, None]).sum(dim=1)
        return self.output(fused)


class DurationAwareERes2NetV2(nn.Module):
    """ERes2NetV2 plus a residual, duration-aware multi-scale pooling path."""

    def __init__(self, backbone_checkpoint, device="cuda"):
        super().__init__()
        self.backbone = build_encoder(backbone_checkpoint, device="cpu", architecture="eres2netv2")
        self.adapter = DurationAwareMultiScalePool()
        self.adapter_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def _feature_map(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(backbone.bn1(backbone.conv1(x)))
        out1 = backbone.layer1(out)
        out2 = backbone.layer2(out1)
        out3 = backbone.layer3(out2)
        out4 = backbone.layer4(out3)
        return backbone.fuse34(out4, backbone.layer3_ds(out3))

    @staticmethod
    def _downsample_lengths(lengths):
        if lengths is None:
            return None
        result = lengths
        for _ in range(3):
            result = torch.div(result + 1, 2, rounding_mode="floor")
        return result

    def forward(self, features, lengths=None):
        feature_map = self._feature_map(features)
        output_lengths = self._downsample_lengths(lengths)
        base_stats = _masked_statistics(feature_map, output_lengths)
        base_embedding = self.backbone.seg_1(base_stats)
        adaptation = self.adapter(feature_map, output_lengths)
        return base_embedding + self.adapter_scale * adaptation


def load_duration_model(backbone_checkpoint, model_checkpoint=None, device="cuda"):
    model = DurationAwareERes2NetV2(backbone_checkpoint, device="cpu")
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)
