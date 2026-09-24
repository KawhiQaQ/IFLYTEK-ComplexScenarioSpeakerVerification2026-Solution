#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition inference for one ERes2Net-family speaker encoder."""

import io
import math
import os
import sys
import wave
import zipfile
from pathlib import Path

# Use the Python 3.8 dependencies bundled by the previously successful R32
# runtime instead of relying on the evaluator's external package index.
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "_vendor"
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "model_lib"))
import kaldi_fbank as kaldi  # noqa: E402
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2  # noqa: E402
from speakerlab.models.eres2net.ERes2Net_huge import (  # noqa: E402
    BasicBlockERes2Net,
    BasicBlockERes2Net_diff_AFF,
    ERes2Net,
)
from speakerlab.models.eres2net.fusion import AFF  # noqa: E402
from speakerlab.models.campplus.DTDNN import CAMPPlus  # noqa: E402
from redimnet2.redimnet2 import ReDimNet2Wrap  # noqa: E402
from tidyvoice_w2vbert2_model import (  # noqa: E402
    load_tidyvoice_w2vbert2_encoder,
)


SCENERY_NAMES = ("data_scenery1", "data_scenery2")


def iter_wavs(root):
    # Some evaluator roots contain protected procfs/device mounts.  Ignore
    # inaccessible branches instead of turning input discovery into a
    # permission failure.
    for directory, _, filenames in os.walk(str(root), onerror=lambda _: None):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                yield Path(directory) / filename


def root_from_wav(path):
    for ancestor in path.parents:
        if ancestor.name not in SCENERY_NAMES:
            continue
        relative = path.relative_to(ancestor)
        if len(relative.parts) >= 2 and relative.parts[0] == "audio":
            return ancestor.parent.resolve()
    return None


def find_input_root():
    candidates = []
    for env_name in ("GAME823_INPUT_DIR", "AIOJ_INPUT_DIR", "INPUT_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    # Keep the official sample paths plus /work/data, which was used by the
    # successful R7 run.  Do not recursively inspect broad system roots.
    candidates.extend(
        Path(value)
        for value in (
            "/work/data/占位符",
            "/work/data/input",
            "/data/input",
            "/input",
            "./data/input",
            "/work/data",
        )
    )
    seen = set()
    searched = []
    for base in candidates:
        key = str(base.resolve()) if base.exists() else str(base)
        if key in seen or not base.is_dir():
            continue
        seen.add(key)
        searched.append(key)
        for wav_path in iter_wavs(base):
            root = root_from_wav(wav_path)
            if root is not None:
                return root
    raise FileNotFoundError(
        "未找到 data_sceneryN/audio/*.wav；已搜索：{}".format(", ".join(searched))
    )


def find_output_dir():
    for env_name in ("AIOJ_OUTPUT_DIR", "OUTPUT_DIR"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).resolve()
    return Path("/work/output")


def canonical_output_path(wav_path, input_root):
    parts = wav_path.relative_to(input_root).parts
    for index, part in enumerate(parts):
        if part in SCENERY_NAMES:
            tail = parts[index:]
            if len(tail) >= 3 and tail[1] == "audio":
                return Path(*tail).with_suffix(".npz")
    raise ValueError("WAV 路径不符合 data_sceneryN/audio/*.wav 目录结构")


class LargeBlock(BasicBlockERes2Net):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=32, scale=2)


class LargeFuseBlock(BasicBlockERes2Net_diff_AFF):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=32, scale=2)


def masked_statistics(x, lengths):
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
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=False,
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


class SpeakerFeaturePyramid(nn.Module):
    def __init__(self, stage_channels=(128, 256, 512, 1024), channels=320):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(value, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64), nn.SiLU(),
            )
            for value in stage_channels
        ])
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64 * 4, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels), nn.SiLU(),
            )
            for _ in stage_channels
        ])
        self.level_gate = nn.Sequential(
            nn.Linear(len(stage_channels) + 2, 32), nn.SiLU(),
            nn.Linear(32, len(stage_channels)),
        )
        self.scales = nn.ModuleList([
            TemporalScaleBlock(channels, dilation) for dilation in (1, 2, 4)
        ])
        self.scale_gate = nn.Sequential(
            nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, len(self.scales))
        )
        self.pools = nn.ModuleList([
            AttentiveScalePool(channels) for _ in self.scales
        ])
        self.output = nn.Sequential(
            nn.Linear(channels * 2, 384), nn.BatchNorm1d(384),
            nn.PReLU(384), nn.Linear(384, 192),
        )

    @staticmethod
    def descriptor(x, lengths):
        steps = torch.arange(x.shape[-1], device=x.device)[None, :]
        mask = (steps < lengths[:, None]).to(x.dtype)[:, None, :]
        return (x.abs() * mask).sum(dim=(1, 2)) / (
            lengths.to(x.dtype).clamp_min(1) * x.shape[1]
        )

    def forward(self, stages, lengths):
        target_steps = stages[0].shape[-1]
        projected = []
        descriptors = []
        for stage, lateral, projection in zip(
            stages, self.lateral, self.projections
        ):
            stage = lateral(stage)
            stage = F.adaptive_avg_pool2d(stage, (4, stage.shape[-1])).flatten(1, 2)
            stage = projection(stage)
            stage = F.interpolate(
                stage, size=target_steps, mode="linear", align_corners=False
            )
            projected.append(stage)
            descriptors.append(self.descriptor(stage, lengths))
        duration = lengths.to(projected[0].dtype)
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        level_features = torch.cat(
            (torch.stack(descriptors, dim=1), duration_features), dim=1
        )
        level_weights = torch.softmax(
            self.level_gate(level_features).float(), dim=1
        ).to(projected[0].dtype)
        fused = sum(
            value * level_weights[:, index, None, None]
            for index, value in enumerate(projected)
        )
        scale_weights = torch.softmax(
            self.scale_gate(duration_features).float(), dim=1
        ).to(fused.dtype)
        statistics = torch.stack([
            pool(block(fused), lengths)
            for block, pool in zip(self.scales, self.pools)
        ], dim=1)
        statistics = (statistics * scale_weights[:, :, None]).sum(dim=1)
        return self.output(statistics)


class QualityGatedPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ERes2NetV2(
            feat_dim=80, embedding_size=192, m_channels=64,
            baseWidth=26, scale=2, expansion=2,
        )
        self.pyramid = SpeakerFeaturePyramid()
        self.adapter_scale = nn.Parameter(torch.tensor(0.10))
        self.quality_gate = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 192)
        )

    @staticmethod
    def final_lengths(lengths):
        result = lengths
        for _ in range(3):
            result = torch.div(result + 1, 2, rounding_mode="floor")
        return result

    @staticmethod
    def quality_features(features, lengths):
        steps = torch.arange(features.shape[1], device=features.device)[None, :]
        valid = steps < lengths[:, None]
        mask = valid.to(features.dtype)[:, :, None]
        count = (lengths * features.shape[2]).to(features.dtype).clamp_min(1)
        rms = torch.sqrt((features.square() * mask).sum(dim=(1, 2)) / count + 1e-6)
        magnitude = (features.abs() * mask).sum(dim=(1, 2)) / count
        differences = (features[:, 1:] - features[:, :-1]).abs()
        valid_pairs = (steps[:, 1:] < lengths[:, None]).to(features.dtype)[:, :, None]
        pair_count = ((lengths - 1).clamp_min(1) * features.shape[2]).to(features.dtype)
        temporal_change = (differences * valid_pairs).sum(dim=(1, 2)) / pair_count
        duration = lengths.to(features.dtype)
        return torch.stack((
            torch.log1p(duration) / 6.0,
            torch.rsqrt(duration.clamp_min(1)) * 4.0,
            rms / 5.0,
            magnitude / 5.0,
            temporal_change / 5.0,
        ), dim=1)

    def _stages(self, features):
        backbone = self.backbone
        value = features.permute(0, 2, 1).unsqueeze(1)
        value = torch.relu(backbone.bn1(backbone.conv1(value)))
        out1 = backbone.layer1(value)
        out2 = backbone.layer2(out1)
        out3 = backbone.layer3(out2)
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final

    def forward(self, features, lengths):
        stages = self._stages(features)
        base = self.backbone.seg_1(masked_statistics(
            stages[-1], self.final_lengths(lengths)
        ))
        adaptation = self.pyramid(stages, lengths)
        delta = self.quality_gate(self.quality_features(features, lengths))
        multiplier = torch.exp(0.5 * torch.tanh(delta))
        return base + self.adapter_scale * multiplier * adaptation


class FeatureMapScale(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.network = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, value):
        mean = value.mean(dim=(-2, -1))
        std = torch.sqrt(
            (value - mean[:, :, None, None]).square().mean(dim=(-2, -1))
            + 1e-6
        )
        scale = self.network(torch.cat((mean, std), dim=1))
        return value * (0.5 + scale[:, :, None, None])


class SourceResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=stride,
                padding=1, groups=input_channels, bias=False,
            ),
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1, bias=False
            ),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1,
                groups=output_channels, bias=False,
            ),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=1, bias=False
            ),
            nn.GroupNorm(8, output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.GroupNorm(8, output_channels),
            )
        )
        self.scale = FeatureMapScale(output_channels)

    def forward(self, value):
        return F.silu(self.scale(self.main(value) + self.skip(value)))


class MultiResolutionSourceEncoder(nn.Module):
    """The V21 raw-waveform source branch, constructed without train-time code."""

    def __init__(self, embedding_dim=192):
        super().__init__()
        self.register_buffer("window", torch.hann_window(640), persistent=True)
        self.register_buffer(
            "short_window", torch.hann_window(320), persistent=True
        )
        self.register_buffer(
            "long_window", torch.hann_window(1280), persistent=True
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                6, 32, kernel_size=(7, 3), stride=(2, 1),
                padding=(3, 1), bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            SourceResidualBlock(32, 64, stride=(2, 2)),
            SourceResidualBlock(64, 96, stride=(2, 2)),
            SourceResidualBlock(96, 128, stride=(2, 1)),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(
                256, 256, kernel_size=7, padding=3,
                groups=256, bias=False,
            ),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(),
            nn.Linear(384, embedding_dim),
        )

    @staticmethod
    def source_quality(source):
        excitation = source[:, 1 if source.shape[1] > 1 else 0]
        lag_positions = torch.linspace(
            20.0 / 240.0, 1.0, excitation.shape[1],
            device=excitation.device,
        )
        lag_weights = torch.softmax(8.0 * excitation.float(), dim=1)
        mean_lag = (lag_weights * lag_positions[None, :, None]).sum(dim=1)
        peak_strength = excitation.float().amax(dim=1)
        return torch.stack(
            (
                peak_strength.mean(dim=1),
                mean_lag.mean(dim=1),
                mean_lag.std(dim=1, unbiased=False),
            ),
            dim=1,
        )

    @staticmethod
    def resolution_view(
        waveforms, frame_size, n_fft, envelope_kernel, window
    ):
        if waveforms.shape[1] < frame_size:
            waveforms = F.pad(
                waveforms, (0, frame_size - waveforms.shape[1])
            )
        frames = waveforms.unfold(1, frame_size, 160)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * window
        spectrum = torch.fft.rfft(frames, n=n_fft, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        raw_correlation = torch.fft.irfft(power, n=n_fft, dim=-1)
        shape = power.shape
        envelope = F.avg_pool1d(
            power.reshape(-1, 1, shape[-1]),
            kernel_size=envelope_kernel, stride=1,
            padding=envelope_kernel // 2,
        ).reshape(shape)
        whitened_power = (power / envelope.clamp_min(1e-6)).clamp_max(20.0)
        excitation_correlation = torch.fft.irfft(
            whitened_power, n=n_fft, dim=-1
        )
        correction = frame_size / (
            frame_size - torch.arange(241, device=waveforms.device)
        )
        outputs = []
        for correlation in (raw_correlation, excitation_correlation):
            correlation = correlation[..., :241]
            correlation = correlation / correlation[..., :1].clamp_min(1e-6)
            correlation = correlation * correction
            outputs.append(
                correlation[..., 20:241:4].permute(0, 2, 1)
            )
        return outputs

    def autocorrelation_features(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        views = []
        for setup in (
            (640, 2048, 31, self.window),
            (320, 1024, 17, self.short_window),
            (1280, 4096, 65, self.long_window),
        ):
            views.extend(self.resolution_view(waveforms, *setup))
        target_steps = views[0].shape[-1]
        views = [
            view if view.shape[-1] == target_steps else F.interpolate(
                view, size=target_steps, mode="linear", align_corners=False
            )
            for view in views
        ]
        return torch.stack(views, dim=1)

    def forward(self, waveforms, return_quality=False):
        source = self.autocorrelation_features(waveforms)
        quality = self.source_quality(source)
        value = self.blocks(self.stem(source))
        value = self.temporal(value.flatten(1, 2))
        weights = torch.softmax(self.attention(value).float(), dim=-1).to(
            value.dtype
        )
        mean = (weights * value).sum(dim=-1)
        variance = (
            weights * (value - mean[..., None]).square()
        ).sum(dim=-1)
        output = self.output(
            torch.cat((mean, torch.sqrt(variance.clamp_min(1e-6))), dim=1)
        )
        if return_quality:
            return output, quality
        return output


class NuisanceInvariantTangent(QualityGatedPyramid):
    """Direct inference-only construction of the complete V21 model."""

    def __init__(self):
        super().__init__()
        self.source_encoder = MultiResolutionSourceEncoder()
        self.source_scale = nn.Parameter(torch.tensor(0.10))
        self.source_gate = nn.Sequential(
            nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 192)
        )
        self.tangent_head = nn.Sequential(
            nn.LayerNorm(192 * 3),
            nn.Linear(192 * 3, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        self.tangent_scale = nn.Parameter(torch.tensor(0.10))
        self.nuisance_expert = nn.Sequential(
            nn.LayerNorm(192 * 3 + 8),
            nn.Linear(192 * 3 + 8, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        self.nuisance_scale = nn.Parameter(torch.tensor(0.10))
        # These auxiliary heads are present in the trained V21 state dict.
        self.device_head = nn.Sequential(
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 8)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 11)
        )

    def periodic_quality(self, waveforms):
        normalized = waveforms.float()
        normalized = normalized - normalized.mean(dim=1, keepdim=True)
        normalized = normalized / (
            normalized.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        views = MultiResolutionSourceEncoder.resolution_view(
            normalized, 640, 2048, 31, self.source_encoder.window
        )
        return self.source_encoder.source_quality(torch.stack(views, dim=1))

    def forward(
        self, features, lengths, waveforms, waveform_lengths=None
    ):
        base = QualityGatedPyramid.forward(self, features, lengths)
        source, periodic_quality = self.source_encoder(
            waveforms, return_quality=True
        )
        quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        source_multiplier = 2.0 * torch.sigmoid(self.source_gate(quality))
        base = base + self.source_scale * source_multiplier * source

        base_direction = F.normalize(base.float(), dim=1).to(base.dtype)
        source_direction = F.normalize(source.float(), dim=1).to(source.dtype)
        interaction = base_direction * source_direction
        candidate = self.tangent_head(
            torch.cat((base_direction, source_direction, interaction), dim=1)
        )
        tangent = candidate - (
            candidate * base_direction
        ).sum(dim=1, keepdim=True) * base_direction
        parent = base + self.tangent_scale * tangent

        periodic_quality = self.periodic_quality(waveforms)
        nuisance_quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        candidate = self.nuisance_expert(torch.cat(
            (
                parent_direction,
                source_direction,
                parent_direction * source_direction,
                nuisance_quality,
            ),
            dim=1,
        ))
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        return parent + self.nuisance_scale * tangent


class DurationConditionedFeatureCompensator(nn.Module):
    """Predict a channel-wise short-to-full correction from observed moments."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.LayerNorm(channels * 2 + 2),
            nn.Linear(channels * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),
        )
        self.strength = nn.Parameter(torch.tensor(0.10))

    @staticmethod
    def channel_statistics(value, lengths):
        steps = torch.arange(value.shape[-1], device=value.device)[None, :]
        valid = steps < lengths[:, None].clamp(max=value.shape[-1])
        mask = valid.float()[:, None, None, :]
        count = (
            valid.sum(dim=1).float() * value.shape[2]
        ).clamp_min(1.0)[:, None]
        working = value.float()
        mean = (working * mask).sum(dim=(2, 3)) / count
        variance = (
            (working - mean[:, :, None, None]).square() * mask
        ).sum(dim=(2, 3)) / count
        return mean, torch.sqrt(variance + 1e-5)

    def forward(self, value, lengths):
        mean, std = self.channel_statistics(value, lengths)
        duration = lengths.float()
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        condition = torch.cat(
            (mean, torch.log(std + 1e-3), duration_features), dim=1
        )
        affine = self.conditioner(
            condition.to(self.conditioner[1].weight.dtype)
        ).float()
        scale, bias = affine.chunk(2, dim=1)
        strength = self.strength.float()
        corrected = value.float() * (
            1.0 + strength * torch.tanh(scale)[:, :, None, None]
        ) + strength * bias[:, :, None, None]
        return corrected.to(value.dtype)


class FrequencyMomentResidual(nn.Module):
    """Recover duration-biased frequency moments after channel compensation."""

    def __init__(self, frequency_bins, hidden):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.LayerNorm(frequency_bins * 2 + 2),
            nn.Linear(frequency_bins * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, frequency_bins * 2),
        )
        self.strength = nn.Parameter(torch.tensor(0.05))

    @staticmethod
    def frequency_statistics(value, lengths):
        steps = torch.arange(value.shape[-1], device=value.device)[None, :]
        valid = steps < lengths[:, None].clamp(max=value.shape[-1])
        mask = valid.float()[:, None, None, :]
        count = (
            valid.sum(dim=1).float() * value.shape[1]
        ).clamp_min(1.0)[:, None]
        working = value.float()
        mean = (working * mask).sum(dim=(1, 3)) / count
        variance = (
            (working - mean[:, None, :, None]).square() * mask
        ).sum(dim=(1, 3)) / count
        return mean, torch.sqrt(variance + 1e-5)

    def forward(self, value, lengths):
        mean, std = self.frequency_statistics(value, lengths)
        duration = lengths.float()
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        condition = torch.cat(
            (mean, torch.log(std + 1e-3), duration_features), dim=1
        )
        affine = self.conditioner(
            condition.to(self.conditioner[1].weight.dtype)
        ).float()
        scale, bias = affine.chunk(2, dim=1)
        strength = self.strength.float()
        corrected = value.float() * (
            1.0 + strength * torch.tanh(scale)[:, None, :, None]
        ) + strength * bias[:, None, :, None]
        return corrected.to(value.dtype)


class ChannelFrequencyFeatureCompensator(nn.Module):
    def __init__(self, channels, channel_hidden, frequency_bins, frequency_hidden):
        super().__init__()
        self.channel_compensator = DurationConditionedFeatureCompensator(
            channels, channel_hidden
        )
        self.frequency_extension = FrequencyMomentResidual(
            frequency_bins, frequency_hidden
        )

    def forward(self, value, lengths):
        value = self.channel_compensator(value, lengths)
        return self.frequency_extension(value, lengths)


class LayerwiseChannelFrequencyCompensation(NuisanceInvariantTangent):
    """Direct inference-only construction of the complete V48 model."""

    def __init__(self):
        super().__init__()
        self.feature_compensators = nn.ModuleList([
            ChannelFrequencyFeatureCompensator(*settings)
            for settings in (
                (128, 64, 80, 64),
                (256, 96, 40, 48),
                (512, 128, 20, 32),
                (1024, 192, 10, 24),
            )
        ])
        self._compensation_lengths = None

    @staticmethod
    def stage_lengths(lengths):
        outputs = [lengths]
        current = lengths
        for _ in range(3):
            current = torch.div(current + 1, 2, rounding_mode="floor")
            outputs.append(current)
        return outputs

    def _stages(self, features):
        if self._compensation_lengths is None:
            raise RuntimeError("compensation lengths were not initialized")
        backbone = self.backbone
        lengths = self.stage_lengths(self._compensation_lengths)
        value = features.permute(0, 2, 1).unsqueeze(1)
        value = torch.relu(backbone.bn1(backbone.conv1(value)))
        out1 = self.feature_compensators[0](
            backbone.layer1(value), lengths[0]
        )
        out2 = self.feature_compensators[1](
            backbone.layer2(out1), lengths[1]
        )
        out3 = self.feature_compensators[2](
            backbone.layer3(out2), lengths[2]
        )
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        final = self.feature_compensators[3](final, lengths[3])
        return out1, out2, out3, final

    def forward(self, features, lengths, waveforms, waveform_lengths=None):
        self._compensation_lengths = lengths
        try:
            return NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths
            )
        finally:
            self._compensation_lengths = None


class V8CampPlusEnsemble(nn.Module):
    """Two complementary public-scale encoders exposed as one embedding."""

    def __init__(self):
        super().__init__()
        self.v8 = QualityGatedPyramid()
        self.campplus = CAMPPlus(
            feat_dim=80,
            embedding_size=192,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
            memory_efficient=True,
        )

    def forward(self, features, lengths):
        # Slice each utterance so padded tails cannot propagate through either
        # convolutional encoder before the final masked statistics.
        if bool(torch.all(lengths == lengths[0])) and int(lengths[0]) == features.shape[1]:
            v8 = F.normalize(self.v8(features, lengths).float(), dim=1)
            campplus = F.normalize(self.campplus(features).float(), dim=1)
            scale = 2.0 ** -0.5
            return torch.cat((scale * v8, scale * campplus), dim=1)
        v8_values = []
        campplus_values = []
        for index, length in enumerate(lengths):
            valid_length = int(length.item())
            utterance = features[index:index + 1, :valid_length]
            utterance_length = lengths.new_tensor([valid_length])
            v8_values.append(self.v8(utterance, utterance_length))
            campplus_values.append(self.campplus(utterance))
        v8 = F.normalize(torch.cat(v8_values, dim=0).float(), dim=1)
        campplus = torch.cat(campplus_values, dim=0)
        campplus = F.normalize(campplus.float(), dim=1)
        scale = 2.0 ** -0.5
        return torch.cat((scale * v8, scale * campplus), dim=1)


class V8CampPlusResNet293Ensemble(nn.Module):
    """V60 plus a complementary deep VoxCeleb residual embedding."""

    def __init__(self):
        super().__init__()
        from wespeaker.models.resnet import ResNet293
        self.base = V8CampPlusEnsemble()
        self.resnet293 = ResNet293(
            feat_dim=80,
            embed_dim=256,
            pooling_func="TSTP",
            two_emb_layer=False,
        )

    def forward(self, features, lengths):
        base = self.base(features, lengths)
        if bool(torch.all(lengths == lengths[0])) and int(lengths[0]) == features.shape[1]:
            deep = self.resnet293(features)[-1]
        else:
            values = []
            for index, length in enumerate(lengths):
                valid_length = int(length.item())
                values.append(self.resnet293(
                    features[index:index + 1, :valid_length]
                )[-1])
            deep = torch.cat(values, dim=0)
        deep = F.normalize(deep.float(), dim=1)
        return torch.cat((math.sqrt(0.90) * base, math.sqrt(0.10) * deep), dim=1)


class V8CampPlusResNet293DeviceEnsemble(nn.Module):
    """V85 plus a small cross-device CAM++ expert."""

    def __init__(self):
        super().__init__()
        from wespeaker.models.resnet import ResNet293
        self.base = V8CampPlusEnsemble()
        self.resnet293 = ResNet293(
            feat_dim=80,
            embed_dim=256,
            pooling_func="TSTP",
            two_emb_layer=False,
        )
        self.campplus3d = CAMPPlus(
            feat_dim=80,
            embedding_size=512,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
            memory_efficient=True,
        )

    def forward(self, features, lengths):
        base = self.base(features, lengths)
        deep_values = []
        device_values = []
        if bool(torch.all(lengths == lengths[0])) and int(lengths[0]) == features.shape[1]:
            deep = self.resnet293(features)[-1]
            device = self.campplus3d(features)
        else:
            for index, length in enumerate(lengths):
                valid_length = int(length.item())
                utterance = features[index:index + 1, :valid_length]
                deep_values.append(self.resnet293(utterance)[-1])
                device_values.append(self.campplus3d(utterance))
            deep = torch.cat(deep_values, dim=0)
            device = torch.cat(device_values, dim=0)
        deep = F.normalize(deep.float(), dim=1)
        device = F.normalize(device.float(), dim=1)
        return torch.cat((
            math.sqrt(0.85) * base,
            math.sqrt(0.10) * deep,
            math.sqrt(0.05) * device,
        ), dim=1)


class V87CenteredGeometryEnsemble(V8CampPlusResNet293DeviceEnsemble):
    """V87 plus centered and low-rank domain-rejected geometry views."""

    def __init__(self):
        super().__init__()
        statistics = torch.load(
            str(SCRIPT_DIR / "model" / "geometry.pt"), map_location="cpu"
        )
        means = statistics["member_means"]
        bases = statistics["domain_bases"]
        if len(means) != 4 or len(bases) != 4:
            raise ValueError("V109 几何统计与四个 V87 成员不匹配")
        for index, (mean, basis) in enumerate(zip(means, bases)):
            self.register_buffer("member_mean_%d" % index, mean.float())
            self.register_buffer("domain_basis_%d" % index, basis.float())

    @staticmethod
    def combine(values):
        weights = (0.425, 0.425, 0.10, 0.05)
        return torch.cat([
            math.sqrt(weight) * value
            for weight, value in zip(weights, values)
        ], dim=1)

    def forward(self, features, lengths):
        weighted = super().forward(features, lengths)
        raw = [
            F.normalize(value.float(), dim=1)
            for value in torch.split(weighted, (192, 192, 256, 512), dim=1)
        ]
        centered = []
        rejected = []
        for index, value in enumerate(raw):
            mean = getattr(self, "member_mean_%d" % index)
            basis = getattr(self, "domain_basis_%d" % index)
            center = F.normalize(value - mean.unsqueeze(0), dim=1)
            centered.append(center)
            if basis.numel():
                center = center - (center @ basis.T) @ basis
            rejected.append(F.normalize(center, dim=1))
        return torch.cat((
            self.combine(raw),
            self.combine(centered),
            self.combine(rejected),
        ), dim=1)


class MultiScaleEvidencePool(nn.Module):
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
            nn.GroupNorm(32, hidden), nn.SiLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(3 * hidden, 128, kernel_size=1), nn.Tanh(),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.Linear(4 * hidden, output_dimension),
            nn.LayerNorm(output_dimension),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

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
        return torch.sigmoid(self.residual_logit) * self.output(
            torch.cat(statistics, dim=1)
        )


class SpectralEvidencePool(nn.Module):
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
            nn.GroupNorm(16, hidden), nn.SiLU(),
        )
        self.frequency_attention = nn.Sequential(
            nn.Conv1d(6 * hidden, hidden, kernel_size=1), nn.Tanh(),
            nn.Conv1d(hidden, 2, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.Linear(8 * hidden, output_dimension),
            nn.LayerNorm(output_dimension),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

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
        weights = torch.softmax(self.frequency_attention(torch.cat((
            sequence, frequency_mean, frequency_std,
        ), dim=1)).float(), dim=2).to(sequence.dtype)
        statistics = []
        for head in range(weights.shape[1]):
            weight = weights[:, head:head + 1]
            mean = torch.sum(weight * sequence, dim=2)
            variance = torch.sum(
                weight * (sequence - mean.unsqueeze(2)).float().square(), dim=2
            ).clamp_min(1.0e-5)
            statistics.extend((mean, variance.sqrt().to(mean.dtype)))
        return torch.sigmoid(self.residual_logit) * self.output(
            torch.cat(statistics, dim=1)
        )


class ChildDualAxisReDimNet2(nn.Module):
    """Child-supervised temporal and spectral paths in one ReDimNet2."""

    def __init__(self, released, checkpoint):
        super().__init__()
        config = dict(released["model_config"])
        config["return_all_outputs"] = True
        self.encoder = ReDimNet2Wrap(**config)
        self.encoder.load_state_dict(released["state_dict"], strict=True)
        self.evidence_pool = MultiScaleEvidencePool(self.encoder.pool.in_dim)
        self.spectral_pool = SpectralEvidencePool(
            self.encoder.backbone.head.out_channels
        )
        package = torch.load(str(checkpoint), map_location="cpu")
        state = package.get("state_dict", package)
        converted = {}
        for name, value in state.items():
            if name.startswith("temporal.encoder."):
                name = "encoder." + name[len("temporal.encoder."):]
            elif name.startswith("temporal.evidence_pool."):
                name = "evidence_pool." + name[len("temporal.evidence_pool."):]
            converted[name] = value
        missing, unexpected = self.load_state_dict(converted, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "儿童 ReDimNet2 权重不匹配：missing={} unexpected={}".format(
                    missing, unexpected
                )
            )

    def forward(self, waveforms):
        features = self.encoder.spec(waveforms)
        if features.ndim == 3:
            features = features.unsqueeze(1)
        frames_2d, _ = self.encoder.backbone(features)
        frames = frames_2d.reshape(
            frames_2d.shape[0], frames_2d.shape[1] * frames_2d.shape[2],
            frames_2d.shape[3],
        )
        base = self.encoder.linear(self.encoder.bn(self.encoder.pool(frames)))
        return (
            base + self.evidence_pool(frames) + self.spectral_pool(frames_2d)
        )


class V109PalabraReDimNet2FlatEnsemble(V87CenteredGeometryEnsemble):
    """R27 with a child-supervised ReDimNet2 route for short utterances."""

    def __init__(self):
        super().__init__()
        released = torch.load(
            str(SCRIPT_DIR / "model" / "redimnet2.ckpt"),
            map_location="cpu",
        )
        if "model_config" not in released or "state_dict" not in released:
            raise RuntimeError("ReDimNet2 检查点缺少结构配置或权重")
        self.redimnet2 = ReDimNet2Wrap(**released["model_config"])
        missing, unexpected = self.redimnet2.load_state_dict(
            released["state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "ReDimNet2 权重不匹配：missing={} unexpected={}".format(
                    missing, unexpected
                )
            )
        self.child_redimnet2 = ChildDualAxisReDimNet2(
            released, SCRIPT_DIR / "model" / "redimnet2_child.ckpt"
        )
        from redimnet2_dense_time import enable_dense_backbone
        enable_dense_backbone(self.child_redimnet2.encoder.backbone)

    def routed_redimnet2(self, waveforms, waveform_lengths):
        # The competition explicitly contains 0.5 s and 1.0 s test copies.
        # The route uses only each utterance's sample count and never sees its
        # pair, filename, scenario, or any evaluation statistic.
        short_limit = 20000
        all_equal = (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        )
        if all_equal:
            encoder = (
                self.child_redimnet2
                if int(waveform_lengths[0]) <= short_limit
                else self.redimnet2
            )
            return encoder(waveforms)
        values = []
        for index, length in enumerate(waveform_lengths):
            size = int(length.item())
            encoder = (
                self.child_redimnet2 if size <= short_limit else self.redimnet2
            )
            values.append(encoder(waveforms[index:index + 1, :size]))
        return torch.cat(values, dim=0)

    def forward(self, features, lengths, waveforms, waveform_lengths):
        geometry = F.normalize(
            super().forward(features, lengths).float(), dim=1
        )
        redimnet2 = self.routed_redimnet2(waveforms, waveform_lengths)
        redimnet2 = F.normalize(redimnet2.float(), dim=1)
        # Both complete schemes have unit norm, so final normalization assigns
        # half of cosine similarity to V109 and half to ReDimNet2.
        return torch.cat((geometry, redimnet2), dim=1)


class V205R33V204ProtectedEnsemble(V109PalabraReDimNet2FlatEnsemble):
    """V292: protected ReDim plus equal geometry/SSL families, shared heads."""

    def __init__(self):
        super().__init__()
        self.w2vbert2 = load_tidyvoice_w2vbert2_encoder(
            SCRIPT_DIR / "model" / "w2vbert2_shards",
            SCRIPT_DIR / "model" / "w2vbert2_config",
            device="cpu",
            context_checkpoint=(
                SCRIPT_DIR / "model" / "w2vbert2_context.ckpt"
            ),
        )

        from shared_w2v_heads import SharedW2VHeads
        self.w2vbert2=SharedW2VHeads(self.w2vbert2,SCRIPT_DIR / "model" / "w2vbert2_grl_head.ckpt",self.redimnet2)
        package=torch.load(str(SCRIPT_DIR / "model" / "geometry_compression.pt"),map_location="cpu")
        if package.get("training_only") is not True:
            raise RuntimeError("Invalid fixed geometry projection provenance")
        self.register_buffer("geometry_projection",package["projection"].float())
        if tuple(self.geometry_projection.shape)!=(1536,1216):
            raise RuntimeError("Invalid fixed geometry projection shape")
        indices=torch.cat([torch.arange(s+640,s+1152) for s in (0,1152,2304)])
        selected=set(indices.tolist())
        self.register_buffer("geometry_projected_indices",indices)
        self.register_buffer("geometry_kept_indices",torch.tensor([i for i in range(3456) if i not in selected]))

    def forward(self, features, lengths, waveforms, waveform_lengths):
        r33 = super().forward(
            features, lengths, waveforms, waveform_lengths
        )
        geometry, redimnet2 = torch.split(r33, (3456, 192), dim=1)
        w2vbert2 = self.w2vbert2(
            waveforms=waveforms, waveform_lengths=waveform_lengths
        )
        original, grl, trained = torch.split(w2vbert2, (256,256,256), dim=1)
        geometry = math.sqrt(0.20) * F.normalize(geometry.float(), dim=1)
        # Float32 projection is identical to the training-only research map.
        with torch.cuda.amp.autocast(enabled=False):
            projected = geometry.float().index_select(1,self.geometry_projected_indices) @ self.geometry_projection.float()
        return torch.cat((
            geometry.index_select(1,self.geometry_kept_indices),
            math.sqrt(0.40) * F.normalize(redimnet2.float(),dim=1),
            projected,
            math.sqrt(0.15) * F.normalize(original.float(),dim=1),
            math.sqrt(0.15) * F.normalize(grl.float(),dim=1),
            math.sqrt(0.10) * F.normalize(trained.float(),dim=1),
        ),dim=1)


class MultiDurationSegmentAggregation(nn.Module):
    """Global, one-second, and half-second identity views for each audio."""

    def __init__(
        self, renormalize_segments=True, reliability_weighted=False
    ):
        super().__init__()
        self.base = V8CampPlusEnsemble()
        self.segment_frames = (98, 48)
        self.max_segments = 4
        self.renormalize_segments = bool(renormalize_segments)
        self.reliability_weighted = bool(reliability_weighted)

    def segment_view(self, features, lengths, global_embedding, wanted):
        values = []
        reliabilities = []
        for row in range(features.shape[0]):
            valid = int(lengths[row])
            if valid <= wanted:
                values.append(global_embedding[row])
                reliabilities.append(global_embedding.new_tensor(1.0))
                continue
            count = min(
                self.max_segments,
                max(2, int(math.ceil(float(valid) / wanted))),
            )
            starts = torch.linspace(
                0, valid - wanted, count, device=features.device
            ).round().long()
            crops = torch.stack([
                features[row, start:start + wanted]
                for start in starts.tolist()
            ])
            if self.renormalize_segments:
                crops = crops - crops.mean(dim=1, keepdim=True)
            crop_lengths = torch.full(
                (count,), wanted, dtype=torch.long, device=features.device
            )
            embeddings = F.normalize(
                self.base(crops, crop_lengths).float(), dim=1
            )
            mean = embeddings.mean(dim=0)
            reliabilities.append(mean.norm().clamp(0.0, 1.0))
            values.append(F.normalize(mean, dim=0))
        return torch.stack(values), torch.stack(reliabilities)

    def forward(self, features, lengths):
        global_embedding = F.normalize(
            self.base(features, lengths).float(), dim=1
        )
        segmented = [
            self.segment_view(features, lengths, global_embedding, wanted)
            for wanted in self.segment_frames
        ]
        views = [global_embedding] + [value for value, _ in segmented]
        if not self.reliability_weighted:
            scale = math.sqrt(1.0 / len(views))
            return torch.cat([scale * value for value in views], dim=1)
        evidence = torch.stack(
            [torch.ones_like(segmented[0][1])]
            + [reliability.square() for _, reliability in segmented],
            dim=1,
        )
        weights = evidence / evidence.sum(dim=1, keepdim=True)
        return torch.cat([
            torch.sqrt(weights[:, index:index + 1]) * value
            for index, value in enumerate(views)
        ], dim=1)


def build_model(architecture):
    if architecture == "eres2netv2":
        return ERes2NetV2(
            feat_dim=80,
            embedding_size=192,
            m_channels=64,
            baseWidth=26,
            scale=2,
            expansion=2,
        )
    if architecture == "eres2net_large":
        model = ERes2Net(
            block=LargeBlock,
            block_fuse=LargeFuseBlock,
            feat_dim=80,
            embedding_size=512,
            m_channels=64,
        )
        model.layer1_downsample = nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False)
        model.layer2_downsample = nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=False)
        model.layer3_downsample = nn.Conv2d(512, 1024, 3, padding=1, stride=2, bias=False)
        model.fuse_mode12 = AFF(channels=256)
        model.fuse_mode123 = AFF(channels=512)
        model.fuse_mode1234 = AFF(channels=1024)
        return model
    if architecture == "quality_pyramid":
        return QualityGatedPyramid()
    if architecture == "nuisance_tangent":
        return NuisanceInvariantTangent()
    if architecture == "channel_frequency_compensation":
        return LayerwiseChannelFrequencyCompensation()
    if architecture == "v8_campplus_ensemble":
        return V8CampPlusEnsemble()
    if architecture == "v8_campplus_resnet293_ensemble":
        return V8CampPlusResNet293Ensemble()
    if architecture == "v8_campplus_resnet293_device_ensemble":
        return V8CampPlusResNet293DeviceEnsemble()
    if architecture == "v87_centered_geometry_ensemble":
        return V87CenteredGeometryEnsemble()
    if architecture == "v109_palabra_redimnet2_flat":
        return V109PalabraReDimNet2FlatEnsemble()
    if architecture == "v205_r33_v204_protected20":
        return V205R33V204ProtectedEnsemble()
    if architecture == "v8_campplus_segment_aggregation":
        return MultiDurationSegmentAggregation()
    if architecture == "v8_campplus_context_segment_aggregation":
        return MultiDurationSegmentAggregation(renormalize_segments=False)
    if architecture == "v8_campplus_reliable_segment_aggregation":
        return MultiDurationSegmentAggregation(
            renormalize_segments=False, reliability_weighted=True
        )
    raise ValueError("未知模型架构：{}".format(architecture))


def load_model(device):
    architecture = (SCRIPT_DIR / "model" / "architecture.txt").read_text().strip()
    model = build_model(architecture)
    state = torch.load(str(SCRIPT_DIR / "model" / "encoder.ckpt"), map_location="cpu")
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    if architecture in (
        "v8_campplus_ensemble", "v8_campplus_segment_aggregation",
        "v8_campplus_context_segment_aggregation",
        "v8_campplus_reliable_segment_aggregation",
        "v8_campplus_resnet293_ensemble",
        "v8_campplus_resnet293_device_ensemble",
        "v87_centered_geometry_ensemble",
        "v109_palabra_redimnet2_flat",
        "v205_r33_v204_protected20",
    ):
        ensemble = (
            model
            if architecture == "v8_campplus_ensemble"
            else model.base
        )
        ensemble.v8.load_state_dict(state, strict=True)
        campplus_state = torch.load(
            str(SCRIPT_DIR / "model" / "campplus.ckpt"), map_location="cpu"
        )
        if "model" in campplus_state and isinstance(campplus_state["model"], dict):
            campplus_state = campplus_state["model"]
        campplus_state = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in campplus_state.items()
        }
        ensemble.campplus.load_state_dict(campplus_state, strict=True)
        if architecture in (
            "v8_campplus_resnet293_ensemble",
            "v8_campplus_resnet293_device_ensemble",
            "v87_centered_geometry_ensemble",
            "v109_palabra_redimnet2_flat",
            "v205_r33_v204_protected20",
        ):
            resnet_state = torch.load(
                str(SCRIPT_DIR / "model" / "resnet293.ckpt"),
                map_location="cpu",
            )
            if "state_dict" in resnet_state and isinstance(resnet_state["state_dict"], dict):
                resnet_state = resnet_state["state_dict"]
            missing, unexpected = model.resnet293.load_state_dict(
                resnet_state, strict=False
            )
            if missing or set(unexpected) - {"projection.weight", "projection.bias"}:
                raise RuntimeError(
                    "ResNet293 权重不匹配：missing={} unexpected={}".format(
                        missing, unexpected
                    )
                )
        if architecture in (
            "v8_campplus_resnet293_device_ensemble",
            "v87_centered_geometry_ensemble",
            "v109_palabra_redimnet2_flat",
            "v205_r33_v204_protected20",
        ):
            campplus3d_state = torch.load(
                str(SCRIPT_DIR / "model" / "campplus3d.ckpt"),
                map_location="cpu",
            )
            if "model" in campplus3d_state and isinstance(campplus3d_state["model"], dict):
                campplus3d_state = campplus3d_state["model"]
            campplus3d_state = {
                (key[len("module.") :] if key.startswith("module.") else key): value
                for key, value in campplus3d_state.items()
            }
            model.campplus3d.load_state_dict(campplus3d_state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return model.to(device).eval(), architecture


def wav_frames(path):
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes()


def load_pcm16(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise ValueError("输入必须是单声道 WAV：{}".format(path))
        if stream.getsampwidth() != 2:
            raise ValueError("输入必须是 16-bit PCM WAV：{}".format(path))
        if stream.getframerate() != 16000:
            raise ValueError("输入必须是 16 kHz WAV：{}".format(path))
        if stream.getcomptype() != "NONE":
            raise ValueError("输入必须是未压缩 PCM WAV：{}".format(path))
        raw = stream.readframes(stream.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return torch.from_numpy(samples / np.float32(32768.0)).unsqueeze(0)


def extract_feature_from_waveform(wav):
    feat = kaldi.fbank(
        wav,
        num_mel_bins=80,
        sample_frequency=16000,
        dither=0.0,
    )
    return feat - feat.mean(dim=0, keepdim=True)


def collate(features):
    lengths = torch.tensor([feature.shape[0] for feature in features], dtype=torch.long)
    length = int(lengths.max())
    padded = torch.stack(
        [F.pad(feature, (0, 0, 0, length - feature.shape[0])) for feature in features]
    )
    return padded, lengths


def dynamic_batches(
    entries, max_items, max_total_samples, require_equal_frames=False
):
    batch = []
    largest = 0
    for entry in entries:
        frames = entry[2]
        next_largest = max(largest, frames)
        if batch and (
            len(batch) >= max_items
            or next_largest * (len(batch) + 1) > max_total_samples
            or (require_equal_frames and frames != largest)
        ):
            yield batch
            batch = []
            largest = 0
        batch.append(entry)
        largest = max(largest, frames)
    if batch:
        yield batch


def main():
    official_root = Path("/work/data/sourcestc-test")
    input_root = (
        official_root.resolve()
        if official_root.is_dir() else find_input_root()
    )
    output_dir = find_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit.zip"
    temporary_zip = output_dir / "submit.zip.tmp"

    wav_entries = []
    seen_outputs = set()
    for wav_path in iter_wavs(input_root):
        relative_output = canonical_output_path(wav_path, input_root)
        output_name = relative_output.as_posix()
        if output_name in seen_outputs:
            raise ValueError("输入中存在重复音频映射：{}".format(output_name))
        seen_outputs.add(output_name)
        wav_entries.append((wav_path, relative_output, wav_frames(wav_path)))
    if not wav_entries:
        raise FileNotFoundError("测试目录中没有 WAV 文件")
    wav_entries.sort(key=lambda entry: (entry[2], str(entry[0])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_items = 24 if device.type == "cuda" else 1
    max_total_samples = 24 * 48000 if device.type == "cuda" else 10**12
    model, architecture = load_model(device)
    if architecture in (
        "v109_palabra_redimnet2_flat", "v205_r33_v204_protected20"
    ):
        limit = 2 if architecture == "v205_r33_v204_protected20" else 8
        max_items = min(max_items, limit)
        max_total_samples = min(max_total_samples, limit * 48000)
    print(
        "device={} architecture={} wavs={}".format(device, architecture, len(wav_entries)),
        flush=True,
    )
    with zipfile.ZipFile(
        str(temporary_zip), "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive, torch.inference_mode():
        completed = 0
        for batch in dynamic_batches(
            wav_entries, max_items, max_total_samples,
            require_equal_frames=architecture in (
                "nuisance_tangent", "channel_frequency_compensation"
            ),
        ):
            waveforms = [load_pcm16(path).squeeze(0) for path, _, _ in batch]
            features, lengths = collate([
                extract_feature_from_waveform(waveform.unsqueeze(0))
                for waveform in waveforms
            ])
            features = features.to(device)
            if architecture in (
                "quality_pyramid", "v8_campplus_ensemble",
                "v8_campplus_segment_aggregation",
                "v8_campplus_context_segment_aggregation",
                "v8_campplus_reliable_segment_aggregation",
                "v8_campplus_resnet293_ensemble",
                "v8_campplus_resnet293_device_ensemble",
                "v87_centered_geometry_ensemble",
            ):
                output = model(features, lengths.to(device))
            elif architecture in (
                "v109_palabra_redimnet2_flat",
                "v205_r33_v204_protected20",
            ):
                waveform_lengths = torch.tensor(
                    [waveform.numel() for waveform in waveforms],
                    dtype=torch.long,
                )
                padded_waveforms = torch.stack([
                    F.pad(
                        waveform,
                        (0, int(waveform_lengths.max()) - waveform.numel()),
                    )
                    for waveform in waveforms
                ])
                output = model(
                    features, lengths.to(device),
                    padded_waveforms.to(device), waveform_lengths.to(device),
                )
            elif architecture in (
                "nuisance_tangent", "channel_frequency_compensation"
            ):
                waveform_lengths = torch.tensor(
                    [waveform.numel() for waveform in waveforms],
                    dtype=torch.long,
                )
                padded_waveforms = torch.stack([
                    F.pad(
                        waveform,
                        (0, int(waveform_lengths.max()) - waveform.numel()),
                    )
                    for waveform in waveforms
                ])
                output = model(
                    features, lengths.to(device),
                    padded_waveforms.to(device), waveform_lengths.to(device),
                )
            else:
                output = model(features)
            embeddings = F.normalize(output.float(), dim=1).cpu().numpy()
            for (_, relative_output, _), embedding in zip(batch, embeddings):
                buffer = io.BytesIO()
                np.savez_compressed(buffer, embedding=embedding.astype(np.float32))
                archive.writestr(relative_output.as_posix(), buffer.getvalue())
            completed += len(batch)
            if completed % 1000 < len(batch) or completed == len(wav_entries):
                print("已生成：{}/{}".format(completed, len(wav_entries)), flush=True)

    os.replace(str(temporary_zip), str(output_zip))
    print("结果文件：{}".format(output_zip))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("推理失败：{}".format(exc), file=sys.stderr)
        raise
