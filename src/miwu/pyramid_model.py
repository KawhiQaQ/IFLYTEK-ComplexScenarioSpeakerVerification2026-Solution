"""Cross-depth feature-pyramid ERes2NetV2 for short speaker verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miwu.duration_model import AttentiveScalePool, TemporalScaleBlock, _masked_statistics
from miwu.encoder import build_encoder


class SpeakerFeaturePyramid(nn.Module):
    """Fuse shallow temporal detail and deep channel-invariant features."""

    def __init__(self, stage_channels=(128, 256, 512, 1024), channels=320):
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(stage_channel, 64, kernel_size=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                )
                for stage_channel in stage_channels
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(64 * 4, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(),
                )
                for _ in stage_channels
            ]
        )
        self.level_gate = nn.Sequential(
            nn.Linear(len(stage_channels) + 2, 32),
            nn.SiLU(),
            nn.Linear(32, len(stage_channels)),
        )
        self.scales = nn.ModuleList(
            [TemporalScaleBlock(channels, dilation) for dilation in (1, 2, 4)]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(2, 24),
            nn.SiLU(),
            nn.Linear(24, len(self.scales)),
        )
        self.pools = nn.ModuleList([AttentiveScalePool(channels) for _ in self.scales])
        self.output = nn.Sequential(
            nn.Linear(channels * 2, 384),
            nn.BatchNorm1d(384),
            nn.PReLU(384),
            nn.Linear(384, 192),
        )
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _masked_descriptor(x, lengths):
        steps = torch.arange(x.shape[-1], device=x.device)[None, :]
        mask = (steps < lengths[:, None]).to(x.dtype)[:, None, :]
        return (x.abs() * mask).sum(dim=(1, 2)) / (
            lengths.to(x.dtype).clamp_min(1) * x.shape[1]
        )

    def forward(self, stages, input_lengths):
        target_steps = stages[0].shape[-1]
        projected = []
        descriptors = []
        for stage, lateral, projection in zip(stages, self.lateral, self.projections):
            stage = lateral(stage)
            stage = F.adaptive_avg_pool2d(stage, (4, stage.shape[-1])).flatten(1, 2)
            stage = projection(stage)
            stage = F.interpolate(stage, size=target_steps, mode="linear", align_corners=False)
            projected.append(stage)
            descriptors.append(self._masked_descriptor(stage, input_lengths))

        duration = input_lengths.to(projected[0].dtype)
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        level_features = torch.cat((torch.stack(descriptors, dim=1), duration_features), dim=1)
        level_weights = torch.softmax(self.level_gate(level_features).float(), dim=1).to(projected[0].dtype)
        fused = sum(
            value * level_weights[:, index, None, None]
            for index, value in enumerate(projected)
        )

        scale_weights = torch.softmax(self.scale_gate(duration_features).float(), dim=1).to(fused.dtype)
        statistics = torch.stack(
            [pool(block(fused), input_lengths) for block, pool in zip(self.scales, self.pools)],
            dim=1,
        )
        statistics = (statistics * scale_weights[:, :, None]).sum(dim=1)
        return self.output(statistics)


class PyramidERes2NetV2(nn.Module):
    def __init__(self, backbone_checkpoint, device="cuda"):
        super().__init__()
        self.backbone = build_encoder(backbone_checkpoint, device="cpu", architecture="eres2netv2")
        self.pyramid = SpeakerFeaturePyramid()
        self.adapter_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = F.relu(backbone.bn1(backbone.conv1(x)))
        out1 = backbone.layer1(x)
        out2 = backbone.layer2(out1)
        out3 = backbone.layer3(out2)
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return (out1, out2, out3, final)

    @staticmethod
    def _final_lengths(lengths):
        result = lengths
        for _ in range(3):
            result = torch.div(result + 1, 2, rounding_mode="floor")
        return result

    def forward(self, features, lengths=None):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long, device=features.device
            )
        stages = self._stages(features)
        base_stats = _masked_statistics(stages[-1], self._final_lengths(lengths))
        base_embedding = self.backbone.seg_1(base_stats)
        return base_embedding + self.adapter_scale * self.pyramid(stages, lengths)


def load_pyramid_model(backbone_checkpoint, model_checkpoint=None, device="cuda"):
    model = PyramidERes2NetV2(backbone_checkpoint, device="cpu")
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)
