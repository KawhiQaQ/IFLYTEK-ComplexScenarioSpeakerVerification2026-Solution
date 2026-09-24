"""Quality-conditioned fusion for the ERes2NetV2 feature pyramid."""

import torch
import torch.nn as nn

from miwu.duration_model import _masked_statistics
from miwu.pyramid_model import load_pyramid_model


class QualityGatedPyramid(nn.Module):
    """Adapt the pyramid contribution per utterance and embedding channel."""

    def __init__(self, backbone_checkpoint, pyramid_checkpoint, device="cuda"):
        super().__init__()
        parent = load_pyramid_model(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        self.backbone = parent.backbone
        self.pyramid = parent.pyramid
        self.adapter_scale = parent.adapter_scale
        self.quality_gate = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, 192),
        )
        # Zero initialization makes the architecture exactly reproduce its
        # V7 parent before learning any duration- or quality-specific routing.
        nn.init.zeros_(self.quality_gate[-1].weight)
        nn.init.zeros_(self.quality_gate[-1].bias)
        self.to(device)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(backbone.bn1(backbone.conv1(x)))
        out1 = backbone.layer1(x)
        out2 = backbone.layer2(out1)
        out3 = backbone.layer3(out2)
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final

    @staticmethod
    def _final_lengths(lengths):
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
        return torch.stack(
            (
                torch.log1p(duration) / 6.0,
                torch.rsqrt(duration.clamp_min(1)) * 4.0,
                rms / 5.0,
                magnitude / 5.0,
                temporal_change / 5.0,
            ),
            dim=1,
        )

    def forward(self, features, lengths=None, return_gate=False):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1],
                dtype=torch.long, device=features.device,
            )
        stages = self._stages(features)
        base_stats = _masked_statistics(stages[-1], self._final_lengths(lengths))
        base_embedding = self.backbone.seg_1(base_stats)
        adaptation = self.pyramid(stages, lengths)
        gate_delta = self.quality_gate(self.quality_features(features, lengths))
        multiplier = torch.exp(0.5 * torch.tanh(gate_delta))
        output = base_embedding + self.adapter_scale * multiplier * adaptation
        if return_gate:
            return output, multiplier
        return output


def load_quality_model(
    backbone_checkpoint, pyramid_checkpoint, model_checkpoint=None, device="cuda"
):
    model = QualityGatedPyramid(
        backbone_checkpoint, pyramid_checkpoint, device="cpu"
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)
