"""Phase-complete temporal stages for short-utterance ReDimNet2.

A strided stage normally computes one temporal phase and repeats every output
frame. This stage computes all stride phases with shared weights and interleaves
their features before the existing normalization and later stage aggregation.
It emits one dense feature sequence, not multiple utterance embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseTimeStage(nn.Sequential):
    def __init__(self, stage, time_stride):
        super().__init__(*list(stage.children()))
        self.time_stride = int(time_stride)
        upsample = [i for i, module in enumerate(self) if isinstance(module, nn.Upsample)]
        if len(upsample) != 1:
            raise ValueError("Expected one nearest temporal upsampling operation")
        self.upsample_index = upsample[0]
        if self[self.upsample_index].mode != 'nearest':
            raise ValueError("Unexpected temporal interpolation")
        if self[self.upsample_index].scale_factor != float(self.time_stride):
            raise ValueError("Stage stride does not match upsampling scale")

    def forward(self, previous):
        value = self[0](previous)
        batch, channels, frames = value.shape
        stride = self.time_stride
        if frames % stride:
            raise ValueError("Backbone must align temporal length before stages")
        # No circular wrap: shifted windows only extend the recording endpoint.
        value = torch.cat([F.pad(value[..., phase:], (0, phase), mode='replicate')
                           if phase else value for phase in range(stride)], dim=0)
        for index in range(1, len(self)):
            if index == self.upsample_index:
                if value.shape[-1] * stride != frames:
                    raise RuntimeError("Dense-time stage lost temporal alignment")
                value = value.reshape(stride, batch, value.shape[1], frames // stride)
                value = value.permute(1, 2, 3, 0).reshape(batch, -1, frames)
            else:
                value = self[index](value)
        return value


def enable_dense_backbone(backbone):
    if any(backbone._stage_has_dual):
        raise ValueError("Dense-time experiment expects the verified R39 single-aggregation backbone")
    before = set(backbone.state_dict())
    cumulative_stride = 1
    changed = []
    for index, setup in enumerate(backbone.stages_setup):
        cumulative_stride *= int(setup[0][1])
        if cumulative_stride > 1:
            name = 'stage%d' % index
            stage = getattr(backbone, name)
            if not isinstance(stage, DenseTimeStage):
                setattr(backbone, name, DenseTimeStage(stage, cumulative_stride))
            changed.append(name)
    if set(backbone.state_dict()) != before:
        raise RuntimeError("Dense-time conversion changed checkpoint tensor names")
    backbone.dense_time_stages = tuple(changed)
    return backbone


def enable_dense_time(model):
    backbone = enable_dense_backbone(model.temporal.encoder.backbone)
    model.dense_time_stages = backbone.dense_time_stages
    return model
