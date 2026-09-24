"""A complete speaker head with global depth and extended temporal interaction."""
import copy
import math
import os
import torch
from torch import nn
from miwu.tidyvoice_w2vbert2_depth_time_head import (
    CompleteGRLHead, DepthTimeGRLW2VBert2,
)


class AxialContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 384)
        self.temporal = nn.Conv2d(128, 128, (1, 5), padding=(0, 4),
                                  dilation=(1, 2), groups=128)
        self.project = nn.Linear(256, 128)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        b, t, d, c = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(b, t, d, 3, 4, 32).permute(3, 0, 1, 4, 2, 5)
        attention = ((q @ k.transpose(-1, -2)).float() / math.sqrt(32)).softmax(-1)
        depth = (attention.to(v.dtype) @ v).transpose(2, 3).reshape(b, t, d, c)
        time = self.temporal(h.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return x + self.project(torch.cat((depth, time), dim=-1))


class AxialGRLHead(CompleteGRLHead):
    def __init__(self, base):
        super().__init__(base)
        self.global_context = AxialContext()

    def forward(self, states):
        # Keep the official CUDA entry independent of caller autocast.
        with torch.cuda.amp.autocast(enabled=states[0].is_cuda, dtype=torch.float16):
            x = torch.stack([m(h) for m, h in zip(self.adapter_layers, states)], dim=2)
            x = self.global_context(self.interaction(x))
            return self.bottleneck(self.pooling(x.flatten(2))).float()


class AxialGRLW2VBert2(DepthTimeGRLW2VBert2):
    initial_checkpoint = os.environ.get(
        "MIWU_V302_CHECKPOINT",
        "checkpoints/training/v302_final.ckpt",
    )
    use_context = True

    def __init__(self, checkpoint, config_directory, head_checkpoint, trained_checkpoint=None):
        super().__init__(checkpoint, config_directory, head_checkpoint, self.initial_checkpoint)
        self.teacher_head = copy.deepcopy(self.head)
        if self.use_context:
            head = AxialGRLHead(self.base)
            missing, unexpected = head.load_state_dict(self.head.state_dict(), strict=False)
            if unexpected or not missing or any(not k.startswith('global_context.') for k in missing):
                raise RuntimeError('Unexpected V302 initialization mismatch')
            self.head = head
        if trained_checkpoint:
            self.head.load_state_dict(torch.load(trained_checkpoint, map_location='cpu')['state_dict'], strict=True)
        self.requires_grad_(False)

    def _forward_equal(self, waveforms, return_parts=False):
        with torch.no_grad():
            h = self.base.encoder.feature_projection(self.base._extract_features(waveforms))[0]
            states = [h]
            for layer in self.base.encoder.encoder.layers:
                h = layer(h)[0]
                states.append(h)
            if return_parts:
                teacher = self.teacher_head(states)
        result = self.head(states)
        return (result, teacher) if return_parts else result


class ContinuedGRLW2VBert2(AxialGRLW2VBert2):
    use_context = False
