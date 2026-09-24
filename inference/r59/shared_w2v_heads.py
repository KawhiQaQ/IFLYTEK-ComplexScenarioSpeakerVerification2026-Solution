"""Four speaker views sharing the proven R58 S2 acoustic encoder.

The deployed 15% GRL slot is the unit-sphere mean of the original GRL head
and V364's independently trained wide/cross-encoder head.  The final 10%
V360 head is kept unchanged from the successful R58 submission.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthTimeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.local = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.expand = nn.Linear(128, 256)
        self.project = nn.Linear(256, 128)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        y = self.local(self.norm(x).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return x + self.project(F.gelu(self.expand(y)))


class WideResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.expand = nn.Linear(1024, 512)
        self.project = nn.Linear(512, 128)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, hidden):
        return self.project(F.gelu(self.expand(self.norm(hidden))))


class CompleteGRLHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.adapter_layers = copy.deepcopy(base.adapter_layers).float()
        self.pooling = copy.deepcopy(base.pooling).float()
        self.bottleneck = copy.deepcopy(base.bottleneck).float()
        self.interaction = DepthTimeResidual()
        self.wide_adapters = nn.ModuleList(
            [WideResidualAdapter() for _ in self.adapter_layers]
        )

    def adapted_frames(self, states):
        return torch.stack(
            [
                old(hidden) + residual(hidden)
                for old, residual, hidden in zip(
                    self.adapter_layers, self.wide_adapters, states
                )
            ],
            dim=2,
        )

    def forward(self, states):
        with torch.cuda.amp.autocast(
            enabled=states[0].is_cuda, dtype=torch.float16
        ):
            value = self.interaction(self.adapted_frames(states)).flatten(2)
            return self.bottleneck(self.pooling(value)).float()


class CrossEncoderInteraction(nn.Module):
    """V364's cross attention from released ReDimNet2 frames to S2 frames."""

    def __init__(self, acoustic_dimension):
        super().__init__()
        self.query_norm = nn.LayerNorm(3200)
        self.acoustic_norm = nn.LayerNorm(acoustic_dimension)
        self.query = nn.Linear(3200, 256)
        self.key_value = nn.Linear(acoustic_dimension, 512)
        self.output = nn.Linear(256, 3200)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, acoustic):
        batch, steps, _ = x.shape
        query = self.query(self.query_norm(x))
        query = query.reshape(batch, steps, 4, 64).transpose(1, 2)
        key_value = self.key_value(self.acoustic_norm(acoustic))
        key, value = key_value.reshape(
            batch, -1, 2, 4, 64
        ).permute(2, 0, 3, 1, 4)
        attention = (
            (query @ key.transpose(-1, -2)).float() / math.sqrt(64)
        ).softmax(-1)
        result = (attention.to(value.dtype) @ value).transpose(1, 2)
        result = result.reshape(batch, steps, 256)
        return x + self.output(result)


class SharedW2VHeads(nn.Module):
    def __init__(self, base, checkpoint, redimnet2):
        super().__init__()
        self.base = base

        self.grl = nn.Module()
        self.grl.adapter_layers = copy.deepcopy(base.adapter_layers)
        self.grl.pooling = copy.deepcopy(base.pooling)
        self.grl.bottleneck = copy.deepcopy(base.bottleneck)
        package = torch.load(str(checkpoint), map_location="cpu")
        self.grl.load_state_dict(package.get("state_dict", package), strict=True)

        # depth_time_grl.ckpt is V364.  Its non-cross parameters are exactly
        # the proven V360 head used by R58; the extra keys implement V364.
        package = torch.load(
            str(checkpoint.parent / "depth_time_grl.ckpt"), map_location="cpu"
        )
        state = package["state_dict"]
        v360_state = {
            name: value
            for name, value in state.items()
            if not name.startswith("cross_encoder.")
        }
        self.trained_grl = CompleteGRLHead(self.grl)
        self.trained_grl.load_state_dict(v360_state, strict=True)

        acoustic_dimension = (
            redimnet2.backbone.head.out_channels
            * (redimnet2.backbone.F // redimnet2.backbone.freq_stride)
        )
        self.cross_encoder = CrossEncoderInteraction(acoustic_dimension)
        cross_state = {
            name[len("cross_encoder.") :]: value
            for name, value in state.items()
            if name.startswith("cross_encoder.")
        }
        self.cross_encoder.load_state_dict(cross_state, strict=True)

        # The released ReDimNet2 already belongs to the outer ensemble.  Keep
        # a non-registering reference so it is neither copied nor serialized
        # twice, while the outer model still moves it to the inference device.
        object.__setattr__(self, "_redimnet2", redimnet2)

    def redimnet2_frames(self, waveforms):
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = self._redimnet2.spec(waveforms.float())
            if spectrum.ndim == 3:
                spectrum = spectrum.unsqueeze(1)
            frames = self._redimnet2.backbone(spectrum)
            if frames.ndim != 4:
                raise RuntimeError("Expected released ReDimNet2 B,C,F,T frames")
            return frames.flatten(1, 2).transpose(1, 2)

    def _forward_equal(self, waveforms):
        inputs = self.base._extract_features(waveforms)
        hidden = self.base.encoder.feature_projection(inputs)[0]
        states = [hidden]
        for layer in self.base.encoder.encoder.layers:
            hidden = layer(hidden)[0]
            states.append(hidden)

        adapted = [
            layer(value) for layer, value in zip(self.base.adapter_layers, states)
        ]
        original = self.base.bottleneck(
            self.base.pooling(torch.cat(adapted, dim=-1))
        )
        correction = self.base.context_mhfa(torch.stack(adapted, dim=2))
        original = original.float() + correction.float()

        grl_features = torch.cat(
            [
                layer(value)
                for layer, value in zip(self.grl.adapter_layers, states)
            ],
            dim=-1,
        )
        grl = self.grl.bottleneck(self.grl.pooling(grl_features)).float()

        acoustic = self.redimnet2_frames(waveforms)
        with torch.cuda.amp.autocast(
            enabled=states[0].is_cuda, dtype=torch.float16
        ):
            wide_frames = self.trained_grl.interaction(
                self.trained_grl.adapted_frames(states)
            ).flatten(2)
            cross_frames = self.cross_encoder(wide_frames, acoustic)
            cross = self.trained_grl.bottleneck(
                self.trained_grl.pooling(cross_frames)
            ).float()
            trained = self.trained_grl.bottleneck(
                self.trained_grl.pooling(wide_frames)
            ).float()

        # Equal unit-sphere fusion is a fixed model-level ensemble.  It uses
        # no pair, filename, scenario, threshold, or test-set statistic.
        fused_grl = F.normalize(
            F.normalize(grl, dim=1) + F.normalize(cross, dim=1), dim=1
        )
        return torch.cat((original.float(), fused_grl, trained), dim=1)

    def forward(self, waveforms, waveform_lengths):
        all_equal = (
            bool(torch.all(waveform_lengths == waveform_lengths[0]))
            and int(waveform_lengths[0]) == waveforms.shape[1]
        )
        if not all_equal:
            return torch.cat(
                [
                    self._forward_equal(waveforms[index:index + 1, : int(length)])
                    for index, length in enumerate(waveform_lengths.tolist())
                ],
                dim=0,
            )
        return self._forward_equal(waveforms)
