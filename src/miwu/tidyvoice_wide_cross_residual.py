"""V364: add a convolutional cross residual to the proven V360 wide head."""
import copy
import os

import torch

from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2
from miwu.tidyvoice_cross_encoder import CrossEncoderInteraction
from miwu.tidyvoice_w2vbert2_depth_time_head import DepthTimeGRLW2VBert2
from miwu.tidyvoice_wide_adapter_head import WideResidualGRLHead


class WideCrossResidualHead(WideResidualGRLHead):
    def __init__(self, base, acoustic_dimension):
        super().__init__(base)
        self.cross_encoder = CrossEncoderInteraction(acoustic_dimension)

    def forward(self, states, acoustic):
        with torch.cuda.amp.autocast(
            enabled=states[0].is_cuda, dtype=torch.float16
        ):
            adapted = [
                old(hidden) + residual(hidden)
                for old, residual, hidden in zip(
                    self.adapter_layers, self.wide_adapters, states
                )
            ]
            value = self.interaction(torch.stack(adapted, dim=2)).flatten(2)
            value = self.cross_encoder(value, acoustic)
            return self.bottleneck(self.pooling(value)).float()


class WideCrossResidualGRLW2VBert2(DepthTimeGRLW2VBert2):
    initial_checkpoint = os.environ.get(
        "MIWU_V360_CHECKPOINT",
        "outputs/v360_wide_adapter/final.ckpt",
    )
    acoustic_checkpoint = os.environ.get(
        "MIWU_REDIMNET2_CHECKPOINT",
        "checkpoints/pretrained/r45_runtime/model/redimnet2.ckpt",
    )

    def __init__(
        self, checkpoint, config_directory, head_checkpoint,
        trained_checkpoint=None,
    ):
        super().__init__(checkpoint, config_directory, head_checkpoint)
        wide_state = torch.load(
            self.initial_checkpoint, map_location="cpu"
        )["state_dict"]
        teacher = WideResidualGRLHead(self.base)
        teacher.load_state_dict(wide_state, strict=True)
        self.teacher_head = teacher

        self.acoustic = load_palabra_multicorpus_redimnet2(
            self.acoustic_checkpoint, device="cpu"
        )
        encoder = self.acoustic.encoder
        acoustic_dimension = (
            encoder.backbone.head.out_channels
            * (encoder.backbone.F // encoder.backbone.freq_stride)
        )
        head = WideCrossResidualHead(self.base, acoustic_dimension)
        missing, unexpected = head.load_state_dict(wide_state, strict=False)
        if unexpected or not missing or any(
            not key.startswith("cross_encoder.") for key in missing
        ):
            raise RuntimeError("V360 initialization mismatch")
        self.head = head
        if trained_checkpoint is not None:
            state = torch.load(trained_checkpoint, map_location="cpu")
            self.head.load_state_dict(state["state_dict"], strict=True)
        self.requires_grad_(False)

    def set_trainable(self):
        self.requires_grad_(False)
        self.eval()
        self.head.cross_encoder.requires_grad_(True)
        return self

    def training_parameter_groups(self, head_lr):
        return [
            {
                "params": self.head.cross_encoder.parameters(),
                "lr": 10.0 * head_lr,
            }
        ]

    def _forward_equal(self, waveforms, return_parts=False):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            hidden = self.base.encoder.feature_projection(
                self.base._extract_features(waveforms)
            )[0]
            states = [hidden]
            for layer in self.base.encoder.encoder.layers:
                hidden = layer(hidden)[0]
                states.append(hidden)
            if return_parts:
                teacher = self.teacher_head(states)
            spectrum = self.acoustic.encoder.spec(
                waveforms.to(hidden.device).float()
            )
            if spectrum.ndim == 3:
                spectrum = spectrum.unsqueeze(1)
            frames = self.acoustic.encoder.backbone(spectrum)
            if frames.ndim != 4:
                raise RuntimeError("Expected released ReDimNet2 B,C,F,T frames")
            acoustic = frames.flatten(1, 2).transpose(1, 2)
        result = self.head(states, acoustic)
        return (result, teacher) if return_parts else result
