"""Fuse convolutional acoustic frames with SSL frames before speaker pooling."""
import copy
import math
import torch
from torch import nn
from miwu.tidyvoice_w2vbert2_depth_time_head import CompleteGRLHead, DepthTimeGRLW2VBert2
from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2


class CrossEncoderInteraction(nn.Module):
    def __init__(self, acoustic_dimension):
        super().__init__()
        self.query_norm = nn.LayerNorm(3200)
        self.acoustic_norm = nn.LayerNorm(acoustic_dimension)
        self.query = nn.Linear(3200, 256)
        self.key_value = nn.Linear(acoustic_dimension, 512)
        self.output = nn.Linear(256, 3200)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def correction(self, x, acoustic):
        b, t, _ = x.shape
        q = self.query(self.query_norm(x)).reshape(b, t, 4, 64).transpose(1, 2)
        kv = self.key_value(self.acoustic_norm(acoustic))
        k, v = kv.reshape(b, -1, 2, 4, 64).permute(2, 0, 3, 1, 4)
        attention = ((q @ k.transpose(-1, -2)).float() / math.sqrt(64)).softmax(-1)
        y = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(b, t, 256)
        return self.output(y)

    def forward(self, x, acoustic):
        return x + self.correction(x, acoustic)


class CrossEncoderHead(CompleteGRLHead):
    def __init__(self, base, acoustic_dimension):
        super().__init__(base)
        self.cross_encoder = CrossEncoderInteraction(acoustic_dimension)

    def forward(self, states, acoustic):
        with torch.cuda.amp.autocast(enabled=states[0].is_cuda, dtype=torch.float16):
            x = torch.stack([m(h) for m, h in zip(self.adapter_layers, states)], dim=2)
            x = self.cross_encoder(self.interaction(x).flatten(2), acoustic)
            return self.bottleneck(self.pooling(x)).float()


class CrossEncoderGRLW2VBert2(DepthTimeGRLW2VBert2):
    initial_checkpoint = 'outputs/v302_hard_age_depth_time_grl_long/final.ckpt'
    acoustic_checkpoint = 'tmp/v263_r40_fresh/model/redimnet2.ckpt'
    head_class = CrossEncoderHead

    def __init__(self, checkpoint, config_directory, head_checkpoint, trained_checkpoint=None):
        super().__init__(checkpoint, config_directory, head_checkpoint, self.initial_checkpoint)
        self.teacher_head = copy.deepcopy(self.head)
        self.acoustic = load_palabra_multicorpus_redimnet2(self.acoustic_checkpoint, device='cpu')
        encoder = self.acoustic.encoder
        dimension = encoder.backbone.head.out_channels * (encoder.backbone.F // encoder.backbone.freq_stride)
        head = self.head_class(self.base, dimension)
        missing, unexpected = head.load_state_dict(self.head.state_dict(), strict=False)
        if unexpected or not missing or any(not k.startswith('cross_encoder.') for k in missing):
            raise RuntimeError('V302 initialization mismatch')
        self.head = head
        if trained_checkpoint:
            self.head.load_state_dict(torch.load(trained_checkpoint, map_location='cpu')['state_dict'], strict=True)
        self.requires_grad_(False)

    def training_parameter_groups(self, head_lr):
        return [
            {'params': [p for n,p in self.head.named_parameters() if not n.startswith('cross_encoder.')], 'lr':head_lr},
            {'params': self.head.cross_encoder.parameters(), 'lr':10*head_lr},
        ]

    def _forward_equal(self, waveforms, return_parts=False):
        # Match production regardless of the training loop's outer AMP scope.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            h = self.base.encoder.feature_projection(self.base._extract_features(waveforms))[0]
            states = [h]
            for layer in self.base.encoder.encoder.layers:
                h = layer(h)[0]
                states.append(h)
            if return_parts:
                with torch.cuda.amp.autocast(enabled=h.is_cuda, dtype=torch.float16):
                    teacher = self.teacher_head(states)
            spectrum = self.acoustic.encoder.spec(waveforms.to(h.device).float())
            if spectrum.ndim == 3:
                spectrum = spectrum.unsqueeze(1)
            frames = self.acoustic.encoder.backbone(spectrum)
            if frames.ndim != 4:
                raise RuntimeError('Expected released ReDimNet2 B,C,F,T frames')
            acoustic = frames.flatten(1, 2).transpose(1, 2)
        result = self.head(states, acoustic)
        return (result, teacher) if return_parts else result


class CrossEvidencePoolHead(CrossEncoderHead):
    """Acoustic evidence changes frame selection, preserving SSL frame values."""
    def forward(self, states, acoustic):
        with torch.cuda.amp.autocast(enabled=states[0].is_cuda, dtype=torch.float16):
            x = torch.stack([m(h) for m, h in zip(self.adapter_layers, states)], dim=2)
            x = self.interaction(x).flatten(2)
            logits = x.transpose(1, 2)
            for layer in self.pooling.attention[:-1]:
                logits = layer(logits)
            logits = logits + self.cross_encoder.correction(x, acoustic).transpose(1, 2)
            weights = self.pooling.attention[-1](logits).transpose(1, 2)
            mean = torch.sum(x * weights, dim=1)
            std = torch.sqrt((torch.sum(x.square() * weights, dim=1)-mean.square()).clamp(min=1e-5))
            return self.bottleneck(torch.cat((mean, std), dim=1)).float()


class CrossEvidencePoolGRLW2VBert2(CrossEncoderGRLW2VBert2):
    head_class = CrossEvidencePoolHead
