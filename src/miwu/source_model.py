"""A waveform-domain harmonic source adapter for the V8 speaker encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from miwu.duration_model import _masked_statistics
from miwu.quality_model import QualityGatedPyramid, load_quality_model


class FeatureMapScale(nn.Module):
    """Utterance-dependent channel scaling used by raw-waveform encoders."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.network = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mean = x.mean(dim=(-2, -1))
        std = torch.sqrt(
            (x - mean[:, :, None, None]).square().mean(dim=(-2, -1)) + 1e-6
        )
        scale = self.network(torch.cat((mean, std), dim=1))
        return x * (0.5 + scale[:, :, None, None])


class SourceResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=stride,
                padding=1, groups=input_channels, bias=False,
            ),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3,
                padding=1, groups=output_channels, bias=False,
            ),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False),
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

    def forward(self, x):
        return F.silu(self.scale(self.main(x) + self.skip(x)))


class HarmonicSourceEncoder(nn.Module):
    """Encode short-term normalized autocorrelation from raw speech."""

    def __init__(self, embedding_dim=192):
        super().__init__()
        self.register_buffer("window", torch.hann_window(640), persistent=True)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1),
                      bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            SourceResidualBlock(32, 64, stride=(2, 2)),
            SourceResidualBlock(64, 96, stride=(2, 2)),
            SourceResidualBlock(96, 128, stride=(2, 1)),
        )
        self.temporal = nn.Sequential(
            nn.LazyConv1d(256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=7, padding=3, groups=256, bias=False),
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

    def initialize_residual(self, waveforms):
        """Materialize lazy weights, then make the adapter start at zero."""
        training = self.training
        self.eval()
        with torch.no_grad():
            self(waveforms)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.train(training)

    def autocorrelation_features(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        if waveforms.shape[1] < 640:
            waveforms = F.pad(waveforms, (0, 640 - waveforms.shape[1]))
        frames = waveforms.unfold(1, 640, 160)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * self.window
        spectrum = torch.fft.rfft(frames, n=2048, dim=-1)
        correlation = torch.fft.irfft(
            spectrum.real.square() + spectrum.imag.square(), n=2048, dim=-1
        )[..., :241]
        correlation = correlation / correlation[..., :1].clamp_min(1e-6)
        overlap_correction = 640.0 / (
            640.0 - torch.arange(241, device=waveforms.device)
        )
        correlation = correlation * overlap_correction
        # Lags 20..240 cover roughly 67--800 Hz and include child pitch.
        return correlation[..., 20:241:4].permute(0, 2, 1).unsqueeze(1)

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

    def forward(self, waveforms, return_quality=False):
        with torch.no_grad():
            source = self.autocorrelation_features(waveforms)
            quality = self.source_quality(source)
        x = self.blocks(self.stem(source))
        x = x.flatten(1, 2)
        x = self.temporal(x)
        weights = torch.softmax(self.attention(x).float(), dim=-1).to(x.dtype)
        mean = (weights * x).sum(dim=-1)
        variance = (weights * (x - mean[..., None]).square()).sum(dim=-1)
        statistics = torch.cat(
            (mean, torch.sqrt(variance.clamp_min(1e-6))), dim=1
        )
        output = self.output(statistics)
        if return_quality:
            return output, quality
        return output


class ExcitationSourceEncoder(HarmonicSourceEncoder):
    """Add an envelope-whitened excitation view to the raw autocorrelation."""

    def __init__(self, embedding_dim=192):
        super().__init__(embedding_dim=embedding_dim)
        self.stem[0] = nn.Conv2d(
            2, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1),
            bias=False,
        )
        self.initialize_residual(torch.zeros(2, 8000))

    @staticmethod
    def _lag_view(correlation, device):
        correlation = correlation[..., :241]
        correlation = correlation / correlation[..., :1].clamp_min(1e-6)
        overlap_correction = 640.0 / (
            640.0 - torch.arange(241, device=device)
        )
        correlation = correlation * overlap_correction
        return correlation[..., 20:241:4].permute(0, 2, 1)

    def autocorrelation_features(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        if waveforms.shape[1] < 640:
            waveforms = F.pad(waveforms, (0, 640 - waveforms.shape[1]))
        frames = waveforms.unfold(1, 640, 160)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * self.window
        spectrum = torch.fft.rfft(frames, n=2048, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        raw_correlation = torch.fft.irfft(power, n=2048, dim=-1)

        shape = power.shape
        envelope = F.avg_pool1d(
            power.reshape(-1, 1, shape[-1]), kernel_size=31,
            stride=1, padding=15,
        ).reshape(shape)
        whitened_power = (power / envelope.clamp_min(1e-6)).clamp_max(20.0)
        excitation_correlation = torch.fft.irfft(
            whitened_power, n=2048, dim=-1
        )
        raw = self._lag_view(raw_correlation, waveforms.device)
        excitation = self._lag_view(
            excitation_correlation, waveforms.device
        )
        return torch.stack((raw, excitation), dim=1)


class MultiResolutionSourceEncoder(HarmonicSourceEncoder):
    """Use 20, 40 and 80 ms source views for variable-duration speech."""

    def __init__(self, embedding_dim=192):
        super().__init__(embedding_dim=embedding_dim)
        self.register_buffer(
            "short_window", torch.hann_window(320), persistent=True
        )
        self.register_buffer(
            "long_window", torch.hann_window(1280), persistent=True
        )
        self.stem[0] = nn.Conv2d(
            6, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1),
            bias=False,
        )
        self.initialize_residual(torch.zeros(2, 8000))

    @staticmethod
    def _resolution_view(
        waveforms, frame_size, n_fft, envelope_kernel, window
    ):
        padded = waveforms
        if padded.shape[1] < frame_size:
            padded = F.pad(padded, (0, frame_size - padded.shape[1]))
        frames = padded.unfold(1, frame_size, 160)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * window
        spectrum = torch.fft.rfft(frames, n=n_fft, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        raw_correlation = torch.fft.irfft(power, n=n_fft, dim=-1)
        shape = power.shape
        envelope = F.avg_pool1d(
            power.reshape(-1, 1, shape[-1]), kernel_size=envelope_kernel,
            stride=1, padding=envelope_kernel // 2,
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
            views.extend(self._resolution_view(waveforms, *setup))
        target_steps = views[0].shape[-1]
        views = [
            view if view.shape[-1] == target_steps else F.interpolate(
                view, size=target_steps, mode="linear", align_corners=False
            )
            for view in views
        ]
        return torch.stack(views, dim=1)


class SourceAwareQualityPyramid(QualityGatedPyramid):
    """Augment the V8 spectral representation with periodic source cues."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(backbone_checkpoint, pyramid_checkpoint, device="cpu")
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.source_encoder = HarmonicSourceEncoder()
        self.source_encoder.initialize_residual(torch.zeros(2, 8000))
        self.source_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the source-aware model")
        parent = super().forward(features, lengths)
        source = self.source_encoder(waveforms)
        output = parent + self.source_scale * source
        if return_source:
            return output, source
        return output


def load_source_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = SourceAwareQualityPyramid(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint, device="cpu"
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ExcitationAwareQualityPyramid(SourceAwareQualityPyramid):
    """Refine V13 with source-filter separated excitation periodicity."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        parent = load_source_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        old_encoder = self.source_encoder
        new_encoder = ExcitationSourceEncoder()
        old_state = old_encoder.state_dict()
        new_state = new_encoder.state_dict()
        for name, value in old_state.items():
            if name == "stem.0.weight":
                new_state[name].zero_()
                new_state[name][:, :1].copy_(value)
            else:
                new_state[name].copy_(value)
        new_encoder.load_state_dict(new_state, strict=True)
        self.source_encoder = new_encoder
        self.to(device)


def load_excitation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, model_checkpoint=None, device="cuda",
):
    model = ExcitationAwareQualityPyramid(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class MultiResolutionQualityPyramid(ExcitationAwareQualityPyramid):
    """Refine V14 with short, medium and long waveform analysis windows."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, device="cpu",
        )
        parent = load_excitation_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        old_encoder = self.source_encoder
        new_encoder = MultiResolutionSourceEncoder()
        old_state = old_encoder.state_dict()
        new_state = new_encoder.state_dict()
        for name, value in old_state.items():
            if name == "stem.0.weight":
                new_state[name].zero_()
                new_state[name][:, :2].copy_(value)
            else:
                new_state[name].copy_(value)
        new_encoder.load_state_dict(new_state, strict=True)
        self.source_encoder = new_encoder
        self.to(device)


def load_multiresolution_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = MultiResolutionQualityPyramid(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class QualityRoutedMultiResolution(MultiResolutionQualityPyramid):
    """Route source evidence per channel using signal and periodic quality."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint, device="cpu",
        )
        parent = load_multiresolution_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.source_gate = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.Linear(64, 192),
        )
        nn.init.zeros_(self.source_gate[-1].weight)
        nn.init.zeros_(self.source_gate[-1].bias)
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the source-aware model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        parent = QualityGatedPyramid.forward(self, features, lengths)
        source, periodic_quality = self.source_encoder(
            waveforms, return_quality=True
        )
        gate_input = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        multiplier = 2.0 * torch.sigmoid(self.source_gate(gate_input))
        output = parent + self.source_scale * multiplier * source
        if return_source:
            return output, source
        return output


def load_source_gate_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = QualityRoutedMultiResolution(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class TangentSourceRefinement(QualityRoutedMultiResolution):
    """Turn source evidence into an explicit angular embedding correction."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, device="cpu",
        )
        parent = load_source_gate_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        # The interaction term lets the head distinguish source cues already
        # represented by the spectral embedding from complementary evidence.
        self.tangent_head = nn.Sequential(
            nn.LayerNorm(192 * 3),
            nn.Linear(192 * 3, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.tangent_head[-1].weight)
        nn.init.zeros_(self.tangent_head[-1].bias)
        self.tangent_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False,
    ):
        parent, source = QualityRoutedMultiResolution.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_source=True,
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        source_direction = F.normalize(source.float(), dim=1).to(source.dtype)
        candidate = self.tangent_head(torch.cat(
            (
                parent_direction,
                source_direction,
                parent_direction * source_direction,
            ),
            dim=1,
        ))
        # Cosine scoring only observes direction. Removing the radial component
        # makes every learned residual act on the geometry used by evaluation.
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.tangent_scale * tangent
        if return_source:
            return output, tangent
        return output


def load_tangent_source_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, model_checkpoint=None, device="cuda",
):
    model = TangentSourceRefinement(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class TemporalNormalizedStageAdapter(nn.Module):
    """Extract channel-robust temporal evidence without replacing base features."""

    def __init__(self, channels, bottleneck):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, bottleneck)
        self.temporal = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=(3, 5), padding=(1, 2),
            groups=bottleneck, bias=False,
        )
        self.mix = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=1, bias=False
        )
        self.output = nn.Conv2d(
            bottleneck, channels, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        # Static channel response is largely constant over time. The residual
        # path sees standardized trajectories while the pretrained path keeps
        # the original magnitude and timbre information.
        mean = x.float().mean(dim=-1, keepdim=True)
        variance = (x.float() - mean).square().mean(dim=-1, keepdim=True)
        normalized = ((x.float() - mean) * torch.rsqrt(variance + 1e-5)).to(x.dtype)
        residual = F.silu(self.norm(self.down(normalized)))
        residual = F.silu(self.mix(self.temporal(residual)))
        return x + self.scale * self.output(residual)


class StageAdaptedTangentSource(TangentSourceRefinement):
    """Insert temporal-normalized residual adapters throughout the backbone."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint, device="cpu",
        )
        parent = load_tangent_source_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.stage_adapters = nn.ModuleList([
            TemporalNormalizedStageAdapter(128, 32),
            TemporalNormalizedStageAdapter(256, 32),
            TemporalNormalizedStageAdapter(512, 64),
            TemporalNormalizedStageAdapter(1024, 64),
        ])
        self.to(device)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(backbone.bn1(backbone.conv1(x)))
        out1 = self.stage_adapters[0](backbone.layer1(x))
        out2 = self.stage_adapters[1](backbone.layer2(out1))
        out3 = self.stage_adapters[2](backbone.layer3(out2))
        out4 = self.stage_adapters[3](backbone.layer4(out3))
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final


def load_stage_adapter_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = StageAdaptedTangentSource(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class QualityRoutedStageAdapter(StageAdaptedTangentSource):
    """Route the learned stage transformation by duration and periodicity."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, stage_adapter_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        expert = load_stage_adapter_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, stage_adapter_checkpoint, device="cpu",
        )
        self.load_state_dict(expert.state_dict(), strict=True)
        self.stage_router = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )
        nn.init.zeros_(self.stage_router[-1].weight)
        nn.init.zeros_(self.stage_router[-1].bias)
        self._stage_multipliers = None
        self.to(device)

    def _apply_expert(self, index, x):
        adapted = self.stage_adapters[index](x)
        multiplier = self._stage_multipliers[:, index, None, None, None]
        return x + multiplier * (adapted - x)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(backbone.bn1(backbone.conv1(x)))
        out1 = self._apply_expert(0, backbone.layer1(x))
        out2 = self._apply_expert(1, backbone.layer2(out1))
        out3 = self._apply_expert(2, backbone.layer3(out2))
        out4 = self._apply_expert(3, backbone.layer4(out3))
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final

    def _periodic_quality(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        views = MultiResolutionSourceEncoder._resolution_view(
            waveforms, 640, 2048, 31, self.source_encoder.window
        )
        return self.source_encoder.source_quality(torch.stack(views, dim=1))

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the routed stage model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        with torch.no_grad():
            periodic_quality = self._periodic_quality(waveforms)
        routing_input = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        # Signed routing can enable, suppress, or reverse the fixed expert.
        # A zero-initialized router exactly reproduces the V17 parent.
        self._stage_multipliers = torch.tanh(self.stage_router(routing_input))
        try:
            return TangentSourceRefinement.forward(
                self, features, lengths, waveforms, waveform_lengths,
                return_source=return_source,
            )
        finally:
            self._stage_multipliers = None


def load_routed_stage_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, stage_adapter_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = QualityRoutedStageAdapter(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, stage_adapter_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class AgeConditionedTangentMixture(TangentSourceRefinement):
    """Use supervised age routing for child/adult angular residual experts."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint, device="cpu",
        )
        parent = load_tangent_source_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.age_head = nn.Sequential(
            nn.Linear(192 + 8, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.child_expert = self._expert()
        self.adult_expert = self._expert()
        self.mixture_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    @staticmethod
    def _expert():
        expert = nn.Sequential(
            nn.LayerNorm(192 * 3 + 8),
            nn.Linear(192 * 3 + 8, 256),
            nn.SiLU(),
            nn.Linear(256, 192),
        )
        nn.init.zeros_(expert[-1].weight)
        nn.init.zeros_(expert[-1].bias)
        return expert

    def _periodic_quality(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        views = MultiResolutionSourceEncoder._resolution_view(
            waveforms, 640, 2048, 31, self.source_encoder.window
        )
        return self.source_encoder.source_quality(torch.stack(views, dim=1))

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the age mixture model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        base, source = QualityRoutedMultiResolution.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_source=True,
        )
        base_direction = F.normalize(base.float(), dim=1).to(base.dtype)
        source_direction = F.normalize(source.float(), dim=1).to(source.dtype)
        v17_candidate = self.tangent_head(torch.cat(
            (
                base_direction,
                source_direction,
                base_direction * source_direction,
            ),
            dim=1,
        ))
        v17_tangent = v17_candidate - (
            v17_candidate * base_direction
        ).sum(dim=1, keepdim=True) * base_direction
        parent = base + self.tangent_scale * v17_tangent

        with torch.no_grad():
            periodic_quality = self._periodic_quality(waveforms)
        quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        age_logit = self.age_head(
            torch.cat((source_direction, quality), dim=1)
        ).squeeze(1)
        child_probability = torch.sigmoid(age_logit).detach()
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        expert_input = torch.cat(
            (
                parent_direction,
                source_direction,
                parent_direction * source_direction,
                quality,
            ),
            dim=1,
        )
        candidate = (
            child_probability[:, None] * self.child_expert(expert_input)
            + (1.0 - child_probability[:, None]) * self.adult_expert(expert_input)
        )
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.mixture_scale * tangent
        if return_aux:
            return output, tangent, age_logit, child_probability
        if return_source:
            return output, tangent
        return output


def load_age_mixture_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = AgeConditionedTangentMixture(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class NuisanceInvariantTangent(TangentSourceRefinement):
    """Learn an angular correction that suppresses device and distance cues."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint, device="cpu",
        )
        parent = load_tangent_source_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.nuisance_expert = nn.Sequential(
            nn.LayerNorm(192 * 3 + 8),
            nn.Linear(192 * 3 + 8, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.nuisance_expert[-1].weight)
        nn.init.zeros_(self.nuisance_expert[-1].bias)
        self.nuisance_scale = nn.Parameter(torch.tensor(0.10))
        self.device_head = nn.Sequential(
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 8)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 11)
        )
        self.to(device)

    def _periodic_quality(self, waveforms):
        waveforms = waveforms.float()
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)
        waveforms = waveforms / (
            waveforms.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        views = MultiResolutionSourceEncoder._resolution_view(
            waveforms, 640, 2048, 31, self.source_encoder.window
        )
        return self.source_encoder.source_quality(torch.stack(views, dim=1))

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the nuisance model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        base, source = QualityRoutedMultiResolution.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_source=True,
        )
        base_direction = F.normalize(base.float(), dim=1).to(base.dtype)
        source_direction = F.normalize(source.float(), dim=1).to(source.dtype)
        v17_input = torch.cat(
            (
                base_direction,
                source_direction,
                base_direction * source_direction,
            ),
            dim=1,
        )
        v17_candidate = self.tangent_head(v17_input)
        v17_tangent = v17_candidate - (
            v17_candidate * base_direction
        ).sum(dim=1, keepdim=True) * base_direction
        parent = base + self.tangent_scale * v17_tangent

        with torch.no_grad():
            periodic_quality = self._periodic_quality(waveforms)
        quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        expert_input = torch.cat(
            (
                parent_direction,
                source_direction,
                parent_direction * source_direction,
                quality,
            ),
            dim=1,
        )
        candidate = self.nuisance_expert(expert_input)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.nuisance_scale * tangent
        if return_components:
            return output, tangent, source, parent
        if return_aux:
            return output, tangent
        if return_source:
            return output, tangent
        return output


def load_nuisance_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = NuisanceInvariantTangent(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class FactorizedNuisanceTangent(NuisanceInvariantTangent):
    """Separate device and distance corrections before combining them."""

    factorized = True

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.device_expert = self._factor_expert()
        self.distance_expert = self._factor_expert()
        self.device_scale = nn.Parameter(torch.tensor(0.10))
        self.distance_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    @staticmethod
    def _factor_expert():
        expert = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Linear(256, 192),
        )
        nn.init.zeros_(expert[-1].weight)
        nn.init.zeros_(expert[-1].bias)
        return expert

    @staticmethod
    def _tangent(candidate, direction):
        return candidate - (
            candidate * direction
        ).sum(dim=1, keepdim=True) * direction

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False,
    ):
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        device_tangent = self._tangent(self.device_expert(direction), direction)
        distance_tangent = self._tangent(
            self.distance_expert(direction), direction
        )
        device_view = parent + self.device_scale * device_tangent
        distance_view = parent + self.distance_scale * distance_tangent
        output = (
            parent + self.device_scale * device_tangent
            + self.distance_scale * distance_tangent
        )
        combined_tangent = device_tangent + distance_tangent
        if return_aux:
            return output, combined_tangent, device_view, distance_view
        if return_source:
            return output, combined_tangent
        return output


def load_factorized_nuisance_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = FactorizedNuisanceTangent(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class PairedChannelCanonicalizer(NuisanceInvariantTangent):
    """Canonicalize channel variation using a single-utterance residual.

    Training can exploit synchronized cross-device recordings, while inference
    needs only the utterance itself.  The correction is conditioned on the V21
    nuisance tangent and its frozen device/distance posteriors.
    """

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        input_dimension = 192 * 3 + 8 + 11
        self.channel_expert = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.channel_expert[-1].weight)
        nn.init.zeros_(self.channel_expert[-1].bias)
        self.channel_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False,
    ):
        parent, nuisance_tangent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_aux=True,
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        nuisance_direction = F.normalize(
            nuisance_tangent.float(), dim=1
        ).to(parent.dtype)
        with torch.no_grad():
            device_posterior = F.softmax(
                self.device_head(parent_direction).float(), dim=1
            ).to(parent.dtype)
            distance_posterior = F.softmax(
                self.distance_head(parent_direction).float(), dim=1
            ).to(parent.dtype)
        expert_input = torch.cat(
            (
                parent_direction,
                nuisance_direction,
                parent_direction * nuisance_direction,
                device_posterior,
                distance_posterior,
            ),
            dim=1,
        )
        candidate = self.channel_expert(expert_input)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.channel_scale * tangent
        if return_aux:
            return output, tangent, parent
        if return_source:
            return output, tangent
        return output


def load_paired_channel_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = PairedChannelCanonicalizer(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class PairedStageChannelAdapter(NuisanceInvariantTangent):
    """Apply synchronized-channel supervision before multiscale pooling."""

    paired_stage = True

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.stage_adapters = nn.ModuleList([
            TemporalNormalizedStageAdapter(128, 32),
            TemporalNormalizedStageAdapter(256, 32),
            TemporalNormalizedStageAdapter(512, 64),
            TemporalNormalizedStageAdapter(1024, 64),
        ])
        self.to(device)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(backbone.bn1(backbone.conv1(x)))
        out1 = self.stage_adapters[0](backbone.layer1(x))
        out2 = self.stage_adapters[1](backbone.layer2(out1))
        out3 = self.stage_adapters[2](backbone.layer3(out2))
        out4 = self.stage_adapters[3](backbone.layer4(out3))
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final


def load_paired_stage_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = PairedStageChannelAdapter(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ExtendedNuisanceTangent(NuisanceInvariantTangent):
    """Refine V21 with one shared device, distance, and dialect correction."""

    extended_nuisance = True

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.dialect_head = nn.Sequential(
            nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 2)
        )
        input_dimension = 192 * 3 + 8 + 11 + 2
        self.extended_expert = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.extended_expert[-1].weight)
        nn.init.zeros_(self.extended_expert[-1].bias)
        self.extended_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False,
    ):
        parent, nuisance_tangent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_aux=True,
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        nuisance_direction = F.normalize(
            nuisance_tangent.float(), dim=1
        ).to(parent.dtype)
        with torch.no_grad():
            device_posterior = F.softmax(
                self.device_head(parent_direction).float(), dim=1
            ).to(parent.dtype)
            distance_posterior = F.softmax(
                self.distance_head(parent_direction).float(), dim=1
            ).to(parent.dtype)
            dialect_posterior = F.softmax(
                self.dialect_head(parent_direction).float(), dim=1
            ).to(parent.dtype)
        expert_input = torch.cat(
            (
                parent_direction,
                nuisance_direction,
                parent_direction * nuisance_direction,
                device_posterior,
                distance_posterior,
                dialect_posterior,
            ),
            dim=1,
        )
        candidate = self.extended_expert(expert_input)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.extended_scale * tangent
        if return_aux:
            return output, tangent
        if return_source:
            return output, tangent
        return output


def load_extended_nuisance_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ExtendedNuisanceTangent(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DualViewSpeakerEmbedding(NuisanceInvariantTangent):
    """Preserve spectral identity and waveform-source evidence in two subspaces."""

    output_dimension = 384

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.source_projection = nn.Linear(192, 192, bias=False)
        nn.init.eye_(self.source_projection.weight)
        self.source_log_weight = nn.Parameter(
            torch.tensor(-1.3862943611198906)
        )
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        parent, _, source, _ = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_components=True,
        )
        parent_view = F.normalize(parent.float(), dim=1).to(parent.dtype)
        source_direction = F.normalize(source.float(), dim=1).to(parent.dtype)
        source_view = F.normalize(
            self.source_projection(source_direction).float(), dim=1
        ).to(parent.dtype)
        source_weight = torch.exp(self.source_log_weight).clamp(0.02, 1.0)
        output = torch.cat(
            (parent_view, source_weight.to(parent.dtype) * source_view), dim=1
        )
        if return_aux:
            return output, source_view, source_direction, source_weight
        return output


def load_dual_view_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = DualViewSpeakerEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class TemporalConsensusEmbedding(NuisanceInvariantTangent):
    """Join the global embedding with consensus across 0.5-second regions."""

    output_dimension = 384

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.local_projection = nn.Linear(192, 192, bias=False)
        nn.init.eye_(self.local_projection.weight)
        self.local_log_weight = nn.Parameter(torch.tensor(-1.3862943611198906))
        self._consensus_stages = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        self._consensus_stages = stages
        return stages

    def _local_consensus(self, final_stage, lengths):
        final_lengths = self._final_lengths(lengths).clamp(
            min=1, max=final_stage.shape[-1]
        )
        # Fifty input frames are approximately 0.5 seconds. After the three
        # temporal strides in ERes2NetV2 this becomes seven feature frames.
        chunk_steps = 7
        outputs = []
        for index, valid_tensor in enumerate(final_lengths):
            valid = int(valid_tensor.item())
            starts = list(range(0, max(1, valid - chunk_steps + 1), chunk_steps))
            last = max(0, valid - chunk_steps)
            if not starts or starts[-1] != last:
                starts.append(last)
            pieces = []
            for start in starts:
                width = min(chunk_steps, valid - start)
                feature_map = final_stage[
                    index:index + 1, :, :, start:start + width
                ]
                piece_lengths = torch.tensor(
                    [width], dtype=torch.long, device=final_stage.device
                )
                statistics = _masked_statistics(feature_map, piece_lengths)
                pieces.append(F.normalize(
                    self.backbone.seg_1(statistics).float(), dim=1
                ))
            outputs.append(F.normalize(torch.cat(pieces, dim=0).mean(0), dim=0))
        return torch.stack(outputs, dim=0).to(final_stage.dtype)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._consensus_stages = None
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        if self._consensus_stages is None:
            raise RuntimeError("backbone stages were not captured")
        local_direction = self._local_consensus(
            self._consensus_stages[-1], lengths
        )
        self._consensus_stages = None
        local_view = F.normalize(
            self.local_projection(local_direction).float(), dim=1
        ).to(parent.dtype)
        parent_view = F.normalize(parent.float(), dim=1).to(parent.dtype)
        local_weight = torch.exp(self.local_log_weight).clamp(0.02, 1.0)
        output = torch.cat(
            (parent_view, local_weight.to(parent.dtype) * local_view), dim=1
        )
        if return_aux:
            return output, local_view, local_direction, local_weight
        return output


def load_temporal_consensus_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = TemporalConsensusEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class HierarchicalTemporalEmbedding(NuisanceInvariantTangent):
    """Aggregate complete V21 embeddings over half-second regions."""

    output_dimension = 384
    chunk_samples = 8000
    chunk_frames = 48

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.local_projection = nn.Linear(192, 192, bias=False)
        nn.init.eye_(self.local_projection.weight)
        self.local_log_weight = nn.Parameter(torch.tensor(-1.3862943611198906))
        self.to(device)

    @staticmethod
    def _starts(valid, width):
        starts = list(range(0, max(1, valid - width + 1), width))
        last = max(0, valid - width)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def _local_consensus(
        self, global_parent, features, lengths, waveforms, waveform_lengths,
    ):
        if waveform_lengths is None:
            waveform_lengths = torch.full(
                (waveforms.shape[0],), waveforms.shape[1], dtype=torch.long,
                device=waveforms.device,
            )
        # A true short utterance already is a local view. Reusing its complete
        # V21 output makes the hierarchical path exactly preserve V21 for it.
        outputs = [
            F.normalize(global_parent[index].float(), dim=0)
            for index in range(features.shape[0])
        ]
        chunk_features = []
        chunk_waveforms = []
        owners = []
        for index in range(features.shape[0]):
            valid_samples = int(waveform_lengths[index].item())
            valid_frames = int(lengths[index].item())
            if valid_samples <= self.chunk_samples:
                continue
            for sample_start in self._starts(
                valid_samples, self.chunk_samples
            ):
                frame_start = min(
                    max(0, valid_frames - self.chunk_frames),
                    sample_start // 160,
                )
                chunk_features.append(features[
                    index, frame_start:frame_start + self.chunk_frames
                ])
                chunk_waveforms.append(waveforms[
                    index, sample_start:sample_start + self.chunk_samples
                ])
                owners.append(index)
        if chunk_features:
            feature_batch = torch.stack(chunk_features)
            waveform_batch = torch.stack(chunk_waveforms)
            feature_lengths = torch.full(
                (len(chunk_features),), self.chunk_frames, dtype=torch.long,
                device=features.device,
            )
            audio_lengths = torch.full(
                (len(chunk_features),), self.chunk_samples, dtype=torch.long,
                device=features.device,
            )
            with torch.no_grad():
                chunk_embeddings = NuisanceInvariantTangent.forward(
                    self, feature_batch, feature_lengths, waveform_batch,
                    audio_lengths,
                )
                chunk_embeddings = F.normalize(
                    chunk_embeddings.float(), dim=1
                )
            for owner in set(owners):
                indexes = [
                    position for position, value in enumerate(owners)
                    if value == owner
                ]
                outputs[owner] = F.normalize(
                    chunk_embeddings[indexes].mean(0), dim=0
                )
        return torch.stack(outputs).to(global_parent.dtype)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the hierarchical model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        local_direction = self._local_consensus(
            parent, features, lengths, waveforms, waveform_lengths
        )
        local_view = F.normalize(
            self.local_projection(local_direction).float(), dim=1
        ).to(parent.dtype)
        parent_view = F.normalize(parent.float(), dim=1).to(parent.dtype)
        local_weight = torch.exp(self.local_log_weight).clamp(0.02, 1.0)
        output = torch.cat(
            (parent_view, local_weight.to(parent.dtype) * local_view), dim=1
        )
        if return_aux:
            return output, local_view, local_direction, local_weight
        return output


def load_hierarchical_temporal_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = HierarchicalTemporalEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class FrequencyPartitionEmbedding(NuisanceInvariantTangent):
    """Expose low, middle, and high-frequency pretrained identity subspaces."""

    output_dimension = 768
    frequency_bands = ((0, 3), (3, 7), (7, 10))

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.band_projections = nn.ModuleList([
            nn.Linear(192, 192, bias=False) for _ in self.frequency_bands
        ])
        for projection in self.band_projections:
            nn.init.eye_(projection.weight)
        self.band_log_weights = nn.Parameter(
            torch.full((len(self.frequency_bands),), -1.6094379124341003)
        )
        self._frequency_stages = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        self._frequency_stages = stages
        return stages

    def _band_directions(self, final_stage, lengths):
        final_lengths = self._final_lengths(lengths).clamp(
            min=1, max=final_stage.shape[-1]
        )
        statistics = _masked_statistics(final_stage, final_lengths)
        statistics = statistics.reshape(statistics.shape[0], 2048, 10)
        weight = self.backbone.seg_1.weight.reshape(192, 2048, 10)
        bias = self.backbone.seg_1.bias / len(self.frequency_bands)
        bands = []
        for start, stop in self.frequency_bands:
            contribution = torch.einsum(
                "bcf,ocf->bo",
                statistics[:, :, start:stop],
                weight[:, :, start:stop],
            ) + bias
            bands.append(F.normalize(contribution.float(), dim=1))
        return torch.stack(bands, dim=1).to(final_stage.dtype)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._frequency_stages = None
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        if self._frequency_stages is None:
            raise RuntimeError("backbone stages were not captured")
        raw_bands = self._band_directions(
            self._frequency_stages[-1], lengths
        )
        self._frequency_stages = None
        band_views = torch.stack([
            F.normalize(
                projection(raw_bands[:, index]).float(), dim=1
            ).to(parent.dtype)
            for index, projection in enumerate(self.band_projections)
        ], dim=1)
        band_weights = torch.exp(self.band_log_weights).clamp(0.02, 1.0)
        weighted_bands = band_views * band_weights.to(parent.dtype)[None, :, None]
        output = torch.cat(
            (
                F.normalize(parent.float(), dim=1).to(parent.dtype),
                weighted_bands.flatten(1),
            ),
            dim=1,
        )
        if return_aux:
            return output, band_views, raw_bands, band_weights
        return output


def load_frequency_partition_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = FrequencyPartitionEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class FrequencyRoutedTangentEmbedding(FrequencyPartitionEmbedding):
    """Use frequency-local evidence as a quality-routed angular correction.

    The frequency views never enter cosine scoring directly.  A learned router
    combines them, and a zero-initialized tangent expert can only rotate the
    strong V21 parent embedding.  This keeps the initial model exactly equal to
    V21 while allowing stable spectral evidence to correct its direction.
    """

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        # Eight duration/acoustic descriptors plus three pairwise agreements
        # between the frequency-local identity directions.
        self.frequency_router = nn.Sequential(
            nn.Linear(11, 32),
            nn.SiLU(),
            nn.Linear(32, len(self.frequency_bands)),
        )
        nn.init.zeros_(self.frequency_router[-1].weight)
        nn.init.zeros_(self.frequency_router[-1].bias)
        self.frequency_expert = nn.Sequential(
            nn.LayerNorm(192 * 3 + 8),
            nn.Linear(192 * 3 + 8, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.frequency_expert[-1].weight)
        nn.init.zeros_(self.frequency_expert[-1].bias)
        self.frequency_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the frequency router")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._frequency_stages = None
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        if self._frequency_stages is None:
            raise RuntimeError("backbone stages were not captured")
        raw_bands = self._band_directions(
            self._frequency_stages[-1], lengths
        )
        self._frequency_stages = None
        band_views = torch.stack([
            F.normalize(
                projection(raw_bands[:, index]).float(), dim=1
            ).to(parent.dtype)
            for index, projection in enumerate(self.band_projections)
        ], dim=1)
        with torch.no_grad():
            periodic_quality = self._periodic_quality(waveforms)
        quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        agreements = torch.stack(
            (
                F.cosine_similarity(band_views[:, 0], band_views[:, 1]),
                F.cosine_similarity(band_views[:, 0], band_views[:, 2]),
                F.cosine_similarity(band_views[:, 1], band_views[:, 2]),
            ),
            dim=1,
        )
        gates = torch.softmax(
            self.frequency_router(torch.cat((quality, agreements), dim=1)).float(),
            dim=1,
        ).to(parent.dtype)
        global_weights = torch.exp(self.band_log_weights).clamp(0.02, 1.0)
        weighted_gates = gates * global_weights.to(parent.dtype)[None, :]
        spectral_direction = F.normalize(
            (band_views * weighted_gates[:, :, None]).sum(dim=1).float(), dim=1
        ).to(parent.dtype)
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        candidate = self.frequency_expert(torch.cat(
            (
                parent_direction,
                spectral_direction,
                parent_direction * spectral_direction,
                quality,
            ),
            dim=1,
        ))
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.frequency_scale * tangent
        if return_aux:
            return output, tangent, gates, spectral_direction
        return output


def load_frequency_routed_tangent_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = FrequencyRoutedTangentEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ParameterizedCepstralMean(nn.Module):
    """Kernel-initialized, trainable local channel normalization.

    This follows the PCMN form beta * x - alpha * local_mean - bias.  A
    zero-initialized residual mix preserves the pretrained input exactly and
    lets synchronized cross-device recordings determine how much local channel
    normalization is useful.
    """

    def __init__(self, feature_dimension=80, window=21):
        super().__init__()
        self.window = window
        self.beta_log = nn.Parameter(torch.zeros(feature_dimension))
        self.alpha_logit = nn.Parameter(torch.zeros(feature_dimension))
        self.bias = nn.Parameter(torch.zeros(feature_dimension))
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "mean_kernel", torch.ones(feature_dimension, 1, window),
            persistent=False,
        )
        self.register_buffer(
            "count_kernel", torch.ones(1, 1, window), persistent=False
        )

    def forward(self, features, lengths=None, return_mix=False):
        channels = features.transpose(1, 2)
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        steps = torch.arange(features.shape[1], device=features.device)[None, :]
        valid = (steps < lengths[:, None])[:, None, :]
        mask = valid.to(features.dtype)
        padding = self.window // 2
        local_sum = F.conv1d(
            channels * mask,
            self.mean_kernel.to(features.dtype),
            padding=padding,
            groups=channels.shape[1],
        )
        count = F.conv1d(
            mask, self.count_kernel.to(features.dtype), padding=padding
        ).clamp_min(1.0)
        local_mean = local_sum / count
        beta = torch.exp(self.beta_log.clamp(-0.35, 0.35))
        alpha = torch.sigmoid(self.alpha_logit)
        normalized = (
            beta[None, :, None] * channels
            - alpha[None, :, None] * local_mean
            - self.bias[None, :, None]
        )
        mix = torch.tanh(self.mix)
        output = channels + mix * (normalized - channels) * mask
        output = output.transpose(1, 2)
        if return_mix:
            return output, mix, alpha, beta
        return output


class PCMNChannelNormalizedEmbedding(NuisanceInvariantTangent):
    """Place trainable local channel normalization before the V21 encoder."""

    paired_stage = True
    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        # The shared paired-training utility treats this ModuleList as the only
        # authorized adaptation path.
        self.stage_adapters = nn.ModuleList([ParameterizedCepstralMean()])
        self.to(device)

    @property
    def pcmn(self):
        return self.stage_adapters[0]

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        corrected, mix, alpha, beta = self.pcmn(
            features, lengths, return_mix=True
        )
        output = NuisanceInvariantTangent.forward(
            self, corrected, lengths, waveforms, waveform_lengths
        )
        if return_aux:
            return output, corrected - features, mix, alpha, beta
        return output


def load_pcmn_channel_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = PCMNChannelNormalizedEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class PairwiseChannelInvariantEmbedding(NuisanceInvariantTangent):
    """Refine V21 with recording-pair channel invariance.

    The pair discriminators are training-only supervision.  Inference remains
    a single-utterance encoder whose zero-initialized expert produces one
    192-dimensional embedding.
    """

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.pairwise_expert = nn.Sequential(
            nn.LayerNorm(192 * 3 + 8),
            nn.Linear(192 * 3 + 8, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.pairwise_expert[-1].weight)
        nn.init.zeros_(self.pairwise_expert[-1].bias)
        self.pairwise_scale = nn.Parameter(torch.tensor(0.10))
        self.pair_device_head = nn.Sequential(
            nn.Linear(192 * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.pair_distance_head = nn.Sequential(
            nn.Linear(192 * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the pairwise model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        parent, nuisance_tangent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths,
            return_aux=True,
        )
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        nuisance_direction = F.normalize(
            nuisance_tangent.float(), dim=1
        ).to(parent.dtype)
        with torch.no_grad():
            periodic_quality = self._periodic_quality(waveforms)
        quality = torch.cat(
            (self.quality_features(features, lengths), periodic_quality), dim=1
        )
        candidate = self.pairwise_expert(torch.cat(
            (
                parent_direction,
                nuisance_direction,
                parent_direction * nuisance_direction,
                quality,
            ),
            dim=1,
        ))
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.pairwise_scale * tangent
        if return_aux:
            return output, tangent
        return output


def load_pairwise_channel_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = PairwiseChannelInvariantEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class FrequencyPositionSEAdapter(nn.Module):
    """Inject absolute frequency position and utterance-level frequency context."""

    def __init__(self, frequency_bins, reduction=4):
        super().__init__()
        hidden = max(4, frequency_bins // reduction)
        self.position = nn.Parameter(torch.zeros(1, 1, frequency_bins, 1))
        self.excitation = nn.Sequential(
            nn.Linear(frequency_bins, hidden),
            nn.SiLU(),
            nn.Linear(hidden, frequency_bins),
        )
        self.position_scale = nn.Parameter(torch.tensor(0.10))
        self.excitation_scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.excitation[-1].weight)
        nn.init.zeros_(self.excitation[-1].bias)

    def forward(self, x):
        positioned = x + self.position_scale * self.position.to(x.dtype)
        descriptor = positioned.float().mean(dim=(1, 3))
        gate = torch.tanh(self.excitation(descriptor)).to(x.dtype)
        return positioned * (
            1.0 + self.excitation_scale * gate[:, None, :, None]
        )


class FrequencyPositionAwareEmbedding(NuisanceInvariantTangent):
    """Adapt V21 inside the ResNet with frequency position and frequency-wise SE.

    Each stage gets a lightweight frequency adapter.  The zero positional
    vectors and zero final excitation layers make construction exactly equal
    to the V21 parent, while training can change how absolute vocal-tract and
    pitch regions are propagated before utterance pooling.
    """

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.frequency_stage_adapters = nn.ModuleList([
            FrequencyPositionSEAdapter(value) for value in (80, 40, 20, 10)
        ])
        self.to(device)

    def _stages(self, features):
        backbone = self.backbone
        x = features.permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(backbone.bn1(backbone.conv1(x)))
        out1 = self.frequency_stage_adapters[0](backbone.layer1(x))
        out2 = self.frequency_stage_adapters[1](backbone.layer2(out1))
        out3 = self.frequency_stage_adapters[2](backbone.layer3(out2))
        out4 = self.frequency_stage_adapters[3](backbone.layer4(out3))
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        return out1, out2, out3, final


def load_frequency_position_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = FrequencyPositionAwareEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DeepFrameReliability(nn.Module):
    """Estimate a scalar reliability for every deep temporal frame."""

    def __init__(self, frequency_bins=10, hidden=32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(frequency_bins * 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, x, lengths, return_weights=False):
        mean = x.float().mean(dim=1)
        deviation = torch.sqrt(
            (x.float() - mean[:, None]).square().mean(dim=1) + 1e-6
        )
        logits = self.context(torch.cat((mean, deviation), dim=1)).squeeze(1)
        steps = torch.arange(x.shape[-1], device=x.device)[None, :]
        valid = steps < lengths[:, None].clamp(max=x.shape[-1])
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * valid.sum(dim=1, keepdim=True).clamp_min(1)
        weights = torch.where(valid, weights, torch.ones_like(weights))
        output = x * weights.to(x.dtype)[:, None, None, :]
        if return_weights:
            return output, weights
        return output


class DeepFrameAttentiveEmbedding(NuisanceInvariantTangent):
    """Reweight deep ResNet frames before all pretrained pooling paths."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.frame_attention = DeepFrameReliability()
        self._frame_lengths = None
        self._last_frame_weights = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        if self._frame_lengths is None:
            raise RuntimeError("deep-frame lengths were not initialized")
        final, weights = self.frame_attention(
            stages[-1], self._frame_lengths, return_weights=True
        )
        self._last_frame_weights = weights
        return stages[0], stages[1], stages[2], final

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._frame_lengths = self._final_lengths(lengths)
        self._last_frame_weights = None
        try:
            output = NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths,
                return_source=return_source, return_aux=return_aux,
                return_components=return_components,
            )
            return output
        finally:
            self._frame_lengths = None
            self._last_frame_weights = None


def load_deep_frame_attention_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = DeepFrameAttentiveEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ResidualRelaxedTemporalFrequencyNorm(nn.Module):
    """Blend frozen BN with instance temporal/frequency normalization.

    For a C x F x T feature map, temporal normalization estimates one
    distribution per frame over C x F, while frequency normalization estimates
    one distribution per frequency over C x T. A zero initialized per-channel
    residual keeps the released encoder exactly unchanged at initialization.
    """

    def __init__(self, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        self.strength = nn.Parameter(torch.zeros(batch_norm.num_features))

    @staticmethod
    def _normalize(value, dimensions, eps):
        mean = value.mean(dim=dimensions, keepdim=True)
        variance = (value - mean).square().mean(dim=dimensions, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + eps)

    def forward(self, value):
        batch_output = self.batch_norm(value)
        working = value.float()
        temporal = self._normalize(working, (1, 2), self.batch_norm.eps)
        frequency = self._normalize(working, (1, 3), self.batch_norm.eps)
        normalized = 0.5 * (temporal + frequency)
        if self.batch_norm.affine:
            normalized = (
                normalized * self.batch_norm.weight.float()[None, :, None, None]
                + self.batch_norm.bias.float()[None, :, None, None]
            )
        interpolation = torch.tanh(self.strength)[None, :, None, None]
        return batch_output + interpolation.to(batch_output.dtype) * (
            normalized.to(batch_output.dtype) - batch_output
        )


class RelaxedTemporalFrequencyEmbedding(NuisanceInvariantTangent):
    """Apply learnable TN/FN residuals throughout the pretrained backbone."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self._replace_batch_norms(self.backbone)
        self.to(device)

    @classmethod
    def _replace_batch_norms(cls, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(
                    module, name,
                    ResidualRelaxedTemporalFrequencyNorm(child),
                )
            else:
                cls._replace_batch_norms(child)

    @property
    def normalization_adapters(self):
        return [
            module for module in self.backbone.modules()
            if isinstance(module, ResidualRelaxedTemporalFrequencyNorm)
        ]

    def _stages(self, features):
        """Run the normalization graph with stage activation checkpointing."""
        backbone = self.backbone
        use_checkpoint = self.training and any(
            adapter.strength.requires_grad
            for adapter in self.normalization_adapters
        )

        def run(function, *values):
            if use_checkpoint:
                return checkpoint(function, *values, use_reentrant=False)
            return function(*values)

        value = features.permute(0, 2, 1).unsqueeze(1)
        value = run(
            lambda item: torch.relu(backbone.bn1(backbone.conv1(item))),
            value,
        )
        out1 = run(backbone.layer1, value)
        out2 = run(backbone.layer2, out1)
        out3 = run(backbone.layer3, out2)
        out4 = run(backbone.layer4, out3)
        final = run(
            lambda deep, middle: backbone.fuse34(
                deep, backbone.layer3_ds(middle)
            ),
            out4, out3,
        )
        return out1, out2, out3, final


def load_relaxed_normalization_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = RelaxedTemporalFrequencyEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DeepAxialConformer(nn.Module):
    """Model global and local time context independently in each frequency bin."""

    def __init__(self, channels=1024, dimension=128, heads=4, kernel_size=7):
        super().__init__()
        self.input_projection = nn.Conv2d(
            channels, dimension, kernel_size=1, bias=False
        )
        self.first_ffn_norm = nn.LayerNorm(dimension)
        self.first_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.convolution_norm = nn.LayerNorm(dimension)
        self.convolution_input = nn.Conv1d(
            dimension, dimension * 2, kernel_size=1
        )
        self.depthwise_convolution = nn.Conv1d(
            dimension, dimension, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dimension, bias=False,
        )
        self.convolution_output_norm = nn.LayerNorm(dimension)
        self.convolution_output = nn.Conv1d(
            dimension, dimension, kernel_size=1
        )
        self.second_ffn_norm = nn.LayerNorm(dimension)
        self.second_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.output_projection = nn.Conv2d(
            dimension, channels, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _mask(sequence, valid):
        return sequence.masked_fill(~valid[:, :, None], 0.0)

    def forward(self, feature_map, lengths):
        batch, _, frequencies, steps = feature_map.shape
        projected = self.input_projection(feature_map)
        sequence = projected.permute(0, 2, 3, 1).reshape(
            batch * frequencies, steps, projected.shape[1]
        )
        expanded_lengths = lengths[:, None].expand(
            batch, frequencies
        ).reshape(-1).clamp(min=1, max=steps)
        positions = torch.arange(steps, device=feature_map.device)[None, :]
        valid = positions < expanded_lengths[:, None]
        sequence = self._mask(sequence, valid)

        sequence = sequence + 0.5 * self.first_ffn(
            self.first_ffn_norm(sequence)
        )
        sequence = self._mask(sequence, valid)
        attended = self.attention_norm(sequence)
        attended = self.attention(
            attended, attended, attended, key_padding_mask=~valid,
            need_weights=False,
        )[0]
        sequence = self._mask(sequence + attended, valid)

        convolved = self.convolution_norm(sequence).transpose(1, 2)
        convolved = F.glu(self.convolution_input(convolved), dim=1)
        convolved = self.depthwise_convolution(convolved).transpose(1, 2)
        convolved = F.silu(self.convolution_output_norm(convolved))
        convolved = self.convolution_output(
            convolved.transpose(1, 2)
        ).transpose(1, 2)
        sequence = self._mask(sequence + convolved, valid)
        sequence = sequence + 0.5 * self.second_ffn(
            self.second_ffn_norm(sequence)
        )
        sequence = self._mask(self.output_norm(sequence), valid)

        residual = sequence.reshape(
            batch, frequencies, steps, projected.shape[1]
        ).permute(0, 3, 1, 2)
        residual = self.output_projection(residual)
        return feature_map + self.scale * residual


class DeepTemporalConformerEmbedding(NuisanceInvariantTangent):
    """Refine the final ResNet map with a temporal Conformer before pooling."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.temporal_conformer = DeepAxialConformer()
        self._conformer_lengths = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        if self._conformer_lengths is None:
            raise RuntimeError("deep temporal lengths were not initialized")
        final = self.temporal_conformer(
            stages[-1], self._conformer_lengths
        )
        return stages[0], stages[1], stages[2], final

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._conformer_lengths = self._final_lengths(lengths)
        try:
            return NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths,
                return_source=return_source, return_aux=return_aux,
                return_components=return_components,
            )
        finally:
            self._conformer_lengths = None


def load_deep_temporal_conformer_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = DeepTemporalConformerEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class TemporalConformerBlock(nn.Module):
    """A compact masked Conformer block for an utterance-level sequence."""

    def __init__(self, dimension=128, heads=4, kernel_size=15):
        super().__init__()
        self.first_ffn_norm = nn.LayerNorm(dimension)
        self.first_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.convolution_norm = nn.LayerNorm(dimension)
        self.convolution_input = nn.Conv1d(
            dimension, dimension * 2, kernel_size=1
        )
        self.depthwise_convolution = nn.Conv1d(
            dimension, dimension, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dimension, bias=False,
        )
        self.convolution_output_norm = nn.LayerNorm(dimension)
        self.convolution_output = nn.Conv1d(
            dimension, dimension, kernel_size=1
        )
        self.second_ffn_norm = nn.LayerNorm(dimension)
        self.second_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )
        self.output_norm = nn.LayerNorm(dimension)

    @staticmethod
    def _mask(sequence, valid):
        return sequence.masked_fill(~valid[:, :, None], 0.0)

    def forward(self, sequence, valid):
        sequence = self._mask(sequence, valid)
        sequence = sequence + 0.5 * self.first_ffn(
            self.first_ffn_norm(sequence)
        )
        sequence = self._mask(sequence, valid)
        attended = self.attention_norm(sequence)
        attended = self.attention(
            attended, attended, attended, key_padding_mask=~valid,
            need_weights=False,
        )[0]
        sequence = self._mask(sequence + attended, valid)
        convolved = self.convolution_norm(sequence).transpose(1, 2)
        convolved = F.glu(self.convolution_input(convolved), dim=1)
        convolved = self.depthwise_convolution(convolved).transpose(1, 2)
        convolved = F.silu(self.convolution_output_norm(convolved))
        convolved = self.convolution_output(
            convolved.transpose(1, 2)
        ).transpose(1, 2)
        sequence = self._mask(sequence + convolved, valid)
        sequence = sequence + 0.5 * self.second_ffn(
            self.second_ffn_norm(sequence)
        )
        return self._mask(self.output_norm(sequence), valid)


class MultiScaleTemporalConformer(nn.Module):
    """Fuse four ResNet depths before masked global/local time modeling."""

    def __init__(self, stage_channels=(128, 256, 512, 1024)):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
            )
            for channels in stage_channels
        ])
        self.stage_projection = nn.ModuleList([
            nn.Conv1d(32 * 4, 64, kernel_size=1, bias=False)
            for _ in stage_channels
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(64 * len(stage_channels), 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.conformer = TemporalConformerBlock()
        self.output = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, stages, lengths):
        target_steps = stages[0].shape[-1]
        projected = []
        for stage, lateral, projection in zip(
            stages, self.lateral, self.stage_projection
        ):
            stage = lateral(stage)
            stage = F.adaptive_avg_pool2d(
                stage, (4, stage.shape[-1])
            ).flatten(1, 2)
            stage = projection(stage)
            if stage.shape[-1] != target_steps:
                stage = F.interpolate(
                    stage, size=target_steps, mode="linear",
                    align_corners=False,
                )
            projected.append(stage)
        sequence = self.fusion(torch.cat(projected, dim=1)).transpose(1, 2)
        valid_lengths = lengths.clamp(min=1, max=target_steps)
        positions = torch.arange(target_steps, device=sequence.device)[None, :]
        valid = positions < valid_lengths[:, None]
        sequence = self.conformer(sequence, valid)
        mask = valid.to(sequence.dtype)[:, :, None]
        count = valid_lengths.to(sequence.dtype)[:, None].clamp_min(1)
        mean = (sequence * mask).sum(dim=1) / count
        variance = (
            (sequence - mean[:, None]).square() * mask
        ).sum(dim=1) / count
        statistics = torch.cat(
            (mean, torch.sqrt(variance + 1e-5)), dim=1
        )
        return self.scale * self.output(statistics)


class MultiScaleConformerEmbedding(NuisanceInvariantTangent):
    """Add a four-depth Conformer speaker residual to the V21 embedding."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.multiscale_conformer = MultiScaleTemporalConformer()
        self._captured_stages = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        self._captured_stages = stages
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._captured_stages = None
        try:
            parent = NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths
            )
            if self._captured_stages is None:
                raise RuntimeError("multi-scale stages were not captured")
            candidate = self.multiscale_conformer(
                self._captured_stages, lengths
            )
            direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
            tangent = candidate - (
                candidate * direction
            ).sum(dim=1, keepdim=True) * direction
            output = parent + tangent
            if return_components:
                return output, tangent, candidate, parent
            if return_aux or return_source:
                return output, tangent
            return output
        finally:
            self._captured_stages = None


def load_multiscale_conformer_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = MultiScaleConformerEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DeepTimeFrequencyConformer(nn.Module):
    """Apply Conformer context along time and then along deep frequency bins."""

    def __init__(self, channels=1024, dimension=128, frequency_bins=10):
        super().__init__()
        self.input_projection = nn.Conv2d(
            channels, dimension, kernel_size=1, bias=False
        )
        self.time_block = TemporalConformerBlock(
            dimension=dimension, heads=4, kernel_size=7
        )
        self.frequency_position = nn.Parameter(
            torch.empty(1, frequency_bins, dimension)
        )
        self.frequency_block = TemporalConformerBlock(
            dimension=dimension, heads=4, kernel_size=5
        )
        self.output_projection = nn.Conv2d(
            dimension, channels, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.normal_(self.frequency_position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, feature_map, lengths):
        batch, _, frequencies, steps = feature_map.shape
        if frequencies > self.frequency_position.shape[1]:
            raise ValueError("deep frequency dimension exceeds configured size")
        projected = self.input_projection(feature_map)
        valid_lengths = lengths.clamp(min=1, max=steps)
        positions = torch.arange(steps, device=feature_map.device)[None, :]
        time_valid = positions < valid_lengths[:, None]

        time_sequence = projected.permute(0, 2, 3, 1).reshape(
            batch * frequencies, steps, projected.shape[1]
        )
        repeated_valid = time_valid[:, None, :].expand(
            batch, frequencies, steps
        ).reshape(batch * frequencies, steps)
        time_sequence = self.time_block(time_sequence, repeated_valid)
        grid = time_sequence.reshape(
            batch, frequencies, steps, projected.shape[1]
        )

        frequency_sequence = grid.permute(0, 2, 1, 3).reshape(
            batch * steps, frequencies, projected.shape[1]
        )
        frequency_sequence = frequency_sequence + self.frequency_position[
            :, :frequencies
        ].to(frequency_sequence.dtype)
        frequency_valid = torch.ones(
            batch * steps, frequencies, dtype=torch.bool,
            device=feature_map.device,
        )
        frequency_sequence = self.frequency_block(
            frequency_sequence, frequency_valid
        )
        grid = frequency_sequence.reshape(
            batch, steps, frequencies, projected.shape[1]
        ).permute(0, 2, 1, 3)
        grid = grid.masked_fill(
            ~time_valid[:, None, :, None], 0.0
        )
        residual = grid.permute(0, 3, 1, 2)
        return feature_map + self.scale * self.output_projection(residual)


class DeepTimeFrequencyConformerEmbedding(NuisanceInvariantTangent):
    """Refine the final V21 map with axial time-frequency Conformer blocks."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.time_frequency_conformer = DeepTimeFrequencyConformer()
        self._axial_lengths = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        if self._axial_lengths is None:
            raise RuntimeError("axial deep lengths were not initialized")
        final = self.time_frequency_conformer(
            stages[-1], self._axial_lengths
        )
        return stages[0], stages[1], stages[2], final

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._axial_lengths = self._final_lengths(lengths)
        try:
            return NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths,
                return_source=return_source, return_aux=return_aux,
                return_components=return_components,
            )
        finally:
            self._axial_lengths = None


def load_deep_time_frequency_conformer_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = DeepTimeFrequencyConformerEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class PitchConditionedFrameAdapter(nn.Module):
    """Condition deep spectral frames on multi-resolution periodic evidence."""

    def __init__(self, channels=1024, dimension=128):
        super().__init__()
        self.spectral_projection = nn.Conv2d(
            channels, dimension, kernel_size=1, bias=False
        )
        self.pitch_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.lag_attention = nn.Conv2d(64, 1, kernel_size=1)
        self.pitch_projection = nn.Conv1d(
            64, dimension, kernel_size=1, bias=False
        )
        self.spectral_norm = nn.LayerNorm(dimension)
        self.pitch_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, 4, dropout=0.0, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )
        self.output = nn.Conv1d(
            dimension, channels * 2, kernel_size=1
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature_map, lengths, autocorrelation):
        steps = feature_map.shape[-1]
        spectral = self.spectral_projection(feature_map).mean(dim=2)
        pitch_map = self.pitch_encoder(autocorrelation.to(feature_map.dtype))
        lag_weights = torch.softmax(
            self.lag_attention(pitch_map).float(), dim=2
        ).to(pitch_map.dtype)
        pitch = (pitch_map * lag_weights).sum(dim=2)
        pitch = self.pitch_projection(pitch)
        pitch = F.interpolate(
            pitch, size=steps, mode="linear", align_corners=False
        )

        valid_lengths = lengths.clamp(min=1, max=steps)
        positions = torch.arange(steps, device=feature_map.device)[None, :]
        valid = positions < valid_lengths[:, None]
        spectral_sequence = spectral.transpose(1, 2).masked_fill(
            ~valid[:, :, None], 0.0
        )
        pitch_sequence = pitch.transpose(1, 2).masked_fill(
            ~valid[:, :, None], 0.0
        )
        attended = self.cross_attention(
            self.spectral_norm(spectral_sequence),
            self.pitch_norm(pitch_sequence),
            self.pitch_norm(pitch_sequence),
            key_padding_mask=~valid,
            need_weights=False,
        )[0]
        fused = self.fusion(torch.cat((spectral_sequence, attended), dim=2))
        fused = fused.masked_fill(~valid[:, :, None], 0.0)
        gain, bias = self.output(fused.transpose(1, 2)).chunk(2, dim=1)
        residual = (
            torch.tanh(gain)[:, :, None, :] * feature_map
            + bias[:, :, None, :]
        )
        residual = residual.masked_fill(~valid[:, None, None, :], 0.0)
        return feature_map + self.scale * residual


class PitchConditionedDeepEmbedding(NuisanceInvariantTangent):
    """Use raw periodic sequences to refine spectral frames before pooling."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.pitch_frame_adapter = PitchConditionedFrameAdapter()
        self._pitch_waveforms = None
        self._pitch_lengths = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        if self._pitch_waveforms is None or self._pitch_lengths is None:
            raise RuntimeError("pitch-conditioned inputs were not initialized")
        with torch.no_grad():
            autocorrelation = self.source_encoder.autocorrelation_features(
                self._pitch_waveforms
            )
        final = self.pitch_frame_adapter(
            stages[-1], self._pitch_lengths, autocorrelation
        )
        return stages[0], stages[1], stages[2], final

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the pitch-conditioned model")
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._pitch_waveforms = waveforms
        self._pitch_lengths = self._final_lengths(lengths)
        try:
            return NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths,
                return_source=return_source, return_aux=return_aux,
                return_components=return_components,
            )
        finally:
            self._pitch_waveforms = None
            self._pitch_lengths = None


def load_pitch_conditioned_deep_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = PitchConditionedDeepEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ResidualConv2dLowRank(nn.Module):
    """Add a trainable low-rank convolution while preserving a frozen Conv2d."""

    def __init__(self, convolution, rank=8):
        super().__init__()
        self.base = convolution
        self.low_rank_down = nn.Conv2d(
            convolution.in_channels, rank,
            kernel_size=convolution.kernel_size,
            stride=convolution.stride,
            padding=convolution.padding,
            dilation=convolution.dilation,
            bias=False,
            padding_mode=convolution.padding_mode,
        )
        self.low_rank_up = nn.Conv2d(
            rank, convolution.out_channels, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.kaiming_normal_(self.low_rank_down.weight, nonlinearity="linear")
        nn.init.zeros_(self.low_rank_up.weight)

    def forward(self, value):
        residual = self.low_rank_up(F.silu(self.low_rank_down(value)))
        return self.base(value) + self.scale * residual


class DeepConvolutionLoRAEmbedding(NuisanceInvariantTangent):
    """Adapt the last ResNet stage and fusion using low-rank convolutions."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        for module in (
            self.backbone.layer4,
            self.backbone.layer3_ds,
            self.backbone.fuse34,
        ):
            self._replace_convolutions(module)
        if not self.convolution_adapters:
            raise RuntimeError("no deep convolutions were available for LoRA")
        self.to(device)

    @classmethod
    def _replace_convolutions(cls, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, ResidualConv2dLowRank(child))
            else:
                cls._replace_convolutions(child)

    @property
    def convolution_adapters(self):
        return [
            module for module in self.backbone.modules()
            if isinstance(module, ResidualConv2dLowRank)
        ]


def load_deep_convolution_lora_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = DeepConvolutionLoRAEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ExcitationConditionedFrameAdapter(PitchConditionedFrameAdapter):
    """Condition frames on lag-normalized, envelope-whitened excitation only."""

    def __init__(self, channels=1024, dimension=128):
        super().__init__(channels=channels, dimension=dimension)
        self.pitch_encoder[0] = nn.Conv2d(
            3, 32, kernel_size=(5, 3), padding=(2, 1), bias=False
        )

    def forward(self, feature_map, lengths, autocorrelation):
        excitation = autocorrelation[:, 1::2].float()
        mean = excitation.mean(dim=2, keepdim=True)
        variance = (
            excitation - mean
        ).square().mean(dim=2, keepdim=True)
        excitation = (
            (excitation - mean) * torch.rsqrt(variance + 1e-5)
        ).clamp(-5.0, 5.0)
        return super().forward(feature_map, lengths, excitation)


class ExcitationConditionedDeepEmbedding(PitchConditionedDeepEmbedding):
    """Use channel-reduced excitation trajectories to refine deep frames."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        self.pitch_frame_adapter = ExcitationConditionedFrameAdapter()
        self.to(device)


def load_excitation_conditioned_deep_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ExcitationConditionedDeepEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class PyramidGhostVLAD(nn.Module):
    """Dictionary pooling with ghost clusters over the frozen feature pyramid."""

    def __init__(self, channels=320, clusters=6, ghost_clusters=2):
        super().__init__()
        self.clusters = clusters
        self.total_clusters = clusters + ghost_clusters
        self.frame_norm = nn.LayerNorm(channels)
        self.assignment = nn.Sequential(
            nn.Conv1d(channels * 3, 192, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(192, self.total_clusters, kernel_size=1),
        )
        self.centers = nn.Parameter(
            torch.empty(self.total_clusters, channels)
        )
        nn.init.normal_(self.centers, std=0.02)
        self.output = nn.Sequential(
            nn.LayerNorm(clusters * channels),
            nn.Linear(clusters * channels, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _pyramid_sequence(stages, lengths, pyramid):
        target_steps = stages[0].shape[-1]
        projected = []
        descriptors = []
        for stage, lateral, projection in zip(
            stages, pyramid.lateral, pyramid.projections
        ):
            stage = lateral(stage)
            stage = F.adaptive_avg_pool2d(
                stage, (4, stage.shape[-1])
            ).flatten(1, 2)
            stage = projection(stage)
            stage = F.interpolate(
                stage, size=target_steps, mode="linear",
                align_corners=False,
            )
            projected.append(stage)
            descriptors.append(pyramid._masked_descriptor(stage, lengths))
        duration = lengths.to(projected[0].dtype)
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        level_features = torch.cat(
            (torch.stack(descriptors, dim=1), duration_features), dim=1
        )
        level_weights = torch.softmax(
            pyramid.level_gate(level_features).float(), dim=1
        ).to(projected[0].dtype)
        return sum(
            value * level_weights[:, index, None, None]
            for index, value in enumerate(projected)
        )

    def forward(self, stages, lengths, pyramid):
        frames = self._pyramid_sequence(stages, lengths, pyramid)
        frames = self.frame_norm(frames.transpose(1, 2)).transpose(1, 2)
        steps = frames.shape[-1]
        valid_lengths = lengths.clamp(min=1, max=steps)
        positions = torch.arange(steps, device=frames.device)[None, :]
        valid = positions < valid_lengths[:, None]
        mask = valid.to(frames.dtype)[:, None, :]
        count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (frames * mask).sum(dim=2, keepdim=True) / count
        variance = (
            (frames - mean).square() * mask
        ).sum(dim=2, keepdim=True) / count
        context = torch.cat(
            (
                frames,
                mean.expand_as(frames),
                torch.sqrt(variance + 1e-5).expand_as(frames),
            ),
            dim=1,
        )
        logits = self.assignment(context).masked_fill(
            ~valid[:, None, :], -1e4
        )
        weights = torch.softmax(logits.float(), dim=1).to(frames.dtype)
        weights = weights * mask
        residuals = (
            weights[:, :, None, :]
            * (frames[:, None, :, :] - self.centers[None, :, :, None])
        ).sum(dim=3)
        residuals = F.normalize(residuals[:, :self.clusters].float(), dim=2)
        descriptor = F.normalize(residuals.flatten(1), dim=1).to(frames.dtype)
        return self.output(descriptor)


class GhostVLADPyramidEmbedding(NuisanceInvariantTangent):
    """Refine V21 using high-resolution multi-scale dictionary pooling."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.ghost_vlad = PyramidGhostVLAD()
        self.ghost_scale = nn.Parameter(torch.tensor(0.10))
        self._ghost_stages = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        self._ghost_stages = stages
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._ghost_stages = None
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        try:
            if self._ghost_stages is None:
                raise RuntimeError("pyramid stages were not captured")
            candidate = self.ghost_vlad(
                self._ghost_stages, lengths, self.pyramid
            )
        finally:
            self._ghost_stages = None
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.ghost_scale * tangent
        if return_components:
            return output, tangent, parent
        if return_aux:
            return output, tangent
        if return_source:
            return output, tangent
        return output


def load_ghost_vlad_pyramid_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = GhostVLADPyramidEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ReliableLocalStatistics(nn.Module):
    """Pool dictionary-local moments without amplifying scarcely visited cells."""

    def __init__(self, channels=320, clusters=6, ghost_clusters=2):
        super().__init__()
        self.clusters = clusters
        self.total_clusters = clusters + ghost_clusters
        self.channels = channels
        self.frame_norm = nn.LayerNorm(channels)
        self.assignment = nn.Sequential(
            nn.Conv1d(channels * 3, 192, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(192, self.total_clusters, kernel_size=1),
        )
        self.centers = nn.Parameter(torch.empty(self.total_clusters, channels))
        nn.init.normal_(self.centers, std=0.02)
        # softplus(log_tau) is the number of effective frames needed for a
        # local cell to approach full reliability.
        self.log_tau = nn.Parameter(torch.tensor(1.8545866))
        descriptor_dimension = clusters * channels * 2 + clusters * 3
        self.output = nn.Sequential(
            nn.LayerNorm(descriptor_dimension),
            nn.Linear(descriptor_dimension, 512),
            nn.SiLU(),
            nn.Linear(512, 192),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def initialize_dictionary(self, state):
        """Reuse V43's learned frame partition while replacing its pooling."""
        own = self.state_dict()
        with torch.no_grad():
            for name in (
                "frame_norm.weight", "frame_norm.bias",
                "assignment.0.weight", "assignment.0.bias",
                "assignment.2.weight", "assignment.2.bias", "centers",
            ):
                own[name].copy_(state["ghost_vlad." + name])
        self.load_state_dict(own, strict=True)

    def forward(self, stages, lengths, pyramid):
        frames = PyramidGhostVLAD._pyramid_sequence(stages, lengths, pyramid)
        frames = self.frame_norm(frames.transpose(1, 2)).transpose(1, 2)
        frames = frames.float()
        steps = frames.shape[-1]
        valid_lengths = lengths.clamp(min=1, max=steps)
        positions = torch.arange(steps, device=frames.device)[None, :]
        valid = positions < valid_lengths[:, None]
        mask = valid.float()[:, None, :]
        count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        global_mean = (frames * mask).sum(dim=2, keepdim=True) / count
        global_variance = (
            (frames - global_mean).square() * mask
        ).sum(dim=2, keepdim=True) / count
        context = torch.cat(
            (
                frames,
                global_mean.expand_as(frames),
                torch.sqrt(global_variance + 1e-5).expand_as(frames),
            ),
            dim=1,
        )
        logits = self.assignment(context.to(self.assignment[0].weight.dtype))
        logits = logits.float().masked_fill(~valid[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=1) * mask
        active = weights[:, :self.clusters]
        mass = active.sum(dim=2).clamp_min(1e-5)
        residual = (
            frames[:, None, :, :]
            - self.centers[:self.clusters].float()[None, :, :, None]
        )
        local_mean = (
            active[:, :, None, :] * residual
        ).sum(dim=3) / mass[:, :, None]
        local_variance = (
            active[:, :, None, :]
            * (residual - local_mean[:, :, :, None]).square()
        ).sum(dim=3) / mass[:, :, None]

        squared_mass = active.square().sum(dim=2).clamp_min(1e-6)
        effective_frames = mass.square() / squared_mass
        occupancy = mass / count.squeeze(2)
        coverage = (occupancy * self.total_clusters).clamp(max=1.0)
        tau = F.softplus(self.log_tau).clamp_min(0.25)
        confidence = effective_frames / (effective_frames + tau)
        reliability = torch.sqrt((coverage * confidence).clamp_min(1e-6))

        mean_direction = F.normalize(local_mean, dim=2)
        local_std = torch.sqrt(local_variance + 1e-5)
        log_std = torch.log(local_std + 1e-3)
        dispersion_shape = F.layer_norm(log_std, (self.channels,))
        local_descriptor = torch.cat(
            (mean_direction, dispersion_shape), dim=2
        ) * reliability[:, :, None]
        auxiliary = torch.stack(
            (
                torch.log(occupancy * self.total_clusters + 1e-5),
                torch.log(local_std.mean(dim=2) + 1e-3),
                confidence,
            ),
            dim=2,
        )
        descriptor = torch.cat(
            (local_descriptor.flatten(1), auxiliary.flatten(1)), dim=1
        )
        return self.output(descriptor.to(self.output[1].weight.dtype))


class ReliableLocalStatisticsEmbedding(NuisanceInvariantTangent):
    """Refine V21 with occupancy-aware local mean and variance evidence."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.local_statistics = ReliableLocalStatistics()
        self.local_statistics_scale = nn.Parameter(torch.tensor(0.10))
        self._local_stages = None
        self.to(device)

    def _stages(self, features):
        stages = super()._stages(features)
        self._local_stages = stages
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._local_stages = None
        parent = NuisanceInvariantTangent.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        try:
            if self._local_stages is None:
                raise RuntimeError("pyramid stages were not captured")
            candidate = self.local_statistics(
                self._local_stages, lengths, self.pyramid
            )
        finally:
            self._local_stages = None
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.local_statistics_scale * tangent
        if return_components:
            return output, tangent, parent
        if return_aux:
            return output, tangent
        if return_source:
            return output, tangent
        return output


def load_reliable_local_statistics_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ReliableLocalStatisticsEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DurationConditionedFeatureCompensator(nn.Module):
    """Predict a channel-wise short-to-full correction from observed moments."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.channels = channels
        self.conditioner = nn.Sequential(
            nn.LayerNorm(channels * 2 + 2),
            nn.Linear(channels * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),
        )
        self.strength = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

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
        condition = torch.cat((mean, torch.log(std + 1e-3), duration_features), dim=1)
        affine = self.conditioner(
            condition.to(self.conditioner[1].weight.dtype)
        ).float()
        scale, bias = affine.chunk(2, dim=1)
        strength = self.strength.float()
        corrected = value.float() * (
            1.0 + strength * torch.tanh(scale)[:, :, None, None]
        ) + strength * bias[:, :, None, None]
        return corrected.to(value.dtype)


class LayerwiseShortToFullCompensation(NuisanceInvariantTangent):
    """Compensate duration bias at every encoder stage before pooling."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, device="cpu",
        )
        state = torch.load(str(nuisance_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.feature_compensators = nn.ModuleList([
            DurationConditionedFeatureCompensator(128, 64),
            DurationConditionedFeatureCompensator(256, 96),
            DurationConditionedFeatureCompensator(512, 128),
            DurationConditionedFeatureCompensator(1024, 192),
        ])
        self._compensation_lengths = None
        self._compensated_stages = None
        self.to(device)

    @staticmethod
    def _stage_lengths(lengths):
        outputs = [lengths]
        current = lengths
        for _ in range(3):
            current = torch.div(current + 1, 2, rounding_mode="floor")
            outputs.append(current)
        return outputs

    @staticmethod
    def stage_statistics(stages, lengths):
        descriptors = []
        for stage, valid_lengths in zip(
            stages, LayerwiseShortToFullCompensation._stage_lengths(lengths)
        ):
            mean, std = DurationConditionedFeatureCompensator.channel_statistics(
                stage, valid_lengths
            )
            descriptors.append(torch.cat((
                F.layer_norm(mean, (mean.shape[1],)),
                F.layer_norm(torch.log(std + 1e-3), (std.shape[1],)),
            ), dim=1))
        return descriptors

    def _stages(self, features):
        if self._compensation_lengths is None:
            raise RuntimeError("compensation lengths were not initialized")
        backbone = self.backbone
        stage_lengths = self._stage_lengths(self._compensation_lengths)
        value = features.permute(0, 2, 1).unsqueeze(1)
        value = torch.relu(backbone.bn1(backbone.conv1(value)))
        out1 = self.feature_compensators[0](
            backbone.layer1(value), stage_lengths[0]
        )
        out2 = self.feature_compensators[1](
            backbone.layer2(out1), stage_lengths[1]
        )
        out3 = self.feature_compensators[2](
            backbone.layer3(out2), stage_lengths[2]
        )
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        final = self.feature_compensators[3](final, stage_lengths[3])
        self._compensated_stages = (out1, out2, out3, final)
        return self._compensated_stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_stage_statistics=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._compensation_lengths = lengths
        self._compensated_stages = None
        try:
            output = NuisanceInvariantTangent.forward(
                self, features, lengths, waveforms, waveform_lengths
            )
            if self._compensated_stages is None:
                raise RuntimeError("compensated stages were not captured")
            statistics = (
                self.stage_statistics(self._compensated_stages, lengths)
                if return_stage_statistics or return_aux or return_source
                else None
            )
        finally:
            self._compensation_lengths = None
            self._compensated_stages = None
        if return_stage_statistics:
            return output, statistics
        if return_aux or return_source:
            return output, statistics[-1]
        return output


def load_layerwise_compensation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = LayerwiseShortToFullCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ParametricSincFilterBank(nn.Module):
    """Compact learnable band-pass front end with physically ordered filters."""

    def __init__(
        self, filters=64, kernel_size=251, sample_rate=16000,
        minimum_low_hz=30.0, minimum_band_hz=50.0,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.minimum_low_hz = minimum_low_hz
        self.minimum_band_hz = minimum_band_hz
        minimum_mel = 2595.0 * torch.log10(torch.tensor(1.0 + 30.0 / 700.0))
        maximum_mel = 2595.0 * torch.log10(
            torch.tensor(1.0 + (sample_rate / 2 - 100.0) / 700.0)
        )
        mel_edges = torch.linspace(minimum_mel, maximum_mel, filters + 1)
        hz_edges = 700.0 * (torch.pow(10.0, mel_edges / 2595.0) - 1.0)
        self.low_hz = nn.Parameter(hz_edges[:-1] - minimum_low_hz)
        self.band_hz = nn.Parameter(
            hz_edges[1:] - hz_edges[:-1] - minimum_band_hz
        )
        positions = torch.arange(kernel_size) - (kernel_size - 1) / 2
        self.register_buffer("positions", positions, persistent=True)
        self.register_buffer(
            "window", torch.hamming_window(kernel_size, periodic=False),
            persistent=True,
        )

    def filters_tensor(self):
        low = self.minimum_low_hz + self.low_hz.abs()
        high = (
            low + self.minimum_band_hz + self.band_hz.abs()
        ).clamp(max=self.sample_rate / 2 - 30.0)
        time = self.positions[None, :] / self.sample_rate
        low = low[:, None]
        high = high[:, None]
        filters = (
            2.0 * high / self.sample_rate
            * torch.sinc(2.0 * high * time)
            - 2.0 * low / self.sample_rate
            * torch.sinc(2.0 * low * time)
        ) * self.window[None, :]
        return F.normalize(filters.float(), dim=1).unsqueeze(1)

    def forward(self, waveform):
        return F.conv1d(
            waveform[:, None].float(), self.filters_tensor(),
            stride=80, padding=self.kernel_size // 2,
        )


class RawSincResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels, input_channels, kernel_size=5, stride=2,
                padding=2, groups=input_channels, bias=False,
            ),
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv1d(
                output_channels, output_channels, kernel_size=5, padding=2,
                groups=output_channels, bias=False,
            ),
            nn.Conv1d(output_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=1, stride=2,
                bias=False,
            ),
            nn.GroupNorm(8, output_channels),
        )
        self.scale = nn.Sequential(
            nn.Linear(output_channels * 2, max(24, output_channels // 4)),
            nn.SiLU(),
            nn.Linear(max(24, output_channels // 4), output_channels),
            nn.Sigmoid(),
        )

    def forward(self, value):
        output = F.silu(self.main(value) + self.skip(value))
        mean = output.mean(dim=2)
        std = torch.sqrt(
            (output - mean[:, :, None]).square().mean(dim=2) + 1e-5
        )
        gate = self.scale(torch.cat((mean, std), dim=1))
        return output * (0.5 + gate[:, :, None])


class SincSpeakerResidualEncoder(nn.Module):
    """Extract narrow-band pitch and formant evidence directly from samples."""

    output_dimension = 192

    def __init__(self):
        super().__init__()
        self.filter_bank = ParametricSincFilterBank()
        self.blocks = nn.Sequential(
            RawSincResidualBlock(64, 96),
            RawSincResidualBlock(96, 128),
            RawSincResidualBlock(128, 192),
        )
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    192, 192, kernel_size=5, padding=2 * dilation,
                    dilation=dilation, groups=192, bias=False,
                ),
                nn.Conv1d(192, 192, kernel_size=1, bias=False),
                nn.GroupNorm(8, 192),
                nn.SiLU(),
            )
            for dilation in (1, 3, 9)
        ])
        self.attention = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 384),
            nn.SiLU(),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _output_lengths(waveform_lengths):
        lengths = torch.div(
            waveform_lengths + 79, 80, rounding_mode="floor"
        )
        for _ in range(3):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths

    def forward(self, waveforms, waveform_lengths=None):
        if waveform_lengths is None:
            waveform_lengths = torch.full(
                (waveforms.shape[0],), waveforms.shape[1], dtype=torch.long,
                device=waveforms.device,
            )
        positions = torch.arange(
            waveforms.shape[1], device=waveforms.device
        )[None, :]
        valid = positions < waveform_lengths[:, None]
        mask = valid.float()
        count = waveform_lengths.float().clamp_min(1.0)[:, None]
        working = waveforms.float()
        mean = (working * mask).sum(dim=1, keepdim=True) / count
        variance = (
            (working - mean).square() * mask
        ).sum(dim=1, keepdim=True) / count
        working = (working - mean) * torch.rsqrt(variance + 1e-6)
        working = torch.cat(
            (working[:, :1], working[:, 1:] - 0.97 * working[:, :-1]), dim=1
        )
        filtered = self.filter_bank(working)
        value = torch.log(filtered.square() + 1e-5)
        value = value - value.mean(dim=(1, 2), keepdim=True)
        value = value / (
            value.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-5
        )
        value = self.blocks(value)
        value = value + sum(branch(value) for branch in self.temporal_branches) / 3.0
        lengths = self._output_lengths(waveform_lengths).clamp(
            min=1, max=value.shape[2]
        )
        steps = torch.arange(value.shape[2], device=value.device)[None, :]
        valid = steps < lengths[:, None]
        logits = self.attention(value).squeeze(1).float().masked_fill(
            ~valid, -1e4
        )
        weights = torch.softmax(logits, dim=1).to(value.dtype)[:, None, :]
        mean = (value * weights).sum(dim=2)
        variance = (weights * (value - mean[:, :, None]).square()).sum(dim=2)
        statistics = torch.cat((mean, torch.sqrt(variance + 1e-5)), dim=1)
        return self.output(statistics)


class SincWaveformResidualEmbedding(LayerwiseShortToFullCompensation):
    """Refine V45 with a compact learnable raw-waveform SincNet branch."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        state = torch.load(str(layerwise_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.sinc_encoder = SincSpeakerResidualEncoder()
        self.sinc_scale = nn.Parameter(torch.tensor(0.10))
        self.to(device)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False, return_components=False,
    ):
        if waveforms is None:
            raise ValueError("waveforms are required by the Sinc residual model")
        parent = LayerwiseShortToFullCompensation.forward(
            self, features, lengths, waveforms, waveform_lengths
        )
        candidate = self.sinc_encoder(waveforms, waveform_lengths)
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.sinc_scale * tangent
        if return_components:
            return output, tangent, parent
        if return_aux or return_source:
            return output, tangent
        return output


def load_sinc_residual_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, model_checkpoint=None, device="cuda",
):
    model = SincWaveformResidualEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, layerwise_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class DurationRoutedFeatureCompensator(DurationConditionedFeatureCompensator):
    """Route an already learned compensation continuously by duration."""

    def __init__(self, channels, hidden):
        super().__init__(channels, hidden)
        self.router = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

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
        gate = 2.0 * torch.sigmoid(
            self.router(duration_features.to(self.router[0].weight.dtype))
        ).float()
        strength = self.strength.float() * gate
        corrected = value.float() * (
            1.0 + strength[:, :, None, None]
            * torch.tanh(scale)[:, :, None, None]
        ) + strength[:, :, None, None] * bias[:, :, None, None]
        return corrected.to(value.dtype)


class DurationRoutedLayerwiseCompensation(LayerwiseShortToFullCompensation):
    """Add explicit continuous duration routing to the effective V45 adapters."""

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        state = torch.load(str(layerwise_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        routed = nn.ModuleList([
            DurationRoutedFeatureCompensator(128, 64),
            DurationRoutedFeatureCompensator(256, 96),
            DurationRoutedFeatureCompensator(512, 128),
            DurationRoutedFeatureCompensator(1024, 192),
        ])
        with torch.no_grad():
            for target, source in zip(routed, self.feature_compensators):
                target.conditioner.load_state_dict(source.conditioner.state_dict())
                target.strength.copy_(source.strength)
        self.feature_compensators = routed
        self.to(device)

    @property
    def duration_routers(self):
        return [adapter.router for adapter in self.feature_compensators]


def load_duration_routed_compensation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, model_checkpoint=None, device="cuda",
):
    model = DurationRoutedLayerwiseCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, layerwise_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class FrequencyMomentResidual(nn.Module):
    """Recover duration-biased frequency moments after channel compensation."""

    def __init__(self, frequency_bins, hidden):
        super().__init__()
        self.frequency_bins = frequency_bins
        self.conditioner = nn.Sequential(
            nn.LayerNorm(frequency_bins * 2 + 2),
            nn.Linear(frequency_bins * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, frequency_bins * 2),
        )
        self.strength = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

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
    """Keep V45's channel correction and add an orthogonal frequency path."""

    def __init__(self, channel_compensator, frequency_bins, hidden):
        super().__init__()
        self.channel_compensator = channel_compensator
        self.frequency_extension = FrequencyMomentResidual(
            frequency_bins, hidden
        )

    def forward(self, value, lengths):
        value = self.channel_compensator(value, lengths)
        return self.frequency_extension(value, lengths)


class LayerwiseChannelFrequencyCompensation(LayerwiseShortToFullCompensation):
    """Extend V45 with same-file short-to-full frequency-moment recovery."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        state = torch.load(str(layerwise_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.feature_compensators = nn.ModuleList([
            ChannelFrequencyFeatureCompensator(channel, bins, hidden)
            for channel, bins, hidden in zip(
                self.feature_compensators,
                (80, 40, 20, 10),
                (64, 48, 32, 24),
            )
        ])
        self.to(device)

    @property
    def frequency_extensions(self):
        return [
            adapter.frequency_extension
            for adapter in self.feature_compensators
        ]

    @staticmethod
    def stage_statistics(stages, lengths):
        descriptors = []
        for stage, valid_lengths in zip(
            stages, LayerwiseShortToFullCompensation._stage_lengths(lengths)
        ):
            channel_mean, channel_std = (
                DurationConditionedFeatureCompensator.channel_statistics(
                    stage, valid_lengths
                )
            )
            frequency_mean, frequency_std = (
                FrequencyMomentResidual.frequency_statistics(
                    stage, valid_lengths
                )
            )
            descriptors.append(torch.cat((
                F.layer_norm(channel_mean, (channel_mean.shape[1],)),
                F.layer_norm(
                    torch.log(channel_std + 1e-3),
                    (channel_std.shape[1],),
                ),
                F.layer_norm(frequency_mean, (frequency_mean.shape[1],)),
                F.layer_norm(
                    torch.log(frequency_std + 1e-3),
                    (frequency_std.shape[1],),
                ),
            ), dim=1))
        return descriptors


def load_channel_frequency_compensation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, model_checkpoint=None, device="cuda",
):
    model = LayerwiseChannelFrequencyCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class SpectralContrastMomentResidual(FrequencyMomentResidual):
    """Recover local spectral contrast while rejecting smooth channel tilt."""

    @staticmethod
    def local_contrast(value):
        width = value.shape[1]
        if width < 3:
            return value - value.mean(dim=1, keepdim=True)
        kernel = 5 if width >= 5 else 3
        padding = kernel // 2
        padded = F.pad(
            value[:, None, :], (padding, padding), mode="replicate"
        )
        smooth = F.avg_pool1d(padded, kernel_size=kernel, stride=1).squeeze(1)
        return value - smooth

    @classmethod
    def contrast_statistics(cls, value, lengths):
        mean, std = cls.frequency_statistics(value, lengths)
        return cls.local_contrast(mean), cls.local_contrast(
            torch.log(std + 1e-3)
        )

    def forward(self, value, lengths):
        mean_contrast, std_contrast = self.contrast_statistics(value, lengths)
        duration = lengths.float()
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        condition = torch.cat(
            (mean_contrast, std_contrast, duration_features), dim=1
        )
        affine = self.conditioner(
            condition.to(self.conditioner[1].weight.dtype)
        ).float()
        scale, bias = affine.chunk(2, dim=1)
        # The correction itself is also high-pass along frequency, preventing
        # this path from recreating broad device or room spectral envelopes.
        scale = self.local_contrast(scale)
        bias = self.local_contrast(bias)
        strength = self.strength.float()
        corrected = value.float() * (
            1.0 + strength * torch.tanh(scale)[:, None, :, None]
        ) + strength * bias[:, None, :, None]
        return corrected.to(value.dtype)


class ChannelSpectralContrastFeatureCompensator(nn.Module):
    def __init__(self, channel_compensator, frequency_bins, hidden):
        super().__init__()
        self.channel_compensator = channel_compensator
        self.spectral_extension = SpectralContrastMomentResidual(
            frequency_bins, hidden
        )

    def forward(self, value, lengths):
        value = self.channel_compensator(value, lengths)
        return self.spectral_extension(value, lengths)


class LayerwiseSpectralContrastCompensation(LayerwiseShortToFullCompensation):
    """Extend V45 with channel-robust local spectral contrast recovery."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, device="cpu",
        )
        state = torch.load(str(layerwise_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.feature_compensators = nn.ModuleList([
            ChannelSpectralContrastFeatureCompensator(channel, bins, hidden)
            for channel, bins, hidden in zip(
                self.feature_compensators,
                (80, 40, 20, 10),
                (64, 48, 32, 24),
            )
        ])
        self.to(device)

    @property
    def spectral_extensions(self):
        return [
            adapter.spectral_extension
            for adapter in self.feature_compensators
        ]

    @staticmethod
    def stage_statistics(stages, lengths):
        descriptors = []
        for stage, valid_lengths in zip(
            stages, LayerwiseShortToFullCompensation._stage_lengths(lengths)
        ):
            channel_mean, channel_std = (
                DurationConditionedFeatureCompensator.channel_statistics(
                    stage, valid_lengths
                )
            )
            frequency_mean, frequency_std = (
                SpectralContrastMomentResidual.contrast_statistics(
                    stage, valid_lengths
                )
            )
            descriptors.append(torch.cat((
                F.layer_norm(channel_mean, (channel_mean.shape[1],)),
                F.layer_norm(
                    torch.log(channel_std + 1e-3),
                    (channel_std.shape[1],),
                ),
                F.layer_norm(frequency_mean, (frequency_mean.shape[1],)),
                F.layer_norm(frequency_std, (frequency_std.shape[1],)),
            ), dim=1))
        return descriptors


def load_spectral_contrast_compensation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, model_checkpoint=None, device="cuda",
):
    model = LayerwiseSpectralContrastCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class LocalPhoneticResidual(nn.Module):
    """Learn content-dependent local time-frequency corrections."""

    def __init__(self, channels, bottleneck):
        super().__init__()
        self.input_projection = nn.Conv2d(
            channels, bottleneck, kernel_size=1, bias=False
        )
        self.duration_projection = nn.Linear(2, bottleneck)
        self.local_branches = nn.ModuleList([
            nn.Conv2d(
                bottleneck, bottleneck, kernel_size=(3, 5),
                padding=(1, 2 * dilation), dilation=(1, dilation),
                groups=bottleneck, bias=False,
            )
            for dilation in (1, 2)
        ])
        self.mix = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=1, bias=False
        )
        self.output = nn.Conv2d(
            bottleneck, channels, kernel_size=1, bias=False
        )
        self.strength = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.output.weight)

    def forward(self, value, lengths):
        steps = torch.arange(value.shape[-1], device=value.device)[None, :]
        valid = steps < lengths[:, None].clamp(max=value.shape[-1])
        mask = valid.to(value.dtype)[:, None, None, :]
        duration = lengths.float()
        duration_features = torch.stack(
            (torch.log1p(duration) / 6.0, torch.rsqrt(duration)), dim=1
        )
        hidden = self.input_projection(value * mask)
        condition = self.duration_projection(
            duration_features.to(self.duration_projection.weight.dtype)
        )[:, :, None, None]
        hidden = F.silu(hidden + condition)
        local = sum(branch(hidden) for branch in self.local_branches) / 2.0
        local = F.silu(self.mix(local))
        residual = self.output(local) * mask
        return value + self.strength.to(value.dtype) * residual


class LocalPhoneticFeatureCompensator(nn.Module):
    def __init__(self, parent_compensator, channels, bottleneck):
        super().__init__()
        self.parent_compensator = parent_compensator
        self.local_extension = LocalPhoneticResidual(channels, bottleneck)

    def forward(self, value, lengths):
        value = self.parent_compensator(value, lengths)
        return self.local_extension(value, lengths)


class LayerwiseLocalPhoneticCompensation(LayerwiseChannelFrequencyCompensation):
    """Extend V48 with phonetic-local corrections at every encoder stage."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, frequency_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, layerwise_checkpoint,
            device="cpu",
        )
        state = torch.load(str(frequency_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.feature_compensators = nn.ModuleList([
            LocalPhoneticFeatureCompensator(parent, channels, bottleneck)
            for parent, channels, bottleneck in zip(
                self.feature_compensators,
                (128, 256, 512, 1024),
                (24, 32, 48, 64),
            )
        ])
        self.to(device)

    @property
    def local_extensions(self):
        return [
            adapter.local_extension
            for adapter in self.feature_compensators
        ]


def load_local_phonetic_compensation_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, frequency_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = LayerwiseLocalPhoneticCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, frequency_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class SerializedInputAwareLayer(nn.Module):
    """One input-aware attentive-statistics head in a serialized stack."""

    def __init__(self, dimension=192, key_dimension=64, expansion=384):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.key = nn.Linear(dimension, key_dimension, bias=False)
        self.query = nn.Linear(dimension * 2, key_dimension)
        self.head = nn.Linear(dimension * 2, dimension)
        self.context_update = nn.Linear(dimension, dimension, bias=False)
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, expansion),
            nn.SiLU(),
            nn.Linear(expansion, dimension),
        )
        self.key_scale = key_dimension ** -0.5

    @staticmethod
    def statistics(sequence, valid):
        weights = valid.to(sequence.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (sequence * weights[:, :, None]).sum(dim=1) / count
        variance = (
            (sequence - mean[:, None]).square() * weights[:, :, None]
        ).sum(dim=1) / count
        return mean, torch.sqrt(variance + 1e-5)

    def forward(self, sequence, valid):
        normalized = self.attention_norm(sequence)
        global_mean, global_std = self.statistics(normalized, valid)
        query = self.query(torch.cat((global_mean, global_std), dim=1))
        keys = self.key(normalized)
        logits = torch.einsum("btd,bd->bt", keys, query) * self.key_scale
        logits = logits.masked_fill(~valid, -1e4)
        attention = torch.softmax(logits.float(), dim=1).to(sequence.dtype)
        attentive_mean = (sequence * attention[:, :, None]).sum(dim=1)
        attentive_variance = (
            (sequence - attentive_mean[:, None]).square()
            * attention[:, :, None]
        ).sum(dim=1)
        attentive_std = torch.sqrt(attentive_variance + 1e-5)
        head = self.head(torch.cat((attentive_mean, attentive_std), dim=1))

        sequence = sequence + self.context_update(attentive_mean)[:, None]
        sequence = sequence + self.feed_forward(
            self.feed_forward_norm(sequence)
        )
        sequence = sequence.masked_fill(~valid[:, :, None], 0.0)
        return sequence, head, attention


class SerializedInputAwarePool(nn.Module):
    """Aggregate deep frames with successive utterance-conditioned heads."""

    def __init__(
        self, input_channels=1024, frequency_bins=10, dimension=192,
        compressed_channels=32, layers=4,
    ):
        super().__init__()
        self.input_projection = nn.Conv2d(
            input_channels, compressed_channels, kernel_size=1, bias=False
        )
        self.frame_projection = nn.Conv1d(
            compressed_channels * frequency_bins,
            dimension,
            kernel_size=1,
            bias=False,
        )
        self.frame_norm = nn.LayerNorm(dimension)
        self.layers = nn.ModuleList([
            SerializedInputAwareLayer(dimension=dimension)
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature_map, lengths, return_attention=False):
        sequence = self.input_projection(feature_map).flatten(1, 2)
        sequence = self.frame_projection(sequence).transpose(1, 2)
        sequence = F.silu(self.frame_norm(sequence))
        positions = torch.arange(
            sequence.shape[1], device=sequence.device
        )[None, :]
        valid = positions < lengths[:, None].clamp(
            min=1, max=sequence.shape[1]
        )
        sequence = sequence.masked_fill(~valid[:, :, None], 0.0)
        heads = []
        attentions = []
        for layer in self.layers:
            sequence, head, attention = layer(sequence, valid)
            heads.append(head)
            attentions.append(attention)
        aggregate = torch.stack(heads, dim=0).mean(dim=0)
        output = self.output(self.output_norm(aggregate))
        if return_attention:
            return output, torch.stack(attentions, dim=1)
        return output


class SerializedInputAwareCompensation(LayerwiseChannelFrequencyCompensation):
    """Add deep serialized input-aware pooling to the frozen V48 encoder."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
        source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
        layerwise_checkpoint, frequency_checkpoint, device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            source_checkpoint, excitation_checkpoint,
            multiresolution_checkpoint, source_gate_checkpoint,
            tangent_checkpoint, nuisance_checkpoint, layerwise_checkpoint,
            device="cpu",
        )
        state = torch.load(str(frequency_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.serialized_pool = SerializedInputAwarePool()
        self.serialized_scale = nn.Parameter(torch.tensor(0.05))
        self._serialized_stage = None
        self.to(device)

    @property
    def serialized_modules(self):
        return (self.serialized_pool, self.serialized_scale)

    def _stages(self, features):
        stages = super()._stages(features)
        self._serialized_stage = stages[-1]
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_source=False, return_aux=False,
        return_stage_statistics=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._serialized_stage = None
        try:
            parent, statistics = super().forward(
                features, lengths, waveforms, waveform_lengths,
                return_stage_statistics=True,
            )
            if self._serialized_stage is None:
                raise RuntimeError("serialized stage was not captured")
            candidate = self.serialized_pool(
                self._serialized_stage, self._final_lengths(lengths)
            )
        finally:
            self._serialized_stage = None
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.serialized_scale.to(parent.dtype) * tangent
        if return_stage_statistics:
            return output, statistics
        if return_aux or return_source:
            return output, tangent
        return output


def load_serialized_input_aware_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    source_checkpoint, excitation_checkpoint, multiresolution_checkpoint,
    source_gate_checkpoint, tangent_checkpoint, nuisance_checkpoint,
    layerwise_checkpoint, frequency_checkpoint, model_checkpoint=None,
    device="cuda",
):
    model = SerializedInputAwareCompensation(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        source_checkpoint, excitation_checkpoint,
        multiresolution_checkpoint, source_gate_checkpoint,
        tangent_checkpoint, nuisance_checkpoint, layerwise_checkpoint,
        frequency_checkpoint, device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class SerializedInputAwareQualityEmbedding(QualityGatedPyramid):
    """Add serialized deep-frame aggregation directly to the trusted V8."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.serialized_pool = SerializedInputAwarePool()
        self.serialized_scale = nn.Parameter(torch.tensor(0.05))
        self._serialized_stage = None
        self.to(device)

    @property
    def serialized_modules(self):
        return (self.serialized_pool, self.serialized_scale)

    def _stages(self, features):
        stages = super()._stages(features)
        self._serialized_stage = stages[-1]
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._serialized_stage = None
        try:
            parent = QualityGatedPyramid.forward(self, features, lengths)
            if self._serialized_stage is None:
                raise RuntimeError("serialized stage was not captured")
            candidate = self.serialized_pool(
                self._serialized_stage, self._final_lengths(lengths)
            )
        finally:
            self._serialized_stage = None
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.serialized_scale.to(parent.dtype) * tangent
        if return_aux:
            return output, tangent
        return output


def load_serialized_quality_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = SerializedInputAwareQualityEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class QualityLayerwiseShortRestoration(QualityGatedPyramid):
    """Restore short-utterance stage moments directly over the trusted V8."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.feature_compensators = nn.ModuleList([
            DurationConditionedFeatureCompensator(128, 64),
            DurationConditionedFeatureCompensator(256, 96),
            DurationConditionedFeatureCompensator(512, 128),
            DurationConditionedFeatureCompensator(1024, 192),
        ])
        self._restoration_lengths = None
        self._restored_stages = None
        self.to(device)

    @staticmethod
    def _stage_lengths(lengths):
        outputs = [lengths]
        current = lengths
        for _ in range(3):
            current = torch.div(current + 1, 2, rounding_mode="floor")
            outputs.append(current)
        return outputs

    @staticmethod
    def stage_statistics(stages, lengths):
        descriptors = []
        for stage, valid_lengths in zip(
            stages, QualityLayerwiseShortRestoration._stage_lengths(lengths)
        ):
            mean, std = DurationConditionedFeatureCompensator.channel_statistics(
                stage, valid_lengths
            )
            descriptors.append(torch.cat((
                F.layer_norm(mean, (mean.shape[1],)),
                F.layer_norm(torch.log(std + 1e-3), (std.shape[1],)),
            ), dim=1))
        return descriptors

    def _stages(self, features):
        if self._restoration_lengths is None:
            raise RuntimeError("restoration lengths were not initialized")
        backbone = self.backbone
        stage_lengths = self._stage_lengths(self._restoration_lengths)
        value = features.permute(0, 2, 1).unsqueeze(1)
        value = torch.relu(backbone.bn1(backbone.conv1(value)))
        out1 = self.feature_compensators[0](
            backbone.layer1(value), stage_lengths[0]
        )
        out2 = self.feature_compensators[1](
            backbone.layer2(out1), stage_lengths[1]
        )
        out3 = self.feature_compensators[2](
            backbone.layer3(out2), stage_lengths[2]
        )
        out4 = backbone.layer4(out3)
        final = backbone.fuse34(out4, backbone.layer3_ds(out3))
        final = self.feature_compensators[3](final, stage_lengths[3])
        self._restored_stages = (out1, out2, out3, final)
        return self._restored_stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_stage_statistics=False, return_aux=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._restoration_lengths = lengths
        self._restored_stages = None
        try:
            output = QualityGatedPyramid.forward(self, features, lengths)
            if self._restored_stages is None:
                raise RuntimeError("restored stages were not captured")
            statistics = (
                self.stage_statistics(self._restored_stages, lengths)
                if return_stage_statistics or return_aux else None
            )
        finally:
            self._restoration_lengths = None
            self._restored_stages = None
        if return_stage_statistics:
            return output, statistics
        if return_aux:
            return output, statistics[-1]
        return output


def load_quality_layerwise_restoration_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = QualityLayerwiseShortRestoration(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class LocalTemporalResidualBlock(nn.Module):
    """Mix nearby deep frames without utterance-conditioned parameters."""

    def __init__(self, dimension=192, kernels=(3, 7, 15)):
        super().__init__()
        self.input_norm = nn.LayerNorm(dimension)
        self.depthwise = nn.ModuleList([
            nn.Conv1d(
                dimension, dimension, kernel_size=kernel,
                padding=kernel // 2, groups=dimension, bias=False,
            )
            for kernel in kernels
        ])
        self.mix = nn.Conv1d(
            dimension * len(kernels), dimension, kernel_size=1, bias=False
        )
        self.local_scale = nn.Parameter(torch.tensor(0.10))
        self.output_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )
        self.feed_forward_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, sequence, valid):
        mask = valid[:, :, None].to(sequence.dtype)
        normalized = self.input_norm(sequence) * mask
        normalized = normalized.transpose(1, 2)
        local = torch.cat([
            F.silu(convolution(normalized))
            for convolution in self.depthwise
        ], dim=1)
        local = self.mix(local).transpose(1, 2)
        sequence = sequence + torch.tanh(self.local_scale) * local
        sequence = sequence * mask
        feed_forward = self.feed_forward(self.output_norm(sequence))
        sequence = sequence + torch.tanh(
            self.feed_forward_scale
        ) * feed_forward
        return sequence * mask


class LocalTemporalPyramidPool(nn.Module):
    """Pool locally normalized identity dynamics at several time scales."""

    def __init__(
        self, input_channels=1024, frequency_bins=10, dimension=192,
        compressed_channels=32, blocks=2,
    ):
        super().__init__()
        self.input_projection = nn.Conv2d(
            input_channels, compressed_channels, kernel_size=1, bias=False
        )
        self.frame_projection = nn.Conv1d(
            compressed_channels * frequency_bins, dimension,
            kernel_size=1, bias=False,
        )
        self.frame_norm = nn.LayerNorm(dimension)
        self.blocks = nn.ModuleList([
            LocalTemporalResidualBlock(dimension=dimension)
            for _ in range(blocks)
        ])
        self.output_norm = nn.LayerNorm(dimension * 2)
        self.output = nn.Linear(dimension * 2, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature_map, lengths):
        sequence = self.input_projection(feature_map).flatten(1, 2)
        sequence = self.frame_projection(sequence).transpose(1, 2)
        sequence = self.frame_norm(sequence)
        positions = torch.arange(
            sequence.shape[1], device=sequence.device
        )[None, :]
        valid = positions < lengths[:, None].clamp(
            min=1, max=sequence.shape[1]
        )
        mask = valid[:, :, None].to(sequence.dtype)
        # Normalize each frame independently, so the local filters cannot use
        # utterance-level energy or channel moments as a routing signal.
        sequence = F.layer_norm(sequence, (sequence.shape[-1],)) * mask
        for block in self.blocks:
            sequence = block(sequence, valid)
        count = valid.sum(dim=1).clamp_min(1).to(sequence.dtype)[:, None]
        mean = (sequence * mask).sum(dim=1) / count
        variance = (
            (sequence - mean[:, None]).square() * mask
        ).sum(dim=1) / count
        statistics = torch.cat((mean, torch.sqrt(variance + 1e-5)), dim=1)
        return self.output(self.output_norm(statistics))


class QualityLocalTemporalEmbedding(QualityGatedPyramid):
    """Add a local multi-scale temporal identity residual to V8."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.local_temporal_pool = LocalTemporalPyramidPool()
        self.local_temporal_scale = nn.Parameter(torch.tensor(0.05))
        self._local_temporal_stage = None
        self.to(device)

    @property
    def trainable_adapter_modules(self):
        return (self.local_temporal_pool, self.local_temporal_scale)

    def _stages(self, features):
        stages = super()._stages(features)
        self._local_temporal_stage = stages[-1]
        return stages

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_aux=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._local_temporal_stage = None
        try:
            parent = QualityGatedPyramid.forward(self, features, lengths)
            if self._local_temporal_stage is None:
                raise RuntimeError("local temporal stage was not captured")
            candidate = self.local_temporal_pool(
                self._local_temporal_stage, self._final_lengths(lengths)
            )
        finally:
            self._local_temporal_stage = None
        parent_direction = F.normalize(parent.float(), dim=1).to(parent.dtype)
        tangent = candidate - (
            candidate * parent_direction
        ).sum(dim=1, keepdim=True) * parent_direction
        output = parent + self.local_temporal_scale.to(parent.dtype) * tangent
        if return_aux:
            return output, tangent
        return output


def load_quality_local_temporal_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = QualityLocalTemporalEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class BoundedQualityRefinement(QualityGatedPyramid):
    """Refine V8's existing pyramid gate using channel-normalized dynamics."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.quality_refinement = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, 192),
        )
        nn.init.zeros_(self.quality_refinement[-1].weight)
        nn.init.zeros_(self.quality_refinement[-1].bias)
        self.to(device)

    @property
    def trainable_adapter_modules(self):
        return (self.quality_refinement,)

    def _refinement_strength(self, lengths):
        return torch.ones_like(lengths, dtype=torch.float32)

    @staticmethod
    def invariant_quality_features(features, lengths):
        working = features.float()
        steps = torch.arange(working.shape[1], device=working.device)[None, :]
        valid = steps < lengths[:, None]
        mask = valid[:, :, None].float()
        count = (valid.sum(dim=1) * working.shape[2]).clamp_min(1).float()
        magnitude = (working.abs() * mask).sum(dim=(1, 2)) / count

        first = working[:, 1:] - working[:, :-1]
        valid_first = (steps[:, 1:] < lengths[:, None])[:, :, None]
        first_count = (
            valid_first[:, :, 0].sum(dim=1) * working.shape[2]
        ).clamp_min(1).float()
        first_energy = (
            first.abs() * valid_first.float()
        ).sum(dim=(1, 2)) / first_count

        second = first[:, 1:] - first[:, :-1]
        valid_second = (steps[:, 2:] < lengths[:, None])[:, :, None]
        second_count = (
            valid_second[:, :, 0].sum(dim=1) * working.shape[2]
        ).clamp_min(1).float()
        second_energy = (
            second.abs() * valid_second.float()
        ).sum(dim=(1, 2)) / second_count

        frame_energy = torch.sqrt(working.square().mean(dim=2) + 1e-6)
        frame_count = valid.sum(dim=1).clamp_min(1).float()
        frame_mean = (frame_energy * valid.float()).sum(dim=1) / frame_count
        frame_variance = (
            (frame_energy - frame_mean[:, None]).square() * valid.float()
        ).sum(dim=1) / frame_count
        variation = torch.sqrt(frame_variance + 1e-6) / (
            frame_mean.abs() + 1e-3
        )
        duration = lengths.float()
        return torch.stack((
            torch.log1p(duration) / 6.0,
            torch.rsqrt(duration.clamp_min(1)) * 4.0,
            torch.log1p(first_energy / (magnitude + 1e-3)),
            torch.log1p(second_energy / (magnitude + 1e-3)),
            torch.log1p(variation),
        ), dim=1)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_gate=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        stages = self._stages(features)
        base_stats = _masked_statistics(
            stages[-1], self._final_lengths(lengths)
        )
        base_embedding = self.backbone.seg_1(base_stats)
        adaptation = self.pyramid(stages, lengths)
        original_delta = self.quality_gate(
            self.quality_features(features, lengths)
        )
        original_multiplier = torch.exp(0.5 * torch.tanh(original_delta))
        refinement_delta = self.quality_refinement(
            self.invariant_quality_features(features, lengths).to(
                self.quality_refinement[1].weight.dtype
            )
        )
        multiplier = original_multiplier * torch.exp(
            0.20 * self._refinement_strength(lengths)[:, None].to(
                refinement_delta.dtype
            ) * torch.tanh(refinement_delta)
        )
        output = base_embedding + self.adapter_scale * multiplier * adaptation
        if return_gate:
            return output, multiplier
        return output


def load_bounded_quality_refinement_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = BoundedQualityRefinement(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ShortOnlyQualityRefinement(BoundedQualityRefinement):
    """Confine the V8 gate refinement to statistically short utterances."""

    def _refinement_strength(self, lengths):
        # Use inverse square-root sample uncertainty: one at about 0.5 s,
        # zero from about 2 s onward, and a smooth transition in between.
        frames = lengths.float().clamp_min(1)
        lower = frames.new_tensor(50.0).rsqrt()
        upper = frames.new_tensor(200.0).rsqrt()
        return ((frames.rsqrt() - upper) / (lower - upper)).clamp(0.0, 1.0)


def load_short_only_quality_refinement_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ShortOnlyQualityRefinement(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ShortReliabilityReweighting(QualityGatedPyramid):
    """Reweight globally stable embedding dimensions for short utterances."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.short_reliability_logits = nn.Parameter(torch.zeros(192))
        self.to(device)

    @property
    def trainable_adapter_modules(self):
        # set_trainable accepts modules; ParameterList keeps this single
        # global reliability vector compatible with the shared trainer.
        return (nn.ParameterList((self.short_reliability_logits,)),)

    @staticmethod
    def _short_strength(lengths):
        frames = lengths.float().clamp_min(1)
        lower = frames.new_tensor(50.0).rsqrt()
        upper = frames.new_tensor(200.0).rsqrt()
        return ((frames.rsqrt() - upper) / (lower - upper)).clamp(0.0, 1.0)

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_reliability=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        parent = QualityGatedPyramid.forward(self, features, lengths)
        logits = self.short_reliability_logits
        # Removing the mean prevents a uniform scale, which cosine scoring
        # would discard, and leaves only relative dimension reliability.
        centered = logits - logits.mean()
        multiplier = torch.exp(
            0.20 * self._short_strength(lengths)[:, None].to(parent.dtype)
            * torch.tanh(centered).to(parent.dtype)[None, :]
        )
        output = parent * multiplier
        if return_reliability:
            return output, multiplier
        return output


def load_short_reliability_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ShortReliabilityReweighting(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)


class ShortFrameAttentiveEmbedding(QualityGatedPyramid):
    """Use learned deep-frame reliability only when temporal evidence is scarce."""

    output_dimension = 192

    def __init__(
        self, backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cuda",
    ):
        super().__init__(
            backbone_checkpoint, pyramid_checkpoint, device="cpu"
        )
        parent = load_quality_model(
            backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
            device="cpu",
        )
        self.load_state_dict(parent.state_dict(), strict=True)
        self.short_frame_attention = DeepFrameReliability()
        self._short_frame_lengths = None
        self._short_input_lengths = None
        self._last_short_frame_weights = None
        self.to(device)

    @property
    def trainable_adapter_modules(self):
        return (self.short_frame_attention,)

    @staticmethod
    def _short_strength(lengths):
        frames = lengths.float().clamp_min(1)
        lower = frames.new_tensor(50.0).rsqrt()
        upper = frames.new_tensor(200.0).rsqrt()
        return ((frames.rsqrt() - upper) / (lower - upper)).clamp(0.0, 1.0)

    def _stages(self, features):
        stages = super()._stages(features)
        if self._short_frame_lengths is None or self._short_input_lengths is None:
            raise RuntimeError("short-frame lengths were not initialized")
        _, weights = self.short_frame_attention(
            stages[-1], self._short_frame_lengths, return_weights=True
        )
        strength = self._short_strength(self._short_input_lengths)[:, None]
        effective = 1.0 + strength.to(weights.dtype) * (weights - 1.0)
        self._last_short_frame_weights = effective
        final = stages[-1] * effective.to(stages[-1].dtype)[:, None, None, :]
        return stages[0], stages[1], stages[2], final

    def forward(
        self, features, lengths=None, waveforms=None, waveform_lengths=None,
        return_attention=False,
    ):
        if lengths is None:
            lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long,
                device=features.device,
            )
        self._short_input_lengths = lengths
        self._short_frame_lengths = self._final_lengths(lengths)
        self._last_short_frame_weights = None
        try:
            output = QualityGatedPyramid.forward(self, features, lengths)
            if return_attention:
                return output, self._last_short_frame_weights
            return output
        finally:
            self._short_input_lengths = None
            self._short_frame_lengths = None
            self._last_short_frame_weights = None


def load_short_frame_attention_model(
    backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
    model_checkpoint=None, device="cuda",
):
    model = ShortFrameAttentiveEmbedding(
        backbone_checkpoint, pyramid_checkpoint, quality_checkpoint,
        device="cpu",
    )
    if model_checkpoint:
        state = torch.load(str(model_checkpoint), map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    return model.to(device)
