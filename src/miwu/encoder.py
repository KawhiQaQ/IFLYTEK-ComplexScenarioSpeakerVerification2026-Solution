"""ERes2NetV2 loading and feature extraction shared by training and evaluation."""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi


ROOT = Path(__file__).resolve().parents[2]
THREED_SPEAKER = ROOT / "vendor" / "3D-Speaker"
if str(THREED_SPEAKER) not in sys.path:
    sys.path.insert(0, str(THREED_SPEAKER))

from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2  # noqa: E402
from speakerlab.models.eres2net.ERes2Net import ERes2Net  # noqa: E402
from speakerlab.models.eres2net.fusion import AFF  # noqa: E402
from speakerlab.models.campplus.DTDNN import CAMPPlus  # noqa: E402
from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import (  # noqa: E402
    ECAPA_TDNN as ECAPATDNN,
)
from speakerlab.models.rdino.ECAPA_TDNN import (  # noqa: E402
    ECAPA_TDNN as RDINOECAPATDNN,
)
from speakerlab.models.resnet.ResNet import ResNet  # noqa: E402
from speakerlab.models.res2net.Res2Net import Res2Net  # noqa: E402
from speakerlab.models.eres2net.ERes2Net_huge import (  # noqa: E402
    BasicBlockERes2Net,
    BasicBlockERes2Net_diff_AFF,
    ERes2Net as ERes2NetLarge,
)


class LargeBlock(BasicBlockERes2Net):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=32, scale=2)


class LargeFuseBlock(BasicBlockERes2Net_diff_AFF):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=32, scale=2)


class AugBlock(BasicBlockERes2Net):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=24, scale=3)


class AugFuseBlock(BasicBlockERes2Net_diff_AFF):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride, baseWidth=24, scale=3)


def build_encoder(checkpoint, device="cuda", architecture="eres2netv2"):
    if architecture == "eres2netv2":
        model = ERes2NetV2(
            feat_dim=80,
            embedding_size=192,
            m_channels=64,
            baseWidth=26,
            scale=2,
            expansion=2,
        )
    elif architecture == "eres2netv2_w24s4ep4":
        model = ERes2NetV2(
            feat_dim=80,
            embedding_size=192,
            m_channels=64,
            baseWidth=24,
            scale=4,
            expansion=4,
        )
    elif architecture in ("eres2net_large", "eres2net_large_192"):
        model = ERes2NetLarge(
            block=LargeBlock,
            block_fuse=LargeFuseBlock,
            feat_dim=80,
            embedding_size=(
                192 if architecture == "eres2net_large_192" else 512
            ),
            m_channels=64,
        )
        # The released 3D-Speaker checkpoint uses the earlier expansion=2
        # topology, while the current repository constructor hard-codes the
        # expansion=4 fusion widths. Restore the published checkpoint widths.
        model.layer1_downsample = nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False)
        model.layer2_downsample = nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=False)
        model.layer3_downsample = nn.Conv2d(512, 1024, 3, padding=1, stride=2, bias=False)
        model.fuse_mode12 = AFF(channels=256)
        model.fuse_mode123 = AFF(channels=512)
        model.fuse_mode1234 = AFF(channels=1024)
    elif architecture == "eres2net_base":
        model = ERes2Net(
            feat_dim=80,
            embedding_size=512,
            m_channels=32,
        )
    elif architecture == "eres2net_aug":
        model = ERes2NetLarge(
            block=AugBlock,
            block_fuse=AugFuseBlock,
            feat_dim=80,
            embedding_size=192,
            m_channels=64,
        )
    elif architecture in ("campplus", "campplus_512"):
        model = CAMPPlus(
            feat_dim=80,
            embedding_size=512 if architecture == "campplus_512" else 192,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
            memory_efficient=True,
        )
    elif architecture == "ecapa_tdnn":
        model = ECAPATDNN(
            input_size=80,
            lin_neurons=192,
            channels=[1024, 1024, 1024, 1024, 3072],
        )
    elif architecture == "rdino_ecapa":
        model = RDINOECAPATDNN(
            input_size=80,
            lin_neurons=512,
            channels=[1024, 1024, 1024, 1024, 3072],
        )
    elif architecture == "resnet34_3d":
        model = ResNet(
            feat_dim=80,
            embedding_size=192,
            m_channels=32,
            pooling_func="TSTP",
            two_emb_layer=True,
        )
    elif architecture == "res2net_3d":
        model = Res2Net(
            feat_dim=80,
            embedding_size=192,
            m_channels=32,
            pooling_func="TSTP",
            two_emb_layer=False,
        )
    else:
        raise ValueError("Unsupported architecture: " + architecture)
    state = torch.load(str(checkpoint), map_location="cpu")
    if architecture == "rdino_ecapa":
        state = {
            key[len("module.backbone.") :]: value
            for key, value in state["teacher"].items()
            if key.startswith("module.backbone.")
        }
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_audio(path, sample_rate=16000):
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform[:1].float()
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.squeeze(0)


def crop_audio(waveform, seconds, key="", sample_rate=16000):
    """Crop deterministically by utterance key without consulting trial labels."""
    if seconds is None:
        return waveform
    wanted = int(round(seconds * sample_rate))
    if waveform.numel() <= wanted:
        return waveform
    # A stable start prevents favorable random-crop selection during evaluation.
    import hashlib

    digest = hashlib.sha256((str(key) + ":" + str(seconds)).encode()).digest()
    start = int.from_bytes(digest[:8], "little") % (waveform.numel() - wanted + 1)
    return waveform[start : start + wanted]


def pad_short(waveform, minimum_seconds=3.0, sample_rate=16000):
    wanted = int(round(minimum_seconds * sample_rate))
    if waveform.numel() < wanted:
        waveform = F.pad(waveform, (0, wanted - waveform.numel()))
    return waveform


def fbank(waveform):
    feat = kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=16000,
        dither=0.0,
    )
    return feat - feat.mean(dim=0, keepdim=True)


@torch.inference_mode()
def embed_waveform(model, waveform, device="cuda"):
    embedding = model(fbank(waveform).unsqueeze(0).to(device)).squeeze(0)
    return F.normalize(embedding.float(), dim=0).cpu()
