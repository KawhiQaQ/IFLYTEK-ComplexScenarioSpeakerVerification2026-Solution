"""Wrapper for the official multi-corpus PalabraAI ReDimNet2 release."""

import sys
from pathlib import Path

import torch
import torch.nn as nn


def _import_redimnet2():
    repository = Path(__file__).resolve().parents[2]
    candidates = (
        repository / ".cache/vendor/palabra-redimnet2",
        repository
        / "checkpoints/pretrained/r45_runtime/model_lib/redimnet2",
    )
    vendor_root = next((path for path in candidates if path.is_dir()), None)
    if vendor_root is None:
        raise RuntimeError(
            "PalabraAI ReDimNet2 source is missing; run "
            "scripts/install_weights.py first"
        )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from redimnet2.redimnet2 import ReDimNet2Wrap
    return ReDimNet2Wrap


class PalabraMultiCorpusReDimNet2(nn.Module):
    """VoxBlink2/VoxCeleb2/CN-Celeb2 B6 large-margin encoder."""

    output_dimension = 192

    def __init__(self, checkpoint):
        super().__init__()
        released = torch.load(str(checkpoint), map_location="cpu")
        if not isinstance(released, dict):
            raise RuntimeError("unexpected ReDimNet2 checkpoint container")
        if "model_config" not in released or "state_dict" not in released:
            raise RuntimeError("ReDimNet2 checkpoint lacks config or weights")
        ReDimNet2Wrap = _import_redimnet2()
        self.encoder = ReDimNet2Wrap(**released["model_config"])
        missing, unexpected = self.encoder.load_state_dict(
            released["state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "ReDimNet2 checkpoint mismatch: missing=%s unexpected=%s"
                % (missing, unexpected)
            )
        self.output_dimension = int(self.encoder.embed_dim)

    def forward(
        self, features=None, lengths=None, waveforms=None,
        waveform_lengths=None,
    ):
        if waveforms is None:
            raise ValueError("PalabraAI ReDimNet2 requires raw waveforms")
        return self.encoder(waveforms)


def load_palabra_multicorpus_redimnet2(checkpoint, device="cuda"):
    return PalabraMultiCorpusReDimNet2(checkpoint).to(device)
