#!/usr/bin/env python3
"""Train only V364's zero-initialized cross residual over frozen V360."""
import json
import sys
from pathlib import Path

import train_v302_hard_age_depth_time_grl_long as training

from miwu.tidyvoice_wide_cross_residual import (
    WideCrossResidualGRLW2VBert2,
)


def main():
    training.DepthTimeGRLW2VBert2 = WideCrossResidualGRLW2VBert2
    defaults = {
        "--config": "configs/sphere_fusion_training.yaml",
        "--prototype-file": "outputs/v328_corrected_prototypes.pt",
        "--output-dir": "outputs/v364_wide_cross_residual",
    }
    for flag, value in defaults.items():
        if not any(
            argument == flag or argument.startswith(flag + "=")
            for argument in sys.argv[1:]
        ):
            sys.argv.extend((flag, value))
    training.main()
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        architecture=(
            "Frozen V360 wide SSL head with one trainable ReDimNet2-to-SSL "
            "cross-attention residual inside the same 256-D speaker head"
        ),
        initialization=WideCrossResidualGRLW2VBert2.initial_checkpoint,
        teacher="Frozen V360 wide-adapter head",
        acoustic_initialization=(
            WideCrossResidualGRLW2VBert2.acoustic_checkpoint
        ),
        encoders_frozen=["complete S2", "complete released ReDimNet2"],
        frozen_v360_head=True,
        trainable_component="zero-initialized cross_encoder only",
        cross_encoder_lr_multiplier=10,
        changed_inference_architecture=True,
        validation_audio_used=False,
        online_feedback_used_for_training=False,
        research_control=(
            "V360 initialization, corrected training identities, original "
            "seven losses, sampler, seed and 1200 steps"
        ),
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
