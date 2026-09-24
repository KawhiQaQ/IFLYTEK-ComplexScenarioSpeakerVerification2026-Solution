"""Single fixed architecture control against V358, initialized from R45/V302."""
import json
import sys
from pathlib import Path
import train_v302_hard_age_depth_time_grl_long as training
from train_v358_coverage_grl import CoverageHardSampler, train_with_observation
from miwu.tidyvoice_wide_adapter_head import WideAdapterGRLW2VBert2


def main():
    training.DepthTimeGRLW2VBert2 = WideAdapterGRLW2VBert2
    training.BalancedAgeComplementSampler = CoverageHardSampler
    training.train = train_with_observation
    defaults = {'--config': 'configs/sphere_fusion_training.yaml',
                '--prototype-file': 'outputs/v328_corrected_prototypes.pt',
                '--output-dir': 'outputs/v360_wide_adapter'}
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    training.main()
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    path = output / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(architecture='Complete R45 depth-time head plus 25 wide residual SSL adapters',
        initialization=WideAdapterGRLW2VBert2.initial_checkpoint,
        teacher='Frozen original V302', changed_inference_architecture=True,
        additional_encoders=0, residual_hidden_width=512, baseline_semifinal_lb=93.35674,
        research_control='V358 same coverage/losses/1200 steps; architecture is the only planned change',
        validation_audio_used=False)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
