#!/usr/bin/env python3
"""Age-balanced continuation of residual-ensemble ReDimNet2."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from miwu.palabra_redimnet2_dual_axis import load_palabra_redimnet2_dual_axis
from train_v1 import AAMHead, stable_split
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v176_child_dual_axis_redimnet2 import add_childmandarin, rebuild_indexes
from train_v179_age_span_dual_axis_redimnet2 import ocean_age_groups
from train_v190_cross_distance_unified_redimnet2 import CrossDistanceShortFullPairDataset
from train_v244_hard_prototype_redimnet2 import nearest_by_group
from train_v258_residual_ensemble_redimnet2 import train_phase


class BalancedAgeComplementSampler:
    """Alternate preschool and school-age hard clusters in every batch."""

    def __init__(self, dataset, config, prototypes, young_groups, adult_indexes):
        self.steps = int(config["steps_per_epoch"])
        self.durations = [float(x) for x in config["duration_probabilities"]]
        self.duration_weights = [float(config["duration_probabilities"][x])
                                 for x in config["duration_probabilities"]]
        count = int(config["hard_neighbor_count"])
        self.topk = int(config["hard_sample_topk"])
        adult = set(adult_indexes)
        groups = {
            "preschool": list(dataset.child_group_indexes.values()),
            "school": list(young_groups),
            "ocean_adult": [[i for i in dataset.ocean_gender_indexes[g] if i in adult]
                            for g in ("m", "f")],
            "3d": [dataset.domain_indexes["3d"]],
            "stcmds": [dataset.domain_indexes["stcmds"]],
            "cv": [dataset.domain_indexes["cv"]],
        }
        self.nearest = {name: nearest_by_group(prototypes, value, count)
                        for name, value in groups.items()}
        self.anchors = {name: sorted(value) for name, value in self.nearest.items()}
        if any(not value for value in self.anchors.values()):
            raise ValueError("an age-balanced mining group has no anchors")

    def __len__(self): return self.steps

    def cluster(self, name):
        anchor = random.choice(self.anchors[name])
        return [anchor] + random.sample(self.nearest[name][anchor][:self.topk], 3)

    def __iter__(self):
        wide = ("3d", "ocean_adult", "stcmds", "cv")
        for step in range(self.steps):
            age = "preschool" if step % 2 == 0 else "school"
            # Both ages must visit every wide domain. The previous step % 4
            # coupled preschool to 3D/ST-CMDS and school to Ocean/CV.
            indexes = self.cluster(age) + self.cluster(wide[(step // 2) % len(wide)])
            random.shuffle(indexes)
            duration = random.choices(self.durations, weights=self.duration_weights, k=1)[0]
            yield [(index, duration) for index in indexes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v260_age_balanced_residual_redimnet2.yaml")
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--mining-prototypes", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS")
    parser.add_argument("--commonvoice-root", default="data/processed/commonvoice17-train")
    parser.add_argument("--childmandarin-root", default="data/raw/childmandarin/train")
    parser.add_argument("--output-dir", default="outputs/v260_age_balanced_residual_redimnet2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.smoke: config.update(head_epochs=0, refiner_epochs=1, upper_epochs=0,
                                 steps_per_epoch=1, workers=0)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(Path(args.three_d_root), 180)
    dataset = CrossDistanceShortFullPairDataset(args.three_d_root, train_speakers, args.ocean_root)
    dataset.ocean_root = args.ocean_root
    add_stcmds(dataset, args.stcmds_root); add_commonvoice(dataset, args.commonvoice_root)
    child_speakers, _, _ = add_childmandarin(dataset, args.childmandarin_root); rebuild_indexes(dataset)
    young_groups, adults, ocean_groups = ocean_age_groups(dataset, args.ocean_root)
    package = torch.load(args.mining_prototypes, map_location="cpu")
    if package["speakers"] != dataset.speakers: raise RuntimeError("prototype speaker order mismatch")
    complements = package["prototypes"].float().clone(); complements[:, 3456:3648] = 0
    complements = F.normalize(complements, dim=1)
    sampler = BalancedAgeComplementSampler(dataset, config, complements, young_groups, adults)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"],
                        pin_memory=True, persistent_workers=config["workers"] > 0)
    model = load_palabra_redimnet2_dual_axis(args.redimnet2_checkpoint, None,
                                             args.initial_checkpoint).cuda()
    teacher = load_palabra_redimnet2_dual_axis(args.redimnet2_checkpoint, None,
                                               args.initial_checkpoint).eval().requires_grad_(False)
    head = AAMHead(len(dataset.speakers), dimension=192,
                   margin=config["margin"], scale=config["scale"]).cuda()
    history, start = [], 1
    for mode, key in (("head", "head_epochs"), ("refiner", "refiner_epochs"), ("upper", "upper_epochs")):
        epochs = int(config[key])
        if epochs:
            history += train_phase(model, teacher, head, loader, complements, config, epochs, mode, start)
            start += epochs
    torch.save({"state_dict": model.state_dict()}, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "age-balanced residual-ensemble ReDimNet2",
        "parent": str(args.initial_checkpoint), "training_speakers": len(dataset.speakers),
        "child_speakers": child_speakers, "ocean_school_groups": ocean_groups,
        "three_d_dev_speakers": len(dev_speakers), "validation_audio_used": False,
        "config": config}, indent=2))


if __name__ == "__main__": main()
