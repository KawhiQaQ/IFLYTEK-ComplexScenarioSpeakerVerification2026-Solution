#!/usr/bin/env python3
"""Train dual-axis ReDimNet2 across toddler, school-age, and adult speech."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from miwu.palabra_redimnet2_dual_axis import (
    load_palabra_redimnet2_dual_axis,
)
from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2
from train_v1 import AAMHead, stable_split
from train_v45_layerwise_compensation import ShortFullPairDataset
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v176_child_dual_axis_redimnet2 import (
    add_childmandarin, rebuild_indexes, train_phase,
)


def ocean_age_groups(dataset, ocean_root):
    age_gender = {}
    ages = {
        "ocean:" + fields[0]: int(fields[1])
        for fields in (
            line.split() for line in
            (Path(ocean_root) / "train" / "spk2age").read_text().splitlines()
        )
    }
    genders = {
        "ocean:" + fields[0]: fields[1]
        for fields in (
            line.split() for line in
            (Path(ocean_root) / "train" / "spk2gender").read_text().splitlines()
        )
    }
    for speaker, age in ages.items():
        if speaker not in dataset.speaker_to_index or speaker not in genders:
            continue
        if age <= 9:
            band = "06_09"
        elif age <= 12:
            band = "10_12"
        elif age <= 15:
            band = "13_15"
        else:
            continue
        age_gender.setdefault((band, genders[speaker]), []).append(
            dataset.speaker_to_index[speaker]
        )
    young_groups = [
        indexes for indexes in age_gender.values() if len(indexes) >= 2
    ]
    adult_indexes = [
        dataset.speaker_to_index[speaker]
        for speaker, age in ages.items()
        if age >= 19 and speaker in dataset.speaker_to_index
    ]
    if not young_groups or not adult_indexes:
        raise ValueError("Speechocean age groups are incomplete")
    return young_groups, adult_indexes, {
        "%s_%s" % key: len(value) for key, value in age_gender.items()
    }


class AgeSpanSampler:
    """Two toddlers, two age-matched school children, four adult domains."""

    def __init__(self, dataset, config, young_groups, adult_indexes):
        if int(config["speakers_per_batch"]) != 8:
            raise ValueError("V179 uses eight speakers per batch")
        if int(config["child_speakers_per_batch"]) != 4:
            raise ValueError("V179 reserves four child speakers per batch")
        self.dataset = dataset
        self.young_groups = young_groups
        self.adult_indexes = adult_indexes
        self.steps = int(config["steps_per_epoch"])
        self.durations = [float(value) for value in config["duration_probabilities"]]
        self.duration_weights = [
            float(config["duration_probabilities"][value])
            for value in config["duration_probabilities"]
        ]

    def __len__(self):
        return self.steps

    def __iter__(self):
        toddler_groups = list(self.dataset.child_group_indexes.values())
        for _ in range(self.steps):
            indexes = random.sample(random.choice(toddler_groups), 2)
            indexes.extend(random.sample(random.choice(self.young_groups), 2))
            indexes.append(random.choice(self.adult_indexes))
            indexes.append(random.choice(self.dataset.domain_indexes["3d"]))
            indexes.append(random.choice(self.dataset.domain_indexes["stcmds"]))
            indexes.append(random.choice(self.dataset.domain_indexes["cv"]))
            random.shuffle(indexes)
            duration = random.choices(
                self.durations, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v179_age_span_dual_axis_redimnet2.yaml"
    )
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument(
        "--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS"
    )
    parser.add_argument(
        "--commonvoice-root", default="data/processed/commonvoice17-train"
    )
    parser.add_argument(
        "--childmandarin-root", default="data/raw/childmandarin/train"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v179_age_span_dual_axis_redimnet2"
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps:
        config["steps_per_epoch"] = args.max_steps
    if args.smoke:
        config["head_epochs"] = 0
        config["refiner_epochs"] = 1
        config["upper_epochs"] = 1
        config["steps_per_epoch"] = 1
        config["workers"] = 0
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_speakers, dev_speakers = stable_split(Path(args.three_d_root), 180)
    dataset = ShortFullPairDataset(
        args.three_d_root, train_speakers, args.ocean_root
    )
    dataset.ocean_root = args.ocean_root
    stcmds_speakers, stcmds_items = add_stcmds(dataset, args.stcmds_root)
    cv_speakers, cv_items = add_commonvoice(dataset, args.commonvoice_root)
    child_speakers, child_items, child_groups = add_childmandarin(
        dataset, args.childmandarin_root
    )
    rebuild_indexes(dataset)
    young_groups, ocean_adults, ocean_groups = ocean_age_groups(
        dataset, args.ocean_root
    )
    sampler = AgeSpanSampler(
        dataset, config, young_groups, ocean_adults
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    model = load_palabra_redimnet2_dual_axis(
        args.redimnet2_checkpoint, None
    ).cuda()
    teacher = load_palabra_multicorpus_redimnet2(
        args.redimnet2_checkpoint
    ).eval().requires_grad_(False)
    head = AAMHead(
        len(dataset.speakers), dimension=192,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    history = []
    start = 1
    for mode, key in (
        ("head", "head_epochs"), ("refiner", "refiner_epochs"),
        ("upper", "upper_epochs"),
    ):
        epochs = int(config[key])
        if epochs:
            history += train_phase(
                model, teacher, head, loader, config, epochs, mode, start
            )
            start += epochs
    torch.save({"state_dict": model.state_dict()}, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": (
            "single ReDimNet2-B6 with temporal receptive-field pyramid and "
            "spectral evidence pooling, trained across the child age span"
        ),
        "training_domains": [
            "ChildMandarin train", "Speechocean train", "3D-Speaker",
            "ST-CMDS", "Common Voice 17 zh-CN train",
        ],
        "batch_composition": (
            "2 ChildMandarin age 3-5 + 2 Speechocean age 6-15 + "
            "1 Speechocean adult + 1 each 3D/ST-CMDS/Common Voice"
        ),
        "training_speakers": len(dataset.speakers),
        "child_train_speakers": child_speakers,
        "child_train_items": child_items,
        "child_groups": child_groups,
        "speechocean_young_groups": ocean_groups,
        "speechocean_adult_speakers": len(ocean_adults),
        "child_dev_test_used_for_training": False,
        "three_d_dev_speakers": len(dev_speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "commonvoice_speakers": cv_speakers,
        "commonvoice_items": cv_items,
        "single_model": True,
        "config": config,
    }, indent=2))


if __name__ == "__main__":
    main()
