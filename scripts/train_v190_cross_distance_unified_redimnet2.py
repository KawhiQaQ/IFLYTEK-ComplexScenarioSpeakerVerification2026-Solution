#!/usr/bin/env python3
"""Continue one child-aware ReDimNet2 with real cross-distance pairs."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from miwu.encoder import fbank, load_audio
from miwu.palabra_redimnet2_dual_axis import load_palabra_redimnet2_dual_axis
from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2
from train_v1 import AAMHead, corrupt, crop_or_repeat, stable_split
from train_v45_layerwise_compensation import ShortFullPairDataset
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v176_child_dual_axis_redimnet2 import (
    add_childmandarin, rebuild_indexes, train_phase,
)
from train_v179_age_span_dual_axis_redimnet2 import ocean_age_groups


def acoustic_fields(path):
    fields = path.stem.split("_")
    return {
        name: next((value for value in fields if value.startswith(name)), "")
        for name in ("Device", "Distance", "Dialect")
    }


class CrossDistanceShortFullPairDataset(ShortFullPairDataset):
    """Prefer a different real distance/device for 3D-Speaker enrollment."""

    def make_query(self, full_waveform, duration):
        return corrupt(crop_or_repeat(full_waveform, int(float(duration) * 16000)))

    def __getitem__(self, request):
        speaker_index, duration = request
        speaker = self.speakers[speaker_index]
        paths = self.by_speaker[speaker]
        query_path = random.choice(paths)
        candidates = [path for path in paths if path != query_path]
        if speaker.startswith("3d:"):
            query_fields = acoustic_fields(query_path)
            cross_distance = [
                path for path in candidates
                if acoustic_fields(path)["Distance"]
                != query_fields["Distance"]
            ]
            if cross_distance:
                candidates = cross_distance
            cross_device = [
                path for path in candidates
                if acoustic_fields(path)["Device"] != query_fields["Device"]
            ]
            if cross_device:
                candidates = cross_device
        # ST-CMDS A/I namespaces are different released speaker identities,
        # not a known cross-device pairing. Sample within the corrected ID.
        enrollment_path = random.choice(candidates)
        full_waveform = crop_or_repeat(load_audio(query_path), 48000)
        query_waveform = self.make_query(full_waveform, duration)
        enrollment_waveform = crop_or_repeat(
            load_audio(enrollment_path), 48000
        )
        return (
            fbank(query_waveform), query_waveform,
            fbank(full_waveform), full_waveform,
            fbank(enrollment_waveform), enrollment_waveform,
            self.speaker_to_index[speaker],
        )


class CrossDistanceAgeSampler:
    """Three toddlers, one school child, two 3D and two adult domains."""

    def __init__(self, dataset, config, young_groups):
        if int(config["speakers_per_batch"]) != 8:
            raise ValueError("V190 uses eight speakers per batch")
        self.dataset = dataset
        self.young_groups = young_groups
        self.steps = int(config["steps_per_epoch"])
        self.durations = [float(x) for x in config["duration_probabilities"]]
        self.duration_weights = [
            float(config["duration_probabilities"][x])
            for x in config["duration_probabilities"]
        ]

    def __len__(self):
        return self.steps

    def __iter__(self):
        toddler_groups = list(self.dataset.child_group_indexes.values())
        for _ in range(self.steps):
            indexes = random.sample(random.choice(toddler_groups), 3)
            indexes.append(random.choice(random.choice(self.young_groups)))
            indexes.extend(random.sample(self.dataset.domain_indexes["3d"], 2))
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
        "--config", default="configs/v190_cross_distance_unified_redimnet2.yaml"
    )
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--child-checkpoint", required=True)
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
        "--output-dir", default="outputs/v190_cross_distance_unified_redimnet2"
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
    dataset = CrossDistanceShortFullPairDataset(
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
    sampler = CrossDistanceAgeSampler(dataset, config, young_groups)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    model = load_palabra_redimnet2_dual_axis(
        args.redimnet2_checkpoint, None, args.child_checkpoint
    ).cuda()
    teacher = load_palabra_multicorpus_redimnet2(
        args.redimnet2_checkpoint
    ).eval().requires_grad_(False)
    head = AAMHead(
        len(dataset.speakers), dimension=192,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    history, start = [], 1
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
            "one child-aware dual-axis ReDimNet2-B6 continued with real "
            "cross-distance and cross-device same-speaker pairs"
        ),
        "initialization": "V176 child dual-axis model",
        "batch_composition": (
            "3 age-3-5 + 1 age-6-15 + 2 cross-distance 3D-Speaker + "
            "1 ST-CMDS + 1 Common Voice"
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
