#!/usr/bin/env python3
"""Train the feature-pyramid architecture with 443 extra Mandarin speakers."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from miwu.encoder import build_encoder
from miwu.pyramid_model import load_pyramid_model
from train_v1 import AAMHead, stable_split
from train_v5_pyramid import SpeakerPairDataset, train_phase


def add_stcmds(dataset, root):
    records = defaultdict(list)
    for path in Path(root).rglob("*.wav"):
        # SLR38 has 855 speakers with 120 recordings each. Pxxxxx is reused
        # in the Android and IOS namespaces: metadata for the same number
        # can have different sex, age and birthplace. Keep that namespace.
        # The old [8:14] slice incorrectly merged 855 people into 443 labels.
        speaker = path.stem[8:15]
        if len(speaker) != 7 or speaker[0] != 'P' or not speaker[1:6].isdigit() or speaker[-1] not in 'AI':
            raise ValueError('Unexpected ST-CMDS identity filename: ' + path.name)
        if speaker:
            records["stcmds:" + speaker].append(path)
    records = {speaker: sorted(paths) for speaker, paths in records.items() if len(paths) >= 2}
    if len(records) < 400:
        raise ValueError("Expected at least 400 ST-CMDS speakers, found %d" % len(records))
    dataset.by_speaker.update(records)
    dataset.speakers = sorted(dataset.by_speaker)
    dataset.speaker_to_index = {
        speaker: index for index, speaker in enumerate(dataset.speakers)
    }
    dataset.domain_indexes = {
        domain: [
            index for index, speaker in enumerate(dataset.speakers)
            if speaker.startswith(domain + ":")
        ]
        for domain in ("3d", "ocean", "stcmds")
    }
    ocean_genders = dict(
        line.split()
        for line in (Path(dataset.ocean_root) / "train" / "spk2gender").read_text().splitlines()
    )
    dataset.ocean_gender_indexes = {
        gender: [
            index for index, speaker in enumerate(dataset.speakers)
            if speaker.startswith("ocean:")
            and ocean_genders[speaker.split(":", 1)[1]] == gender
        ]
        for gender in ("m", "f")
    }
    return len(records), sum(len(paths) for paths in records.values())


class ThreeDomainSpeakerSampler:
    def __init__(self, dataset, speakers_per_batch, durations, steps):
        if speakers_per_batch % 3:
            raise ValueError("speakers_per_batch must be divisible by three")
        self.dataset = dataset
        self.per_domain = speakers_per_batch // 3
        self.duration_values = [float(value) for value in durations]
        self.duration_weights = [float(durations[value]) for value in durations]
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            indexes = random.sample(self.dataset.domain_indexes["3d"], self.per_domain)
            gender = random.choice(("m", "f"))
            indexes += random.sample(
                self.dataset.ocean_gender_indexes[gender], self.per_domain
            )
            indexes += random.sample(
                self.dataset.domain_indexes["stcmds"], self.per_domain
            )
            random.shuffle(indexes)
            duration = random.choices(
                self.duration_values, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7_pyramid_large.yaml")
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument(
        "--stcmds-root",
        default="data/processed/stcmds/ST-CMDS-20170001_1-OS",
    )
    parser.add_argument("--output-dir", default="outputs/v7_pyramid_large")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(Path(args.three_d_root), 180)
    (output / "train_speakers.txt").write_text("\n".join(train_speakers) + "\n")
    (output / "dev_speakers.txt").write_text("\n".join(dev_speakers) + "\n")

    dataset = SpeakerPairDataset(args.three_d_root, train_speakers, args.ocean_root)
    dataset.ocean_root = args.ocean_root
    extra_speakers, extra_items = add_stcmds(dataset, args.stcmds_root)
    sampler = ThreeDomainSpeakerSampler(
        dataset,
        config["speakers_per_batch"],
        config["duration_probabilities"],
        args.max_steps or config["steps_per_epoch"],
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0
    )
    model = load_pyramid_model(args.backbone_checkpoint).train()
    teacher = build_encoder(args.teacher_checkpoint).eval()
    teacher.requires_grad_(False)
    head = AAMHead(
        len(dataset.speakers), dimension=192,
        margin=config["margin"], scale=config["scale"]
    ).cuda()
    history = train_phase(
        model, teacher, head, loader, config,
        config["head_epochs"], False, output, 1
    )
    history += train_phase(
        model, teacher, head, loader, config,
        config["joint_epochs"], True, output, config["head_epochs"] + 1
    )
    torch.save(model.state_dict(), output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "cross-depth duration-gated feature pyramid ERes2NetV2",
        "training_domains": ["3D-Speaker", "Speechocean train", "ST-CMDS"],
        "objective": "AAM + same-sex margin prototype + teacher anchor",
        "config": config,
        "training_speakers": len(dataset.speakers),
        "stcmds_speakers": extra_speakers,
        "stcmds_items": extra_items,
        "three_d_dev_speakers": len(dev_speakers),
        "speechocean_test_used_for_training": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
