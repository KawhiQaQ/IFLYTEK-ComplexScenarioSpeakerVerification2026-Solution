#!/usr/bin/env python3
"""Train V54 with a fourth, speaker-disjoint Common Voice domain."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from miwu.quality_model import load_quality_model
from miwu.source_model import load_quality_local_temporal_model
from train_v1 import AAMHead, stable_split
from train_v7_pyramid_large import add_stcmds
from train_v45_layerwise_compensation import ShortFullPairDataset, train_phase


def add_commonvoice(dataset, root):
    records = {
        "cv:" + speaker.name: sorted(
            path for path in speaker.iterdir() if path.is_file()
        )
        for speaker in Path(root).iterdir() if speaker.is_dir()
    }
    records = {
        speaker: paths for speaker, paths in records.items() if len(paths) >= 2
    }
    if len(records) < 500:
        raise ValueError(
            "expected at least 500 Common Voice train speakers, found %d"
            % len(records)
        )
    dataset.by_speaker.update(records)
    dataset.speakers = sorted(dataset.by_speaker)
    dataset.speaker_to_index = {
        speaker: index for index, speaker in enumerate(dataset.speakers)
    }
    dataset.domain_indexes["cv"] = [
        index for index, speaker in enumerate(dataset.speakers)
        if speaker.startswith("cv:")
    ]
    # Rebuild existing domain indexes after the global speaker ordering changes.
    for domain in ("3d", "ocean", "stcmds"):
        dataset.domain_indexes[domain] = [
            index for index, speaker in enumerate(dataset.speakers)
            if speaker.startswith(domain + ":")
        ]
    ocean_gender_path = Path(dataset.ocean_root) / "train" / "spk2gender"
    ocean_genders = {
        "ocean:" + speaker: gender
        for speaker, gender in (
            line.split() for line in ocean_gender_path.read_text().splitlines()
        )
    }
    dataset.ocean_gender_indexes = {
        gender: [
            index for index, speaker in enumerate(dataset.speakers)
            if ocean_genders.get(speaker) == gender
        ]
        for gender in ("m", "f")
    }
    return len(records), sum(len(paths) for paths in records.values())


class FourDomainSpeakerSampler:
    def __init__(self, dataset, speakers_per_batch, durations, steps):
        if speakers_per_batch % 4:
            raise ValueError("speakers_per_batch must be divisible by four")
        self.dataset = dataset
        self.per_domain = speakers_per_batch // 4
        self.duration_values = [float(value) for value in durations]
        self.duration_weights = [float(durations[value]) for value in durations]
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            indexes = []
            indexes += random.sample(
                self.dataset.domain_indexes["3d"], self.per_domain
            )
            gender = random.choice(("m", "f"))
            indexes += random.sample(
                self.dataset.ocean_gender_indexes[gender], self.per_domain
            )
            indexes += random.sample(
                self.dataset.domain_indexes["stcmds"], self.per_domain
            )
            indexes += random.sample(
                self.dataset.domain_indexes["cv"], self.per_domain
            )
            random.shuffle(indexes)
            duration = random.choices(
                self.duration_values, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def main(
    model_loader=load_quality_local_temporal_model,
    default_config="configs/v55_v8_local_temporal_four_domain.yaml",
    default_output="outputs/v55_four_domain_local_temporal",
    architecture="V8 local temporal residual trained on four domains",
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=default_config
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--pyramid-checkpoint", required=True)
    parser.add_argument("--quality-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument(
        "--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS"
    )
    parser.add_argument(
        "--commonvoice-root", default="data/processed/commonvoice17-train"
    )
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
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
    sampler = FourDomainSpeakerSampler(
        dataset, config["speakers_per_batch"],
        config["duration_probabilities"],
        args.max_steps or config["steps_per_epoch"],
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    model = model_loader(
        args.backbone_checkpoint, args.pyramid_checkpoint,
        args.quality_checkpoint,
    ).train()
    teacher = load_quality_model(
        args.backbone_checkpoint, args.pyramid_checkpoint,
        args.quality_checkpoint,
    ).eval()
    teacher.requires_grad_(False)
    head = AAMHead(
        len(dataset.speakers), dimension=model.output_dimension,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    history = train_phase(
        model, teacher, None, head, loader, config,
        config["head_epochs"], False, output, 1,
        layerwise_supervision=False, teacher_audio_only=True,
    )
    history += train_phase(
        model, teacher, None, head, loader, config,
        config["joint_epochs"], True, output,
        config["head_epochs"] + 1,
        layerwise_supervision=False, teacher_audio_only=True,
    )
    torch.save(model.state_dict(), output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": architecture,
        "parent": str(Path(args.quality_checkpoint).resolve()),
        "embedding_dimension": model.output_dimension,
        "single_model": True,
        "training_domains": [
            "3D-Speaker", "Speechocean train", "ST-CMDS",
            "Common Voice 17 zh-CN train-speaker split",
        ],
        "training_speakers": len(dataset.speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "commonvoice_speakers": cv_speakers,
        "commonvoice_items": cv_items,
        "commonvoice_dev_speakers": 260,
        "speechocean_test_used_for_training": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
    }, indent=2))


if __name__ == "__main__":
    main()
