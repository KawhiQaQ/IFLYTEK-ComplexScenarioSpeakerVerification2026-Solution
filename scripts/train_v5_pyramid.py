#!/usr/bin/env python3
"""Train a cross-depth pyramid with short-query/long-enrollment prototypes."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from miwu.encoder import build_encoder, fbank, load_audio
from miwu.pyramid_model import load_pyramid_model
from train_v1 import AAMHead, corrupt, crop_or_repeat, stable_split
from train_v4_arch import read_ocean_training


class SpeakerPairDataset(Dataset):
    def __init__(self, three_d_root, three_d_speakers, ocean_root):
        self.by_speaker = {
            "3d:" + speaker: sorted((Path(three_d_root) / "test" / speaker).glob("*.wav"))
            for speaker in three_d_speakers
        }
        self.by_speaker.update(read_ocean_training(ocean_root))
        self.by_speaker = {
            speaker: paths for speaker, paths in self.by_speaker.items() if len(paths) >= 2
        }
        self.speakers = sorted(self.by_speaker)
        self.speaker_to_index = {
            speaker: index for index, speaker in enumerate(self.speakers)
        }
        self.domain_indexes = {
            domain: [
                index for index, speaker in enumerate(self.speakers)
                if speaker.startswith(domain + ":")
            ]
            for domain in ("3d", "ocean")
        }
        ocean_genders = dict(
            line.split()
            for line in (Path(ocean_root) / "train" / "spk2gender").read_text().splitlines()
        )
        self.ocean_gender_indexes = {
            gender: [
                index for index, speaker in enumerate(self.speakers)
                if speaker.startswith("ocean:")
                and ocean_genders[speaker.split(":", 1)[1]] == gender
            ]
            for gender in ("m", "f")
        }

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, request):
        speaker_index, duration = request
        speaker = self.speakers[speaker_index]
        query_path, enrollment_path = random.sample(self.by_speaker[speaker], 2)
        query = crop_or_repeat(load_audio(query_path), int(duration * 16000))
        enrollment = crop_or_repeat(load_audio(enrollment_path), 48000)
        return (
            fbank(corrupt(query)),
            fbank(enrollment),
            self.speaker_to_index[speaker],
        )


class BalancedSpeakerBatchSampler:
    def __init__(self, domain_indexes, ocean_gender_indexes, speakers_per_batch, durations, steps):
        if speakers_per_batch % 2:
            raise ValueError("speakers_per_batch must be even")
        self.domain_indexes = domain_indexes
        self.ocean_gender_indexes = ocean_gender_indexes
        self.per_domain = speakers_per_batch // 2
        self.duration_values = [float(value) for value in durations]
        self.duration_weights = [float(durations[value]) for value in durations]
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            indexes = []
            indexes.extend(random.sample(self.domain_indexes["3d"], self.per_domain))
            gender = random.choice(("m", "f"))
            indexes.extend(random.sample(self.ocean_gender_indexes[gender], self.per_domain))
            random.shuffle(indexes)
            duration = random.choices(
                self.duration_values, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def set_trainable(model, joint):
    model.requires_grad_(False)
    model.pyramid.requires_grad_(True)
    model.adapter_scale.requires_grad_(True)
    if joint:
        for name in ("layer3", "layer4", "layer3_ds", "fuse34", "seg_1"):
            getattr(model.backbone, name).requires_grad_(True)
    model.train()
    for name, module in model.backbone.named_children():
        if not joint or name not in ("layer3", "layer4", "layer3_ds", "fuse34", "seg_1"):
            module.eval()


def train_phase(model, teacher, head, loader, config, epochs, joint, output, start_epoch):
    set_trainable(model, joint)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    ]
    pyramid_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    groups = [
        {"params": pyramid_parameters, "lr": config["pyramid_lr"]},
        {"params": head.parameters(), "lr": config["head_lr"]},
    ]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, epochs * len(loader))
    )
    history = []
    for phase_epoch in range(epochs):
        epoch = start_epoch + phase_epoch
        totals = defaultdict(float)
        for step, (query_features, enrollment_features, labels) in enumerate(loader, 1):
            query_features = query_features.cuda(non_blocking=True)
            enrollment_features = enrollment_features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            query_lengths = torch.full(
                (query_features.shape[0],), query_features.shape[1],
                dtype=torch.long, device=query_features.device
            )
            enrollment_lengths = torch.full(
                (enrollment_features.shape[0],), enrollment_features.shape[1],
                dtype=torch.long, device=enrollment_features.device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_embedding = F.normalize(teacher(enrollment_features), dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                query_embedding = model(query_features, query_lengths)
                enrollment_embedding = model(enrollment_features, enrollment_lengths)
                classification = 0.5 * (
                    F.cross_entropy(head(query_embedding, labels), labels)
                    + F.cross_entropy(head(enrollment_embedding, labels), labels)
                )
                query_normalized = F.normalize(query_embedding, dim=1)
                enrollment_normalized = F.normalize(enrollment_embedding, dim=1)
                pair_cosine = torch.matmul(
                    query_normalized, enrollment_normalized.T
                )
                pair_labels = torch.arange(len(labels), device=labels.device)
                pair_cosine = pair_cosine - torch.eye(
                    len(labels), dtype=pair_cosine.dtype, device=pair_cosine.device
                ) * config["prototype_margin"]
                pair_logits = config["prototype_scale"] * pair_cosine
                prototype = 0.5 * (
                    F.cross_entropy(pair_logits, pair_labels)
                    + F.cross_entropy(pair_logits.T, pair_labels)
                )
                consistency = 0.5 * (
                    (1.0 - F.cosine_similarity(query_embedding, teacher_embedding)).mean()
                    + (1.0 - F.cosine_similarity(enrollment_embedding, teacher_embedding)).mean()
                )
                loss = (
                    config["classification_weight"] * classification
                    + config["prototype_weight"] * prototype
                    + config["teacher_weight"] * consistency
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]], 5.0
            )
            optimizer.step()
            scheduler.step()
            totals["loss"] += loss.item()
            totals["classification"] += classification.item()
            totals["prototype"] += prototype.item()
            totals["consistency"] += consistency.item()
            if step % 25 == 0:
                print(
                    "epoch=%d phase=%s step=%d loss=%.4f cls=%.4f proto=%.4f teacher=%.4f scale=%.4f"
                    % (
                        epoch, "joint" if joint else "head", step,
                        loss.item(), classification.item(), prototype.item(),
                        consistency.item(), model.adapter_scale.item()
                    ),
                    flush=True,
                )
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(epoch=epoch, phase="joint" if joint else "head", steps=len(loader))
        history.append(stats)
        torch.save(model.state_dict(), output / ("epoch_%02d.ckpt" % epoch))
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v5_pyramid_proto.yaml")
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--output-dir", default="outputs/v5_pyramid_proto")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(
        Path(args.three_d_root), 180
    )
    (output / "train_speakers.txt").write_text("\n".join(train_speakers) + "\n")
    (output / "dev_speakers.txt").write_text("\n".join(dev_speakers) + "\n")
    dataset = SpeakerPairDataset(args.three_d_root, train_speakers, args.ocean_root)
    sampler = BalancedSpeakerBatchSampler(
        dataset.domain_indexes,
        dataset.ocean_gender_indexes,
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
        "objective": "AAM + bidirectional short-query/long-enrollment prototype + teacher anchor",
        "config": config,
        "training_speakers": len(dataset.speakers),
        "three_d_dev_speakers": len(dev_speakers),
        "speechocean_test_used_for_training": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
