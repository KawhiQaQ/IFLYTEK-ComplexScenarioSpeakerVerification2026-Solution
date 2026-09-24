#!/usr/bin/env python3
"""Train the duration-aware multi-scale architecture on two public domains."""

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

from miwu.duration_model import load_duration_model
from miwu.encoder import build_encoder, fbank, load_audio
from train_v1 import AAMHead, corrupt, crop_or_repeat, stable_split


def read_ocean_training(root):
    metadata = Path(root) / "train"
    utterance_to_speaker = dict(
        line.split() for line in (metadata / "utt2spk").read_text().splitlines()
    )
    utterance_to_path = dict(
        line.split(maxsplit=1) for line in (metadata / "wav.scp").read_text().splitlines()
    )
    records = defaultdict(list)
    for utterance, speaker in utterance_to_speaker.items():
        path = Path(root) / utterance_to_path[utterance]
        if path.is_file():
            records["ocean:" + speaker].append(path)
    return records


class MixedDomainDataset(Dataset):
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
        self.items = [
            (speaker, path)
            for speaker in self.speakers
            for path in sorted(self.by_speaker[speaker])
        ]
        self.domain_indexes = {
            domain: [
                index
                for index, (speaker, _) in enumerate(self.items)
                if speaker.startswith(domain + ":")
            ]
            for domain in ("3d", "ocean")
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, request):
        index, duration = request
        speaker, path = self.items[index]
        student_waveform = crop_or_repeat(load_audio(path), int(duration * 16000))
        student_waveform = corrupt(student_waveform)
        alternatives = self.by_speaker[speaker]
        teacher_path = random.choice(alternatives)
        if len(alternatives) > 1:
            while teacher_path == path:
                teacher_path = random.choice(alternatives)
        teacher_waveform = crop_or_repeat(load_audio(teacher_path), 48000)
        return (
            fbank(student_waveform),
            fbank(teacher_waveform),
            self.speaker_to_index[speaker],
        )


class BalancedDomainBatchSampler:
    """Give the two public training domains equal mass in every batch."""

    def __init__(self, domain_indexes, batch_size, duration_probabilities, max_steps=0):
        if batch_size % 2:
            raise ValueError("Domain-balanced batch size must be even")
        self.domain_indexes = domain_indexes
        self.batch_size = batch_size
        self.per_domain = batch_size // 2
        self.steps = max_steps or sum(len(values) for values in domain_indexes.values()) // batch_size
        pairs = [
            (float(duration), float(probability))
            for duration, probability in duration_probabilities.items()
        ]
        self.durations = [duration for duration, _ in pairs]
        self.probabilities = [probability for _, probability in pairs]

    def __len__(self):
        return self.steps

    def __iter__(self):
        pools = {domain: list(values) for domain, values in self.domain_indexes.items()}
        positions = {domain: len(values) for domain, values in pools.items()}
        for _ in range(self.steps):
            batch = []
            for domain in ("3d", "ocean"):
                if positions[domain] + self.per_domain > len(pools[domain]):
                    random.shuffle(pools[domain])
                    positions[domain] = 0
                start = positions[domain]
                batch.extend(pools[domain][start : start + self.per_domain])
                positions[domain] += self.per_domain
            random.shuffle(batch)
            duration = random.choices(
                self.durations, weights=self.probabilities, k=1
            )[0]
            yield [(index, duration) for index in batch]


def set_trainable(model, joint):
    model.requires_grad_(False)
    model.adapter.requires_grad_(True)
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
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    ]
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    groups = [
        {"params": adapter_parameters, "lr": config["adapter_lr"]},
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
        batches = 0
        for student_features, teacher_features, labels in loader:
            batches += 1
            student_features = student_features.cuda(non_blocking=True)
            teacher_features = teacher_features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            feature_lengths = torch.full(
                (student_features.shape[0],),
                student_features.shape[1],
                dtype=torch.long,
                device=student_features.device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                target = F.normalize(teacher(teacher_features), dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embedding = model(student_features, feature_lengths)
                classification = F.cross_entropy(head(embedding, labels), labels)
                consistency = (1.0 - F.cosine_similarity(embedding, target)).mean()
                loss = (
                    config["classification_weight"] * classification
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
            totals["consistency"] += consistency.item()
            if batches % 25 == 0:
                print(
                    "epoch=%d phase=%s step=%d loss=%.4f cls=%.4f teacher=%.4f scale=%.4f"
                    % (
                        epoch,
                        "joint" if joint else "head",
                        batches,
                        loss.item(),
                        classification.item(),
                        consistency.item(),
                        model.adapter_scale.item(),
                    ),
                    flush=True,
                )
        stats = {name: value / batches for name, value in totals.items()}
        stats.update(epoch=epoch, phase="joint" if joint else "head", steps=batches)
        history.append(stats)
        torch.save(model.state_dict(), output / ("epoch_%02d.ckpt" % epoch))
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_duration_arch.yaml")
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--output-dir", default="outputs/v4_duration_arch")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(
        Path(args.three_d_root), config["train_speakers"]
    )
    (output / "train_speakers.txt").write_text("\n".join(train_speakers) + "\n")
    (output / "dev_speakers.txt").write_text("\n".join(dev_speakers) + "\n")

    dataset = MixedDomainDataset(
        args.three_d_root, train_speakers, args.ocean_root
    )
    sampler = BalancedDomainBatchSampler(
        dataset.domain_indexes,
        config["batch_size"],
        config["duration_probabilities"],
        max_steps=args.max_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config["workers"],
        pin_memory=True,
        persistent_workers=config["workers"] > 0,
    )
    model = load_duration_model(args.backbone_checkpoint).train()
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
    metadata = {
        "architecture": "duration-aware multi-scale residual ERes2NetV2",
        "config": config,
        "training_speakers": len(dataset.speakers),
        "training_items": len(dataset),
        "three_d_dev_speakers": len(dev_speakers),
        "speechocean_test_used_for_training": False,
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
