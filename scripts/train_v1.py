#!/usr/bin/env python3
"""Robust single-model adaptation with speaker-disjoint validation.

The student learns from short/noisy/reverberant speech. A frozen pretrained
teacher anchors each sample to a different clean utterance of the same speaker,
which limits small-corpus drift while encouraging duration/channel invariance.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

from miwu.encoder import build_encoder, fbank, load_audio


def stable_split(root, train_count):
    speakers = sorted(path.name for path in (root / "test").iterdir() if path.is_dir())
    rng = random.Random(823)
    rng.shuffle(speakers)
    return speakers[:train_count], speakers[train_count:]


def random_crop(wav, wanted):
    if wav.numel() <= wanted:
        return wav
    start = random.randint(0, wav.numel() - wanted)
    return wav[start : start + wanted]


def crop_or_repeat(wav, wanted):
    if wav.numel() >= wanted:
        return random_crop(wav, wanted)
    repeats = (wanted + wav.numel() - 1) // wav.numel()
    return wav.repeat(repeats)[:wanted]


def corrupt(wav, sample_rate=16000):
    # Identity-preserving channel/noise transforms only; no label-conditioned logic.
    if random.random() < 0.65:
        snr_db = random.uniform(5.0, 25.0)
        noise = torch.randn_like(wav)
        scale = wav.square().mean().sqrt() / (noise.square().mean().sqrt() + 1e-8)
        wav = wav + noise * scale * (10.0 ** (-snr_db / 20.0))
    if random.random() < 0.45:
        tail = random.randint(800, 4800)
        decay = random.uniform(3.0, 8.0)
        rir = torch.randn(tail) * torch.exp(-torch.linspace(0, decay, tail))
        rir[0] += 1.0
        rir = rir / (rir.square().sum().sqrt() + 1e-8)
        wav = F.conv1d(wav[None, None], rir.flip(0)[None, None], padding=tail - 1)[0, 0, : wav.numel()]
    if random.random() < 0.35:
        low_rate = random.choice([8000, 12000])
        wav = torchaudio.functional.resample(wav, sample_rate, low_rate)
        wav = torchaudio.functional.resample(wav, low_rate, sample_rate)
    wav = wav * random.uniform(0.5, 1.2)
    return wav.clamp(-1, 1)


class RobustDataset(Dataset):
    def __init__(self, root, speakers, duration_probabilities, teacher_same_file=False):
        self.speakers = list(speakers)
        self.speaker_to_index = {speaker: index for index, speaker in enumerate(self.speakers)}
        self.by_speaker = {
            speaker: sorted((root / "test" / speaker).glob("*.wav")) for speaker in speakers
        }
        self.items = [path for speaker in speakers for path in self.by_speaker[speaker]]
        self.teacher_same_file = teacher_same_file

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        index, duration = index
        path = self.items[index]
        speaker = path.parent.name
        student = load_audio(path)
        student = corrupt(crop_or_repeat(student, int(duration * 16000)))

        teacher_path = path if self.teacher_same_file else random.choice(self.by_speaker[speaker])
        teacher = load_audio(teacher_path)
        teacher = crop_or_repeat(teacher, 48000)
        return fbank(student), fbank(teacher), self.speaker_to_index[speaker]


class DurationBatchSampler:
    def __init__(self, size, batch_size, duration_probabilities, drop_last=True):
        self.size = size
        self.batch_size = batch_size
        self.drop_last = drop_last
        pairs = [(float(value), float(probability)) for value, probability in duration_probabilities.items()]
        self.durations = [pair[0] for pair in pairs]
        self.probabilities = [pair[1] for pair in pairs]

    def __len__(self):
        if self.drop_last:
            return self.size // self.batch_size
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indexes = list(range(self.size))
        random.shuffle(indexes)
        for start in range(0, self.size, self.batch_size):
            chunk = indexes[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            duration = random.choices(self.durations, weights=self.probabilities, k=1)[0]
            yield [(index, duration) for index in chunk]


class AAMHead(nn.Module):
    def __init__(self, speakers, dimension=192, margin=.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(speakers, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cosine)
        target = torch.cos(theta.float() + self.margin).to(cosine.dtype)
        logits = cosine.clone()
        logits[torch.arange(len(labels), device=labels.device), labels] = target[
            torch.arange(len(labels), device=labels.device), labels
        ]
        return logits * self.scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_robust.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument(
        "--architecture",
        choices=("eres2netv2", "eres2net_large"),
        default="eres2netv2",
    )
    parser.add_argument("--data-root", default="data/processed/3dspeaker")
    parser.add_argument("--output-dir", default="outputs/v1_robust")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(root, config["train_speakers"])
    assert set(train_speakers).isdisjoint(dev_speakers)
    (output / "train_speakers.txt").write_text("\n".join(train_speakers) + "\n")
    (output / "dev_speakers.txt").write_text("\n".join(dev_speakers) + "\n")

    student = build_encoder(args.checkpoint, architecture=args.architecture).train()
    teacher_checkpoint = args.teacher_checkpoint or args.checkpoint
    teacher = build_encoder(
        teacher_checkpoint, architecture=args.architecture
    ).eval()
    teacher.requires_grad_(False)
    prefixes = tuple(config["trainable_prefixes"])
    for name, parameter in student.named_parameters():
        parameter.requires_grad = name.startswith(prefixes)
    for name, module in student.named_children():
        if not name.startswith(prefixes):
            module.eval()
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)

    dataset = RobustDataset(
        root,
        train_speakers,
        config["duration_probabilities"],
        config.get("teacher_same_file", False),
    )
    batch_sampler = DurationBatchSampler(
        len(dataset), config["batch_size"], config["duration_probabilities"]
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config["workers"],
        pin_memory=True,
        persistent_workers=config["workers"] > 0,
    )
    embedding_dimension = 512 if args.architecture == "eres2net_large" else 192
    head = AAMHead(
        len(train_speakers),
        dimension=embedding_dimension,
        margin=config["margin"],
        scale=config["scale"],
    ).cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in student.parameters() if p.requires_grad], "lr": config["encoder_lr"]},
            {"params": head.parameters(), "lr": config["head_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    total_steps = config["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps))
    history = []
    global_step = 0
    for epoch in range(1, config["epochs"] + 1):
        totals = defaultdict(float)
        batches = 0
        for student_feat, teacher_feat, labels in loader:
            global_step += 1
            batches += 1
            student_feat = student_feat.cuda(non_blocking=True)
            teacher_feat = teacher_feat.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                targets = F.normalize(teacher(teacher_feat), dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = student(student_feat)
                classification = F.cross_entropy(head(embeddings, labels), labels)
                consistency = (1 - F.cosine_similarity(embeddings, targets)).mean()
                loss = (
                    config["classification_weight"] * classification
                    + config["teacher_weight"] * consistency
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            scheduler.step()
            totals["loss"] += loss.item()
            totals["classification"] += classification.item()
            totals["consistency"] += consistency.item()
            if global_step % 25 == 0:
                print(
                    "epoch=%d step=%d loss=%.4f cls=%.4f teacher=%.4f"
                    % (epoch, global_step, loss.item(), classification.item(), consistency.item()),
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        stats = {name: value / batches for name, value in totals.items()}
        stats.update(epoch=epoch, steps=batches)
        history.append(stats)
        torch.save(student.state_dict(), output / ("epoch_%02d.ckpt" % epoch))
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(stats), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break

    torch.save(student.state_dict(), output / "final.ckpt")
    metadata = dict(
        config=config,
        architecture=args.architecture,
        source_checkpoint=str(Path(args.checkpoint).resolve()),
        teacher_checkpoint=str(Path(teacher_checkpoint).resolve()),
        train_speakers=len(train_speakers),
        dev_speakers=len(dev_speakers),
        overlap=sorted(set(train_speakers) & set(dev_speakers)),
        trainable_parameters=trainable,
        total_parameters=sum(p.numel() for p in student.parameters()),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
