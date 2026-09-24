#!/usr/bin/env python3
"""Train a child-aware dual-axis ReDimNet2 on five speaker domains."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from miwu.palabra_redimnet2_dual_axis import (
    load_palabra_redimnet2_dual_axis,
)
from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2
from train_v1 import AAMHead, stable_split
from train_v45_layerwise_compensation import (
    ShortFullPairDataset, symmetric_prototype,
)
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v165_redimnet2_temporal_pyramid import _affinity_loss


def rebuild_indexes(dataset):
    dataset.speakers = sorted(dataset.by_speaker)
    dataset.speaker_to_index = {
        speaker: index for index, speaker in enumerate(dataset.speakers)
    }
    dataset.domain_indexes = {
        domain: [
            index for index, speaker in enumerate(dataset.speakers)
            if speaker.startswith(domain + ":")
        ]
        for domain in ("3d", "ocean", "stcmds", "cv", "child")
    }
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


def add_childmandarin(dataset, root):
    records = {}
    attributes = {}
    for directory in Path(root).iterdir():
        if not directory.is_dir():
            continue
        paths = sorted(directory.glob("*.wav"))
        if len(paths) < 2:
            continue
        speaker = "child:" + directory.name
        records[speaker] = paths
        fields = paths[0].stem.split("_")
        if len(fields) < 3:
            raise ValueError("invalid ChildMandarin filename: %s" % paths[0])
        attributes[speaker] = (fields[1], fields[2])
    if len(records) < 300:
        raise ValueError(
            "expected at least 300 ChildMandarin train speakers, found %d"
            % len(records)
        )
    dataset.by_speaker.update(records)
    rebuild_indexes(dataset)
    groups = defaultdict(list)
    for speaker, values in attributes.items():
        groups[values].append(dataset.speaker_to_index[speaker])
    dataset.child_group_indexes = {
        key: indexes for key, indexes in groups.items() if len(indexes) >= 4
    }
    return len(records), sum(len(paths) for paths in records.values()), {
        "%s_%s" % key: len(indexes) for key, indexes in groups.items()
    }


class FiveDomainChildSampler:
    def __init__(self, dataset, config):
        if int(config["speakers_per_batch"]) != 8:
            raise ValueError("V176 uses an eight-speaker balanced batch")
        if int(config["child_speakers_per_batch"]) != 4:
            raise ValueError("V176 reserves four hard child identities per batch")
        self.dataset = dataset
        self.steps = int(config["steps_per_epoch"])
        self.durations = [float(value) for value in config["duration_probabilities"]]
        self.duration_weights = [
            float(config["duration_probabilities"][value])
            for value in config["duration_probabilities"]
        ]

    def __len__(self):
        return self.steps

    def __iter__(self):
        child_groups = list(self.dataset.child_group_indexes.values())
        for _ in range(self.steps):
            indexes = random.sample(random.choice(child_groups), 4)
            indexes.append(random.choice(self.dataset.domain_indexes["3d"]))
            gender = random.choice(("m", "f"))
            indexes.append(random.choice(
                self.dataset.ocean_gender_indexes[gender]
            ))
            indexes.append(random.choice(self.dataset.domain_indexes["stcmds"]))
            indexes.append(random.choice(self.dataset.domain_indexes["cv"]))
            random.shuffle(indexes)
            duration = random.choices(
                self.durations, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def set_trainable(model, mode):
    model.requires_grad_(False)
    if mode == "head":
        model.eval()
        return
    model.temporal.evidence_pool.requires_grad_(True)
    model.spectral_pool.requires_grad_(True)
    if mode == "upper":
        backbone = model.temporal.encoder.backbone
        for name in ("stage4", "stage5", "fin_wght1d", "head"):
            getattr(backbone, name).requires_grad_(True)
        model.temporal.encoder.pool.requires_grad_(True)
        model.temporal.encoder.bn.requires_grad_(True)
        model.temporal.encoder.linear.requires_grad_(True)
    model.train()
    for module in model.temporal.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_phase(model, teacher, head, loader, config, epochs, mode, start):
    set_trainable(model, mode)
    groups = [{"params": head.parameters(), "lr": config["head_lr"]}]
    refiners = []
    if mode != "head":
        refiners = list(model.temporal.evidence_pool.parameters())
        refiners += list(model.spectral_pool.parameters())
        groups.append({"params": refiners, "lr": config["refiner_lr"]})
    if mode == "upper":
        refiner_ids = {id(parameter) for parameter in refiners}
        upper = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in refiner_ids
        ]
        groups.append({"params": upper, "lr": config["upper_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, epochs * len(loader))
    )
    history = []
    for phase_epoch in range(epochs):
        epoch = start + phase_epoch
        totals = defaultdict(float)
        for step, batch in enumerate(loader, 1):
            (
                _, query_waveform, _, full_waveform,
                _, enrollment_waveform, labels,
            ) = [value.cuda(non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_query = teacher(waveforms=query_waveform)
                teacher_full = teacher(waveforms=full_waveform)
                teacher_enrollment = teacher(waveforms=enrollment_waveform)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                query = model(waveforms=query_waveform)
                full = model(waveforms=full_waveform)
                enrollment = model(waveforms=enrollment_waveform)
                classification = 0.5 * (
                    F.cross_entropy(head(query, labels), labels)
                    + F.cross_entropy(head(enrollment, labels), labels)
                )
                query_norm = F.normalize(query.float(), dim=1)
                full_norm = F.normalize(full.float(), dim=1)
                enrollment_norm = F.normalize(enrollment.float(), dim=1)
                teacher_query_norm = F.normalize(teacher_query.float(), dim=1)
                teacher_full_norm = F.normalize(teacher_full.float(), dim=1)
                teacher_enrollment_norm = F.normalize(
                    teacher_enrollment.float(), dim=1
                )
                prototype = symmetric_prototype(
                    query_norm, enrollment_norm,
                    config["prototype_scale"], config["prototype_margin"],
                )
                completion = (
                    1.0 - F.cosine_similarity(query_norm, teacher_full_norm)
                ).mean()
                coordinate = 0.5 * (
                    (1.0 - F.cosine_similarity(
                        full_norm, teacher_full_norm
                    )).mean()
                    + (1.0 - F.cosine_similarity(
                        enrollment_norm, teacher_enrollment_norm
                    )).mean()
                )
                duration = (
                    1.0 - F.cosine_similarity(query_norm, full_norm)
                ).mean()
                affinity = _affinity_loss(
                    torch.cat((query, enrollment), dim=0),
                    torch.cat((teacher_query_norm, teacher_enrollment_norm), dim=0),
                )
                loss = (
                    config["classification_weight"] * classification
                    + config["prototype_weight"] * prototype
                    + config["completion_weight"] * completion
                    + config["coordinate_weight"] * coordinate
                    + config["duration_weight"] * duration
                    + config["affinity_weight"] * affinity
                )
            loss.backward()
            if mode != "head" and phase_epoch == 0 and step == 1:
                gradients = [
                    parameter.grad for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients or not all(
                    torch.isfinite(value).all() for value in gradients
                ):
                    raise RuntimeError("V176 trainable gradients are invalid")
                print("gradient_check=%s_all_trainable_finite" % mode, flush=True)
            parameters = [p for group in groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            scheduler.step()
            values = {
                "loss": loss.item(), "classification": classification.item(),
                "prototype": prototype.item(), "completion": completion.item(),
                "coordinate": coordinate.item(), "duration": duration.item(),
                "affinity": affinity.item(),
            }
            for name, value in values.items():
                totals[name] += value
            if step % 25 == 0 or len(loader) == 1:
                temporal_gate = torch.sigmoid(
                    model.temporal.evidence_pool.residual_logit.detach()
                ).item()
                spectral_gate = torch.sigmoid(
                    model.spectral_pool.residual_logit.detach()
                ).item()
                print(
                    "epoch=%d phase=%s step=%d loss=%.4f cls=%.4f "
                    "proto=%.4f completion=%.4f coordinate=%.4f duration=%.4f "
                    "gates=%.4f/%.4f"
                    % (
                        epoch, mode, step, values["loss"],
                        values["classification"], values["prototype"],
                        values["completion"], values["coordinate"],
                        values["duration"], temporal_gate, spectral_gate,
                    ), flush=True,
                )
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(
            epoch=epoch, phase=mode, steps=len(loader),
            temporal_fraction=torch.sigmoid(
                model.temporal.evidence_pool.residual_logit.detach()
            ).item(),
            spectral_fraction=torch.sigmoid(
                model.spectral_pool.residual_logit.detach()
            ).item(),
        )
        history.append(stats)
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v176_child_dual_axis_redimnet2.yaml"
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
        "--output-dir", default="outputs/v176_child_dual_axis_redimnet2"
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
    sampler = FiveDomainChildSampler(dataset, config)
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
            "single ReDimNet2-B6 with jointly trained temporal receptive-field "
            "pyramid and spectral evidence pooling"
        ),
        "training_domains": [
            "ChildMandarin train", "3D-Speaker", "Speechocean train",
            "ST-CMDS", "Common Voice 17 zh-CN train",
        ],
        "training_speakers": len(dataset.speakers),
        "child_train_speakers": child_speakers,
        "child_train_items": child_items,
        "child_groups": child_groups,
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
