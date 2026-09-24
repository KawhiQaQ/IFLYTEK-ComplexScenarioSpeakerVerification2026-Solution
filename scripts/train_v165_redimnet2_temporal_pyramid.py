#!/usr/bin/env python3
"""Train a duration-completing temporal-pyramid ReDimNet2 single model."""

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

from miwu.palabra_redimnet2_model import load_palabra_multicorpus_redimnet2
from miwu.palabra_redimnet2_temporal_pyramid import (
    load_palabra_redimnet2_temporal_pyramid,
)
from train_v1 import AAMHead, stable_split
from train_v45_layerwise_compensation import (
    ShortFullPairDataset, symmetric_prototype,
)
from train_v55_four_domain_local_temporal import (
    FourDomainSpeakerSampler, add_commonvoice,
)
from train_v7_pyramid_large import add_stcmds


def _affinity_loss(student, teacher):
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)
    mask = ~torch.eye(
        student.shape[0], dtype=torch.bool, device=student.device
    )
    return F.smooth_l1_loss(
        (student @ student.T)[mask], (teacher @ teacher.T)[mask], beta=0.05
    )


def train_phase(model, teacher, head, loader, config, epochs, mode, start):
    model.set_trainable(upper=mode == "upper")
    if mode == "head":
        model.requires_grad_(False)
        model.eval()
    groups = [{"params": head.parameters(), "lr": config["head_lr"]}]
    if mode != "head":
        refiner = list(model.evidence_pool.parameters())
        groups.append({"params": refiner, "lr": config["refiner_lr"]})
        if mode == "upper":
            refiner_ids = {id(parameter) for parameter in refiner}
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
                prototype = symmetric_prototype(
                    query_norm, enrollment_norm,
                    config["prototype_scale"], config["prototype_margin"],
                )
                completion = (
                    1.0 - F.cosine_similarity(
                        query_norm, F.normalize(teacher_full.float(), dim=1)
                    )
                ).mean()
                coordinate = 0.5 * (
                    (1.0 - F.cosine_similarity(
                        full_norm, F.normalize(teacher_full.float(), dim=1)
                    )).mean()
                    + (1.0 - F.cosine_similarity(
                        enrollment_norm,
                        F.normalize(teacher_enrollment.float(), dim=1),
                    )).mean()
                )
                duration = (
                    1.0 - F.cosine_similarity(query_norm, full_norm)
                ).mean()
                student_bank = torch.cat((query, enrollment), dim=0)
                teacher_bank = torch.cat(
                    (teacher_query, teacher_enrollment), dim=0
                )
                affinity = _affinity_loss(student_bank, teacher_bank)
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
                refiner_gradients = [
                    parameter.grad for parameter in model.evidence_pool.parameters()
                    if parameter.grad is not None
                ]
                if not refiner_gradients or not all(
                    torch.isfinite(value).all() for value in refiner_gradients
                ):
                    raise RuntimeError("temporal-pyramid gradients are invalid")
                if mode == "upper":
                    upper_gradients = [
                        parameter.grad for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                        and not name.startswith("evidence_pool.")
                        and parameter.grad is not None
                    ]
                    if not upper_gradients:
                        raise RuntimeError("upper ReDimNet2 stages received no gradient")
                print("gradient_check=%s_finite" % mode, flush=True)
            parameters = [
                parameter for group in groups for parameter in group["params"]
            ]
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
                print(
                    "epoch=%d phase=%s step=%d loss=%.4f cls=%.4f "
                    "proto=%.4f completion=%.4f coordinate=%.4f duration=%.4f gate=%.4f"
                    % (
                        epoch, mode, step, values["loss"],
                        values["classification"], values["prototype"],
                        values["completion"], values["coordinate"],
                        values["duration"],
                        torch.sigmoid(
                            model.evidence_pool.residual_logit.detach()
                        ).item(),
                    ), flush=True,
                )
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(
            epoch=epoch, phase=mode, steps=len(loader),
            residual_fraction=torch.sigmoid(
                model.evidence_pool.residual_logit.detach()
            ).item(),
        )
        history.append(stats)
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v165_redimnet2_temporal_pyramid.yaml")
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS")
    parser.add_argument("--commonvoice-root", default="data/processed/commonvoice17-train")
    parser.add_argument("--output-dir", default="outputs/v165_redimnet2_temporal_pyramid")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps:
        config["steps_per_epoch"] = args.max_steps
    if args.smoke:
        config["head_epochs"] = 0
        config["refiner_epochs"] = 1
        config["upper_epochs"] = 0
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
    sampler = FourDomainSpeakerSampler(
        dataset, config["speakers_per_batch"],
        config["duration_probabilities"], config["steps_per_epoch"],
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    model = load_palabra_redimnet2_temporal_pyramid(
        args.redimnet2_checkpoint
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
    torch.save({
        "state_dict": {
            name: value for name, value in model.state_dict().items()
            if name.startswith("evidence_pool.")
        }
    }, output / "refiner.ckpt")
    torch.save({"state_dict": model.state_dict()}, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": (
            "single ReDimNet2-B6 with integrated three-receptive-field, "
            "two-query temporal evidence pool and protected residual coordinates"
        ),
        "training_speakers": len(dataset.speakers),
        "three_d_dev_speakers": len(dev_speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "commonvoice_speakers": cv_speakers,
        "commonvoice_items": cv_items,
        "commonvoice_dev_speakers": 260,
        "speechocean_test_used_for_training": False,
        "single_model": True,
        "config": config,
    }, indent=2))


if __name__ == "__main__":
    main()
