#!/usr/bin/env python3
"""Train a contextual multi-layer pooling path over frozen W2V-BERT2."""

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

from miwu.tidyvoice_w2vbert2_context_mhfa import (
    load_tidyvoice_w2vbert2_context_mhfa,
)
from train_v1 import AAMHead, stable_split
from train_v45_layerwise_compensation import (
    ShortFullPairDataset, symmetric_prototype,
)
from train_v55_four_domain_local_temporal import (
    FourDomainSpeakerSampler, add_commonvoice,
)
from train_v7_pyramid_large import add_stcmds


def hard_impostor_loss(query, enrollment, base_query, base_enrollment, margin):
    query = F.normalize(query.float(), dim=1)
    enrollment = F.normalize(enrollment.float(), dim=1)
    with torch.no_grad():
        base = F.normalize(base_query.float(), dim=1) @ F.normalize(
            base_enrollment.float(), dim=1
        ).T
        base.fill_diagonal_(-2.0)
        row_negative = base.argmax(dim=1)
        column_negative = base.argmax(dim=0)
    positive = (query * enrollment).sum(dim=1)
    row_score = (query * enrollment[row_negative]).sum(dim=1)
    column_score = (enrollment * query[column_negative]).sum(dim=1)
    return 0.5 * (
        F.relu(margin + row_score - positive).mean()
        + F.relu(margin + column_score - positive).mean()
    )


def affinity_loss(student, teacher):
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)
    mask = ~torch.eye(
        len(student), dtype=torch.bool, device=student.device
    )
    return F.smooth_l1_loss(
        (student @ student.T)[mask], (teacher @ teacher.T)[mask], beta=0.05
    )


def train(model, head, loader, config, output, smoke=False):
    model.set_trainable()
    iterator = iter(loader)
    head_optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["head_lr"],
        weight_decay=config["weight_decay"],
    )
    head_steps = 0 if smoke else config["head_steps"]
    for step in range(1, head_steps + 1):
        batch = next(iterator)
        labels = batch[-1].cuda(non_blocking=True)
        head_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            query = model.base(waveforms=batch[1])
            enrollment = model.base(waveforms=batch[5])
        classification = 0.5 * (
            F.cross_entropy(head(query.float(), labels), labels)
            + F.cross_entropy(head(enrollment.float(), labels), labels)
        )
        classification.backward()
        head_optimizer.step()
        if step % 10 == 0:
            print(
                "phase=head step=%d classification=%.4f"
                % (step, classification.item()), flush=True,
            )

    optimizer = torch.optim.AdamW([
        {"params": model.context_mhfa.parameters(), "lr": config["mhfa_lr"]},
        {"params": head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    steps = 1 if smoke else config["steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    totals = defaultdict(float)
    for step in range(1, steps + 1):
        batch = next(iterator)
        query_waveform, full_waveform, enrollment_waveform = (
            batch[1], batch[3], batch[5]
        )
        labels = batch[-1].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            query, base_query, _ = model(
                waveforms=query_waveform, return_parts=True
            )
            full, base_full, _ = model(
                waveforms=full_waveform, return_parts=True
            )
            enrollment, base_enrollment, _ = model(
                waveforms=enrollment_waveform, return_parts=True
            )
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
        hard = hard_impostor_loss(
            query, enrollment, base_query, base_enrollment,
            config["hard_margin"],
        )
        completion = (
            1.0 - F.cosine_similarity(
                query_norm, F.normalize(base_full.float(), dim=1)
            )
        ).mean()
        duration = (
            1.0 - F.cosine_similarity(query_norm, full_norm)
        ).mean()
        coordinate = 0.5 * (
            (1.0 - F.cosine_similarity(
                full_norm, F.normalize(base_full.float(), dim=1)
            )).mean()
            + (1.0 - F.cosine_similarity(
                enrollment_norm,
                F.normalize(base_enrollment.float(), dim=1),
            )).mean()
        )
        affinity = affinity_loss(
            torch.cat((query, enrollment), dim=0),
            torch.cat((base_query, base_enrollment), dim=0),
        )
        loss = (
            config["classification_weight"] * classification
            + config["prototype_weight"] * prototype
            + config["hard_weight"] * hard
            + config["completion_weight"] * completion
            + config["duration_weight"] * duration
            + config["coordinate_weight"] * coordinate
            + config["affinity_weight"] * affinity
        )
        loss.backward()
        if step == 1:
            gradients = [
                value.grad for value in model.context_mhfa.parameters()
                if value.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(value).all() for value in gradients
            ):
                raise RuntimeError("context-MHFA gradients are invalid")
            if any(
                value.grad is not None for value in model.base.parameters()
            ):
                raise RuntimeError("published W2V-BERT2 received gradients")
            print(
                "gradient_check=frozen_w2vbert2_context_mhfa_finite",
                flush=True,
            )
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            3.0,
        )
        optimizer.step()
        scheduler.step()
        values = {
            "loss": loss.item(), "classification": classification.item(),
            "prototype": prototype.item(), "hard": hard.item(),
            "completion": completion.item(), "duration": duration.item(),
            "coordinate": coordinate.item(), "affinity": affinity.item(),
        }
        for name, value in values.items():
            totals[name] += value
        if step % 10 == 0 or steps == 1:
            print(
                "phase=mhfa step=%d loss=%.4f cls=%.4f proto=%.4f "
                "hard=%.4f completion=%.4f coordinate=%.4f gate=%.4f"
                % (
                    step, values["loss"], values["classification"],
                    values["prototype"], values["hard"],
                    values["completion"], values["coordinate"],
                    torch.sigmoid(model.context_mhfa.residual_logit).item(),
                ), flush=True,
            )
    stats = {name: value / steps for name, value in totals.items()}
    stats.update(
        steps=steps,
        residual_fraction=torch.sigmoid(
            model.context_mhfa.residual_logit.detach()
        ).item(),
    )
    print(json.dumps(stats), flush=True)
    torch.save({
        "state_dict": {
            "context_mhfa." + name: value.detach().cpu()
            for name, value in model.context_mhfa.state_dict().items()
        }
    }, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps([stats], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v169_w2vbert2_context_mhfa.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-directory", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument(
        "--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS"
    )
    parser.add_argument(
        "--commonvoice-root", default="data/processed/commonvoice17-train"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v169_w2vbert2_context_mhfa"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
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
        config["duration_probabilities"],
        config["head_steps"] + config["steps"],
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    model = load_tidyvoice_w2vbert2_context_mhfa(
        args.checkpoint, args.config_directory, device="cuda"
    )
    head = AAMHead(
        len(dataset.speakers), dimension=256,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    train(model, head, loader, config, output, args.smoke)
    (output / "metadata.json").write_text(json.dumps({
        "architecture": (
            "frozen multi-corpus W2V-BERT2 with content-aware 25-layer "
            "factorized key/value attentive statistics residual"
        ),
        "training_speakers": len(dataset.speakers),
        "three_d_dev_speakers": len(dev_speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "commonvoice_speakers": cv_speakers,
        "commonvoice_items": cv_items,
        "commonvoice_dev_speakers": 260,
        "speechocean_test_used_for_training": False,
        "published_encoder_trainable": False,
        "trainable_parameters": sum(
            value.numel() for value in model.context_mhfa.parameters()
        ),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "config": config,
    }, indent=2))


if __name__ == "__main__":
    main()
