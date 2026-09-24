#!/usr/bin/env python3
"""Train ReDimNet2 to resolve impostors left ambiguous by frozen R39 peers."""

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

from miwu.palabra_redimnet2_dual_axis import load_palabra_redimnet2_dual_axis
from train_v1 import AAMHead, stable_split
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v165_redimnet2_temporal_pyramid import _affinity_loss
from train_v176_child_dual_axis_redimnet2 import add_childmandarin, rebuild_indexes, set_trainable
from train_v179_age_span_dual_axis_redimnet2 import ocean_age_groups
from train_v190_cross_distance_unified_redimnet2 import CrossDistanceShortFullPairDataset
from train_v244_hard_prototype_redimnet2 import HardPrototypeSampler, mining_groups


def residual_ensemble_loss(query, prototype, complement, scale, margin):
    # Match the deployed R39 allocation: 40% ReDimNet2 and 60% frozen peers.
    student = query @ prototype.T
    frozen = complement @ complement.T
    logits = float(scale) * (0.40 * student + 0.60 * frozen)
    labels = torch.arange(logits.shape[0], device=logits.device)
    logits[labels, labels] -= float(scale) * float(margin)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def standalone_loss(query, prototype, scale, margin):
    logits = float(scale) * (query @ prototype.T)
    labels = torch.arange(logits.shape[0], device=logits.device)
    logits[labels, labels] -= float(scale) * float(margin)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_phase(model, teacher, head, loader, complements, config, epochs, mode, start):
    set_trainable(model, mode)
    groups = [{"params": head.parameters(), "lr": config["head_lr"]}]
    refiners = []
    if mode != "head":
        refiners = list(model.temporal.evidence_pool.parameters()) + list(model.spectral_pool.parameters())
        groups.append({"params": refiners, "lr": config["refiner_lr"]})
    if mode == "upper":
        refiner_ids = {id(p) for p in refiners}
        upper = [p for p in model.parameters() if p.requires_grad and id(p) not in refiner_ids]
        groups.append({"params": upper, "lr": config["upper_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * len(loader)))
    history = []
    for phase_epoch in range(epochs):
        totals = defaultdict(float)
        for step, batch in enumerate(loader, 1):
            _, query_wave, _, full_wave, _, enroll_wave, labels = [x.cuda(non_blocking=True) for x in batch]
            fixed = F.normalize(complements[labels.cpu()].cuda(non_blocking=True), dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tq = teacher(waveforms=query_wave); tf = teacher(waveforms=full_wave)
                te = teacher(waveforms=enroll_wave)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model(waveforms=query_wave); f = model(waveforms=full_wave)
                e = model(waveforms=enroll_wave)
                qn, fn, en = [F.normalize(x.float(), dim=1) for x in (q, f, e)]
                tqn, tfn, ten = [F.normalize(x.float(), dim=1) for x in (tq, tf, te)]
                prototype = F.normalize(fn + en, dim=1)
                teacher_prototype = F.normalize(tfn + ten, dim=1)
                classification = 0.5 * (
                    F.cross_entropy(head(q, labels), labels)
                    + F.cross_entropy(head(prototype, labels), labels)
                )
                residual = residual_ensemble_loss(
                    qn, prototype, fixed, config["prototype_scale"], config["prototype_margin"]
                )
                standalone = standalone_loss(
                    qn, prototype, config["prototype_scale"],
                    config["prototype_margin"]
                )
                completion = (1.0 - F.cosine_similarity(qn, teacher_prototype)).mean()
                coordinate = 0.5 * ((1.0 - F.cosine_similarity(fn, tfn)).mean()
                                    + (1.0 - F.cosine_similarity(en, ten)).mean())
                duration = (1.0 - F.cosine_similarity(qn, prototype)).mean()
                affinity = _affinity_loss(torch.cat((qn, prototype), 0),
                                          torch.cat((tqn, teacher_prototype), 0))
                loss = (config["classification_weight"] * classification
                        + config["residual_weight"] * residual
                        + config["standalone_weight"] * standalone
                        + config["completion_weight"] * completion
                        + config["coordinate_weight"] * coordinate
                        + config["duration_weight"] * duration
                        + config["affinity_weight"] * affinity)
            loss.backward()
            if mode != "head" and phase_epoch == 0 and step == 1:
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not gradients or not all(torch.isfinite(x).all() for x in gradients):
                    raise RuntimeError("V258 trainable gradients are invalid")
                print("gradient_check=%s_all_trainable_finite" % mode, flush=True)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
            optimizer.step(); scheduler.step()
            values = {"loss": loss.item(), "classification": classification.item(),
                      "residual": residual.item(), "standalone": standalone.item(),
                      "completion": completion.item(), "coordinate": coordinate.item(),
                      "duration": duration.item(), "affinity": affinity.item()}
            for name, value in values.items(): totals[name] += value
            if step % 25 == 0 or len(loader) == 1:
                print("epoch=%d phase=%s step=%d loss=%.4f cls=%.4f residual=%.4f own=%.4f coord=%.4f"
                      % (start + phase_epoch, mode, step, values["loss"],
                         values["classification"], values["residual"],
                         values["standalone"], values["coordinate"]), flush=True)
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(epoch=start + phase_epoch, phase=mode, steps=len(loader))
        history.append(stats); print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v258_residual_ensemble_redimnet2.yaml")
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--mining-prototypes", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS")
    parser.add_argument("--commonvoice-root", default="data/processed/commonvoice17-train")
    parser.add_argument("--childmandarin-root", default="data/raw/childmandarin/train")
    parser.add_argument("--output-dir", default="outputs/v258_residual_ensemble_redimnet2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.smoke: config.update(head_epochs=0, refiner_epochs=1, upper_epochs=0, steps_per_epoch=1, workers=0)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train_speakers, dev_speakers = stable_split(Path(args.three_d_root), 180)
    dataset = CrossDistanceShortFullPairDataset(args.three_d_root, train_speakers, args.ocean_root)
    dataset.ocean_root = args.ocean_root
    add_stcmds(dataset, args.stcmds_root); add_commonvoice(dataset, args.commonvoice_root)
    child_speakers, _, _ = add_childmandarin(dataset, args.childmandarin_root); rebuild_indexes(dataset)
    young_groups, adults, _ = ocean_age_groups(dataset, args.ocean_root)
    package = torch.load(args.mining_prototypes, map_location="cpu")
    if package["speakers"] != dataset.speakers: raise RuntimeError("prototype speaker order mismatch")
    complements = package["prototypes"].float().clone()
    complements[:, 3456:3648] = 0
    complements = F.normalize(complements, dim=1)
    sampler = HardPrototypeSampler(dataset, config, mining_groups(dataset, young_groups, adults), complements)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"],
                        pin_memory=True, persistent_workers=config["workers"] > 0)
    model = load_palabra_redimnet2_dual_axis(args.redimnet2_checkpoint, None, args.initial_checkpoint).cuda()
    teacher = load_palabra_redimnet2_dual_axis(
        args.redimnet2_checkpoint, None, args.initial_checkpoint).eval().requires_grad_(False)
    head = AAMHead(len(dataset.speakers), dimension=192,
                   margin=config["margin"], scale=config["scale"]).cuda()
    history, start = [], 1
    for mode, key in (("head", "head_epochs"), ("refiner", "refiner_epochs"), ("upper", "upper_epochs")):
        epochs = int(config[key])
        if epochs:
            history += train_phase(model, teacher, head, loader, complements, config, epochs, mode, start)
            start += epochs
    torch.save({"state_dict": model.state_dict()}, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "V244 residual-ensemble ReDimNet2", "parent": str(args.initial_checkpoint),
        "training_speakers": len(dataset.speakers), "child_speakers": child_speakers,
        "three_d_dev_speakers": len(dev_speakers), "validation_audio_used": False,
        "deployed_training_weights": {"frozen_complement": 0.60, "redimnet2": 0.40},
        "config": config}, indent=2))


if __name__ == "__main__": main()
