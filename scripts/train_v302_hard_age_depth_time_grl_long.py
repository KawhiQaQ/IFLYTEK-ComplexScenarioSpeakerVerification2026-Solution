#!/usr/bin/env python3
"""Train a depth/time GRL head on age-balanced acoustic hard-negative clusters."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from miwu.tidyvoice_w2vbert2_depth_time_head import DepthTimeGRLW2VBert2
from train_v1 import AAMHead, stable_split
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v176_child_dual_axis_redimnet2 import (
    add_childmandarin, rebuild_indexes,
)
from train_v179_age_span_dual_axis_redimnet2 import ocean_age_groups
from train_v260_age_balanced_residual_redimnet2 import BalancedAgeComplementSampler
from train_v190_cross_distance_unified_redimnet2 import (
    CrossDistanceAgeSampler, CrossDistanceShortFullPairDataset,
)
from collections import defaultdict
import torch.nn.functional as F
from train_v45_layerwise_compensation import symmetric_prototype
from train_v169_w2vbert2_context_mhfa import affinity_loss, hard_impostor_loss

def train(model, head, loader, config, output, smoke=False):
    model.set_trainable()
    iterator = iter(loader)
    head_optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["head_lr"],
        weight_decay=config["weight_decay"],
    )
    for step in range(1, (1 if smoke else config["head_steps"]) + 1):
        batch = next(iterator)
        labels = batch[-1].cuda(non_blocking=True)
        head_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            query = model(waveforms=batch[1])
            enrollment = model(waveforms=batch[5])
        classification = 0.5 * (
            F.cross_entropy(head(query.float(), labels), labels)
            + F.cross_entropy(head(enrollment.float(), labels), labels)
        )
        classification.backward()
        head_optimizer.step()

    model_groups = (model.training_parameter_groups(config["model_lr"])
                    if hasattr(model, "training_parameter_groups") else
                    [{"params": model.head.parameters(), "lr": config["model_lr"]}])
    optimizer = torch.optim.AdamW(model_groups + [
        {"params": head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    steps = 1 if smoke else int(config["steps"])
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
            query, teacher_query = model(
                waveforms=query_waveform, return_parts=True
            )
            full, teacher_full = model(
                waveforms=full_waveform, return_parts=True
            )
            enrollment, teacher_enrollment = model(
                waveforms=enrollment_waveform, return_parts=True
            )
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
        hard = hard_impostor_loss(
            query, enrollment, teacher_query, teacher_enrollment,
            config["hard_margin"],
        )
        completion = (1.0 - F.cosine_similarity(
            query_norm, teacher_full_norm
        )).mean()
        duration = (1.0 - F.cosine_similarity(
            query_norm, full_norm
        )).mean()
        coordinate = 0.5 * (
            (1.0 - F.cosine_similarity(
                full_norm, teacher_full_norm
            )).mean()
            + (1.0 - F.cosine_similarity(
                enrollment_norm, teacher_enrollment_norm
            )).mean()
        )
        affinity = affinity_loss(
            torch.cat((query, enrollment), dim=0),
            torch.cat((teacher_query, teacher_enrollment), dim=0),
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
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        loss.backward()
        if step == 1:
            gradients = [
                value.grad for value in model.head.parameters()
                if value.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(value).all() for value in gradients
            ):
                raise RuntimeError("V293 head gradients are invalid")
            if any(value.grad is not None for value in model.base.parameters()):
                raise RuntimeError("V293 frozen S2/GRL teacher received gradients")
            print("gradient_check=frozen_encoder_teacher_finite_head", flush=True)
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups
             for value in group["params"]], 3.0
        )
        optimizer.step()
        scheduler.step()
        gate_deviation = (1.0 - F.cosine_similarity(query_norm, teacher_query_norm)).mean()
        values = {
            "loss": loss.item(), "classification": classification.item(),
            "prototype": prototype.item(), "hard": hard.item(),
            "completion": completion.item(), "duration": duration.item(),
            "coordinate": coordinate.item(), "affinity": affinity.item(),
            "gate_deviation": gate_deviation.item(),
        }
        for name, value in values.items():
            totals[name] += value
        if step % 20 == 0 or steps == 1:
            print(
                "phase=depth_time_head step=%d loss=%.4f cls=%.4f proto=%.4f "
                "hard=%.4f completion=%.4f coordinate=%.4f gate_dev=%.4f"
                % (step, values["loss"], values["classification"],
                   values["prototype"], values["hard"],
                   values["completion"], values["coordinate"],
                   values["gate_deviation"]), flush=True,
            )
    stats = {name: value / steps for name, value in totals.items()}
    stats["steps"] = steps
    print(json.dumps(stats), flush=True)
    torch.save({
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.head.state_dict().items()
        }
    }, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps([stats], indent=2))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/sphere_fusion_training.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-directory", required=True)
    parser.add_argument("--head-checkpoint", required=True)
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
        "--output-dir", default="outputs/v302_hard_age_depth_time_grl_long"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prototype-file",default="outputs/v294_grl_train_prototypes.pt")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
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
    total_steps = 2 if args.smoke else config["head_steps"] + config["steps"]
    bank=torch.load(args.prototype_file,map_location="cpu")
    assert bank["speakers"]==dataset.speakers and bank["metadata"]["validation_audio_used"] is False
    sampler=BalancedAgeComplementSampler(dataset,{**config,"steps_per_epoch":total_steps},F.normalize(bank["prototypes"].float(),dim=1),young_groups,ocean_adults)
    loader = DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=0 if args.smoke else config["workers"],
        pin_memory=True,
        persistent_workers=not args.smoke and config["workers"] > 0,
    )
    model = DepthTimeGRLW2VBert2(
        args.checkpoint, args.config_directory, args.head_checkpoint
    ).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        value,teacher=model(waveforms=next(iter(loader))[1][:2],return_parts=True)
        error=(value-teacher).abs().max().item()
        print("initial_teacher_max_abs=%.9g"%error,flush=True)
        if getattr(model, "allow_nonidentity_initialization", False):
            cosine_deviation = (
                1.0 - F.cosine_similarity(value.float(), teacher.float())
            ).max().item()
            print(
                "initial_teacher_max_cosine_deviation=%.9g"
                % cosine_deviation, flush=True,
            )
            if (
                not torch.isfinite(value).all()
                or cosine_deviation > model.initial_max_cosine_deviation
            ):
                raise RuntimeError("Bounded parent initialization failed")
        elif error>1e-5:
            raise RuntimeError("Identity initialization failed")
    head = AAMHead(
        len(dataset.speakers), 256, config["margin"], config["scale"]
    ).cuda()
    bank=torch.load(args.prototype_file,map_location="cpu")
    if bank["speakers"]!=dataset.speakers or bank["metadata"].get("validation_audio_used") is not False:
        raise RuntimeError("Training-only prototype provenance or identity order mismatch")
    if hasattr(head, "initialize_from_prototypes"):
        head.initialize_from_prototypes(bank)
    else:
        if bank["prototypes"].shape!=head.weight.shape:raise RuntimeError("Prototype shape mismatch")
        with torch.no_grad():head.weight.copy_(F.normalize(bank["prototypes"].float(),dim=1).cuda())
    print("classifier=%s real_training_prototypes=%d" % (type(head).__name__, len(dataset.speakers)),flush=True)
    train(model, head, loader, config, output, args.smoke)
    (output / "metadata.json").write_text(json.dumps({
        "architecture": (
            "frozen S2 encoder, complete adapted GRL head with depth-time residual interaction"
        ),
        "initialization": "exact V223 S2 plus GRL head",
        "teacher": "frozen V223 relation geometry",
        "batch_composition": (
            "alternating preschool/school four-speaker hard cluster + rotating four-speaker broad-domain cluster"
        ),
        "training_speakers": len(dataset.speakers),
        "child_train_speakers": child_speakers,
        "child_train_items": child_items,
        "child_groups": child_groups,
        "speechocean_young_groups": ocean_groups,
        "speechocean_adult_speakers": len(ocean_adults),
        "three_d_dev_speakers": len(dev_speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "commonvoice_speakers": cv_speakers,
        "commonvoice_items": cv_items,
        "child_dev_test_used_for_training": False,
        "validation_trials_used_for_training": False,
        "online_feedback_used_for_training": False,
        "single_model": True,
        "acoustic_encoder_trainable": False, "complete_speaker_head_trainable": True,
        "trainable_parameters": sum(
            value.numel() for value in model.head.parameters()
        ),
        "classifier_initialization": args.prototype_file,
        "config": config,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
