#!/usr/bin/env python3
"""Train V176 against acoustically nearest training-speaker prototypes.

The mining index is built only from training recordings with the frozen V176
teacher.  Each batch contains two four-speaker acoustic-neighbour clusters; one
cluster is always same-age/same-gender ChildMandarin.  A short corrupted query
is matched to a prototype made from two different recordings of the speaker.
"""

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

from miwu.encoder import load_audio
from miwu.palabra_redimnet2_dual_axis import load_palabra_redimnet2_dual_axis
from train_v1 import AAMHead, crop_or_repeat, stable_split
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v165_redimnet2_temporal_pyramid import _affinity_loss
from train_v176_child_dual_axis_redimnet2 import (
    add_childmandarin, rebuild_indexes, set_trainable,
)
from train_v179_age_span_dual_axis_redimnet2 import ocean_age_groups
from train_v190_cross_distance_unified_redimnet2 import (
    CrossDistanceShortFullPairDataset,
)


@torch.inference_mode()
def frozen_speaker_prototypes(dataset, teacher, utterances):
    """Average up to ``utterances`` training recordings for every identity."""
    items = []
    for index, speaker in enumerate(dataset.speakers):
        paths = sorted(dataset.by_speaker[speaker])
        if len(paths) > utterances:
            positions = np.linspace(0, len(paths) - 1, utterances).round().astype(int)
            paths = [paths[position] for position in positions]
        items.extend((index, path) for path in paths)

    sums = torch.zeros(len(dataset.speakers), 192)
    counts = torch.zeros(len(dataset.speakers), 1)
    batch_indexes, batch_waves = [], []

    def flush():
        if not batch_waves:
            return
        waveforms = torch.stack(batch_waves).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = F.normalize(teacher(waveforms=waveforms).float(), dim=1)
        for row, speaker_index in enumerate(batch_indexes):
            sums[speaker_index] += embeddings[row].cpu()
            counts[speaker_index] += 1
        batch_indexes.clear()
        batch_waves.clear()

    for speaker_index, path in items:
        batch_indexes.append(speaker_index)
        batch_waves.append(crop_or_repeat(load_audio(path), 48000))
        if len(batch_waves) == 48:
            flush()
    flush()
    if not torch.all(counts > 0):
        raise RuntimeError("speaker prototype extraction missed an identity")
    prototypes = F.normalize(sums / counts, dim=1)
    if not torch.isfinite(prototypes).all():
        raise RuntimeError("speaker prototypes contain non-finite values")
    return prototypes


def nearest_by_group(prototypes, groups, neighbor_count):
    """Find acoustic neighbours only inside an allowed demographic/domain group."""
    nearest = {}
    for group in groups:
        group = sorted(set(group))
        if len(group) < 4:
            continue
        values = prototypes[group]
        similarity = values @ values.T
        similarity.fill_diagonal_(-2.0)
        count = min(neighbor_count, len(group) - 1)
        positions = similarity.topk(count, dim=1).indices.tolist()
        for row, speaker_index in enumerate(group):
            nearest[speaker_index] = [group[position] for position in positions[row]]
    return nearest


def mining_groups(dataset, young_groups, adult_indexes):
    child_groups = list(dataset.child_group_indexes.values())
    adult = set(adult_indexes)
    ocean_adult_groups = [
        [index for index in dataset.ocean_gender_indexes[gender] if index in adult]
        for gender in ("m", "f")
    ]
    groups = {
        "child": child_groups,
        "ocean": list(young_groups) + ocean_adult_groups,
        "3d": [dataset.domain_indexes["3d"]],
        "stcmds": [dataset.domain_indexes["stcmds"]],
        "cv": [dataset.domain_indexes["cv"]],
    }
    return groups


class HardPrototypeSampler:
    """One child neighbour cluster plus one rotating wide-domain cluster."""

    def __init__(self, dataset, config, groups, prototypes):
        if int(config["speakers_per_batch"]) != 8:
            raise ValueError("V244 uses two four-speaker clusters")
        self.dataset = dataset
        self.steps = int(config["steps_per_epoch"])
        self.durations = [float(value) for value in config["duration_probabilities"]]
        self.duration_weights = [
            float(config["duration_probabilities"][value])
            for value in config["duration_probabilities"]
        ]
        self.topk = int(config["hard_sample_topk"])
        self.nearest = {
            domain: nearest_by_group(
                prototypes, domain_groups, int(config["hard_neighbor_count"])
            )
            for domain, domain_groups in groups.items()
        }
        self.anchors = {
            domain: sorted(mapping) for domain, mapping in self.nearest.items()
        }
        for domain, anchors in self.anchors.items():
            if not anchors:
                raise ValueError("no hard-mining anchors for %s" % domain)

    def __len__(self):
        return self.steps

    def cluster(self, domain):
        anchor = random.choice(self.anchors[domain])
        candidates = self.nearest[domain][anchor][:self.topk]
        return [anchor] + random.sample(candidates, 3)

    def __iter__(self):
        wide_domains = ("3d", "ocean", "stcmds", "cv")
        for step in range(self.steps):
            indexes = self.cluster("child")
            indexes += self.cluster(wide_domains[step % len(wide_domains)])
            random.shuffle(indexes)
            duration = random.choices(
                self.durations, weights=self.duration_weights, k=1
            )[0]
            yield [(index, duration) for index in indexes]


def symmetric_hard_prototype(query, prototype, scale, margin):
    logits = float(scale) * (query @ prototype.T)
    indexes = torch.arange(logits.shape[0], device=logits.device)
    logits[indexes, indexes] -= float(scale) * float(margin)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


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
            (_, query_waveform, _, full_waveform, _, enrollment_waveform, labels) = [
                value.cuda(non_blocking=True) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_query = teacher(waveforms=query_waveform)
                teacher_full = teacher(waveforms=full_waveform)
                teacher_enrollment = teacher(waveforms=enrollment_waveform)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                query = model(waveforms=query_waveform)
                full = model(waveforms=full_waveform)
                enrollment = model(waveforms=enrollment_waveform)
                query_norm = F.normalize(query.float(), dim=1)
                full_norm = F.normalize(full.float(), dim=1)
                enrollment_norm = F.normalize(enrollment.float(), dim=1)
                student_prototype = F.normalize(full_norm + enrollment_norm, dim=1)
                teacher_query_norm = F.normalize(teacher_query.float(), dim=1)
                teacher_full_norm = F.normalize(teacher_full.float(), dim=1)
                teacher_enrollment_norm = F.normalize(teacher_enrollment.float(), dim=1)
                teacher_prototype = F.normalize(
                    teacher_full_norm + teacher_enrollment_norm, dim=1
                )

                classification = 0.5 * (
                    F.cross_entropy(head(query, labels), labels)
                    + F.cross_entropy(head(student_prototype, labels), labels)
                )
                prototype = symmetric_hard_prototype(
                    query_norm, student_prototype,
                    config["prototype_scale"], config["prototype_margin"],
                )
                completion = (
                    1.0 - F.cosine_similarity(query_norm, teacher_prototype)
                ).mean()
                coordinate = 0.5 * (
                    (1.0 - F.cosine_similarity(full_norm, teacher_full_norm)).mean()
                    + (1.0 - F.cosine_similarity(
                        enrollment_norm, teacher_enrollment_norm
                    )).mean()
                )
                duration = (
                    1.0 - F.cosine_similarity(query_norm, student_prototype)
                ).mean()
                affinity = _affinity_loss(
                    torch.cat((query_norm, student_prototype), dim=0),
                    torch.cat((teacher_query_norm, teacher_prototype), dim=0),
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
                    raise RuntimeError("V244 trainable gradients are invalid")
                print("gradient_check=%s_all_trainable_finite" % mode, flush=True)
            parameters = [parameter for group in groups for parameter in group["params"]]
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
                    "hard_proto=%.4f completion=%.4f coordinate=%.4f duration=%.4f"
                    % (
                        epoch, mode, step, values["loss"],
                        values["classification"], values["prototype"],
                        values["completion"], values["coordinate"], values["duration"],
                    ), flush=True,
                )
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(epoch=epoch, phase=mode, steps=len(loader))
        history.append(stats)
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v244_hard_prototype_redimnet2.yaml")
    parser.add_argument("--redimnet2-checkpoint", required=True)
    parser.add_argument("--child-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument("--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS")
    parser.add_argument("--commonvoice-root", default="data/processed/commonvoice17-train")
    parser.add_argument("--childmandarin-root", default="data/raw/childmandarin/train")
    parser.add_argument("--output-dir", default="outputs/v244_hard_prototype_redimnet2")
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
        config["prototype_utterances"] = 1
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
    young_groups, ocean_adults, ocean_groups = ocean_age_groups(dataset, args.ocean_root)

    model = load_palabra_redimnet2_dual_axis(
        args.redimnet2_checkpoint, None, args.child_checkpoint
    ).cuda()
    teacher = load_palabra_redimnet2_dual_axis(
        args.redimnet2_checkpoint, None, args.child_checkpoint
    ).eval().requires_grad_(False)
    prototypes = frozen_speaker_prototypes(
        dataset, teacher, int(config["prototype_utterances"])
    )
    groups = mining_groups(dataset, young_groups, ocean_adults)
    sampler = HardPrototypeSampler(dataset, config, groups, prototypes)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    print(
        "mining_ready speakers=%d child_anchors=%d domains=%s" % (
            len(dataset.speakers), len(sampler.anchors["child"]),
            ",".join("%s:%d" % (name, len(indexes)) for name, indexes in sampler.anchors.items()),
        ), flush=True,
    )
    head = AAMHead(
        len(dataset.speakers), dimension=192,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    history, start = [], 1
    for mode, key in (("head", "head_epochs"), ("refiner", "refiner_epochs"), ("upper", "upper_epochs")):
        epochs = int(config[key])
        if epochs:
            history += train_phase(model, teacher, head, loader, config, epochs, mode, start)
            start += epochs

    torch.save({"state_dict": model.state_dict()}, output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "single V176 dual-axis ReDimNet2 trained by multi-recording hard prototypes",
        "initialization": "V176 child dual-axis model",
        "teacher": "frozen V176 child dual-axis model",
        "mining": "training-only V176 prototypes; same-demographic/domain acoustic neighbours",
        "batch_composition": "four same-age/gender ChildMandarin neighbours plus four neighbours from a rotating wide domain",
        "training_speakers": len(dataset.speakers),
        "child_train_speakers": child_speakers,
        "child_train_items": child_items,
        "child_groups": child_groups,
        "speechocean_young_groups": ocean_groups,
        "speechocean_adult_speakers": len(ocean_adults),
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
