#!/usr/bin/env python3
"""Train layerwise short-to-full feature compensation from a frozen V21."""

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

from miwu.encoder import fbank, load_audio
from miwu.source_model import (
    LayerwiseShortToFullCompensation,
    load_layerwise_compensation_model,
    load_nuisance_model,
)
from train_v1 import AAMHead, corrupt, crop_or_repeat, stable_split
from train_v5_pyramid import SpeakerPairDataset
from train_v7_pyramid_large import ThreeDomainSpeakerSampler, add_stcmds


class ShortFullPairDataset(SpeakerPairDataset):
    """Return a short crop, its clean same-file context, and another utterance."""

    def __getitem__(self, request):
        speaker_index, duration = request
        speaker = self.speakers[speaker_index]
        paths = self.by_speaker[speaker]
        query_path = random.choice(paths)
        enrollment_candidates = [path for path in paths if path != query_path]
        if speaker.startswith("stcmds:"):
            cross_device = [
                path for path in enrollment_candidates
                if path.stem[14:15] != query_path.stem[14:15]
            ]
            if cross_device:
                enrollment_candidates = cross_device
        enrollment_path = random.choice(enrollment_candidates)
        full_waveform = crop_or_repeat(load_audio(query_path), 48000)
        query_waveform = corrupt(crop_or_repeat(
            full_waveform, int(duration * 16000)
        ))
        enrollment_waveform = crop_or_repeat(
            load_audio(enrollment_path), 48000
        )
        return (
            fbank(query_waveform), query_waveform,
            fbank(full_waveform), full_waveform,
            fbank(enrollment_waveform), enrollment_waveform,
            self.speaker_to_index[speaker],
        )


class BackboneStageCapture:
    def __init__(self, model):
        self.values = {}
        modules = (
            ("stage1", model.backbone.layer1),
            ("stage2", model.backbone.layer2),
            ("stage3", model.backbone.layer3),
            ("stage4", model.backbone.fuse34),
        )
        self.handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in modules
        ]

    def _hook(self, name):
        def save(_module, _inputs, output):
            self.values[name] = output
        return save

    def stages(self):
        return tuple(
            self.values[name]
            for name in ("stage1", "stage2", "stage3", "stage4")
        )

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def set_trainable(model, joint):
    model.requires_grad_(False)
    if joint:
        if hasattr(model, "trainable_adapter_modules"):
            for module in model.trainable_adapter_modules:
                module.requires_grad_(True)
        elif hasattr(model, "serialized_modules"):
            for module in model.serialized_modules:
                module.requires_grad_(True)
        elif hasattr(model, "duration_routers"):
            for router in model.duration_routers:
                router.requires_grad_(True)
        elif hasattr(model, "frequency_extensions"):
            for extension in model.frequency_extensions:
                extension.requires_grad_(True)
        elif hasattr(model, "spectral_extensions"):
            for extension in model.spectral_extensions:
                extension.requires_grad_(True)
        elif hasattr(model, "local_extensions"):
            for extension in model.local_extensions:
                extension.requires_grad_(True)
        else:
            model.feature_compensators.requires_grad_(True)
    model.train()
    model.backbone.eval()
    model.pyramid.eval()
    model.quality_gate.eval()
    if hasattr(model, "source_encoder"):
        for module in model.source_encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    if not joint and hasattr(model, "feature_compensators"):
        model.feature_compensators.eval()


def symmetric_prototype(query, enrollment, scale, margin):
    similarities = torch.matmul(query, enrollment.T)
    labels = torch.arange(len(query), device=query.device)
    similarities = similarities - torch.eye(
        len(query), dtype=similarities.dtype, device=query.device
    ) * margin
    logits = scale * similarities
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def train_phase(
    model, teacher, capture, head, loader, config, epochs, joint, output, start,
    teacher_returns_statistics=False, layerwise_supervision=True,
    teacher_audio_only=False,
):
    set_trainable(model, joint)
    groups = [{"params": head.parameters(), "lr": config["head_lr"]}]
    if joint:
        groups.append({
            "params": [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ],
            "lr": config["adapter_lr"],
        })
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
                query, query_waveform, full, full_waveform,
                enrollment, enrollment_waveform, labels,
            ) = [value.cuda(non_blocking=True) for value in batch]
            query_lengths = torch.full(
                (len(labels),), query.shape[1], dtype=torch.long,
                device=query.device,
            )
            full_lengths = torch.full(
                (len(labels),), full.shape[1], dtype=torch.long,
                device=query.device,
            )
            enrollment_lengths = torch.full(
                (len(labels),), enrollment.shape[1], dtype=torch.long,
                device=query.device,
            )
            optimizer.zero_grad(set_to_none=True)
            if joint:
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    if not layerwise_supervision:
                        if teacher_audio_only:
                            teacher_query = teacher(full, full_lengths)
                            teacher_enrollment = teacher(
                                enrollment, enrollment_lengths
                            )
                        else:
                            teacher_query = teacher(
                                full, full_lengths, full_waveform, None
                            )
                            teacher_enrollment = teacher(
                                enrollment, enrollment_lengths,
                                enrollment_waveform, None,
                            )
                    elif teacher_returns_statistics:
                        (
                            teacher_query,
                            teacher_query_statistics,
                        ) = teacher(
                            full, full_lengths, full_waveform, None,
                            return_stage_statistics=True,
                        )
                        (
                            teacher_enrollment,
                            teacher_enrollment_statistics,
                        ) = teacher(
                            enrollment, enrollment_lengths,
                            enrollment_waveform, None,
                            return_stage_statistics=True,
                        )
                    else:
                        capture.clear()
                        if teacher_audio_only:
                            teacher_query = teacher(full, full_lengths)
                        else:
                            teacher_query = teacher(
                                full, full_lengths, full_waveform, None
                            )
                        teacher_query_statistics = model.stage_statistics(
                            capture.stages(), full_lengths
                        )
                        capture.clear()
                        if teacher_audio_only:
                            teacher_enrollment = teacher(
                                enrollment, enrollment_lengths
                            )
                        else:
                            teacher_enrollment = teacher(
                                enrollment, enrollment_lengths,
                                enrollment_waveform, None,
                            )
                        teacher_enrollment_statistics = model.stage_statistics(
                            capture.stages(), enrollment_lengths
                        )
                        capture.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if joint and layerwise_supervision:
                    query_embedding, query_statistics = model(
                        query, query_lengths, query_waveform, None,
                        return_stage_statistics=True,
                    )
                    enrollment_embedding, enrollment_statistics = model(
                        enrollment, enrollment_lengths,
                        enrollment_waveform, None,
                        return_stage_statistics=True,
                    )
                else:
                    query_embedding = model(
                        query, query_lengths, query_waveform, None
                    )
                    enrollment_embedding = model(
                        enrollment, enrollment_lengths,
                        enrollment_waveform, None,
                    )
                classification = 0.5 * (
                    F.cross_entropy(head(query_embedding, labels), labels)
                    + F.cross_entropy(
                        head(enrollment_embedding, labels), labels
                    )
                )
                query_normalized = F.normalize(query_embedding, dim=1)
                enrollment_normalized = F.normalize(
                    enrollment_embedding, dim=1
                )
                prototype = symmetric_prototype(
                    query_normalized, enrollment_normalized,
                    config["prototype_scale"], config["prototype_margin"],
                )
                if joint:
                    consistency = 0.5 * (
                        (1.0 - F.cosine_similarity(
                            query_embedding, teacher_query
                        )).mean()
                        + (1.0 - F.cosine_similarity(
                            enrollment_embedding, teacher_enrollment
                        )).mean()
                    )
                    if layerwise_supervision:
                        layerwise = torch.stack([
                            0.5 * (
                                (1.0 - F.cosine_similarity(
                                    student_query, target_query
                                )).mean()
                                + (1.0 - F.cosine_similarity(
                                    student_enrollment, target_enrollment
                                )).mean()
                            )
                            for (
                                student_query, target_query,
                                student_enrollment, target_enrollment,
                            ) in zip(
                                query_statistics, teacher_query_statistics,
                                enrollment_statistics,
                                teacher_enrollment_statistics,
                            )
                        ]).mean()
                    else:
                        layerwise = query_embedding.new_zeros(())
                    if config.get("affinity_weight", 0.0) > 0:
                        student_bank = F.normalize(torch.cat(
                            (query_embedding, enrollment_embedding), dim=0
                        ), dim=1)
                        teacher_bank = F.normalize(torch.cat(
                            (teacher_query, teacher_enrollment), dim=0
                        ), dim=1)
                        student_affinity = torch.matmul(
                            student_bank, student_bank.T
                        )
                        teacher_affinity = torch.matmul(
                            teacher_bank, teacher_bank.T
                        )
                        off_diagonal = ~torch.eye(
                            student_affinity.shape[0], dtype=torch.bool,
                            device=student_affinity.device,
                        )
                        affinity = F.smooth_l1_loss(
                            student_affinity[off_diagonal],
                            teacher_affinity[off_diagonal], beta=0.05,
                        )
                    else:
                        affinity = query_embedding.new_zeros(())
                else:
                    consistency = query_embedding.new_zeros(())
                    layerwise = query_embedding.new_zeros(())
                    affinity = query_embedding.new_zeros(())
                loss = (
                    config["classification_weight"] * classification
                    + config["prototype_weight"] * prototype
                    + config["teacher_weight"] * consistency
                    + config["layerwise_weight"] * layerwise
                    + config.get("affinity_weight", 0.0) * affinity
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                5.0,
            )
            optimizer.step()
            scheduler.step()
            values = {
                "loss": loss.item(),
                "classification": classification.item(),
                "prototype": prototype.item(),
                "consistency": consistency.item(),
                "layerwise": layerwise.item(),
                "affinity": affinity.item(),
            }
            for name, value in values.items():
                totals[name] += value
            if step % 25 == 0:
                print(
                    "epoch=%d phase=%s step=%d loss=%.4f cls=%.4f "
                    "proto=%.4f teacher=%.4f layer=%.4f affinity=%.4f"
                    % (
                        epoch, "joint" if joint else "head", step,
                        values["loss"], values["classification"],
                        values["prototype"], values["consistency"],
                        values["layerwise"], values["affinity"],
                    ),
                    flush=True,
                )
        stats = {name: value / len(loader) for name, value in totals.items()}
        stats.update(
            epoch=epoch, phase="joint" if joint else "head",
            steps=len(loader),
        )
        history.append(stats)
        torch.save(model.state_dict(), output / ("epoch_%02d.ckpt" % epoch))
        print(json.dumps(stats), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v45_layerwise_compensation.yaml"
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--pyramid-checkpoint", required=True)
    parser.add_argument("--quality-checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--excitation-checkpoint", required=True)
    parser.add_argument("--multiresolution-checkpoint", required=True)
    parser.add_argument("--source-gate-checkpoint", required=True)
    parser.add_argument("--tangent-checkpoint", required=True)
    parser.add_argument("--nuisance-checkpoint", required=True)
    parser.add_argument("--three-d-root", default="data/processed/3dspeaker")
    parser.add_argument("--ocean-root", default="data/processed/speechocean")
    parser.add_argument(
        "--stcmds-root", default="data/processed/stcmds/ST-CMDS-20170001_1-OS"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v45_layerwise_compensation"
    )
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
    sampler = ThreeDomainSpeakerSampler(
        dataset, config["speakers_per_batch"],
        config["duration_probabilities"],
        args.max_steps or config["steps_per_epoch"],
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["workers"],
        pin_memory=True, persistent_workers=config["workers"] > 0,
    )
    checkpoints = (
        args.backbone_checkpoint, args.pyramid_checkpoint,
        args.quality_checkpoint, args.source_checkpoint,
        args.excitation_checkpoint, args.multiresolution_checkpoint,
        args.source_gate_checkpoint, args.tangent_checkpoint,
    )
    model = load_layerwise_compensation_model(
        *checkpoints, args.nuisance_checkpoint
    ).train()
    teacher = load_nuisance_model(*checkpoints, args.nuisance_checkpoint).eval()
    teacher.requires_grad_(False)
    capture = BackboneStageCapture(teacher)
    head = AAMHead(
        len(dataset.speakers), dimension=model.output_dimension,
        margin=config["margin"], scale=config["scale"],
    ).cuda()
    history = train_phase(
        model, teacher, capture, head, loader, config,
        config["head_epochs"], False, output, 1,
    )
    history += train_phase(
        model, teacher, capture, head, loader, config,
        config["joint_epochs"], True, output,
        config["head_epochs"] + 1,
    )
    capture.close()
    torch.save(model.state_dict(), output / "final.ckpt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "layerwise same-file short-to-full compensation",
        "parent": str(Path(args.nuisance_checkpoint).resolve()),
        "embedding_dimension": model.output_dimension,
        "single_model": True,
        "training_domains": ["3D-Speaker", "Speechocean train", "ST-CMDS"],
        "training_speakers": len(dataset.speakers),
        "stcmds_speakers": stcmds_speakers,
        "stcmds_items": stcmds_items,
        "three_d_dev_speakers": len(dev_speakers),
        "speechocean_test_used_for_training": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
    }, indent=2))


if __name__ == "__main__":
    main()
