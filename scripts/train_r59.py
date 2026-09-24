#!/usr/bin/env python3
"""Reproduce the task-specific R59 training pipeline with the released setup."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "checkpoints/pretrained/r45_runtime"
TRAINING = ROOT / "checkpoints/training"


def execute(command: list[str], environment: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


def require(paths: list[Path]) -> None:
    missing = []
    for path in paths:
        if path.exists():
            continue
        try:
            missing.append(str(path.relative_to(ROOT)))
        except ValueError:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing reproduction assets: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "prototypes", "v360", "v364", "package"),
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one optimization step for code/environment validation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate assets and print commands without using a GPU.")
    args = parser.parse_args()

    data = args.data_root.resolve()
    output = args.output_root.resolve()
    prototypes = output / "v328_corrected_prototypes.pt"
    v360 = output / ("smoke/v360" if args.smoke else "v360_wide_adapter")
    v364 = output / ("smoke/v364" if args.smoke else "v364_wide_cross_residual")

    required = [
        RUNTIME / "model/w2vbert2_shards/manifest.json",
        RUNTIME / "model/w2vbert2_config/config.json",
        RUNTIME / "model/w2vbert2_grl_head.ckpt",
        RUNTIME / "model/redimnet2.ckpt",
        TRAINING / "v302_final.ckpt",
        TRAINING / "v324_recording_prototypes.pt",
        data / "processed/3dspeaker",
        data / "processed/speechocean",
        data / "processed/stcmds/ST-CMDS-20170001_1-OS",
        data / "processed/commonvoice17-train",
        data / "raw/childmandarin/train",
    ]
    require(required)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    environment["MIWU_V302_CHECKPOINT"] = str(TRAINING / "v302_final.ckpt")
    environment["MIWU_REDIMNET2_CHECKPOINT"] = str(
        RUNTIME / "model/redimnet2.ckpt"
    )

    common_data = [
        "--three-d-root", str(data / "processed/3dspeaker"),
        "--ocean-root", str(data / "processed/speechocean"),
        "--stcmds-root", str(data / "processed/stcmds/ST-CMDS-20170001_1-OS"),
        "--commonvoice-root", str(data / "processed/commonvoice17-train"),
        "--childmandarin-root", str(data / "raw/childmandarin/train"),
    ]
    encoder = [
        "--checkpoint", str(RUNTIME / "model/w2vbert2_shards"),
        "--config-directory", str(RUNTIME / "model/w2vbert2_config"),
        "--head-checkpoint", str(RUNTIME / "model/w2vbert2_grl_head.ckpt"),
    ]

    stages = {args.stage} if args.stage != "all" else {
        "prototypes", "v360", "v364", "package"
    }
    if "prototypes" in stages and not prototypes.exists():
        execute([
            sys.executable, "-u", "scripts/mine_v328_corrected_prototypes.py",
            "--runtime", str(RUNTIME),
            "--teacher-checkpoint", str(TRAINING / "v302_final.ckpt"),
            "--legacy-prototypes", str(TRAINING / "v324_recording_prototypes.pt"),
            "--output", str(prototypes),
            "--report", str(ROOT / "reports/v328_corrected_prototypes.json"),
            *common_data,
        ], environment, args.dry_run)
    elif "prototypes" in stages:
        print("reuse=" + str(prototypes), flush=True)

    if "v360" in stages:
        if not args.dry_run:
            require([prototypes])
        command = [
            sys.executable, "-u", "scripts/train_v360_wide_adapter.py",
            *encoder, *common_data,
            "--prototype-file", str(prototypes),
            "--output-dir", str(v360),
        ]
        if args.smoke:
            command.append("--smoke")
        execute(command, environment, args.dry_run)

    environment["MIWU_V360_CHECKPOINT"] = str(v360 / "final.ckpt")
    if "v364" in stages:
        if not args.dry_run:
            require([prototypes, v360 / "final.ckpt"])
        command = [
            sys.executable, "-u", "scripts/train_v364_wide_cross_residual.py",
            *encoder, *common_data,
            "--prototype-file", str(prototypes),
            "--output-dir", str(v364),
        ]
        if args.smoke:
            command.append("--smoke")
        execute(command, environment, args.dry_run)

    if "package" in stages and not args.smoke:
        checkpoint = (
            v364 / "final.ckpt"
            if (v364 / "final.ckpt").exists()
            else TRAINING / "v364_final.ckpt"
        )
        execute([
            sys.executable, "scripts/build_r59_package.py",
            "--v364-checkpoint", str(checkpoint),
        ], environment, args.dry_run)


if __name__ == "__main__":
    main()
