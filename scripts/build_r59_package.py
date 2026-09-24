#!/usr/bin/env python3
"""Build an R59 runtime from the verified package template and a V364 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEMPLATE_SHA256 = (
    "415856babca0285f63c052acbd74e79a23c46e688bd708d5f5341137efc1c7a3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path,
        default=ROOT / "artifacts/releases/miwu-v386-v364-grl-sphere-fixed-input-official-r59.tar.gz",
    )
    parser.add_argument(
        "--v364-checkpoint", type=Path,
        default=ROOT / "outputs/v364_wide_cross_residual/final.ckpt",
    )
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist/miwu-r59-retrained.tar.gz")
    args = parser.parse_args()
    template = args.template.resolve()
    checkpoint = args.v364_checkpoint.resolve()
    output = args.output.resolve()
    if sha256(template) != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("R59 template SHA-256 mismatch")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    with tempfile.TemporaryDirectory(prefix="miwu-r59-") as temporary:
        stage = Path(temporary)
        with tarfile.open(template, "r:gz") as archive:
            archive.extractall(stage)
        roots = [path for path in stage.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Expected one project root in the R59 template")
        project = roots[0]
        shutil.copy2(
            ROOT / "deployment/sphere_fusion/shared_w2v_heads.py",
            project / "shared_w2v_heads.py",
        )
        shutil.copy2(checkpoint, project / "model/depth_time_grl.ckpt")
        metadata = {
            "architecture": "R59/V386",
            "v364_checkpoint_sha256": sha256(checkpoint),
            "template_sha256": EXPECTED_TEMPLATE_SHA256,
        }
        (project / "REPRODUCTION.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for path in project.rglob("*"):
            path.chmod(0o777 if path.is_dir() or path.name == "start.sh" else 0o666)
        with tarfile.open(output, "w:gz", format=tarfile.GNU_FORMAT) as archive:
            archive.add(project, arcname=project.name)
    print(str(output))
    print("sha256=" + sha256(output))


if __name__ == "__main__":
    main()
