#!/usr/bin/env python3
"""Install the separately distributed model assets into this repository."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "miwu-v386-v364-grl-sphere-fixed-input-official-r59.tar.gz"
EXPECTED_SHA256 = "415856babca0285f63c052acbd74e79a23c46e688bd708d5f5341137efc1c7a3"

def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    source = args.weights.resolve()
    checkpoint_source = source / "checkpoints"
    release_source = source / "artifacts/releases"
    archive_source = release_source / ARCHIVE
    for required in (checkpoint_source, archive_source):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256(archive_source) != EXPECTED_SHA256:
        raise RuntimeError("Final model archive SHA-256 mismatch")

    shutil.copytree(checkpoint_source, ROOT / "checkpoints", dirs_exist_ok=True)
    shutil.copytree(release_source, ROOT / "artifacts/releases", dirs_exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="miwu-weights-") as temporary:
        stage = Path(temporary)
        with tarfile.open(archive_source, "r:gz") as archive:
            archive.extractall(stage)
        roots = [path for path in stage.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Unexpected model archive layout")
        shutil.copytree(roots[0], ROOT / "inference/r59", dirs_exist_ok=True)
    print("weights_installed=" + str(ROOT))
    print("model_sha256=" + EXPECTED_SHA256)

if __name__ == "__main__":
    main()
