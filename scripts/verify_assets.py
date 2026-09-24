#!/usr/bin/env python3
"""Verify the local backup against manifests/assets.sha256."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "manifests/assets.sha256")
    args = parser.parse_args()
    failures = []
    checked = 0
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip(" *")
        path = ROOT / relative
        if not path.is_file():
            failures.append(relative + " (missing)")
        elif digest(path) != expected:
            failures.append(relative + " (SHA-256 mismatch)")
        checked += 1
    if failures:
        raise RuntimeError("Asset verification failed: " + ", ".join(failures))
    print(f"verified={checked}")


if __name__ == "__main__":
    main()
