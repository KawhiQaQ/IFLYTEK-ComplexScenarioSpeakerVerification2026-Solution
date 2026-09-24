#!/usr/bin/env python3
"""Run the exact R59 runtime on an official-layout WAV directory."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing data_sceneryN/audio/*.wav.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path,
                        default=ROOT / "inference/r59")
    args = parser.parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    runtime = args.runtime.resolve()
    required = [runtime / "run.py", runtime / "model/encoder.ckpt",
                runtime / "model/depth_time_grl.ckpt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete R59 runtime: " + ", ".join(missing))
    wavs = list(input_root.glob("data_scenery*/audio/**/*.wav"))
    if not wavs:
        raise FileNotFoundError(
            "Expected at least one WAV under data_sceneryN/audio/: " + str(input_root)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["GAME823_INPUT_DIR"] = str(input_root)
    environment["OUTPUT_DIR"] = str(output_root)
    subprocess.run(
        [sys.executable, "-u", str(runtime / "run.py")],
        cwd=runtime, env=environment, check=True,
    )
    archive = output_root / "submit.zip"
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("R59 did not create a non-empty submit.zip")
    print(archive)


if __name__ == "__main__":
    main()
