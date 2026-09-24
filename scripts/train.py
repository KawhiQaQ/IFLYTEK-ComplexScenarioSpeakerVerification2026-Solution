#!/usr/bin/env python3
"""Train the released wide-residual and cross-encoder speaker heads."""
from __future__ import annotations
import argparse
import sys
import train_r59

STAGES = {
    "all": "all",
    "prototypes": "prototypes",
    "wide-residual": "v360",
    "cross-encoder": "v364",
    "package": "package",
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="all")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    forwarded = [
        sys.argv[0], "--stage", STAGES[args.stage],
        "--data-root", args.data_root,
        "--output-root", args.output_root,
    ]
    if args.smoke:
        forwarded.append("--smoke")
    if args.dry_run:
        forwarded.append("--dry-run")
    sys.argv = forwarded
    train_r59.main()

if __name__ == "__main__":
    main()
