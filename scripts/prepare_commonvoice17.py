#!/usr/bin/env python3
"""Build the released speaker-disjoint Common Voice 17 training subset."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_PREFIX = "data/raw/commonvoice17-zhcn-validation"


def stable_order(value: object) -> bytes:
    return hashlib.sha256(str(value).encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--train-speakers", type=int, default=600)
    parser.add_argument("--utterances-per-speaker", type=int, default=12)
    args = parser.parse_args()

    files = sorted(args.root.glob("validation_*.parquet"))
    expected_names = {f"validation_{index}.parquet" for index in range(26)}
    if {path.name for path in files} != expected_names:
        raise RuntimeError(
            "expected validation_0.parquet through validation_25.parquet"
        )

    locations = defaultdict(list)
    for path in files:
        clients = pq.read_table(path, columns=["client_id"]).column(
            "client_id"
        ).to_pylist()
        for row, client in enumerate(clients):
            locations[client].append((path, row))

    speakers = sorted(locations, key=stable_order)
    train_speakers = speakers[: args.train_speakers]
    dev_speakers = speakers[args.train_speakers :]
    if not train_speakers or not dev_speakers:
        raise RuntimeError("both Common Voice splits must be non-empty")

    args.split_root.mkdir(parents=True, exist_ok=True)
    (args.split_root / "train_speakers.txt").write_text(
        "\n".join(train_speakers) + "\n"
    )
    (args.split_root / "dev_speakers.txt").write_text(
        "\n".join(dev_speakers) + "\n"
    )

    rows_by_file = defaultdict(list)
    for speaker in train_speakers:
        selected = sorted(
            locations[speaker],
            key=lambda item: stable_order(
                f"{SOURCE_PREFIX}/{item[0].name}:{item[1]}"
            ),
        )[: args.utterances_per_speaker]
        if len(selected) < 2:
            raise RuntimeError(
                "speaker has fewer than two recordings: " + speaker
            )
        for path, row in selected:
            rows_by_file[path].append((row, speaker))

    written = 0
    for path, wanted in rows_by_file.items():
        column = pq.read_table(path, columns=["audio"]).column("audio")
        for row, speaker in wanted:
            value = column[row].as_py()
            suffix = Path(value.get("path") or "recording.mp3").suffix or ".mp3"
            destination = (
                args.output_root
                / speaker
                / (path.stem + "_%04d%s" % (row, suffix))
            )
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(value["bytes"])
            written += 1
        print(path.name, written, flush=True)

    print(
        {
            "train_speakers": len(train_speakers),
            "dev_speakers": len(dev_speakers),
            "train_recordings": written,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
