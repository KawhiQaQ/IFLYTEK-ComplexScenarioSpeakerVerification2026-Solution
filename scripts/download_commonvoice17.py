#!/usr/bin/env python3
"""Download the exact Common Voice 17.0 zh-CN snapshot used for training."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple


REPOSITORY = "masuidrive/cv-corpus-17.0-zh-CN-client_id-grouped"
REVISION = "61f9e8718b832db5fb1aecf80650f84fa90f8b2c"
FILES: Dict[str, Tuple[int, str]] = {
    "validation_0.parquet": (31753468, "a14d5044baff881e3c880792d617f1ed42cc2245f364912c38bf2d58a0c8eaf0"),
    "validation_1.parquet": (30796200, "9f26f798144011bf56922584d6484b0aca7ab785c39ff55ce2797167c852cdcc"),
    "validation_2.parquet": (31895374, "f80bf84816211d9b80cf52657b7134d398fef68d2252ae3f5c6d2fb7ab7edc93"),
    "validation_3.parquet": (31949971, "deb1b93edd743aebab89eb0038702bb5a882c7de63874bab9b70e621a4d69965"),
    "validation_4.parquet": (31347502, "6da729bb291ebc0091d93e2a939d77bb37e9d62b022e9d0f92ce1ce79dc7f72c"),
    "validation_5.parquet": (31933027, "20ce2b7c9cb9d48045702d8b713d049deaf0d72ef20908e0380ff29481aba331"),
    "validation_6.parquet": (30900270, "507f077d5d044848a9e0d0a16d7bfb3aa0ca9dc896949a36519fa1188548c6c8"),
    "validation_7.parquet": (31009135, "90ca74b01731a6a3686042ae620488ba7f5c496b5de5c7a2cea585ab5322b673"),
    "validation_8.parquet": (30714403, "6b65e262caf6423ebf37dcb4d5cb940d992ff1e36f012424522e02f9435cbbe8"),
    "validation_9.parquet": (31164483, "da433cf2b5e68c287e526f86a9d31a1b87425bba10c52dced9aeec2169302cf8"),
    "validation_10.parquet": (31164108, "29030b4566205af74f4fe84b8cc6075e83e77e22431fc86260439af4ddab68d6"),
    "validation_11.parquet": (30550954, "cbeaebe566c393bb1f85a7186b7b03b1d5a7e649942a094bd8cebdc1739aa25e"),
    "validation_12.parquet": (31458003, "a9caeb78c531e957e274a1217b2d8b6f1ea20f80e5288a9423224036f9136c3f"),
    "validation_13.parquet": (30566988, "ff0c8f0440c9ee8c1e6f02cc027f517412a1088a6acb206b3bdd7d9e6e5c9759"),
    "validation_14.parquet": (31614355, "4798349af0506032ae519e41203340b9546a40bb5c0f8da57ef6fd075fe54099"),
    "validation_15.parquet": (31998323, "1a6f1b5975ef11680dd7ca0014859c38567d40cbd8bd2754431d3484c18cbf47"),
    "validation_16.parquet": (31722250, "99ad16e081d7215b539aa82ce1c40b2cf66d2059feafb105b82994605b6c4625"),
    "validation_17.parquet": (30962893, "d7ae85642800300ff9eb6cb3b74c691d359c3cd4257e0c25bd485f3c02c11a56"),
    "validation_18.parquet": (32000934, "1857d3345da43aedd9da0780865d969f0cbe3a212a95ed213cbebdd4ba5b2459"),
    "validation_19.parquet": (31488248, "cbd93024f5e768991788a9fdcb057c7b9b5f0152c3464d40f2112b6e52e1bbec"),
    "validation_20.parquet": (31861350, "32fd64921ce66a6de6986527b930198258ece7ae5d0badafca92e327ee0eb563"),
    "validation_21.parquet": (31961201, "b40c3889e44c4f89a2e9c29c0d44d11121f4f400f98e470106601b5d5ab12137"),
    "validation_22.parquet": (31930429, "a560a7345cba6665871fc6db9238621e7cc0a05f4f7d45ca89e5433a43c0e95b"),
    "validation_23.parquet": (31901531, "fabe86969c1ab36e26c2666cd36f37bb201105ea952fde58e8199c892f09732c"),
    "validation_24.parquet": (31124522, "8f1e65231f5665c64720707a3c28c92950e2784c30821d643e6b8dddd362a363"),
    "validation_25.parquet": (8838535, "41e87a50a03605286b7cf511e00364b61e579f5c69f6e99637a67469348738e4"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def download_one(
    name: str,
    output_root: Path,
    endpoint: str,
    force: bool,
    retries: int,
) -> str:
    expected_size, expected_sha256 = FILES[name]
    destination = output_root / name
    if verify(destination, expected_size, expected_sha256):
        return f"verified  {name}"
    if destination.exists() and not force:
        raise RuntimeError(
            f"{destination} exists but does not match the released snapshot; "
            "delete it or rerun with --force"
        )

    url = (
        f"{endpoint.rstrip('/')}/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{name}"
    )
    temporary = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        written = 0
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "miwu-reproduction/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
            if written != expected_size or digest.hexdigest() != expected_sha256:
                raise RuntimeError(
                    f"checksum mismatch: got {written} bytes/{digest.hexdigest()}"
                )
            temporary.replace(destination)
            return f"downloaded {name}"
        except Exception as error:  # retry transient HTTP and connection failures
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to download {name}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/commonvoice17-zhcn-validation"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint (also read from HF_ENDPOINT)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check local files without downloading",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        invalid = [
            name
            for name, (size, checksum) in FILES.items()
            if not verify(args.output_root / name, size, checksum)
        ]
        if invalid:
            raise RuntimeError("missing or invalid files: " + ", ".join(invalid))
        print(f"verified {len(FILES)} files in {args.output_root}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                name,
                args.output_root,
                args.endpoint,
                args.force,
                args.retries,
            ): name
            for name in FILES
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print(f"ready: {len(FILES)} verified files in {args.output_root}")


if __name__ == "__main__":
    main()
