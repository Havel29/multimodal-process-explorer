#!/usr/bin/env python3
"""Download the three checkpoints of the open X-ray example from the GitHub release (1.8 GB).

The Docker image runs this at build time, so the example is preloaded on the first start
(config/xray_pathways.toml, [preload]). Files already present are kept.

    python scripts/download_example_checkpoints.py [--dest models/xray-pathways] [--release-url URL]
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_URL = "https://github.com/Havel29/multimodal-process-explorer/releases/download/v1.0"
FILES = ("text.pth", "image.pth", "multimodal.pth")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=ROOT / "models" / "xray-pathways")
    parser.add_argument("--release-url", default=RELEASE_URL)
    args = parser.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = args.dest / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"{target} already there", flush=True)
            continue
        partial = target.with_suffix(".part")
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(f"{args.release_url}/{name}") as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1 << 20)
        partial.replace(target)


if __name__ == "__main__":
    main()
