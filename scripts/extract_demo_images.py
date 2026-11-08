#!/usr/bin/env python3
"""Extract only the images referenced by a demo XES (or a CSV log) from the full CDSL ZIP."""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup import read_xes  # noqa: E402
from scripts.prepare_demo_data import image_reference  # noqa: E402

DEFAULT_PREFIX = "physionet.org/files/covid-data-shared-learning/1.0.0/IMAGES/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xes", "--log", dest="xes", required=True, type=Path, help="XES or CSV event log")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-field", default="path")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--extract", action="store_true", help="Extract after reporting the required size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = pd.read_csv(args.xes, usecols=[args.image_field]) if args.xes.suffix == ".csv" else read_xes(args.xes)
    if args.image_field not in log:
        raise SystemExit(f"Image field not found: {args.image_field}")
    references = sorted({reference for value in log[args.image_field].dropna()
                         if (reference := image_reference(value))})
    prefix = args.prefix.strip("/") + "/"
    with zipfile.ZipFile(args.archive) as archive:
        entries, missing = [], []
        for reference in references:
            member = prefix + PurePosixPath(reference).as_posix().lstrip("/")
            try:
                entries.append((reference, archive.getinfo(member)))
            except KeyError:
                missing.append(reference)
        size = sum(info.file_size for _, info in entries)
        print(f"Referenced images: {len(references)}")
        print(f"Found in archive: {len(entries)}")
        print(f"Missing from archive: {len(missing)}")
        print(f"Uncompressed size: {size / 1024 ** 3:.2f} GiB")
        if missing:
            print("First missing paths:")
            for reference in missing[:10]:
                print(f"  {reference}")
        if not args.extract:
            print("Dry run only; pass --extract to write the files")
            return

        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for index, (reference, info) in enumerate(entries, start=1):
            destination = (root / Path(*PurePosixPath(reference).parts)).resolve()
            if root not in destination.parents:
                raise SystemExit(f"Unsafe image path: {reference}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            if index % 100 == 0 or index == len(entries):
                print(f"Extracted {index}/{len(entries)}")


if __name__ == "__main__":
    main()
