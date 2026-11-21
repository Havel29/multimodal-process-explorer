#!/usr/bin/env python3
"""Download the base models of the configurations, so the Docker image works offline.

Without --config, every configuration in config/ is read, so that switching EXPLORER_CONFIG needs no
rebuild. Models with an adapter file are skipped unless their class defines download(spec, num_classes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.models import adapter_class  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, nargs="*", help="Configurations (default: every file in config/)")
    args = parser.parse_args(argv)

    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        FlavaProcessor,
    )

    done = set()
    for path in args.config or sorted((ROOT / "config").glob("*.toml")):
        config = load_config(path)
        num_labels = max(2, len(config.outcome_classes))  # classes may be read from the log at setup
        for spec in config.models:
            if (spec.kind, spec.base, spec.adapter) in done:
                continue
            done.add((spec.kind, spec.base, spec.adapter))
            if spec.adapter:
                download = getattr(adapter_class(spec), "download", None)
                if callable(download):
                    print(f"Preparing {spec.adapter} for {spec.name}", flush=True)
                    download(spec, num_labels)
                continue
            print(f"Downloading {spec.base} for {spec.name}", flush=True)
            if spec.kind == "text":
                AutoTokenizer.from_pretrained(spec.base)
                AutoModelForSequenceClassification.from_pretrained(spec.base, num_labels=num_labels)
            elif spec.kind == "image":
                AutoImageProcessor.from_pretrained(spec.base)
                AutoModelForImageClassification.from_pretrained(spec.base, num_labels=num_labels, ignore_mismatched_sizes=True)
            elif spec.kind == "flava":
                FlavaProcessor.from_pretrained(spec.base)


if __name__ == "__main__":
    main()
