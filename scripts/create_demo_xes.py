#!/usr/bin/env python3
"""Export the chronological-test cases of a log (all of them, or a balanced sample of the longest) as XES."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from scripts.prepare_demo_data import choose_cases, load_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Source CSV or XES event log")
    parser.add_argument("--images", type=Path, help="Optional image root used to verify image files while selecting cases")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/cdsl_test.xes"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cases", "--patients", dest="cases", type=int, default=20,
                        help="Number of cases (default: 20; --patients is kept as an alias)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log = load_log(args.log, config)
    selected = choose_cases(log, args.cases, args.images, config)
    frame = pd.concat([group for _, group in selected], ignore_index=True)
    columns = config.preparation.columns
    try:
        import pm4py
    except ImportError as exc:
        raise SystemExit("Pm4py is required; run pip install -r requirements.txt") from exc
    frame = pm4py.format_dataframe(
        frame,
        case_id=columns["case"],
        activity_key=columns["activity"],
        timestamp_key=columns["timestamp"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(frame, str(args.output), case_id_key="case:concept:name")
    print(f"Wrote {len(selected)} cases and {len(frame)} events to {args.output}")


if __name__ == "__main__":
    main()
