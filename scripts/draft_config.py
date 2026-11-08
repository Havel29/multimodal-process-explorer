#!/usr/bin/env python3
"""Draft a configuration for a new dataset from its XES log, to edit by hand afterwards.

    python scripts/draft_config.py log.xes --image path --label outcome -o config/my_dataset.toml

Field names default to the XES standard (case:concept:name, concept:name, time:timestamp).
The draft has one event type per activity, marks the activities with an image as prediction
stages, lists the label values as outcome classes and verbalises events with all their
attributes. Things to check are printed at the end, above all the order of the outcome classes,
which must be the order of the models' outputs.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import config_from_mapping  # noqa: E402
from app.config_writer import dump_config  # noqa: E402
from app.inference import complete_config  # noqa: E402
from app.setup import read_xes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xes", type=Path)
    parser.add_argument("--case", default="case:concept:name")
    parser.add_argument("--activity", default="concept:name")
    parser.add_argument("--timestamp", default="time:timestamp")
    parser.add_argument("--image", default="", help="event attribute with the image path")
    parser.add_argument("--label", default="", help="attribute with the outcome (optional)")
    parser.add_argument("--risk", default="", help="outcome value to treat as the risk class (default: the rarest)")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of printing")
    args = parser.parse_args(argv)

    log = read_xes(args.xes)
    columns = {"case": args.case, "activity": args.activity, "timestamp": args.timestamp,
               "image": args.image, "outcome": args.label}
    missing = [value for value in columns.values() if value and value not in log.columns]
    if missing:
        parser.error(f"not in the log: {', '.join(missing)}. Attributes: {', '.join(log.columns)}")

    config = config_from_mapping({"preparation": {"columns": columns}})
    if args.risk:
        config = dataclasses.replace(config, risk_class=args.risk)
    config, notes = complete_config(config, log)
    if args.risk and args.risk not in {outcome.code for outcome in config.outcome_classes}:
        match = next((outcome.code for outcome in config.outcome_classes if outcome.source_value == args.risk), None)
        if match is None:
            parser.error(f"--risk {args.risk!r} is not one of the label values")
        config = dataclasses.replace(config, risk_class=match, _activities={})

    text = dump_config(config, header=f"Drafted from {args.xes.name} by scripts/draft_config.py. Check the notes it printed.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"written {args.output}")
    else:
        print(text)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
