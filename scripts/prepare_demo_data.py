#!/usr/bin/env python3
"""Pick demo cases from the event log and write a pseudonymised bundle."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DemoConfig, load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="Explorer configuration (default: EXPLORER_CONFIG or the CDSL file)")
    parser.add_argument("--log", type=Path, help="Event log; auto-detected from the configured candidates")
    parser.add_argument("--images", type=Path, help="Root directory of the image paths in the log")
    parser.add_argument("--without-images", action="store_true",
                        help="Build a pathway-only bundle; image stages and predictions stay disabled")
    parser.add_argument("--output", type=Path, help="Output directory (default: the configured bundle directory)")
    parser.add_argument("--cases", "--patients", dest="cases", type=int, default=20,
                        help="Number of cases to prepare (default: 20; --patients is kept as an alias)")
    return parser.parse_args(argv)



def first_existing(candidates) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def resolve_inputs(args: argparse.Namespace, config: DemoConfig) -> tuple[Path, Path | None]:
    settings = config.preparation
    log_path = args.log or first_existing(settings.log_candidates)
    if log_path is None or not log_path.is_file():
        searched = ", ".join(str(path) for path in settings.log_candidates)
        raise SystemExit(f"Event log not found. Pass --log or place it in one of: {searched}")
    if args.without_images:
        return log_path, None

    image_root = args.images
    if image_root is None and settings.image_env and os.getenv(settings.image_env):
        image_root = Path(os.environ[settings.image_env])
    image_root = image_root or first_existing(settings.image_candidates)
    if image_root is None or not image_root.is_dir():
        raise SystemExit(
            f"Event log found at {log_path}, but the image directory is missing. Pass --images"
            + (f", set {settings.image_env}" if settings.image_env else "")
            + ", or use one of the configured locations. The CSV contains image paths, not the image files themselves."
        )
    return log_path, image_root


def load_log(path: Path, config: DemoConfig) -> pd.DataFrame:
    columns = config.preparation.columns
    if path.suffix.lower() == ".xes":
        try:
            import pm4py
        except ImportError as exc:
            raise SystemExit("Reading XES requires pm4py; install requirements.txt") from exc
        log = pm4py.read_xes(str(path), return_legacy_log_object=False)
        if not isinstance(log, pd.DataFrame):
            log = pm4py.convert_to_dataframe(log)
    else:
        log = pd.read_csv(path)
    required = [columns[key] for key in ("case", "activity", "timestamp", "image")]
    if columns.get("outcome"):
        required.append(columns["outcome"])
    missing = set(required) - set(log.columns)
    if missing:
        raise SystemExit(f"Event log is missing columns: {', '.join(sorted(missing))}")
    log[columns["timestamp"]] = pd.to_datetime(log[columns["timestamp"]], errors="coerce")
    return log.dropna(subset=[columns["case"], columns["activity"], columns["timestamp"]])



def image_reference(value: object) -> str | None:
    """The last path of the log's image list (a Python-literal list or a string)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        paths = ast.literal_eval(str(value))
        relative = paths[-1] if isinstance(paths, list) else paths
    except (SyntaxError, ValueError):
        relative = str(value).strip("[]'\" ").split(",")[-1].strip("'\" ")
    return str(relative).strip() or None


def image_source(value: object, image_root: Path) -> Path | None:
    relative = image_reference(value)
    if relative is None:
        return None
    root = image_root.resolve()
    candidate = (root / relative).resolve()
    return candidate if root in candidate.parents and candidate.is_file() else None



def event_attributes(row: pd.Series, config: DemoConfig) -> tuple[str, ...]:
    """The attributes the templates can use: the configured features, or else every column except
    the mapped ones (case, activity, timestamp, image and, so the text never gives away the label, outcome)."""
    if config.preparation.features:
        return config.preparation.features
    mapped = {value for value in config.preparation.columns.values() if value}
    # @@index and @@case_index are added by pm4py when it reads the log
    return tuple(column for column in row.index
                 if column not in mapped and not column.startswith(("lifecycle:", "@@")))


def _missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _plain(value: object) -> object:
    # A numeric column with gaps is read as float: show 6461 rather than 6461.0
    return int(value) if isinstance(value, float) and value.is_integer() else value


def render_event(row: pd.Series, elapsed: float, config: DemoConfig) -> str:
    """Verbalise one event with its MEDUSA template, exactly as the authors do."""
    name = str(row[config.preparation.columns["activity"]])
    activity = config.activity(name)
    values = {feature: row.get(feature, float("nan")) for feature in event_attributes(row, config)}
    # For the generic template: the event's attributes that have a value, in log order
    values["attributes"] = [(key, _plain(value)) for key, value in values.items() if not _missing(value)]
    values["timesincecasestart"] = elapsed
    values["activity"] = name
    template = activity.template if activity and activity.template else config.preparation.default_template
    text = Template(template).render(values)
    for char in "{}'[]":
        text = text.replace(char, "")
    return (" " if activity and activity.leading_space else "") + text


def concise_summary(text: str, length: int) -> str:
    clean = " ".join(text.split()).strip()
    return clean if len(clean) <= length else clean[: length - 3].rstrip() + "..."



def choose_cases(log: pd.DataFrame, count: int, image_root: Path | None, config: DemoConfig) -> list[tuple[object, pd.DataFrame]]:
    """Balanced, information-rich cases from the chronological test partition."""
    columns = config.preparation.columns
    case, activity, timestamp, outcome, image = (columns[key] for key in ("case", "activity", "timestamp", "outcome", "image"))
    starts = log.groupby(case)[timestamp].min().sort_values(kind="mergesort")
    test_ids = set(starts.index[int(config.preparation.test_split * len(starts)):])
    stage_activities = {
        activity_.name for event_type in config.event_types for activity_ in event_type.activities if activity_.prediction_stage
    }

    # Risk class last; earlier classes get the extra case.
    classes = sorted(config.outcome_classes, key=lambda outcome_: outcome_.code == config.risk_class)
    candidates: dict[str, list[tuple[int, object, pd.DataFrame]]] = {outcome_.source_value: [] for outcome_ in classes}
    for case_id, group in log[log[case].isin(test_ids)].groupby(case, sort=False):
        group = group.sort_values(timestamp, kind="mergesort").copy()
        result = str(group[outcome].dropna().iloc[0]) if group[outcome].notna().any() else ""
        if result not in candidates:
            continue
        stages = group[group[activity].isin(stage_activities)][image]
        found = image_reference if image_root is None else (lambda value: image_source(value, image_root))
        valid_stages = sum(found(value) is not None for value in stages)
        if not valid_stages:
            continue
        score = valid_stages * 100 + group[activity].nunique() * 10 + min(len(group), 99)
        candidates[result].append((score, case_id, group))

    ranked = lambda items: sorted(items, key=lambda item: (-item[0], str(item[1])))  # noqa: E731
    per_class = math.ceil(count / len(candidates))
    selected = [item for bucket in candidates.values() for item in ranked(bucket)[:per_class]][:count]
    if len(selected) < count:
        used = {item[1] for item in selected}
        extras = ranked([item for bucket in candidates.values() for item in bucket if item[1] not in used])
        selected += extras[: count - len(selected)]
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} eligible test cases were found; requested {count}")
    return [(case_id, group) for _, case_id, group in selected]


def build_patient(alias: str, group: pd.DataFrame, image_root: Path | None, output: Path, config: DemoConfig) -> dict:
    columns = config.preparation.columns
    start = group[columns["timestamp"]].iloc[0]
    events = []
    for index, (_, row) in enumerate(group.iterrows(), start=1):
        name = str(row[columns["activity"]])
        offset = max(0.0, (row[columns["timestamp"]] - start).total_seconds())
        event_id = f"E{index:03d}"
        text = render_event(row, offset, config)
        event = {
            "id": event_id,
            "activity": name,
            "offset_seconds": int(offset),
            "summary": concise_summary(text, config.preparation.summary_length),
            "model_text": text,
            "prediction_stage": False,
        }
        if image_root is not None and config.is_prediction_stage(name):
            source = image_source(row.get(columns["image"]), image_root)
            if source is not None:
                destination = output / "images" / alias / f"{event_id}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                event["prediction_stage"] = True
                event["image"] = destination.relative_to(output).as_posix()
        events.append(event)

    patient = {
        "id": alias,
        "duration_seconds": events[-1]["offset_seconds"],
        "events": events,
    }
    outcome = columns.get("outcome")
    if outcome and group[outcome].notna().any():
        patient["true_outcome"] = str(group[outcome].dropna().iloc[0])
    return patient


def prepare_bundle(log: pd.DataFrame, image_root: Path | None, output: Path, config: DemoConfig,
                   case_ids: list[object] | None = None) -> Path:
    """Create the runtime bundle from an already parsed event table."""
    columns = config.preparation.columns
    case, timestamp = columns["case"], columns["timestamp"]
    log = log.copy()
    log[timestamp] = pd.to_datetime(log[timestamp], errors="coerce")
    log = log.dropna(subset=[case, columns["activity"], timestamp])
    groups = {case_id: group.sort_values(timestamp, kind="mergesort") for case_id, group in log.groupby(case, sort=False)}
    selected_ids = [case_id for case_id in (case_ids or list(groups)) if case_id in groups]
    if not selected_ids:
        raise ValueError("The event log contains no valid cases")
    output.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output / "images", ignore_errors=True)
    width = max(2, len(str(len(selected_ids))))  # Case 01 ... Case 20, or Case 001 ... Case 500
    patients = [build_patient(f"Case {index:0{width}d}", groups[case_id], image_root, output, config)
                for index, case_id in enumerate(selected_ids, start=1)]
    payload = {
        "metadata": {
            "source": config.preparation.source,
            "selection": "Cases loaded from the configured XES event log",
            "case_count": len(patients),
            "identifier_policy": "Source case identifiers and absolute timestamps omitted",
            "mode": "multimodal" if image_root is not None else "pathway_only",
            "labels_available": config.labels_available,
        },
        "patients": patients,
    }
    target = output / config.bundle_path.name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    count = args.cases
    if count < 2:
        raise SystemExit("--cases must be at least 2")
    output = args.output or config.bundle_path.parent
    log_path, image_root = resolve_inputs(args, config)
    log = load_log(log_path, config)

    output.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output / "images", ignore_errors=True)
    selected = choose_cases(log, count, image_root, config)
    patients = [
        build_patient(f"Case {index:02d}", group, image_root, output, config)
        for index, (_, group) in enumerate(selected, start=1)
    ]
    payload = {
        "metadata": {
            "source": config.preparation.source,
            "selection": "Balanced high-information cases from the authors' chronological test partition",
            "case_count": len(patients),
            "identifier_policy": "Source case identifiers and absolute timestamps omitted",
            "event_log": log_path.name,
            "mode": "multimodal" if image_root is not None else "pathway_only",
        },
        "patients": patients,
    }
    target = output / config.bundle_path.name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(patients)} cases at {target}")


if __name__ == "__main__":
    main()
