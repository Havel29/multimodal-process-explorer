"""Completes a configuration from the event log, so that a new dataset needs little or no TOML.

Only what the file leaves out is inferred; whatever it does say is kept as it is:
- event types: one timeline lane per activity, if the file defines none;
- image events: the activities whose events have a value in the image field, if no activity is
  marked with prediction_stage;
- outcome classes: the values of the label field, if the file defines none. Their order must be the
  order of the models' outputs, which the log cannot tell, so it is a guess and is reported. Classes
  without a colour get one.
"""
from __future__ import annotations

import colorsys
import dataclasses
import re

import pandas as pd

from .config import Activity, DemoConfig, EventType, OutcomeClass

# Distinct, readable colours for lanes and outcomes
def _palette(count: int) -> list[str]:
    colours = []
    for index in range(count):
        red, green, blue = colorsys.hls_to_rgb((index * 0.618034) % 1.0, 0.42, 0.55)
        colours.append(f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}")
    return colours


def _key(text: str, taken: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "event"
    candidate, number = key, 2
    while candidate in taken:
        candidate, number = f"{key}_{number}", number + 1
    taken.add(candidate)
    return candidate


def _has_image(value: object) -> bool:
    from scripts.prepare_demo_data import image_reference

    return image_reference(value) is not None


def complete_config(config: DemoConfig, log: pd.DataFrame) -> tuple[DemoConfig, list[str]]:
    """The config with the missing parts read from the log, and a note for every assumption made."""
    columns = config.preparation.columns
    activity_column, image_column, outcome_column = columns["activity"], columns.get("image", ""), columns.get("outcome", "")
    notes: list[str] = []

    activities = list(dict.fromkeys(str(value) for value in log[activity_column].dropna()))
    with_images: list[str] = []
    if image_column and image_column in log.columns:
        has_image = log[image_column].map(_has_image)
        with_images = list(dict.fromkeys(str(value) for value in log.loc[has_image, activity_column].dropna()))

    # Event types
    event_types = list(config.event_types)
    if not event_types:
        taken: set[str] = set()
        colours = _palette(len(activities))
        event_types = [
            EventType(key=_key(name, taken), label=name, color=colour, activities=(Activity(name=name, display=name),))
            for name, colour in zip(activities, colours)
        ]
        notes.append(f"{len(activities)} event types, one per activity")
    else:
        known = {activity.name for event_type in event_types for activity in event_type.activities}
        unknown = [name for name in activities if name not in known]
        if unknown:
            notes.append(f"{len(unknown)} activities are not in the configuration and are shown as Other: "
                         + ", ".join(unknown[:5]) + (", ..." if len(unknown) > 5 else ""))

    # Image events
    if not any(activity.prediction_stage for event_type in event_types for activity in event_type.activities):
        stages = set(with_images)
        event_types = [
            dataclasses.replace(event_type, activities=tuple(
                dataclasses.replace(activity, prediction_stage=activity.name in stages) for activity in event_type.activities
            ))
            for event_type in event_types
        ]
        if stages:
            notes.append("image events: " + ", ".join(sorted(stages)))
        else:
            notes.append("no event has an image in the image field, so the image models have nothing to read")

    # Outcome classes
    outcome_classes, risk_class = config.outcome_classes, config.risk_class
    values: list[str] = []
    if outcome_column and outcome_column in log.columns:
        values = sorted(str(value) for value in log[outcome_column].dropna().unique())
    if not outcome_classes:
        if not values:
            raise ValueError("The outcome classes are not configured and no label field was selected: "
                             "select a label field or list the classes under [outcome] in the configuration")
        taken = set()
        outcome_classes = tuple(
            OutcomeClass(code=_key(value, taken), label=value, source_value=value, color="") for value in values
        )
        codes = {outcome.code for outcome in outcome_classes}
        if risk_class not in codes:
            # The rarer outcome is usually the one to watch (e.g. non-survival, rejection)
            case_column = columns["case"]
            per_case = log.dropna(subset=[outcome_column]).groupby(case_column)[outcome_column].first().astype(str)
            rarest = per_case.value_counts().sort_values(kind="mergesort").index[0]
            risk_class = next(outcome.code for outcome in outcome_classes if outcome.source_value == rarest)
        notes.append("outcome classes in model output order (alphabetical, a guess: check it matches the models): "
                     + ", ".join(outcome.source_value for outcome in outcome_classes)
                     + f"; risk class: {next(o.label for o in outcome_classes if o.code == risk_class)}")
    elif values:
        configured = {outcome.source_value for outcome in outcome_classes}
        unknown = [value for value in values if value not in configured]
        if unknown:
            raise ValueError(
                f"The labels of this log ({', '.join(unknown)}) are not outcomes of the configuration in use, "
                f"which expects {', '.join(sorted(configured))}. Choose the configuration that belongs to this log "
                "with EXPLORER_CONFIG in .env (config/xray_pathways.toml for the open example, config/generic.toml "
                "for your own data) and restart with docker compose up -d."
            )

    # Classes without a colour: red for the risk class, the CDSL colours otherwise
    others = iter(["#2a78d6", "#1baf7a", "#eda100", "#6250d6", *_palette(len(outcome_classes))])
    outcome_classes = tuple(
        outcome if outcome.color else dataclasses.replace(outcome, color="#e34948" if outcome.code == risk_class else next(others))
        for outcome in outcome_classes
    )

    completed = dataclasses.replace(
        config, event_types=tuple(event_types), outcome_classes=outcome_classes, risk_class=risk_class, _activities={},
    )
    return completed, notes
