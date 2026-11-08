"""Writes a configuration back to TOML: the setup saves the completed config of a session, and
scripts/draft_config.py drafts one for a new dataset that can then be edited by hand."""
from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, DemoConfig, GENERIC_TEMPLATE
from .texts import DEFAULT_TEXTS


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)  # a JSON string is a valid TOML basic string


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.as_posix()


def _table(name: str, values: dict, indent: str = "") -> list[str]:
    lines = [f"{indent}[{name}]"]
    lines += [f"{indent}{key} = {_value(value)}" for key, value in values.items() if value not in ("", None)]
    return lines


def dump_config(config: DemoConfig, header: str = "") -> str:
    lines = [f"# {line}" for line in header.splitlines()] + ([""] if header else [])
    lines += _table("demo", {
        "title": config.title, "subtitle": config.subtitle, "institution": config.institution, "logo": config.logo,
        "case_noun": config.case_noun, "pathway_noun": config.pathway_noun, "image_noun": config.image_noun,
        "stage_noun": config.stage_noun, "day_noun": config.day_noun, "disclaimer": config.disclaimer,
    })
    # model_cache is left out: it often comes from HF_HOME, which belongs to the machine, not the dataset
    lines += [""] + _table("data", {"grayscale_images": config.grayscale_images})

    lines += ["", "# Outcome classes in the order of the models' outputs"]
    lines += _table("outcome", {"risk_class": config.risk_class})
    for outcome in config.outcome_classes:
        lines += ["", "[[outcome.classes]]"] + [line for line in _table("x", {
            "code": outcome.code, "source_value": outcome.source_value, "label": outcome.label,
            "short": outcome.short, "color": outcome.color,
        })[1:]]

    for model in config.models:
        lines += ["", "[[models]]"] + _table("x", {
            "id": model.id, "name": model.name, "description": model.description, "kind": model.kind,
            "base": model.base, "checkpoint": model.checkpoint.name if model.checkpoint.is_file() else "", "color": model.color,
            "adapter": model.adapter,
        })[1:]
    lines += ["", "# One explanation per explained model; method: a built-in one (integrated_gradients) or a",
              "# Python file with explain(), see app/explainers.py"]
    for explanation in config.explanations:
        lines += ["", "[[explanation]]"] + _table("x", {
            "model": explanation.model, "method": explanation.method, "label": explanation.label,
            "steps": explanation.steps, "internal_batch_size": explanation.internal_batch_size,
            "reference": explanation.reference,
        })[1:]
        if explanation.options:
            lines += [""] + _table("explanation.options", explanation.options)

    lines += ["", "# One timeline lane per event type; prediction_stage marks the activities with an image"]
    for event_type in config.event_types:
        lines += ["", "[[event_types]]"] + _table("x", {"key": event_type.key, "label": event_type.label, "color": event_type.color})[1:]
        for activity in event_type.activities:
            lines += ["", "  [[event_types.activities]]"] + _table("x", {
                "name": activity.name, "display": activity.display, "short": activity.short,
                "prediction_stage": activity.prediction_stage or "", "leading_space": activity.leading_space or "",
                "reveals_outcome": activity.reveals_outcome or "",
                "template": activity.template,
            }, indent="  ")[1:]

    preparation = config.preparation
    lines += [""] + _table("preparation", {
        "source": preparation.source, "test_split": preparation.test_split, "summary_length": preparation.summary_length,
        "default_template": preparation.default_template if preparation.default_template != GENERIC_TEMPLATE else "",
        "features": list(preparation.features) or "",
        "log_candidates": [_relative(path) for path in preparation.log_candidates] or "",
        "image_candidates": [_relative(path) for path in preparation.image_candidates] or "",
        "image_env": preparation.image_env or "",
    })
    lines += [""] + _table("preparation.columns", dict(preparation.columns))

    # Every text, so that the wording can be adapted to the scenario without looking for it in the code
    lines += ["", "# The texts the interface prints. {case}, {pathway}, {image}, {stage} and {day} are the nouns of [demo]",
              "# (add an s for the plural); the other {fields} are filled in by the page."]
    lines += _table("texts", {**DEFAULT_TEXTS, **config.texts})
    return "\n".join(lines) + "\n"
