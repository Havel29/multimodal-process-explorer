"""What the models know at a cutoff: the end of a day, an image event, or a time."""
from __future__ import annotations

import math
from dataclasses import dataclass

DAY_SECONDS = 86400


class CutoffError(ValueError):
    """The cutoff or the event selection is invalid for this pathway."""


@dataclass(frozen=True)
class Knowledge:
    available: list[dict]
    image_event: dict | None


def resolve_cutoff(
    events: list[dict],
    *,
    cutoff_day: int | None = None,
    stage_event_id: str | None = None,
    cutoff_seconds: int | None = None,
) -> Knowledge:
    if sum(value is not None for value in (cutoff_day, stage_event_id, cutoff_seconds)) != 1:
        if cutoff_day is None and stage_event_id is None and cutoff_seconds is None:
            raise CutoffError("Select a day cutoff or an image event")
        raise CutoffError("Choose either a day cutoff or an imaging stage")

    if stage_event_id is not None:
        index = next((i for i, event in enumerate(events) if event["id"] == stage_event_id), None)
        if index is None or not events[index].get("prediction_stage"):
            raise CutoffError("Select an image event as the prediction stage")
        return Knowledge(events[: index + 1], events[index])

    if cutoff_day is not None:
        last_event = max((event.get("offset_seconds", 0) for event in events), default=0)
        last_day = max(1, math.ceil(last_event / DAY_SECONDS))
        if not 1 <= cutoff_day <= last_day:
            raise CutoffError(f"Day cutoff must be between 1 and {last_day}")
        cutoff_seconds = cutoff_day * DAY_SECONDS
    if cutoff_seconds < 0:
        raise CutoffError("Cutoff time cannot be negative")
    available = [event for event in events if event.get("offset_seconds", 0) <= cutoff_seconds]
    image_event = next((event for event in reversed(available) if event.get("prediction_stage")), None)
    return Knowledge(available, image_event)


def compose_history(available: list[dict], enabled: set[str]) -> tuple[str, list[tuple[str, int, int]]]:
    """Concatenate the enabled event texts; also return each event's span."""
    parts, spans, cursor = [], [], 0
    for event in available:
        if event["id"] not in enabled:
            continue
        text = event.get("model_text", "")
        parts.append(text)
        spans.append((event["id"], cursor, cursor + len(text)))
        cursor += len(text)
    history = "".join(parts)
    if not history.strip():
        return "Event history unavailable.", []
    return history, spans


def trajectory_points(patient: dict, ignore=None) -> list[dict]:
    """The end of each day of the stay, plus each image event; the days end with the last event that
    `ignore` does not exclude (e.g. a discharge that states the outcome)."""
    events = [event for event in patient["events"] if not (ignore and ignore(event))]
    end = events[-1]["offset_seconds"] if events else 0
    days = max(1, math.ceil(end / DAY_SECONDS))
    points = [{"kind": "day", "day": day, "offset_seconds": day * DAY_SECONDS} for day in range(1, days + 1)]
    points += [
        {"kind": "stage", "stage_event_id": event["id"], "offset_seconds": event["offset_seconds"]}
        for event in events
        if event.get("prediction_stage")
    ]
    return sorted(points, key=lambda point: point["offset_seconds"])
