"""The texts the interface prints, overridable one by one in the [texts] section of the configuration.

In any text, {case}, {cases}, {pathway}, {image}, {images}, {stage} and {day} become the nouns of
[demo] (plurals add an "s"); the other {fields} are filled in by the page (counts, names, days).
"""
from __future__ import annotations

import string

DEFAULT_TEXTS: dict[str, str] = {
    # Inputs page
    "setup_title": "Configure the multimodal explorer",
    "setup_intro": "Upload an XES event log and select the image and model folders.",
    # Case view
    "case_meta": "{event_count} over {day_count}, {image_count}. Times are relative to the first event.",
    "observed_outcome": "Observed outcome",
    "timeline_help": "Set what the models know: drag the black line, click the timeline, pick a {day}, or click any "
                     "{stage} marker. Drag a box over events to use only them, exclude them, or add them back. "
                     "Click a mark to read the event.",
    "start_hint": "Set a cutoff on the timeline to begin.",
    "events_title": "Events given to the models",
    "events_empty": "The events known at the cutoff appear here. Untick an event to hide it from the models.",
    "predictions_title": "Predictions",
    # Explanations
    "why_title": "Why?",
    "scope_context": "Model context (what the model reads)",
    "scope_history": "Whole history (all events)",
    "explain_intro": "Each Explain button runs {method} on its model, over what that model reads: the event texts, the "
                     "{image} or both. To see what the prediction depends on, you can also switch event types off by "
                     "clicking their names in the timeline.",
    "context_title": "What the model read",
    "context_legend": "Shaded words push the prediction toward {risk} or away from it; the darker, the stronger.",
    "history_title": "Most influential events",
    # Statistics
    "statistics_title": "Statistics",
    "statistics_intro": "How the models behave as knowledge accumulates along each {pathway}: discrimination and "
                        "calibration by {day}, earliness, stability, and which kinds of events move the predictions.",
    "statistics_caveat": "Curves stop at {last_day}, the last sampled horizon with at least {minimum} continuing "
                         "{pathway}s; each point reports its own n on hover. The table still extends through {table_day}.",
    "summary_note": "At each {image}: one prediction per {image} cutoff, on the cutoffs where every model runs, as in "
                    "the usual evaluation of multimodal models. All cutoffs: pooled over every {day} and {image} cutoff "
                    "of each model, including the {days} before the first {image} for the text models; AUC and Brier "
                    "score compare the risk outcome against the others. A model settles once every later prediction "
                    "matches the observed outcome (earliness, in predictive process monitoring terms); a flip is a "
                    "change of predicted class between consecutive cutoffs.",
    "importance_note": "Mean size of the change in {risk}, in percentage points, when an event type is removed (or the "
                       "{image} blanked) at each of the {count} {image} cutoffs. Hover a bar for the average direction. "
                       "A bar near zero usually means those events are out of context: the text models only read the "
                       "most recent 512 tokens, so the earliest events are rarely read.",
    # The unit of the timeline
    "day": "Day",
}

NOUNS = ("case", "pathway", "image", "stage", "day")


class _Keep(dict):
    """Leaves the fields the page fills in as they are."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_texts(overrides: dict[str, str], nouns: dict[str, str]) -> dict[str, str]:
    """Defaults with the configured overrides, nouns filled in."""
    unknown = set(overrides) - set(DEFAULT_TEXTS)
    if unknown:
        raise ValueError(f"Unknown keys in [texts]: {', '.join(sorted(unknown))}; see app/texts.py")
    fields = _Keep({**nouns, **{f"{noun}s": f"{word}s" for noun, word in nouns.items()}})
    texts = {**DEFAULT_TEXTS, **overrides}
    return {key: string.Formatter().vformat(value, (), fields) for key, value in texts.items()}
