"""Loads the TOML file that describes the dataset and its models."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .texts import resolve_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "xray_pathways.toml"
MODEL_KINDS = {"text": (True, False), "image": (False, True), "flava": (True, True)}


@dataclass(frozen=True)
class Activity:
    name: str
    display: str = ""
    template: str = ""  # empty: the preparation's default template
    short: str = ""
    prediction_stage: bool = False
    leading_space: bool = False
    # The event states the outcome (e.g. the hospital discharge with its diagnoses): it stays on the
    # timeline but is never given to the models, so that no prediction is made once the outcome is known
    reveals_outcome: bool = False


@dataclass(frozen=True)
class EventType:
    key: str
    label: str
    color: str
    activities: tuple[Activity, ...] = ()


OTHER_EVENT_TYPE = EventType(key="other", label="Other", color="#8a96a1")


@dataclass(frozen=True)
class OutcomeClass:
    code: str
    label: str
    source_value: str
    color: str = ""  # picked at setup when empty
    short: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    kind: str  # the role: text, image or flava (multimodal)
    base: str
    checkpoint: Path
    color: str
    adapter: str = ""  # a Python file with the model class (see app/models/__init__.py); empty: the built-in one

    @property
    def uses_text(self) -> bool:
        return MODEL_KINDS[self.kind][0]

    @property
    def uses_image(self) -> bool:
        return MODEL_KINDS[self.kind][1]


@dataclass(frozen=True)
class ExplanationSettings:
    model: str
    # A built-in method (app/explainers.py) or the path of a Python file with an explain() function
    method: str = "integrated_gradients"
    label: str = ""  # how the interface names the method; empty: from the method
    steps: int = 20
    internal_batch_size: int = 4
    reference: str = ""
    options: dict = field(default_factory=dict)  # passed to the method as keyword arguments

    @property
    def display(self) -> str:
        if self.label:
            return self.label
        return "Layer Integrated Gradients" if self.method == "integrated_gradients" else Path(self.method).stem.replace("_", " ")

    def arguments(self) -> dict:
        return {"steps": self.steps, "internal_batch_size": self.internal_batch_size, **self.options}


@dataclass(frozen=True)
class PreparationSettings:
    columns: dict[str, str]
    features: tuple[str, ...]
    default_template: str
    test_split: float
    summary_length: int
    log_candidates: tuple[Path, ...]
    image_candidates: tuple[Path, ...]
    image_env: str | None
    source: str


@dataclass(frozen=True)
class DemoConfig:
    title: str
    subtitle: str
    institution: str
    logo: str
    disclaimer: str
    image_noun: str
    stage_noun: str
    case_noun: str
    pathway_noun: str
    day_noun: str
    bundle_path: Path
    analytics_dir: Path
    model_cache: Path
    grayscale_images: bool
    risk_class: str
    outcome_classes: tuple[OutcomeClass, ...]
    models: tuple[ModelSpec, ...]
    explanations: tuple[ExplanationSettings, ...]  # at most one per model; the page shows a button each
    event_types: tuple[EventType, ...]
    preparation: PreparationSettings
    labels_available: bool = True
    texts: dict = field(default_factory=dict)  # the [texts] overrides; resolved_texts() adds the defaults
    _activities: dict[str, tuple[EventType, Activity]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for event_type in self.event_types:
            for activity in event_type.activities:
                self._activities[activity.name] = (event_type, activity)
        # Outcome classes may be left out of the file and read from the event log at setup
        codes = [outcome.code for outcome in self.outcome_classes]
        if codes and self.risk_class not in codes:
            raise ValueError(f"risk_class {self.risk_class!r} is not one of the outcome classes {codes}")
        if not self.models:
            raise ValueError("configure at least one model")
        kinds = [model.kind for model in self.models]
        if len(kinds) != len(set(kinds)):
            raise ValueError(f"one model per role at most (text, image, flava); got {kinds}")
        explained = [explanation.model for explanation in self.explanations]
        for model_id in explained:
            if model_id not in {model.id for model in self.models}:
                raise ValueError(f"explanation model {model_id!r} is not configured")
        if len(explained) != len(set(explained)):
            raise ValueError(f"one explanation per model at most; got {explained}")
        self.resolved_texts()  # rejects unknown [texts] keys

    @property
    def explanation(self) -> ExplanationSettings | None:
        """The main explanation: the multimodal model's if it has one, else the first."""
        kinds = {model.id: model.kind for model in self.models}
        return next((item for item in self.explanations if kinds[item.model] == "flava"),
                    self.explanations[0] if self.explanations else None)

    def explanation_of(self, model_id: str) -> ExplanationSettings | None:
        return next((item for item in self.explanations if item.model == model_id), None)

    def with_models(self, models) -> "DemoConfig":
        """The same configuration with only these models (one to three) and their explanations."""
        models = tuple(models)
        ids = {model.id for model in models}
        return replace(self, models=models, explanations=tuple(item for item in self.explanations if item.model in ids))

    def resolved_texts(self) -> dict[str, str]:
        return resolve_texts(self.texts, {"case": self.case_noun, "pathway": self.pathway_noun, "image": self.image_noun,
                                          "stage": self.stage_noun, "day": self.day_noun})

    def event_type_of(self, activity: str) -> EventType:
        return self._activities.get(activity, (OTHER_EVENT_TYPE, None))[0]

    def activity(self, name: str) -> Activity | None:
        entry = self._activities.get(name)
        return entry[1] if entry else None

    def is_prediction_stage(self, activity: str) -> bool:
        entry = self.activity(activity)
        return bool(entry and entry.prediction_stage)

    def reveals_outcome(self, activity: str) -> bool:
        entry = self.activity(activity)
        return bool(entry and entry.reveals_outcome)

    def model(self, model_id: str) -> ModelSpec | None:
        return next((model for model in self.models if model.id == model_id), None)

    def outcome_for_source(self, value: str | None) -> OutcomeClass:
        """Map an outcome value of the event log to its class."""
        match = next((outcome for outcome in self.outcome_classes if outcome.source_value == value), None)
        if match is None:
            raise ValueError(f"Unknown outcome value {value!r}")
        return match

    def class_index(self, code: str) -> int:
        return [outcome.code for outcome in self.outcome_classes].index(code)

    def public(self) -> dict:
        """The part of the configuration the browser needs."""
        return {
            "title": self.title,
            "imageNoun": self.image_noun,
            "stageNoun": self.stage_noun,
            "nouns": {"case": self.case_noun, "pathway": self.pathway_noun, "day": self.day_noun},
            "texts": self.resolved_texts(),
            "riskClass": self.risk_class,
            "hasLabels": self.labels_available,
            "outcomes": [
                {"code": outcome.code, "label": outcome.label, "short": outcome.short or outcome.label, "color": outcome.color}
                for outcome in self.outcome_classes
            ],
            "models": [
                {
                    "id": model.id, "name": model.name, "description": model.description,
                    "color": model.color, "usesText": model.uses_text, "usesImage": model.uses_image,
                }
                for model in self.models
            ],
            "explanations": [
                {"model": item.model, "steps": item.steps, "method": item.display, "reference": item.reference}
                for item in self.explanations
            ],
            "eventTypes": [
                {
                    "key": event_type.key, "label": event_type.label, "color": event_type.color,
                    "activities": [
                        {"name": activity.name, "display": activity.display, "short": activity.short,
                         "predictionStage": activity.prediction_stage, "revealsOutcome": activity.reveals_outcome}
                        for activity in event_type.activities
                    ],
                }
                for event_type in (*self.event_types, OTHER_EVENT_TYPE)
            ],
        }


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# Used for whatever the TOML file leaves out, so that a new dataset needs no file at all:
# the setup reads event types, image events and outcome classes from the event log.
DEFAULT_MODELS = {
    "text": {"id": "text", "name": "Text model", "description": "Event history",
             "base": "FacebookAI/roberta-base", "color": "#eda100"},
    "image": {"id": "image", "name": "Image model", "description": "Image only",
              "base": "google/vit-base-patch16-224", "color": "#1baf7a"},
    "flava": {"id": "multimodal", "name": "Multimodal model", "description": "History + image",
              "base": "facebook/flava-full", "color": "#6250d6"},
}
DEFAULT_COLUMNS = {"case": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp",
                   "outcome": "", "image": ""}
# Verbalises an event with all its attributes (see scripts/prepare_demo_data.render_event)
GENERIC_TEMPLATE = "{{activity}}{% for name, value in attributes %}; {{name}}: {{value}}{% endfor %}."


def load_config(path: str | Path | None = None) -> DemoConfig:
    """Load the explorer config, accepting the legacy MEDUSA_CONFIG name."""
    path = _path(path or os.getenv("EXPLORER_CONFIG") or os.getenv("MEDUSA_CONFIG") or DEFAULT_CONFIG)
    return config_from_mapping(tomllib.loads(path.read_text(encoding="utf-8")))


def config_from_mapping(raw: dict) -> DemoConfig:
    """Build the config from parsed TOML; every section is optional."""
    demo, data, preparation = raw.get("demo", {}), raw.get("data", {}), raw.get("preparation", {})
    outcome = raw.get("outcome", {})

    models = []
    for entry in raw.get("models") or [{"kind": kind} for kind in DEFAULT_MODELS]:
        if entry.get("kind") not in MODEL_KINDS:
            raise ValueError(f"Model {entry.get('id')}: unknown kind {entry.get('kind')!r}; use one of {sorted(MODEL_KINDS)}")
        entry = {**DEFAULT_MODELS[entry["kind"]], **entry}
        if entry.get("adapter") and not _path(entry["adapter"]).is_file():
            raise ValueError(f"Model {entry['id']}: adapter file {entry['adapter']!r} not found")
        models.append(ModelSpec(**{**entry, "checkpoint": _path(entry.get("checkpoint", ""))}))
    # [[explanation]] per explained model ([explanation] for one). By default, the built-in multimodal model
    # gets Integrated Gradients, as in MEDUSA; any other model needs an entry with a method file
    explanations = raw.get("explanation")
    if isinstance(explanations, dict):
        explanations = [explanations]
    if explanations is None:
        explanations = [{"model": model.id} for model in models if model.kind == "flava" and not model.adapter]
    bundle = data.get("bundle", "data/runtime/demo.json")

    return DemoConfig(
        title=demo.get("title", "Multimodal"),
        subtitle=demo.get("subtitle", "Process Explorer"),
        institution=demo.get("institution", ""),
        logo=demo.get("logo", ""),
        disclaimer=demo.get("disclaimer", ""),
        image_noun=demo.get("image_noun", "image"),
        stage_noun=demo.get("stage_noun", "image"),
        case_noun=demo.get("case_noun", "case"),
        pathway_noun=demo.get("pathway_noun", "trace"),
        day_noun=demo.get("day_noun", "day"),
        bundle_path=_path(os.getenv("DEMO_DATA_PATH") or bundle),
        analytics_dir=_path(data.get("analytics_dir") or Path(bundle).parent),
        model_cache=_path(os.getenv("HF_HOME") or data.get("model_cache", "hf-cache")),
        grayscale_images=data.get("grayscale_images", False),
        risk_class=outcome.get("risk_class", ""),
        outcome_classes=tuple(OutcomeClass(**entry) for entry in outcome.get("classes", [])),
        models=tuple(models),
        explanations=tuple(ExplanationSettings(**entry) for entry in explanations),
        event_types=tuple(
            EventType(
                key=entry["key"], label=entry.get("label", entry["key"]), color=entry.get("color", OTHER_EVENT_TYPE.color),
                activities=tuple(
                    Activity(**{**activity, "display": activity.get("display") or activity["name"]})
                    for activity in entry.get("activities", [])
                ),
            )
            for entry in raw.get("event_types", [])
        ),
        preparation=PreparationSettings(
            columns={**DEFAULT_COLUMNS, **preparation.get("columns", {})},
            features=tuple(preparation.get("features", [])),
            default_template=preparation.get("default_template", GENERIC_TEMPLATE),
            test_split=preparation.get("test_split", 0.66),
            summary_length=preparation.get("summary_length", 230),
            log_candidates=tuple(_path(item) for item in preparation.get("log_candidates", [])),
            image_candidates=tuple(_path(item) for item in preparation.get("image_candidates", [])),
            image_env=preparation.get("image_env"),
            source=preparation.get("source", ""),
        ),
        labels_available=True,
        texts=dict(raw.get("texts", {})),
    )
