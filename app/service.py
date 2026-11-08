"""The demo's operations, independent of the web framework."""
from __future__ import annotations

import gc
import platform
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from threading import Lock
from time import perf_counter

from .attribution import event_attributions, history_windows
from .cache import AnalyticsCache
from .config import DemoConfig
from .explainers import load_explainer
from .cutoffs import CutoffError, Knowledge, compose_history, resolve_cutoff, trajectory_points
from .models import Classifier, ModelUnavailable, build_models
from .models.base import cuda_available
from .store import DemoDataError, PatientStore

EXPLANATION_CACHE_SIZE = 32
EXPLANATION_SCOPES = ("context", "history")


class ServiceError(Exception):
    status = 400


class BadRequest(ServiceError):
    status = 400


class NotFound(ServiceError):
    status = 404


class Unavailable(ServiceError):
    status = 503


@dataclass
class Selection:
    """A cutoff (exactly one of the three) plus the events given to the models."""

    patient_id: str
    enabled_event_ids: list[str] = field(default_factory=list)
    cutoff_day: int | None = None
    stage_event_id: str | None = None
    cutoff_seconds: int | None = None

    def key(self) -> tuple:
        return (self.patient_id, self.cutoff_day, self.stage_event_id, self.cutoff_seconds,
                frozenset(self.enabled_event_ids))


class DemoService:
    def __init__(
        self,
        config: DemoConfig,
        store: PatientStore,
        models: dict[str, Classifier],
        trajectories: AnalyticsCache,
        importance: AnalyticsCache,
        device_preference: str = "auto",
    ):
        self.config = config
        self.store = store
        self.models = models
        self.trajectories = trajectories
        self.importance_cache = importance
        self.device_preference = device_preference
        self._performance: list[dict] = []
        self._last_timing: dict[str, dict] = {}
        self._performance_lock = Lock()
        # PyTorch modules and CUDA execution are shared by every request. Keep
        # model loading, device changes, and forward passes single-filed.
        self._inference_lock = Lock()
        self._explanations: OrderedDict[tuple, dict] = OrderedDict()
        self._explanations_lock = Lock()

    @classmethod
    def from_config(cls, config: DemoConfig, device_preference: str = "auto") -> "DemoService":
        bundle = config.bundle_path
        return cls(
            config=config,
            store=PatientStore(bundle),
            models=build_models(config, device_preference),
            trajectories=AnalyticsCache(config.analytics_dir / "trajectories.json", bundle),
            importance=AnalyticsCache(config.analytics_dir / "importance.json", bundle),
            device_preference=device_preference,
        )

    @staticmethod
    def _gpu_info() -> tuple[bool, str | None]:
        try:
            import torch
            available = cuda_available(torch)
            return available, torch.cuda.get_device_name(0) if available else None
        except (ImportError, RuntimeError):
            return False, None

    def runtime(self) -> dict:
        available, name = self._gpu_info()
        try:
            import torch
            torch_version, cuda_version = torch.__version__, torch.version.cuda
        except ImportError:
            torch_version, cuda_version = None, None
        return {
            "preference": self.device_preference,
            "gpu_available": available,
            "gpu_name": name,
            "cpu": platform.processor() or platform.machine(),
            "python_version": platform.python_version(),
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "models_loaded": any(model.loaded for model in self.models.values()),
        }

    def set_device(self, preference: str) -> dict:
        if preference not in {"auto", "cpu", "gpu"}:
            raise BadRequest("Inference device must be auto, cpu, or gpu")
        available, _ = self._gpu_info()
        if preference == "gpu" and not available:
            raise BadRequest("Compatible CUDA GPU is unavailable")
        if preference == self.device_preference:
            return self.runtime()
        with self._inference_lock:
            old_models = self.models
            self.models = build_models(self.config, preference)
            self.device_preference = preference
            del old_models
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.runtime()

    def _data(self, function, *args):
        try:
            return function(*args)
        except DemoDataError as exc:
            raise Unavailable(str(exc)) from exc
        except KeyError as exc:
            raise NotFound(exc.args[0] if exc.args else "Not found") from exc

    def list_patients(self) -> list[dict]:
        return self._data(self.store.list_patients)

    def metadata(self) -> dict:
        return self._data(lambda: self.store.metadata)

    def patient(self, patient_id: str) -> dict:
        return self._data(self.store.get_patient, patient_id)

    def outcome(self, patient: dict) -> dict:
        if not self.config.labels_available or "true_outcome" not in patient:
            raise BadRequest("This event log has no ground-truth label field")
        outcome = self.config.outcome_for_source(patient.get("true_outcome"))
        return {"code": outcome.code, "label": outcome.label}

    def public_patient(self, patient_id: str) -> dict:
        patient = self.patient(patient_id)
        public = {key: value for key, value in patient.items() if key != "true_outcome"}
        if self.config.labels_available:
            public["ground_truth"] = self.outcome(patient)
        return public

    def image_path(self, patient_id: str, event_id: str) -> Path:
        return self._data(self.store.image_path, patient_id, event_id)

    def knowledge(self, selection: Selection) -> tuple[dict, Knowledge, set[str]]:
        patient = self.patient(selection.patient_id)
        try:
            known = resolve_cutoff(
                patient["events"],
                cutoff_day=selection.cutoff_day,
                stage_event_id=selection.stage_event_id,
                cutoff_seconds=selection.cutoff_seconds,
            )
        except CutoffError as exc:
            raise BadRequest(str(exc)) from exc
        enabled = set(selection.enabled_event_ids)
        if not enabled <= {event["id"] for event in known.available}:
            boundary = "chosen stage" if selection.stage_event_id is not None else "cutoff"
            raise BadRequest(f"Knowledge selection contains events after the {boundary}")
        # Events that state the outcome stay on the timeline, but the models never get them
        hidden = {event["id"] for event in known.available if self.config.reveals_outcome(event["activity"])}
        if hidden:
            known = Knowledge([event for event in known.available if event["id"] not in hidden], known.image_event)
            enabled -= hidden
        return patient, known, enabled

    def _model(self, model_id: str) -> Classifier:
        if model_id not in self.models:
            raise BadRequest(f"Unknown model {model_id!r}")
        return self.models[model_id]

    @staticmethod
    def _enabled_image(known: Knowledge, enabled: set[str]) -> dict | None:
        """Latest enabled image event at the cutoff, mirroring the browser."""
        return next(
            (event for event in reversed(known.available)
             if event.get("prediction_stage") and event["id"] in enabled),
            None,
        )

    def probabilities(
        self, model_id: str, history: str, patient_id: str, image_event: dict | None, blank_image: bool = False
    ) -> dict[str, float]:
        image_path = None
        with self._inference_lock:
            model = self._model(model_id)
            if model.spec.uses_image and not blank_image:
                if image_event is None:
                    raise BadRequest(f"{model.spec.name.upper()} requires a {self.config.image_noun} available by this cutoff")
                image_path = self.image_path(patient_id, image_event["id"])
            cold_start = not model.loaded
            started = perf_counter()
            try:
                values = model.predict(history, image_path)
            except ModelUnavailable as exc:
                raise Unavailable(str(exc)) from exc
            total_ms = (perf_counter() - started) * 1000
            device_name = model.device_name or self.device_preference
            load_ms = round(float(getattr(model, "last_load_ms", 0) or 0), 3) if cold_start else 0.0
            inference_ms = round(float(getattr(model, "last_inference_ms", total_ms) or total_ms), 3)
        timing = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_id,
            "device": device_name,
            "batch_size": 1,
            "cold_start": cold_start,
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "total_ms": round(total_ms, 3),
        }
        with self._performance_lock:
            self._performance.append(timing)
            self._performance = self._performance[-10000:]
            self._last_timing[model_id] = timing
        return {outcome.code: round(value, 6) for outcome, value in zip(self.config.outcome_classes, values)}

    def _label(self, code: str) -> dict:
        outcome = self.config.outcome_classes[self.config.class_index(code)]
        return {"code": outcome.code, "label": outcome.label}

    def predict(self, selection: Selection, model_id: str) -> dict:
        patient, known, enabled = self.knowledge(selection)
        history, _ = compose_history(known.available, enabled)
        probabilities = self.probabilities(model_id, history, patient["id"], self._enabled_image(known, enabled))
        model = self.models[model_id]
        response = {
            "prediction": self._label(max(probabilities, key=probabilities.get)),
            "probabilities": probabilities,
            "device": model.device_name,
            "model": model_id,
            "performance": dict(self._last_timing[model_id]),
            "knowledge_event_count": len(enabled),
            "available_event_count": len(known.available),
        }
        if self.config.labels_available:
            response["ground_truth"] = self.outcome(patient)
        for name in ("cutoff_day", "cutoff_seconds", "stage_event_id"):
            if getattr(selection, name) is not None:
                response[name] = getattr(selection, name)
        return response

    def performance(self) -> dict:
        with self._performance_lock:
            samples = [dict(sample) for sample in self._performance]
        groups: dict[tuple[str, str], list[dict]] = {}
        for sample in samples:
            groups.setdefault((sample["model"], sample["device"]), []).append(sample)
        summary = []
        for (model, device), rows in sorted(groups.items()):
            warm = [row for row in rows if not row["cold_start"]] or rows
            values = sorted(row["inference_ms"] for row in warm)
            p95 = values[min(len(values) - 1, max(0, int(0.95 * len(values) + 0.999999) - 1))]
            summary.append({
                "model": model,
                "device": device,
                "samples": len(rows),
                "warm_samples": len(warm),
                "cold_starts": sum(row["cold_start"] for row in rows),
                "inference_mean_ms": round(mean(values), 3),
                "inference_median_ms": round(median(values), 3),
                "inference_p95_ms": round(p95, 3),
                "inference_min_ms": round(min(values), 3),
                "inference_max_ms": round(max(values), 3),
            })
        return {"environment": self.runtime(), "device_preference": self.device_preference,
                "summary": summary, "samples": samples}

    def reset_statistics(self) -> None:
        """Discard the cached trajectories and event-type importance of every case."""
        self.trajectories.clear()
        self.importance_cache.clear()

    def reset_performance(self) -> None:
        with self._performance_lock:
            self._performance.clear()
            self._last_timing.clear()

    def _ablation(self, patient_id: str, known: Knowledge, enabled: set[str]) -> dict:
        """Change in risk when one event type is removed, or the image blanked."""
        risk = self.config.risk_class
        text_models = [
            spec.id for spec in self.config.models
            if spec.uses_text and (not spec.uses_image or known.image_event is not None)
        ]

        def risk_of(model_id: str, ids: set[str], blank_image: bool = False) -> float:
            history, _ = compose_history(known.available, ids)
            return self.probabilities(model_id, history, patient_id, known.image_event, blank_image)[risk]

        baseline = {model_id: risk_of(model_id, enabled) for model_id in text_models}
        groups: dict[str, tuple[str, set[str]]] = {}
        for event in known.available:
            if event["id"] in enabled:
                event_type = self.config.event_type_of(event.get("activity", ""))
                groups.setdefault(event_type.key, (event_type.label, set()))[1].add(event["id"])

        rows = [
            {
                "key": key, "label": label, "event_count": len(ids),
                "delta": {model_id: round(risk_of(model_id, enabled - ids) - baseline[model_id], 6) for model_id in text_models},
            }
            for key, (label, ids) in groups.items()
        ]
        multimodal = [spec.id for spec in self.config.models if spec.uses_text and spec.uses_image and spec.id in baseline]
        if multimodal:
            rows.append({
                "key": "image", "label": self.config.image_noun.capitalize(), "event_count": 1,
                "delta": {model_id: round(risk_of(model_id, enabled, blank_image=True) - baseline[model_id], 6) for model_id in multimodal},
            })
        rows.sort(key=lambda row: max(abs(value) for value in row["delta"].values()), reverse=True)
        return {"baseline": baseline, "rows": rows}

    def explain(self, selection: Selection, scope: str = "context", model_id: str | None = None) -> dict:
        """Run the explanation method configured for a model; positive scores push toward the risk class.

        ``model_id`` picks the model (default: the main explanation, the multimodal one if there is one).
        "context" explains what the model reads, as in the paper. "history" explains the
        whole history window by window, so out-of-context events get scores too.
        """
        if scope not in EXPLANATION_SCOPES:
            raise BadRequest(f"Unknown explanation scope {scope!r}")
        settings = self.config.explanation_of(model_id) if model_id else self.config.explanation
        if settings is None:
            raise BadRequest(f"No explanation is configured for {model_id or 'any model'}")
        key = (*selection.key(), scope, settings.model)
        with self._explanations_lock:
            if key in self._explanations:
                self._explanations.move_to_end(key)
                return self._explanations[key]

        patient, known, enabled = self.knowledge(selection)
        model = self._model(settings.model)
        image_event = self._enabled_image(known, enabled) if model.spec.uses_image else None
        if model.spec.uses_image and image_event is None:
            raise BadRequest(f"Explanations need a {self.config.image_noun} available by this cutoff")
        if scope == "history" and not hasattr(model, "count_tokens"):
            raise BadRequest(f"{model.spec.name} reads no text, so there is no history to explain")
        try:
            explainer = load_explainer(settings.method)
        except Exception as exc:  # a user's file can fail in any way; say which file
            raise Unavailable(f"Explanation method {settings.method!r} could not be loaded: {exc}") from exc
        options = settings.arguments()
        history, spans = compose_history(known.available, enabled)
        image_path = self.image_path(patient["id"], image_event["id"]) if image_event else None
        # Two classes: explain the predicted one and flip the sign if it is not the risk class.
        # More classes: explain the risk class itself, so that the colours keep meaning towards/away from it.
        target = self.config.class_index(self.config.risk_class) if len(self.config.outcome_classes) > 2 else None
        try:
            result = explainer(model, history, image_path, target=target, **options)
            context = model.context_spans(history) if model.spec.uses_text else []
            in_context = event_attributions(history, spans, [(start, end, 0.0) for start, end in context])
            tokens, windows = result.tokens, 1
            if scope == "history" and spans:
                parts = history_windows(spans, [model.count_tokens(history[start:end]) for _, start, end in spans], model.window_tokens)
                tokens, windows = [], len(parts)
                for part in parts:
                    offset = part[0][1]
                    scores = explainer(model, history[offset:part[-1][2]], image_path, target=result.target, with_image=False, **options)
                    tokens += [(start + offset, end + offset, score) for start, end, score in scores.tokens]
        except ModelUnavailable as exc:
            raise Unavailable(str(exc)) from exc

        sign = 1 if result.target == self.config.class_index(self.config.risk_class) else -1
        events = event_attributions(history, spans, [(start, end, score * sign) for start, end, score in tokens])
        probabilities = {
            outcome.code: round(value, 6) for outcome, value in zip(self.config.outcome_classes, result.probabilities)
        }
        response = {
            "model": settings.model,
            "method": settings.display,
            "steps": settings.steps,
            "scope": scope,
            "windows": windows,
            "prediction": self._label(max(probabilities, key=probabilities.get)),
            "probabilities": probabilities,
            "image_event_id": image_event["id"] if image_event else None,
            "events": events,
            "out_of_context_event_ids": [event_id for event_id, _, _ in spans if model.spec.uses_text and event_id not in in_context],
            "patches": [[value * sign for value in row] for row in result.patches],
        }
        if self.config.labels_available:
            response["ground_truth"] = self.outcome(patient)
        with self._explanations_lock:
            self._explanations[key] = response
            while len(self._explanations) > EXPLANATION_CACHE_SIZE:
                self._explanations.popitem(last=False)
        return response

    def _compute_trajectory(self, patient: dict) -> list[dict]:
        risk = self.config.risk_class
        points = []
        for point in trajectory_points(patient, ignore=lambda event: self.config.reveals_outcome(event["activity"])):
            selection = Selection(patient["id"], stage_event_id=point.get("stage_event_id"))
            if selection.stage_event_id is None:
                selection.cutoff_seconds = point["offset_seconds"]
            _, known, _ = self.knowledge(selection)
            history, _ = compose_history(known.available, {event["id"] for event in known.available})
            values, predicted = {}, {}
            for spec in self.config.models:
                if spec.uses_image and known.image_event is None:
                    values[spec.id] = predicted[spec.id] = None
                else:
                    probabilities = self.probabilities(spec.id, history, patient["id"], known.image_event)
                    values[spec.id] = probabilities[risk]
                    predicted[spec.id] = max(probabilities, key=probabilities.get)  # any number of classes
            points.append({
                **point,
                "event_count": len(known.available),
                "image_event_id": known.image_event["id"] if known.image_event else None,
                "risk": values,
                "predicted": predicted,
            })
        return points

    def _compute_importance(self, patient: dict) -> list[dict]:
        """Ablation at every image event with the full history."""
        profile = []
        for event in patient["events"]:
            if event.get("prediction_stage"):
                _, known, _ = self.knowledge(Selection(patient["id"], stage_event_id=event["id"]))
                result = self._ablation(patient["id"], known, {item["id"] for item in known.available})
                profile.append({"stage_event_id": event["id"], "offset_seconds": event["offset_seconds"], **result})
        return profile

    def trajectory(self, patient_id: str, compute: bool = False) -> list[dict] | None:
        patient = self.patient(patient_id)
        if compute:
            return self.trajectories.get_or_compute(patient_id, lambda: self._compute_trajectory(patient))
        return self.trajectories.get(patient_id)

    def importance(self, patient_id: str, compute: bool = False) -> list[dict] | None:
        patient = self.patient(patient_id)
        if compute:
            return self.importance_cache.get_or_compute(patient_id, lambda: self._compute_importance(patient))
        return self.importance_cache.get(patient_id)

    def statistics(self) -> dict:
        if not self.config.labels_available:
            raise BadRequest("Statistics are unavailable because no label field was selected")
        trajectories, importance = self.trajectories.all(), self.importance_cache.all()
        return {
            "patients": [
                {
                    "id": summary["id"],
                    "ground_truth": self.outcome(self.patient(summary["id"])),
                    "points": trajectories.get(summary["id"]),
                    "importance": importance.get(summary["id"]),
                }
                for summary in self.list_patients()
            ]
        }

    def health(self) -> dict:
        return {
            "status": "ok" if self.store.ready and all(spec.checkpoint.is_file() for spec in self.config.models) else "setup_required",
            "data_ready": self.store.ready,
            "models": {
                spec.id: {"checkpoint_ready": spec.checkpoint.is_file(), "loaded": self.models[spec.id].loaded,
                          "device": self.models[spec.id].device_name}
                for spec in self.config.models
            },
        }
