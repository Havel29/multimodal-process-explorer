import dataclasses
import json
import os
import re
from pathlib import Path

# The tests use the complete CDSL configuration (event types, templates, three models)
os.environ["EXPLORER_CONFIG"] = str(Path(__file__).resolve().parents[1] / "config" / "cdsl_covid.toml")

import pytest
from fastapi.testclient import TestClient

from app.cache import AnalyticsCache
from app.config import load_config
from app.main import create_app
from app.models import Attributions
from app.service import DemoService
from app.store import PatientStore


class FakeModel:
    """Stands in for a classifier: risk is 0.8 if the history mentions fever.

    Its context is the last four words, and every word it reads scores 1.
    """

    loaded = True
    device_name = "test"
    window_tokens = 4

    def count_tokens(self, text):
        return len(text.split())

    def context_spans(self, history):
        return [match.span() for match in re.finditer(r"\S+", history)][-self.window_tokens:]

    def explain(self, history, image_path, steps=20, internal_batch_size=4, target=None, with_image=True):
        probabilities = self.predict(history, image_path)
        if target is None:
            target = max(range(len(probabilities)), key=probabilities.__getitem__)
        tokens = [(start, end, 1.0) for start, end in self.context_spans(history)]
        return Attributions(probabilities, target, tokens, [[0.5]] if with_image else [])

    def __init__(self, spec):
        self.spec = spec
        self.histories = []

    def predict(self, history, image_path):
        self.histories.append(history)
        risk = 0.8 if "fever" in history else 0.3
        if image_path is None and self.spec.uses_image:
            risk -= 0.1
        return [risk, 1 - risk]


def event(event_id, activity, offset, text, stage=False, image=None):
    item = {"id": event_id, "activity": activity, "offset_seconds": offset, "model_text": text, "summary": text,
            "prediction_stage": stage}
    if image:
        item["image"] = image
    return item


EVENTS = [
    event("E001", "entered in ED", 0, "admission "),
    event("E002", "received vital signs check", 86400, "day one fever "),
    event("E003", "performed the digital radiography", 90000, "xray ", stage=True, image="images/scan.jpg"),
    event("E004", "was dispensed medicine", 200000, "CEFTRIAXONA "),
]


@pytest.fixture
def config(tmp_path):
    return dataclasses.replace(load_config(), bundle_path=tmp_path / "demo.json", analytics_dir=tmp_path)


def write_bundle(config, events=EVENTS, outcome="Home"):
    root = config.bundle_path.parent
    (root / "images").mkdir(exist_ok=True)
    (root / "images" / "scan.jpg").write_bytes(b"jpeg")
    bundle = {"patients": [{"id": "Patient 01", "true_outcome": outcome, "duration_seconds": events[-1]["offset_seconds"], "events": events}]}
    config.bundle_path.write_text(json.dumps(bundle), encoding="utf-8")


@pytest.fixture
def service(config):
    write_bundle(config)
    return DemoService(
        config=config,
        store=PatientStore(config.bundle_path),
        models={spec.id: FakeModel(spec) for spec in config.models},
        trajectories=AnalyticsCache(config.analytics_dir / "trajectories.json", config.bundle_path),
        importance=AnalyticsCache(config.analytics_dir / "importance.json", config.bundle_path),
    )


@pytest.fixture
def client(service, monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    return TestClient(create_app(service=service))


ROOT = Path(__file__).resolve().parents[1]
