from fastapi.testclient import TestClient

from app.cache import AnalyticsCache
from app.config import load_config
from app.main import create_app
from app.service import DemoService
from app.store import PatientStore

from .conftest import FakeModel
from .synthetic import write_bundle, write_config

TYPES = 50


def test_fifty_event_types_are_served(tmp_path, monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    config = load_config(write_config(tmp_path, TYPES))
    write_bundle(tmp_path, TYPES)
    service = DemoService(
        config, PatientStore(config.bundle_path), {spec.id: FakeModel(spec) for spec in config.models},
        AnalyticsCache(tmp_path / "trajectories.json", config.bundle_path), AnalyticsCache(tmp_path / "importance.json", config.bundle_path),
    )
    client = TestClient(create_app(service=service))

    assert len(config.public()["eventTypes"]) == TYPES + 2  # plus the image type and "other"
    assert client.get("/").status_code == 200
    importance = client.get("/api/patients/Patient 01/importance?compute=true")
    assert importance.status_code == 200
    assert len(importance.json()["stages"][-1]["rows"]) > 40
