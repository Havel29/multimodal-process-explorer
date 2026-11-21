import json

import pytest

from app.store import DemoDataError, PatientStore


def test_store_loads_and_resolves_an_image(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "scan.jpg").write_bytes(b"jpeg")
    bundle = {
        "metadata": {"source": "test"},
        "patients": [{
            "id": "Patient 01", "duration_seconds": 42, "true_outcome": "Home",
            "events": [{
                "id": "E001", "activity": "performed the digital radiography",
                "prediction_stage": True, "image": "images/scan.jpg",
            }],
        }],
    }
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    store = PatientStore(path)
    assert store.ready
    assert store.list_patients()[0]["prediction_stage_count"] == 1
    assert store.image_path("Patient 01", "E001").name == "scan.jpg"


def test_store_rejects_missing_bundle(tmp_path):
    store = PatientStore(tmp_path / "missing.json")
    assert not store.ready
    with pytest.raises(DemoDataError):
        store.list_patients()


def test_store_blocks_paths_outside_bundle(tmp_path):
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"jpeg")
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"patients": [{
        "id": "Patient 01", "events": [{"id": "E001", "image": "../outside.jpg"}],
    }]}), encoding="utf-8")
    store = PatientStore(path)
    with pytest.raises(DemoDataError):
        store.image_path("Patient 01", "E001")

