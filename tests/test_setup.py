import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.setup import SetupError, SetupManager, read_xes
from app.service import BadRequest


def make_inputs(tmp_path):
    xes = tmp_path / "patients.xes"
    xes.write_text("placeholder", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / "scan.jpg").write_bytes(b"jpeg")
    models = tmp_path / "models"
    models.mkdir()
    for name in ("text.pth", "image.pth", "multimodal.pth"):
        (models / name).write_bytes(b"weights")
    return xes, images, models


def frame(case_count=20):
    rows = []
    for index in range(1, case_count + 1):
        rows += [
            {"case:id": f"P{index}", "act": "entered in ED", "when": "2020-01-01T00:00:00Z", "scan": None, "result": "Home"},
            {"case:id": f"P{index}", "act": "performed the digital radiography", "when": "2020-01-01T01:00:00Z", "scan": "scan.jpg", "result": "Home"},
        ]
    return pd.DataFrame(rows)


def selections(label="result"):
    return {
        "case": "case:id", "activity": "act", "timestamp": "when", "image": "scan", "outcome": label,
    }


MODELS = {"text": "text.pth", "image": "image.pth", "flava": "multimodal.pth"}


def test_setup_inspects_xes_and_builds_runtime_service(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")

    inspected = manager.inspect(str(xes), str(images), str(models))
    service = manager.apply(inspected["token"], selections(), MODELS)

    assert inspected["event_count"] == 40
    assert set(inspected["model_files"]) == set(MODELS.values())
    assert service.config.labels_available
    assert service.public_patient("Case 01")["ground_truth"]["code"] == "home"
    payload = json.loads(service.config.bundle_path.read_text(encoding="utf-8"))
    assert payload["patients"][0]["events"][1]["prediction_stage"]

    restored = SetupManager(config, runtime_dir=tmp_path / "runtime").restore()
    assert restored is not None
    assert restored.public_patient("Case 01")["ground_truth"]["code"] == "home"


def test_optional_label_removes_ground_truth_and_statistics(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(images), str(models))

    service = manager.apply(inspected["token"], selections(None), MODELS)

    assert not service.config.labels_available
    assert "ground_truth" not in service.public_patient("Case 01")
    with pytest.raises(BadRequest, match="no label field"):
        service.statistics()


def test_setup_migrates_the_latest_complete_legacy_upload(config, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "demo.json").write_text(json.dumps({
        "metadata": {"labels_available": True},
        "patients": [{"id": "Case 01", "true_outcome": "Home", "events": [{
            "id": "E001", "activity": "entered in ED", "offset_seconds": 0,
            "model_text": "admission", "summary": "admission", "prediction_stage": False,
        }]}],
    }), encoding="utf-8")
    model_dir = runtime / "uploads" / "legacy" / "models"
    model_dir.mkdir(parents=True)
    for spec in config.models:
        (model_dir / spec.checkpoint.name).write_bytes(b"weights")

    manager = SetupManager(config, runtime_dir=runtime)
    restored = manager.restore()

    assert restored is not None
    assert manager.state_path.is_file()
    assert restored.public_patient("Case 01")["ground_truth"]["code"] == "home"


def test_setup_rejects_missing_paths(config, tmp_path):
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")

    with pytest.raises(SetupError, match="File not found"):
        manager.inspect(str(tmp_path / "missing.xes"), str(tmp_path), str(tmp_path))


def test_setup_accepts_any_number_of_cases(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame(case_count=3))
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(images), str(models))

    service = manager.apply(inspected["token"], selections(), MODELS)

    assert [case["id"] for case in service.list_patients()] == ["Case 01", "Case 02", "Case 03"]


def test_setup_rejects_image_folder_that_does_not_match_xes_paths(config, tmp_path, monkeypatch):
    xes, _, models = make_inputs(tmp_path)
    wrong_images = tmp_path / "prepared-images"
    wrong_images.mkdir()
    (wrong_images / "E001.jpg").write_bytes(b"jpeg")
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(wrong_images), str(models))

    with pytest.raises(SetupError, match="No XES image paths match"):
        manager.apply(inspected["token"], selections(), MODELS)


def test_browser_upload_is_saved_and_parsed_before_showing_dropdowns(config, tmp_path, monkeypatch):
    _, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda path: frame() if path.read_bytes() == b"uploaded-xes" else None)
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")

    inspected = manager.inspect_upload("selected-test-subset.xes", b"uploaded-xes", str(images), str(models))

    assert inspected["event_count"] == 40
    assert "case:id" in inspected["fields"]
    assert set(inspected["model_files"]) == set(MODELS.values())


def test_setup_upload_endpoint_accepts_the_xes_file_body(config, tmp_path, monkeypatch):
    _, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    app = create_app(config)
    app.state.setup.runtime_dir = tmp_path / "runtime"

    response = TestClient(app).post(
        "/api/setup/upload",
        params={"filename": "subset.xes", "image_dir": str(images), "model_dir": str(models)},
        content=b"uploaded-xes",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    assert response.json()["event_count"] == 40


def test_folder_picker_uploads_are_staged_before_xes_parsing(config, tmp_path, monkeypatch):
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    app = create_app(config)
    app.state.setup.runtime_dir = tmp_path / "runtime"
    client = TestClient(app)

    token = client.post("/api/setup/staging").json()["token"]
    image = client.post(
        f"/api/setup/staging/{token}/file/images", params={"path": "nested/scan.jpg"}, content=b"jpeg",
    )
    for name in MODELS.values():
        response = client.post(
            f"/api/setup/staging/{token}/file/models", params={"path": name}, content=b"weights",
        )
        assert response.status_code == 200
    inspected = client.post(
        f"/api/setup/staging/{token}/inspect", params={"filename": "subset.xes"}, content=b"uploaded-xes",
    )

    assert image.status_code == 200
    assert (tmp_path / "runtime" / "uploads" / token / "images" / "nested" / "scan.jpg").is_file()
    assert inspected.status_code == 200
    assert inspected.json()["event_count"] == 40
    assert set(inspected.json()["model_files"]) == set(MODELS.values())


def test_pm4py_xes_round_trip_exposes_dropdown_fields(tmp_path):
    import pm4py

    source = pd.DataFrame([{
        "patient_id": "P1", "activity": "entered in ED",
        "time:timestamp": pd.Timestamp("2020-01-01T00:00:00Z"), "path": "scan.jpg", "outcome": "Home",
    }])
    formatted = pm4py.format_dataframe(
        source, case_id="patient_id", activity_key="activity", timestamp_key="time:timestamp"
    )
    path = tmp_path / "patient.xes"
    pm4py.write_xes(formatted, str(path), case_id_key="case:concept:name")

    parsed = read_xes(path)

    assert {"case:concept:name", "concept:name", "time:timestamp", "path", "outcome"} <= set(parsed.columns)


def test_healthcheck_stays_available_before_setup():
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "setup_required"


def test_a_setup_deletes_the_uploads_of_earlier_setups(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    uploads = tmp_path / "runtime" / "uploads"
    (uploads / "earlier" / "models").mkdir(parents=True)
    (uploads / "earlier" / "models" / "text.pth").write_bytes(b"weights")
    (uploads / "earlier.xes").write_text("log", encoding="utf-8")
    in_progress = manager.create_upload_session()  # another upload, not parsed yet

    inspected = manager.inspect(str(xes), str(images), str(models))
    manager.apply(inspected["token"], selections(), MODELS)

    assert sorted(entry.name for entry in uploads.iterdir()) == [in_progress]


def test_a_setup_can_use_a_single_model_and_is_restored_with_it(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(images), str(models))

    service = manager.apply(inspected["token"], selections(), {"text": "text.pth", "image": "", "flava": None})

    assert [spec.kind for spec in service.config.models] == ["text"]
    # Only the explanations of the models in use are kept
    assert all(item.model == service.config.models[0].id for item in service.config.explanations)
    restored = SetupManager(config, runtime_dir=tmp_path / "runtime").restore()
    assert [spec.kind for spec in restored.config.models] == ["text"]


def test_an_edited_configuration_for_the_same_data_replaces_the_saved_one(config, tmp_path, monkeypatch):
    import dataclasses

    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(images), str(models))
    manager.apply(inspected["token"], selections(), {"text": "text.pth", "image": None, "flava": None})

    # A copy of the configuration with another risk class (same classes, same event types) is used
    edited = dataclasses.replace(config, risk_class="home")
    assert SetupManager(edited, runtime_dir=tmp_path / "runtime").restore().config.risk_class == "home"
    # A configuration for other data (other classes) is not: the saved one stays
    other = dataclasses.replace(config, outcome_classes=config.outcome_classes[:1], risk_class=config.outcome_classes[0].code)
    restored = SetupManager(other, runtime_dir=tmp_path / "runtime").restore()
    assert [item.code for item in restored.config.outcome_classes] == [item.code for item in config.outcome_classes]


def test_a_setup_needs_at_least_one_model(config, tmp_path, monkeypatch):
    xes, images, models = make_inputs(tmp_path)
    monkeypatch.setattr("app.setup.read_xes", lambda _: frame())
    manager = SetupManager(config, runtime_dir=tmp_path / "runtime")
    inspected = manager.inspect(str(xes), str(images), str(models))

    with pytest.raises(SetupError, match="at least one model"):
        manager.apply(inspected["token"], selections(), {"text": "", "image": None, "flava": ""})
