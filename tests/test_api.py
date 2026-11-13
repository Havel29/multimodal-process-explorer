from fastapi.testclient import TestClient

from app.main import create_app


def predict(client, **body):
    return client.post("/api/predict", json={"patient_id": "Patient 01", "model": "roberta", **body})


def test_day_cutoff_uses_only_available_enabled_events(client, service):
    response = predict(client, cutoff_day=1, enabled_event_ids=["E001", "E002"])

    assert response.status_code == 200
    assert response.json()["available_event_count"] == 2
    assert response.json()["probabilities"] == {"non_survival": 0.8, "home": 0.2}
    assert response.json()["prediction"]["code"] == "non_survival"
    assert service.models["roberta"].histories[-1] == "admission day one fever "


def test_day_cutoff_rejects_future_knowledge(client):
    response = predict(client, cutoff_day=1, enabled_event_ids=["E004"])

    assert response.status_code == 400
    assert "after the cutoff" in response.json()["detail"]


def test_image_model_requires_a_radiograph_by_the_cutoff(client):
    response = predict(client, model="flava", cutoff_day=1, enabled_event_ids=["E001"])

    assert response.status_code == 400
    assert "requires a radiograph" in response.json()["detail"]


def test_unknown_model_is_rejected(client):
    assert predict(client, model="gpt", cutoff_day=1, enabled_event_ids=[]).status_code == 400


def test_free_cutoff_selects_the_latest_radiograph(client):
    response = predict(client, model="vit", cutoff_seconds=150000, enabled_event_ids=["E001", "E002", "E003"])

    assert response.status_code == 200
    assert response.json()["cutoff_seconds"] == 150000


def test_excluded_radiograph_is_not_given_to_image_models(client):
    response = predict(client, model="vit", cutoff_seconds=150000, enabled_event_ids=["E001", "E002"])

    assert response.status_code == 400
    assert "requires a radiograph" in response.json()["detail"]


def test_importance_reports_the_change_in_risk_per_event_type(client):
    response = client.get("/api/patients/Patient 01/importance?compute=true")

    assert response.status_code == 200
    body = response.json()["stages"][0]
    assert body["baseline"] == {"roberta": 0.8, "flava": 0.8}
    deltas = {row["key"]: row["delta"] for row in body["rows"]}
    assert deltas["vitals"]["roberta"] == -0.5
    assert deltas["ed"]["roberta"] == 0.0
    assert deltas["image"] == {"flava": -0.1}


EXPLAIN_AT_XRAY = {"patient_id": "Patient 01", "stage_event_id": "E003", "enabled_event_ids": ["E001", "E002", "E003"]}


def test_explanation_marks_events_outside_the_model_context(client):
    # The fake model reads the last four words: "day one fever xray".
    body = client.post("/api/explain", json=EXPLAIN_AT_XRAY).json()

    assert body["scope"] == "context"
    assert body["out_of_context_event_ids"] == ["E001"]
    assert set(body["events"]) == {"E002", "E003"}


def test_whole_history_explanation_scores_every_event_window_by_window(client):
    body = client.post("/api/explain", json={**EXPLAIN_AT_XRAY, "scope": "history"}).json()

    assert body["windows"] == 2
    assert body["out_of_context_event_ids"] == ["E001"]
    assert set(body["events"]) == {"E001", "E002", "E003"}
    assert body["patches"] == [[0.5]]


def test_unknown_explanation_scope_is_rejected(client):
    assert client.post("/api/explain", json={**EXPLAIN_AT_XRAY, "scope": "everything"}).status_code == 400


def test_trajectory_runs_to_the_end_of_the_stay(client):
    points = client.get("/api/patients/Patient 01/trajectory?compute=true").json()["points"]

    assert [point["kind"] for point in points] == ["day", "stage", "day", "day"]
    assert points[-1]["offset_seconds"] >= 200000
    assert points[0]["risk"]["vit"] is None
    assert points[-1]["risk"]["vit"] is not None
    assert points[-1]["predicted"]["vit"] in {"non_survival", "home"} and points[0]["predicted"]["vit"] is None


def test_events_that_state_the_outcome_are_never_given_to_the_models(config, service):
    from .conftest import EVENTS, event, write_bundle
    from app.store import PatientStore

    # The discharge (reveals_outcome in the CDSL configuration) a day after the last event, mentioning fever
    write_bundle(config, [*EVENTS, event("E005", "discharged", 300000, "discharged with fever ")])
    service.store = PatientStore(config.bundle_path)
    client = TestClient(create_app(service=service))

    response = predict(client, cutoff_day=4, enabled_event_ids=["E001", "E003", "E004", "E005"])
    assert response.status_code == 200
    assert response.json()["available_event_count"] == 4
    assert response.json()["probabilities"]["non_survival"] == 0.3  # "fever" only in E005 and E002 (off)

    points = client.get("/api/patients/Patient 01/trajectory?compute=true").json()["points"]
    assert points[-1]["offset_seconds"] < 300000  # the days end with the last event the models can get
    assert all("discharged" not in history for history in service.models["roberta"].histories)


def test_statistics_can_be_reset(client):
    client.get("/api/patients/Patient 01/trajectory?compute=true")

    assert client.delete("/api/statistics").json() == {"reset": True}
    assert client.get("/api/statistics").json()["patients"][0]["points"] is None


def test_statistics_lists_cached_results(client):
    client.get("/api/patients/Patient 01/trajectory?compute=true")

    patients = client.get("/api/statistics").json()["patients"]

    assert patients[0]["ground_truth"]["code"] == "home"
    assert patients[0]["points"] and patients[0]["importance"] is None


def test_no_login_is_asked(service, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    client = TestClient(create_app(service=service))

    assert client.get("/api/patients").status_code == 200
    assert client.get("/").status_code == 200


def test_page_is_rendered_from_the_configuration(client, config):
    page = client.get("/").text

    for model in config.models:
        assert f'data-model="{model.id}"' in page
        assert f"--model-{model.id}: {model.color}" in page
    assert config.disclaimer in page


def test_prediction_records_exportable_performance_metrics(client):
    result = predict(client, cutoff_day=1, enabled_event_ids=["E001", "E002"]).json()

    assert result["performance"]["model"] == "roberta"
    assert result["performance"]["device"] == "test"
    assert result["performance"]["inference_ms"] >= 0
    metrics = client.get("/api/performance").json()
    assert metrics["summary"][0]["samples"] == 1
    assert "timestamp,model,device,batch_size,cold_start,load_ms,inference_ms,total_ms" in client.get("/api/performance.csv").text


def test_runtime_device_and_metric_reset(client):
    assert client.get("/api/runtime").json()["preference"] == "auto"
    assert client.post("/api/runtime/device", json={"device": "cpu"}).json()["preference"] == "cpu"
    assert client.post("/api/runtime/device", json={"device": "quantum"}).status_code == 400
    assert client.delete("/api/performance").json() == {"reset": True}


def test_explanation_method_can_be_a_file(client, service, tmp_path):
    import dataclasses

    method = tmp_path / "first_word.py"
    method.write_text(
        "def explain(model, history, image_path, target=None, with_image=True, scale=1.0, **options):\n"
        "    return {'probabilities': [0.6, 0.4], 'target': 0, 'tokens': [(0, 3, scale)], 'patches': [[0.2, 0.1], [0.0, 0.3]]}\n",
        encoding="utf-8",
    )
    explanation = dataclasses.replace(service.config.explanation, method=str(method), options={"scale": 0.5})
    service.config = dataclasses.replace(service.config, explanations=(explanation,))

    body = client.post("/api/explain", json=EXPLAIN_AT_XRAY).json()

    assert body["method"] == "first word"
    assert body["events"]["E001"]["segments"][0] == ["adm", 0.5]
    assert len(body["patches"]) == 2


def test_missing_explanation_file_is_reported(client, service):
    import dataclasses

    explanation = dataclasses.replace(service.config.explanation, method="explainers/missing.py")
    service.config = dataclasses.replace(service.config, explanations=(explanation,))

    response = client.post("/api/explain", json=EXPLAIN_AT_XRAY)

    assert response.status_code == 503
    assert "missing.py" in response.json()["detail"]


def test_each_explained_model_has_its_own_explanation(client, service):
    import dataclasses

    from app.config import ExplanationSettings
    from app.models import Attributions

    calls = []

    def recording(model, history, image_path, target=None, with_image=True, **options):
        calls.append(model.spec.id)
        return Attributions(probabilities=[0.6, 0.4], target=0, tokens=[(0, 3, 1.0)], patches=[])

    import app.explainers as explainers
    explainers.BUILT_IN["recording"] = recording
    ids = [model.id for model in service.config.models]
    service.config = dataclasses.replace(service.config, explanations=tuple(
        ExplanationSettings(model=model_id, method="recording") for model_id in ids))

    for model_id in ids:
        body = client.post("/api/explain", json={**EXPLAIN_AT_XRAY, "model": model_id}).json()
        assert body["model"] == model_id
    assert calls == ids
    del explainers.BUILT_IN["recording"]


def test_more_than_two_outcome_classes(config):
    import dataclasses

    from app.cache import AnalyticsCache
    from app.config import OutcomeClass
    from app.service import DemoService
    from app.store import PatientStore

    from .conftest import FakeModel, write_bundle

    class ThreeClasses(FakeModel):
        def predict(self, history, image_path):
            self.histories.append(history)
            return [0.2, 0.3, 0.5]  # non_survival (risk), home, transfer

        def explain(self, history, image_path, target=None, **options):
            self.targets = [*getattr(self, "targets", []), target]
            return super().explain(history, image_path, target=target, **options)

    three = dataclasses.replace(config, outcome_classes=(*config.outcome_classes, OutcomeClass("transfer", "Transfer", "Transfer")))
    write_bundle(three)
    service = DemoService(
        config=three, store=PatientStore(three.bundle_path), models={spec.id: ThreeClasses(spec) for spec in three.models},
        trajectories=AnalyticsCache(three.analytics_dir / "trajectories.json", three.bundle_path),
        importance=AnalyticsCache(three.analytics_dir / "importance.json", three.bundle_path),
    )
    client = TestClient(create_app(service=service))

    points = client.get("/api/patients/Patient 01/trajectory?compute=true").json()["points"]
    assert points[-1]["predicted"]["roberta"] == "transfer" and points[-1]["risk"]["roberta"] == 0.2

    result = client.post("/api/explain", json=EXPLAIN_AT_XRAY).json()
    assert result["prediction"]["code"] == "transfer"
    assert service.models["flava"].targets[0] == 0  # the risk class is explained, not the predicted one


def test_everything_but_the_explanation_works_without_one(client, service):
    import dataclasses

    service.config = dataclasses.replace(service.config, explanations=())
    page = TestClient(create_app(service=service)).get("/").text

    assert 'class="analysis hidden"' in page and "data-explain=" not in page
    assert predict(client, cutoff_day=1, enabled_event_ids=["E001"]).status_code == 200
    assert client.get("/api/patients/Patient 01/trajectory?compute=true").status_code == 200
    assert client.get("/api/patients/Patient 01/importance?compute=true").status_code == 200
    assert client.post("/api/explain", json=EXPLAIN_AT_XRAY).status_code == 400
