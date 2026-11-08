"""HTTP routes. Each one validates input and delegates to ``DemoService``."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .service import DemoService, Selection, Unavailable
from .setup import SetupError, SetupManager


class SelectionBody(BaseModel):
    patient_id: str
    enabled_event_ids: list[str]
    cutoff_day: int | None = None
    stage_event_id: str | None = None
    cutoff_seconds: int | None = None

    def selection(self) -> Selection:
        return Selection(**self.model_dump(exclude={"model", "scope"}))


class PredictionBody(SelectionBody):
    model: str


class ExplanationBody(SelectionBody):
    scope: str = "context"
    model: str | None = None  # the model to explain; default: the main explanation


class SetupPathsBody(BaseModel):
    xes_path: str
    image_dir: str
    model_dir: str


class SetupFieldsBody(BaseModel):
    token: str
    case_field: str
    activity_field: str
    timestamp_field: str
    image_field: str
    label_field: str | None = None
    # One to three models; a role left empty (or not in the configuration) is not used
    text_model: str | None = None
    image_model: str | None = None
    multimodal_model: str | None = None


class RuntimeDeviceBody(BaseModel):
    device: str


def get_service(request: Request) -> DemoService:
    service = request.app.state.service
    if service is None:
        raise Unavailable("Complete the explorer setup first")
    return service


def get_setup(request: Request) -> SetupManager:
    return request.app.state.setup


pages = APIRouter()
api = APIRouter(prefix="/api")


@pages.get("/", response_class=HTMLResponse)
def index(request: Request):
    service = request.app.state.service
    config = service.config if service is not None else request.app.state.setup.base_config
    return request.app.state.templates.TemplateResponse(
        request, "index.html", {"config": config, "public_config": config.public()}
    )


@api.get("/patients")
def list_patients(service: DemoService = Depends(get_service)):
    return {"patients": service.list_patients(), "metadata": service.metadata()}


@api.get("/patients/{patient_id}")
def get_patient(patient_id: str, service: DemoService = Depends(get_service)):
    return service.public_patient(patient_id)


@api.get("/patients/{patient_id}/events/{event_id}/image")
def get_image(patient_id: str, event_id: str, service: DemoService = Depends(get_service)):
    return FileResponse(
        service.image_path(patient_id, event_id),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@api.get("/patients/{patient_id}/trajectory")
def trajectory(patient_id: str, compute: bool = False, service: DemoService = Depends(get_service)):
    return {"patient_id": patient_id, "points": service.trajectory(patient_id, compute)}


@api.get("/patients/{patient_id}/importance")
def importance(patient_id: str, compute: bool = False, service: DemoService = Depends(get_service)):
    return {"patient_id": patient_id, "stages": service.importance(patient_id, compute)}


@api.post("/predict")
def predict(body: PredictionBody, service: DemoService = Depends(get_service)):
    return service.predict(body.selection(), body.model)


@api.post("/explain")
def explain(body: ExplanationBody, service: DemoService = Depends(get_service)):
    return service.explain(body.selection(), body.scope, body.model)


@api.get("/statistics")
def statistics(service: DemoService = Depends(get_service)):
    return service.statistics()


@api.get("/runtime")
def runtime(service: DemoService = Depends(get_service)):
    return service.runtime()


@api.post("/runtime/device")
def set_runtime_device(body: RuntimeDeviceBody, service: DemoService = Depends(get_service)):
    return service.set_device(body.device)


@api.get("/performance")
def performance(service: DemoService = Depends(get_service)):
    return service.performance()


@api.delete("/performance")
def reset_performance(service: DemoService = Depends(get_service)):
    service.reset_performance()
    return {"reset": True}


@api.get("/performance.csv")
def performance_csv(service: DemoService = Depends(get_service)):
    samples = service.performance()["samples"]
    output = io.StringIO()
    fields = ["timestamp", "model", "device", "batch_size", "cold_start", "load_ms", "inference_ms", "total_ms"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(samples)
    return Response(
        output.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inference-performance.csv"},
    )


health = APIRouter(prefix="/api")


@health.get("/setup/status")
def setup_status(setup: SetupManager = Depends(get_setup)):
    return {"configured": setup.configured}


@health.post("/setup/inspect")
def setup_inspect(body: SetupPathsBody, setup: SetupManager = Depends(get_setup)):
    try:
        return setup.inspect(body.xes_path, body.image_dir, body.model_dir)
    except SetupError as exc:
        raise HTTPException(400, str(exc)) from exc


@health.post("/setup/upload")
async def setup_upload(
    request: Request,
    filename: str,
    image_dir: str,
    model_dir: str,
    setup: SetupManager = Depends(get_setup),
):
    try:
        return setup.inspect_upload(filename, await request.body(), image_dir, model_dir)
    except SetupError as exc:
        raise HTTPException(400, str(exc)) from exc


@health.post("/setup/staging")
def setup_staging(setup: SetupManager = Depends(get_setup)):
    return {"token": setup.create_upload_session()}


@health.post("/setup/staging/{token}/file/{kind}")
async def setup_staging_file(
    token: str,
    kind: str,
    path: str,
    request: Request,
    setup: SetupManager = Depends(get_setup),
):
    try:
        target = setup.staging_target(token, kind, path)
        maximum = 3 * 1024 * 1024 * 1024 if kind == "models" else 250 * 1024 * 1024
        size = 0
        with target.open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > maximum:
                    raise SetupError(f"Uploaded {kind} file exceeds the size limit")
                output.write(chunk)
        return {"path": path, "bytes": size}
    except SetupError as exc:
        try:
            target.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        raise HTTPException(400, str(exc)) from exc


@health.post("/setup/staging/{token}/inspect")
async def setup_staging_inspect(
    token: str,
    filename: str,
    request: Request,
    setup: SetupManager = Depends(get_setup),
):
    try:
        return setup.inspect_staged(token, filename, await request.body())
    except SetupError as exc:
        raise HTTPException(400, str(exc)) from exc


@health.post("/setup/apply")
def setup_apply(body: SetupFieldsBody, request: Request, setup: SetupManager = Depends(get_setup)):
    try:
        request.app.state.service = setup.apply(
            body.token,
            {
                "case": body.case_field,
                "activity": body.activity_field,
                "timestamp": body.timestamp_field,
                "image": body.image_field,
                "outcome": body.label_field,
            },
            {"text": body.text_model, "image": body.image_model, "flava": body.multimodal_model},
        )
    except (SetupError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"configured": True, "notes": setup.notes}


@health.get("/health")
def health_check(request: Request):
    service = request.app.state.service
    if service is None:
        return {"status": "setup_required", "data_ready": False, "models": {}}
    return service.health()
