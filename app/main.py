"""Application factory. ``uvicorn app.main:app`` serves the configured demo."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import api
from .config import DemoConfig, load_config
from .service import DemoService, ServiceError
from .setup import SetupManager

APP_DIR = Path(__file__).resolve().parent


class FreshStaticFiles(StaticFiles):
    """Static files the browser revalidates on every load."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(config: DemoConfig | None = None, service: DemoService | None = None) -> FastAPI:
    explicit_config = config is not None
    config = config or (service.config if service else load_config())
    setup = SetupManager(config, configured=service is not None)
    if service is None and not explicit_config:
        service = setup.restore()
    app = FastAPI(title=f"{config.title} {config.subtitle}".strip())
    app.state.service = service
    app.state.setup = setup
    app.state.templates = Jinja2Templates(directory=APP_DIR / "templates")

    @app.exception_handler(ServiceError)
    def service_error(request: Request, exc: ServiceError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    app.mount("/static", FreshStaticFiles(directory=APP_DIR / "static"), name="static")
    app.include_router(api.health)
    app.include_router(api.pages)
    app.include_router(api.api)
    return app


app = create_app()
