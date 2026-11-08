"""Runtime setup: load an XES log, its images and one to three model checkpoints (text, image, multimodal)."""
from __future__ import annotations

import dataclasses
import json
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock

import pandas as pd

from .config import DemoConfig, PROJECT_ROOT, load_config
from .config_writer import dump_config
from .inference import complete_config
from .service import DemoService

SESSION_CONFIG = "config.toml"


class SetupError(ValueError):
    pass


@dataclass
class SetupDraft:
    token: str
    log: pd.DataFrame
    xes_path: Path
    image_dir: Path
    model_dir: Path
    fields: tuple[str, ...]
    model_files: tuple[str, ...]


def read_xes(path: Path) -> pd.DataFrame:
    try:
        import pm4py
    except ImportError as exc:
        raise SetupError("Pm4py is not installed. Run pip install -r requirements.txt") from exc
    try:
        log = pm4py.read_xes(str(path), return_legacy_log_object=False)
        frame = log if isinstance(log, pd.DataFrame) else pm4py.convert_to_dataframe(log)
    except Exception as exc:
        raise SetupError(f"Unable to parse the XES file: {exc}") from exc
    if frame.empty:
        raise SetupError("The XES file contains no events")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _existing_path(value: str, kind: str) -> Path:
    path = Path(value).expanduser().resolve()
    valid = path.is_file() if kind == "file" else path.is_dir()
    if not valid:
        raise SetupError(f"{kind.capitalize()} not found: {path}")
    return path


class SetupManager:
    def __init__(self, base_config: DemoConfig, configured: bool = False, runtime_dir: Path | None = None):
        self.base_config = base_config
        self.configured = configured
        self.runtime_dir = runtime_dir or PROJECT_ROOT / "data" / "runtime"
        self._drafts: dict[str, SetupDraft] = {}
        self._uploads: dict[str, Path] = {}
        self._lock = RLock()
        # What the last apply read from the log instead of the configuration
        self.notes: list[str] = []

    @property
    def state_path(self) -> Path:
        return self.runtime_dir / "setup.json"

    def restore(self) -> DemoService | None:
        """Restore the last applied setup after an application restart."""
        try:
            migrated = not self.state_path.is_file()
            state = self._legacy_state() if migrated else json.loads(self.state_path.read_text(encoding="utf-8"))
            if state is None:
                return None
            checkpoints = {
                kind: Path(value).resolve() for kind, value in state["checkpoints"].items()
            }
            if not checkpoints or not all(path.is_file() for path in checkpoints.values()):
                return None
            # The configuration completed from the log at setup, if there is one
            saved = self.runtime_dir / state["config"] if state.get("config") else None
            base = load_config(saved) if saved and saved.is_file() else self.base_config
            # The models of that setup: one to three, each with its checkpoint
            models = tuple(
                dataclasses.replace(spec, checkpoint=checkpoints[spec.kind])
                for spec in base.models if spec.kind in checkpoints
            )
            bundle = self.runtime_dir / "demo.json"
            if not models or not bundle.is_file():
                return None
            config = dataclasses.replace(
                base.with_models(models),
                bundle_path=bundle,
                analytics_dir=self.runtime_dir,
                labels_available=bool(state["labels_available"]),
            )
            service = DemoService.from_config(config)
            if migrated:
                self._save(config)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        self.configured = True
        return service

    def _legacy_state(self) -> dict | None:
        """Recover the newest complete upload made before setup persistence existed."""
        bundle = self.runtime_dir / "demo.json"
        uploads = self.runtime_dir / "uploads"
        if not bundle.is_file() or not uploads.is_dir():
            return None
        names = {spec.kind: spec.checkpoint.name for spec in self.base_config.models}
        candidates = []
        for model_dir in uploads.glob("*/models"):
            checkpoints = {kind: model_dir / name for kind, name in names.items()}
            if all(path.is_file() for path in checkpoints.values()):
                candidates.append((model_dir.stat().st_mtime_ns, checkpoints))
        if not candidates:
            return None
        checkpoints = max(candidates, key=lambda item: item[0])[1]
        metadata = json.loads(bundle.read_text(encoding="utf-8")).get("metadata", {})
        return {
            "version": 1,
            "labels_available": metadata.get("labels_available", True),
            "checkpoints": {kind: str(path.resolve()) for kind, path in checkpoints.items()},
        }

    def _save(self, config: DemoConfig) -> None:
        """Atomically persist only what is needed to rebuild the service."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "labels_available": config.labels_available,
            "checkpoints": {spec.kind: str(spec.checkpoint.resolve()) for spec in config.models},
        }
        if (self.runtime_dir / SESSION_CONFIG).is_file():
            state["config"] = SESSION_CONFIG
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def create_upload_session(self) -> str:
        token = secrets.token_urlsafe(18)
        root = self.runtime_dir / "uploads" / token
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._uploads[token] = root
        return token

    def staging_target(self, token: str, kind: str, relative_path: str) -> Path:
        if kind not in {"images", "models"}:
            raise SetupError("Unknown setup folder type")
        with self._lock:
            root = self._uploads.get(token)
        if root is None:
            raise SetupError("Folder-upload session expired; select the folders again")
        relative = PurePosixPath(relative_path.replace("\\", "/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise SetupError("Invalid relative file path")
        target = root / kind / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def inspect_staged(self, token: str, filename: str, content: bytes) -> dict:
        with self._lock:
            root = self._uploads.get(token)
        if root is None:
            raise SetupError("Folder-upload session expired; select the folders again")
        name = Path(filename).name
        suffixes = "".join(Path(name).suffixes[-2:]).lower()
        if not name or not (suffixes.endswith(".xes") or suffixes.endswith(".xes.gz")):
            raise SetupError("The event log must be an .xes or .xes.gz file")
        if not content:
            raise SetupError("The uploaded XES file is empty")
        if len(content) > 100 * 1024 * 1024:
            raise SetupError("The uploaded XES file exceeds the 100 MB demo limit")
        xes = root / ("input.xes.gz" if suffixes.endswith(".xes.gz") else "input.xes")
        xes.write_bytes(content)
        result = self._inspect(xes, root / "images", root / "models", token)
        with self._lock:
            self._uploads.pop(token, None)
        return result

    def inspect(self, xes_path: str, image_dir: str, model_dir: str) -> dict:
        xes = _existing_path(xes_path, "file")
        if xes.suffix.lower() not in {".xes", ".gz"}:
            raise SetupError("The event log must be an .xes or .xes.gz file")
        images = _existing_path(image_dir, "directory")
        models = _existing_path(model_dir, "directory")
        return self._inspect(xes, images, models)

    def inspect_upload(self, filename: str, content: bytes, image_dir: str, model_dir: str) -> dict:
        """Persist and parse an XES uploaded by the browser."""
        name = Path(filename).name
        suffixes = "".join(Path(name).suffixes[-2:]).lower()
        if not name or not (suffixes.endswith(".xes") or suffixes.endswith(".xes.gz")):
            raise SetupError("The event log must be an .xes or .xes.gz file")
        if not content:
            raise SetupError("The uploaded XES file is empty")
        if len(content) > 100 * 1024 * 1024:
            raise SetupError("The uploaded XES file exceeds the 100 MB demo limit")
        images = _existing_path(image_dir, "directory")
        models = _existing_path(model_dir, "directory")
        token = secrets.token_urlsafe(18)
        uploads = self.runtime_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        xes = uploads / f"{token}{'.xes.gz' if suffixes.endswith('.xes.gz') else '.xes'}"
        xes.write_bytes(content)
        return self._inspect(xes, images, models, token)

    def _inspect(self, xes: Path, images: Path, models: Path, token: str | None = None) -> dict:
        frame = read_xes(xes)
        files = tuple(sorted(
            path.name
            for path in models.iterdir()
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".bin", ".ckpt", ".safetensors"}
        ))
        if not files:
            raise SetupError(f"No model checkpoint was found in {models}")
        token = token or secrets.token_urlsafe(18)
        fields = tuple(column for column in frame.columns if not str(column).startswith("@@"))  # pm4py row indices
        draft = SetupDraft(token, frame, xes, images, models, fields, files)
        with self._lock:
            self._drafts = {token: draft}
        return {
            "token": token,
            "fields": list(draft.fields),
            "model_files": list(files),
            "field_defaults": {
                name: value if value in draft.fields else None
                for name, value in self.base_config.preparation.columns.items()
            },
            # The roles of the configuration (the form shows one field each) and their configured files
            "model_defaults": {
                ("multimodal" if spec.kind == "flava" else spec.kind):
                    spec.checkpoint.name if spec.checkpoint.name in files else None
                for spec in self.base_config.models
            },
            "event_count": len(frame),
        }

    def apply(self, token: str, fields: dict[str, str | None], model_files: dict[str, str]) -> DemoService:
        with self._lock:
            draft = self._drafts.get(token)
        if draft is None:
            raise SetupError("Setup session expired; parse the paths again")

        required = ("case", "activity", "timestamp", "image")
        missing = [name for name in required if not fields.get(name)]
        if missing:
            raise SetupError(f"Select the required fields: {', '.join(missing)}")
        unknown = [value for value in fields.values() if value and value not in draft.fields]
        if unknown:
            raise SetupError(f"Unknown XES field: {unknown[0]}")
        from scripts.prepare_demo_data import image_reference, image_source

        image_values = draft.log[str(fields["image"])].dropna()
        references = [value for value in image_values if image_reference(value)]
        if not references:
            raise SetupError("The selected image field contains no image paths")
        matched_images = sum(image_source(value, draft.image_dir) is not None for value in references)
        if not matched_images:
            raise SetupError(
                "No XES image paths match the selected source image folder. "
                "Select the folder whose internal paths correspond to the image field."
            )
        # One to three models: a role left empty is not used
        configured = {spec.kind for spec in self.base_config.models}
        model_files = {kind: name for kind, name in model_files.items() if name and kind in configured}
        if not model_files:
            raise SetupError("Select at least one model checkpoint")
        for kind, name in model_files.items():
            if name not in draft.model_files:
                raise SetupError(f"Select a valid {'multimodal' if kind == 'flava' else kind} model file")

        columns = dict(self.base_config.preparation.columns)
        columns.update({name: str(fields[name]) for name in required})
        columns["outcome"] = str(fields["outcome"]) if fields.get("outcome") else ""
        preparation = dataclasses.replace(self.base_config.preparation, columns=columns)
        checkpoints = {kind: (draft.model_dir / model_files[kind]).resolve() for kind in model_files}
        models = tuple(dataclasses.replace(spec, checkpoint=checkpoints[spec.kind])
                       for spec in self.base_config.models if spec.kind in checkpoints)
        output = self.runtime_dir
        config = dataclasses.replace(
            self.base_config.with_models(models),
            bundle_path=output / "demo.json",
            analytics_dir=output,
            preparation=preparation,
            labels_available=bool(fields.get("outcome")),
        )

        try:
            config, notes = complete_config(config, draft.log)
        except ValueError as exc:
            raise SetupError(str(exc)) from exc

        from scripts.prepare_demo_data import prepare_bundle

        prepare_bundle(draft.log, draft.image_dir, output, config)
        output.mkdir(parents=True, exist_ok=True)
        (output / SESSION_CONFIG).write_text(
            dump_config(config, header="Completed by the setup from the event log; edit it and load it with EXPLORER_CONFIG."),
            encoding="utf-8",
        )
        self._save(config)
        self.notes = notes
        service = DemoService.from_config(config)
        with self._lock:
            self.configured = True
            self._drafts.clear()
        self._prune_uploads(keep=token)
        return service

    def _prune_uploads(self, keep: str) -> None:
        """Delete the uploads of earlier setups: each holds a copy of the checkpoints (GBs)."""
        uploads = self.runtime_dir / "uploads"
        if not uploads.is_dir():
            return
        with self._lock:
            in_progress = set(self._uploads)  # folder uploads not yet parsed
        for entry in uploads.iterdir():
            token = entry.name.split(".")[0]  # a folder <token> or a file <token>.xes
            if token == keep or token in in_progress:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
