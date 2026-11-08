from __future__ import annotations

import json
from pathlib import Path
from threading import RLock


class DemoDataError(RuntimeError):
    """Raised when the curated demo bundle is missing or malformed."""


class PatientStore:
    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self._lock = RLock()
        self._mtime_ns: int | None = None
        self._patients: dict[str, dict] = {}
        self._metadata: dict = {}

    @property
    def ready(self) -> bool:
        try:
            self._reload_if_needed()
        except DemoDataError:
            return False
        return bool(self._patients)

    @property
    def metadata(self) -> dict:
        self._reload_if_needed()
        return dict(self._metadata)

    def list_patients(self) -> list[dict]:
        self._reload_if_needed()
        return [
            {
                "id": patient["id"],
                "event_count": len(patient["events"]),
                "prediction_stage_count": sum(
                    bool(event.get("prediction_stage")) for event in patient["events"]
                ),
                "duration_seconds": patient.get("duration_seconds", 0),
            }
            for patient in self._patients.values()
        ]

    def get_patient(self, patient_id: str) -> dict:
        self._reload_if_needed()
        try:
            return self._patients[patient_id]
        except KeyError as exc:
            raise KeyError(f"Unknown patient: {patient_id}") from exc

    def image_path(self, patient_id: str, event_id: str) -> Path:
        patient = self.get_patient(patient_id)
        event = next((item for item in patient["events"] if item["id"] == event_id), None)
        if event is None or not event.get("image"):
            raise KeyError("No image exists for this patient event")

        root = self.bundle_path.parent.resolve()
        candidate = (root / event["image"]).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise DemoDataError("Curated image path is invalid")
        return candidate

    def _reload_if_needed(self) -> None:
        with self._lock:
            if not self.bundle_path.is_file():
                raise DemoDataError(
                    f"Runtime bundle not found at {self.bundle_path}. Complete the explorer setup first."
                )

            stat = self.bundle_path.stat()
            if stat.st_mtime_ns == self._mtime_ns:
                return

            try:
                payload = json.loads(self.bundle_path.read_text(encoding="utf-8"))
                patients = payload["patients"]
                if not isinstance(patients, list) or not patients:
                    raise ValueError("patients must be a non-empty list")
                indexed = {patient["id"]: patient for patient in patients}
                if len(indexed) != len(patients):
                    raise ValueError("patient aliases must be unique")
                for patient in patients:
                    if not patient.get("events"):
                        raise ValueError(f"{patient['id']} has no events")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DemoDataError(f"Invalid curated bundle: {exc}") from exc

            self._patients = indexed
            self._metadata = payload.get("metadata", {})
            self._mtime_ns = stat.st_mtime_ns

