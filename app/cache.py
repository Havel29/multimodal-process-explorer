"""Per-case analysis results persisted next to the runtime bundle."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Callable

# Bump to discard files in an older format.
FORMAT_VERSION = 4  # 3: events that reveal the outcome are no longer given to the models; 4: predicted class per point


class AnalyticsCache:
    """Results per case, discarded when the bundle changes."""

    def __init__(self, path: Path, bundle_path: Path):
        self.path = path
        self.bundle_path = bundle_path
        self._lock = Lock()
        self._compute_locks: dict[str, Lock] = {}
        # Bumped by clear(), so results computed before a reset are not saved after it.
        self._generation = 0

    def _signature(self) -> str:
        stat = self.bundle_path.stat()
        return f"v{FORMAT_VERSION}:{stat.st_size}:{stat.st_mtime_ns}"

    def _read(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload.get("patients", {}) if payload.get("signature") == self._signature() else {}

    def get(self, patient_id: str):
        with self._lock:
            return self._read().get(patient_id)

    def all(self) -> dict:
        with self._lock:
            return self._read()

    def put(self, patient_id: str, value) -> None:
        with self._lock:
            self._write(patient_id, value)

    def _write(self, patient_id: str, value) -> None:
        """Store one case; the caller holds the lock."""
        patients = self._read()
        patients[patient_id] = value
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"signature": self._signature(), "patients": patients}), encoding="utf-8")
        temporary.replace(self.path)

    def clear(self) -> None:
        """Forget every case; results are computed again on request."""
        with self._lock:
            self._generation += 1
            self.path.unlink(missing_ok=True)

    def get_or_compute(self, patient_id: str, compute: Callable[[], object]):
        """Compute once, even with concurrent requests."""
        cached = self.get(patient_id)
        if cached is not None:
            return cached
        with self._lock:
            compute_lock = self._compute_locks.setdefault(patient_id, Lock())
        with compute_lock:
            cached = self.get(patient_id)
            if cached is None:
                with self._lock:
                    generation = self._generation
                cached = compute()
                with self._lock:
                    if generation == self._generation:
                        self._write(patient_id, cached)
            return cached
