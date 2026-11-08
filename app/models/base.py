"""Shared machinery for the lazily loaded classifiers."""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter

from ..config import ModelSpec


class ModelUnavailable(RuntimeError):
    """The model cannot run: missing checkpoint, dependency, or weights."""


def cuda_available(torch) -> bool:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    return False


def select_device(torch, preference: str = "auto"):
    """Resolve auto/cpu/gpu, rejecting a requested but unusable GPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference not in {"auto", "gpu"}:
        raise ModelUnavailable(f"Unknown inference device: {preference}")
    if cuda_available(torch):
        return torch.device("cuda")
    if preference == "gpu":
        raise ModelUnavailable("GPU was requested, but compatible CUDA support is unavailable")
    return torch.device("cpu")


def load_state_dict(torch, checkpoint: Path) -> dict:
    """Read a checkpoint saved as a bare or wrapped (DataParallel) state dict, or as safetensors."""
    if checkpoint.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(checkpoint), device="cpu")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    if not isinstance(payload, dict):
        raise TypeError("expected a PyTorch state dictionary")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


class Classifier:
    """A classifier loaded from its checkpoint on first use."""

    def __init__(self, spec: ModelSpec, num_classes: int, model_cache: Path, local_files_only: bool = False,
                 grayscale_images: bool = False, device_preference: str = "auto"):
        self.spec = spec
        self.num_classes = num_classes
        self.model_cache = model_cache
        self.local_files_only = local_files_only
        self.grayscale_images = grayscale_images
        self.device_preference = device_preference
        self._lock = Lock()
        self._model = None
        self._device = None
        self.torch = None
        self.last_load_ms: float | None = None
        self.last_inference_ms: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str | None:
        return str(self._device) if self._device is not None else None

    def hub_options(self) -> dict:
        # cache_dir expects HF_HOME/hub.
        return {"cache_dir": str(self.model_cache / "hub"), "local_files_only": self.local_files_only}

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            if not self.spec.checkpoint.is_file():
                raise ModelUnavailable(f"Checkpoint not found: {self.spec.checkpoint}")
            try:
                import torch
            except ImportError as exc:
                raise ModelUnavailable(f"PyTorch is unavailable: {exc}") from exc
            self.torch = torch
            started = perf_counter()
            try:
                model = self._build()
                model.load_state_dict(self._prepare_weights(load_state_dict(torch, self.spec.checkpoint)), strict=True)
                device = select_device(torch, self.device_preference)
                model.to(device).eval()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable(f"Unable to initialize {self.spec.name}: {exc}") from exc
            self._device = device
            self._model = model
            self.last_load_ms = (perf_counter() - started) * 1000

    def timed_inference(self, function):
        """Measure one inference accurately, synchronising asynchronous CUDA work."""
        if self._device.type == "cuda":
            self.torch.cuda.synchronize(self._device)
        started = perf_counter()
        result = function()
        if self._device.type == "cuda":
            self.torch.cuda.synchronize(self._device)
        self.last_inference_ms = (perf_counter() - started) * 1000
        return result

    def _prepare_weights(self, weights: dict) -> dict:
        return weights

    def _build(self):
        raise NotImplementedError

    def load_image(self, path: Path):
        """Three-channel image; grayscale first when the dataset is grayscale."""
        from PIL import Image

        image = Image.open(path)
        return (image.convert("L") if self.grayscale_images else image).convert("RGB")

    def _probabilities(self, logits) -> list[float]:
        return [float(value) for value in self.torch.softmax(logits, dim=-1)[0].detach().cpu()]

    def predict(self, history: str, image_path: Path | None) -> list[float]:
        raise NotImplementedError
