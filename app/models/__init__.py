"""Classifier adapters: how each configured model is built, fed and run.

Every [[models]] entry has a ``kind``, its role: ``text`` (reads the history), ``image`` (reads the
latest image) or ``flava`` (multimodal, reads both). By default the role picks a built-in adapter
(RoBERTa, ViT or FLAVA through Hugging Face). To use another PyTorch model, set ``adapter`` to the
path of a Python file (relative to the repository) that defines a class ``Model``, a subclass of
``app.models.base.Classifier`` with two methods:

    def _build(self) -> torch.nn.Module
        # the network, untrained; the tool then loads the checkpoint into it (a state_dict, strict)
    def predict(self, history: str, image_path: Path | None) -> list[float]
        # class probabilities, in the order of [outcome] classes; call self.ensure_loaded() first

and, optionally, ``download(cls, spec, num_classes)``, called when the Docker image is built, to fetch
what ``_build`` needs while the explorer runs offline. See adapters/example_bag_of_words.py.
"""
from __future__ import annotations

import importlib.util
import os

from ..config import DemoConfig, ModelSpec, _path
from .base import Classifier, ModelUnavailable
from .flava import Attributions, FlavaClassifier
from .image import ImageClassifier
from .text import TextClassifier

ADAPTERS: dict[str, type[Classifier]] = {
    "text": TextClassifier,
    "image": ImageClassifier,
    "flava": FlavaClassifier,
}


def adapter_class(spec: ModelSpec) -> type[Classifier]:
    """The class that runs a configured model: from its adapter file, or the built-in one for its role."""
    if not spec.adapter:
        return ADAPTERS[spec.kind]
    path = _path(spec.adapter)
    module_spec = importlib.util.spec_from_file_location(f"adapter_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = getattr(module, "Model", None)
    if not (isinstance(model, type) and issubclass(model, Classifier)):
        raise ValueError(f"{path} defines no class Model(Classifier)")
    return model


def build_models(config: DemoConfig, device_preference: str = "auto") -> dict[str, Classifier]:
    local_only = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
    return {
        spec.id: adapter_class(spec)(
            spec, len(config.outcome_classes), config.model_cache, local_only, config.grayscale_images,
            device_preference=device_preference,
        )
        for spec in config.models
    }


__all__ = ["ADAPTERS", "Attributions", "Classifier", "ModelUnavailable", "adapter_class", "build_models"]
