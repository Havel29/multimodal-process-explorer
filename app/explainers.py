"""Explanation methods, chosen with [explanation] method in the configuration.

A method is either the name of a built-in one or the path of a Python file (relative to the
repository) that defines

    def explain(model, history, image_path, target=None, with_image=True, **options) -> Attributions | dict

``model`` is the configured classifier (app/models; its ``predict``, ``torch`` and loaded network
are there to use), ``history`` the text the models read and ``image_path`` the image, or None.
``target`` is the class to explain (None: the predicted one); ``with_image`` False means only the
text is needed. ``options`` are the keys of [explanation.options]. The result gives the class
probabilities, the explained class, the scores of character spans of ``history`` as
(start, end, score) and a square grid of image scores (empty without an image), positive toward the
explained class. The explorer sums the span scores per word and per event and draws the grid over
the image; see explainers/example_occlusion.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from .config import PROJECT_ROOT
from .models import Attributions, ModelUnavailable

Explainer = Callable[..., Attributions]


def integrated_gradients(model, history: str, image_path: Path | None, target: int | None = None,
                         with_image: bool = True, **options) -> Attributions:
    """Layer Integrated Gradients of the model adapter (the multimodal one, as in MEDUSA)."""
    if not hasattr(model, "explain"):
        raise ModelUnavailable(f"{model.spec.name} has no Integrated Gradients; set [explanation] method to a file")
    return model.explain(history, image_path, target=target, with_image=with_image, **options)


BUILT_IN: dict[str, Explainer] = {"integrated_gradients": integrated_gradients}


def load_explainer(method: str) -> Explainer:
    if method in BUILT_IN:
        return BUILT_IN[method]
    path = Path(method)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.is_file():
        raise ValueError(f"Explanation method {method!r} is neither built in ({', '.join(BUILT_IN)}) nor a file")
    spec = importlib.util.spec_from_file_location(f"explainer_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "explain", None)):
        raise ValueError(f"{path} defines no explain() function")
    function = module.explain

    def explain(*args, **kwargs) -> Attributions:
        result = function(*args, **kwargs)
        return result if isinstance(result, Attributions) else Attributions(
            probabilities=list(result["probabilities"]), target=int(result["target"]),
            tokens=[tuple(token) for token in result.get("tokens", [])], patches=result.get("patches", []),
        )
    return explain
