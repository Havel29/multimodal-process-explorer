"""Image-only classifier (ViT on radiographs in the paper)."""
from __future__ import annotations

from pathlib import Path

from .base import Classifier, ModelUnavailable


class ImageClassifier(Classifier):
    def _build(self):
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ModelUnavailable(f"transformers is unavailable: {exc}") from exc
        self.processor = AutoImageProcessor.from_pretrained(self.spec.base, **self.hub_options())
        return AutoModelForImageClassification.from_pretrained(
            self.spec.base, num_labels=self.num_classes, ignore_mismatched_sizes=True, **self.hub_options()
        )

    def predict(self, history: str, image_path: Path | None) -> list[float]:
        if image_path is None:
            raise ModelUnavailable(f"{self.spec.name} needs an image")
        self.ensure_loaded()
        inputs = self.processor(images=self.load_image(image_path), return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        def run():
            with self._lock, self.torch.inference_mode():
                return self._probabilities(self._model(**inputs).logits)
        return self.timed_inference(run)
