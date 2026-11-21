"""Clinical-history classifier (RoBERTa in the paper)."""
from __future__ import annotations

from pathlib import Path

from .base import Classifier, ModelUnavailable

MAX_TOKENS = 512


class TextClassifier(Classifier):
    def _build(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailable(f"transformers is unavailable: {exc}") from exc
        # Left truncation keeps the most recent events, as in MEDUSA.
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.base, truncation_side="left", **self.hub_options())
        return AutoModelForSequenceClassification.from_pretrained(
            self.spec.base, num_labels=self.num_classes, **self.hub_options()
        )

    def predict(self, history: str, image_path: Path | None = None) -> list[float]:
        self.ensure_loaded()
        inputs = self.tokenizer(
            history, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        def run():
            with self._lock, self.torch.inference_mode():
                return self._probabilities(self._model(**inputs).logits)
        return self.timed_inference(run)

    # For explanation methods given as a file (app/explainers.py), as in the multimodal adapter

    window_tokens = MAX_TOKENS - 2

    def count_tokens(self, text: str) -> int:
        self.ensure_loaded()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def context_spans(self, history: str) -> list[tuple[int, int]]:
        self.ensure_loaded()
        offsets = self.tokenizer(history, truncation=True, max_length=MAX_TOKENS, return_offsets_mapping=True)["offset_mapping"]
        return [(start, end) for start, end in offsets if end > start]
