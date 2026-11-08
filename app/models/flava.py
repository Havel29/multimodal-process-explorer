"""Multimodal classifier over the history and an image (FLAVA in the paper)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .base import Classifier, ModelUnavailable

MAX_TOKENS = 512
IMAGE_SIZE = 224


@dataclass
class Attributions:
    """Integrated Gradients scores for the tokens and the image patches."""

    probabilities: list[float]
    target: int
    tokens: list[tuple[int, int, float]]
    patches: list[list[float]]


class FlavaClassifier(Classifier):
    # Room for text in one input, after [CLS] and [SEP].
    window_tokens = MAX_TOKENS - 2

    def _build(self):
        try:
            from torchmultimodal.models.flava.model import flava_model_for_classification
            from transformers import FlavaProcessor
        except ImportError as exc:
            raise ModelUnavailable(f"FLAVA dependencies are unavailable: {exc}") from exc
        self.processor = FlavaProcessor.from_pretrained(self.spec.base, truncation_side="left", **self.hub_options())
        return flava_model_for_classification(num_classes=self.num_classes, pretrained=False)

    def _prepare_weights(self, weights: dict) -> dict:
        # Only used in training.
        weights.pop("loss.weight", None)
        return weights

    def _token_ids(self, history: str, **options):
        return self.processor.tokenizer(
            history, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS, **options
        )

    def _pixels(self, image_path: Path | None):
        if image_path is None:
            # Zero image: the authors' IG baseline, also used to remove the image.
            return self.torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self._device)
        pixels = self.processor.image_processor(self.load_image(image_path), return_tensors="pt")["pixel_values"]
        return pixels.to(self._device)

    def _logits(self, token_ids, pixels):
        # The wrapper always computes a loss, so it needs some labels.
        labels = self.torch.zeros(token_ids.shape[0], dtype=self.torch.long, device=self._device)
        return self._model(text=token_ids, image=pixels, labels=labels).logits

    def predict(self, history: str, image_path: Path | None) -> list[float]:
        """Class probabilities; ``image_path=None`` blanks the image."""
        self.ensure_loaded()
        token_ids = self._token_ids(history)["input_ids"].to(self._device)
        pixels = self._pixels(image_path)
        def run():
            with self._lock, self.torch.inference_mode():
                return self._probabilities(self._logits(token_ids, pixels))
        return self.timed_inference(run)

    def count_tokens(self, text: str) -> int:
        """Tokens of a text, without special tokens."""
        self.ensure_loaded()
        return len(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def context_spans(self, history: str) -> list[tuple[int, int]]:
        """Character spans of the tokens kept in the model's context."""
        self.ensure_loaded()
        offsets = self._token_ids(history, return_offsets_mapping=True)["offset_mapping"][0].tolist()
        return [(start, end) for start, end in offsets if end > start]

    def explain(
        self,
        history: str,
        image_path: Path,
        steps: int = 20,
        internal_batch_size: int = 4,
        target: int | None = None,
        with_image: bool = True,
    ) -> Attributions:
        """Layer Integrated Gradients, as in the authors' xai/ig_flava.py.

        By default the predicted class is explained; ``target`` fixes it instead.
        """
        self.ensure_loaded()
        try:
            from captum.attr import LayerIntegratedGradients, TokenReferenceBase
        except ImportError as exc:
            raise ModelUnavailable(f"Explanations need captum: {exc}") from exc
        torch = self.torch

        encoded = self._token_ids(history, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        token_ids = encoded["input_ids"].to(self._device)
        pixels = self._pixels(image_path)
        encoders = self._model.model

        with self._lock:
            with torch.no_grad():
                probabilities = self._probabilities(self._logits(token_ids, pixels))
            if target is None:
                target = max(range(len(probabilities)), key=probabilities.__getitem__)

            reference = TokenReferenceBase(reference_token_idx=self.processor.tokenizer.pad_token_id)
            reference_ids = reference.generate_reference(token_ids.shape[1], device=self._device).unsqueeze(0)
            text_scores = LayerIntegratedGradients(self._logits, encoders.text_encoder.embeddings).attribute(
                token_ids, reference_ids, additional_forward_args=(pixels,),
                target=target, n_steps=steps, internal_batch_size=internal_batch_size,
            )
            patches = []
            if with_image:
                image_scores = LayerIntegratedGradients(
                    lambda image, ids: self._logits(ids, image), encoders.image_encoder.embeddings
                ).attribute(
                    pixels, torch.zeros_like(pixels), additional_forward_args=(token_ids,),
                    target=target, n_steps=steps, internal_batch_size=internal_batch_size,
                )
                patch_scores = normalised_scores(image_scores)[1:]  # drop the [CLS] position
                side = int(math.isqrt(len(patch_scores)))
                patches = [patch_scores[row * side:(row + 1) * side] for row in range(side)]

        token_scores = normalised_scores(text_scores)
        return Attributions(
            probabilities=probabilities,
            target=target,
            tokens=[(start, end, score) for (start, end), score in zip(offsets, token_scores) if end > start],
            patches=patches,
        )


def normalised_scores(attributions) -> list[float]:
    """Sum over the embedding dimension, then scale to unit norm (as in the authors' code)."""
    scores = attributions.sum(dim=2).squeeze(0)
    return (scores / scores.norm()).detach().cpu().tolist()
