"""Example adapter: a text model that is not from Hugging Face, written as plain PyTorch.

Why it is here: no configuration uses it and no checkpoint comes with it. It is a template showing
the minimum a file needs to plug your own model into the explorer, a class Model with _build() and
predict(). A bag of words was picked because it is tiny and downloads nothing, so those two methods
are all there is to read. To use your model, copy this file and replace the network in _network()
and the input preparation in predict(); app/models/__init__.py documents the interface.

The model reads the event history as a bag of words (each word hashed into one of 2**16 buckets), averages
their embeddings and classifies the average. Use it in a configuration as

    [[models]]
    kind = "text"                                  # its role: it reads the history only
    adapter = "adapters/example_bag_of_words.py"
    checkpoint = "models/my-data/bag_of_words.pth"

The checkpoint is the state_dict of the network that _build returns, trained on your data, e.g.

    model = Model(spec, num_classes, model_cache)._network()   # or your own training code
    ...                                                          # train it
    torch.save(model.state_dict(), "models/my-data/bag_of_words.pth")

with one output per outcome class, in the order of [outcome] in the configuration.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

from app.models.base import Classifier

BUCKETS = 2 ** 16
WORD = re.compile(r"\w+")


class Model(Classifier):
    def _network(self):
        import torch

        class BagOfWords(torch.nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.embedding = torch.nn.EmbeddingBag(BUCKETS, 64, mode="mean")
                self.output = torch.nn.Linear(64, num_classes)

            def forward(self, word_ids):
                return self.output(self.embedding(word_ids))

        return BagOfWords(self.num_classes)

    def _build(self):
        # Called once, on the first prediction; the tool then loads the checkpoint into the result
        return self._network()

    def predict(self, history: str, image_path: Path | None = None) -> list[float]:
        self.ensure_loaded()
        words = [zlib.crc32(word.lower().encode()) % BUCKETS for word in WORD.findall(history)] or [0]
        word_ids = self.torch.tensor([words], device=self._device)

        def run():
            with self._lock, self.torch.inference_mode():
                return self._probabilities(self._model(word_ids))
        return self.timed_inference(run)
