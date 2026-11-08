#!/usr/bin/env python3
"""Fine-tune the text, image and multimodal models of a configuration on an event log.

The explorer needs trained checkpoints; this script makes them for a new log, with the texts made
exactly as the explorer makes them (the configuration, completed from the log as the setup does)
and the architectures of the adapters in app/models, so that the checkpoints load as they are.

  - text model: prefixes of every case (the history up to an event)
  - image model: every image event's image
  - multimodal model: the history up to every image event, and up to a few later events, with the latest image

Labels are the case outcomes, in the order of [outcome] in the configuration. The cases are split
into training and validation; the checkpoint of the epoch with the best validation accuracy is kept
and written to the [[models]] checkpoint path.

    python scripts/train_models.py --config config/xray_pathways.toml \\
        --log data/xray-pathways/pathways_full.xes --images data/xray-pathways/images
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.cutoffs import compose_history  # noqa: E402
from app.inference import complete_config  # noqa: E402
from app.models import build_models  # noqa: E402
from app.setup import read_xes  # noqa: E402
from scripts.prepare_demo_data import image_source, render_event  # noqa: E402

LEARNING_RATES = {"text": 1e-5, "image": 5e-5, "flava": 1e-5}  # flava as in MEDUSA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True, help="XES event log with the outcome label")
    parser.add_argument("--images", type=Path, required=True, help="Root of the image paths in the log")
    parser.add_argument("--kinds", nargs="+", default=["text", "image", "flava"], choices=["text", "image", "flava"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--validation", type=float, default=0.1, help="Share of the cases kept for validation")
    parser.add_argument("--text-prefixes", type=int, default=8, help="Text prefixes sampled per case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, help="Only the first N cases (for a quick test)")
    parser.add_argument("--resume", action="store_true", help="Continue from the checkpoints already written")
    parser.add_argument("--lr", type=float, help="Learning rate for every kind (default: per kind, see LEARNING_RATES)")
    return parser.parse_args()


def load_cases(config, log: pd.DataFrame, image_root: Path) -> list[dict]:
    columns = config.preparation.columns
    log[columns["timestamp"]] = pd.to_datetime(log[columns["timestamp"]], utc=True, errors="coerce")
    cases = []
    for case_id, group in log.groupby(columns["case"], sort=False):
        group = group.sort_values(columns["timestamp"], kind="mergesort")
        start = group[columns["timestamp"]].iloc[0]
        events = []
        for index, (_, row) in enumerate(group.iterrows(), start=1):
            if config.reveals_outcome(str(row[columns["activity"]])):
                continue  # as in the explorer, the models never get the events that state the outcome
            offset = max(0.0, (row[columns["timestamp"]] - start).total_seconds())
            stage = config.is_prediction_stage(str(row[columns["activity"]]))
            events.append({"id": f"E{index:03d}", "model_text": render_event(row, offset, config),
                           "image": image_source(row.get(columns["image"]), image_root) if stage else None})
        outcome = config.outcome_for_source(str(group[columns["outcome"]].dropna().iloc[0]))
        cases.append({"id": case_id, "events": events, "label": config.class_index(outcome.code)})
    return cases


def history(events: list[dict]) -> str:
    return compose_history(events, {event["id"] for event in events})[0]


def samples(cases: list[dict], kind: str, text_prefixes: int, rng: random.Random, flava_extra: int = 2) -> list[tuple]:
    """(history, image, label) triples for one model kind."""
    items = []
    for case in cases:
        events, label = case["events"], case["label"]
        stages = [index for index, event in enumerate(events) if event["image"]]
        if kind == "image":
            items += [(None, events[index]["image"], label) for index in stages]
        elif kind == "flava":
            # Every image event, plus a few later cutoffs with the latest image: the explorer also asks the
            # model at the end of each day, when the history has grown since the last image
            later = [index for index in range(stages[0] + 1, len(events)) if index not in stages] if stages else []
            ends = stages + rng.sample(later, min(flava_extra, len(later)))
            for index in sorted(ends):
                image = next(events[i]["image"] for i in range(index, -1, -1) if events[i]["image"])
                items.append((history(events[:index + 1]), image, label))
        else:
            ends = sorted(rng.sample(range(len(events)), min(text_prefixes, len(events))))
            items += [(history(events[:index + 1]), None, label) for index in ends]
    return items


class Trainer:
    """Builds a model kind as its adapter does and prepares its inputs the same way."""

    def __init__(self, adapter, num_classes: int, torch, device):
        self.adapter, self.torch, self.device = adapter, torch, device
        adapter.torch = torch
        if adapter.spec.kind == "flava":
            from torchmultimodal.models.flava.model import flava_model_for_classification

            adapter._build()  # sets the processor
            self.model = flava_model_for_classification(num_classes=num_classes, pretrained=True)
        else:
            self.model = adapter._build()
        self.model.to(device)

    def logits_and_loss(self, batch: list[tuple]):
        torch, adapter, kind = self.torch, self.adapter, self.adapter.spec.kind
        labels = torch.tensor([label for _, _, label in batch], device=self.device)
        if kind == "text":
            inputs = adapter.tokenizer([text for text, _, _ in batch], return_tensors="pt", padding="max_length",
                                       truncation=True, max_length=512).to(self.device)
            logits = self.model(**inputs).logits
        elif kind == "image":
            inputs = adapter.processor(images=[adapter.load_image(image) for _, image, _ in batch],
                                       return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
        else:
            ids = adapter._token_ids([text for text, _, _ in batch])["input_ids"].to(self.device)
            pixels = adapter.processor.image_processor([adapter.load_image(image) for _, image, _ in batch],
                                                       return_tensors="pt")["pixel_values"].to(self.device)
            logits = self.model(text=ids, image=pixels, labels=labels).logits
        return logits, torch.nn.functional.cross_entropy(logits.float(), labels)

    def accuracy(self, items: list[tuple], batch_size: int) -> float:
        torch = self.torch
        self.model.eval()
        right = 0
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                logits, _ = self.logits_and_loss(batch)
                right += sum(int(p == label) for p, (_, _, label) in zip(logits.argmax(-1).tolist(), batch))
        return right / max(1, len(items))


def main() -> None:
    args = parse_args()
    import torch

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    log = read_xes(args.log)
    config, notes = complete_config(load_config(args.config), log)
    for note in notes:
        print("note:", note)
    cases = load_cases(config, log, args.images)[:args.cases]
    rng.shuffle(cases)
    held = int(len(cases) * args.validation)
    train_cases, validation_cases = cases[held:], cases[:held]
    print(f"{len(train_cases)} training and {len(validation_cases)} validation cases")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapters = {adapter.spec.kind: adapter for adapter in build_models(config, "cpu").values()}

    for kind in args.kinds:
        adapter = adapters[kind]
        if adapter.spec.adapter:
            print(f"== {adapter.spec.name}: comes from {adapter.spec.adapter}; train it with your own code", flush=True)
            continue
        train = samples(train_cases, kind, args.text_prefixes, rng)
        validation = samples(validation_cases, kind, args.text_prefixes, rng)
        trainer = Trainer(adapter, len(config.outcome_classes), torch, device)
        best = -1.0
        if args.resume and adapter.spec.checkpoint.is_file():
            # Go on from the saved checkpoint; it is only replaced by a better epoch
            trainer.model.load_state_dict(torch.load(adapter.spec.checkpoint, map_location=device, weights_only=True))
            best = trainer.accuracy(validation, args.batch * 2)
            print(f"resuming {adapter.spec.checkpoint.name}: validation accuracy {best:.3f}", flush=True)
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr or LEARNING_RATES[kind])
        print(f"== {adapter.spec.name}: {len(train)} training and {len(validation)} validation samples", flush=True)
        for epoch in range(1, args.epochs + 1):
            trainer.model.train()
            rng.shuffle(train)
            started = perf_counter()
            for start in range(0, len(train), args.batch):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = trainer.logits_and_loss(train[start:start + args.batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            accuracy = trainer.accuracy(validation, args.batch * 2)
            print(f"epoch {epoch}: validation accuracy {accuracy:.3f} ({perf_counter() - started:.0f} s)", flush=True)
            if accuracy > best:
                best = accuracy
                adapter.spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(trainer.model.state_dict(), adapter.spec.checkpoint)
        print(f"best validation accuracy {best:.3f}, saved to {adapter.spec.checkpoint}", flush=True)
        del trainer, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
