#!/usr/bin/env python3
"""Prediction and explanation time by prefix length, on every multimodal case of the full log.

Every prefix of at least --min-prefix events is predicted by every model that can take it, as the
explorer does: the text model always, the image and multimodal models with the latest image once the
prefix has one. A random sample of the prefixes with an image is also explained with Integrated
Gradients, both on the model's context (the default) and on the whole history, window by window.
Times are end to end: composing and tokenising the history, reading the image and running the model,
and each of these steps is also written apart.

    python scripts/benchmark_scalability.py --log data/curated/covid_log.csv --images data/images
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attribution import event_attributions, history_windows  # noqa: E402
from app.config import load_config  # noqa: E402
from app.cutoffs import compose_history  # noqa: E402
from app.models import build_models  # noqa: E402
from scripts.prepare_demo_data import image_source, load_log, render_event  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--min-prefix", type=int, default=3)
    parser.add_argument("--explain-share", type=float, default=0.02,
                        help="Share of the prefixes with an image that are also explained")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=int, help="Only the first N multimodal cases (for a quick test)")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/scalability.csv"))
    parser.add_argument("--resume", action="store_true",
                        help="Keep the cases already complete in --output and go on from the next one "
                             "(the explanation sample then differs from an uninterrupted run)")
    parser.add_argument("--case-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Only these cases (numbered from 1), e.g. to time again those measured under load")
    return parser.parse_args()


def completed_rows(path: Path) -> tuple[list[list[str]], int]:
    """Rows of the cases before the last one in a partial output (which may be cut short), and that case."""
    if not path.is_file():
        return [], 1
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    rows = [row for row in rows if len(row) == 9]
    last = int(rows[-1][0]) if rows else 1
    return [row for row in rows if int(row[0]) < last], last


def multimodal_cases(log, image_root: Path, config, limit: int | None = None) -> list[list[dict]]:
    """Every case with at least one image event whose image exists, as explorer events."""
    columns = config.preparation.columns
    stages = log[log[columns["activity"]].map(config.is_prediction_stage) & log[columns["image"]].notna()]
    log = log[log[columns["case"]].isin(stages[columns["case"]])]
    cases = []
    for _, group in log.groupby(columns["case"], sort=False):
        if limit is not None and len(cases) == limit:
            break
        group = group.sort_values(columns["timestamp"], kind="mergesort")
        start = group[columns["timestamp"]].iloc[0]
        events = []
        # Outcome-revealing events (reveals_outcome) are kept: the explorer never gives them to the models, but
        # for timing they are ordinary text (at most one more prefix per case, usually the last one).
        for index, (_, row) in enumerate(group.iterrows(), start=1):
            name = str(row[columns["activity"]])
            offset = max(0.0, (row[columns["timestamp"]] - start).total_seconds())
            image = image_source(row.get(columns["image"]), image_root) if config.is_prediction_stage(name) else None
            events.append({"id": f"E{index:03d}", "model_text": render_event(row, offset, config), "image": image})
        if any(event["image"] for event in events):
            cases.append(events)
    return cases


def timed(model, function):
    return clocked(model, function)[1]


def clocked(model, function):
    if model._device.type == "cuda":
        model.torch.cuda.synchronize(model._device)
    started = perf_counter()
    result = function()
    if model._device.type == "cuda":
        model.torch.cuda.synchronize(model._device)
    return result, (perf_counter() - started) * 1000


def predict_steps(model, history: str, image: Path | None) -> tuple[float, float, float]:
    """The steps of the adapter's predict(), timed apart: text, image and model milliseconds."""
    torch, device, kind = model.torch, model._device, model.spec.kind
    text_ms = image_ms = 0.0
    if model.spec.adapter:  # a model from an adapter file: only its whole predict() can be timed
        return text_ms, image_ms, timed(model, lambda: model.predict(history, image))
    if kind == "flava":
        token_ids, text_ms = clocked(model, lambda: model._token_ids(history)["input_ids"].to(device))
        pixels, image_ms = clocked(model, lambda: model._pixels(image))
        forward = lambda: model._logits(token_ids, pixels)  # noqa: E731
    elif kind == "text":
        inputs, text_ms = clocked(model, lambda: {key: value.to(device) for key, value in model.tokenizer(
            history, return_tensors="pt", padding="max_length", truncation=True, max_length=512).items()})
        forward = lambda: model._model(**inputs).logits  # noqa: E731
    else:
        inputs, image_ms = clocked(model, lambda: {key: value.to(device) for key, value in model.processor(
            images=model.load_image(image), return_tensors="pt").items()})
        forward = lambda: model._model(**inputs).logits  # noqa: E731
    def run():
        with model._lock, torch.inference_mode():
            return model._probabilities(forward())
    return text_ms, image_ms, timed(model, run)


def explain(model, available: list[dict], image: Path, scope: str, steps: int, batch: int) -> int:
    """The model calls of DemoService.explain; returns the number of windows."""
    history, spans = compose_history(available, {event["id"] for event in available})
    result = model.explain(history, image, steps=steps, internal_batch_size=batch)
    event_attributions(history, spans, [(start, end, 0.0) for start, end in model.context_spans(history)])
    tokens, windows = result.tokens, 1
    if scope == "history":
        parts = history_windows(spans, [model.count_tokens(history[start:end]) for _, start, end in spans],
                                model.window_tokens)
        tokens, windows = [], len(parts)
        for part in parts:
            offset = part[0][1]
            scores = model.explain(history[offset:part[-1][2]], image, steps=steps, internal_batch_size=batch,
                                   target=result.target, with_image=False)
            tokens += [(start + offset, end + offset, score) for start, end, score in scores.tokens]
    event_attributions(history, spans, tokens)
    return windows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = build_models(config, args.device)
    multimodal = models[config.explanation.model]
    for model in models.values():
        model.ensure_loaded()
    steps, batch = config.explanation.steps, config.explanation.internal_batch_size

    cases = multimodal_cases(load_log(args.log, config), args.images, config, args.cases)
    print(f"{len(cases)} multimodal cases, {sum(map(len, cases))} events, device {multimodal.device_name}", flush=True)

    history, _ = compose_history(cases[0], {event["id"] for event in cases[0]})
    image = next(event["image"] for event in cases[0] if event["image"])
    for _ in range(args.warmup):
        for model in models.values():
            predict_steps(model, history, image)
    if args.explain_share:
        explain(multimodal, cases[0], image, "context", steps, batch)

    rng = random.Random(args.seed)
    kept, first = completed_rows(args.output) if args.resume else ([], 1)
    last = len(cases)
    if args.case_range:
        first, last = args.case_range
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "prefix_length", "task", "model", "windows", "ms", "text_ms", "image_ms", "model_ms"])
        writer.writerows(kept)
        if kept:
            print(f"Resuming from case {first}", flush=True)
        for number, events in enumerate(cases[first - 1:last], start=first):
            for length in range(args.min_prefix, len(events) + 1):
                available = events[:length]
                image = next((event["image"] for event in reversed(available) if event["image"]), None)
                for model in models.values():
                    if model.spec.uses_image and image is None:
                        continue
                    (history, _), compose_ms = clocked(model, lambda: compose_history(
                        available, {event["id"] for event in available})) if model.spec.uses_text else (("", []), 0.0)
                    text_ms, image_ms, model_ms = predict_steps(model, history, image)
                    text_ms += compose_ms
                    writer.writerow([number, length, "predict", model.spec.id, 1, round(text_ms + image_ms + model_ms, 3),
                                     round(text_ms, 3), round(image_ms, 3), round(model_ms, 3)])
                if image and rng.random() < args.explain_share:
                    for scope in ("context", "history"):
                        windows = 0
                        def run():
                            nonlocal windows
                            windows = explain(multimodal, available, image, scope, steps, batch)
                        ms = timed(multimodal, run)
                        writer.writerow([number, length, f"explain_{scope}", multimodal.spec.id, windows, round(ms, 3), "", "", ""])
            if number % 50 == 0 or number == len(cases):
                handle.flush()
                print(f"{number}/{len(cases)} cases", flush=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
