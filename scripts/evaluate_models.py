#!/usr/bin/env python3
"""Evaluate the three models of a configuration on a test log, at every image cutoff.

At the image events and at the end of each case, every model predicts from what the explorer would give it (the history
up to that point and the latest image); the predictions are compared with the case outcome. Reports, per model
for the first image of each case, the later ones and the end of the pathway: accuracy, ROC AUC of the risk class
and macro F1, next to the baseline that always predicts the most frequent outcome of the training log.

    python scripts/evaluate_models.py --config config/xray_pathways.toml \\
        --log data/xray-pathways/pathways_test.xes --images data/xray-pathways/images \\
        --train-log data/xray-pathways/pathways_full.xes
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.inference import complete_config  # noqa: E402
from app.models import build_models  # noqa: E402
from app.setup import read_xes  # noqa: E402
from scripts.train_models import history, load_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True, help="Test log, not used for training")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True, help="Training log, for the majority baseline")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def auc(scores: list[float], positives: list[bool]) -> float:
    """ROC AUC as the chance that a positive scores above a negative (ties count half)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def class_f1(predicted: list[int], actual: list[int], c: int) -> float:
    tp = sum(p == c and a == c for p, a in zip(predicted, actual))
    fp = sum(p == c and a != c for p, a in zip(predicted, actual))
    fn = sum(p != c and a == c for p, a in zip(predicted, actual))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def macro_f1(predicted: list[int], actual: list[int], classes: int) -> float:
    return sum(class_f1(predicted, actual, c) for c in range(classes)) / classes


def main() -> None:
    args = parse_args()
    train_log = read_xes(args.train_log)
    config, _ = complete_config(load_config(args.config), train_log)
    classes = len(config.outcome_classes)
    risk = config.class_index(config.risk_class)
    train_labels = Counter(case["label"] for case in load_cases(config, train_log, args.images))
    majority = train_labels.most_common(1)[0][0]

    cases = load_cases(config, read_xes(args.log), args.images)
    models = build_models(config, args.device)
    rows = []  # (group, model id or "majority", predicted class, risk probability, label)
    for case in cases:
        events = case["events"]
        images = [index for index, event in enumerate(events) if event["image"]]
        if not images:
            continue
        # The first image, every later one, and the end of the pathway (everything known, latest image)
        cutoffs = [("first image", images[0])] + [("later images", index) for index in images[1:]]
        cutoffs.append(("end of pathway", len(events) - 1))
        for group, index in cutoffs:
            image = events[max(i for i in images if i <= index)]["image"]
            text = history(events[:index + 1])
            for model_id, model in models.items():
                probabilities = model.predict(text, image)
                predicted = max(range(classes), key=probabilities.__getitem__)
                rows.append((group, model_id, predicted, probabilities[risk], case["label"]))
            rows.append((group, "majority", majority, float(majority == risk), case["label"]))

    labels = Counter(case["label"] for case in cases)
    print(f"{len(cases)} test cases: " + ", ".join(f"{config.outcome_classes[c].code} {n}" for c, n in sorted(labels.items())))
    print(f"majority baseline: always {config.outcome_classes[majority].code} (most frequent in the training log)")
    # "every image" pools the first and later images: one prediction per image, as in the MEDUSA evaluation
    groups = {"first image": ("first image",), "later images": ("later images",),
              "every image": ("first image", "later images"), "end of pathway": ("end of pathway",)}
    codes = [outcome.code for outcome in config.outcome_classes]
    for group, members in groups.items():
        selected = [row for row in rows if row[0] in members]
        cutoffs = sum(row[1] == "majority" for row in selected)
        print(f"\n== {group}: {cutoffs} cutoffs")
        print(f"{'model':<12}{'accuracy':>10}{'ROC AUC':>10}{'macro F1':>10}" + "".join(f"{'F1 ' + code:>16}" for code in codes))
        for model_id in [*models, "majority"]:
            mine = [row for row in selected if row[1] == model_id]
            predicted, scores, actual = [r[2] for r in mine], [r[3] for r in mine], [r[4] for r in mine]
            accuracy = sum(p == a for p, a in zip(predicted, actual)) / len(mine)
            area = auc(scores, [a == risk for a in actual])
            per_class = [class_f1(predicted, actual, c) for c in range(classes)]
            print(f"{model_id:<12}{accuracy:>10.3f}{area:>10.3f}{macro_f1(predicted, actual, classes):>10.3f}"
                  + "".join(f"{value:>16.3f}" for value in per_class))


if __name__ == "__main__":
    main()
