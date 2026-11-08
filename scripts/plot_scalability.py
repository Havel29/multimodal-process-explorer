#!/usr/bin/env python3
"""Plot the output of benchmark_scalability.py: prediction and explanation time by prefix length.

    python scripts/plot_scalability.py data/runtime/scalability.csv data/runtime/scalability.png
    python scripts/plot_scalability.py predictions.csv out.png --explanations explanations.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Model colours of the explorer (config/cdsl_covid.toml); IG scopes take the first categorical slots
MODELS = {
    "roberta": ("RoBERTa (history)", "#eda100"),
    "vit": ("ViT (radiograph)", "#1baf7a"),
    "flava": ("FLAVA (history + radiograph)", "#6250d6"),
}
SCOPES = {
    "explain_context": ("Model context (512 tokens)", "#2a78d6"),
    "explain_history": ("Whole history", "#eb6834"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--explanations", type=Path, help="Results with the explanation times (default: results)")
    parser.add_argument("--min-cases", type=int, default=20,
                        help="Prediction lengths reached by fewer cases are left out")
    parser.add_argument("--bin", type=int, default=25, help="Prefix-length bin of the explanation means")
    parser.add_argument("--min-explained", type=int, default=5, help="Explanation bins with fewer samples show no mean")
    return parser.parse_args()


def style(axis, title: str, log: bool = False) -> None:
    axis.set_title(title, loc="left", fontsize=9, color=INK)
    axis.set_xlabel("Prefix length (events)", fontsize=8, color=MUTED)
    axis.set_ylabel("Time (ms)", fontsize=8, color=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7, colors=MUTED, length=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    if log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=1000)
        axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.results)
    explanations = pd.read_csv(args.explanations) if args.explanations else data
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    predictions = data[data.task == "predict"]
    for model, (label, color) in MODELS.items():
        by_length = predictions[predictions.model == model].groupby("prefix_length").ms.agg(["mean", "count"])
        by_length = by_length[by_length["count"] >= args.min_cases]
        if by_length.empty:
            continue
        left.plot(by_length.index, by_length["mean"], color=color, linewidth=1.5, label=label)
    style(left, "Prediction")
    left.legend(fontsize=7, frameon=False, loc="upper left")

    explanations = explanations[explanations.task.str.startswith("explain")]
    for task, (label, color) in SCOPES.items():
        rows = explanations[explanations.task == task]
        right.scatter(rows.prefix_length, rows.ms, s=6, color=color, alpha=0.25, linewidths=0)
        binned = rows.groupby(rows.prefix_length // args.bin * args.bin + args.bin / 2).ms.agg(["mean", "count"])
        binned = binned[binned["count"] >= args.min_explained]["mean"]
        right.plot(binned.index, binned.values, color=color, linewidth=1.5, label=label)
    style(right, "Integrated Gradients (FLAVA)", log=True)
    right.legend(fontsize=7, frameon=False, loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
