#!/usr/bin/env python3
"""
Benchmark River streaming ML models on the Electricity dataset using
prequential (test-then-train) evaluation.

Protocol
--------
For each sample in arrival order:
  1. predict_one(x)   — TEST first
  2. metric.update()   — RECORD
  3. learn_one(x, y)   — TRAIN after

This is the standard interleaved test-then-train protocol for online learners.

Metrics
-------
* Accuracy   — cumulative over all samples
* Cohen's kappa — cumulative, same sample set
* Wall-clock time (ms)
* Throughput  — samples / wall-clock seconds
* Checkpoints every 5000 samples printed to stderr

# pip install river
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import river.forest
import river.metrics
import river.tree


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "date", "day", "period",
    "nswprice", "nswdemand",
    "vicprice", "vicdemand",
    "transfer",
]


def load_electricity(path: str | Path) -> list[tuple[dict[str, float], int]]:
    """Return list of (x_dict, y) from the Electricity CSV.

    Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    Label:    class  (UP -> 1, DOWN -> 0)
    """
    stream: list[tuple[dict[str, float], int]] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x = {name: float(row[name]) for name in FEATURE_NAMES}
            y = 1 if row["class"].strip() == "UP" else 0
            stream.append((x, y))

    return stream


# ---------------------------------------------------------------------------
# Prequential evaluation
# ---------------------------------------------------------------------------

def evaluate_prequential(
    model,
    stream: list[tuple[dict[str, float], int]],
    model_name: str,
    checkpoint_interval: int = 5000,
) -> dict:
    """Run prequential (test-then-train) evaluation and return metrics dict."""

    acc_metric = river.metrics.Accuracy()
    kappa_metric = river.metrics.CohenKappa()

    n_samples = len(stream)

    t0 = time.perf_counter()

    for i, (x, y) in enumerate(stream):
        # 1. TEST first
        y_pred = model.predict_one(x)

        # 2. RECORD
        # predict_one returns None before the model has seen any data;
        # treat None as class 0 (arbitrary but consistent)
        if y_pred is None:
            y_pred = 0
        acc_metric.update(y, y_pred)
        kappa_metric.update(y, y_pred)

        # 3. TRAIN after
        model.learn_one(x, y)

        # Checkpoint
        count = i + 1
        if count % checkpoint_interval == 0:
            elapsed = time.perf_counter() - t0
            throughput = count / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{model_name}] {count:>6d}/{n_samples}  "
                f"acc={acc_metric.get():.4f}  "
                f"kappa={kappa_metric.get():.4f}  "
                f"{throughput:.0f} s/s",
                file=sys.stderr,
            )

    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    return {
        "n_samples": n_samples,
        "accuracy": acc_metric.get(),
        "kappa": kappa_metric.get(),
        "time_ms": elapsed_s * 1000.0,
        "throughput": n_samples / elapsed_s if elapsed_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_models() -> list[tuple[str, object]]:
    """Return list of (label, model_instance)."""
    return [
        (
            "hoeffding_tree",
            river.tree.HoeffdingTreeClassifier(),
        ),
        (
            "hoeffding_adaptive_tree",
            river.tree.HoeffdingAdaptiveTreeClassifier(),
        ),
        (
            "arf_n10",
            river.forest.ARFClassifier(n_models=10),
        ),
        (
            "arf_n25",
            river.forest.ARFClassifier(n_models=25),
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / ".." / ".." / "datasets" / "electricity.csv"
    output_csv = script_dir / "electricity_results.csv"

    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading dataset...", file=sys.stderr)
    stream = load_electricity(dataset_path)
    n_samples = len(stream)
    n_features = len(FEATURE_NAMES)

    header = (
        f"\n=== Electricity Dataset Prequential Results (River) ===\n"
        f"Dataset: {n_samples} samples, {n_features} features, binary classification\n"
        f"Protocol: prequential (test-then-train)\n"
    )
    print(header, file=sys.stderr)

    col_fmt = "{:<30s} {:>6s}  {:>6s}  {:>10s}  {:>12s}"
    row_fmt = "{:<30s} {:>6.4f}  {:>6.4f}  {:>10.1f}  {:>10.0f} s/s"

    print(col_fmt.format("Model", "Acc", "Kappa", "Time(ms)", "Throughput"), file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    results: list[dict] = []

    for label, model in make_models():
        print(f"\nRunning {label}...", file=sys.stderr)
        res = evaluate_prequential(model, stream, model_name=label, checkpoint_interval=5000)
        print(
            row_fmt.format(label, res["accuracy"], res["kappa"], res["time_ms"], res["throughput"]),
            file=sys.stderr,
        )
        results.append({
            "model": label,
            "accuracy": f"{res['accuracy']:.6f}",
            "kappa": f"{res['kappa']:.6f}",
            "time_ms": f"{res['time_ms']:.1f}",
            "throughput": f"{res['throughput']:.0f}",
            "n_samples": res["n_samples"],
        })

    # Print final summary table
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(header, file=sys.stderr)
    print(col_fmt.format("Model", "Acc", "Kappa", "Time(ms)", "Throughput"), file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    for r in results:
        print(
            row_fmt.format(
                r["model"],
                float(r["accuracy"]),
                float(r["kappa"]),
                float(r["time_ms"]),
                float(r["throughput"]),
            ),
            file=sys.stderr,
        )

    # Write CSV
    os.makedirs(output_csv.parent, exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
