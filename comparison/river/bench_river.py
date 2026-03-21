#!/usr/bin/env python3
"""
Benchmark River streaming ML models on three datasets using
prequential (test-then-train) evaluation.

Datasets
--------
* Electricity — 45,312 samples, 8 features, binary (UP/DOWN)
* Airlines    — 539,383 samples, 7 features, binary (Delay 0/1)
* Covertype   — 581,012 samples, 54 features, multi-class (7 classes)

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
* Learning curves saved per dataset for best model (ARF n=25)

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
# Feature name lists
# ---------------------------------------------------------------------------

ELECTRICITY_FEATURES = [
    "date", "day", "period",
    "nswprice", "nswdemand",
    "vicprice", "vicdemand",
    "transfer",
]

AIRLINES_FEATURES = [
    "Airline", "Flight", "AirportFrom", "AirportTo",
    "DayOfWeek", "Time", "Length",
]

COVERTYPE_FEATURES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [
    f"Wilderness_Area_{i}" for i in range(1, 5)
] + [
    f"Soil_Type_{i}" for i in range(1, 41)
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_electricity(path: str | Path) -> list[tuple[dict[str, float], int]]:
    """Return list of (x_dict, y) from the Electricity CSV.

    Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    Label:    class  (UP -> 1, DOWN -> 0)
    """
    stream: list[tuple[dict[str, float], int]] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x = {name: float(row[name]) for name in ELECTRICITY_FEATURES}
            y = 1 if row["class"].strip() == "UP" else 0
            stream.append((x, y))

    return stream


def load_airlines(path: str | Path) -> list[tuple[dict[str, float], int]]:
    """Return list of (x_dict, y) from the Airlines CSV.

    Features: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length
    Label:    Delay (0/1 int)
    """
    stream: list[tuple[dict[str, float], int]] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x = {name: float(row[name]) for name in AIRLINES_FEATURES}
            y = int(row["Delay"])
            stream.append((x, y))

    return stream


def load_covertype(path: str | Path) -> list[tuple[dict[str, float], int]]:
    """Return list of (x_dict, y) from the Covertype CSV.

    Features: 54 numeric columns (10 continuous + 4 wilderness + 40 soil type)
    Label:    Cover_Type (1-7 int, kept 1-based)
    """
    stream: list[tuple[dict[str, float], int]] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x = {name: float(row[name]) for name in COVERTYPE_FEATURES}
            y = int(row["Cover_Type"])
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
) -> tuple[dict, list[tuple[int, float, float]]]:
    """Run prequential (test-then-train) evaluation.

    Returns (metrics_dict, learning_curve) where learning_curve is a list of
    (step, accuracy, kappa) tuples recorded at each checkpoint.
    """

    acc_metric = river.metrics.Accuracy()
    kappa_metric = river.metrics.CohenKappa()

    n_samples = len(stream)
    learning_curve: list[tuple[int, float, float]] = []

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
            acc_val = acc_metric.get()
            kappa_val = kappa_metric.get()
            learning_curve.append((count, acc_val, kappa_val))
            print(
                f"  [{model_name}] {count:>7d}/{n_samples}  "
                f"acc={acc_val:.4f}  "
                f"kappa={kappa_val:.4f}  "
                f"{throughput:.0f} s/s",
                file=sys.stderr,
            )

    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    # Final checkpoint (if not already recorded)
    final_acc = acc_metric.get()
    final_kappa = kappa_metric.get()
    if not learning_curve or learning_curve[-1][0] != n_samples:
        learning_curve.append((n_samples, final_acc, final_kappa))

    metrics = {
        "n_samples": n_samples,
        "accuracy": final_acc,
        "kappa": final_kappa,
        "time_ms": elapsed_s * 1000.0,
        "throughput": n_samples / elapsed_s if elapsed_s > 0 else 0.0,
    }

    return metrics, learning_curve


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
# Learning curve CSV writer
# ---------------------------------------------------------------------------

def write_learning_curve(
    output_path: Path,
    curve: list[tuple[int, float, float]],
) -> None:
    """Write a learning curve CSV with columns: step, accuracy, kappa."""
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "accuracy", "kappa"])
        for step, acc, kappa in curve:
            writer.writerow([step, f"{acc:.6f}", f"{kappa:.6f}"])


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------

def run_dataset(
    name: str,
    stream: list[tuple[dict[str, float], int]],
    n_features: int,
    n_classes: int,
    checkpoint_interval: int,
    output_dir: Path,
) -> None:
    """Evaluate all models on a single dataset, print results, write CSVs."""

    n_samples = len(stream)
    class_desc = "binary" if n_classes == 2 else f"{n_classes}-class"

    header = (
        f"\n=== {name.capitalize()} Dataset Prequential Results (River) ===\n"
        f"Dataset: {n_samples} samples, {n_features} features, {class_desc} classification\n"
        f"Protocol: prequential (test-then-train)\n"
    )
    print(header, file=sys.stderr)

    col_fmt = "{:<30s} {:>6s}  {:>6s}  {:>10s}  {:>12s}"
    row_fmt = "{:<30s} {:>6.4f}  {:>6.4f}  {:>10.1f}  {:>10.0f} s/s"

    print(col_fmt.format("Model", "Acc", "Kappa", "Time(ms)", "Throughput"), file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    results: list[dict] = []
    best_curve: list[tuple[int, float, float]] = []

    for label, model in make_models():
        print(f"\nRunning {label}...", file=sys.stderr)
        res, curve = evaluate_prequential(
            model, stream, model_name=label, checkpoint_interval=checkpoint_interval,
        )
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

        # Keep the learning curve from ARF n=25 (best model)
        if label == "arf_n25":
            best_curve = curve

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

    # Write results CSV
    results_csv = output_dir / f"{name}_results.csv"
    os.makedirs(results_csv.parent, exist_ok=True)
    with open(results_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {results_csv}", file=sys.stderr)

    # Write learning curve CSV (ARF n=25)
    if best_curve:
        curve_csv = output_dir / f"{name}_learning_curve.csv"
        write_learning_curve(curve_csv, best_curve)
        print(f"Learning curve written to {curve_csv}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    datasets_dir = script_dir / ".." / ".." / "datasets"

    # Dataset configurations: (name, csv_filename, loader, features_list, n_classes, checkpoint_interval)
    datasets = [
        (
            "electricity",
            "electricity.csv",
            load_electricity,
            ELECTRICITY_FEATURES,
            2,
            5000,
        ),
        (
            "airlines",
            "airlines.csv",
            load_airlines,
            AIRLINES_FEATURES,
            2,
            50000,
        ),
        (
            "covertype",
            "covertype.csv",
            load_covertype,
            COVERTYPE_FEATURES,
            7,
            50000,
        ),
    ]

    for name, csv_filename, loader, features, n_classes, checkpoint_interval in datasets:
        dataset_path = datasets_dir / csv_filename

        if not dataset_path.exists():
            print(
                f"WARNING: {name} dataset not found at {dataset_path}, skipping.",
                file=sys.stderr,
            )
            continue

        print(f"\nLoading {name} dataset...", file=sys.stderr)
        stream = loader(dataset_path)

        run_dataset(
            name=name,
            stream=stream,
            n_features=len(features),
            n_classes=n_classes,
            checkpoint_interval=checkpoint_interval,
            output_dir=script_dir,
        )


if __name__ == "__main__":
    main()
