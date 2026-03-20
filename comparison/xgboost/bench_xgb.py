#!/usr/bin/env python3
"""
Benchmark XGBoost and LightGBM on the Electricity dataset using periodic
batch retraining — the standard way to deploy batch learners on a stream.

Protocol
--------
For each (model, retrain_window) pair we iterate through every sample in
arrival order.  A sliding *window* accumulates samples.  Whenever the window
fills we retrain from scratch on the window contents and flush it.  Predictions
(and therefore metric updates) only happen **after** the first retrain so the
model always has something to predict with.

Metrics
-------
* Accuracy   — cumulative over all evaluated samples
* Cohen's κ  — cumulative, same sample set
* Wall-clock time (ms)
* Throughput  — evaluated samples / wall-clock seconds

# pip install xgboost lightgbm scikit-learn numpy
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_electricity(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from the Electricity CSV.

    Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    Label:    class  (UP → 1, DOWN → 0)
    """
    xs: list[list[float]] = []
    ys: list[int] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xs.append([
                float(row["date"]),
                float(row["day"]),
                float(row["period"]),
                float(row["nswprice"]),
                float(row["nswdemand"]),
                float(row["vicprice"]),
                float(row["vicdemand"]),
                float(row["transfer"]),
            ])
            ys.append(1 if row["class"].strip() == "UP" else 0)

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int32)


# ---------------------------------------------------------------------------
# Batch-retrain evaluation loop
# ---------------------------------------------------------------------------

def evaluate_batch_retrain(
    model,
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> dict:
    """Run periodic batch retrain and return metrics dict."""

    n_samples = X.shape[0]
    model_trained = False

    # Accumulators for the window
    win_X: list[np.ndarray] = []
    win_y: list[int] = []

    # Accumulators for metrics (only post-first-train samples)
    all_true: list[int] = []
    all_pred: list[int] = []

    t0 = time.perf_counter()

    for i in range(n_samples):
        xi = X[i : i + 1]  # keep 2-D
        yi = int(y[i])

        # --- predict (only if we have a trained model) ---
        if model_trained:
            yp = int(model.predict(xi)[0])
            all_true.append(yi)
            all_pred.append(yp)

        # --- accumulate ---
        win_X.append(X[i])
        win_y.append(yi)

        # --- retrain when the window fills ---
        if len(win_X) >= window_size:
            X_win = np.array(win_X, dtype=np.float64)
            y_win = np.array(win_y, dtype=np.int32)
            model.fit(X_win, y_win)
            win_X.clear()
            win_y.clear()
            model_trained = True

    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    n_eval = len(all_true)
    acc = accuracy_score(all_true, all_pred) if n_eval > 0 else 0.0
    kappa = cohen_kappa_score(all_true, all_pred) if n_eval > 0 else 0.0

    return {
        "n_eval": n_eval,
        "accuracy": acc,
        "kappa": kappa,
        "time_ms": elapsed_s * 1000.0,
        "throughput": n_eval / elapsed_s if elapsed_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_models() -> list[tuple[str, object, list[int]]]:
    """Return list of (label, model_constructor, retrain_windows)."""

    windows = [500, 1000, 5000]

    entries: list[tuple[str, object, list[int]]] = []

    # XGBoost variants
    for w in windows:
        model = XGBClassifier(
            n_estimators=50,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        entries.append((f"xgb_w{w}", model, [w]))

    # LightGBM variants
    for w in windows:
        model = LGBMClassifier(
            n_estimators=50,
            max_depth=6,
            verbose=-1,
        )
        entries.append((f"lgbm_w{w}", model, [w]))

    return entries


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
    X, y = load_electricity(dataset_path)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    header = (
        f"\n=== Electricity Dataset Batch Retrain Results ===\n"
        f"Dataset: {n_samples} samples, {n_features} features, "
        f"{'binary' if n_classes == 2 else str(n_classes) + '-class'} classification\n"
        f"Protocol: periodic batch retrain\n"
    )
    print(header, file=sys.stderr)

    col_fmt = "{:<30s} {:>6s}  {:>6s}  {:>10s}  {:>12s}"
    row_fmt = "{:<30s} {:>6.4f}  {:>6.4f}  {:>10.1f}  {:>10.0f} s/s"

    print(col_fmt.format("Model", "Acc", "Kappa", "Time(ms)", "Throughput"), file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    results: list[dict] = []

    for label, model, windows in make_models():
        w = windows[0]
        res = evaluate_batch_retrain(model, X, y, window_size=w)
        print(
            row_fmt.format(label, res["accuracy"], res["kappa"], res["time_ms"], res["throughput"]),
            file=sys.stderr,
        )
        results.append({
            "model": label,
            "window_size": w,
            "accuracy": f"{res['accuracy']:.6f}",
            "kappa": f"{res['kappa']:.6f}",
            "time_ms": f"{res['time_ms']:.1f}",
            "throughput": f"{res['throughput']:.0f}",
            "n_eval": res["n_eval"],
        })

    # Write CSV
    os.makedirs(output_csv.parent, exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
