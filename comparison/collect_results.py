#!/usr/bin/env python3
"""
collect_results.py - Aggregate benchmark results from irithyll, River, and XGBoost/LightGBM
into a unified comparison table and BENCHMARKS.md file.

irithyll results are hardcoded (from cargo bench stderr output on Electricity dataset).
River and XGBoost/LightGBM results are read from CSV files if they exist.

Usage:
    python comparison/collect_results.py
"""

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RIVER_CSV = SCRIPT_DIR / "river" / "electricity_results.csv"
XGBOOST_CSV = SCRIPT_DIR / "xgboost" / "electricity_results.csv"
BENCHMARKS_MD = PROJECT_ROOT / "BENCHMARKS.md"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str
    model_type: str  # "streaming" or "batch retrain"
    accuracy: Optional[float]
    kappa: Optional[float]
    throughput: Optional[int]  # samples/second


# ---------------------------------------------------------------------------
# irithyll hardcoded results (from cargo bench stderr on Electricity dataset)
# ---------------------------------------------------------------------------

IRITHYLL_RESULTS = [
    BenchmarkResult(
        model="irithyll SGBT 25t d4 (lr=0.05)",
        model_type="streaming",
        accuracy=0.7159,
        kappa=0.3709,
        throughput=67063,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.05)",
        model_type="streaming",
        accuracy=0.8188,
        kappa=0.6155,
        throughput=16347,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.1)",
        model_type="streaming",
        accuracy=0.8583,
        kappa=0.7041,
        throughput=19011,
    ),
    BenchmarkResult(
        model="irithyll SGBT 100t d6 (lr=0.1)",
        model_type="streaming",
        accuracy=0.8852,
        kappa=0.7619,
        throughput=8184,
    ),
]


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_csv_results(path: Path, source_label: str) -> list[BenchmarkResult]:
    """Read benchmark results from a CSV file.

    Expected CSV columns: model, type, accuracy, kappa, throughput
    Columns are matched case-insensitively and stripped of whitespace.
    """
    if not path.exists():
        return []

    results = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize header names
            if reader.fieldnames is None:
                print(f"  [WARN] {source_label}: CSV has no header row")
                return []

            # Build a lowercase -> original mapping for flexible column matching
            col_map: dict[str, str] = {}
            for name in reader.fieldnames:
                col_map[name.strip().lower()] = name

            def get_col(row: dict, *candidates: str) -> Optional[str]:
                for c in candidates:
                    original = col_map.get(c)
                    if original and row.get(original):
                        return row[original].strip()
                return None

            for i, row in enumerate(reader):
                model = get_col(row, "model", "name", "algorithm")
                model_type = get_col(row, "type", "model_type", "paradigm") or "unknown"
                acc_str = get_col(row, "accuracy", "acc")
                kappa_str = get_col(row, "kappa", "cohen_kappa")
                tp_str = get_col(row, "throughput", "throughput (s/s)", "samples_per_sec")

                if model is None:
                    print(f"  [WARN] {source_label} row {i}: no model name, skipping")
                    continue

                accuracy = float(acc_str) if acc_str else None
                kappa = float(kappa_str) if kappa_str else None
                throughput = int(float(tp_str)) if tp_str else None

                results.append(BenchmarkResult(
                    model=model,
                    model_type=model_type,
                    accuracy=accuracy,
                    kappa=kappa,
                    throughput=throughput,
                ))

        print(f"  [OK] {source_label}: loaded {len(results)} result(s) from {path.name}")
    except Exception as e:
        print(f"  [ERROR] {source_label}: failed to read {path} -- {e}")

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_acc(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "---"


def fmt_kappa(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "---"


def fmt_throughput(val: Optional[int]) -> str:
    return f"{val:,}" if val is not None else "---"


def build_table_rows(results: list[BenchmarkResult]) -> list[str]:
    """Build markdown table rows from results."""
    rows = []
    for r in results:
        rows.append(
            f"| {r.model} | {r.model_type} | {fmt_acc(r.accuracy)} "
            f"| {fmt_kappa(r.kappa)} | {fmt_throughput(r.throughput)} |"
        )
    return rows


def build_table(results: list[BenchmarkResult]) -> str:
    """Build the full markdown table string."""
    header = "| Model | Type | Accuracy | Kappa | Throughput (s/s) |"
    sep = "|-------|------|----------|-------|------------------|"
    rows = build_table_rows(results)
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# BENCHMARKS.md generation
# ---------------------------------------------------------------------------

def generate_benchmarks_md(all_results: list[BenchmarkResult],
                           has_river: bool,
                           has_xgboost: bool) -> str:
    """Generate full BENCHMARKS.md content."""
    table = build_table(all_results)

    missing_notes = []
    if not has_river:
        missing_notes.append(
            "- **River:** Results not yet available. "
            "Run `python comparison/river/run_electricity.py` to generate."
        )
    if not has_xgboost:
        missing_notes.append(
            "- **XGBoost/LightGBM:** Results not yet available. "
            "Run `python comparison/xgboost/run_electricity.py` to generate."
        )

    missing_section = ""
    if missing_notes:
        missing_section = (
            "\n### Missing Results\n\n"
            + "\n".join(missing_notes)
            + "\n"
        )

    return f"""\
# irithyll Benchmarks

Streaming ML benchmark comparisons for irithyll against established libraries.

## Methodology

All benchmarks use **prequential evaluation** (test-then-train): each sample is first
used for prediction (to measure accuracy), then used for training. This is the standard
evaluation protocol for online/streaming learners and reflects real-world deployment
where models must predict on unseen data before learning from it.

For batch learners (XGBoost, LightGBM), we simulate a realistic streaming deployment:
the model is periodically retrained on a sliding window of recent samples. This is how
batch models are typically deployed in production streaming scenarios.

Throughput is measured as **samples per second** (s/s) including both the predict and
train steps, reflecting end-to-end online learning speed.

## Dataset: Electricity

The [Electricity dataset](https://www.openml.org/d/151) is a standard benchmark for
concept drift in streaming ML. It contains 45,312 samples with 8 features, representing
electricity demand in New South Wales, Australia. The task is binary classification
(price up/down), and the data exhibits real-world concept drift.

- **Samples:** 45,312
- **Features:** 8
- **Task:** Binary classification
- **Drift:** Yes (real-world, non-stationary)

## Evaluation Protocol

- **Metric: Accuracy** -- proportion of correct predictions
- **Metric: Cohen's Kappa** -- agreement corrected for chance (more informative under class imbalance)
- **Throughput** -- samples/second for the full train+predict loop
- **Protocol:** Prequential (test-then-train), single pass over the dataset

## Results

{table}
{missing_section}
## Hardware

> Results collected on: _(fill in your hardware here)_
>
> - **CPU:**
> - **RAM:**
> - **OS:**

## Limitations

This comparison is provided for transparency and to help users make informed decisions.
We acknowledge the following limitations:

1. **Single dataset.** Electricity is a well-known benchmark, but no single dataset
   captures the full range of streaming ML challenges. Results may differ on other data.

2. **Hyperparameter sensitivity.** Each model was tuned with reasonable but not
   exhaustive hyperparameter search. Better configurations may exist for all models.

3. **Apples-to-oranges on batch vs streaming.** Batch models (XGBoost, LightGBM) with
   sliding-window retraining have a fundamentally different compute profile than true
   streaming learners. Throughput comparisons across paradigms should be interpreted
   with this in mind.

4. **No ensemble/pipeline comparisons.** River supports rich pipelines
   (preprocessing + drift detection + model). We compare base models only.

5. **Single machine, single thread.** All benchmarks are single-threaded. Libraries
   with parallel prediction (e.g., LightGBM) may perform differently in multi-threaded
   settings.

6. **irithyll results are from Rust `cargo bench`.** River and XGBoost results are from
   Python. The language runtime difference is a real factor in throughput but not in
   accuracy/kappa.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== irithyll -- Electricity Benchmark Result Collector ===\n")

    # 1. irithyll (hardcoded)
    print("[irithyll] Using hardcoded results (from cargo bench stderr)")
    all_results: list[BenchmarkResult] = list(IRITHYLL_RESULTS)

    # 2. River
    print(f"[River] Looking for {RIVER_CSV}")
    river_results = read_csv_results(RIVER_CSV, "River")
    has_river = len(river_results) > 0
    if not has_river:
        print("  [SKIP] River results not yet run -- will show as missing")
        all_results.append(BenchmarkResult(
            model="River (not yet run)",
            model_type="streaming",
            accuracy=None,
            kappa=None,
            throughput=None,
        ))
    else:
        all_results.extend(river_results)

    # 3. XGBoost / LightGBM
    print(f"[XGBoost/LightGBM] Looking for {XGBOOST_CSV}")
    xgboost_results = read_csv_results(XGBOOST_CSV, "XGBoost/LightGBM")
    has_xgboost = len(xgboost_results) > 0
    if not has_xgboost:
        print("  [SKIP] XGBoost/LightGBM results not yet run -- will show as missing")
        all_results.append(BenchmarkResult(
            model="XGBoost/LightGBM (not yet run)",
            model_type="batch retrain",
            accuracy=None,
            kappa=None,
            throughput=None,
        ))
    else:
        all_results.extend(xgboost_results)

    # Print unified table to stdout
    print("\n")
    print("=" * 70)
    print("  irithyll v8.1.2 -- Electricity Benchmark Comparison")
    print("=" * 70)
    print()
    table = build_table(all_results)
    print(table)
    print()

    # Write BENCHMARKS.md
    md_content = generate_benchmarks_md(all_results, has_river, has_xgboost)
    BENCHMARKS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARKS_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Wrote {BENCHMARKS_MD}")


if __name__ == "__main__":
    main()
