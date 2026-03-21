#!/usr/bin/env python3
"""
collect_results.py - Aggregate benchmark results from irithyll, River, and XGBoost/LightGBM
across multiple datasets into a unified comparison and BENCHMARKS.md file.

Datasets: Electricity, Airlines, Covertype

irithyll results are hardcoded (from cargo bench stderr output).
River and XGBoost/LightGBM results are read from CSV files if they exist.

Usage:
    python comparison/collect_results.py
"""

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARKS_MD = PROJECT_ROOT / "BENCHMARKS.md"

DATASETS = ["electricity", "airlines", "covertype"]

DATASET_META = {
    "electricity": {
        "title": "Electricity",
        "description": "45K samples, binary, concept drift",
        "samples": "45,312",
        "features": "8",
        "task": "Binary classification",
        "drift": "Yes (real-world, non-stationary)",
        "detail": (
            "The [Electricity dataset](https://www.openml.org/d/151) is a standard benchmark for\n"
            "concept drift in streaming ML. It contains 45,312 samples with 8 features, representing\n"
            "electricity demand in New South Wales, Australia. The task is binary classification\n"
            "(price up/down), and the data exhibits real-world concept drift."
        ),
    },
    "airlines": {
        "title": "Airlines",
        "description": "539K samples, binary, large scale",
        "samples": "539,383",
        "features": "7",
        "task": "Binary classification",
        "drift": "Yes (temporal, seasonal patterns)",
        "detail": (
            "The [Airlines dataset](https://www.openml.org/d/1169) is a large-scale benchmark for\n"
            "streaming classification. It contains 539,383 samples with 7 features, representing\n"
            "US flight delay records. The task is binary classification (delayed/on-time), with\n"
            "temporal concept drift from seasonal and operational changes."
        ),
    },
    "covertype": {
        "title": "Covertype",
        "description": "581K samples, 7-class, high dimensionality",
        "samples": "581,012",
        "features": "54",
        "task": "7-class classification",
        "drift": "No (static distribution, but challenging multi-class)",
        "detail": (
            "The [Covertype dataset](https://www.openml.org/d/150) is a large-scale multi-class\n"
            "benchmark. It contains 581,012 samples with 54 features (10 quantitative, 44 binary),\n"
            "representing forest cover types in the Roosevelt National Forest. The task is 7-class\n"
            "classification, testing a model's ability to handle high dimensionality and class imbalance."
        ),
    },
}


def river_csv(dataset: str) -> Path:
    return SCRIPT_DIR / "river" / f"{dataset}_results.csv"


def xgboost_csv(dataset: str) -> Path:
    return SCRIPT_DIR / "xgboost" / f"{dataset}_results.csv"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str
    dataset: str
    model_type: str  # "streaming" or "batch retrain"
    accuracy: Optional[float]
    kappa: Optional[float]
    throughput: Optional[int]  # samples/second
    macro_f1: Optional[float] = None  # only for covertype


# ---------------------------------------------------------------------------
# irithyll hardcoded results (from cargo bench stderr)
# ---------------------------------------------------------------------------

IRITHYLL_RESULTS = [
    # --- Electricity ---
    BenchmarkResult(
        model="irithyll SGBT 25t d4 (lr=0.05)",
        dataset="electricity",
        model_type="streaming",
        accuracy=0.7159,
        kappa=0.3709,
        throughput=67063,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.05)",
        dataset="electricity",
        model_type="streaming",
        accuracy=0.8188,
        kappa=0.6155,
        throughput=16347,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.1)",
        dataset="electricity",
        model_type="streaming",
        accuracy=0.8583,
        kappa=0.7041,
        throughput=19011,
    ),
    BenchmarkResult(
        model="irithyll SGBT 100t d6 (lr=0.1)",
        dataset="electricity",
        model_type="streaming",
        accuracy=0.8852,
        kappa=0.7619,
        throughput=8184,
    ),
    # --- Airlines ---
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.05)",
        dataset="airlines",
        model_type="streaming",
        accuracy=0.6253,
        kappa=0.1802,
        throughput=9222,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.1)",
        dataset="airlines",
        model_type="streaming",
        accuracy=0.6488,
        kappa=0.2449,
        throughput=9054,
    ),
    BenchmarkResult(
        model="irithyll SGBT 100t d6 (lr=0.1)",
        dataset="airlines",
        model_type="streaming",
        accuracy=0.6558,
        kappa=0.2684,
        throughput=4094,
    ),
    # --- Covertype ---
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.05)",
        dataset="covertype",
        model_type="streaming",
        accuracy=0.8938,
        kappa=0.8265,
        throughput=591,
        macro_f1=0.8173,
    ),
    BenchmarkResult(
        model="irithyll SGBT 50t d6 (lr=0.1)",
        dataset="covertype",
        model_type="streaming",
        accuracy=0.9247,
        kappa=0.8780,
        throughput=584,
        macro_f1=0.8710,
    ),
    BenchmarkResult(
        model="irithyll SGBT 100t d6 (lr=0.1)",
        dataset="covertype",
        model_type="streaming",
        accuracy=0.9456,
        kappa=0.9122,
        throughput=200,
        macro_f1=0.9098,
    ),
]


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_csv_results(path: Path, dataset: str, source_label: str) -> list[BenchmarkResult]:
    """Read benchmark results from a CSV file.

    Expected CSV columns: model, type, accuracy, kappa, throughput
    Columns are matched case-insensitively and stripped of whitespace.
    The dataset name is tagged onto each result.
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
                f1_str = get_col(row, "macro_f1", "f1", "macro-f1", "macro_f1_score")

                # Infer model_type from context when CSV doesn't have a type column
                window_str = get_col(row, "window_size", "window")
                if model_type == "unknown":
                    if window_str:
                        model_type = "batch retrain"
                    elif "river" in source_label.lower():
                        model_type = "streaming"
                    elif "xgboost" in source_label.lower():
                        model_type = "batch retrain"
                    else:
                        model_type = "streaming"

                if model is None:
                    print(f"  [WARN] {source_label} row {i}: no model name, skipping")
                    continue

                accuracy = float(acc_str) if acc_str else None
                kappa = float(kappa_str) if kappa_str else None
                throughput = int(float(tp_str)) if tp_str else None
                macro_f1 = float(f1_str) if f1_str else None

                results.append(BenchmarkResult(
                    model=model,
                    dataset=dataset,
                    model_type=model_type,
                    accuracy=accuracy,
                    kappa=kappa,
                    throughput=throughput,
                    macro_f1=macro_f1,
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


def fmt_f1(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "---"


def fmt_throughput(val: Optional[int]) -> str:
    return f"{val:,}" if val is not None else "---"


def fmt_pct(val: Optional[float]) -> str:
    """Format as percentage with one decimal, e.g. 88.5%."""
    return f"{val * 100:.1f}%" if val is not None else "---"


def fmt_tp_short(val: Optional[int]) -> str:
    """Format throughput for summary table, e.g. '8K s/s'."""
    if val is None:
        return "---"
    if val >= 1000:
        return f"{val // 1000}K s/s"
    return f"{val:,} s/s"


def build_streaming_table(results: list[BenchmarkResult], include_f1: bool = False) -> str:
    """Build markdown table for streaming models."""
    if include_f1:
        header = "| Model | Library | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |"
        sep = "|-------|---------|----------|-------|----------|------------------|"
    else:
        header = "| Model | Library | Accuracy | Kappa | Throughput (s/s) |"
        sep = "|-------|---------|----------|-------|------------------|"

    rows = []
    for r in results:
        lib = _infer_library(r.model)
        if include_f1:
            rows.append(
                f"| {r.model} | {lib} | {fmt_acc(r.accuracy)} "
                f"| {fmt_kappa(r.kappa)} | {fmt_f1(r.macro_f1)} "
                f"| {fmt_throughput(r.throughput)} |"
            )
        else:
            rows.append(
                f"| {r.model} | {lib} | {fmt_acc(r.accuracy)} "
                f"| {fmt_kappa(r.kappa)} | {fmt_throughput(r.throughput)} |"
            )
    return "\n".join([header, sep] + rows)


def build_batch_table(results: list[BenchmarkResult], include_f1: bool = False) -> str:
    """Build markdown table for batch retrain models."""
    if include_f1:
        header = "| Model | Library | Window | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |"
        sep = "|-------|---------|--------|----------|-------|----------|------------------|"
    else:
        header = "| Model | Library | Window | Accuracy | Kappa | Throughput (s/s) |"
        sep = "|-------|---------|--------|----------|-------|------------------|"

    rows = []
    for r in results:
        lib = _infer_library(r.model)
        window = _extract_window(r.model)
        if include_f1:
            rows.append(
                f"| {r.model} | {lib} | {window} | {fmt_acc(r.accuracy)} "
                f"| {fmt_kappa(r.kappa)} | {fmt_f1(r.macro_f1)} "
                f"| {fmt_throughput(r.throughput)} |"
            )
        else:
            rows.append(
                f"| {r.model} | {lib} | {window} | {fmt_acc(r.accuracy)} "
                f"| {fmt_kappa(r.kappa)} | {fmt_throughput(r.throughput)} |"
            )
    return "\n".join([header, sep] + rows)


def _infer_library(model_name: str) -> str:
    """Infer library name from model name string."""
    name_lower = model_name.lower()
    if "irithyll" in name_lower:
        return "irithyll"
    elif "arf" in name_lower or "hoeffding" in name_lower:
        return "River"
    elif "xgb" in name_lower or "xgboost" in name_lower:
        return "XGBoost"
    elif "lgbm" in name_lower or "lightgbm" in name_lower:
        return "LightGBM"
    elif "river" in name_lower:
        return "River"
    return "unknown"


def _extract_window(model_name: str) -> str:
    """Extract window size from model name like 'xgb_w500' -> '500'."""
    name_lower = model_name.lower()
    if "_w" in name_lower:
        parts = name_lower.split("_w")
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[-1]
    return "---"


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def best_result_for_library(results: list[BenchmarkResult], dataset: str, library: str) -> Optional[BenchmarkResult]:
    """Find the best result (by accuracy) for a given library and dataset."""
    candidates = [
        r for r in results
        if r.dataset == dataset and _infer_library(r.model) == library and r.accuracy is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.accuracy)  # type: ignore


def build_summary_table(all_results: list[BenchmarkResult]) -> str:
    """Build a unified summary table showing best result per library per dataset."""
    header = "| Dataset | irithyll Best | River Best | XGBoost Best | LightGBM Best |"
    sep = "|---------|--------------|------------|--------------|---------------|"

    rows = []
    for ds in DATASETS:
        cells = []
        for lib in ["irithyll", "River", "XGBoost", "LightGBM"]:
            best = best_result_for_library(all_results, ds, lib)
            if best is not None:
                cells.append(f"{fmt_pct(best.accuracy)} ({fmt_tp_short(best.throughput)})")
            else:
                cells.append("---")
        meta = DATASET_META[ds]
        rows.append(f"| {meta['title']} | {' | '.join(cells)} |")

    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# BENCHMARKS.md generation
# ---------------------------------------------------------------------------

def generate_dataset_section(dataset: str, all_results: list[BenchmarkResult]) -> str:
    """Generate the markdown section for a single dataset."""
    meta = DATASET_META[dataset]
    ds_results = [r for r in all_results if r.dataset == dataset]
    streaming = [r for r in ds_results if r.model_type == "streaming"]
    batch = [r for r in ds_results if r.model_type != "streaming"]
    include_f1 = dataset == "covertype"

    lines = []
    lines.append(f"## {meta['title']} ({meta['description']})")
    lines.append("")
    lines.append(meta["detail"])
    lines.append("")
    lines.append(f"- **Samples:** {meta['samples']}")
    lines.append(f"- **Features:** {meta['features']}")
    lines.append(f"- **Task:** {meta['task']}")
    lines.append(f"- **Drift:** {meta['drift']}")
    lines.append("")

    if streaming:
        lines.append("### Streaming Models")
        lines.append("")
        lines.append(build_streaming_table(streaming, include_f1=include_f1))
        lines.append("")

    if batch:
        lines.append("### Batch Models")
        lines.append("")
        lines.append(build_batch_table(batch, include_f1=include_f1))
        lines.append("")

    if not streaming and not batch:
        lines.append("_No results available yet for this dataset._")
        lines.append("")

    return "\n".join(lines)


def generate_benchmarks_md(all_results: list[BenchmarkResult],
                           dataset_availability: dict[str, dict[str, bool]]) -> str:
    """Generate full BENCHMARKS.md content.

    dataset_availability maps dataset -> {"river": bool, "xgboost": bool}
    """
    # Build per-dataset sections
    dataset_sections = []
    for ds in DATASETS:
        dataset_sections.append(generate_dataset_section(ds, all_results))

    # Build summary table
    summary_table = build_summary_table(all_results)

    # Build missing results notes
    missing_notes = []
    for ds in DATASETS:
        avail = dataset_availability.get(ds, {})
        meta = DATASET_META[ds]
        if not avail.get("river", False):
            missing_notes.append(
                f"- **River ({meta['title']}):** Results not yet available. "
                f"Run `python comparison/river/run_{ds}.py` to generate."
            )
        if not avail.get("xgboost", False):
            missing_notes.append(
                f"- **XGBoost/LightGBM ({meta['title']}):** Results not yet available. "
                f"Run `python comparison/xgboost/run_{ds}.py` to generate."
            )

    missing_section = ""
    if missing_notes:
        missing_section = (
            "\n### Missing Results\n\n"
            + "\n".join(missing_notes)
            + "\n"
        )

    dataset_content = "\n".join(dataset_sections)

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

## Evaluation Protocol

- **Metric: Accuracy** -- proportion of correct predictions
- **Metric: Cohen's Kappa** -- agreement corrected for chance (more informative under class imbalance)
- **Metric: Macro-F1** -- macro-averaged F1 score (used for multi-class datasets like Covertype)
- **Throughput** -- samples/second for the full train+predict loop
- **Protocol:** Prequential (test-then-train), single pass over the dataset

{dataset_content}
## Summary

Best result per library across all datasets:

{summary_table}
{missing_section}
## Learning Curves

Learning curve CSVs (accuracy over time) are available in `datasets/results/` for plotting:
- `electricity_learning_curve.csv`
- `airlines_learning_curve.csv`
- `covertype_learning_curve.csv`

## Hardware

> Results collected on: _(fill in your hardware here)_
>
> - **CPU:**
> - **RAM:**
> - **OS:**

## Reproducing

```bash
# 0. Download datasets
python datasets/download.py

# 1. Run irithyll benchmarks (Rust)
cargo bench --bench real_dataset_bench -- detailed

# 2. Run River benchmarks (Python)
python comparison/river/bench_river.py

# 3. Run XGBoost/LightGBM benchmarks (Python)
python comparison/xgboost/bench_xgb.py

# 4. Collect and generate BENCHMARKS.md
python comparison/collect_results.py
```

## Limitations

This comparison is provided for transparency and to help users make informed decisions.
We acknowledge the following limitations:

1. **Hyperparameter sensitivity.** Each model was tuned with reasonable but not
   exhaustive hyperparameter search. Better configurations may exist for all models.

2. **Apples-to-oranges on batch vs streaming.** Batch models (XGBoost, LightGBM) with
   sliding-window retraining have a fundamentally different compute profile than true
   streaming learners. Throughput comparisons across paradigms should be interpreted
   with this in mind.

3. **No ensemble/pipeline comparisons.** River supports rich pipelines
   (preprocessing + drift detection + model). We compare base models only.

4. **Single machine, single thread.** All benchmarks are single-threaded. Libraries
   with parallel prediction (e.g., LightGBM) may perform differently in multi-threaded
   settings.

5. **irithyll results are from Rust `cargo bench`.** River and XGBoost results are from
   Python. The language runtime difference is a real factor in throughput but not in
   accuracy/kappa.

6. **Dataset selection.** These three datasets are well-known streaming ML benchmarks,
   but they do not capture the full range of streaming ML challenges. Results may differ
   on other data distributions, feature types, or class structures.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== irithyll v8.2.1 -- Multi-Dataset Benchmark Result Collector ===\n")

    all_results: list[BenchmarkResult] = list(IRITHYLL_RESULTS)
    dataset_availability: dict[str, dict[str, bool]] = {}

    for ds in DATASETS:
        meta = DATASET_META[ds]
        print(f"--- {meta['title']} ---")
        avail: dict[str, bool] = {"river": False, "xgboost": False}

        # River
        rcsv = river_csv(ds)
        print(f"  [River] Looking for {rcsv}")
        river_results = read_csv_results(rcsv, ds, f"River/{meta['title']}")
        if river_results:
            avail["river"] = True
            all_results.extend(river_results)
        else:
            print(f"  [SKIP] River {meta['title']} results not yet run")

        # XGBoost / LightGBM
        xcsv = xgboost_csv(ds)
        print(f"  [XGBoost/LightGBM] Looking for {xcsv}")
        xgb_results = read_csv_results(xcsv, ds, f"XGBoost/{meta['title']}")
        if xgb_results:
            avail["xgboost"] = True
            all_results.extend(xgb_results)
        else:
            print(f"  [SKIP] XGBoost/LightGBM {meta['title']} results not yet run")

        dataset_availability[ds] = avail
        print()

    # Print per-dataset tables to stdout
    for ds in DATASETS:
        meta = DATASET_META[ds]
        ds_results = [r for r in all_results if r.dataset == ds]
        streaming = [r for r in ds_results if r.model_type == "streaming"]
        batch = [r for r in ds_results if r.model_type != "streaming"]
        include_f1 = ds == "covertype"

        print("=" * 70)
        print(f"  irithyll v8.2.1 -- {meta['title']} Benchmark Comparison")
        print("=" * 70)
        print()

        if streaming:
            print("Streaming Models:")
            print(build_streaming_table(streaming, include_f1=include_f1))
            print()
        if batch:
            print("Batch Models:")
            print(build_batch_table(batch, include_f1=include_f1))
            print()

    # Print summary
    print("=" * 70)
    print("  Summary -- Best per Library")
    print("=" * 70)
    print()
    print(build_summary_table(all_results))
    print()

    # Write BENCHMARKS.md
    md_content = generate_benchmarks_md(all_results, dataset_availability)
    BENCHMARKS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARKS_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Wrote {BENCHMARKS_MD}")


if __name__ == "__main__":
    main()
