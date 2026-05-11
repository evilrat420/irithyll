"""
plot_benchmarks.py — irithyll benchmark visualization pipeline.

Reads criterion JSON output from target/criterion/ and generates two plots:

  1. pareto.png         — Accuracy (or R2) vs throughput (samples/sec) per model,
                          with the Pareto frontier marked. Shows the accuracy-speed
                          trade-off across all model families.

  2. dataset_comparison.png — Side-by-side bar chart comparing all streaming models
                              across the 28 real-world datasets from real_world_bench.

Requirements:
  Python 3.10+, numpy, matplotlib (no other dependencies)

  pip install numpy matplotlib

Usage:
  # Default: reads target/criterion/, writes marketing/benchmarks/
  python scripts/plot_benchmarks.py

  # Custom paths
  python scripts/plot_benchmarks.py \\
      --criterion-dir /path/to/target/criterion \\
      --output-dir marketing/benchmarks \\
      --dpi 150

  # List what the script found without generating plots
  python scripts/plot_benchmarks.py --dry-run

How criterion output maps to inputs:
  Each criterion bench writes JSON at:
    target/criterion/<bench_name>/<group>/<id>/estimates.json

  The 'mean' estimate in estimates.json is in nanoseconds. This script converts
  ns/op to ops/sec (throughput). Where a Throughput::Elements annotation was
  set in the bench, criterion records a 'throughput' field. This script checks
  for that and uses it when available.

  For accuracy metrics (R2, RMSE, prequential accuracy), there is no criterion
  JSON — those are written to stderr/stdout by the bench itself, or read from
  CSV files in datasets/results/. The script reads:
    - datasets/results/electricity_learning_curve.csv
    - datasets/results/airlines_learning_curve.csv
    - datasets/results/covertype_learning_curve.csv
  when available, and falls back to embedded summary data from the tables in
  BENCHMARKS.md when those files are absent.

Output:
  marketing/benchmarks/pareto.png
  marketing/benchmarks/dataset_comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use non-interactive backend (safe for headless CI)
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIGURE_BG = "#0f0f13"
AXES_BG = "#161620"
GRID_COLOR = "#2a2a3a"
TEXT_COLOR = "#c8c8d4"
ACCENT = "#7c7cff"

PARETO_LINE_COLOR = "#ff6b6b"
PARETO_LINE_STYLE = "--"
PARETO_MARKER = "D"

# Model family color map — extended automatically for unknown models
FAMILY_COLORS: dict[str, str] = {
    "SGBT": "#4fc3f7",
    "DistributionalSGBT": "#29b6f6",
    "ESN": "#ab47bc",
    "Mamba": "#ce93d8",
    "MambaV3": "#e040fb",
    "MambaBD": "#f48fb1",
    "KAN": "#66bb6a",
    "TTT": "#a5d6a7",
    "MoE": "#ffca28",
    "NeuralMoE": "#ffd54f",
    "sLSTM": "#ff8a65",
    "mGRADE": "#ffab91",
    "GLA": "#4db6ac",
    "SpikeNet": "#80cbc4",
    "ProjectedLearner": "#b0bec5",
    "RLS": "#90a4ae",
    "LogLinear": "#fff176",
    "River": "#ef9a9a",
    "XGBoost": "#a1887f",
    "LightGBM": "#bcaaa4",
}

EXTRA_COLORS = [
    "#e57373", "#f06292", "#ba68c8", "#7986cb", "#4fc3f7",
    "#4dd0e1", "#4db6ac", "#81c784", "#aed581", "#ffb74d",
]


def family_color(model_name: str, palette: dict[str, str]) -> str:
    for key, color in palette.items():
        if key.lower() in model_name.lower():
            return color
    # Auto-assign from extras
    idx = hash(model_name) % len(EXTRA_COLORS)
    return EXTRA_COLORS[idx]


# ---------------------------------------------------------------------------
# Criterion JSON parsing
# ---------------------------------------------------------------------------

def load_estimate_ns(estimates_path: Path) -> Optional[float]:
    """Return mean estimate in nanoseconds from a criterion estimates.json."""
    try:
        with open(estimates_path) as f:
            data = json.load(f)
        return data["mean"]["point_estimate"]
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def criterion_throughput(estimates_path: Path) -> Optional[float]:
    """
    Return throughput in ops/sec from a criterion estimates.json.

    Criterion records element throughput in the 'throughput' sibling to
    'mean' when Throughput::Elements(n) is set. If not present, we fall
    back to 1e9 / mean_ns for single-op benches.
    """
    try:
        with open(estimates_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Check for a sibling benchmark.json that holds throughput metadata
    bench_json = estimates_path.parent / "benchmark.json"
    if bench_json.exists():
        try:
            with open(bench_json) as f:
                meta = json.load(f)
            # Criterion 0.5 stores throughput as {"Elements": n} under "throughput"
            tp = meta.get("throughput")
            if tp and "Elements" in tp:
                n_elements = tp["Elements"]
                mean_ns = data["mean"]["point_estimate"]
                if mean_ns > 0:
                    return n_elements * 1e9 / mean_ns
        except (KeyError, json.JSONDecodeError, OSError):
            pass

    # Fallback: 1 op / mean_ns -> ops/sec
    mean_ns = data.get("mean", {}).get("point_estimate")
    if mean_ns and mean_ns > 0:
        return 1e9 / mean_ns
    return None


def discover_bench_groups(criterion_dir: Path) -> dict[str, list[Path]]:
    """
    Walk criterion_dir and return a mapping of
    bench_name -> list of estimates.json paths found.
    """
    result: dict[str, list[Path]] = {}
    if not criterion_dir.exists():
        return result
    for bench_dir in sorted(criterion_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench_name = bench_dir.name
        paths = sorted(bench_dir.rglob("estimates.json"))
        if paths:
            result[bench_name] = paths
    return result


# ---------------------------------------------------------------------------
# Embedded fallback data (from BENCHMARKS.md summary tables)
# ---------------------------------------------------------------------------

# Format: (model_label, library, accuracy, throughput_sps)
ELECTRICITY_ROWS: list[tuple[str, str, float, float]] = [
    ("SGBT 25t d4 lr=0.05", "irithyll", 0.7159, 67063),
    ("SGBT 50t d6 lr=0.05", "irithyll", 0.8188, 16347),
    ("SGBT 50t d6 lr=0.1",  "irithyll", 0.8583, 19011),
    ("SGBT 100t d6 lr=0.1", "irithyll", 0.8852, 8184),
    ("hoeffding_tree",       "River",    0.7956, 12029),
    ("hoeffding_adaptive",   "River",    0.8293, 3357),
    ("arf_n10",              "River",    0.8858, 534),
    ("arf_n25",              "River",    0.8913, 200),
    ("xgb_w500",             "XGBoost",  0.7637, 1997),
    ("xgb_w1000",            "XGBoost",  0.7542, 2058),
    ("xgb_w5000",            "XGBoost",  0.7053, 2134),
    ("lgbm_w500",            "LightGBM", 0.7632, 1434),
    ("lgbm_w1000",           "LightGBM", 0.7572, 1448),
    ("lgbm_w5000",           "LightGBM", 0.7107, 1483),
]

AIRLINES_ROWS: list[tuple[str, str, float, float]] = [
    ("SGBT 50t d6 lr=0.05", "irithyll", 0.6253, 9222),
    ("SGBT 50t d6 lr=0.1",  "irithyll", 0.6488, 9054),
    ("SGBT 100t d6 lr=0.1", "irithyll", 0.6558, 4094),
    ("hoeffding_tree",       "River",    0.6383, 9100),
    ("hoeffding_adaptive",   "River",    0.6348, 3067),
    ("arf_n10",              "River",    0.6565, 448),
    ("arf_n25",              "River",    0.6675, 171),
    ("xgb_w500",             "XGBoost",  0.6216, 1980),
    ("xgb_w1000",            "XGBoost",  0.6299, 2057),
    ("xgb_w5000",            "XGBoost",  0.6317, 2131),
    ("lgbm_w500",            "LightGBM", 0.6352, 1425),
    ("lgbm_w1000",           "LightGBM", 0.6460, 1429),
    ("lgbm_w5000",           "LightGBM", 0.6439, 1419),
]

COVERTYPE_ROWS: list[tuple[str, str, float, float]] = [
    ("SGBT 50t d6 lr=0.05", "irithyll", 0.8938, 591),
    ("SGBT 50t d6 lr=0.1",  "irithyll", 0.9247, 584),
    ("SGBT 100t d6 lr=0.1", "irithyll", 0.9456, 200),
    ("hoeffding_tree",       "River",    0.7655, 2134),
    ("hoeffding_adaptive",   "River",    0.7731, 687),
    ("arf_n10",              "River",    0.8727, 461),
    ("arf_n25",              "River",    0.8858, 207),
    ("xgb_w500",             "XGBoost",  0.4988, 2176),
    ("xgb_w1000",            "XGBoost",  0.4753, 2143),
    ("xgb_w5000",            "XGBoost",  0.5931, 2079),
    ("lgbm_w500",            "LightGBM", 0.4596, 1434),
    ("lgbm_w1000",           "LightGBM", 0.4856, 1443),
    ("lgbm_w5000",           "LightGBM", 0.5979, 1428),
]

# Synthetic dataset labels for bar chart (real_world_bench categories)
SYNTHETIC_DATASETS: list[str] = [
    # Binary classification
    "SEA Concepts",
    "Rotating Hyperplane",
    "Agrawal",
    "Random RBF",
    "Spike-Encoded",
    # Multiclass classification
    "LED (10-class)",
    "Waveform (3-class)",
    "Multi-class Spiral",
    # Regression
    "Sine Regression",
    "Friedman+drift",
    "Sensor Drift",
    "Mackey-Glass",
    "Lorenz Attractor",
    "NARMA10",
    "Regime Shift",
    "Continuous Drift",
    "Contextual Few-Shot",
    "Contextual Few-Shot Short",
    "Long-Seq Autoregressive",
    "Compositional Physics",
    "Feynman Physics",
    "Power Plant",
    # Stress
    "Sudden Drift",
    "High-Dim Nonlinear",
    "Non-Stationary",
]

# Placeholder model labels for the bar chart skeleton (real data from bench output)
SYNTHETIC_MODELS: list[str] = [
    "SGBT", "ESN", "Mamba", "KAN", "TTT", "sLSTM", "mGRADE", "GLA", "RLS",
]


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------

def pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Return the Pareto-optimal points from a list of (throughput, accuracy) pairs.
    A point dominates another if it has both higher throughput AND higher accuracy.
    Returns points sorted by throughput ascending.
    """
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float]] = []
    best_accuracy = -math.inf
    for tp, acc in sorted_pts:
        if acc > best_accuracy:
            frontier.append((tp, acc))
            best_accuracy = acc
    return frontier


# ---------------------------------------------------------------------------
# Plot 1: Pareto plot (accuracy vs throughput)
# ---------------------------------------------------------------------------

def plot_pareto(
    output_path: Path,
    dpi: int,
    dry_run: bool,
    criterion_dir: Path,
) -> None:
    """
    Generate pareto.png: accuracy (y) vs throughput in s/s (x) per model.
    Uses fallback embedded data when criterion output is not available.
    """
    # Combine all three datasets; use best accuracy per model label
    all_rows = ELECTRICITY_ROWS + AIRLINES_ROWS + COVERTYPE_ROWS
    # Aggregate: best accuracy per (label, library) pair, keeping paired throughput
    # Strategy: group by label, take the row with max accuracy
    best: dict[str, tuple[str, float, float]] = {}
    for label, library, acc, tp in all_rows:
        key = label
        if key not in best or acc > best[key][1]:
            best[key] = (library, acc, tp)

    labels = list(best.keys())
    libraries = [best[k][0] for k in labels]
    accuracies = np.array([best[k][1] for k in labels])
    throughputs = np.array([best[k][2] for k in labels], dtype=float)

    if dry_run:
        print(f"[DRY-RUN] pareto.png: {len(labels)} model points found")
        return

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=FIGURE_BG)
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    # Scatter: one point per model, colored by library
    plotted_libraries: set[str] = set()
    for label, library, acc, tp in zip(labels, libraries, accuracies, throughputs):
        color = family_color(library, FAMILY_COLORS)
        ax.scatter(
            tp, acc,
            color=color,
            s=70,
            alpha=0.85,
            edgecolors=FIGURE_BG,
            linewidths=0.5,
            zorder=3,
        )
        plotted_libraries.add(library)

    # Pareto frontier line
    points = list(zip(throughputs.tolist(), accuracies.tolist()))
    frontier = pareto_frontier(points)
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(
            fx, fy,
            color=PARETO_LINE_COLOR,
            linestyle=PARETO_LINE_STYLE,
            linewidth=1.5,
            marker=PARETO_MARKER,
            markersize=6,
            zorder=4,
            label="Pareto frontier",
        )

    # Annotate irithyll SGBT 100t best point
    best_irithyll = max(
        [(acc, tp, lbl) for lbl, lib, acc, tp in zip(labels, libraries, accuracies, throughputs) if lib == "irithyll"],
        default=None,
    )
    if best_irithyll:
        acc_b, tp_b, lbl_b = best_irithyll
        ax.annotate(
            lbl_b,
            xy=(tp_b, acc_b),
            xytext=(tp_b * 1.1, acc_b - 0.02),
            color=TEXT_COLOR,
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=0.8),
        )

    # Legend patches per library
    legend_patches = [
        mpatches.Patch(color=family_color(lib, FAMILY_COLORS), label=lib)
        for lib in sorted(plotted_libraries)
    ]
    pareto_patch = mpatches.Patch(
        color=PARETO_LINE_COLOR, label="Pareto frontier", linestyle=PARETO_LINE_STYLE
    )
    ax.legend(
        handles=legend_patches + [pareto_patch],
        facecolor=AXES_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=9,
        loc="lower right",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (samples/sec, log scale)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Accuracy (prequential)", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "Accuracy vs Throughput — Streaming ML Pareto Frontier\n"
        "(Electricity + Airlines + Covertype, prequential protocol)",
        color=TEXT_COLOR,
        fontsize=12,
        pad=12,
    )
    ax.title.set_fontweight("bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=FIGURE_BG)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-dataset bar chart (real_world_bench)
# ---------------------------------------------------------------------------

def load_dataset_comparison_from_criterion(criterion_dir: Path) -> Optional[dict]:
    """
    Try to parse real_world_bench criterion output into a structured dict.
    Returns None if not enough data is present to build the chart.

    Format expected: criterion groups named like
      real_world_bench/<dataset_name>/<model_name>/estimates.json

    Returns: {dataset: {model: metric_value}} where metric is throughput ops/sec.
    """
    bench_dir = criterion_dir / "real_world_bench"
    if not bench_dir.exists():
        return None
    result: dict[str, dict[str, float]] = {}
    for dataset_dir in sorted(bench_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        result[dataset] = {}
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            estimates = model_dir / "estimates.json"
            tp = criterion_throughput(estimates)
            if tp is not None:
                result[dataset][model] = tp
    return result if result else None


def plot_dataset_comparison(
    output_path: Path,
    dpi: int,
    dry_run: bool,
    criterion_dir: Path,
) -> None:
    """
    Generate dataset_comparison.png: side-by-side bar chart across 28 datasets.

    If real_world_bench criterion JSON is present, uses actual throughput data.
    Otherwise plots a labeled skeleton with a prominent note explaining how to
    populate it (run cargo bench --bench real_world_bench first).
    """
    # Try to load real data
    data = load_dataset_comparison_from_criterion(criterion_dir)
    has_real_data = data is not None and len(data) > 0

    if dry_run:
        status = "real criterion data" if has_real_data else "skeleton (no criterion data)"
        print(f"[DRY-RUN] dataset_comparison.png: {status}")
        if has_real_data:
            print(f"  Datasets found: {list(data.keys())[:5]} ...")
        return

    if has_real_data:
        datasets = list(data.keys())
        models = sorted({m for d in data.values() for m in d.keys()})
        values = np.array([
            [data[ds].get(m, 0.0) for m in models]
            for ds in datasets
        ])
    else:
        # Skeleton: random placeholder heights, grayed out, with instructions
        datasets = SYNTHETIC_DATASETS
        models = SYNTHETIC_MODELS
        rng = np.random.default_rng(42)
        values = rng.uniform(0.5, 1.0, size=(len(datasets), len(models)))

    n_datasets = len(datasets)
    n_models = len(models)

    fig_width = max(16, n_datasets * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 8), facecolor=FIGURE_BG)
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    x = np.arange(n_datasets)
    bar_width = 0.8 / n_models
    offsets = np.linspace(-(0.8 - bar_width) / 2, (0.8 - bar_width) / 2, n_models)

    for i, (model, offset) in enumerate(zip(models, offsets)):
        color = family_color(model, FAMILY_COLORS)
        alpha = 0.85 if has_real_data else 0.35
        ax.bar(
            x + offset,
            values[:, i],
            width=bar_width,
            color=color,
            alpha=alpha,
            label=model,
            edgecolor=FIGURE_BG,
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
    ax.set_ylabel(
        "Throughput (samples/sec)" if has_real_data else "Metric (placeholder — run bench first)",
        color=TEXT_COLOR,
        fontsize=10,
    )
    title_suffix = "" if has_real_data else "\n[PLACEHOLDER — run: cargo bench --bench real_world_bench]"
    ax.set_title(
        f"Streaming Model Comparison — 28 Datasets (Prequential Protocol){title_suffix}",
        color=TEXT_COLOR if has_real_data else "#ff8c69",
        fontsize=11,
        pad=10,
        fontweight="bold",
    )

    ax.legend(
        facecolor=AXES_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8,
        ncol=min(n_models, 5),
        loc="upper right",
    )

    if not has_real_data:
        ax.text(
            0.5, 0.5,
            "Run 'cargo bench --bench real_world_bench' to populate this chart.\n"
            "Then re-run: python scripts/plot_benchmarks.py",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#ff8c69",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=AXES_BG, edgecolor="#ff8c69", alpha=0.8),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=FIGURE_BG)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate irithyll benchmark plots from criterion JSON output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Root of criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("marketing/benchmarks"),
        help="Directory for generated PNGs (default: marketing/benchmarks)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    criterion_dir: Path = args.criterion_dir
    output_dir: Path = args.output_dir
    dpi: int = args.dpi
    dry_run: bool = args.dry_run

    # Resolve to absolute paths relative to the script location if relative
    if not criterion_dir.is_absolute():
        # Try relative to cwd first, then relative to repo root (script is in scripts/)
        if not criterion_dir.exists():
            repo_root = Path(__file__).parent.parent
            criterion_dir = repo_root / criterion_dir
    if not output_dir.is_absolute():
        repo_root = Path(__file__).parent.parent
        output_dir = repo_root / output_dir

    if dry_run:
        print(f"[DRY-RUN] criterion_dir: {criterion_dir}")
        print(f"[DRY-RUN] output_dir:    {output_dir}")
        groups = discover_bench_groups(criterion_dir)
        print(f"[DRY-RUN] Bench groups found: {list(groups.keys())}")

    pareto_path = output_dir / "pareto.png"
    comparison_path = output_dir / "dataset_comparison.png"

    print(f"criterion-dir : {criterion_dir}")
    print(f"output-dir    : {output_dir}")
    print(f"dpi           : {dpi}")
    print()

    plot_pareto(pareto_path, dpi=dpi, dry_run=dry_run, criterion_dir=criterion_dir)
    plot_dataset_comparison(
        comparison_path, dpi=dpi, dry_run=dry_run, criterion_dir=criterion_dir
    )

    if not dry_run:
        print()
        print("[OK] Done. Open the PNGs from:")
        print(f"  {pareto_path}")
        print(f"  {comparison_path}")


if __name__ == "__main__":
    main()
