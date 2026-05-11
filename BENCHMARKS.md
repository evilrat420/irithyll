# irithyll Benchmarks

Streaming ML benchmark comparisons for irithyll against established libraries.

Numerical data and comparisons live here. The README links to this document; do
not duplicate performance tables in README.

---

## Contents

1. [Evaluation philosophy](#evaluation-philosophy)
2. [Bench taxonomy](#bench-taxonomy)
3. [Running benchmarks](#running-benchmarks)
4. [Real-dataset results](#real-dataset-results-sgbt)
5. [Visualization](#visualization)
6. [Hardware](#hardware)
7. [Limitations](#limitations)

---

## Evaluation philosophy

All irithyll benchmarks follow **prequential evaluation** (test-then-train): each
sample is predicted on first, then used for training. This is the standard protocol
for online learners and reflects real deployment where models must predict before
seeing the label (Gama et al., 2013).

For batch learners (XGBoost, LightGBM), a realistic streaming deployment is
simulated: the model is periodically retrained on a sliding window of recent samples.
Throughput is measured as **samples per second** (s/s) including both predict and
train steps, reflecting end-to-end online learning speed.

No artificial or simulated data is used for accuracy benchmarks. Throughput benches
use deterministic synthetic data (seeded xorshift64) for reproducibility.

---

## Bench taxonomy

The bench suite is split into seven categories. Each bench file is a standalone
criterion harness unless noted.

### 1. Throughput benches

**File:** `benches/throughput_bench.rs`

Measures sustained samples/sec across SGBT configurations. Three groups:

- `train_throughput` — batch of 1000 samples, varying `n_steps` (10/50/100) and
  `n_features` (3/20/100). Establishes training throughput baseline.
- `predict_throughput` — batch of 10 000 predictions, varying `n_steps` and
  `n_features`. Measures pure inference speed at scale.
- `prequential_throughput` — interleaved predict+train loop at batch size 1000.
  This is the figure most representative of live streaming deployment.

```bash
cargo bench --bench throughput_bench
```

Output lands in `target/criterion/train_throughput/`, `target/criterion/predict_throughput/`,
and `target/criterion/prequential_throughput/`.

---

### 2. Scaling benches

**File:** `benches/scaling_bench.rs`

Isolates how latency grows with individual dimensions: `n_steps`, `n_features`,
`max_depth`, and `n_bins`. Each dimension is swept while the others are held at
a sensible default, making it straightforward to reason about the cost of each
configuration knob independently.

```bash
cargo bench --bench scaling_bench
```

Output lands in `target/criterion/scale_n_steps/`, `target/criterion/scale_n_features/`,
`target/criterion/scale_max_depth/`, `target/criterion/scale_n_bins/`.

---

### 3. Packed-inference benches

**File:** `benches/packed_inference_bench.rs`

Six groups covering the `EnsembleView` (f32 packed) and `QuantizedEnsembleView`
(i16 packed) fast-inference paths:

| Group | What it measures |
|---|---|
| `packed_predict_trees` | Single-sample latency vs tree count (10/50/100/500) |
| `packed_predict_depth` | Latency vs max_depth (2/4/6/8) |
| `packed_predict_features` | Latency vs n_features (5/10/50/100) |
| `packed_batch_scaling` | Throughput (Elements/s) vs batch size (1/1K/10K/100K) |
| `packed_vs_sgbt` | Head-to-head: f32 packed vs f64 SGBT on same model |
| `packed_model_size` | Export (serialization) speed; prints packed binary sizes to stderr |

```bash
cargo bench --bench packed_inference_bench
```

Output lands in `target/criterion/packed_predict_trees/` etc.

---

### 4. Quantized-inference benches

**File:** `benches/quantized_bench.rs`

Compares the i16 quantized inference path against f32 packed inference:
latency, throughput (Elements/s), accuracy degradation across configurations,
and binary size. Useful for validating the quantization trade-off before
deploying to memory-constrained targets.

```bash
cargo bench --bench quantized_bench
```

---

### 5. Comparison benches

**File:** `benches/comparison_bench.rs`

Self-comparison across all three irithyll inference paths on the same trained
model: SGBT (f64 tree walk), EnsembleView (f32 branch-free packed),
QuantizedEnsembleView (i16 TurboQuant). No external libraries — pure
internal comparison for auditing the speedup from each export path.

```bash
cargo bench --bench comparison_bench
```

---

### 6. Pipeline benches

**File:** `benches/pipeline_bench.rs`

End-to-end roundtrip: train → export → predict. Covers:

- Export latency scaling with model size.
- Accuracy preservation across the packed pipeline.
- Train + export + predict as a single elapsed measurement.

```bash
cargo bench --bench pipeline_bench
```

---

### 7. Histogram benches

**File:** `benches/histogram_bench.rs`

Low-level benchmarks for histogram accumulation and split-finding:

- `accumulate_64_bins` — 10 000 samples into a 64-bin histogram.
- Additional groups for split-finding and bin-edge queries.

These are the micro-benchmarks closest to the tree-learning inner loop.

```bash
cargo bench --bench histogram_bench
```

---

### 8. Ensemble benches

**File:** `benches/ensemble_bench.rs`

SGBT ensemble train and predict with varying `n_steps`. Complements
`throughput_bench.rs` by focusing on the ensemble-level structure (tree growth,
split-finding overhead) rather than raw sample throughput.

```bash
cargo bench --bench ensemble_bench
```

---

### 9. Parallel benches

**File:** `benches/parallel_bench.rs`

Parallel training and prediction under the `parallel` feature flag (Rayon-backed).
Requires `--features parallel`.

```bash
cargo bench --bench parallel_bench --features parallel
```

---

### 10. Training benches (distributional + MoE)

**File:** `benches/training_bench.rs`

Train and predict throughput for `DistributionalSGBT` and `MoEDistributionalSGBT`.
Complements the plain SGBT throughput bench with the distributional variants.

```bash
cargo bench --bench training_bench
```

---

### 11. Real-dataset benches (SGBT vs batch)

**File:** `benches/real_dataset_bench.rs`

Prequential evaluation of SGBT and MulticlassSGBT on three standard streaming
ML benchmarks sourced from OpenML. Requires downloading datasets first:

```bash
python datasets/download.py
cargo bench --bench real_dataset_bench
# Or with detailed prequential output:
cargo bench --bench real_dataset_bench -- detailed
```

Datasets:

| Dataset | Samples | Features | Task |
|---|---|---|---|
| Electricity (Elec2) | 45,312 | 8 | Binary classification, concept drift |
| Airlines | 539,383 | 7 | Binary classification, temporal drift |
| Covertype | 581,012 | 54 | 7-class classification |

---

### 12. Real-world bench (28-dataset neural suite)

**File:** `benches/real_world_bench.rs`

Expanded suite evaluating all streaming model families across 28 synthetic datasets
using prequential protocol. No external data downloads required — all datasets are
generated deterministically from the `irithyll::generators` module.

Dataset categories:

| Category | Count | Examples |
|---|---|---|
| Binary classification | 5 | SEA Concepts, Rotating Hyperplane, Agrawal, Random RBF, Spike-Encoded |
| Multiclass classification | 3 | LED (10-class), Waveform (3-class), Multi-class Spiral |
| Regression | 14 | Sine, Friedman+drift, Sensor Drift, Mackey-Glass, Lorenz, NARMA10, Regime Shift, Continuous Drift, Contextual Few-Shot, Long-Seq Autoregressive, Compositional Physics, Feynman Physics, Power Plant, Contextual Few-Shot Short |
| Stress tests | 3 | Sudden Drift, High-Dim Nonlinear (50 features), Non-Stationary Sequence |

Models evaluated:

- Tree ensembles: SGBT, DistributionalSGBT
- Neural streaming: ESN, Mamba (V1), KAN, TTT, MoE, sLSTM, mGRADE, MambaV3,
  MambaBD, GLA, SpikeNet
- Composites: ProjectedLearner+Mamba, NeuralMoE (3-expert)
- Linear: RecursiveLeastSquares
- Classification wrappers: `binary_classifier()`, `multiclass_classifier()`

Metrics: RMSE (regression), Accuracy + Kappa + Kappa-T (binary), Accuracy + Kappa (multiclass).

```bash
cargo bench --bench real_world_bench
```

---

### 13. Log-Linear Attention bench

**File:** `benches/log_linear_bench.rs`

Architectural claims verification for the Log-Linear Attention model (Han Guo et al.,
ICLR 2026, arXiv:2506.04761) in the streaming fixed-weights setting:

1. **Multi-scale prequential RMSE** — composite target combining short, medium, and
   long lags. LogLinear's O(log T) Fenwick state versus GLA's single fixed state.
2. **MQAR top-1 recall** — relative gap between LogLinear and GLA at high `n_pairs`.
   Absolute thresholds are paper-empirical and not portable to the fixed-weight
   streaming setting; the bench asserts relative improvement.
3. **Needle-in-haystack stability** — LogLinear must not regress on short-horizon
   retrieval relative to GLA.

```bash
cargo bench --bench log_linear_bench
# Build only (no execution):
cargo bench --bench log_linear_bench --no-run
```

---

### 14. Instruction-count regression bench (iai-callgrind)

**File:** `irithyll-core/benches/packed_inference_iai.rs`

Deterministic instruction-count, branch-count, and cache-behavior measurements
for the packed-inference hot path. Unlike criterion (wall-clock), iai-callgrind
runs each function under valgrind and reports CPU instructions retired — values
are stable across runs and machines, suitable as a CI regression gate.

| Bench | What it measures |
|---|---|
| `single_predict` | 50t, depth 4, single-sample `EnsembleView::predict` |
| `batch_predict` | 50t, depth 4, `EnsembleView::predict_batch` over 1000 samples |
| `deserialize_view` | 100t, depth 6, `EnsembleView::from_bytes` parse + validate |

**Platform constraint:** valgrind is Linux-only. The bench source compiles on
all platforms (default `cargo check` exercise) but execution requires Linux +
valgrind. The bench is feature-gated and dormant by default; opt in via
`--features iai-bench`.

```bash
# Compile-check (any platform):
cargo check -p irithyll-core --features iai-bench --bench packed_inference_iai

# Run on Linux:
cargo bench -p irithyll-core --features iai-bench --bench packed_inference_iai
```

---

---

## Running benchmarks

### Quick reference

```bash
# All criterion benches (long — expect 30-60 min for the full suite)
cargo bench

# Single bench
cargo bench --bench throughput_bench

# Single group within a bench
cargo bench --bench throughput_bench -- prequential_throughput

# Real-dataset benches (download first)
python datasets/download.py
cargo bench --bench real_dataset_bench

# Neural suite (28 synthetic datasets)
cargo bench --bench real_world_bench

# Log-Linear Attention claims
cargo bench --bench log_linear_bench
```

### Criterion output

Criterion writes HTML reports and raw JSON to `target/criterion/`. The JSON for a
given measurement lives at:

```
target/criterion/<bench_name>/<group>/<measurement>/estimates.json
```

These JSON files are the input for `scripts/plot_benchmarks.py`.

### River + XGBoost comparison

```bash
# Download datasets
python datasets/download.py

# River (streaming Python)
python comparison/river/bench_river.py

# XGBoost / LightGBM (batch with sliding window)
python comparison/xgboost/bench_xgb.py

# Collect + aggregate
python comparison/collect_results.py
```

---

## Real-dataset results (SGBT)

### Methodology

- Protocol: prequential (test-then-train), single pass
- Metrics: Accuracy, Cohen's Kappa, Macro-F1 (Covertype only), samples/sec
- Batch model baseline: sliding-window retraining (window sizes 500, 1000, 5000)

### Electricity (45K samples, binary, concept drift)

Source: [OpenML #151](https://www.openml.org/d/151) — electricity demand in New South Wales,
Australia. Binary (price up/down). Real-world concept drift.

**Streaming models**

| Model | Library | Accuracy | Kappa | Throughput (s/s) |
|---|---|---|---|---|
| SGBT 25t d4 (lr=0.05) | irithyll | 0.7159 | 0.3709 | 67,063 |
| SGBT 50t d6 (lr=0.05) | irithyll | 0.8188 | 0.6155 | 16,347 |
| SGBT 50t d6 (lr=0.1) | irithyll | 0.8583 | 0.7041 | 19,011 |
| SGBT 100t d6 (lr=0.1) | irithyll | 0.8852 | 0.7619 | 8,184 |
| hoeffding_tree | River | 0.7956 | 0.5779 | 12,029 |
| hoeffding_adaptive_tree | River | 0.8293 | 0.6476 | 3,357 |
| arf_n10 | River | 0.8858 | 0.7652 | 534 |
| arf_n25 | River | 0.8913 | 0.7767 | 200 |

**Batch models (sliding-window)**

| Model | Library | Window | Accuracy | Kappa | Throughput (s/s) |
|---|---|---|---|---|---|
| xgb_w500 | XGBoost | 500 | 0.7637 | 0.5169 | 1,997 |
| xgb_w1000 | XGBoost | 1000 | 0.7542 | 0.4960 | 2,058 |
| xgb_w5000 | XGBoost | 5000 | 0.7053 | 0.4111 | 2,134 |
| lgbm_w500 | LightGBM | 500 | 0.7632 | 0.5155 | 1,434 |
| lgbm_w1000 | LightGBM | 1000 | 0.7572 | 0.5026 | 1,448 |
| lgbm_w5000 | LightGBM | 5000 | 0.7107 | 0.4234 | 1,483 |

---

### Airlines (539K samples, binary, large-scale)

Source: [OpenML #1169](https://www.openml.org/d/1169) — US flight delay records.
Binary (delayed/on-time). Temporal and seasonal drift.

**Streaming models**

| Model | Library | Accuracy | Kappa | Throughput (s/s) |
|---|---|---|---|---|
| SGBT 50t d6 (lr=0.05) | irithyll | 0.6253 | 0.1802 | 9,222 |
| SGBT 50t d6 (lr=0.1) | irithyll | 0.6488 | 0.2449 | 9,054 |
| SGBT 100t d6 (lr=0.1) | irithyll | 0.6558 | 0.2684 | 4,094 |
| hoeffding_tree | River | 0.6383 | 0.2429 | 9,100 |
| hoeffding_adaptive_tree | River | 0.6348 | 0.2413 | 3,067 |
| arf_n10 | River | 0.6565 | 0.2895 | 448 |
| arf_n25 | River | 0.6675 | 0.3102 | 171 |

**Batch models (sliding-window)**

| Model | Library | Window | Accuracy | Kappa | Throughput (s/s) |
|---|---|---|---|---|---|
| xgb_w500 | XGBoost | 500 | 0.6216 | 0.2287 | 1,980 |
| xgb_w1000 | XGBoost | 1000 | 0.6299 | 0.2457 | 2,057 |
| xgb_w5000 | XGBoost | 5000 | 0.6317 | 0.2501 | 2,131 |
| lgbm_w500 | LightGBM | 500 | 0.6352 | 0.2532 | 1,425 |
| lgbm_w1000 | LightGBM | 1000 | 0.6460 | 0.2751 | 1,429 |
| lgbm_w5000 | LightGBM | 5000 | 0.6439 | 0.2738 | 1,419 |

---

### Covertype (581K samples, 7-class, high-dimensional)

Source: [OpenML #150](https://www.openml.org/d/150) — forest cover types in Roosevelt National
Forest. 54 features (10 quantitative, 44 binary). 7-class, class imbalance.

**Streaming models**

| Model | Library | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |
|---|---|---|---|---|---|
| SGBT 50t d6 (lr=0.05) | irithyll | 0.8938 | 0.8265 | 0.8173 | 591 |
| SGBT 50t d6 (lr=0.1) | irithyll | 0.9247 | 0.8780 | 0.8710 | 584 |
| SGBT 100t d6 (lr=0.1) | irithyll | 0.9456 | 0.9122 | 0.9098 | 200 |
| hoeffding_tree | River | 0.7655 | 0.6186 | --- | 2,134 |
| hoeffding_adaptive_tree | River | 0.7731 | 0.6309 | --- | 687 |
| arf_n10 | River | 0.8727 | 0.7921 | --- | 461 |
| arf_n25 | River | 0.8858 | 0.8133 | --- | 207 |

**Batch models (sliding-window)**

| Model | Library | Window | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |
|---|---|---|---|---|---|---|
| xgb_w500 | XGBoost | 500 | 0.4988 | 0.2312 | --- | 2,176 |
| xgb_w1000 | XGBoost | 1000 | 0.4753 | 0.1735 | --- | 2,143 |
| xgb_w5000 | XGBoost | 5000 | 0.5931 | 0.3232 | --- | 2,079 |
| lgbm_w500 | LightGBM | 500 | 0.4596 | 0.1868 | --- | 1,434 |
| lgbm_w1000 | LightGBM | 1000 | 0.4856 | 0.1905 | --- | 1,443 |
| lgbm_w5000 | LightGBM | 5000 | 0.5979 | 0.3577 | --- | 1,428 |

---

### Summary: best per library

| Dataset | irithyll best | River best | XGBoost best | LightGBM best |
|---|---|---|---|---|
| Electricity | 88.5% acc, 8K s/s | 89.1% acc, 200 s/s | 76.4% acc, 2K s/s | 76.3% acc, 1.5K s/s |
| Airlines | 65.6% acc, 4K s/s | 66.8% acc, 171 s/s | 63.2% acc, 2.1K s/s | 64.6% acc, 1.4K s/s |
| Covertype | 94.6% acc, 200 s/s | 88.6% acc, 207 s/s | 59.3% acc, 2.2K s/s | 59.8% acc, 1.4K s/s |

---

## Visualization

The `scripts/plot_benchmarks.py` script reads criterion JSON output and generates
two plots in `marketing/benchmarks/`:

| Plot | File | What it shows |
|---|---|---|
| Pareto frontier | `pareto.png` | Accuracy vs throughput per model, Pareto frontier overlay |
| Dataset comparison | `dataset_comparison.png` | Side-by-side bar chart across 28 real-world datasets |

### How to generate

Run the benchmark suite first to populate `target/criterion/`:

```bash
cargo bench --bench real_world_bench
cargo bench --bench real_dataset_bench
```

Then generate plots:

```bash
# Dependencies: numpy, matplotlib (Python 3.10+)
pip install numpy matplotlib

# Generate both plots (defaults: reads target/criterion/, writes marketing/benchmarks/)
python scripts/plot_benchmarks.py

# Custom paths
python scripts/plot_benchmarks.py \
    --criterion-dir /path/to/target/criterion \
    --output-dir marketing/benchmarks \
    --dpi 150
```

The generated PNGs in `marketing/benchmarks/` are committed to the repository and
kept up to date with each release sweep.

---

## Hardware

Results collected on:

- **CPU:** AMD Ryzen 5 5500 (6C/12T, 3.6 GHz base)
- **RAM:** 16 GB DDR4 3200 MHz (2x8 GB)
- **OS:** Windows 11 Home 10.0.26200
- **Rust:** stable (single-threaded, release profile)

---

## Limitations

1. **Hyperparameter sensitivity.** Models were tuned with reasonable but not exhaustive
   search. Better configurations may exist for all models.

2. **Apples-to-oranges on batch vs streaming.** Batch models (XGBoost, LightGBM) with
   sliding-window retraining have a fundamentally different compute profile than true
   streaming learners. Throughput comparisons across paradigms should be interpreted
   accordingly.

3. **No ensemble/pipeline comparisons.** River supports rich preprocessing + drift
   detection + model pipelines. This comparison uses base models only.

4. **Single machine, single thread.** Libraries with parallel prediction (e.g.,
   LightGBM) may perform differently under multi-threaded settings.

5. **irithyll results are from Rust `cargo bench`.** River and XGBoost results are
   from Python. The runtime difference is a real factor in throughput but not in
   accuracy/kappa.

6. **Dataset selection.** These three datasets are well-known streaming ML benchmarks
   but do not capture the full range of streaming ML challenges. Results may differ
   on other data distributions, feature types, or class structures.

7. **iai-callgrind Linux-only.** The instruction-count regression bench runs under
   valgrind and is therefore unavailable on Windows and macOS. The bench source still
   compiles on those platforms (so CI catches API drift) but cannot be executed.

