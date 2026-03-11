# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-11

### Added

- **Streaming Adaptation** -- EWMA leaf decay (`leaf_half_life`), time-based proactive
  tree replacement (`max_tree_samples`), and EFDT-inspired split re-evaluation at
  max-depth leaves (`split_reeval_interval`). All three are opt-in via `SGBTConfigBuilder`
  and backward-compatible with existing serialized models (`#[serde(default)]`).
- **Lazy histogram decay** -- O(1) amortized forward decay for histogram bins. Tracks a
  scalar `decay_scale` per histogram and defers the O(n_bins) materialization pass to
  split evaluation time. Mathematically exact; automatic renormalization prevents f64
  underflow at ~16K samples per leaf.
- **SIMD histogram acceleration** (`simd` feature) -- AVX2 intrinsics for histogram
  gradient/hessian summation, gated behind the `simd` feature flag.
- **Neural leaf models** (`neural-leaves` feature) -- experimental MLP-based leaf
  predictions as an alternative to closed-form constant leaves.
- **Apache Arrow integration** (`arrow` feature) -- `train_from_record_batch()`,
  `predict_from_record_batch()`, `record_batch_to_samples()` for columnar data
  interop. Hardened NaN/Inf filtering.
- **Parquet I/O** (`parquet` feature) -- `train_from_parquet()` for bulk training
  directly from Parquet files.
- **ONNX export** (`onnx` feature) -- `export_onnx()` and `save_onnx()` for
  interoperability with ONNX-compatible runtimes.
- **Parallel training** (`parallel` feature) -- `ParallelSGBT` using Rayon for
  data-parallel tree training across boosting steps.
- **Model serialization** -- `save_model()` / `load_model()` for JSON checkpoint/restore
  with `ModelState` snapshots. `LossType` tags enable loss function reconstruction.
  Optional bincode backend via `serde-bincode` feature.
- **Deterministic seeding** -- `SGBTConfig::seed` ensures reproducible results across
  feature subsampling and variant stochastic decisions.
- **Feature importance** -- `FeatureImportance` tracker accumulates split gain per
  feature across the ensemble.
- **Online classification metrics** -- `ClassificationMetrics` with incremental accuracy,
  precision, recall, F1, and log loss. `MetricSet` combines regression and classification.
- **Async streaming** -- `AsyncSGBT` with tokio channels, bounded backpressure,
  `Predictor` for concurrent read-only access, and `PredictionStream` adapter.
- **Multi-class support** -- `MulticlassSGBT` with one-vs-rest committees and softmax
  normalization.
- **Three SGBT variants** -- Standard, Skip (SGBT-SK), and MultipleIterations (SGBT-MI)
  per Gunasekara et al. (2024).
- **Three drift detectors** -- Page-Hinkley Test, ADWIN, and DDM with configurable
  parameters via `DriftDetectorType`.
- **Pluggable loss functions** -- `SquaredLoss`, `LogisticLoss`, `SoftmaxLoss`,
  `HuberLoss`, and the `Loss` trait for custom implementations.
- **Three binning strategies** -- `UniformBinning`, `QuantileBinning`, and optional
  `KMeansBinning` for histogram construction.
- **Hoeffding tree splitting** -- statistically-grounded split decisions using
  histogram-binned gradient statistics with configurable confidence (`delta`).
- **XGBoost-style regularization** -- L2 (`lambda`) and minimum gain (`gamma`)
  regularization on leaf weights and split decisions.
- **Five examples** -- `basic_regression`, `classification`, `async_ingestion`,
  `custom_loss`, `drift_detection`, `model_checkpointing`, `streaming_metrics`.
- **Four Criterion benchmarks** -- `histogram_bench`, `tree_bench`, `ensemble_bench`,
  `parallel_bench`.
- **Property-based tests** -- proptest-based correctness verification.
- **GitHub Actions CI** -- test matrix across stable/beta/nightly Rust on Linux, macOS,
  and Windows. Includes clippy, rustfmt, doc generation, and MSRV verification.

## [0.1.0] - 2026-03-10

Initial development release. Core SGBT algorithm with Hoeffding trees, histogram
binning, drift detection, and online metrics.

[1.0.0]: https://github.com/evilrat420/irithyll/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/evilrat420/irithyll/releases/tag/v0.1.0
