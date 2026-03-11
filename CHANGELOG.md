# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-03-11

### Added

- **Expectile loss** -- `ExpectileLoss { tau }` for asymmetric L2 regression. Natively
  compatible with second-order boosting (well-defined positive Hessian for all tau).
  `tau > 0.5` penalizes under-prediction; `tau < 0.5` penalizes over-prediction.
- **Quantile (pinball) loss** -- `QuantileLoss { tau }` for conditional quantile
  regression. Uses hessian=1 pseudo-Huber trick for streaming tree compatibility,
  matching the approach used by LightGBM and XGBoost internally.
- **Rolling metrics** -- `RollingRegressionMetrics` and `RollingClassificationMetrics`
  with sliding window + revert pattern. Sliding Welford for O(1) R-squared over
  finite windows. MAE, MSE, RMSE, accuracy, precision, recall, F1, log loss.
- **EWMA metrics** -- `EwmaRegressionMetrics` and `EwmaClassificationMetrics` for
  exponentially weighted metric tracking. O(1) memory, no buffer needed.
- **Adaptive Conformal Intervals** -- `AdaptiveConformalInterval` implements ACI
  (Gibbs & Candes, 2021) for distribution-free prediction intervals under drift.
  Adapts miscoverage rate online to maintain target coverage probability.
- **Feature importance drift monitor** -- `ImportanceDriftMonitor` feeds per-feature
  |SHAP| values into drift detectors (PHT, ADWIN, or DDM). Detects reasoning shifts
  before accuracy drops. Configurable sample rate for amortized SHAP cost. No
  existing open-source library offers this capability.
- **Bincode serialization** -- `serde-bincode` feature flag now functional with
  `to_bincode`, `from_bincode`, `save_model_bincode`, `load_model_bincode` for
  compact binary model persistence (typically 3-5x smaller than JSON).

## [3.0.0] - 2026-03-11

### Added

- **TreeSHAP explanations** -- path-dependent TreeSHAP (Lundberg et al., 2020) for
  per-feature SHAP contributions. `SGBT::explain(features)` returns `ShapValues` with
  the invariant `base_value + sum(values) == predict(features)`. Named explanations
  via `explain_named()` when `feature_names` are configured.
- **StreamingShap** -- online running-mean |SHAP| tracker for real-time feature
  importance monitoring without storing past predictions.
- **Named features** -- `SGBTConfig::builder().feature_names(vec!["price", "volume"])`
  enables `named_feature_importances()` and `explain_named()`. Duplicate names are
  rejected at build time.
- **Multi-target regression** -- `MultiTargetSGBT` wraps T independent `SGBT<L>` models,
  one per target dimension. `train_one(features, targets)` and `predict(features) -> Vec<f64>`.
  Custom loss via `with_loss()`.
- **Drift detector state serialization** -- `DriftDetector::serialize_state()` and
  `restore_state()` preserve Page-Hinkley, ADWIN, and DDM internal state across
  save/load cycles. No more spurious drift after checkpoint restore.
- **PyO3 Python bindings** -- `irithyll-python` workspace crate providing
  `StreamingGBT`, `StreamingGBTConfig`, `ShapExplanation`, and `MultiTargetGBT` as
  Python classes. GIL-released train/predict, numpy zero-copy, JSON save/load.

### Performance

- **Vec leaf states** -- replaced `HashMap<u32, LeafState>` with `Vec<Option<LeafState>>`
  indexed by NodeId. Eliminates hashing overhead on the hot path for dense node indices.

## [2.0.0] - 2026-03-11

### Breaking Changes

- **Generic Loss parameter** -- `SGBT<L: Loss = SquaredLoss>` replaces the old boxed
  `Box<dyn Loss>` design. Loss gradient/hessian calls are now monomorphized and inlined
  by the compiler. Use `SGBT::with_loss(config, LogisticLoss)` instead of passing
  `Box::new()`. `DynSGBT = SGBT<Box<dyn Loss>>` is provided for dynamic dispatch.
  Cascades to `ParallelSGBT<L>`, `AsyncSGBT<L>`, `MulticlassSGBT<L>`, and `Predictor<L>`.
- **Observation trait** -- `train_one()` now accepts `&impl Observation` instead of
  `&Sample`. Zero-copy training via `SampleRef<'a>` (borrows `&[f64]`). Tuple impls
  for `(&[f64], f64)` and `(Vec<f64>, f64)` enable quick usage without constructing
  `Sample`. `train_one_slice()` is removed (subsumed by `Observation`).
- **Structured errors** -- `ConfigError` is now a sub-enum with `OutOfRange` and
  `Invalid` variants carrying `param`, `constraint`, and `value` fields. Replaces
  `InvalidConfig(String)`.
- **Auto LossType** -- `Loss` trait now requires `fn loss_type(&self) -> Option<LossType>`.
  `save_model()` auto-detects loss type from the model, no more manual `LossType` tag.
- **MulticlassSGBT::new() returns Result** -- no longer panics on `n_classes < 2`.

### Added

- **`SampleRef<'a>`** -- borrowed observation type for zero-allocation training.
- **`Observation` trait** -- unified interface with default `weight() -> 1.0`. Implemented
  for `Sample`, `SampleRef`, `(&[f64], f64)`, `(Vec<f64>, f64)`.
- **`Clone` for `SGBT<L>`** -- deep clone including all tree state, drift detectors, and
  leaf accumulators. Requires `L: Clone`. Also implemented for `ParallelSGBT<L>`.
- **`DriftDetector::clone_boxed()`** -- deep clone preserving internal state (unlike
  `clone_fresh()` which resets).
- **`BinnerKind` enum** -- replaces `Box<dyn BinningStrategy>` per-feature per-leaf with
  stack-allocated enum dispatch. Eliminates N_features heap allocations per new leaf.
- **`PartialEq` for `SGBTConfig`** and `DriftDetectorType`.
- **`Display` for `SGBTConfig`**, `SGBTVariant`, `DriftSignal`, `Sample`.
- **`From` conversions** -- `From<(Vec<f64>, f64)>` and `From<(&[f64], f64)>` for Sample.
- **`Debug` for `MulticlassSGBT`**.
- **Throughput benchmarks** -- `throughput_bench` and `scaling_bench` for regression
  tracking across configurations.

### Performance

- **Bitset feature mask** -- O(1) membership testing via `Vec<u64>` bitset, replacing
  O(n) `Vec::contains()`. The fallback fill loop in `generate_feature_mask()` drops
  from O(n^2) to O(n). Single `u64` for <=64 features.
- **Enum BinningStrategy dispatch** -- `BinnerKind` enum eliminates heap allocation and
  vtable indirection for histogram binners.
- **Pre-allocated train_counts buffer** -- `ParallelSGBT` reuses a buffer instead of
  allocating a fresh `Vec<usize>` per `train_one()` call.
- **Monomorphized loss** -- generic `L: Loss` parameter enables the compiler to inline
  gradient/hessian computations, eliminating vtable dispatch in the hot path.
- **Measured improvement** -- 8.5% throughput gain on 100-step 20-feature training
  workloads (Criterion benchmark).

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

[4.0.0]: https://github.com/evilrat420/irithyll/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/evilrat420/irithyll/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/evilrat420/irithyll/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/evilrat420/irithyll/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/evilrat420/irithyll/releases/tag/v0.1.0
