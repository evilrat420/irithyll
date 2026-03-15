# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.0.0] - 2026-03-15

### Added

- **Empirical σ estimation** (`ScaleMode::Empirical`, now default) -- replaces
  the scale tree chain with an EWMA of squared prediction errors:
  `σ = sqrt(ewma_sq_err)`.  Always calibrated, zero tuning, O(1) compute.
  The scale tree chain was frozen in practice (scale gradients too weak for
  Hoeffding trees to split), making σ-modulated learning a no-op.  Empirical σ
  is self-calibrating by definition: high recent errors → σ large → location LR
  scales up → model corrects faster.
- `ScaleMode` enum (`Empirical` / `TreeChain`) on `SGBTConfig` for choosing
  the σ estimation strategy.  `ScaleMode::TreeChain` preserves the full
  dual-chain NGBoost behavior for users who need feature-conditional uncertainty.
- `empirical_sigma_alpha` config parameter (default `0.01`) -- controls EWMA
  adaptation speed for empirical σ.
- `empirical_sigma()` method on `DistributionalSGBT` -- returns current σ.
- `scale_mode()` method on `DistributionalSGBT`.
- `ModelDiagnostics` now includes `empirical_sigma`, `scale_mode`,
  `scale_trees_active`, and separate `location_trees` / `scale_trees` vectors
  for independent inspection.
- `ScaleMode` re-exported at crate root.
- ASCII art header in README.
- **DistributionalSGBT diagnostics** -- three new diagnostic capabilities:
  - `ModelDiagnostics` struct with per-tree summaries (leaf count, max depth,
    samples seen, leaf weight statistics, split features) and global feature
    split counts. Access via `model.diagnostics()`.
  - `DecomposedPrediction` struct with `predict_decomposed(features)` -- returns
    per-tree additive contributions for both location and scale ensembles.
    Includes `mu()`, `log_sigma()`, `sigma()` reconstruction methods.
  - `feature_importances()` and `feature_importances_split()` -- normalized
    split-gain-weighted feature importance vectors, with optional separate
    location/scale views.
- `TreeDiagnostic`, `ModelDiagnostics`, `DecomposedPrediction` re-exported at
  crate root.

### Changed

- **BREAKING:** `DistributionalSGBT` defaults to empirical σ (`ScaleMode::Empirical`).
  Users who relied on the dual-chain behavior must explicitly set
  `scale_mode(ScaleMode::TreeChain)` in their config builder.
- `ModelDiagnostics` struct has new fields: `location_trees`, `scale_trees`,
  `empirical_sigma`, `scale_mode`, `scale_trees_active`.

## [6.5.1] - 2026-03-15

### Changed

- Made AdaGrad optional on `LinearLeafModel` via `use_adagrad: bool` field on
  `LeafModelType::Linear`. Default is `false` (plain Newton-scaled SGD).

## [6.5.0] - 2026-03-14

### Added

- **Adaptive leaf promotion** -- leaves start as constant (zero overhead) and
  auto-promote to a trainable model when the Hoeffding bound confirms it is
  statistically superior. Uses the tree's existing `delta` parameter -- no
  arbitrary thresholds. New `LeafModelType::Adaptive { promote_to }` variant.
- **Optional AdaGrad optimization for linear leaves** -- `use_adagrad: true`
  enables per-weight squared gradient accumulators, giving each feature its
  own adaptive learning rate. Default is `false` (plain Newton-scaled SGD).
- **Exponential forgetting** -- optional `decay` parameter on `Linear` and `MLP`
  leaf models. Applies exponential weight decay before each update, giving the
  model a finite memory horizon for non-stationary streams.
- **Warm-start on split** -- new `LeafModel::clone_warm()` trait method. When a
  leaf splits, child leaves inherit the parent's learned weights (resetting
  optimizer state), converging faster than starting from scratch.
- Integration tests for adaptive leaves, decay leaves, warm-start cloning.

### Changed

- `LeafModelType::Linear` now includes `decay: Option<f64>` and
  `use_adagrad: bool` fields (both serde-defaulted). `LeafModelType::MLP`
  includes `decay: Option<f64>`. Existing configs deserialize correctly.
- `LeafModelType::create()` now takes an additional `delta: f64` parameter
  for adaptive leaf construction.
- Tree cloning now uses `clone_warm()` instead of `clone_fresh()`, preserving
  learned leaf model weights across clone operations.
- Child leaves created during splits warm-start from the parent's model
  instead of initializing from scratch.

## [6.4.0] - 2026-03-14

### Added

- **Pluggable leaf models** -- leaves can now use trainable prediction models
  instead of constant weights. Three variants via `LeafModelType`:
  - `ClosedForm` (default) -- standard constant leaf weight, zero overhead.
  - `Linear { learning_rate }` -- per-leaf online ridge regression. Each leaf
    learns a local linear surface `w . x + b` via Newton-scaled gradient descent.
    Significantly improves accuracy for low-depth trees (depth 2-4).
  - `MLP { hidden_size, learning_rate }` -- per-leaf single-hidden-layer neural
    network with ReLU activation and backpropagation.
- `LeafModelType` re-exported at crate root for ergonomic access.
- `SGBTConfig::builder().leaf_model_type(...)` for ensemble-level configuration.
- Full integration across all ensemble variants: SGBT, DistributionalSGBT,
  ParallelSGBT.

### Changed

- Leaf model module (`tree::leaf_model`) is now always compiled -- previously
  gated behind the `neural-leaves` feature flag. No external dependencies added.
- Removed all em-dash characters from source code for consistent ASCII encoding.

## [6.3.0] - 2026-03-13

### Added

- **Kernel Recursive Least Squares (KRLS)** -- non-linear streaming regression in
  reproducing kernel Hilbert space. Implements Engel, Mannor & Meir (2004) with
  ALD (Approximate Linear Dependency) sparsification for automatic dictionary
  pruning. Configurable budget caps dictionary size; once full, new samples fall
  through to weight-only updates. Three kernel implementations: `RBFKernel`,
  `PolynomialKernel`, `LinearKernel` via the object-safe `Kernel` trait. O(N²)
  per sample where N ≤ budget.
- **CCIPCA** (Candid Covariance-free Incremental PCA) -- streaming dimensionality
  reduction via Weng, Zhang & Hwang (2003). Incrementally estimates leading
  eigenvectors of the covariance matrix in O(kd) time per sample without ever
  forming the covariance matrix. Configurable amnestic parameter for non-stationary
  streams. Implements `StreamingPreprocessor` for pipeline integration.
- **RLS confidence intervals** -- `RecursiveLeastSquares` now provides
  `prediction_variance()`, `prediction_std()`, `predict_interval(z)`, and
  `noise_variance()` for lightweight uncertainty quantification. Variance combines
  noise estimate (EWMA of squared residuals) with parameter uncertainty through
  the P matrix.
- **DistributionalSGBT serialization** -- `to_distributional_state()`,
  `from_distributional_state()` with full round-trip fidelity including drift
  detector state, rolling sigma, and alternate trees. `save_distributional_model()`
  / `load_distributional_model()` and bincode equivalents.
- Factory functions: `krls()`, `ccipca()` for ergonomic construction.

### Fixed

- Broken intra-doc links for `DistributionalModelState` and missing
  `StreamingLearner` import in `Pipeline` doc example.

## [6.2.0] - 2026-03-13

### Added

- **Pipeline composition** -- `PipelineBuilder` for chaining `StreamingPreprocessor`
  steps with a terminal `StreamingLearner`. `Pipeline` itself implements
  `StreamingLearner`, enabling recursive composition and stacking. Training updates
  preprocessor statistics; prediction uses frozen transforms.
- **`StreamingPreprocessor` trait** -- object-safe trait for streaming feature
  transformers with `update_and_transform()` (training) and `transform()` (prediction).
- **`IncrementalNormalizer`** -- Welford online standardization implementing
  `StreamingPreprocessor`. Zero-mean, unit-variance with lazy feature count
  initialization and configurable variance floor.
- **`OnlineFeatureSelector`** -- EWMA importance tracking with dynamic feature
  masking after configurable warmup period.
- **`AdaptiveSGBT`** -- SGBT wrapper with pluggable learning rate schedulers.
  Five built-in schedulers: `ConstantLR`, `LinearDecayLR`, `ExponentialDecayLR`,
  `CosineAnnealingLR`, `PlateauLR`.
- Factory functions for ergonomic model construction: `sgbt()`, `linear()`,
  `rls()`, `gaussian_nb()`, `mondrian()`, `normalizer()`, `pipe()`,
  `adaptive_sgbt()`.

## [6.1.1] - 2026-03-13

### Added

- **σ-modulated learning rate** for `DistributionalSGBT` -- when `uncertainty_modulated_lr`
  is enabled, the location (μ) ensemble's learning rate is scaled by
  `sigma_ratio = current_sigma / rolling_sigma_mean`. The model learns μ faster during
  high-uncertainty regimes and conserves during stable periods. The scale (σ) ensemble
  always trains at the unmodulated base rate to prevent positive feedback loops.
  Rolling sigma mean uses slow EWMA (alpha = 0.001), initialized from initial targets std.
- `predict_distributional()` method on `DistributionalSGBT` returning `(mu, sigma, sigma_ratio)`
  for real-time monitoring of the effective learning rate.
- `rolling_sigma_mean()` and `is_uncertainty_modulated()` accessors on `DistributionalSGBT`.
- `uncertainty_modulated_lr()` builder method on `SGBTConfigBuilder`.

### Fixed

- Rustdoc warnings causing CI Documentation job failure: unresolved `StreamingLearner` link,
  private `MondrianTree` link, and redundant explicit link targets in learners module.

## [6.1.0] - 2026-03-12

### Added

- **StreamingLearner trait** -- object-safe `StreamingLearner` trait for polymorphic model
  composition. `train_one()`, `predict()`, `n_samples_seen()`, `reset()` with default
  `train()` (unit weight) and `predict_batch()`. `SGBTLearner<L>` adapter wraps any
  `SGBT<L>` into the trait. Supports `Box<dyn StreamingLearner>` for runtime-polymorphic
  stacking ensembles.
- **Mixture of Experts** -- `MoESGBT` with K specialist SGBT experts and a learned
  linear softmax gating network. Gate updated via online SGD cross-entropy toward the
  best-performing expert. Soft gating (all experts weighted) and Hard gating (top-k only).
  Methods: `predict_with_gating()`, `gating_probabilities()`, `expert_losses()`.
- **Model stacking meta-learner** -- `StackedEnsemble` combining heterogeneous
  `Box<dyn StreamingLearner>` base learners through a trainable meta-learner. Temporal
  holdout prevents leakage: base predictions collected before training bases. Optional
  feature passthrough. Itself implements `StreamingLearner` for recursive stacking.
- **Learning rate scheduling** -- `LRScheduler` trait with 5 implementations:
  `ConstantLR`, `LinearDecayLR`, `ExponentialDecayLR`, `CosineAnnealingLR`, `PlateauLR`.
  `SGBT::set_learning_rate()` method for external scheduler integration.
- **Streaming linear model** -- `StreamingLinearModel` with SGD and pluggable
  regularization: `None`, `Ridge` (L2 weight decay), `Lasso` (L1 proximal/soft-threshold),
  `ElasticNet` (combined L1+L2). Lazy feature initialization. Implements `StreamingLearner`.
- **Gaussian Naive Bayes** -- `GaussianNB` incremental classifier with per-class Welford
  mean/variance tracking. Weighted samples, automatic class discovery, sklearn-style
  `var_smoothing`. `predict_proba()` and `predict_log_proba()` for probability output.
  Implements `StreamingLearner`.
- **Mondrian Forest** -- `MondrianForest` online random forest with arena-based SoA storage.
  Feature-range-proportional random splits, configurable via `MondrianForestConfig` builder
  (n_trees, max_depth, lifetime, seed). Implements `StreamingLearner`.
- **Recursive Least Squares** -- `RecursiveLeastSquares` exact streaming OLS via
  Sherman-Morrison matrix inversion lemma. O(d²) per sample, forgetting factor for
  non-stationary environments. Implements `StreamingLearner`.
- **Streaming polynomial regression** -- `StreamingPolynomialRegression` wrapping RLS
  with online polynomial feature expansion up to arbitrary degree.
  Implements `StreamingLearner`.
- **Locally weighted regression** -- `LocallyWeightedRegression` Nadaraya-Watson kernel
  regression over a fixed-capacity circular buffer with Gaussian kernel weighting.
  Implements `StreamingLearner`.
- **Incremental normalizer** -- `IncrementalNormalizer` with Welford online mean/variance
  per feature. `update()`, `transform()`, `update_and_transform()`, `transform_in_place()`.
  Lazy feature count initialization.
- **Online feature selector** -- `OnlineFeatureSelector` with EWMA importance tracking
  and configurable `keep_fraction`. Masks low-importance features after warmup period.

## [6.0.0] - 2026-03-12

### Added

- **NGBoost distributional output** -- `DistributionalSGBT` maintains two independent
  streaming tree ensembles (location + scale) to output a full Gaussian N(mu, sigma^2)
  predictive distribution. Scale parameterized in log-space for positivity. Gradients
  derived from Gaussian NLL (Duan et al., 2020). Enables per-prediction uncertainty
  quantification with `predict() -> GaussianPrediction { mu, sigma, log_sigma }` and
  `predict_interval(alpha)` for calibrated prediction intervals.
- **Error-weighted sample importance** -- streaming AdaBoost-style gradient reweighting.
  Samples with high absolute error (relative to a rolling EWMA mean) receive amplified
  gradients, forcing the ensemble to focus on hard examples. Configurable via
  `SGBTConfig::builder().error_weight_alpha(0.01)`. Weight capped at 10x to prevent
  instability.
- **Quality-based tree pruning** -- per-step EWMA tracking of |marginal contribution|.
  Trees contributing below a threshold for `patience` consecutive samples are reset,
  eliminating dead wood from extinct regimes. Configurable via `quality_prune_alpha`,
  `quality_prune_threshold`, `quality_prune_patience` in SGBTConfig.
- **Half-Space Trees anomaly detection** -- `HalfSpaceTree` implements the HS-Tree
  algorithm (Tan, Ting & Liu, 2011) for streaming anomaly detection. Random axis-aligned
  partitions with mass-based scoring, window rotation for reference/latest profiles.
  `score_and_update()` returns normalized anomaly scores in [0, 1]. Configurable via
  `HSTConfig` with n_trees, max_depth, window_size, and explicit feature ranges.
- **Serde state for v6 fields** -- `ModelState` now persists `rolling_mean_error`,
  `contribution_ewma`, and `low_contrib_count` across save/load cycles with
  `#[serde(default)]` for backward compatibility with v5 checkpoints.

## [5.1.0] - 2026-03-12

### Added

- **Gradient clipping per leaf** -- `SGBTConfig::builder().gradient_clip_sigma(3.0)` clips
  gradient/hessian outliers beyond N standard deviations, preventing explosive leaf weights
  from noisy streams.
- **`predict_with_confidence()`** -- returns `(prediction, variance)` using per-leaf
  variance estimates. Enables lightweight uncertainty quantification without the full
  distributional model.
- **Monotonic constraints** -- `SGBTConfig::builder().monotone_constraints(vec![1, -1, 0])`
  enforces feature-level monotonicity during tree growth. +1 = increasing, -1 = decreasing,
  0 = unconstrained.
- **`train_batch_with_callback()`** -- train on a slice of samples with a user-supplied
  callback invoked every N samples for logging, metrics, checkpointing, etc.
- **`train_batch_subsampled()`** -- train on a batch with probabilistic subsampling per
  sample, useful for large-batch streaming scenarios.

## [5.0.0] - 2026-03-11

### Added

- **Categorical feature handling** -- Fisher optimal binary partitioning for categorical
  splits in Hoeffding trees. Categories are sorted by gradient/hessian ratio, then the
  existing XGBoostGain evaluator finds the best contiguous partition. Split routing uses
  a `u64` bitmask (up to 64 categories per feature). Configure via
  `SGBTConfig::builder().feature_types(vec![FeatureType::Categorical, FeatureType::Continuous])`.
  `CategoricalBinning` strategy creates one bin per observed distinct value.
- **BaggedSGBT (Oza online bagging)** -- `BaggedSGBT` wraps M independent `SGBT<L>`
  instances with Poisson(1) weighting per bag per sample. Implements the SGB(Oza)
  algorithm from Gunasekara et al. (2025) for variance reduction in streaming gradient
  boosted regression. Final prediction is the mean across all bags. Deterministic
  seeding with xorshift64 PRNG and Knuth's Poisson sampling.
- **Non-crossing multi-quantile regression** -- `QuantileRegressorSGBT` trains K
  independent `SGBT<QuantileLoss>` models (one per quantile level) with PAVA (Pool
  Adjacent Violators Algorithm) post-prediction to enforce monotonicity. O(K) isotonic
  regression guarantees `predict(tau_i) <= predict(tau_j)` for `tau_i < tau_j`.
  `predict_interval(features, lower_tau, upper_tau)` returns calibrated prediction
  intervals. Auto-sorts and validates quantile levels at construction time.
- **`FeatureType` enum** -- `FeatureType::Continuous` and `FeatureType::Categorical`
  propagated from `SGBTConfig` through `TreeConfig` to leaf state binner creation.
  Backward-compatible: `feature_types` defaults to `None` (all continuous).
- **Categorical bitmask routing in `TreeArena`** -- `categorical_mask: Vec<Option<u64>>`
  stored per node. `split_leaf_categorical()` sets bitmask; `route_to_leaf()` dispatches
  via `(mask >> cat_val) & 1`. Serializable with `#[serde(default)]` for backward compat.

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

[6.4.0]: https://github.com/evilrat420/irithyll/compare/v6.3.0...v6.4.0
[6.3.0]: https://github.com/evilrat420/irithyll/compare/v6.2.0...v6.3.0
[6.2.0]: https://github.com/evilrat420/irithyll/compare/v6.1.1...v6.2.0
[6.1.1]: https://github.com/evilrat420/irithyll/compare/v6.1.0...v6.1.1
[6.1.0]: https://github.com/evilrat420/irithyll/compare/v6.0.0...v6.1.0
[6.0.0]: https://github.com/evilrat420/irithyll/compare/v5.1.0...v6.0.0
[5.1.0]: https://github.com/evilrat420/irithyll/compare/v5.0.0...v5.1.0
[5.0.0]: https://github.com/evilrat420/irithyll/compare/v4.0.0...v5.0.0
[4.0.0]: https://github.com/evilrat420/irithyll/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/evilrat420/irithyll/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/evilrat420/irithyll/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/evilrat420/irithyll/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/evilrat420/irithyll/releases/tag/v0.1.0
