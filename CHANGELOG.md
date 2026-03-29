# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [9.8.3] - 2026-03-29

### Added

- **Adaptive config on `StreamingLearner` trait** — three new default methods:
  - `diagnostics_array()` — raw `[f64; 5]` diagnostic signals (residual alignment,
    regularization sensitivity, depth sufficiency, effective DOF, uncertainty).
    Overridden with real values in 17 models.
  - `adjust_config(lr_multiplier, lambda_delta)` — smooth runtime config mutation.
    Implemented for SGBT, DistributionalSGBT, TTT, KAN, RLS, KRLS, Linear, Mondrian.
  - `apply_structural_change(depth_delta, steps_delta)` — queued structural changes
    at tree replacement boundaries.
  - `replacement_count()` — total internal model replacements for boundary detection.
- `set_lambda()`, `set_max_depth()`, `set_n_steps()` on SGBT and DistributionalSGBT.
- `total_replacements()` on SGBT and DistributionalSGBT.
- **AutoTuner fully wired**: `after_train()` adjustments applied to champion config,
  `at_replacement()` structural changes applied at replacement boundaries,
  champion `diagnostics_array()` feeds full 5-signal diagnostics to adaptor.
- CITATION.cff, RESPONSIBLE_USE.md, CONTRIBUTING.md added to repository.

### Fixed

- Uncertainty: all models return real computed values (no more 0.0 stubs).
  SGBT uses `rolling_contribution_sigma`, neural models use RLS noise variance
  or model-specific signals.
- Uncertainty-modulated learning rate for TTT (`eta`) and KAN (`lr`).

### Changed

- irithyll-core `StreamingLearner` trait now has 4 additional default methods
  (non-breaking: all return no-op/zero defaults).

## [9.8.2] - 2026-03-28

### Added

- **Auto-builder engine** (`automl::auto_builder`):
  - `FeasibleRegion` — SRM-inspired complexity budget from data characteristics.
  - `WelfordRace` — all-see-all racing with center + perturbation candidates.
  - `DiagnosticAdaptor` — streams smooth LR/lambda adjustments and structural changes
    from model diagnostic signals.
  - `ConfigDiagnostics` — 5-signal struct (residual alignment, regularization sensitivity,
    depth sufficiency, effective DOF, uncertainty).
- **`DiagnosticSource` trait** — model-agnostic diagnostic extraction. Implemented for
  all 30+ models (SGBT, DistributionalSGBT, ESN, Mamba, SpikeNet, TTT, KAN, MoE,
  KRLS, RLS, Linear, Mondrian, Pipeline, ContinualLearner, and more).
- **SGBT diagnostic cache** — `update_diagnostic_cache()` computes 4 real signals
  from tree G/H leaf traversal every `train_one()`.
- Comprehensive diagnostics module (`ensemble::diagnostics`) with `TreeDiagnostics`,
  `EnsembleDiagnostics`, `DistributionalDiagnostics`, and `Display` impls.

## [9.8.1] - 2026-03-27

### Added

- KAN and TTT integrated into AutoML `Factory` (Algorithm::Kan, Algorithm::Ttt).
- Python bindings for `StreamingTTT`, `StreamingKAN`, `NeuralMoE`, `AutoTuner`.
- CLI support for `ttt` and `kan` model types in `irithyll train`.
- Example programs: `kan_regression.rs`, `neural_moe.rs`.
- `grace_period` added as 7th hyperparameter in SGBT/Distributional search spaces.

## [9.8.0] - 2026-03-27

### Added

- **Streaming Kolmogorov-Arnold Networks** (`kan`):
  - `StreamingKAN` — B-spline edge activations with De Boor's algorithm.
  - `KANConfig` builder with `n_in`, `n_out`, `hidden_sizes`, `spline_order`,
    `grid_size`, `lr`, `w_b`, `w_s` parameters.
  - `KANLayer` — per-edge cubic B-spline coefficients, sparse gradient updates
    (only 4 coefficients update per edge per sample).
  - Welford input normalization, rolling loss for LR modulation.
  - Implements `StreamingLearner` and `DiagnosticSource`.

## [9.7.2] - 2026-03-26

### Added

- **Unified `Factory`** struct with `Algorithm` enum (8 variants: Sgbt, Distributional,
  Esn, Mamba, Attention, SpikeNet, Kan, Ttt). Replaces per-model factory types
  (old types deprecated).
- Complexity-adjusted elimination: `score = ewma_error + complexity/n_seen` (MDL-inspired).
- Drift-triggered re-racing: ADWIN fires → new tournament with expanded bracket.
- `DiscountedThompsonSampling` bandit (Qi+ 2023) for factory selection with
  exponential forgetting.

## [9.7.1] - 2026-03-26

### Added

- **Per-node auto-bandwidth soft routing** — `predict_soft_routed()` on
  DistributionalSGBT and SGBT. Per-node Gaussian kernels with auto-calibrated
  bandwidths from median split threshold gaps. Continuous weighted blends
  eliminate step discontinuities at decision boundaries.
- irithyll-core bumped to 0.9.0 for soft routing support.

## [9.7.0] - 2026-03-25

### Added

- **Streaming Test-Time Training** (`ttt`):
  - `StreamingTTT` — hidden state IS a linear model (W), updated by gradient
    descent on reconstruction loss per sample. Titans extensions (momentum +
    weight decay).
  - `TTTConfig` builder with `d_model`, `n_heads`, `eta`, `alpha`, `momentum`.
  - `TTTLayer` — fast weight updates with optional momentum decay.
  - `prediction_uncertainty()` from rolling variance.
- **Honest sigma** (`DistributionalSGBT::honest_sigma()`) — tree contribution
  variance for instant epistemic uncertainty, zero hyperparameters.
- **Adaptive MTS** — `adaptive_mts(sigma_ratio)` modulates tree lifetime based
  on model uncertainty. High uncertainty → shorter lifetime → faster adaptation.
- **Proactive pruning** — `proactive_prune()` replaces least-contributing trees
  at configurable intervals.

## [9.6.0] - 2026-03-25

### Added

- **Streaming Neural Mixture of Experts** (`moe`):
  - `NeuralMoE` — polymorphic `Box<dyn StreamingLearner>` experts with top-k
    sparse routing and DeepSeek-v3 load balancing.
  - `LinearRouter` — softmax gating with SGD updates and auxiliary load balance loss.
  - `NeuralMoEBuilder` for ergonomic construction.
  - `expert_disagreement()` and `cached_disagreement()` for uncertainty.
  - Implements `StreamingLearner` and `DiagnosticSource`.

## [9.5.2] - 2026-03-24

### Added

- Warmup-aware elimination: `ModelFactory::warmup_hint()` protects neural models
  during cold-start (warmup-excluded median in early stopping).
- Statistical early stopping: Welford z-test (z > 2.0, min 30 samples) kills
  bad candidates early.
- Adaptive bracket sizing: `n_initial` doubles on promotion, halves when champion holds.

## [9.5.1] - 2026-03-24

### Added

- Tournament successive halving: 8 candidates → eliminate bottom half each round → finalist.
- Multi-factory racing across heterogeneous model types.
- Perturbation sampling: 50% candidates from champion's config neighborhood,
  50% random/bandit-guided.
- `ConfigSpace`, `HyperParam`, `ConfigSampler` (random, latin hypercube, perturb).
- `RewardNormalizer` (EWMA baseline, normalize to [0,1]).

## [9.5.0] - 2026-03-24

### Added

- **Streaming AutoML** (`automl`):
  - `AutoTuner` with champion-challenger racing.
  - `ModelFactory` trait and `SgbtFactory` for hyperparameter search.
  - `AutoTunerConfig` with `round_budget`, `n_initial`, `metric`, `ewma_span`.
  - `auto_tune()` convenience factory function.
  - `AutoMetric` enum (MAE, RMSE, R2).

## [9.4.1] - 2026-03-24

### Changed

- TUI overhaul: improved dashboard layout, better metric display.
- Documentation and reference updates across all modules.
- CLI and Python bindings updated for v9.4 features.

## [9.4.0] - 2026-03-23

### Added

- **SIMD-accelerated inference** in irithyll-core:
  - `simd::dot_product_f64()` — 4-wide f64 SIMD dot product.
  - `simd::matvec_f64()` — SIMD matrix-vector multiply.
  - Feature-gated behind `simd` feature flag.
  - Auto-vectorization hints for non-SIMD fallback paths.

## [9.3.0] - 2026-03-23

### Added

- **Conformal prediction v2** (`conformal`):
  - PID control for adaptive coverage (Angelopoulos+ 2024).
  - Strongly adaptive conformal inference (Gibbs & Candes 2024).
  - Platt scaling for probability calibration.
  - `ConformalPID` with configurable P/I/D gains.
  - Step, exponential, and cosine learning rate schedules.

## [9.2.0] - 2026-03-23

### Added

- **Drift-aware continual learning** (`continual`):
  - `ContinualLearner` wrapper with pluggable `ContinualStrategy`.
  - Elastic Weight Consolidation (EWC) — Fisher information regularization.
  - Neuron regeneration — periodic random re-initialization of dead neurons.
  - Parameter isolation — freeze important parameters, learn new ones.

## [9.1.0] - 2026-03-22

### Added

- **Unified streaming linear attention** (`attention`):
  - GLA (Gated Linear Attention).
  - DeltaNet (delta rule attention).
  - RWKV-style (time-mixing with exponential decay).
  - Hawk (gated recurrence with 1D depthwise convolution).
  - RetNet (multi-scale exponential decay).
  - mLSTM (matrix LSTM with exponential gating).
  - `StreamingAttentionModel` with RLS readout.
  - `AttentionPreprocessor` as `StreamingPreprocessor`.

## [9.0.1] - 2026-03-22

### Added

- Per-split information criterion scoring (Lunde, Kleppe & Skaug 2020) for
  Hoeffding tree split evaluation.

## [9.0.0] - 2026-03-22

### Added

- **Reservoir Computing** (`neural::reservoir`):
  - `NextGenRC` — time-delay embedding + polynomial features (degree 2–5) + RLS readout.
    No random matrices needed. Factory: `ngrc(k, s, degree)`.
  - `EchoStateNetwork` — cycle/ring reservoir topology (O(N) weights) + leaky integrator
    + RLS readout. Factory: `esn(n_reservoir, spectral_radius)`.
  - `ESNPreprocessor` — ESN as `StreamingPreprocessor` for pipeline composition.
    Factory: `esn_preprocessor(...)`.
  - `DelayBuffer` circular buffer for NG-RC time-delay embedding.
  - `HighDegreePolynomial` for degree 2–5 monomial generation.
  - `CycleReservoir` ring topology with leaky integration.
  - `Xorshift64Rng` deterministic PRNG for reservoir/weight initialization.

- **SSM/Mamba — Selective State Space Models** (`neural::ssm`):
  - `StreamingMamba` — Mamba-style selective SSM with input-dependent B, C, Delta
    + RLS readout. Factory: `mamba(d_in, n_state)`.
  - `DiagonalSSM` — non-selective S4D baseline.
  - `SelectiveSSM` — full Mamba architecture in irithyll-core (`no_std`).
  - `MambaPreprocessor` — SSM as `StreamingPreprocessor`.
    Factory: `mamba_preprocessor(...)`.

- **Spiking Neural Networks with e-prop** (`neural::snn`):
  - `SpikeNet` — Vec-based SNN with e-prop online learning, f64 interface.
  - `SpikeNetFixed` — `no_std` SNN with Q1.14 integer arithmetic throughout
    (full training in fixed-point, no floats).
  - LIF neurons, delta spike encoding, piecewise linear surrogate gradient.
  - Random feedback alignment (no weight transport).
  - `SpikePreprocessor` — SNN as `StreamingPreprocessor`.
    Factory: `spikenet(n_hidden)`.

- **Math utilities:** `softplus()` and `sigmoid()` in irithyll-core math module.

- **All architectures implement `StreamingLearner`** — composable with existing
  pipelines (`pipe()`, `normalizer()`, ensembles, evaluation).

- **432 new tests** (total: 1,997). 31 new files, ~8,600 new lines of code.
  No new external dependencies. All `no_std` compatible.

- irithyll-core: 0.5.0 → 0.6.0. irithyll: 8.4.1 → 9.0.0.

## [8.4.0] - 2026-03-22

### Added

- **irithyll-cli v0.1.0** — command-line interface for streaming ML:
  - `irithyll train` — train from CSV with progress bar, save model to JSON
  - `irithyll predict` — batch predictions from a saved model
  - `irithyll eval` — prequential (test-then-train) evaluation with accuracy/RMSE/MAE/kappa
  - `irithyll inspect` — model summary, feature importances, config dump
  - `irithyll export` — export to packed embedded format
  - `irithyll config` — generate/validate TOML config files
  - **TUI dashboard** (`--tui` flag) — ratatui + Catppuccin Mocha theme, live training
    metrics table, braille loss curve chart, throughput display
  - Auto label-encoding for non-numeric CSV columns (e.g. "UP"/"DOWN")
  - Feature names from CSV headers propagated to model config

## [8.3.0] - 2026-03-22

### Added

- **Python bindings overhaul** (irithyll-python v4.0.0, 490 → 1,287 lines):
  - **Batch API:** `fit(X, y)`, `partial_fit(X, y)`, `predict_batch(X)`, `score(X, y)`
    for numpy 2D arrays with GIL released during computation.
  - **Streaming training:** `train_stream(iterator, callback, every_n)` from Python iterators.
  - **PrequentialEvaluator:** test-then-train evaluation returning accuracy/RMSE/MAE dicts,
    with `evaluate_streaming()` for periodic snapshots.
  - **DistributionalGBT:** Python class wrapping DistributionalSGBT — returns (μ, σ) predictions.
  - **ClassifierGBT:** Python class wrapping MulticlassSGBT — `predict()`, `predict_proba()`,
    batch `fit(X, y)` for multi-class classification.
  - **Config expansion:** 5 new hyperparameter fields (gradient_clip_sigma, max_leaf_output,
    adaptive_leaf_bound, min_hessian_sum, split_reeval_interval).
  - **Jupyter support:** `_repr_html_()` for rich notebook display.
  - **Convenience:** `StreamingGBTConfig.fit(X, y)` creates and trains a model in one call.

## [8.2.3] - 2026-03-21

### Changed

- **Track B complete: full training pipeline migrated to irithyll-core** (v0.5.0).
  Histogram binning, Hoeffding trees, SGBT ensembles, and all 10 ensemble variants
  now live in `irithyll-core` behind `#[cfg(feature = "alloc")]`. The `irithyll`
  crate becomes a thin re-export layer plus std-specific extensions (async streaming,
  Arrow/Parquet, ONNX export, preprocessing, metrics, evaluation, pipeline).
  - **Phase 3:** Histogram module (7 files, 3,236 lines) → irithyll-core
  - **Phase 4:** Tree module (8 files, 6,589 lines) → irithyll-core
  - **Phase 5:** Ensemble core (6 files, 5,164 lines) → irithyll-core
  - **Phase 6:** Ensemble variants (11 files, 8,571 lines) → irithyll-core
  - New core modules: `feature` (FeatureType), `learner` (StreamingLearner trait)
  - All float ops converted to `libm`/`crate::math` for no_std compatibility
  - serde derives gated behind `#[cfg_attr(feature = "serde", ...)]`
  - HashMap-dependent methods gated behind `#[cfg(feature = "std")]`
  - irithyll-core: 686 tests. irithyll: 879 tests. Total: 1,565.

## [8.2.2] - 2026-03-21

### Added

- **Per-leaf adaptive output bounds** -- `adaptive_leaf_bound` config option. Each leaf
  tracks EWMA of its own output weight and clamps predictions to `|mean| + k * std`.
  Synchronized with `leaf_decay_alpha` when `leaf_half_life` is set, Welford online
  otherwise. Warmup (10 samples) falls back to `max_leaf_output` if set.
  - New config field: `SGBTConfig::adaptive_leaf_bound: Option<f64>`
  - New builder method: `.adaptive_leaf_bound(k)`
  - 9 new tests covering warmup, clamping, EWMA adaptation, and backward compat.

## [8.2.1] - 2026-03-21

### Added

- **Airlines + Covertype benchmarks** -- prequential evaluation on 3 real datasets:
  - Electricity (45K, binary, concept drift) -- existing
  - Airlines (539K, binary, large scale flight delay prediction)
  - Covertype (581K, 7-class forest cover type, high dimensionality)
- **Multi-class prequential evaluation** using `MulticlassSGBT` with per-class confusion
  matrix, multi-class Cohen's Kappa, and macro-F1.
- **Learning curve CSV output** -- checkpoint-level accuracy/kappa/F1 written to
  `datasets/results/` for each dataset, enabling accuracy-over-time plots.
- **Memory tracking** -- peak memory estimates reported in benchmark output.
- **Dataset download script** -- `datasets/download.py` fetches Airlines and Covertype
  from standard ML repositories.
- Comparison scripts extended for all 3 datasets (River, XGBoost, LightGBM).
- `comparison/collect_results.py` generates unified multi-dataset BENCHMARKS.md.

## [8.2.0] - 2026-03-21

### Added

- **Real dataset benchmarks** -- prequential evaluation on the Electricity dataset (45K samples,
  binary classification with concept drift). Best config achieves 88.5% accuracy, Kappa 0.762.
  Includes comparison harness for River, XGBoost, and LightGBM.
- **`irithyll-core` v0.4.0: drift detection migration** -- all three drift detectors
  (Page-Hinkley, DDM, ADWIN) are now `no_std + alloc` compatible in `irithyll-core`.
  Trait, signal enum, and state types all available in the core crate.
  - `DriftSignal` enum available without `alloc` (pure `core`).
  - `DriftDetector` trait, `DriftDetectorState`, implementations behind `alloc` feature.
  - ADWIN exponential histogram fully ported (907 lines).
- **`irithyll-core` v0.4.0: error type migration** -- `ConfigError` and `IrithyllError`
  now defined in `irithyll-core` with hand-written `core::fmt::Display` impls.
  `std::error::Error` impl available behind `std` feature.
- Benchmark harness: `benches/real_dataset_bench.rs` with CSV dataset loading,
  prequential evaluation, and detailed results reporting.
- Comparison scripts: `comparison/river/bench_river.py`, `comparison/xgboost/bench_xgb.py`,
  `comparison/collect_results.py`.

### Changed

- `irithyll` now depends on `irithyll-core` with `std` feature (was `alloc` only).
- `irithyll/src/drift/` is now a thin re-export layer over `irithyll-core::drift`.
- `irithyll/src/error.rs` re-exports `ConfigError` from `irithyll-core`.

## [8.1.2] - 2026-03-20

### Added

- **Per-expert MoE configs** -- `MoEDistributionalSGBT::with_expert_configs()` allows each
  expert to use its own `SGBTConfig` (different depths, lambda, learning rates, etc.).
  Shadow respawn uses per-expert config when available.
- **Entropy gate regularization** -- `entropy_weight` field on `MoEDistributionalSGBT`
  adds entropy bonus to gate SGD, preventing collapse to a single expert.
  Gradient: `d(-H)/dz_k = p_k * (log(p_k) + 1) - mean_term`.
- **Distributional packed f32 export** -- `export_distributional_packed()` exports the
  location ensemble of `DistributionalSGBT` to irithyll-core packed binary format.
  Returns `(bytes, location_base)` for f64-precision base handling.
- **Dual-path inference** -- `packed_refresh_interval` config field enables a packed f32
  cache on `DistributionalSGBT` that refreshes every N training samples. `predict()`
  uses cache-optimal BFS traversal when available, falls back to full tree walk.
  `enable_packed_cache(interval)` for runtime activation.
- New accessors on `DistributionalSGBT`: `location_steps()`, `location_base()`,
  `learning_rate()`, `has_packed_cache()`.
- New accessors on `MoEDistributionalSGBT`: `entropy_weight()`, `expert_configs()`.
- Builder method `SGBTConfigBuilder::packed_refresh_interval()`.
- 7 new tests covering all additions.

## [8.1.0] - 2026-03-18

### Added

- **Int16 quantized inference (`irithyll-core`)** -- 8-byte `PackedNodeI16` with integer-only
  tree traversal for FPU-less embedded targets.
  - `QuantizedEnsembleHeader` with "IR16" magic, per-feature quantization scales, leaf scale.
  - `QuantizedEnsembleView` -- zero-copy view with `predict()` (inline quantization, zero-alloc)
    and `predict_prequantized()` (pure integer hot loop, zero float ops).
  - `predict_tree_i16()` / `predict_tree_i16_x4()` -- branch-free integer traversal.
  - `predict_tree_i16_inline()` -- hybrid path with on-the-fly f32-to-i16 quantization.
  - 8 nodes per 64-byte cache line (vs 5 for f32). 48% more nodes in same flash.
  - On Cortex-M0+ (no FPU): integer compare is ~25x faster than software f32 compare.
- **`export_packed_i16()`** -- converts trained SGBT to int16 quantized packed binary.
  - Per-feature scale: `32767 / max(|thresholds|)` for optimal i16 range utilization.
  - Global leaf scale with i32 accumulation, single dequantization at prediction end.
  - `validate_export_i16()` for accuracy comparison against original model.
- **`math` module (`irithyll-core`)** -- libm-backed f64 math operations for no_std.
  - 20 functions: abs, sqrt, exp, ln, log2, powf, powi, sin, cos, tanh, floor, ceil, round, etc.
  - Foundation for Track B (migrating training algorithms to irithyll-core).

### Changed

- `irithyll-core` now depends on `libm` 0.2 (pure Rust, no_std, thumbv6m compatible).
- `irithyll-core` version bumped to 0.2.0 (new public types + dependency).

## [8.0.1] - 2026-03-18

### Added

- **`MoEDistributionalSGBT`** -- Mixture of Experts over `DistributionalSGBT`
  ensembles with shadow expert competition via Hoeffding bounds.
  - K active experts + K shadow challengers, each a full `DistributionalSGBT`.
  - Learned softmax gate: `W · x + b → softmax → routing probabilities`.
  - Shadow competition uses Gaussian NLL difference with Hoeffding bound test.
  - When shadow proves statistically superior, it replaces the active expert.
  - Gate-weighted mixture prediction via law of total variance.
  - Supports soft gating (all experts see all samples) and hard top-k routing.
  - `predict()` returns `GaussianPrediction { mu, sigma }` with correct mixture
    variance accounting for both within-expert and between-expert variance.

## [8.0.0] - 2026-03-18

### Added

- **`irithyll-core` crate** — new `#![no_std]` zero-alloc inference engine for
  deploying trained models on embedded targets (Cortex-M0+, 32KB flash).
  - 12-byte `PackedNode` AoS format (5 nodes per cache line).
  - `EnsembleView<'a>` — zero-copy view over `&[u8]`, validated once at construction.
  - Branch-free `predict_tree()` traversal (`cmov`/`csel`, no pipeline stalls).
  - `predict_tree_x4()` interleaved batch prediction exploiting ILP.
  - f64→f32 quantization with tolerance validation.
- **`export_embedded` module** — converts trained `SGBT` to packed binary format.
  - BFS tree reindexing with contiguous u16 node indices.
  - Learning rate baked into leaf values (one less multiply per tree).
  - `validate_export()` for comparing original vs packed predictions.
- **Re-exports** — `EnsembleView`, `PackedNode`, `FormatError` available from
  root `irithyll` crate via `pub use irithyll_core::*`.
- **CI** — `thumbv6m-none-eabi` cross-compilation check for `irithyll-core`,
  workspace-wide clippy and docs.
- **Benchmark** — `packed_bench` comparing packed inference vs standard SGBT prediction.

### Changed

- Workspace restructured: `irithyll-core` added as workspace member.
- `irithyll` now depends on `irithyll-core` (path dependency).

## [7.9.1] - 2026-03-17

### Fixed

- **Cascading swap bug in graduated handoff.** When a shadow tree was promoted
  to active, its total `n_samples_seen()` already exceeded `max_tree_samples`,
  causing an immediate second swap that promoted a cold stump. Added
  `samples_at_activation` tracking: time-based replacement and blend weight
  decay now use samples-since-activation, not total lifetime sample count.

## [7.9.0] - 2026-03-17

### Added

- **Graduated tree handoff** -- eliminates prediction dips during tree
  replacement in streaming GBDT ensembles (Boulevard/GOOWE-inspired).
- **`shadow_warmup`** -- new `SGBTConfig` option enabling always-on shadow
  trees. When set, each tree slot immediately spawns a shadow that trains
  alongside the active tree. After `shadow_warmup` samples, the shadow
  contributes to graduated predictions.
- **`predict_graduated()`** -- on `SGBT`, `DistributionalSGBT`, `TreeSlot`,
  and `BoostingStep`. Blends active and shadow predictions based on
  relative maturity: active weight decays linearly from 1.0 (at 80% of
  `max_tree_samples`) to 0.0 (at 120%), shadow weight ramps from 0.0
  to 1.0 over `shadow_warmup` samples. When active fully decays, shadow
  is promoted and a new shadow spawns -- zero cold-start.
- **`predict_graduated_sibling_interpolated()`** -- premium prediction path
  combining graduated handoff (no replacement dips) with sibling-based
  spatial interpolation (no step-function artifacts). The smoothest
  possible prediction surface for streaming GBDT.
- Soft replacement: when active tree's weight reaches 0, the shadow is
  promoted to active and a new shadow is spawned immediately. The drift
  detector is reset. No moment where a cold tree serves predictions.

## [7.8.2] - 2026-03-17

### Added

- **`predict_sibling_interpolated()`** -- new prediction method on `SGBT`,
  `DistributionalSGBT`, and `HoeffdingTree`. At each split node, linearly
  blends left and right subtree predictions based on the feature's distance
  from the threshold, using auto-calibrated bandwidths as the interpolation
  margin. Predictions vary continuously as features move near split
  boundaries, eliminating step-function artifacts. This complements the
  existing `predict_interpolated()` (temporal smoothing) with spatial
  smoothing in feature space.

## [7.8.1] - 2026-03-17

### Added

- **`max_leaf_output`** -- new `SGBTConfig`/`TreeConfig` option clamping leaf
  predictions to `[-max, max]`. Breaks the feedback loop that causes
  prediction explosions in streaming settings.
- **`min_hessian_sum`** -- new `SGBTConfig`/`TreeConfig` option suppressing
  leaf output until hessian accumulation exceeds the threshold. Prevents
  post-replacement spikes from fresh leaves with insufficient samples.
- **`predict_interpolated()`** -- new prediction method on `SGBT`,
  `DistributionalSGBT`, and `HoeffdingTree`. Blends leaf predictions with
  parent node predictions via `alpha = hess / (hess + lambda)`, fixing
  static EWRV/OIF predictions from recently-split leaves.
- **Ensemble-level gradient statistics** -- `DistributionalSGBT` now tracks
  Welford running mean/variance of gradients across all trees. Exposed via
  `ensemble_grad_mean()`, `ensemble_grad_std()`, and in `ModelDiagnostics`.
  Enables new trees to inherit the ensemble's gradient distribution,
  fixing the cold-start flaw in `gradient_clip_sigma`.
- **`huber_k`** -- new `SGBTConfig` option enabling Huber loss for
  `DistributionalSGBT` location gradients with adaptive
  `delta = k * empirical_sigma`. Standard value 1.345 (95% Gaussian
  efficiency). Bounds gradients by construction without per-leaf clipping.

## [7.8.0] - 2026-03-16

### Added

- **Preprocessing expansion** -- 5 new streaming preprocessors for feature
  engineering in online pipelines.
- **`FeatureHasher`** -- hashing trick (Weinberger et al., 2009) for
  fixed-size dimensionality reduction. Dual-seed multiply-xorshift hash
  with sign bit preserving inner products in expectation. Implements
  `StreamingPreprocessor` (stateless).
- **`OneHotEncoder`** -- streaming one-hot encoding with online category
  discovery. Configurable `max_categories` cap per feature. Sorted
  category storage for deterministic encoding. Implements
  `StreamingPreprocessor`.
- **`TargetEncoder`** -- Bayesian-smoothed target encoding for categorical
  features. Formula: `(n * cat_mean + m * global_mean) / (n + m)`.
  Standalone (requires target, does not implement `StreamingPreprocessor`).
- **`MinMaxScaler`** -- streaming min/max tracking with configurable output
  range (default `[0, 1]`). Constant features map to range midpoint.
  Implements `StreamingPreprocessor`.
- **`PolynomialFeatures`** -- degree-2 polynomial feature generation
  (squares + pairwise interactions). Optional `interaction_only` mode
  for cross-terms without squared terms. Stateless, implements
  `StreamingPreprocessor`.
- New factory functions: `feature_hasher()`, `min_max_scaler()`, `one_hot()`,
  `polynomial_features()`, `target_encoder()`.
- New re-exports at crate root: `FeatureHasher`, `MinMaxScaler`,
  `OneHotEncoder`, `PolynomialFeatures`, `TargetEncoder`.

## [7.7.0] - 2026-03-16

### Added

- **Multi-armed bandits** (`bandits` module) -- online decision-making algorithms
  for the exploration-exploitation trade-off in streaming settings.
- **`Bandit` trait** -- unified interface for context-free bandits: `select_arm()`,
  `update(arm, reward)`, `arm_values()`, `arm_counts()`.
- **`ContextualBandit` trait** -- feature-conditioned arm selection:
  `select_arm(context)`, `update(arm, context, reward)`.
- **`EpsilonGreedy`** -- random exploration with probability epsilon (optionally
  decaying via exponential schedule). Constructors: `new()`, `with_seed()`,
  `with_decay()`.
- **`UCB1`** -- Upper Confidence Bound (Auer et al., 2002). Deterministic,
  round-robin initial exploration, `Q(a) + sqrt(2 ln(t) / N(a))` selection.
- **`UCBTuned`** -- UCB with per-arm variance estimates for tighter confidence
  bounds. Tracks sum of squared rewards for empirical variance computation.
- **`ThompsonSampling`** -- Bayesian arm selection via Beta posterior sampling.
  Marsaglia-Tsang Gamma sampler + Box-Muller normals for dependency-free Beta
  samples. Constructors: `new()`, `with_seed()`, `with_prior()`.
- **`LinUCB`** -- contextual bandit with per-arm ridge regression and
  Sherman-Morrison inverse updates (Li et al., 2010). O(d²) per update,
  O(kd²) total space for k arms with d features.
- New factory functions: `epsilon_greedy()`, `ucb1()`, `ucb_tuned()`,
  `thompson()`, `lin_ucb()`.
- New re-exports at crate root: `Bandit`, `ContextualBandit`, `EpsilonGreedy`,
  `UCB1`, `UCBTuned`, `ThompsonSampling`, `LinUCB`.

## [7.6.1] - 2026-03-16

### Changed

- **Auto-bandwidth smooth routing** -- `predict()` on both `SGBT` and
  `DistributionalSGBT` now always uses sigmoid-blended soft routing with
  per-feature auto-calibrated bandwidths. Bandwidths are computed as
  `median_gap × 2.0` from the split thresholds of all trees in the ensemble.
  Features with fewer than 3 unique thresholds fall back to `range / n_bins × 2`,
  and features never split on use hard routing (bandwidth = ∞).
- **Removed `bandwidth` from `SGBTConfig`** -- the `bandwidth()` builder method,
  field, and validation have been removed. Auto-bandwidth is the only mode.
- Bandwidth cache refreshes automatically when tree replacements (drift or
  time-based) are detected, via a lightweight replacement counter on `TreeSlot`.

### Added

- `HoeffdingTree::collect_split_thresholds_per_feature()` -- collects all
  continuous split thresholds per feature from the tree arena.
- `HoeffdingTree::predict_smooth_auto()` -- per-feature bandwidth smooth
  prediction.
- `TreeSlot::replacements()` -- counts tree replacements for cache invalidation.
- `TreeSlot::prediction_mean()` / `prediction_std()` -- Welford online stats
  tracking per-tree prediction distribution for anomaly detection.
- `SGBT::auto_bandwidths()` / `DistributionalSGBT::auto_bandwidths()` --
  exposes the current per-feature bandwidth vector.
- `auto_bandwidths: Vec<f64>` in `ModelDiagnostics`.
- `prediction_mean` / `prediction_std` in `TreeDiagnostic`.

### Fixed

- Clippy `needless_range_loop` warning in Holt-Winters seasonal initialization.

## [7.6.0] - 2026-03-16

### Added

- **SNARIMAX** (`SNARIMAX`) -- streaming non-linear ARIMA with seasonal,
  exogenous, and autoregressive/moving-average components. Online SGD
  parameter updates with gradient clipping for stability on unnormalized data.
  Configurable AR/MA orders, seasonal period, and exogenous feature count.
  Multi-step recursive forecasting via `forecast(horizon)`.
- **Holt-Winters** (`HoltWinters`) -- streaming triple exponential smoothing
  with additive and multiplicative seasonality modes. Buffered initialization
  from the first seasonal period. Level, trend, and seasonal components
  updated incrementally per observation. Multi-step forecasting.
- **Streaming Decomposition** (`StreamingDecomposition`) -- online seasonal
  decomposition into trend, seasonal, and residual components. EWMA-based
  trend estimation with per-position seasonal factor tracking. Returns
  `DecomposedPoint` with the identity `observed = trend + seasonal + residual`.
- `Seasonality` enum (`Additive` / `Multiplicative`) for Holt-Winters.
- `DecomposedPoint` struct for decomposition output.
- `SNARIMAXCoefficients` snapshot struct for model introspection.
- New re-exports at crate root: `SNARIMAX`, `SNARIMAXConfig`,
  `SNARIMAXCoefficients`, `HoltWinters`, `HoltWintersConfig`, `Seasonality`,
  `StreamingDecomposition`, `DecompositionConfig`, `DecomposedPoint`.

## [7.5.0] - 2026-03-16

### Added

- **Hoeffding Tree Classifier** (`HoeffdingTreeClassifier`) -- standalone streaming
  decision tree for classification based on the VFDT algorithm (Domingos & Hulten,
  2000). Maintains per-leaf class distributions, splits using information gain with
  Hoeffding bound. Configurable via `HoeffdingClassifierConfig` builder. Implements
  `StreamingLearner`.
- **Multinomial Naive Bayes** (`MultinomialNB`) -- streaming classifier for
  count/frequency features with Laplace smoothing. Automatic class discovery,
  predict_proba/predict_log_proba support.
- **Bernoulli Naive Bayes** (`BernoulliNB`) -- streaming classifier for binary
  features with configurable binarization threshold. Explicitly models feature
  absence unlike Multinomial NB.
- **Adaptive Random Forest** (`AdaptiveRandomForest`) -- ensemble of streaming
  learners with ADWIN drift detection and automatic tree replacement
  (Gomes et al., 2017). Poisson(lambda)-weighted bootstrap, random feature
  subspaces, generic over any `StreamingLearner` via factory closure.
- New re-exports at crate root: `HoeffdingTreeClassifier`, `MultinomialNB`,
  `BernoulliNB`, `AdaptiveRandomForest`.

## [7.4.0] - 2026-03-16

### Added

- **Streaming K-Means** (`StreamingKMeans`) -- mini-batch online K-Means with
  optional forgetting factor for non-stationary streams (Sculley, 2010). Lazy
  initialization from first k distinct samples. Decaying learning rate
  `eta = 1/count` per centroid. Configurable via `StreamingKMeansConfig` builder.
- **DBSTREAM** (`DBStream`) -- density-based streaming clustering with weighted
  micro-clusters and shared-density graph for macro-cluster merging
  (Hahsler & Bolanos, 2016). Exponential decay, periodic cleanup of dead MCs,
  connected-component macro-cluster extraction via DFS.
- **CluStream** (`CluStream`) -- micro/macro streaming clustering via Cluster
  Feature vectors (Aggarwal et al., 2003). Online phase maintains CF summaries
  with absorb/merge logic. Offline phase produces k macro-clusters via weighted
  K-Means on micro-cluster centers.
- `ClusterFeature` -- sufficient statistics `(n, LS, SS)` for streaming cluster
  summarization. Supports absorb, merge, center, and radius computation.
- `MicroCluster` -- weighted centroid with creation time for DBSTREAM.
- New re-exports at crate root: `StreamingKMeans`, `StreamingKMeansConfig`,
  `DBStream`, `DBStreamConfig`, `MicroCluster`, `CluStream`, `CluStreamConfig`,
  `ClusterFeature`.

## [7.3.0] - 2026-03-16

### Added

- **Prequential evaluation** -- `PrequentialEvaluator` implements the standard
  streaming ML evaluation protocol (Gama et al., 2009): test-then-train on each
  sample. Configurable warmup period and step interval. Optional rolling-window
  and EWMA metric tracking alongside cumulative `RegressionMetrics`.
- **Progressive validation** -- `ProgressiveValidator` with configurable holdout
  strategies: `HoldoutStrategy::None` (pure prequential), `Periodic { period }`
  (every N-th sample held out), `Random { holdout_fraction, seed }` (stochastic
  holdout with deterministic xorshift64 PRNG). All strategies always evaluate;
  only training is gated.
- **Cohen's Kappa** (`CohenKappa`) -- streaming classification agreement metric
  adjusted for chance. Auto-growing confusion matrix discovers new classes on
  the fly. Tie-breaking by lowest class index.
- **Kappa-M** (`KappaM`) -- compares model against a majority-class baseline
  classifier. Tracks per-class counts to determine the current majority class.
- **Kappa-T** (`KappaT`) -- compares model against a no-change (temporal)
  classifier that always predicts the previous true label.
- **Streaming AUC-ROC** (`StreamingAUC`) -- windowed approximation of AUC-ROC
  via the Wilcoxon-Mann-Whitney U statistic over a fixed-size circular buffer
  of `(score, is_positive)` pairs. O(n^2) over the window per computation.
- New re-exports at crate root: `PrequentialEvaluator`, `PrequentialConfig`,
  `ProgressiveValidator`, `HoldoutStrategy`, `CohenKappa`, `KappaM`, `KappaT`,
  `StreamingAUC`.

## [7.2.0] - 2026-03-16

### Added

- **PD sigma modulation** -- proportional-derivative controller for uncertainty-
  modulated learning rate in `DistributionalSGBT`. Tracks `sigma_velocity`
  (EWMA-smoothed derivative of empirical σ) with self-calibrating derivative
  gain `k_d = |sigma_velocity| / rolling_sigma_mean`. The PD ratio
  `(sigma + k_d * sigma_velocity) / rolling_sigma_mean` anticipates regime
  changes, boosting the learning rate *before* errors fully propagate.
  New `sigma_velocity()` getter exposes the signal.
- **Smooth prediction on SGBT** -- `predict_smooth(features, bandwidth)` now
  available on `SGBT` (not just `DistributionalSGBT`), aggregating per-tree
  smooth predictions through the full boosting ensemble.
- **Config-driven smooth prediction** -- `bandwidth: Option<f64>` field on
  `SGBTConfig`. When set, `predict()` on both `SGBT` and `DistributionalSGBT`
  automatically uses sigmoid-blended soft routing with no API change required.
  Builder method: `.bandwidth(0.5)`. Validated as positive and finite.

## [7.1.0] - 2026-03-16

### Added

- **Smooth prediction** (`predict_smooth`) -- sigmoid-blended soft routing
  through tree split nodes. Instead of hard left/right decisions, each split
  uses `alpha = sigmoid((threshold - feature) / bandwidth)` and recursively
  blends subtree predictions. Produces a continuous function with no bins,
  boundaries, or jumps. Available on `HoeffdingTree`, `TreeSlot`,
  `BoostingStep`, and `DistributionalSGBT`. Bandwidth parameter controls
  transition sharpness.
- **Leaf state accessor** (`HoeffdingTree::leaf_grad_hess`) -- read-only access
  to per-leaf gradient and hessian sums. Enables inverse-hessian confidence
  estimation: `confidence = 1.0 / (hess_sum + lambda)`.
- **Per-leaf sample counts in diagnostics** -- `TreeDiagnostic` now includes
  `leaf_sample_counts: Vec<u64>` showing data distribution across leaves.
- Internal `leaf_prediction` helper on `HoeffdingTree` reducing code duplication
  across `predict`, `predict_with_variance`, and `predict_smooth`.

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

[7.7.0]: https://github.com/evilrat420/irithyll/compare/v7.6.1...v7.7.0
[7.6.1]: https://github.com/evilrat420/irithyll/compare/v7.6.0...v7.6.1
[7.6.0]: https://github.com/evilrat420/irithyll/compare/v7.5.0...v7.6.0
[7.5.0]: https://github.com/evilrat420/irithyll/compare/v7.4.0...v7.5.0
[7.4.0]: https://github.com/evilrat420/irithyll/compare/v7.3.0...v7.4.0
[7.3.0]: https://github.com/evilrat420/irithyll/compare/v7.2.0...v7.3.0
[7.2.0]: https://github.com/evilrat420/irithyll/compare/v7.1.0...v7.2.0
[7.1.0]: https://github.com/evilrat420/irithyll/compare/v7.0.0...v7.1.0
[7.0.0]: https://github.com/evilrat420/irithyll/compare/v6.4.0...v7.0.0
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
