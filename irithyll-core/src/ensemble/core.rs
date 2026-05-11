//! SGBT core: struct definition, Clone/Debug, and constructors.
//!
//! This module isolates the structural definition and initialization logic,
//! keeping the hot path (train_one, predict) separate for clarity.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use core::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::Observation;
#[allow(unused_imports)] // Used in doc links + tests
use crate::sample::Sample;

/// Cached diagnostic state for SGBT, separated from the core training state
/// to improve struct clarity and cache locality in the prediction path.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct DiagnosticCache {
    /// Previous per-tree contributions for residual alignment (cosine similarity).
    pub(crate) prev_contributions: Vec<f64>,
    /// Contributions from two calls ago, for delta-based alignment.
    pub(crate) prev_prev_contributions: Vec<f64>,
    /// Cached cosine similarity of consecutive tree contribution vectors.
    pub(crate) cached_residual_alignment: f64,
    /// Cached mean |G|/(H+λ)² across all leaves.
    pub(crate) cached_reg_sensitivity: f64,
    /// Cached F-statistic (between-leaf / within-leaf variance).
    pub(crate) cached_depth_sufficiency: f64,
    /// Cached trace(H/(H+λ)) across all leaves.
    pub(crate) cached_effective_dof: f64,
    /// Per-tree EWMA of signed contribution accuracy. Positive = helps, negative = hurts.
    pub(crate) contribution_accuracy: Vec<f64>,
    /// EWMA alpha for contribution accuracy tracking.
    pub(crate) prune_alpha: f64,
}

/// Streaming Gradient Boosted Trees ensemble.
///
/// The primary entry point for training and prediction. Generic over `L: Loss`
/// so the loss function's gradient/hessian calls are monomorphized (inlined)
/// into the boosting hot loop -- no virtual dispatch overhead.
///
/// The default type parameter `L = SquaredLoss` means `SGBT::new(config)`
/// creates a regression model without specifying the loss type explicitly.
///
/// # Examples
///
/// ```ignore
/// use irithyll::{SGBTConfig, SGBT};
///
/// // Regression with squared loss (default):
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::new(config);
/// ```
///
/// ```ignore
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::logistic::LogisticLoss;
///
/// // Classification with logistic loss -- no Box::new()!
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::with_loss(config, LogisticLoss);
/// ```
pub struct SGBT<L: Loss = SquaredLoss> {
    /// Configuration.
    pub(crate) config: SGBTConfig,
    /// Boosting steps (one tree + drift detector each).
    pub(crate) steps: Vec<BoostingStep>,
    /// Loss function (monomorphized -- no vtable).
    pub(crate) loss: L,
    /// Base prediction (initial constant, computed from first batch of targets).
    pub(crate) base_prediction: f64,
    /// Whether base_prediction has been initialized.
    pub(crate) base_initialized: bool,
    /// Running collection of initial targets for computing base_prediction.
    pub(crate) initial_targets: Vec<f64>,
    /// Number of initial targets to collect before setting base_prediction.
    pub(crate) initial_target_count: usize,
    /// Total samples trained.
    pub(crate) samples_seen: u64,
    /// RNG state for variant skip logic.
    pub(crate) rng_state: u64,
    /// Per-step EWMA of |marginal contribution| for quality-based pruning.
    /// Empty when `quality_prune_alpha` is `None`.
    pub(crate) contribution_ewma: Vec<f64>,
    /// Per-step consecutive low-contribution sample counter.
    /// Empty when `quality_prune_alpha` is `None`.
    pub(crate) low_contrib_count: Vec<u64>,
    /// Rolling mean absolute error for error-weighted sample importance.
    /// Only used when `error_weight_alpha` is `Some`.
    pub(crate) rolling_mean_error: f64,
    /// Per-feature auto-calibrated bandwidths for smooth prediction.
    /// Computed from median split threshold gaps across all trees.
    pub(crate) auto_bandwidths: Vec<f64>,
    /// Sum of replacement counts across all steps at last bandwidth computation.
    /// Used to detect when trees have been replaced and bandwidths need refresh.
    pub(crate) last_replacement_sum: u64,
    /// EWMA of contribution variance (sigma) across trees for adaptive_mts.
    /// Used as the denominator when computing sigma_ratio for tree lifetime modulation.
    pub(crate) rolling_contribution_sigma: f64,
    /// Ring buffer of sigma_ratio values for end-of-cycle adaptive MTS.
    /// Capacity = grace_period. MTS updates only at tree replacement boundaries.
    pub(crate) sigma_ring: VecDeque<f64>,
    /// Sum of replacement counts at last MTS update (replacement boundary detection).
    pub(crate) mts_replacement_sum: u64,
    // -----------------------------------------------------------------------
    // Diagnostic caches — not used in predict hot path.
    // -----------------------------------------------------------------------
    /// Diagnostic caches — not used in predict hot path.
    pub(crate) diag: DiagnosticCache,
}

impl<L: Loss + Clone> Clone for SGBT<L> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            steps: self.steps.clone(),
            loss: self.loss.clone(),
            base_prediction: self.base_prediction,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            contribution_ewma: self.contribution_ewma.clone(),
            low_contrib_count: self.low_contrib_count.clone(),
            rolling_mean_error: self.rolling_mean_error,
            auto_bandwidths: self.auto_bandwidths.clone(),
            last_replacement_sum: self.last_replacement_sum,
            rolling_contribution_sigma: self.rolling_contribution_sigma,
            sigma_ring: self.sigma_ring.clone(),
            mts_replacement_sum: self.mts_replacement_sum,
            diag: self.diag.clone(),
        }
    }
}

impl<L: Loss> fmt::Debug for SGBT<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SGBT")
            .field("n_steps", &self.steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("base_prediction", &self.base_prediction)
            .field("base_initialized", &self.base_initialized)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor for the default loss (SquaredLoss)
// ---------------------------------------------------------------------------

impl SGBT<SquaredLoss> {
    /// Create a new SGBT ensemble with squared loss (regression).
    ///
    /// This is the most common constructor. For classification or custom
    /// losses, use [`with_loss`](SGBT::with_loss).
    pub fn new(config: SGBTConfig) -> Self {
        Self::with_loss(config, SquaredLoss)
    }
}

// ---------------------------------------------------------------------------
// General impl for all Loss types
// ---------------------------------------------------------------------------

impl<L: Loss> SGBT<L> {
    /// Create a new SGBT ensemble with a specific loss function.
    ///
    /// The loss is stored by value (monomorphized), giving zero-cost
    /// gradient/hessian dispatch.
    ///
    /// ```ignore
    /// use irithyll::{SGBTConfig, SGBT};
    /// use irithyll::loss::logistic::LogisticLoss;
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let model = SGBT::with_loss(config, LogisticLoss);
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L) -> Self {
        let leaf_decay_alpha = config
            .leaf_half_life
            .map(|hl| crate::math::exp(-crate::math::ln(2.0) / hl as f64));

        let tree_config = crate::ensemble::config::build_tree_config(&config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let max_tree_samples = config.max_tree_samples;

        let shadow_warmup = config.shadow_warmup.unwrap_or(0);
        let steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, max_tree_samples, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
                }
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        let n = config.n_steps;
        let has_pruning =
            config.quality_prune_alpha.is_some() || config.proactive_prune_interval.is_some();
        let grace_period = config.grace_period;
        Self {
            config,
            steps,
            loss,
            base_prediction: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
            contribution_ewma: if has_pruning {
                vec![0.0; n]
            } else {
                Vec::new()
            },
            low_contrib_count: if has_pruning { vec![0; n] } else { Vec::new() },
            rolling_mean_error: 0.0,
            rolling_contribution_sigma: 0.0,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            sigma_ring: VecDeque::with_capacity(grace_period),
            mts_replacement_sum: 0,
            diag: DiagnosticCache {
                contribution_accuracy: vec![0.0; n],
                ..Default::default()
            },
        }
    }

    // ---------------------------------------------------------------------------
    // Training
    // ---------------------------------------------------------------------------

    /// Train on a single observation.
    ///
    /// Accepts any type implementing [`Observation`], including [`Sample`],
    /// `SampleRef`, or tuples like `(&[f64], f64)` for zero-copy training.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let target = sample.target();
        let features = sample.features();

        // Guard: skip non-finite inputs to prevent NaN/Inf from corrupting model state.
        if !target.is_finite() || !features.iter().all(|f| f.is_finite()) {
            return;
        }

        // Initialize base prediction from first few targets
        if !self.base_initialized {
            self.initial_targets.push(target);
            if self.initial_targets.len() >= self.initial_target_count {
                self.base_prediction = self.loss.initial_prediction(&self.initial_targets);
                self.base_initialized = true;
                self.initial_targets.clear();
                self.initial_targets.shrink_to_fit();
            }
        }

        // Current prediction starts from base
        let mut current_pred = self.base_prediction;

        // Adaptive MTS: compute contribution variance and set effective max_tree_samples
        if let Some((base_mts, k)) = self.config.adaptive_mts {
            let sigma = self.contribution_variance(features);
            self.rolling_contribution_sigma =
                0.999 * self.rolling_contribution_sigma + 0.001 * sigma;

            let normalized = if self.rolling_contribution_sigma > 1e-10 {
                sigma / self.rolling_contribution_sigma
            } else {
                1.0
            };
            let factor = 1.0 / (1.0 + k * normalized);
            let floor = (base_mts as f64 * self.config.adaptive_mts_floor)
                .max(self.config.grace_period as f64 * 2.0);
            let effective_mts = ((base_mts as f64) * factor).max(floor) as u64;
            for step in &mut self.steps {
                step.slot_mut().set_max_tree_samples(Some(effective_mts));
            }
        }

        let prune_alpha = self
            .config
            .quality_prune_alpha
            .or_else(|| self.config.proactive_prune_interval.map(|_| 0.01));
        let prune_threshold = self.config.quality_prune_threshold;
        let prune_patience = self.config.quality_prune_patience;

        // Track which trees were replaced by quality pruning this step (for double-fire prevention).
        let mut replaced_this_step = vec![false; self.steps.len()];

        // Error-weighted sample importance: compute weight from prediction error
        let error_weight = if let Some(ew_alpha) = self.config.error_weight_alpha {
            let abs_error = crate::math::abs(target - current_pred);
            if self.rolling_mean_error > 1e-15 {
                let w = (1.0 + abs_error / (self.rolling_mean_error + 1e-15)).min(10.0);
                self.rolling_mean_error =
                    ew_alpha * abs_error + (1.0 - ew_alpha) * self.rolling_mean_error;
                w
            } else {
                self.rolling_mean_error = abs_error.max(1e-15);
                1.0 // first sample, no reweighting
            }
        } else {
            1.0
        };

        // Sequential boosting: each step targets the residual of all prior steps
        #[allow(clippy::needless_range_loop)]
        for s in 0..self.steps.len() {
            let gradient = self.loss.gradient(target, current_pred) * error_weight;
            let hessian = self.loss.hessian(target, current_pred) * error_weight;
            let train_count = self
                .config
                .variant
                .train_count(hessian, &mut self.rng_state);

            let step_pred =
                self.steps[s].train_and_predict(features, gradient, hessian, train_count);

            current_pred += self.config.learning_rate * step_pred;

            // Quality-based tree pruning: track contribution and replace dead wood
            if let Some(alpha) = prune_alpha {
                let contribution = crate::math::abs(self.config.learning_rate * step_pred);
                self.contribution_ewma[s] =
                    alpha * contribution + (1.0 - alpha) * self.contribution_ewma[s];

                if self.contribution_ewma[s] < prune_threshold {
                    self.low_contrib_count[s] += 1;
                    if self.low_contrib_count[s] >= prune_patience {
                        self.steps[s].reset();
                        self.contribution_ewma[s] = 0.0;
                        self.low_contrib_count[s] = 0;
                        replaced_this_step[s] = true;
                    }
                } else {
                    self.low_contrib_count[s] = 0;
                }
            }
        }

        // Proactive pruning: replace worst-contributing tree at interval
        if let Some(interval) = self.config.proactive_prune_interval {
            if self.samples_seen % interval == 0
                && self.samples_seen > 0
                && !self.contribution_ewma.is_empty()
            {
                let min_age = interval / 2;

                // Collect (idx, ewma) for mature trees that weren't already replaced by quality pruning.
                let mature: Vec<(usize, f64)> = self
                    .steps
                    .iter()
                    .enumerate()
                    .zip(self.contribution_ewma.iter())
                    .filter(|((i, step), _)| {
                        step.n_samples_seen() >= min_age && !replaced_this_step[*i]
                    })
                    .map(|((i, _), &ewma)| (i, ewma))
                    .collect();

                if !mature.is_empty() {
                    // Compute p25 of contribution_ewma across mature trees
                    let mut sorted_ewma: Vec<f64> = mature.iter().map(|(_, e)| *e).collect();
                    sorted_ewma
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                    let p25_idx = (sorted_ewma.len().saturating_sub(1)) / 4;
                    let p25 = sorted_ewma[p25_idx];

                    // Only prune if the worst is below p25
                    let worst = mature.iter().min_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                    });

                    if let Some(&(worst_idx, worst_ewma)) = worst {
                        if worst_ewma < p25 {
                            self.steps[worst_idx].reset();
                            self.contribution_ewma[worst_idx] = 0.0;
                            self.low_contrib_count[worst_idx] = 0;
                        }
                    }
                }
            }
        }

        // Refresh auto-bandwidths when trees have been replaced or not yet computed.
        self.refresh_bandwidths();
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Train on a batch with periodic callback for cooperative yielding.
    pub fn train_batch_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        interval: usize,
        mut callback: F,
    ) {
        let interval = interval.max(1);
        for (i, sample) in samples.iter().enumerate() {
            self.train_one(sample);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        let total = samples.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    /// Train on a random subsample of a batch using reservoir sampling (Algorithm R).
    pub fn train_batch_subsampled<O: Observation>(&mut self, samples: &[O], max_samples: usize) {
        if max_samples >= samples.len() {
            self.train_batch(samples);
            return;
        }
        let mut reservoir: Vec<usize> = (0..max_samples).collect();
        let mut rng = self.rng_state;
        for i in max_samples..samples.len() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng % (i as u64 + 1)) as usize;
            if j < max_samples {
                reservoir[j] = i;
            }
        }
        self.rng_state = rng;
        reservoir.sort_unstable();
        for &idx in &reservoir {
            self.train_one(&samples[idx]);
        }
    }

    /// Train on a batch with both subsampling and periodic callbacks.
    pub fn train_batch_subsampled_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        max_samples: usize,
        interval: usize,
        mut callback: F,
    ) {
        if max_samples >= samples.len() {
            self.train_batch_with_callback(samples, interval, callback);
            return;
        }
        let mut reservoir: Vec<usize> = (0..max_samples).collect();
        let mut rng = self.rng_state;
        for i in max_samples..samples.len() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng % (i as u64 + 1)) as usize;
            if j < max_samples {
                reservoir[j] = i;
            }
        }
        self.rng_state = rng;
        reservoir.sort_unstable();
        let interval = interval.max(1);
        for (i, &idx) in reservoir.iter().enumerate() {
            self.train_one(&samples[idx]);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        let total = reservoir.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    // ---------------------------------------------------------------------------
    // Prediction
    // ---------------------------------------------------------------------------

    /// Predict the raw output for a feature vector.
    ///
    /// Uses auto-calibrated per-feature bandwidths for smooth (soft) routing.
    /// Falls back to hard routing before any training has occurred.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        if self.auto_bandwidths.is_empty() {
            for step in &self.steps {
                pred += self.config.learning_rate * step.predict(features);
            }
        } else {
            for step in &self.steps {
                pred += self.config.learning_rate
                    * step.predict_smooth_auto(features, &self.auto_bandwidths);
            }
        }
        pred
    }

    /// Predict using sigmoid-blended soft routing with an explicit bandwidth.
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_smooth(features, bandwidth);
        }
        pred
    }

    /// Per-feature auto-calibrated bandwidths used by `predict()`.
    pub fn auto_bandwidths(&self) -> &[f64] {
        &self.auto_bandwidths
    }

    /// Predict with parent-leaf linear interpolation.
    pub fn predict_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_interpolated(features);
        }
        pred
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    pub fn predict_sibling_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate
                * step.predict_sibling_interpolated(features, &self.auto_bandwidths);
        }
        pred
    }

    /// Predict with graduated active-shadow blending.
    pub fn predict_graduated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_graduated(features);
        }
        pred
    }

    /// Predict with graduated blending + sibling interpolation.
    pub fn predict_graduated_sibling_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate
                * step.predict_graduated_sibling_interpolated(features, &self.auto_bandwidths);
        }
        pred
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic loss).
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        self.loss.predict_transform(self.predict(features))
    }

    /// Predict probability (alias for `predict_transformed`).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        self.predict_transformed(features)
    }

    /// Predict with confidence estimation.
    ///
    /// Returns `(prediction, confidence)` where confidence = 1 / sqrt(sum_variance).
    pub fn predict_with_confidence(&self, features: &[f64]) -> (f64, f64) {
        let mut pred = self.base_prediction;
        let mut total_variance = 0.0;
        let lr2 = self.config.learning_rate * self.config.learning_rate;
        for step in &self.steps {
            let (value, variance) = step.predict_with_variance(features);
            pred += self.config.learning_rate * value;
            total_variance += lr2 * variance;
        }
        let confidence = if total_variance > 0.0 && total_variance.is_finite() {
            1.0 / crate::math::sqrt(total_variance)
        } else {
            0.0
        };
        (pred, confidence)
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    // ---------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------

    /// Number of boosting steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Total trees (active + alternates).
    pub fn n_trees(&self) -> usize {
        self.steps.len() + self.steps.iter().filter(|s| s.has_alternate()).count()
    }

    /// Total leaves across all active trees.
    pub fn total_leaves(&self) -> usize {
        self.steps.iter().map(|s| s.n_leaves()).sum()
    }

    /// Total samples trained.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// The current base prediction.
    pub fn base_prediction(&self) -> f64 {
        self.base_prediction
    }

    /// Whether the base prediction has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Access the configuration.
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }

    /// Set the learning rate for future boosting rounds.
    #[inline]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Immutable access to the boosting steps.
    pub fn steps(&self) -> &[BoostingStep] {
        &self.steps
    }

    /// Immutable access to the loss function.
    pub fn loss(&self) -> &L {
        &self.loss
    }

    /// Feature importances based on accumulated split gains across all trees.
    ///
    /// Returns normalized importances (sum to 1.0) indexed by feature.
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut totals: Vec<f64> = Vec::new();
        for step in &self.steps {
            let gains = step.slot().split_gains();
            if totals.is_empty() && !gains.is_empty() {
                totals.resize(gains.len(), 0.0);
            }
            for (i, &g) in gains.iter().enumerate() {
                if i < totals.len() {
                    totals[i] += g;
                }
            }
        }
        let sum: f64 = totals.iter().sum();
        if sum > 0.0 {
            totals.iter_mut().for_each(|v| *v /= sum);
        }
        totals
    }

    /// Feature names, if configured.
    pub fn feature_names(&self) -> Option<&[String]> {
        self.config.feature_names.as_deref()
    }

    /// Feature importances paired with their names.
    ///
    /// Returns `None` if feature names are not configured. Otherwise returns
    /// `(name, importance)` pairs sorted by importance descending.
    pub fn named_feature_importances(&self) -> Option<Vec<(String, f64)>> {
        let names = self.config.feature_names.as_ref()?;
        let importances = self.feature_importances();
        let mut pairs: Vec<(String, f64)> = names
            .iter()
            .zip(importances.iter().chain(core::iter::repeat(&0.0)))
            .map(|(n, &v)| (n.clone(), v))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        Some(pairs)
    }

    /// Train on a single sample with named features.
    #[cfg(feature = "std")]
    pub fn train_one_named(
        &mut self,
        features: &std::collections::HashMap<alloc::string::String, f64>,
        target: f64,
    ) {
        let names = self
            .config
            .feature_names
            .as_ref()
            .expect("train_one_named requires feature_names to be configured");
        let vec: Vec<f64> = names
            .iter()
            .map(|name| features.get(name).copied().unwrap_or(0.0))
            .collect();
        self.train_one(&(&vec[..], target));
    }

    /// Predict with named features.
    #[cfg(feature = "std")]
    pub fn predict_named(
        &self,
        features: &std::collections::HashMap<alloc::string::String, f64>,
    ) -> f64 {
        let names = self
            .config
            .feature_names
            .as_ref()
            .expect("predict_named requires feature_names to be configured");
        let vec: Vec<f64> = names
            .iter()
            .map(|name| features.get(name).copied().unwrap_or(0.0))
            .collect();
        self.predict(&vec)
    }

    // ---------------------------------------------------------------------------
    // Reset
    // ---------------------------------------------------------------------------

    /// Reset the ensemble to initial state.
    pub fn reset(&mut self) {
        for step in &mut self.steps {
            step.reset();
        }
        self.base_prediction = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = self.config.seed;
        self.rolling_mean_error = 0.0;
        self.rolling_contribution_sigma = 0.0;
        self.auto_bandwidths.clear();
        self.last_replacement_sum = 0;
        self.sigma_ring.clear();
        self.mts_replacement_sum = 0;
        self.diag = DiagnosticCache {
            contribution_accuracy: vec![0.0; self.steps.len()],
            ..Default::default()
        };
        if !self.contribution_ewma.is_empty() {
            self.contribution_ewma.iter_mut().for_each(|v| *v = 0.0);
        }
        if !self.low_contrib_count.is_empty() {
            self.low_contrib_count.iter_mut().for_each(|v| *v = 0);
        }
    }

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    /// Compute tree contribution standard deviation (σ proxy for adaptive_mts).
    fn contribution_variance(&self, features: &[f64]) -> f64 {
        let n = self.steps.len();
        if n <= 1 {
            return 0.0;
        }
        let lr = self.config.learning_rate;
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for step in &self.steps {
            let c = lr * step.predict(features);
            sum += c;
            sq_sum += c * c;
        }
        let n_f = n as f64;
        let mean = sum / n_f;
        let var = (sq_sum / n_f) - (mean * mean);
        crate::math::sqrt((var.abs() * n_f / (n_f - 1.0)).max(0.0))
    }

    /// Refresh auto-bandwidths if any tree has been replaced since last computation.
    fn refresh_bandwidths(&mut self) {
        let current_sum: u64 = self.steps.iter().map(|s| s.slot().replacements()).sum();
        if current_sum != self.last_replacement_sum || self.auto_bandwidths.is_empty() {
            self.auto_bandwidths = self.compute_auto_bandwidths();
            self.last_replacement_sum = current_sum;
        }
    }

    /// Compute per-feature auto-calibrated bandwidths from all trees.
    fn compute_auto_bandwidths(&self) -> Vec<f64> {
        const K: f64 = 2.0;
        let n_features = self
            .steps
            .iter()
            .filter_map(|s| s.slot().active_tree().n_features())
            .max()
            .unwrap_or(0);

        if n_features == 0 {
            return Vec::new();
        }

        let mut all_thresholds: Vec<Vec<f64>> = vec![Vec::new(); n_features];
        for step in &self.steps {
            let tree_thresholds = step
                .slot()
                .active_tree()
                .collect_split_thresholds_per_feature();
            for (i, ts) in tree_thresholds.into_iter().enumerate() {
                if i < n_features {
                    all_thresholds[i].extend(ts);
                }
            }
        }

        let n_bins = self.config.n_bins as f64;

        all_thresholds
            .iter()
            .map(|ts| {
                if ts.is_empty() {
                    return f64::INFINITY;
                }
                let mut sorted = ts.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                sorted.dedup_by(|a, b| crate::math::abs(*a - *b) < 1e-15);

                if sorted.len() < 2 {
                    return f64::INFINITY;
                }

                let mut gaps: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();

                if sorted.len() < 3 {
                    let range = sorted.last().unwrap() - sorted.first().unwrap();
                    if range < 1e-15 {
                        return f64::INFINITY;
                    }
                    return (range / n_bins) * K;
                }

                gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                let median_gap = if gaps.len() % 2 == 0 {
                    (gaps[gaps.len() / 2 - 1] + gaps[gaps.len() / 2]) / 2.0
                } else {
                    gaps[gaps.len() / 2]
                };

                if median_gap < 1e-15 {
                    f64::INFINITY
                } else {
                    median_gap * K
                }
            })
            .collect()
    }
}
