//! SGBT core: struct definition, Clone/Debug, and constructors.
//!
//! This module isolates the structural definition and initialization logic,
//! keeping the hot path (train_one, predict) separate for clarity.

use std::collections::VecDeque;
use std::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;

/// Cached diagnostic state for SGBT, separated from the core training state
/// to improve struct clarity and cache locality in the prediction path.
#[derive(Debug, Clone, Default)]
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
/// ```
/// use irithyll::{SGBTConfig, SGBT};
///
/// // Regression with squared loss (default):
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::new(config);
/// ```
///
/// ```
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
    /// ```
    /// use irithyll::{SGBTConfig, SGBT};
    /// use irithyll::loss::logistic::LogisticLoss;
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let model = SGBT::with_loss(config, LogisticLoss);
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L) -> Self {
        let leaf_decay_alpha = config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());

        let tree_config = crate::ensemble::config::build_tree_config(&config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let max_tree_samples = if let Some((base_mts, _)) = config.adaptive_mts {
            Some(base_mts)
        } else {
            config.max_tree_samples
        };

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
        let has_pruning = config.quality_prune_alpha.is_some();
        let prune_alpha = if config.proactive_prune_interval.is_some() {
            let hl = config.prune_half_life.unwrap_or_else(|| {
                if let Some((base_mts, _)) = config.adaptive_mts {
                    base_mts as usize
                } else if let Some(mts) = config.max_tree_samples {
                    mts as usize
                } else {
                    config.grace_period.max(1)
                }
            });
            1.0 - (-2.0 / hl.max(1) as f64).exp()
        } else {
            0.01
        };
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
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            rolling_contribution_sigma: 0.0,
            sigma_ring: VecDeque::new(),
            mts_replacement_sum: 0,
            diag: DiagnosticCache {
                contribution_accuracy: vec![0.0; n],
                prune_alpha,
                ..Default::default()
            },
        }
    }

    /// Compute contribution sigma (std dev of tree contributions for a feature vector).
    pub(crate) fn compute_contribution_sigma(&self, features: &[f64]) -> f64 {
        let n = self.steps.len();
        if n <= 1 {
            return 0.0;
        }
        let lr = self.config.learning_rate;
        let mut sum = 0.0_f64;
        let mut sq_sum = 0.0_f64;
        for step in &self.steps {
            let c = lr * step.predict(features);
            sum += c;
            sq_sum += c * c;
        }
        let nf = n as f64;
        let mean_c = sum / nf;
        let var = (sq_sum / nf) - (mean_c * mean_c);
        let var_corrected = var * nf / (nf - 1.0);
        var_corrected.max(0.0).sqrt()
    }
}
