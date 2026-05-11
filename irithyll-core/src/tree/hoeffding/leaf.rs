//! Per-leaf state and histogram management for Hoeffding tree splits.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::feature::FeatureType;
use crate::histogram::bins::LeafHistograms;
use crate::histogram::BinnerKind;
use crate::math;
use crate::tree::leaf_model::LeafModel;

/// State tracked per leaf node for split decisions.
///
/// Each active leaf owns its own set of histogram accumulators (one per feature)
/// and running gradient/hessian sums for leaf weight updates.
pub(crate) struct LeafState {
    /// Histogram accumulators for this leaf. `None` until bin edges are computed
    /// (after the grace period).
    pub histograms: Option<LeafHistograms>,

    /// Per-feature binning strategies that collect observed values to compute
    /// bin edges. Uses `BinnerKind` enum dispatch instead of `Box<dyn>` to
    /// eliminate N_features heap allocations per new leaf.
    pub binners: Vec<BinnerKind>,

    /// Whether bin edges have been computed (after grace period samples).
    pub bins_ready: bool,

    /// Running gradient sum for leaf weight updates.
    pub grad_sum: f64,

    /// Running hessian sum for leaf weight updates.
    pub hess_sum: f64,

    /// Sample count at last split re-evaluation (for EFDT-inspired re-eval).
    pub last_reeval_count: u64,

    /// EWMA gradient mean for gradient clipping (Welford online algorithm).
    pub clip_grad_mean: f64,

    /// EWMA gradient M2 accumulator (Welford) for variance estimation.
    pub clip_grad_m2: f64,

    /// Number of gradients observed for clipping statistics.
    pub clip_grad_count: u64,

    /// EWMA/Welford mean of this leaf's output weight (for adaptive bounds).
    pub output_mean: f64,

    /// EWMA/Welford M2 (variance accumulator) of this leaf's output weight.
    pub output_m2: f64,

    /// Number of output weight observations.
    pub output_count: u64,

    /// Optional trainable leaf model (linear / MLP). `None` for closed-form leaves.
    pub leaf_model: Option<Box<dyn LeafModel>>,
}

impl Clone for LeafState {
    fn clone(&self) -> Self {
        Self {
            histograms: self.histograms.clone(),
            binners: self.binners.clone(),
            bins_ready: self.bins_ready,
            grad_sum: self.grad_sum,
            hess_sum: self.hess_sum,
            last_reeval_count: self.last_reeval_count,
            clip_grad_mean: self.clip_grad_mean,
            clip_grad_m2: self.clip_grad_m2,
            clip_grad_count: self.clip_grad_count,
            output_mean: self.output_mean,
            output_m2: self.output_m2,
            output_count: self.output_count,
            leaf_model: self.leaf_model.as_ref().map(|m| m.clone_warm()),
        }
    }
}

/// Clip a gradient using Welford online stats tracked per leaf.
///
/// Updates running mean/variance, then clamps the gradient to `mean ± sigma * std_dev`.
/// Returns the (possibly clamped) gradient. During warmup (< 10 samples), no clipping
/// is applied to let the statistics stabilize.
#[inline]
pub(crate) fn clip_gradient(state: &mut LeafState, gradient: f64, sigma: f64) -> f64 {
    state.clip_grad_count += 1;
    let n = state.clip_grad_count as f64;

    // Welford online update
    let delta = gradient - state.clip_grad_mean;
    state.clip_grad_mean += delta / n;
    let delta2 = gradient - state.clip_grad_mean;
    state.clip_grad_m2 += delta * delta2;

    // No clipping during warmup
    if state.clip_grad_count < 10 {
        return gradient;
    }

    let variance = state.clip_grad_m2 / (n - 1.0);
    let std_dev = math::sqrt(variance);

    if std_dev < 1e-15 {
        return gradient; // All gradients identical -- no clipping needed
    }

    let lo = state.clip_grad_mean - sigma * std_dev;
    let hi = state.clip_grad_mean + sigma * std_dev;
    gradient.clamp(lo, hi)
}

/// Update per-leaf output weight tracking for adaptive bounds.
///
/// If `decay_alpha` is Some, uses EWMA synchronized with leaf_decay_alpha.
/// Otherwise uses Welford online algorithm (batch scenarios).
#[inline]
pub(crate) fn update_output_stats(state: &mut LeafState, weight: f64, decay_alpha: Option<f64>) {
    state.output_count += 1;

    if let Some(alpha) = decay_alpha {
        // EWMA — synchronized with leaf gradient decay
        if state.output_count == 1 {
            state.output_mean = weight;
            state.output_m2 = 0.0;
        } else {
            let diff = weight - state.output_mean;
            state.output_mean = alpha * state.output_mean + (1.0 - alpha) * weight;
            let diff2 = weight - state.output_mean;
            state.output_m2 = alpha * state.output_m2 + (1.0 - alpha) * diff * diff2;
        }
    } else {
        // Welford online (no decay — batch scenarios)
        let delta = weight - state.output_mean;
        state.output_mean += delta / (state.output_count as f64);
        let delta2 = weight - state.output_mean;
        state.output_m2 += delta * delta2;
    }
}

/// Get the adaptive output bound for this leaf.
///
/// Returns `|mean| + k * std`, with a floor of 0.01 to never fully suppress a leaf.
/// During warmup (< 10 samples), returns `f64::MAX` (no bound).
#[inline]
pub(crate) fn adaptive_bound(state: &LeafState, k: f64, decay_alpha: Option<f64>) -> f64 {
    if state.output_count < 10 {
        return f64::MAX; // warmup — no bound yet
    }

    let variance = if decay_alpha.is_some() {
        // EWMA variance is stored directly in output_m2
        state.output_m2.max(0.0)
    } else {
        // Welford: variance = M2 / (n - 1)
        state.output_m2 / (state.output_count as f64 - 1.0)
    };
    let std = math::sqrt(variance);

    // Bound = |mean| + k * std, floor at 0.01 to never fully suppress a leaf
    (math::abs(state.output_mean) + k * std).max(0.01)
}

/// Create binners according to feature types.
pub(crate) fn make_binners(
    n_features: usize,
    feature_types: Option<&[FeatureType]>,
) -> Vec<BinnerKind> {
    (0..n_features)
        .map(|i| {
            if let Some(ft) = feature_types {
                if i < ft.len() && ft[i] == FeatureType::Categorical {
                    return BinnerKind::categorical();
                }
            }
            BinnerKind::uniform()
        })
        .collect()
}

impl LeafState {
    /// Create a fresh leaf state for a leaf with `n_features` features.
    pub(crate) fn new(n_features: usize) -> Self {
        Self::new_with_types(n_features, None)
    }

    /// Create a fresh leaf state respecting per-feature type declarations.
    pub(crate) fn new_with_types(n_features: usize, feature_types: Option<&[FeatureType]>) -> Self {
        let binners = make_binners(n_features, feature_types);

        Self {
            histograms: None,
            binners,
            bins_ready: false,
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
            clip_grad_mean: 0.0,
            clip_grad_m2: 0.0,
            clip_grad_count: 0,
            output_mean: 0.0,
            output_m2: 0.0,
            output_count: 0,
            leaf_model: None,
        }
    }

    /// Create a leaf state with pre-computed histograms (used after a split
    /// when we can initialize the child via histogram subtraction).
    #[allow(dead_code)]
    pub(crate) fn with_histograms(histograms: LeafHistograms) -> Self {
        let n_features = histograms.n_features();
        let binners: Vec<BinnerKind> = (0..n_features).map(|_| BinnerKind::uniform()).collect();

        // Recover grad/hess sums from the histograms.
        let grad_sum: f64 = histograms
            .histograms
            .first()
            .map_or(0.0, |h| h.total_gradient());
        let hess_sum: f64 = histograms
            .histograms
            .first()
            .map_or(0.0, |h| h.total_hessian());

        Self {
            histograms: Some(histograms),
            binners,
            bins_ready: true,
            grad_sum,
            hess_sum,
            last_reeval_count: 0,
            clip_grad_mean: 0.0,
            clip_grad_m2: 0.0,
            clip_grad_count: 0,
            output_mean: 0.0,
            output_m2: 0.0,
            output_count: 0,
            leaf_model: None,
        }
    }
}
