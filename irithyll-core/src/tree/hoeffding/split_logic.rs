//! Split evaluation and Hoeffding-bound logic for streaming tree growth.

use alloc::vec::Vec;

use crate::feature::FeatureType;
use crate::histogram::BinEdges;
use crate::math;
use crate::rng::xorshift64;
use crate::tree::split::{leaf_weight, SplitCandidate, SplitCriterion, XGBoostGain};

use super::leaf::{make_binners, LeafState};

/// Tie-breaking threshold (tau). When `epsilon < tau`, we accept the best split
/// even if the gap between best and second-best gain is small, because the
/// Hoeffding bound is already tight enough that further samples won't help.
pub(crate) const TAU: f64 = 0.05;

/// Methods for split decision logic on HoeffdingTree.
///
/// Kept separate from the main struct definition for clarity. These methods handle:
/// - Feature mask generation for subsampling
/// - Candidate evaluation with XGBoost gain
/// - Hoeffding bound and CIR checks
/// - Actual split execution and child initialization
pub(crate) mod private {
    use super::*;
    use crate::histogram::bins::LeafHistograms;
    use crate::tree::builder::TreeConfig;
    use alloc::boxed::Box;

    /// Core state needed for split evaluation.
    /// This is a pass-through helper to avoid large borrowing chains.
    pub struct SplitContext<'a> {
        pub config: &'a TreeConfig,
        pub n_features: Option<usize>,
        pub n_feature_mask: &'a [usize],
        pub split_criterion: &'a XGBoostGain,
        /// RNG state for subsampling; threaded through for future use.
        #[allow(dead_code)]
        pub rng_state: &'a mut u64,
    }

    /// Collect split candidates for a single leaf by evaluating all features.
    ///
    /// Returns a Vec of (feature_idx, SplitCandidate, optional_fisher_order) tuples,
    /// sorted by gain descending.
    pub fn evaluate_split_candidates(
        histograms: &LeafHistograms,
        feature_types: Option<&[FeatureType]>,
        ctx: &SplitContext,
    ) -> Vec<(usize, SplitCandidate, Option<Vec<usize>>)> {
        let mut candidates: Vec<(usize, SplitCandidate, Option<Vec<usize>>)> = Vec::new();

        for &feat_idx in ctx.n_feature_mask {
            if feat_idx >= histograms.n_features() {
                continue;
            }
            let hist = &histograms.histograms[feat_idx];
            let total_grad = hist.total_gradient();
            let total_hess = hist.total_hessian();

            let is_categorical = feature_types
                .as_ref()
                .is_some_and(|ft| feat_idx < ft.len() && ft[feat_idx] == FeatureType::Categorical);

            if is_categorical {
                // Fisher optimal binary partitioning:
                // 1. Compute gradient_sum/hessian_sum ratio per bin
                // 2. Sort bins by this ratio
                // 3. Evaluate splits on the sorted order
                let n_bins = hist.grad_sums.len();
                if n_bins < 2 {
                    continue;
                }

                // Build (bin_index, ratio) pairs, filtering out empty bins
                let mut bin_order: Vec<usize> = (0..n_bins)
                    .filter(|&i| math::abs(hist.hess_sums[i]) > 1e-15)
                    .collect();

                if bin_order.len() < 2 {
                    continue;
                }

                // Sort by grad_sum / hess_sum ratio (ascending)
                bin_order.sort_by(|&a, &b| {
                    let ratio_a = hist.grad_sums[a] / hist.hess_sums[a];
                    let ratio_b = hist.grad_sums[b] / hist.hess_sums[b];
                    ratio_a
                        .partial_cmp(&ratio_b)
                        .unwrap_or(core::cmp::Ordering::Equal)
                });

                // Reorder grad/hess sums according to Fisher order
                let sorted_grads: Vec<f64> = bin_order.iter().map(|&i| hist.grad_sums[i]).collect();
                let sorted_hess: Vec<f64> = bin_order.iter().map(|&i| hist.hess_sums[i]).collect();

                if let Some(candidate) = ctx.split_criterion.evaluate(
                    &sorted_grads,
                    &sorted_hess,
                    total_grad,
                    total_hess,
                    ctx.config.gamma,
                    ctx.config.lambda,
                ) {
                    candidates.push((feat_idx, candidate, Some(bin_order)));
                }
            } else {
                // Standard continuous feature -- evaluate as-is
                if let Some(candidate) = ctx.split_criterion.evaluate(
                    &hist.grad_sums,
                    &hist.hess_sums,
                    total_grad,
                    total_hess,
                    ctx.config.gamma,
                    ctx.config.lambda,
                ) {
                    candidates.push((feat_idx, candidate, None));
                }
            }
        }

        // Filter out candidates that violate monotonic constraints.
        if let Some(ref mc) = ctx.config.monotone_constraints {
            candidates.retain(|(feat_idx, candidate, _)| {
                if *feat_idx >= mc.len() {
                    return true; // No constraint for this feature
                }
                let constraint = mc[*feat_idx];
                if constraint == 0 {
                    return true; // Unconstrained
                }

                let left_val =
                    leaf_weight(candidate.left_grad, candidate.left_hess, ctx.config.lambda);
                let right_val = leaf_weight(
                    candidate.right_grad,
                    candidate.right_hess,
                    ctx.config.lambda,
                );

                if constraint > 0 {
                    // Non-decreasing: left_value <= right_value
                    left_val <= right_val
                } else {
                    // Non-increasing: left_value >= right_value
                    left_val >= right_val
                }
            });
        }

        // Sort candidates by gain descending.
        candidates.sort_by(|a, b| {
            b.1.gain
                .partial_cmp(&a.1.gain)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Check if split passes Hoeffding bound and adaptive depth tests.
    ///
    /// Returns true if the split should proceed.
    pub fn should_split_hoeffding(
        best_gain: f64,
        second_best_gain: f64,
        sample_count: u64,
        ctx: &SplitContext,
    ) -> bool {
        // Per-split information criterion (Lunde-Kleppe-Skaug 2020).
        // Acts as a FIRST gate before the Hoeffding bound check.
        if let Some(cir_factor) = ctx.config.adaptive_depth {
            let n = sample_count as f64;
            if n > 1.0 {
                // Use effective_n with EWMA decay if configured
                let effective_n = match ctx.config.leaf_decay_alpha {
                    Some(alpha) => n.min(1.0 / (1.0 - alpha)),
                    None => n,
                };
                let n_feat = ctx.n_features.unwrap_or(1) as f64;
                let penalty = cir_factor / (effective_n * n_feat);

                if best_gain <= penalty {
                    return false; // Don't split — insufficient generalization evidence
                }
            }
        }

        // Hoeffding bound: epsilon = sqrt(R^2 * ln(1/delta) / (2 * n))
        // R bounds the range of the gain function. Default R=1.0 is conservative;
        // set config.hoeffding_r = sqrt(target_variance) for data-proportional bounds.
        //
        // With EWMA decay, the effective sample size is bounded by 1/(1-alpha).
        // We cap n at this value to prevent spurious splits from artificially
        // tight bounds when decay is active.
        let r = ctx.config.hoeffding_r.unwrap_or(1.0);
        let r_squared = r * r;
        let n = sample_count as f64;
        let effective_n = match ctx.config.leaf_decay_alpha {
            Some(alpha) => n.min(1.0 / (1.0 - alpha)),
            None => n,
        };
        let ln_inv_delta = math::ln(1.0 / ctx.config.delta);
        let epsilon = math::sqrt(r_squared * ln_inv_delta / (2.0 * effective_n));

        // Split condition: the best is significantly better than second-best,
        // OR the bound is already so tight that more samples won't help.
        let gap = best_gain - second_best_gain;
        !(gap <= epsilon && epsilon >= TAU)
    }

    /// Initialize child leaf states after a split.
    ///
    /// Handles histogram subtraction trick and model warm-start.
    #[allow(dead_code)]
    pub fn initialize_child_states(
        n_features: usize,
        feature_types: Option<&[FeatureType]>,
        leaf_model_factory: Option<Box<dyn Fn() -> Box<dyn crate::tree::leaf_model::LeafModel>>>,
        parent_hists: Option<LeafHistograms>,
    ) -> (LeafState, LeafState) {
        let edges_per_feature: Vec<BinEdges> = parent_hists
            .as_ref()
            .map(|h| h.histograms.iter().map(|hist| hist.edges.clone()).collect())
            .unwrap_or_default();

        let (left_hists, right_hists) = if parent_hists.is_some() {
            (
                LeafHistograms::new(&edges_per_feature),
                LeafHistograms::new(&edges_per_feature),
            )
        } else {
            (LeafHistograms::new(&[]), LeafHistograms::new(&[]))
        };

        let child_binners_l = make_binners(n_features, feature_types);
        let child_binners_r = make_binners(n_features, feature_types);

        let left_model = leaf_model_factory.as_ref().map(|f| f());
        let right_model = leaf_model_factory.as_ref().map(|f| f());

        let left_state = LeafState {
            histograms: if parent_hists.is_some() {
                Some(left_hists)
            } else {
                None
            },
            binners: child_binners_l,
            bins_ready: parent_hists.is_some(),
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
            clip_grad_mean: 0.0,
            clip_grad_m2: 0.0,
            clip_grad_count: 0,
            output_mean: 0.0,
            output_m2: 0.0,
            output_count: 0,
            leaf_model: left_model,
        };

        let right_state = LeafState {
            histograms: if parent_hists.is_some() {
                Some(right_hists)
            } else {
                None
            },
            binners: child_binners_r,
            bins_ready: parent_hists.is_some(),
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
            clip_grad_mean: 0.0,
            clip_grad_m2: 0.0,
            clip_grad_count: 0,
            output_mean: 0.0,
            output_m2: 0.0,
            output_count: 0,
            leaf_model: right_model,
        };

        (left_state, right_state)
    }
}

/// Feature mask generation with O(1) membership testing.
///
/// If `feature_subsample_rate` is 1.0, all features are included.
/// Otherwise, a random subset is selected via xorshift64.
pub(crate) fn generate_feature_mask(
    mut feature_mask: Vec<usize>,
    mut feature_mask_bits: Vec<u64>,
    rng_state: &mut u64,
    subsample_rate: f64,
    n_features: usize,
) -> (Vec<usize>, Vec<u64>) {
    feature_mask.clear();

    if subsample_rate >= 1.0 {
        feature_mask.extend(0..n_features);
        return (feature_mask, feature_mask_bits);
    }

    let target_count = math::ceil((n_features as f64) * subsample_rate) as usize;
    let target_count = target_count.max(1).min(n_features);

    // Prepare the bitset: one bit per feature, O(1) membership test.
    let n_words = n_features.div_ceil(64);
    feature_mask_bits.clear();
    feature_mask_bits.resize(n_words, 0u64);

    // Include each feature with probability = subsample_rate.
    for i in 0..n_features {
        let r = xorshift64(rng_state);
        let p = (r as f64) / (u64::MAX as f64);
        if p < subsample_rate {
            feature_mask.push(i);
            feature_mask_bits[i / 64] |= 1u64 << (i % 64);
        }
    }

    // If we didn't get enough features, fill up deterministically.
    // Now O(n) instead of O(n²) thanks to the bitset.
    if feature_mask.len() < target_count {
        for i in 0..n_features {
            if feature_mask.len() >= target_count {
                break;
            }
            if feature_mask_bits[i / 64] & (1u64 << (i % 64)) == 0 {
                feature_mask.push(i);
                feature_mask_bits[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    (feature_mask, feature_mask_bits)
}
