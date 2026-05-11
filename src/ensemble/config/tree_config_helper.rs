//! Helper for constructing TreeConfig from SGBT configuration.

use super::SGBTConfig;

/// Build a [`TreeConfig`] from an [`SGBTConfig`], threading every knob.
///
/// Every SGBT construction site (ctor, grow, restore) must call this function
/// rather than constructing `TreeConfig` inline. This eliminates the entire
/// class of "knob added to SGBTConfig but missed in one construction path" bugs.
///
/// The `leaf_decay_alpha` (computed from `leaf_half_life`) and `seed` (per-step
/// offset) are left at their defaults — callers must set them via `.leaf_decay_alpha_opt()`
/// and `.seed()` after calling this function.
pub(crate) fn build_tree_config(cfg: &SGBTConfig) -> crate::tree::builder::TreeConfig {
    crate::tree::builder::TreeConfig::new()
        .max_depth(cfg.max_depth)
        .n_bins(cfg.n_bins)
        .lambda(cfg.lambda)
        .gamma(cfg.gamma)
        .grace_period(cfg.grace_period)
        .delta(cfg.delta)
        .feature_subsample_rate(cfg.feature_subsample_rate)
        .split_reeval_interval_opt(cfg.split_reeval_interval)
        .feature_types_opt(cfg.feature_types.clone())
        .gradient_clip_sigma_opt(cfg.gradient_clip_sigma)
        .monotone_constraints_opt(cfg.monotone_constraints.clone())
        .max_leaf_output_opt(cfg.max_leaf_output)
        .adaptive_leaf_bound_opt(cfg.adaptive_leaf_bound)
        .adaptive_depth_opt(cfg.adaptive_depth)
        .min_hessian_sum_opt(cfg.min_hessian_sum)
        .hoeffding_r_opt(cfg.hoeffding_r)
        .leaf_model_type(cfg.leaf_model_type.clone())
}
