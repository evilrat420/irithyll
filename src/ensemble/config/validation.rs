//! Validation logic for SGBT configuration builder.

use crate::error::{ConfigError, Result};

use super::DriftDetectorType;
use super::SGBTConfig;
use crate::ensemble::variants::SGBTVariant;

/// Validate and build the configuration.
///
/// # Errors
///
/// Returns [`InvalidConfig`](crate::IrithyllError::InvalidConfig) with a structured
/// [`ConfigError`] if any parameter is out of its valid range.
pub(super) fn validate_and_build(config: SGBTConfig) -> Result<SGBTConfig> {
    let c = &config;

    // -- Ensemble-level parameters --
    if c.n_steps == 0 {
        return Err(ConfigError::out_of_range("n_steps", "must be > 0", c.n_steps).into());
    }
    if c.learning_rate <= 0.0 || c.learning_rate > 1.0 {
        return Err(ConfigError::out_of_range(
            "learning_rate",
            "must be in (0, 1]",
            c.learning_rate,
        )
        .into());
    }
    if c.feature_subsample_rate <= 0.0 || c.feature_subsample_rate > 1.0 {
        return Err(ConfigError::out_of_range(
            "feature_subsample_rate",
            "must be in (0, 1]",
            c.feature_subsample_rate,
        )
        .into());
    }

    // -- Tree-level parameters --
    if c.max_depth == 0 {
        return Err(ConfigError::out_of_range("max_depth", "must be > 0", c.max_depth).into());
    }
    if c.n_bins < 2 {
        return Err(ConfigError::out_of_range("n_bins", "must be >= 2", c.n_bins).into());
    }
    if c.lambda < 0.0 {
        return Err(ConfigError::out_of_range("lambda", "must be >= 0", c.lambda).into());
    }
    if c.gamma < 0.0 {
        return Err(ConfigError::out_of_range("gamma", "must be >= 0", c.gamma).into());
    }
    if c.grace_period == 0 {
        return Err(
            ConfigError::out_of_range("grace_period", "must be > 0", c.grace_period).into(),
        );
    }
    if c.delta <= 0.0 || c.delta >= 1.0 {
        return Err(ConfigError::out_of_range("delta", "must be in (0, 1)", c.delta).into());
    }

    if c.initial_target_count == 0 {
        return Err(ConfigError::out_of_range(
            "initial_target_count",
            "must be > 0",
            c.initial_target_count,
        )
        .into());
    }

    // -- Streaming adaptation parameters --
    if let Some(hl) = c.leaf_half_life {
        if hl == 0 {
            return Err(ConfigError::out_of_range("leaf_half_life", "must be >= 1", hl).into());
        }
    }
    if let Some(max) = c.max_tree_samples {
        if max < 100 {
            return Err(
                ConfigError::out_of_range("max_tree_samples", "must be >= 100", max).into(),
            );
        }
    }
    if let Some(interval) = c.split_reeval_interval {
        if interval < c.grace_period {
            return Err(ConfigError::invalid(
                "split_reeval_interval",
                format!(
                    "must be >= grace_period ({}), got {}",
                    c.grace_period, interval
                ),
            )
            .into());
        }
    }

    // -- Feature names --
    if let Some(ref names) = c.feature_names {
        let mut seen = std::collections::HashSet::new();
        for name in names {
            if !seen.insert(name.as_str()) {
                return Err(ConfigError::invalid(
                    "feature_names",
                    format!("duplicate feature name: '{}'", name),
                )
                .into());
            }
        }
    }

    // -- Feature types --
    if let Some(ref types) = c.feature_types {
        if let Some(ref names) = c.feature_names {
            if !names.is_empty() && !types.is_empty() && names.len() != types.len() {
                return Err(ConfigError::invalid(
                    "feature_types",
                    format!(
                        "length ({}) must match feature_names length ({})",
                        types.len(),
                        names.len()
                    ),
                )
                .into());
            }
        }
    }

    // -- Gradient clipping --
    if let Some(sigma) = c.gradient_clip_sigma {
        if sigma <= 0.0 {
            return Err(
                ConfigError::out_of_range("gradient_clip_sigma", "must be > 0", sigma).into(),
            );
        }
    }

    // -- Monotonic constraints --
    if let Some(ref mc) = c.monotone_constraints {
        for (i, &v) in mc.iter().enumerate() {
            if v != -1 && v != 0 && v != 1 {
                return Err(ConfigError::invalid(
                    "monotone_constraints",
                    format!("feature {}: must be -1, 0, or +1, got {}", i, v),
                )
                .into());
            }
        }
    }

    // -- Leaf output clamping --
    if let Some(max) = c.max_leaf_output {
        if max <= 0.0 {
            return Err(ConfigError::out_of_range("max_leaf_output", "must be > 0", max).into());
        }
    }

    // -- Per-leaf adaptive output bound --
    if let Some(k) = c.adaptive_leaf_bound {
        if k <= 0.0 {
            return Err(ConfigError::out_of_range("adaptive_leaf_bound", "must be > 0", k).into());
        }
    }

    // -- Adaptive depth (per-split information criterion) --
    if let Some(factor) = c.adaptive_depth {
        if factor <= 0.0 {
            return Err(ConfigError::out_of_range("adaptive_depth", "must be > 0", factor).into());
        }
    }

    // -- Minimum hessian sum --
    if let Some(min_h) = c.min_hessian_sum {
        if min_h <= 0.0 {
            return Err(ConfigError::out_of_range("min_hessian_sum", "must be > 0", min_h).into());
        }
    }

    // -- Huber loss multiplier --
    if let Some(k) = c.huber_k {
        if k <= 0.0 {
            return Err(ConfigError::out_of_range("huber_k", "must be > 0", k).into());
        }
    }

    // -- Empirical sigma alpha --
    if !(0.0..=1.0).contains(&c.empirical_sigma_alpha) {
        return Err(ConfigError::out_of_range(
            "empirical_sigma_alpha",
            "must be in [0.0, 1.0]",
            c.empirical_sigma_alpha,
        )
        .into());
    }

    // -- Shadow warmup --
    if let Some(warmup) = c.shadow_warmup {
        if warmup == 0 {
            return Err(
                ConfigError::out_of_range("shadow_warmup", "must be > 0", warmup as f64).into(),
            );
        }
    }

    // -- Quality-based pruning parameters --
    if let Some(alpha) = c.quality_prune_alpha {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ConfigError::out_of_range(
                "quality_prune_alpha",
                "must be in (0, 1)",
                alpha,
            )
            .into());
        }
    }
    if c.quality_prune_threshold <= 0.0 {
        return Err(ConfigError::out_of_range(
            "quality_prune_threshold",
            "must be > 0",
            c.quality_prune_threshold,
        )
        .into());
    }
    if c.quality_prune_patience == 0 {
        return Err(ConfigError::out_of_range(
            "quality_prune_patience",
            "must be > 0",
            c.quality_prune_patience,
        )
        .into());
    }

    // -- Error-weighted sample importance --
    if let Some(alpha) = c.error_weight_alpha {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ConfigError::out_of_range(
                "error_weight_alpha",
                "must be in (0, 1)",
                alpha,
            )
            .into());
        }
    }

    // -- Adaptive MTS --
    if let Some((base_mts, k)) = c.adaptive_mts {
        if base_mts < 100 {
            return Err(ConfigError::out_of_range(
                "adaptive_mts.base_mts",
                "must be >= 100",
                base_mts,
            )
            .into());
        }
        if k <= 0.0 {
            return Err(ConfigError::out_of_range("adaptive_mts.k", "must be > 0", k).into());
        }
    }
    if !(0.0..=1.0).contains(&c.adaptive_mts_floor) {
        return Err(ConfigError::out_of_range(
            "adaptive_mts_floor",
            "must be in [0.0, 1.0]",
            c.adaptive_mts_floor,
        )
        .into());
    }

    // -- Proactive prune interval --
    if let Some(interval) = c.proactive_prune_interval {
        if interval < 10 {
            return Err(ConfigError::out_of_range(
                "proactive_prune_interval",
                "must be >= 10",
                interval,
            )
            .into());
        }
    }

    // -- Hoeffding R --
    if let Some(r) = c.hoeffding_r {
        if !(r > 0.0 && r.is_finite()) {
            return Err(
                ConfigError::out_of_range("hoeffding_r", "must be finite and > 0", r).into(),
            );
        }
    }

    // -- Packed cache refresh interval --
    if c.packed_refresh_interval != 0 && c.packed_refresh_interval < 10 {
        return Err(ConfigError::out_of_range(
            "packed_refresh_interval",
            "must be 0 (disabled) or >= 10",
            c.packed_refresh_interval,
        )
        .into());
    }

    // -- Drift detector parameters --
    match &c.drift_detector {
        DriftDetectorType::PageHinkley { delta, lambda } => {
            if *delta <= 0.0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.PageHinkley.delta",
                    "must be > 0",
                    delta,
                )
                .into());
            }
            if *lambda <= 0.0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.PageHinkley.lambda",
                    "must be > 0",
                    lambda,
                )
                .into());
            }
        }
        DriftDetectorType::Adwin { delta } => {
            if *delta <= 0.0 || *delta >= 1.0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.Adwin.delta",
                    "must be in (0, 1)",
                    delta,
                )
                .into());
            }
        }
        DriftDetectorType::Ddm {
            warning_level,
            drift_level,
            min_instances,
        } => {
            if *warning_level <= 0.0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.Ddm.warning_level",
                    "must be > 0",
                    warning_level,
                )
                .into());
            }
            if *drift_level <= 0.0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.Ddm.drift_level",
                    "must be > 0",
                    drift_level,
                )
                .into());
            }
            if *drift_level <= *warning_level {
                return Err(ConfigError::invalid(
                    "drift_detector.Ddm.drift_level",
                    format!(
                        "must be > warning_level ({}), got {}",
                        warning_level, drift_level
                    ),
                )
                .into());
            }
            if *min_instances == 0 {
                return Err(ConfigError::out_of_range(
                    "drift_detector.Ddm.min_instances",
                    "must be > 0",
                    min_instances,
                )
                .into());
            }
        }
    }

    // -- Variant parameters --
    match &c.variant {
        SGBTVariant::Standard => {} // no extra validation
        SGBTVariant::Skip { k } => {
            if *k == 0 {
                return Err(ConfigError::out_of_range("variant.Skip.k", "must be > 0", k).into());
            }
        }
        SGBTVariant::MultipleIterations { multiplier } => {
            if *multiplier <= 0.0 {
                return Err(ConfigError::out_of_range(
                    "variant.MultipleIterations.multiplier",
                    "must be > 0",
                    multiplier,
                )
                .into());
            }
        }
    }

    Ok(config)
}
