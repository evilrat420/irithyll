//! Data-derived config bounds using SRM complexity budget.
//!
//! The [`FeasibleRegion`] derives feasible configuration space from observed
//! data characteristics (sample count, feature count, target variance).
//! Bounds respect structural regularization margin constraints.

use crate::ensemble::config::SGBTConfig;

/// Data-derived config bounds using SRM complexity budget.
///
/// The complexity budget constrains the config space:
/// `eta * n_steps * 2^max_depth <= n_samples * epsilon^2 / ln(n_features)`
#[derive(Debug, Clone)]
pub struct FeasibleRegion {
    n_samples: usize,
    n_features: usize,
    target_epsilon: f64,
    budget: f64,
}

impl FeasibleRegion {
    /// Derive feasible region from data characteristics.
    ///
    /// `target_variance` is the observed variance of the target variable
    /// (used to set target_epsilon = sqrt(target_variance) * 0.1).
    pub fn from_data(n_samples: usize, n_features: usize, target_variance: f64) -> Self {
        let target_epsilon = target_variance.sqrt().max(1e-8) * 0.1;
        let log_p = (n_features as f64).max(1.0).ln().max(1.0);
        let budget = n_samples as f64 * target_epsilon.powi(2) / log_p;

        Self {
            n_samples,
            n_features,
            target_epsilon,
            budget,
        }
    }

    /// Derive config bounds from the complexity budget.
    pub fn config_bounds(&self) -> super::ConfigBounds {
        let max_depth_upper =
            ((self.budget.max(1.0).ln() / core::f64::consts::LN_2).floor() as usize).clamp(2, 6);

        let n_steps_upper = ((self.budget / 4.0).ceil() as usize).clamp(3, 50);

        // Hoeffding-derived grace period: R^2 * ln(1/delta) / (2 * margin^2)
        // R=1.0 (normalized), delta=0.05, margin=target_epsilon
        let gp_upper = ((2.0 * (1.0_f64 / 0.05).ln() / self.target_epsilon.powi(2).max(1e-10))
            .ceil() as usize)
            .clamp(3, 200);

        let n_bins_upper = 64usize.min(self.n_samples / 4).max(8);

        // Lambda bounds: derived from target scale so regularization matches
        // gradient magnitude. High-variance targets need higher lambda, low-variance
        // need lower. Prevents stumpy trees from over-regularization.
        let target_std = self.target_epsilon * 10.0; // undo the 0.1 scaling from from_data()
        let lambda_center = target_std.max(0.01);
        let lambda_bounds = (
            (lambda_center * 0.1).clamp(0.1, 5.0),
            (lambda_center * 3.0).clamp(0.1, 5.0),
        );

        super::ConfigBounds {
            max_depth: (2, max_depth_upper),
            n_steps: (3, n_steps_upper),
            grace_period: (3, gp_upper),
            learning_rate: (0.05, 0.3),
            lambda: lambda_bounds,
            n_bins: (8, n_bins_upper),
            feature_subsample: (0.5, 1.0),
        }
    }

    /// Build an SGBTConfig at the CENTER of the feasible region.
    pub fn center_config(&self) -> SGBTConfig {
        let bounds = self.config_bounds();
        SGBTConfig::builder()
            .max_depth((bounds.max_depth.0 + bounds.max_depth.1) / 2)
            .n_steps((bounds.n_steps.0 + bounds.n_steps.1) / 2)
            .grace_period((bounds.grace_period.0 + bounds.grace_period.1) / 2)
            .learning_rate((bounds.learning_rate.0 * bounds.learning_rate.1).sqrt())
            .lambda((bounds.lambda.0 * bounds.lambda.1).sqrt())
            .n_bins((bounds.n_bins.0 + bounds.n_bins.1) / 2)
            .feature_subsample_rate((bounds.feature_subsample.0 + bounds.feature_subsample.1) / 2.0)
            .build()
            .expect("FeasibleRegion::center_config: feasible region produced invalid config")
    }

    /// Generate center + directional perturbation configs.
    ///
    /// Returns 1 center config + up to 14 perturbations (2 per 7 params).
    /// Each perturbation sets one param to its lower or upper bound while
    /// keeping all other params at center values.
    pub fn perturbation_configs(&self) -> Vec<SGBTConfig> {
        let bounds = self.config_bounds();
        let center = self.center_config();
        let mut configs = vec![center.clone()];

        // Helper: center values for building perturbation configs.
        let center_depth = (bounds.max_depth.0 + bounds.max_depth.1) / 2;
        let center_steps = (bounds.n_steps.0 + bounds.n_steps.1) / 2;
        let center_gp = (bounds.grace_period.0 + bounds.grace_period.1) / 2;
        let center_lr = (bounds.learning_rate.0 * bounds.learning_rate.1).sqrt();
        let center_lambda = (bounds.lambda.0 * bounds.lambda.1).sqrt();
        let center_bins = (bounds.n_bins.0 + bounds.n_bins.1) / 2;
        let center_fsr = (bounds.feature_subsample.0 + bounds.feature_subsample.1) / 2.0;

        // Macro to build a config with one parameter overridden from center.
        // Since SGBTConfig fields are pub and it derives Clone, we clone and mutate.
        macro_rules! perturb {
            ($field:ident, $val:expr) => {{
                let mut cfg = center.clone();
                cfg.$field = $val;
                configs.push(cfg);
            }};
        }

        // max_depth perturbations
        if bounds.max_depth.0 != bounds.max_depth.1 {
            perturb!(max_depth, bounds.max_depth.0);
            perturb!(max_depth, bounds.max_depth.1);
        }

        // n_steps perturbations
        if bounds.n_steps.0 != bounds.n_steps.1 {
            perturb!(n_steps, bounds.n_steps.0);
            perturb!(n_steps, bounds.n_steps.1);
        }

        // grace_period perturbations
        if bounds.grace_period.0 != bounds.grace_period.1 {
            perturb!(grace_period, bounds.grace_period.0);
            perturb!(grace_period, bounds.grace_period.1);
        }

        // learning_rate perturbations
        perturb!(learning_rate, bounds.learning_rate.0);
        perturb!(learning_rate, bounds.learning_rate.1);

        // lambda perturbations
        perturb!(lambda, bounds.lambda.0);
        perturb!(lambda, bounds.lambda.1);

        // n_bins perturbations
        if bounds.n_bins.0 != bounds.n_bins.1 {
            perturb!(n_bins, bounds.n_bins.0);
            perturb!(n_bins, bounds.n_bins.1);
        }

        // feature_subsample_rate perturbations
        perturb!(feature_subsample_rate, bounds.feature_subsample.0);
        perturb!(feature_subsample_rate, bounds.feature_subsample.1);

        // Suppress unused variable warnings for the center values used by
        // the macro's field-mutation approach rather than a builder rebuild.
        let _ = (
            center_depth,
            center_steps,
            center_gp,
            center_lr,
            center_lambda,
            center_bins,
            center_fsr,
        );

        configs
    }

    /// Update bounds as more data arrives.
    pub fn update(&mut self, n_samples: usize) {
        self.n_samples = n_samples;
        let log_p = (self.n_features as f64).max(1.0).ln().max(1.0);
        self.budget = n_samples as f64 * self.target_epsilon.powi(2) / log_p;
    }

    /// Update target variance estimate from observed data.
    ///
    /// Should be called periodically as the stream progresses so that
    /// config bounds (especially lambda and grace period) stay calibrated.
    pub fn update_variance(&mut self, target_variance: f64) {
        self.target_epsilon = target_variance.sqrt().max(1e-8) * 0.1;
        // Recompute budget with updated epsilon.
        let log_p = (self.n_features as f64).max(1.0).ln().max(1.0);
        self.budget = self.n_samples as f64 * self.target_epsilon.powi(2) / log_p;
    }

    /// Current complexity budget.
    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Current sample count.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Current feature count.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}
