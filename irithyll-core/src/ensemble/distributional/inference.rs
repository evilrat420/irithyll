//! Inference (prediction) logic for distributional SGBT.

use super::{DistributionalSGBT, GaussianPrediction};
use crate::ensemble::config::ScaleMode;

pub(crate) fn predict_distributional(
    model: &DistributionalSGBT,
    features: &[f64],
) -> GaussianPrediction {
    // Try packed cache for mu if available (no_std skips this)
    let mu = predict_full_trees(model, features);

    let (sigma, log_sigma) = match model.scale_mode {
        ScaleMode::Empirical => {
            let s = crate::math::sqrt(model.ewma_sq_err).max(1e-8);
            (s, crate::math::ln(s))
        }
        ScaleMode::TreeChain => {
            let mut ls = model.scale_base;
            if model.auto_bandwidths.is_empty() {
                for s in 0..model.scale_steps.len() {
                    ls += model.config.learning_rate * model.scale_steps[s].predict(features);
                }
            } else {
                for s in 0..model.scale_steps.len() {
                    ls += model.config.learning_rate
                        * model.scale_steps[s]
                            .predict_smooth_auto(features, &model.auto_bandwidths);
                }
            }
            (crate::math::exp(ls).max(1e-8), ls)
        }
    };

    let honest_sigma = compute_honest_sigma(model, features);

    GaussianPrediction {
        mu,
        sigma,
        log_sigma,
        honest_sigma,
    }
}

pub(crate) fn predict_full_trees(model: &DistributionalSGBT, features: &[f64]) -> f64 {
    let mut mu = model.location_base;
    if model.auto_bandwidths.is_empty() {
        for s in 0..model.location_steps.len() {
            mu += model.config.learning_rate * model.location_steps[s].predict(features);
        }
    } else {
        for s in 0..model.location_steps.len() {
            mu += model.config.learning_rate
                * model.location_steps[s].predict_smooth_auto(features, &model.auto_bandwidths);
        }
    }
    mu
}

pub(crate) fn predict_smooth(
    model: &DistributionalSGBT,
    features: &[f64],
    bandwidth: f64,
) -> GaussianPrediction {
    let mut mu = model.location_base;
    for s in 0..model.location_steps.len() {
        mu += model.config.learning_rate
            * model.location_steps[s].predict_smooth(features, bandwidth);
    }

    let (sigma, log_sigma) = match model.scale_mode {
        ScaleMode::Empirical => {
            let s = crate::math::sqrt(model.ewma_sq_err).max(1e-8);
            (s, crate::math::ln(s))
        }
        ScaleMode::TreeChain => {
            let mut ls = model.scale_base;
            for s in 0..model.scale_steps.len() {
                ls += model.config.learning_rate
                    * model.scale_steps[s].predict_smooth(features, bandwidth);
            }
            (crate::math::exp(ls).max(1e-8), ls)
        }
    };

    let honest_sigma = compute_honest_sigma(model, features);

    GaussianPrediction {
        mu,
        sigma,
        log_sigma,
        honest_sigma,
    }
}

pub(crate) fn predict_interpolated(
    model: &DistributionalSGBT,
    features: &[f64],
) -> GaussianPrediction {
    let mut mu = model.location_base;
    for s in 0..model.location_steps.len() {
        mu += model.config.learning_rate * model.location_steps[s].predict_interpolated(features);
    }

    let (sigma, log_sigma) = match model.scale_mode {
        ScaleMode::Empirical => {
            let s = crate::math::sqrt(model.ewma_sq_err).max(1e-8);
            (s, crate::math::ln(s))
        }
        ScaleMode::TreeChain => {
            let mut ls = model.scale_base;
            for s in 0..model.scale_steps.len() {
                ls += model.config.learning_rate
                    * model.scale_steps[s].predict_interpolated(features);
            }
            (crate::math::exp(ls).max(1e-8), ls)
        }
    };

    let honest_sigma = compute_honest_sigma(model, features);

    GaussianPrediction {
        mu,
        sigma,
        log_sigma,
        honest_sigma,
    }
}

pub(crate) fn predict_sibling_interpolated(
    model: &DistributionalSGBT,
    features: &[f64],
) -> GaussianPrediction {
    let mut mu = model.location_base;
    for s in 0..model.location_steps.len() {
        mu += model.config.learning_rate
            * model.location_steps[s]
                .predict_sibling_interpolated(features, &model.auto_bandwidths);
    }

    let (sigma, log_sigma) = match model.scale_mode {
        ScaleMode::Empirical => {
            let s = crate::math::sqrt(model.ewma_sq_err).max(1e-8);
            (s, crate::math::ln(s))
        }
        ScaleMode::TreeChain => {
            let mut ls = model.scale_base;
            for s in 0..model.scale_steps.len() {
                ls += model.config.learning_rate
                    * model.scale_steps[s]
                        .predict_sibling_interpolated(features, &model.auto_bandwidths);
            }
            (crate::math::exp(ls).max(1e-8), ls)
        }
    };

    let honest_sigma = compute_honest_sigma(model, features);

    GaussianPrediction {
        mu,
        sigma,
        log_sigma,
        honest_sigma,
    }
}

fn compute_honest_sigma(model: &DistributionalSGBT, features: &[f64]) -> f64 {
    let n = model.location_steps.len();
    if n <= 1 {
        return 0.0;
    }
    let lr = model.config.learning_rate;
    let mut sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    for step in &model.location_steps {
        let c = lr * step.predict(features);
        sum += c;
        sq_sum += c * c;
    }
    let nf = n as f64;
    let mean_c = sum / nf;
    let var = (sq_sum / nf) - (mean_c * mean_c);
    let var_corrected = var * nf / (nf - 1.0);
    crate::math::sqrt(var_corrected.max(0.0))
}
