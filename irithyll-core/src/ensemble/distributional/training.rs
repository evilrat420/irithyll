//! Training logic for distributional SGBT.

use super::DistributionalSGBT;
use crate::ensemble::config::ScaleMode;
use crate::sample::Observation;

pub(crate) fn train_distributional_one(model: &mut DistributionalSGBT, sample: &impl Observation) {
    model.samples_seen += 1;
    let target = sample.target();
    let features = sample.features();

    // Guard: skip non-finite inputs to prevent NaN/Inf corruption
    if !target.is_finite() || !features.iter().all(|f| f.is_finite()) {
        return;
    }

    // Initialize base predictions from first few targets
    if !model.base_initialized {
        model.initial_targets.push(target);
        if model.initial_targets.len() >= model.initial_target_count {
            // Location base = mean
            let sum: f64 = model.initial_targets.iter().sum();
            let mean = sum / model.initial_targets.len() as f64;
            model.location_base = mean;

            // Scale base = log(std) -- clamped for stability
            let var: f64 = model
                .initial_targets
                .iter()
                .map(|&y| (y - mean) * (y - mean))
                .sum::<f64>()
                / model.initial_targets.len() as f64;
            let initial_std = crate::math::sqrt(var).max(1e-6);
            model.scale_base = crate::math::ln(initial_std);

            // Initialize rolling sigma mean and ewma from initial targets std
            model.rolling_sigma_mean = initial_std;
            model.ewma_sq_err = var.max(1e-12);

            // Initialize PD sigma state
            model.prev_sigma = initial_std;
            model.sigma_velocity = 0.0;

            model.base_initialized = true;
            model.initial_targets.clear();
            model.initial_targets.shrink_to_fit();
        }
        return;
    }

    match model.scale_mode {
        ScaleMode::Empirical => train_empirical(model, target, features),
        ScaleMode::TreeChain => train_tree_chain(model, target, features),
    }
}

fn train_empirical(model: &mut DistributionalSGBT, target: f64, features: &[f64]) {
    // Current location prediction (before this step's update)
    let mut mu = model.location_base;
    for s in 0..model.location_steps.len() {
        mu += model.config.learning_rate * model.location_steps[s].predict(features);
    }

    // Update rolling honest_sigma mean (tree contribution variance EWMA)
    let honest_sigma = compute_honest_sigma(model, features);
    const HONEST_SIGMA_ALPHA: f64 = 0.001;
    model.rolling_honest_sigma_mean = (1.0 - HONEST_SIGMA_ALPHA) * model.rolling_honest_sigma_mean
        + HONEST_SIGMA_ALPHA * honest_sigma;

    // Empirical sigma from EWMA of squared prediction errors
    let err = target - mu;
    let alpha = model.empirical_sigma_alpha;
    model.ewma_sq_err = (1.0 - alpha) * model.ewma_sq_err + alpha * err * err;
    let empirical_sigma = crate::math::sqrt(model.ewma_sq_err).max(1e-8);

    // Compute σ-ratio for uncertainty-modulated learning rate (PD controller)
    let sigma_ratio = if model.uncertainty_modulated_lr {
        // Compute sigma velocity (derivative of sigma over time)
        let d_sigma = empirical_sigma - model.prev_sigma;
        model.prev_sigma = empirical_sigma;

        // EWMA-smooth the velocity (same alpha as empirical sigma for synchronization)
        model.sigma_velocity = (1.0 - alpha) * model.sigma_velocity + alpha * d_sigma;

        // Adaptive derivative gain: self-calibrating, no config needed
        let k_d = if model.rolling_sigma_mean > 1e-12 {
            crate::math::abs(model.sigma_velocity) / model.rolling_sigma_mean
        } else {
            0.0
        };

        // PD ratio: proportional + derivative
        let pd_sigma = empirical_sigma + k_d * model.sigma_velocity;
        let ratio = (pd_sigma / model.rolling_sigma_mean).clamp(0.1, 10.0);

        // Update rolling sigma mean with slow EWMA
        const SIGMA_EWMA_ALPHA: f64 = 0.001;
        model.rolling_sigma_mean = (1.0 - SIGMA_EWMA_ALPHA) * model.rolling_sigma_mean
            + SIGMA_EWMA_ALPHA * empirical_sigma;

        ratio
    } else {
        1.0
    };

    let base_lr = model.config.learning_rate;

    // Train location steps only -- no scale trees needed
    let mut mu_accum = model.location_base;
    for s in 0..model.location_steps.len() {
        let (g_mu, h_mu) = location_gradient(mu_accum, target);
        // Welford update for ensemble gradient stats
        update_ensemble_grad_stats(model, g_mu);
        let train_count = model.config.variant.train_count(h_mu, &mut model.rng_state);
        let loc_pred = model.location_steps[s].train_and_predict(features, g_mu, h_mu, train_count);
        mu_accum += (base_lr * sigma_ratio) * loc_pred;
    }
}

fn train_tree_chain(model: &mut DistributionalSGBT, target: f64, features: &[f64]) {
    let mut mu = model.location_base;
    let mut log_sigma = model.scale_base;

    // Update rolling honest_sigma mean (tree contribution variance EWMA)
    let honest_sigma = compute_honest_sigma(model, features);
    const HONEST_SIGMA_ALPHA: f64 = 0.001;
    model.rolling_honest_sigma_mean = (1.0 - HONEST_SIGMA_ALPHA) * model.rolling_honest_sigma_mean
        + HONEST_SIGMA_ALPHA * honest_sigma;

    // Compute σ-ratio for uncertainty-modulated learning rate (PD controller).
    let sigma_ratio = if model.uncertainty_modulated_lr {
        let current_sigma = crate::math::exp(log_sigma).max(1e-8);

        // Compute sigma velocity (derivative of sigma over time)
        let d_sigma = current_sigma - model.prev_sigma;
        model.prev_sigma = current_sigma;

        // EWMA-smooth the velocity (same alpha as empirical sigma for synchronization)
        let alpha = model.empirical_sigma_alpha;
        model.sigma_velocity = (1.0 - alpha) * model.sigma_velocity + alpha * d_sigma;

        // Adaptive derivative gain: self-calibrating, no config needed
        let k_d = if model.rolling_sigma_mean > 1e-12 {
            crate::math::abs(model.sigma_velocity) / model.rolling_sigma_mean
        } else {
            0.0
        };

        // PD ratio: proportional + derivative
        let pd_sigma = current_sigma + k_d * model.sigma_velocity;
        let ratio = (pd_sigma / model.rolling_sigma_mean).clamp(0.1, 10.0);

        const SIGMA_EWMA_ALPHA: f64 = 0.001;
        model.rolling_sigma_mean =
            (1.0 - SIGMA_EWMA_ALPHA) * model.rolling_sigma_mean + SIGMA_EWMA_ALPHA * current_sigma;

        ratio
    } else {
        1.0
    };

    let base_lr = model.config.learning_rate;

    // Sequential boosting: both ensembles target their respective residuals
    for s in 0..model.location_steps.len() {
        let sigma = crate::math::exp(log_sigma).max(1e-8);
        let z = (target - mu) / sigma;

        // Location gradients
        let (g_mu, h_mu) = location_gradient(mu, target);
        // Welford update for ensemble gradient stats
        update_ensemble_grad_stats(model, g_mu);

        // Scale gradients (NLL w.r.t. log_sigma)
        let g_sigma = 1.0 - z * z;
        let h_sigma = (2.0 * z * z).clamp(0.01, 100.0);

        let train_count = model.config.variant.train_count(h_mu, &mut model.rng_state);

        // Train location step -- σ-modulated LR when enabled
        let loc_pred = model.location_steps[s].train_and_predict(features, g_mu, h_mu, train_count);
        mu += (base_lr * sigma_ratio) * loc_pred;

        // Train scale step -- ALWAYS at unmodulated base rate.
        let scale_pred =
            model.scale_steps[s].train_and_predict(features, g_sigma, h_sigma, train_count);
        log_sigma += base_lr * scale_pred;
    }

    // Also update empirical sigma tracker for diagnostics
    let err = target - mu;
    let alpha = model.empirical_sigma_alpha;
    model.ewma_sq_err = (1.0 - alpha) * model.ewma_sq_err + alpha * err * err;
}

fn location_gradient(mu: f64, target: f64) -> (f64, f64) {
    let err = target - mu;
    let g = -err;
    let h = 1.0;
    (g, h)
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

fn update_ensemble_grad_stats(model: &mut DistributionalSGBT, g: f64) {
    let _n = model.ensemble_grad_count;
    let mean_old = model.ensemble_grad_mean;
    model.ensemble_grad_count += 1;
    let delta = g - mean_old;
    model.ensemble_grad_mean = mean_old + delta / model.ensemble_grad_count as f64;
    model.ensemble_grad_m2 += delta * (g - model.ensemble_grad_mean);
}
