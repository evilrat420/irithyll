//! StreamingLearner trait implementation for Mamba.
//!
//! Contains the core train_one, predict, and reset logic.

use super::{SSMVariant, StreamingMamba};
use irithyll_core::continual::ContinualStrategy;
use irithyll_core::learner::StreamingLearner;
use irithyll_core::math::silu;

/// Train on one sample with weighted update.
///
/// Follows the Option D prequential protocol: readout features are computed from
/// the pre-update SSM state (via the cached `last_ssm_output` from the prior step),
/// RLS is trained on those pre-update features, then the SSM state is advanced.
/// This ensures train and predict both query the same feature distribution
/// (pre-update SSM output × current gate), eliminating the one-step
/// train/predict feature-distribution mismatch.
pub(super) fn train_one(model: &mut StreamingMamba, features: &[f64], target: f64, weight: f64) {
    let d_in = model.config.d_in;

    // Guard: skip non-finite inputs to prevent NaN from corrupting SSM state.
    if !features.iter().all(|f| f.is_finite()) {
        return;
    }

    // Option D step 1: compute readout features from PRE-update SSM output
    // (last_ssm_output holds the output from the previous train_one call).
    // This mirrors exactly what predict() does, making train/predict features identical.
    //
    // Gate computation: gate[i] = SiLU(W[i,:] · x + b[i])
    let pre_gated_output: Vec<f64> = (0..d_in)
        .map(|i| {
            let mut sum = model.gate_bias[i];
            let row = &model.gate_weights[i * d_in..(i + 1) * d_in];
            for (w, &x) in row.iter().zip(features.iter()) {
                sum += w * x;
            }
            let gate_val = silu(sum);
            // last_ssm_output = output from previous step = pre-update SSM output
            model.last_ssm_output[i] * gate_val + features[i]
        })
        .collect();

    // Build readout features using pre-update state.
    let pre_state = model.ssm.state().to_vec();
    let pre_readout_features =
        model.build_readout_features(&pre_gated_output, &pre_state, features);

    // Track state Frobenius squared norm for utilization ratio (pre-update state).
    let frob_sq: f64 = pre_state.iter().map(|s| s * s).sum();
    const FROB_ALPHA: f64 = 0.001;
    model.max_frob_sq_ewma = if frob_sq > model.max_frob_sq_ewma {
        frob_sq
    } else {
        (1.0 - FROB_ALPHA) * model.max_frob_sq_ewma + FROB_ALPHA * frob_sq
    };

    // Update residual alignment tracking (acceleration-based, on pre-update features).
    let current_pred = model.readout.predict(&pre_readout_features);
    let current_change = current_pred - model.prev_prediction;
    if model.n_samples > 0 {
        let acceleration = current_change - model.prev_change;
        let prev_acceleration = model.prev_change - model.prev_prev_change;
        let agreement = if acceleration.abs() > 1e-15 && prev_acceleration.abs() > 1e-15 {
            if (acceleration > 0.0) == (prev_acceleration > 0.0) {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
        const ALIGN_ALPHA: f64 = 0.05;
        if model.n_samples == 1 {
            model.alignment_ewma = agreement;
        } else {
            model.alignment_ewma =
                (1.0 - ALIGN_ALPHA) * model.alignment_ewma + ALIGN_ALPHA * agreement;
        }
    }
    model.prev_prev_change = model.prev_change;
    model.prev_change = current_change;
    model.prev_prediction = current_pred;

    // Option D step 2: train RLS on pre-update features before advancing state.
    if !pre_readout_features.iter().all(|f| f.is_finite()) {
        model.last_features = pre_readout_features;
        model.n_samples += 1;
        return;
    }
    model
        .readout
        .train_one(&pre_readout_features, target, weight);

    // Option D step 3: advance SSM state by processing x_t.
    let ssm_output = model.ssm.forward(features);

    // Guard: if the SSM produced non-finite output, reset state and skip caching.
    if !ssm_output.iter().all(|f| f.is_finite()) {
        model.ssm.reset();
        model.last_features = pre_readout_features;
        model.n_samples += 1;
        return;
    }

    // Cache the new SSM output — this becomes the pre-update output for the next step.
    model.last_ssm_output.copy_from_slice(&ssm_output);

    // Plasticity maintenance: track per-unit SSM state energy (post-update state)
    // and trigger surgical reinit when dead units are detected.
    if let Some(ref mut guard) = model.plasticity_guard {
        let state = model.ssm.state();
        let n_state = model.config.n_state;
        let n_units = guard.n_groups();
        let mut unit_energy: Vec<f64> = match &model.ssm {
            SSMVariant::V1(_) => (0..n_units)
                .map(|d| {
                    let mut e = 0.0;
                    for n in 0..n_state {
                        let idx = n * model.config.d_in + d;
                        if idx < state.len() {
                            e += state[idx].abs();
                        }
                    }
                    e / n_state.max(1) as f64
                })
                .collect(),
            // V3 (Tustin), V3Exp, V3Mimo: per-group energy over all state elements
            // For V3Mimo, state layout is 2*n_groups*n_state*cpg; dividing into
            // n_groups slices gives per-group Frobenius norm.
            SSMVariant::V3(_) | SSMVariant::V3Exp(_) | SSMVariant::V3Mimo(_) => {
                let per_group = state.len().checked_div(n_units).unwrap_or(0);
                (0..n_units)
                    .map(|g| {
                        let start = g * per_group;
                        let end = (start + per_group).min(state.len());
                        let e: f64 = state[start..end].iter().map(|s| s.abs()).sum();
                        e / per_group.max(1) as f64
                    })
                    .collect()
            }
            SSMVariant::BD(ssm) => {
                let bs = ssm.block_size();
                (0..n_units)
                    .map(|b| {
                        let start = b * n_state * bs;
                        let end = (start + n_state * bs).min(state.len());
                        let e: f64 = state[start..end].iter().map(|s| s.abs()).sum();
                        e / (n_state * bs).max(1) as f64
                    })
                    .collect()
            }
        };
        guard.pre_update(&model.prev_state_energy, &mut unit_energy);
        guard.post_update(&model.prev_state_energy);

        // Surgical per-unit reinit based on SSM variant
        let mut reinit_rng = model
            .config
            .seed
            .wrapping_add(0xCAFE_BABE_u64.wrapping_mul(model.n_samples));
        for j in 0..guard.n_groups() {
            if guard.was_regenerated(j) {
                match &mut model.ssm {
                    SSMVariant::V1(ssm) => ssm.reinitialize_channel(j, &mut reinit_rng),
                    SSMVariant::V3(ssm) => ssm.reinitialize_group(j, &mut reinit_rng),
                    SSMVariant::V3Exp(ssm) => ssm.reinitialize_group(j, &mut reinit_rng),
                    SSMVariant::V3Mimo(ssm) => ssm.reinitialize_group(j, &mut reinit_rng),
                    SSMVariant::BD(ssm) => ssm.reinitialize_block(j, &mut reinit_rng),
                }
            }
        }

        model.prev_state_energy = unit_energy;
    }

    // Cache readout features for diagnostics (post-update gated features for state_frob_ratio).
    let post_gated_output: Vec<f64> = (0..d_in)
        .map(|i| {
            let mut sum = model.gate_bias[i];
            let row = &model.gate_weights[i * d_in..(i + 1) * d_in];
            for (w, &x) in row.iter().zip(features.iter()) {
                sum += w * x;
            }
            let gate_val = silu(sum);
            ssm_output[i] * gate_val + features[i]
        })
        .collect();
    let post_state = model.ssm.state();
    model.last_features = model.build_readout_features(&post_gated_output, post_state, features);

    model.n_samples += 1;
}

/// Predict using the current input and cached SSM state.
pub(super) fn predict(model: &StreamingMamba, features: &[f64]) -> f64 {
    // Design: predict() must use the current input for gating/residual
    // combined with SSM state from the previous train_one() call.
    if model.n_samples == 0 || features.len() != model.config.d_in {
        return 0.0;
    }

    let d_in = model.config.d_in;

    // Recompute gate + residual using the current input x_t and the cached
    // SSM output (temporal signal from x_{t-1} forward pass).
    let gated_output: Vec<f64> = (0..d_in)
        .map(|i| {
            let mut sum = model.gate_bias[i];
            let row = &model.gate_weights[i * d_in..(i + 1) * d_in];
            for (w, &x) in row.iter().zip(features.iter()) {
                sum += w * x;
            }
            let gate_val = silu(sum);
            model.last_ssm_output[i] * gate_val + features[i]
        })
        .collect();

    // Rebuild readout features with recomputed gated output + cached state energy
    let state = model.ssm.state();
    let readout_features = model.build_readout_features(&gated_output, state, features);

    model.readout.predict(&readout_features)
}

/// Reset model to initial state.
pub(super) fn reset(model: &mut StreamingMamba) {
    model.ssm.reset();
    model.readout.reset();

    // Re-initialize gate weights from scratch (deterministic from seed).
    let (gw, gb) = StreamingMamba::init_gate_weights(model.config.d_in, model.config.seed);
    model.gate_weights = gw;
    model.gate_bias = gb;

    for f in model.last_features.iter_mut() {
        *f = 0.0;
    }
    model.n_samples = 0;
    model.prev_prediction = 0.0;
    model.prev_change = 0.0;
    model.prev_prev_change = 0.0;
    model.alignment_ewma = 0.0;
    model.max_frob_sq_ewma = 0.0;

    if let Some(ref mut guard) = model.plasticity_guard {
        guard.reset();
    }

    model.prev_state_energy.fill(0.0);
    model.last_ssm_output.fill(0.0);
}
