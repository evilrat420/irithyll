//! State update functions for each attention architecture variant.
//!
//! Each function takes a mutable reference to the state plus the current
//! timestep's key, value, and gating parameters, and applies the
//! architecture-specific recurrence in-place.
//!
//! # Update Rules
//!
//! | Architecture     | Rule                                          |
//! |-----------------|-----------------------------------------------|
//! | RetNet / GLA    | `S = decay * S + k * v^T`                     |
//! | DeltaNet        | `S = S + (v - S^T k) * k^T`                  |
//! | GatedDeltaNet   | `S = decay * S + beta * (v - S^T k_norm) * k_norm^T` |
//! | RWKV            | `S = exp(-w) * S + exp(k) * v^T`              |
//! | Hawk            | `h = alpha * h + beta * x` (element-wise)     |
//! | mLSTM           | `S = f * S + i * v * k^T`                     |

use alloc::vec;

use super::state::AttentionState;
use crate::math;

/// Additive update (RetNet, basic GLA).
///
/// `S = decay * S + k * v^T`
///
/// The state decays by a fixed or data-dependent factor, then accumulates
/// the outer product of the current key and value.
///
/// # Arguments
///
/// * `state` -- matrix state of shape `d_k x d_v`
/// * `k` -- key vector (length `d_k`)
/// * `v` -- value vector (length `d_v`)
/// * `decay` -- scalar decay factor
pub fn additive_update(state: &mut AttentionState, k: &[f64], v: &[f64], decay: f64) {
    state.scale(decay);
    state.add_outer_product(k, v);
}

/// Delta rule update (DeltaNet).
///
/// `S = S + (v - S^T k) * k^T`
///
/// Error-corrective: the update writes the "correct" value `v` for key `k`
/// by computing the prediction error `e = v - S^T k` and adjusting the state
/// by `e * k^T`. This is a Hebbian-like associative memory update.
///
/// # Arguments
///
/// * `state` -- matrix state of shape `d_k x d_v`
/// * `k` -- key vector (length `d_k`)
/// * `v` -- value vector (length `d_v`)
pub fn delta_update(state: &mut AttentionState, k: &[f64], v: &[f64]) {
    // Compute prediction: pred = S^T * k (length d_v)
    let pred = state.query(k);

    // Compute error: e = v - pred
    let d_v = v.len();
    let mut error = vec![0.0; d_v];
    for j in 0..d_v {
        error[j] = v[j] - pred[j];
    }

    // S += error * k^T (i.e., k * error^T in row-major terms)
    state.add_outer_product(k, &error);
}

/// Gated delta update (GatedDeltaNet, Yang et al. ICLR 2025).
///
/// `S = decay * S + beta_scale * (v - S^T k_norm) * k_norm^T`
///
/// Combines GLA's data-dependent gating with DeltaNet's error-corrective
/// delta rule, using L2-normalized keys and learnable beta scaling.
/// The state first decays, then the beta-scaled delta correction is applied
/// with normalized keys for bounded state growth.
///
/// # Arguments
///
/// * `state` -- matrix state of shape `d_k x d_v`
/// * `k` -- key vector (length `d_k`), will be L2-normalized internally
/// * `v` -- value vector (length `d_v`)
/// * `decay` -- scalar decay factor from sigmoid gate
/// * `beta_scale` -- learnable scaling for the delta correction term
pub fn gated_delta_update(
    state: &mut AttentionState,
    k: &[f64],
    v: &[f64],
    decay: f64,
    beta_scale: f64,
) {
    // First decay the state
    state.scale(decay);

    // L2-normalize keys for bounded state growth
    let d_k = k.len();
    let norm_sq: f64 = k.iter().map(|&x| x * x).sum();
    let norm = math::sqrt(norm_sq);
    let k_norm: alloc::vec::Vec<f64> = if norm < 1e-12 {
        vec![0.0; d_k]
    } else {
        let inv = 1.0 / norm;
        k.iter().map(|&x| x * inv).collect()
    };

    // Compute prediction with normalized key: pred = S^T * k_norm
    let pred = state.query(&k_norm);

    // Compute beta-scaled error: e = beta_scale * (v - pred)
    let d_v = v.len();
    let mut error = vec![0.0; d_v];
    for j in 0..d_v {
        error[j] = beta_scale * (v[j] - pred[j]);
    }

    // S += e * k_norm^T
    state.add_outer_product(&k_norm, &error);
}

/// Exponential update (RWKV).
///
/// `S = exp(-w) * S + exp(k_i) * v^T`
///
/// RWKV uses exponential weighting: the state decays by `exp(-w)` and the
/// key is exponentiated before forming the outer product. This creates a
/// "receptance-weighted" mechanism where keys compete exponentially.
///
/// # Arguments
///
/// * `state` -- matrix state of shape `d_k x d_v`
/// * `k` -- key vector (length `d_k`), exponentiated element-wise
/// * `v` -- value vector (length `d_v`)
/// * `w` -- scalar decay parameter (pre-computed from gate)
pub fn exponential_update(state: &mut AttentionState, k: &[f64], v: &[f64], w: f64) {
    let decay = math::exp(-w);
    state.scale(decay);

    // exp(k) * v^T
    let d_k = k.len();
    let mut exp_k = vec![0.0; d_k];
    for i in 0..d_k {
        exp_k[i] = math::exp(k[i]);
    }
    state.add_outer_product(&exp_k, v);
}

/// Hawk update (vector state).
///
/// `h = alpha * h + beta * x` (element-wise)
///
/// Hawk (from the Griffin architecture) uses a simple gated recurrence on a
/// vector state. Each dimension has its own learned `alpha` (decay) and
/// `beta` (input scaling) parameters.
///
/// # Arguments
///
/// * `state` -- vector state of dimension `d`
/// * `x` -- input vector (length `d`)
/// * `alpha` -- per-dimension decay factors (length `d`)
/// * `beta` -- per-dimension input scaling (length `d`)
///
/// # Panics
///
/// Panics if the state is not a Vector, or if lengths don't match.
pub fn hawk_update(state: &mut AttentionState, x: &[f64], alpha: &[f64], beta: &[f64]) {
    match state {
        AttentionState::Vector(h) => {
            debug_assert_eq!(h.len(), x.len(), "state and input must have same length");
            debug_assert_eq!(
                h.len(),
                alpha.len(),
                "state and alpha must have same length"
            );
            debug_assert_eq!(h.len(), beta.len(), "state and beta must have same length");
            for i in 0..h.len() {
                h[i] = alpha[i] * h[i] + beta[i] * x[i];
            }
        }
        AttentionState::Matrix { .. } => panic!("hawk_update requires Vector state"),
    }
}

/// mLSTM update.
///
/// `S = f * S + i * v * k^T`
///
/// The xLSTM matrix memory variant uses separate forget (`f`) and input (`i`)
/// gates. The forget gate controls state retention and the input gate scales
/// the new association strength.
///
/// # Arguments
///
/// * `state` -- matrix state of shape `d_k x d_v`
/// * `k` -- key vector (length `d_k`)
/// * `v` -- value vector (length `d_v`)
/// * `forget` -- forget gate value in (0, 1)
/// * `input` -- input gate value in (0, 1)
pub fn mlstm_update(state: &mut AttentionState, k: &[f64], v: &[f64], forget: f64, input: f64) {
    state.scale(forget);

    // i * v * k^T: scale the outer product by input gate
    let _d_k = k.len();
    let d_v = v.len();
    let mut scaled_v = vec![0.0; d_v];
    for (j, sv) in scaled_v.iter_mut().enumerate() {
        *sv = input * v[j];
    }
    state.add_outer_product(k, &scaled_v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_update_from_zero_state() {
        let mut state = AttentionState::new_matrix(2, 3);
        let k = [1.0, 2.0];
        let v = [3.0, 4.0, 5.0];
        additive_update(&mut state, &k, &v, 0.9);
        // From zero: decay does nothing, so S = k * v^T
        assert!(
            (state.get_matrix(0, 0) - 3.0).abs() < 1e-12,
            "S[0][0] should be 1*3=3, got {}",
            state.get_matrix(0, 0)
        );
        assert!(
            (state.get_matrix(1, 2) - 10.0).abs() < 1e-12,
            "S[1][2] should be 2*5=10, got {}",
            state.get_matrix(1, 2)
        );
    }

    #[test]
    fn additive_update_decay_applied() {
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 10.0);
        state.set_matrix(1, 1, 20.0);
        let k = [0.0, 0.0];
        let v = [0.0, 0.0];
        additive_update(&mut state, &k, &v, 0.5);
        assert!(
            (state.get_matrix(0, 0) - 5.0).abs() < 1e-12,
            "decayed S[0][0] should be 10*0.5=5, got {}",
            state.get_matrix(0, 0)
        );
        assert!(
            (state.get_matrix(1, 1) - 10.0).abs() < 1e-12,
            "decayed S[1][1] should be 20*0.5=10, got {}",
            state.get_matrix(1, 1)
        );
    }

    #[test]
    fn delta_update_error_corrective() {
        let mut state = AttentionState::new_matrix(2, 2);
        // Write key [1, 0] -> value [5, 3]
        let k = [1.0, 0.0];
        let v = [5.0, 3.0];
        delta_update(&mut state, &k, &v);
        // Now query with k: S^T * k should approximate v
        let out = state.query(&k);
        assert!(
            (out[0] - 5.0).abs() < 1e-12,
            "after delta write, read-back should be ~5.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.0).abs() < 1e-12,
            "after delta write, read-back should be ~3.0, got {}",
            out[1]
        );
    }

    #[test]
    fn delta_update_corrects_existing() {
        let mut state = AttentionState::new_matrix(2, 2);
        // First write
        let k = [1.0, 0.0];
        let v1 = [5.0, 3.0];
        delta_update(&mut state, &k, &v1);
        // Overwrite same key with new value
        let v2 = [10.0, 7.0];
        delta_update(&mut state, &k, &v2);
        let out = state.query(&k);
        assert!(
            (out[0] - 10.0).abs() < 1e-12,
            "after second delta write, should read 10.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 7.0).abs() < 1e-12,
            "after second delta write, should read 7.0, got {}",
            out[1]
        );
    }

    #[test]
    fn gated_delta_update_combines_decay_and_correction() {
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 100.0);
        let k = [1.0, 0.0];
        let v = [5.0, 3.0];
        gated_delta_update(&mut state, &k, &v, 0.0, 1.0);
        // With decay=0, previous state is wiped, then delta writes fresh
        // Key [1,0] is already unit norm so normalization is identity
        let out = state.query(&k);
        assert!(
            (out[0] - 5.0).abs() < 1e-12,
            "with decay=0, should read fresh value 5.0, got {}",
            out[0]
        );
    }

    #[test]
    fn exponential_update_changes_state() {
        let mut state = AttentionState::new_matrix(2, 3);
        let k = [0.1, -0.1];
        let v = [1.0, 2.0, 3.0];
        exponential_update(&mut state, &k, &v, 0.5);
        // State should be non-zero after update
        let s = state.as_slice();
        let sum: f64 = s.iter().map(|&x| if x < 0.0 { -x } else { x }).sum();
        assert!(
            sum > 0.0,
            "state should be non-zero after exponential update"
        );
    }

    #[test]
    fn exponential_update_exp_k_applied() {
        let mut state = AttentionState::new_matrix(1, 1);
        let k = [0.0]; // exp(0) = 1
        let v = [7.0];
        exponential_update(&mut state, &k, &v, 0.0);
        // exp(-0) * 0 + exp(0) * 7 = 1 * 7 = 7
        assert!(
            (state.get_matrix(0, 0) - 7.0).abs() < 1e-12,
            "with w=0 and k=0, state should be exp(0)*7=7, got {}",
            state.get_matrix(0, 0)
        );
    }

    #[test]
    fn hawk_update_vector_recurrence() {
        let mut state = AttentionState::new_vector(3);
        let x = [1.0, 2.0, 3.0];
        let alpha = [0.9, 0.8, 0.7];
        let beta = [0.1, 0.2, 0.3];
        hawk_update(&mut state, &x, &alpha, &beta);
        // From zero: h = alpha*0 + beta*x = beta*x
        let s = state.as_slice();
        assert!(
            (s[0] - 0.1).abs() < 1e-12,
            "h[0] should be 0.1*1=0.1, got {}",
            s[0]
        );
        assert!(
            (s[1] - 0.4).abs() < 1e-12,
            "h[1] should be 0.2*2=0.4, got {}",
            s[1]
        );
        assert!(
            (s[2] - 0.9).abs() < 1e-12,
            "h[2] should be 0.3*3=0.9, got {}",
            s[2]
        );
    }

    #[test]
    fn hawk_update_accumulates() {
        let mut state = AttentionState::new_vector(2);
        let alpha = [0.5, 0.5];
        let beta = [1.0, 1.0];
        hawk_update(&mut state, &[2.0, 4.0], &alpha, &beta);
        // h = [2, 4]
        hawk_update(&mut state, &[1.0, 1.0], &alpha, &beta);
        // h = [0.5*2+1*1, 0.5*4+1*1] = [2, 3]
        let s = state.as_slice();
        assert!(
            (s[0] - 2.0).abs() < 1e-12,
            "h[0] should be 2.0, got {}",
            s[0]
        );
        assert!(
            (s[1] - 3.0).abs() < 1e-12,
            "h[1] should be 3.0, got {}",
            s[1]
        );
    }

    #[test]
    fn mlstm_update_from_zero() {
        let mut state = AttentionState::new_matrix(2, 2);
        let k = [1.0, 0.0];
        let v = [5.0, 3.0];
        mlstm_update(&mut state, &k, &v, 0.9, 0.8);
        // From zero: f*0 + i*v*k^T = 0.8 * [5,3] * [1,0]^T
        assert!(
            (state.get_matrix(0, 0) - 4.0).abs() < 1e-12,
            "S[0][0] should be 0.8*5*1=4.0, got {}",
            state.get_matrix(0, 0)
        );
        assert!(
            (state.get_matrix(0, 1) - 2.4).abs() < 1e-12,
            "S[0][1] should be 0.8*3*1=2.4, got {}",
            state.get_matrix(0, 1)
        );
        assert!(
            state.get_matrix(1, 0).abs() < 1e-12,
            "S[1][0] should be 0.8*5*0=0, got {}",
            state.get_matrix(1, 0)
        );
    }

    #[test]
    fn mlstm_forget_gate_decays_state() {
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 10.0);
        state.set_matrix(1, 1, 20.0);
        let k = [0.0, 0.0];
        let v = [0.0, 0.0];
        mlstm_update(&mut state, &k, &v, 0.5, 1.0);
        assert!(
            (state.get_matrix(0, 0) - 5.0).abs() < 1e-12,
            "forget gate 0.5 should halve state, got {}",
            state.get_matrix(0, 0)
        );
    }

    #[test]
    fn all_updates_change_state_from_zero() {
        // Verify every update rule produces non-zero state from zero init
        // (with non-zero inputs)
        let k = [1.0, 0.5];
        let v = [2.0, 3.0];
        let x = [1.0, 2.0];
        let alpha = [0.9, 0.8];
        let beta = [0.1, 0.2];

        let mut s1 = AttentionState::new_matrix(2, 2);
        additive_update(&mut s1, &k, &v, 0.9);
        let sum1: f64 = s1.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum1 > 0.0, "additive_update should change state");

        let mut s2 = AttentionState::new_matrix(2, 2);
        delta_update(&mut s2, &k, &v);
        let sum2: f64 = s2.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum2 > 0.0, "delta_update should change state");

        let mut s3 = AttentionState::new_matrix(2, 2);
        gated_delta_update(&mut s3, &k, &v, 0.9, 1.0);
        let sum3: f64 = s3.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum3 > 0.0, "gated_delta_update should change state");

        let mut s4 = AttentionState::new_matrix(2, 2);
        exponential_update(&mut s4, &k, &v, 0.5);
        let sum4: f64 = s4.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum4 > 0.0, "exponential_update should change state");

        let mut s5 = AttentionState::new_vector(2);
        hawk_update(&mut s5, &x, &alpha, &beta);
        let sum5: f64 = s5.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum5 > 0.0, "hawk_update should change state");

        let mut s6 = AttentionState::new_matrix(2, 2);
        mlstm_update(&mut s6, &k, &v, 0.9, 0.8);
        let sum6: f64 = s6.as_slice().iter().map(|x| math::abs(*x)).sum();
        assert!(sum6 > 0.0, "mlstm_update should change state");
    }

    #[test]
    fn gated_delta_net_beta_scale_default_matches_original() {
        // With beta_scale=1.0 and a unit-norm key, the new gated_delta_update
        // should produce identical results to the original formulation.
        let mut state1 = AttentionState::new_matrix(2, 2);
        let mut state2 = AttentionState::new_matrix(2, 2);
        let k = [0.6, 0.8]; // already unit norm (0.36 + 0.64 = 1.0)
        let v = [5.0, 3.0];
        let decay = 0.9;

        // Manually replicate old behavior: decay, then delta rule
        state1.scale(decay);
        let pred = state1.query(&k);
        let mut error = vec![0.0; 2];
        for j in 0..2 {
            error[j] = v[j] - pred[j];
        }
        state1.add_outer_product(&k, &error);

        // Use new function with beta_scale=1.0
        gated_delta_update(&mut state2, &k, &v, decay, 1.0);

        let s1 = state1.as_slice();
        let s2 = state2.as_slice();
        for i in 0..s1.len() {
            assert!(
                (s1[i] - s2[i]).abs() < 1e-12,
                "beta_scale=1.0 should match original at index {}: {} vs {}",
                i,
                s1[i],
                s2[i]
            );
        }
    }

    #[test]
    fn gated_delta_net_key_normalization_bounded_state() {
        // Feed large-magnitude keys; state norm should stay bounded because
        // keys are L2-normalized internally.
        let mut state = AttentionState::new_matrix(2, 2);
        let v = [1.0, 1.0];
        let decay = 0.95;

        for i in 0..100 {
            // Keys with magnitude growing up to 1000
            let scale = (i + 1) as f64 * 10.0;
            let k = [scale, scale];
            gated_delta_update(&mut state, &k, &v, decay, 1.0);
        }

        let state_norm_sq: f64 = state.as_slice().iter().map(|&x| x * x).sum();
        let state_norm = math::sqrt(state_norm_sq);
        // With normalized keys and bounded values, state should stay bounded.
        // The maximum per-element magnitude with unit-norm keys and value magnitude
        // ~1 should stay well within a reasonable range.
        assert!(
            state_norm < 100.0,
            "state norm should be bounded with normalized keys, got {}",
            state_norm
        );
    }

    #[test]
    fn gated_delta_net_beta_scale_zero_freezes_state() {
        // With beta_scale=0, the delta correction is disabled entirely,
        // so state only decays (no new associations are written).
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 10.0);
        state.set_matrix(1, 1, 20.0);
        let k = [1.0, 0.0];
        let v = [999.0, 888.0]; // large values that would normally be written
        let decay = 0.5;

        gated_delta_update(&mut state, &k, &v, decay, 0.0);

        // State should just be decayed: 10*0.5=5, 20*0.5=10
        assert!(
            (state.get_matrix(0, 0) - 5.0).abs() < 1e-12,
            "with beta=0, S[0][0] should be 10*0.5=5.0, got {}",
            state.get_matrix(0, 0)
        );
        assert!(
            (state.get_matrix(1, 1) - 10.0).abs() < 1e-12,
            "with beta=0, S[1][1] should be 20*0.5=10.0, got {}",
            state.get_matrix(1, 1)
        );
        // Off-diagonal should remain zero (no delta correction applied)
        assert!(
            state.get_matrix(0, 1).abs() < 1e-12,
            "with beta=0, S[0][1] should remain 0, got {}",
            state.get_matrix(0, 1)
        );
        assert!(
            state.get_matrix(1, 0).abs() < 1e-12,
            "with beta=0, S[1][0] should remain 0, got {}",
            state.get_matrix(1, 0)
        );
    }
}
