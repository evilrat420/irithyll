//! Gate computation functions for streaming attention variants.
//!
//! Each attention mode uses a different gating mechanism to control
//! how much of the previous state to retain and how much new information
//! to write. This module provides the gate computations as pure functions.
//!
//! # Gate Types
//!
//! - **Fixed decay** (RetNet): constant `gamma` per timestep
//! - **Sigmoid gate** (GLA, GatedDeltaNet, mLSTM): `sigma(w^T x)`
//! - **Exponential gate** (RWKV): `exp(-(initial_decay + softplus(w^T x)))`
//! - **LSTM gates** (mLSTM): separate forget and input sigmoid gates

use alloc::vec::Vec;

use crate::math;

/// Xorshift64 PRNG for deterministic weight initialization.
///
/// Same algorithm as [`crate::ssm::projection::Xorshift64`], duplicated here
/// to keep the attention module self-contained within `irithyll-core`.
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    /// Generate the next random u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Generate a uniform random f64 in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a standard normal random value via Box-Muller transform.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let u1 = loop {
            let u = self.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.next_f64();
        math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * core::f64::consts::PI * u2)
    }
}

/// Dot product of two slices.
///
/// Delegates to [`crate::simd::simd_dot`] for AVX2 acceleration
/// when available.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal lengths");
    crate::simd::simd_dot(a, b)
}

/// Row-major matrix-vector multiply: `out = W * x`.
///
/// `w` is `rows x cols` row-major, `x` is `cols`-vector, `out` is `rows`-vector.
///
/// Delegates to [`crate::simd::simd_mat_vec`] for AVX2 acceleration
/// when available.
#[inline]
pub fn mat_vec(w: &[f64], x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    debug_assert_eq!(w.len(), rows * cols, "w must be rows*cols");
    debug_assert_eq!(x.len(), cols, "x must have cols elements");
    debug_assert_eq!(out.len(), rows, "out must have rows elements");
    crate::simd::simd_mat_vec(w, x, rows, cols, out);
}

/// Initialize a weight vector with small random normal values (scale 0.01).
pub fn init_weights(rng: &mut Xorshift64, len: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(len);
    for _ in 0..len {
        w.push(rng.next_normal() * 0.01);
    }
    w
}

/// Fixed decay gate (RetNet).
///
/// Returns the constant `gamma` regardless of input. This is the simplest
/// gating: the state decays by a fixed factor each timestep.
///
/// # Arguments
///
/// * `gamma` -- decay factor in (0, 1)
#[inline]
pub fn fixed_decay(gamma: f64) -> f64 {
    gamma
}

/// Data-dependent sigmoid gate (GLA, GatedDeltaNet, mLSTM).
///
/// Computes `sigmoid(w_gate^T * x)`, producing a gate value in (0, 1).
///
/// # Arguments
///
/// * `w_gate` -- gate weight vector (length must match `x`)
/// * `x` -- input vector
#[inline]
pub fn sigmoid_gate(w_gate: &[f64], x: &[f64]) -> f64 {
    math::sigmoid(dot(w_gate, x))
}

/// Exponential gate (RWKV).
///
/// Computes `exp(-(initial_decay + softplus(w_decay^T * x)))`, producing
/// a decay factor in (0, 1) that is input-dependent.
///
/// # Arguments
///
/// * `w_decay` -- decay weight vector (length must match `x`)
/// * `x` -- input vector
/// * `initial_decay` -- base decay rate (positive)
#[inline]
pub fn exponential_gate(w_decay: &[f64], x: &[f64], initial_decay: f64) -> f64 {
    let raw = initial_decay + math::softplus(dot(w_decay, x));
    math::exp(-raw)
}

/// LSTM-style forget and input gates (mLSTM).
///
/// Computes:
/// - forget gate: `sigmoid(w_f^T * x)`
/// - input gate: `sigmoid(w_i^T * x)`
///
/// # Arguments
///
/// * `w_f` -- forget gate weight vector
/// * `w_i` -- input gate weight vector
/// * `x` -- input vector
///
/// # Returns
///
/// `(forget_gate, input_gate)` both in (0, 1).
#[inline]
pub fn lstm_gates(w_f: &[f64], w_i: &[f64], x: &[f64]) -> (f64, f64) {
    (math::sigmoid(dot(w_f, x)), math::sigmoid(dot(w_i, x)))
}

/// Per-dimension vector decay (RWKV-7).
///
/// Computes bounded decay per dimension:
/// `w[i] = exp(-exp(-0.5) * sigmoid(dot_i))` where `dot_i = w_decay_row_i . x`.
///
/// Each element of the output is in (exp(-exp(-0.5)), 1) ≈ (0.545, 1.0),
/// ensuring stable but non-trivial decay per dimension.
///
/// # Arguments
///
/// * `w_decay` -- decay weight matrix, `d_key x d_model` row-major
/// * `x` -- input vector (length `d_model`)
/// * `d_key` -- number of output dimensions
///
/// # Returns
///
/// Vector of length `d_key` with decay factors in (0.545, 1.0).
pub fn vector_decay(w_decay: &[f64], x: &[f64], d_key: usize) -> Vec<f64> {
    let d_model = x.len();
    debug_assert_eq!(
        w_decay.len(),
        d_key * d_model,
        "w_decay must be d_key * d_model"
    );
    let scale = math::exp(-0.5); // ~0.6065
    let mut w = Vec::with_capacity(d_key);
    for i in 0..d_key {
        let row = &w_decay[i * d_model..(i + 1) * d_model];
        let raw = dot(row, x);
        w.push(math::exp(-scale * math::sigmoid(raw)));
    }
    w
}

/// Per-dimension sigmoid gate (RWKV-7 ICLR).
///
/// Computes `sigmoid(w_gate_row_i . x)` for each dimension, producing
/// a vector of gate values in (0, 1).
///
/// # Arguments
///
/// * `w_gate` -- gate weight matrix, `d_key x d_model` row-major
/// * `x` -- input vector (length `d_model`)
/// * `d_key` -- number of output dimensions
///
/// # Returns
///
/// Vector of length `d_key` with gate values in (0, 1).
pub fn vector_sigmoid_gate(w_gate: &[f64], x: &[f64], d_key: usize) -> Vec<f64> {
    let d_model = x.len();
    debug_assert_eq!(
        w_gate.len(),
        d_key * d_model,
        "w_gate must be d_key * d_model"
    );
    let mut g = Vec::with_capacity(d_key);
    for i in 0..d_key {
        let row = &w_gate[i * d_model..(i + 1) * d_model];
        g.push(math::sigmoid(dot(row, x)));
    }
    g
}

/// Extended sigmoid for DeltaProduct beta (range [0, 2]).
///
/// Computes `2 * sigmoid(w . x)`, mapping to [0, 2] for Householder
/// reflections. At beta=2, the transformation is a full reflection.
///
/// # Arguments
///
/// * `w` -- weight vector (length must match `x`)
/// * `x` -- input vector
#[inline]
pub fn extended_sigmoid_gate(w: &[f64], x: &[f64]) -> f64 {
    2.0 * math::sigmoid(dot(w, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn fixed_decay_returns_gamma() {
        assert!(
            (fixed_decay(0.9) - 0.9).abs() < 1e-12,
            "fixed_decay(0.9) should return 0.9"
        );
        assert!(
            (fixed_decay(0.0) - 0.0).abs() < 1e-12,
            "fixed_decay(0.0) should return 0.0"
        );
    }

    #[test]
    fn sigmoid_gate_at_zero_bias() {
        // When w and x produce dot=0, sigmoid should return 0.5
        let w = vec![0.0; 4];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let g = sigmoid_gate(&w, &x);
        assert!(
            (g - 0.5).abs() < 1e-12,
            "sigmoid(0) should be 0.5, got {}",
            g
        );
    }

    #[test]
    fn sigmoid_gate_large_positive() {
        let w = vec![10.0; 4];
        let x = vec![1.0; 4];
        let g = sigmoid_gate(&w, &x);
        assert!(
            g > 0.99,
            "sigmoid of large positive should be > 0.99, got {}",
            g
        );
    }

    #[test]
    fn sigmoid_gate_large_negative() {
        let w = vec![-10.0; 4];
        let x = vec![1.0; 4];
        let g = sigmoid_gate(&w, &x);
        assert!(
            g < 0.01,
            "sigmoid of large negative should be < 0.01, got {}",
            g
        );
    }

    #[test]
    fn exponential_gate_in_unit_interval() {
        let w = vec![0.1, -0.1, 0.05, 0.0];
        let x = vec![1.0, 2.0, -1.0, 0.5];
        let g = exponential_gate(&w, &x, 0.5);
        assert!(
            g > 0.0 && g < 1.0,
            "exponential gate should be in (0, 1), got {}",
            g
        );
    }

    #[test]
    fn exponential_gate_large_decay_small_output() {
        // Large initial_decay -> very small gate value (aggressive decay)
        let w = vec![0.0; 4];
        let x = vec![0.0; 4];
        let g = exponential_gate(&w, &x, 10.0);
        // exp(-(10 + softplus(0))) = exp(-(10 + ln(2))) ~ exp(-10.69)
        assert!(
            g < 0.001,
            "large decay should produce very small gate, got {}",
            g
        );
    }

    #[test]
    fn lstm_gates_at_zero() {
        let w_f = vec![0.0; 4];
        let w_i = vec![0.0; 4];
        let x = vec![1.0; 4];
        let (f, i) = lstm_gates(&w_f, &w_i, &x);
        assert!(
            (f - 0.5).abs() < 1e-12,
            "forget gate at zero should be 0.5, got {}",
            f
        );
        assert!(
            (i - 0.5).abs() < 1e-12,
            "input gate at zero should be 0.5, got {}",
            i
        );
    }

    #[test]
    fn lstm_gates_independent() {
        // Forget gate large positive, input gate large negative
        let w_f = vec![10.0; 2];
        let w_i = vec![-10.0; 2];
        let x = vec![1.0; 2];
        let (f, i) = lstm_gates(&w_f, &w_i, &x);
        assert!(f > 0.99, "forget gate should be near 1, got {}", f);
        assert!(i < 0.01, "input gate should be near 0, got {}", i);
    }

    #[test]
    fn xorshift_deterministic_same_seed() {
        let mut rng1 = Xorshift64(42);
        let mut rng2 = Xorshift64(42);
        for _ in 0..50 {
            assert_eq!(
                rng1.next_u64(),
                rng2.next_u64(),
                "same seed must produce same sequence"
            );
        }
    }

    #[test]
    fn init_weights_correct_length_and_small() {
        let mut rng = Xorshift64(123);
        let w = init_weights(&mut rng, 100);
        assert_eq!(w.len(), 100, "should produce 100 weights");
        let max_abs = w.iter().fold(0.0f64, |m, &x| {
            let a = if x < 0.0 { -x } else { x };
            if a > m {
                a
            } else {
                m
            }
        });
        assert!(
            max_abs < 0.5,
            "weights with scale 0.01 should be small, max_abs={}",
            max_abs
        );
    }

    #[test]
    fn vector_decay_bounded() {
        let w = vec![0.1, -0.2, 0.3, -0.1, 0.05, 0.15, -0.05, 0.2];
        let x = vec![1.0, 2.0, -1.0, 0.5];
        let decay = vector_decay(&w, &x, 2);
        assert_eq!(decay.len(), 2, "should produce 2 decay values");
        for (i, &d) in decay.iter().enumerate() {
            assert!(
                d > 0.54 && d < 1.0,
                "decay[{}] should be in (0.54, 1.0), got {}",
                i,
                d
            );
        }
    }

    #[test]
    fn vector_sigmoid_gate_bounded() {
        let w = vec![10.0, 0.0, -10.0, 0.0]; // 2 dims, d_model=2
        let x = vec![1.0, 0.0];
        let g = vector_sigmoid_gate(&w, &x, 2);
        assert!(g[0] > 0.99, "large positive should give ~1, got {}", g[0]);
        assert!(g[1] < 0.01, "large negative should give ~0, got {}", g[1]);
    }

    #[test]
    fn extended_sigmoid_gate_range() {
        let w_pos = vec![100.0];
        let w_neg = vec![-100.0];
        let x = vec![1.0];
        let high = extended_sigmoid_gate(&w_pos, &x);
        let low = extended_sigmoid_gate(&w_neg, &x);
        assert!(high > 1.99, "large positive should give ~2.0, got {}", high);
        assert!(low < 0.01, "large negative should give ~0.0, got {}", low);
        let mid = extended_sigmoid_gate(&[0.0], &[0.0]);
        assert!(
            (mid - 1.0).abs() < 1e-6,
            "zero input should give 1.0, got {}",
            mid
        );
    }

    #[test]
    fn mat_vec_basic() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        mat_vec(&w, &x, 2, 2, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-12, "row 0: 1+2=3, got {}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-12, "row 1: 3+4=7, got {}", out[1]);
    }
}
