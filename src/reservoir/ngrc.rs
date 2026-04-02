//! Next Generation Reservoir Computing (NG-RC) streaming learner.
//!
//! [`NextGenRC`] implements a reservoir-free temporal learning approach that
//! replaces the random reservoir with time-delay embeddings and polynomial
//! nonlinear features. This is trained online via Recursive Least Squares.
//!
//! # Algorithm
//!
//! Given an input time series with feature dimension `d`:
//!
//! 1. **Delay embedding**: Collect the last `k` observations at skip `s`:
//!    `O_lin = [x(t), x(t-s), x(t-2s), ..., x(t-(k-1)s)]` (length `k * d`)
//!
//! 2. **Polynomial expansion**: Generate all monomials of degree `p` from `O_lin`:
//!    `O_nonlin = poly(O_lin)` (e.g., all pairwise products for `p = 2`)
//!
//! 3. **Feature vector**: `features = [1.0 (bias), O_lin, O_nonlin]`
//!
//! 4. **Online readout**: Train an RLS model on these features to predict the target.
//!
//! # References
//!
//! Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
//! "Next generation reservoir computing." *Nature Communications*, 12, 5564.

use std::fmt;

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use irithyll_core::reservoir::{DelayBuffer, HighDegreePolynomial};

use super::ngrc_config::NGRCConfig;

/// Next Generation Reservoir Computing model.
///
/// Uses time-delay embeddings and polynomial features to create a nonlinear
/// feature space from a time series, trained online via RLS. No random weights
/// are involved — the model is fully deterministic.
///
/// The model requires `k * s` observations before producing meaningful
/// predictions (the "warmup" period while the delay buffer fills).
///
/// # Examples
///
/// ```
/// use irithyll::reservoir::{NextGenRC, NGRCConfig};
/// use irithyll::learner::StreamingLearner;
///
/// let config = NGRCConfig::builder()
///     .k(2)
///     .s(1)
///     .degree(2)
///     .build()
///     .unwrap();
///
/// let mut ngrc = NextGenRC::new(config);
///
/// // Feed a sine wave: train on x(t) to predict x(t+1).
/// for i in 0..200 {
///     let t = i as f64 * 0.1;
///     let x = t.sin();
///     let target = (t + 0.1).sin();
///     ngrc.train(&[x], target);
/// }
///
/// // Predict the next value.
/// let t_test: f64 = 200.0 * 0.1;
/// let pred = ngrc.predict(&[t_test.sin()]);
/// assert!(pred.is_finite());
/// ```
pub struct NextGenRC {
    /// Configuration.
    config: NGRCConfig,
    /// Circular buffer storing recent observations for delay embedding.
    /// Capacity = k * s (we need observations at delays 0, s, 2s, ..., (k-1)*s).
    buffer: DelayBuffer,
    /// Polynomial feature generator.
    poly: HighDegreePolynomial,
    /// Online readout model.
    rls: RecursiveLeastSquares,
    /// Total observations pushed into the buffer.
    total_pushed: u64,
    /// Number of observations needed before the model is warm.
    warmup_count: usize,
    /// Number of input features per observation (set on first call).
    n_inputs: Option<usize>,
    /// Samples trained on (post-warmup).
    samples_seen: u64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// Per-feature EWMA of |feature[i]| for feature utilization entropy.
    state_activity_ewma: Vec<f64>,
}

impl NextGenRC {
    /// Create a new NG-RC model from the given configuration.
    pub fn new(config: NGRCConfig) -> Self {
        // We need k delays at skip s. The buffer must hold enough observations
        // so that delay index (k-1)*s is available. That means we need
        // (k-1)*s + 1 entries in the buffer.
        let buf_capacity = (config.k - 1) * config.s + 1;
        let warmup_count = buf_capacity;

        let poly = HighDegreePolynomial::new(config.degree);
        let rls = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta);

        Self {
            buffer: DelayBuffer::new(buf_capacity),
            poly,
            rls,
            total_pushed: 0,
            warmup_count,
            n_inputs: None,
            samples_seen: 0,
            config,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            state_activity_ewma: Vec::new(),
        }
    }

    /// Whether the delay buffer is full and the model can produce predictions.
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.total_pushed >= self.warmup_count as u64
    }

    /// Build the NG-RC feature vector from the current buffer state.
    ///
    /// Returns `None` if the buffer is not yet warm.
    fn build_features(&self) -> Option<Vec<f64>> {
        if !self.is_warm() {
            return None;
        }

        let d = self.n_inputs?;
        let k = self.config.k;
        let s = self.config.s;

        // O_lin: concatenation of k delayed observations at skip s.
        let lin_dim = k * d;
        let mut o_lin = Vec::with_capacity(lin_dim);
        for delay_k in 0..k {
            let delay_idx = delay_k * s;
            if let Some(obs) = self.buffer.get(delay_idx) {
                o_lin.extend_from_slice(obs);
            } else {
                return None;
            }
        }

        // O_nonlin: polynomial monomials of O_lin.
        let o_nonlin = self.poly.generate(&o_lin);

        // Assemble full feature vector: [bias?, O_lin, O_nonlin].
        let bias_dim = if self.config.include_bias { 1 } else { 0 };
        let total_dim = bias_dim + o_lin.len() + o_nonlin.len();
        let mut features = Vec::with_capacity(total_dim);

        if self.config.include_bias {
            features.push(1.0);
        }
        features.extend_from_slice(&o_lin);
        features.extend_from_slice(&o_nonlin);

        Some(features)
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &NGRCConfig {
        &self.config
    }

    /// Number of observations pushed so far (including warmup).
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }
}

impl StreamingLearner for NextGenRC {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Record input dimension on first call.
        if self.n_inputs.is_none() {
            self.n_inputs = Some(features.len());
        }

        // Push the observation into the delay buffer.
        self.buffer.push(features);
        self.total_pushed += 1;

        // If warm, build features and train the RLS readout.
        if let Some(feat_vec) = self.build_features() {
            // Update per-feature activity EWMA for utilization entropy.
            const STATE_ALPHA: f64 = 0.01;
            if self.state_activity_ewma.len() != feat_vec.len() {
                self.state_activity_ewma = vec![0.0; feat_vec.len()];
            }
            for (ewma, &f) in self.state_activity_ewma.iter_mut().zip(feat_vec.iter()) {
                *ewma = (1.0 - STATE_ALPHA) * *ewma + STATE_ALPHA * f.abs();
            }

            // Update residual alignment tracking (acceleration-based).
            let current_pred = self.rls.predict(&feat_vec);
            let current_change = current_pred - self.prev_prediction;
            if self.samples_seen > 0 {
                let acceleration = current_change - self.prev_change;
                let prev_acceleration = self.prev_change - self.prev_prev_change;
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
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;

            self.rls.train_one(&feat_vec, target, weight);
            self.samples_seen += 1;
        }
    }

    fn predict(&self, _features: &[f64]) -> f64 {
        // Side-effect-free: uses the current buffer state without pushing.
        // If we are not warm or have never trained, return 0.0.
        if !self.is_warm() || self.n_inputs.is_none() {
            return 0.0;
        }

        // Build features from current buffer state (most recent observation
        // is whatever was last pushed, NOT the `features` argument).
        // This matches the streaming convention: predict uses current state.
        if let Some(feat_vec) = self.build_features() {
            self.rls.predict(&feat_vec)
        } else {
            0.0
        }
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.buffer.reset();
        self.rls.reset();
        self.total_pushed = 0;
        self.n_inputs = None;
        self.samples_seen = 0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.state_activity_ewma.clear();
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        use crate::automl::DiagnosticSource;
        match self.config_diagnostics() {
            Some(d) => [
                d.residual_alignment,
                d.regularization_sensitivity,
                d.depth_sufficiency,
                d.effective_dof,
                d.uncertainty,
            ],
            None => [0.0; 5],
        }
    }
}

impl fmt::Debug for NextGenRC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NextGenRC")
            .field("k", &self.config.k)
            .field("s", &self.config.s)
            .field("degree", &self.config.degree)
            .field("warmup_count", &self.warmup_count)
            .field("total_pushed", &self.total_pushed)
            .field("samples_seen", &self.samples_seen)
            .field("is_warm", &self.is_warm())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for NextGenRC {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // RLS saturation: 1.0 - trace(P) / (delta * d).
        let rls_saturation = {
            let p = self.rls.p_matrix();
            let d = self.rls.weights().len();
            if d > 0 && self.rls.delta() > 0.0 {
                let trace: f64 = (0..d).map(|i| p[i * d + i]).sum();
                (1.0 - trace / (self.rls.delta() * d as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        // Feature utilization entropy: normalized Shannon entropy of per-feature activity.
        let feature_entropy = {
            let sum: f64 = self.state_activity_ewma.iter().sum();
            if sum > 1e-15 && self.state_activity_ewma.len() > 1 {
                let n = self.state_activity_ewma.len();
                let ln_n = (n as f64).ln();
                let mut h = 0.0;
                for &a in &self.state_activity_ewma {
                    let p_i = a / sum;
                    if p_i > 1e-15 {
                        h -= p_i * p_i.ln();
                    }
                }
                (h / ln_n).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        let depth_sufficiency = 0.5 * rls_saturation + 0.5 * feature_entropy;

        // Weight magnitude: ||w||_2 / sqrt(d).
        let w = self.rls.weights();
        let effective_dof = if !w.is_empty() {
            let sq_sum: f64 = w.iter().map(|wi| wi * wi).sum();
            sq_sum.sqrt() / (w.len() as f64).sqrt()
        } else {
            0.0
        };

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            depth_sufficiency,
            effective_dof,
            uncertainty: self.rls.noise_variance().sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ngrc() -> NextGenRC {
        let config = NGRCConfig::builder().build().unwrap();
        NextGenRC::new(config)
    }

    #[test]
    fn cold_start_returns_zero() {
        let ngrc = default_ngrc();
        assert_eq!(ngrc.predict(&[1.0]), 0.0);
        assert_eq!(ngrc.n_samples_seen(), 0);
        assert!(!ngrc.is_warm());
    }

    #[test]
    fn warmup_then_trains() {
        let mut ngrc = default_ngrc();
        // Default k=2, s=1, so warmup = (2-1)*1 + 1 = 2.
        ngrc.train(&[1.0], 2.0);
        assert!(!ngrc.is_warm());
        assert_eq!(ngrc.n_samples_seen(), 0);

        ngrc.train(&[2.0], 3.0);
        assert!(ngrc.is_warm());
        assert_eq!(ngrc.n_samples_seen(), 1);
    }

    #[test]
    fn predict_is_side_effect_free() {
        let mut ngrc = default_ngrc();
        // Push enough to warm up.
        ngrc.train(&[1.0], 2.0);
        ngrc.train(&[2.0], 3.0);
        ngrc.train(&[3.0], 4.0);

        let samples_before = ngrc.n_samples_seen();
        let pushed_before = ngrc.total_pushed();

        let _ = ngrc.predict(&[99.0]);

        assert_eq!(
            ngrc.n_samples_seen(),
            samples_before,
            "predict should not change samples_seen",
        );
        assert_eq!(
            ngrc.total_pushed(),
            pushed_before,
            "predict should not push to buffer",
        );
    }

    #[test]
    fn reset_returns_to_cold_state() {
        let mut ngrc = default_ngrc();
        for i in 0..20 {
            ngrc.train(&[i as f64], (i + 1) as f64);
        }
        assert!(ngrc.is_warm());
        assert!(ngrc.n_samples_seen() > 0);

        ngrc.reset();
        assert!(!ngrc.is_warm());
        assert_eq!(ngrc.n_samples_seen(), 0);
        assert_eq!(ngrc.total_pushed(), 0);
        assert_eq!(ngrc.predict(&[1.0]), 0.0);
    }

    #[test]
    fn learns_linear_trend() {
        // y(t) = 3.0 * x(t) + 1.0
        let config = NGRCConfig::builder()
            .k(2)
            .s(1)
            .degree(2)
            .forgetting_factor(1.0)
            .build()
            .unwrap();
        let mut ngrc = NextGenRC::new(config);

        for i in 0..300 {
            let x = i as f64 * 0.01;
            let y = 3.0 * x + 1.0;
            ngrc.train(&[x], y);
        }

        // After training, the model should predict well on the training distribution.
        // Push one more sample to update the buffer, then predict.
        let x_test = 3.0;
        ngrc.train(&[x_test], 3.0 * x_test + 1.0);
        let pred = ngrc.predict(&[x_test]);
        let expected = 3.0 * x_test + 1.0;
        assert!(
            (pred - expected).abs() < 1.0,
            "expected ~{}, got {} (error = {})",
            expected,
            pred,
            (pred - expected).abs(),
        );
    }

    #[test]
    fn sine_wave_regression() {
        // Train NG-RC to predict sin(t + dt) from sin(t).
        let config = NGRCConfig::builder()
            .k(3)
            .s(1)
            .degree(2)
            .forgetting_factor(0.999)
            .delta(100.0)
            .build()
            .unwrap();
        let mut ngrc = NextGenRC::new(config);

        let dt = 0.1;
        let n_train = 500;

        for i in 0..n_train {
            let t = i as f64 * dt;
            let x = (t).sin();
            let target = (t + dt).sin();
            ngrc.train(&[x], target);
        }

        // Evaluate on a few test points (continue the series).
        let mut total_error = 0.0;
        let n_test = 50;
        for i in n_train..(n_train + n_test) {
            let t = i as f64 * dt;
            let x = (t).sin();
            let target = (t + dt).sin();
            ngrc.train(&[x], target);
            let pred = ngrc.predict(&[x]);
            total_error += (pred - target).abs();
        }
        let mae = total_error / n_test as f64;
        assert!(mae < 0.5, "sine wave MAE should be < 0.5, got {}", mae,);
    }

    #[test]
    fn trait_object_compatibility() {
        let config = NGRCConfig::builder().build().unwrap();
        let ngrc = NextGenRC::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(ngrc);
        boxed.train(&[1.0], 2.0);
        boxed.train(&[2.0], 3.0);
        boxed.train(&[3.0], 4.0);
        let pred = boxed.predict(&[4.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn skip_greater_than_one() {
        let config = NGRCConfig::builder().k(2).s(3).degree(2).build().unwrap();
        let mut ngrc = NextGenRC::new(config);

        // warmup = (2-1)*3 + 1 = 4
        for i in 0..3 {
            ngrc.train(&[i as f64], 0.0);
            assert!(!ngrc.is_warm(), "should not be warm after {} pushes", i + 1);
        }
        ngrc.train(&[3.0], 0.0);
        assert!(
            ngrc.is_warm(),
            "should be warm after 4 pushes with k=2, s=3"
        );
    }

    #[test]
    fn no_bias_config() {
        let config = NGRCConfig::builder().include_bias(false).build().unwrap();
        let mut ngrc = NextGenRC::new(config);

        for i in 0..10 {
            ngrc.train(&[i as f64], (i * 2) as f64);
        }
        // Should still produce finite predictions.
        let pred = ngrc.predict(&[5.0]);
        assert!(pred.is_finite());
    }
}
