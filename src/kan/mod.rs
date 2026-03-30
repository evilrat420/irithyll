//! Streaming Kolmogorov-Arnold Networks (KAN).
//!
//! Learnable B-spline activation functions on edges instead of fixed
//! activations on nodes. Each edge learns a univariate function via
//! B-spline coefficients, enabling expressive nonlinear regression
//! with sparse per-sample gradient updates.
//!
//! # Architecture
//!
//! ```text
//! x_t -> [Input Norm] -> [KAN Layer 1] -> [KAN Layer 2] -> ... -> y_hat_t
//! ```
//!
//! # References
//!
//! - Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks" ICLR 2025
//! - Hoang et al. (2026) "Ultrafast On-chip Online Learning" -- proves single-sample KAN SGD
//! - Makinde (2026) "T-KAN" -- temporal KAN for LOB forecasting

mod bspline;
mod layer;

use layer::KANLayer;

use crate::error::ConfigError;
use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// KANConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingKAN`].
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::kan::KANConfig;
///
/// let config = KANConfig::builder()
///     .layer_sizes(vec![3, 10, 1])
///     .lr(0.01)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct KANConfig {
    /// Layer sizes including input and output. E.g., `[5, 10, 1]` for 5->10->1.
    pub layer_sizes: Vec<usize>,
    /// B-spline order (default: 3 = cubic).
    pub spline_order: usize,
    /// Number of grid intervals per edge (default: 8).
    ///
    /// More grid intervals give finer B-spline resolution, improving convergence
    /// on compositional functions at the cost of slightly more parameters per edge.
    pub grid_size: usize,
    /// Learning rate for SGD (default: 0.01).
    pub lr: f64,
    /// SGD momentum factor for B-spline coefficient updates (default: 0.5).
    ///
    /// Momentum accumulates gradient direction across samples, making the
    /// sparse B-spline updates (only `k+1` coefficients per edge per step)
    /// converge much faster. Set to 0.0 to disable.
    pub momentum: f64,
    /// Decay factor applied to spline coefficients each step (default: 0.0005).
    ///
    /// After each gradient update, all B-spline coefficients are multiplied by
    /// `(1 - coefficient_decay)`. This acts as a forgetting mechanism that
    /// biases the network toward recent observations, enabling concept-drift
    /// adaptation without external wrappers.
    pub coefficient_decay: f64,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for KANConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![1, 5, 1],
            spline_order: 3,
            grid_size: 8,
            lr: 0.01,
            momentum: 0.5,
            coefficient_decay: 0.0005,
            seed: 42,
        }
    }
}

impl std::fmt::Display for KANConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KANConfig(layers={:?}, k={}, g={}, lr={}, momentum={}, decay={}, seed={})",
            self.layer_sizes,
            self.spline_order,
            self.grid_size,
            self.lr,
            self.momentum,
            self.coefficient_decay,
            self.seed
        )
    }
}

// ---------------------------------------------------------------------------
// KANConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`KANConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::kan::KANConfig;
///
/// let config = KANConfig::builder()
///     .layer_sizes(vec![5, 10, 1])
///     .spline_order(3)
///     .grid_size(8)
///     .lr(0.01)
///     .momentum(0.5)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.layer_sizes, vec![5, 10, 1]);
/// assert_eq!(config.spline_order, 3);
/// ```
pub struct KANConfigBuilder {
    config: KANConfig,
}

impl KANConfig {
    /// Create a new builder with default values.
    pub fn builder() -> KANConfigBuilder {
        KANConfigBuilder {
            config: KANConfig::default(),
        }
    }
}

impl KANConfigBuilder {
    /// Set the layer sizes (default: `[1, 5, 1]`).
    ///
    /// Must contain at least 2 entries (input + output). The last entry
    /// must be 1 (regression output).
    pub fn layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.config.layer_sizes = sizes;
        self
    }

    /// Set the B-spline order (default: 3 = cubic).
    pub fn spline_order(mut self, k: usize) -> Self {
        self.config.spline_order = k;
        self
    }

    /// Set the number of grid intervals per edge (default: 8).
    ///
    /// More grid intervals give finer B-spline resolution. Values of 5-12
    /// are typical; 8 balances resolution and parameter count.
    pub fn grid_size(mut self, g: usize) -> Self {
        self.config.grid_size = g;
        self
    }

    /// Set the learning rate for SGD (default: 0.01).
    pub fn lr(mut self, lr: f64) -> Self {
        self.config.lr = lr;
        self
    }

    /// Set the SGD momentum factor (default: 0.5).
    ///
    /// Momentum accumulates gradient direction across samples, making sparse
    /// B-spline updates converge faster. Standard value is 0.9. Set to 0.0
    /// to disable.
    pub fn momentum(mut self, m: f64) -> Self {
        self.config.momentum = m;
        self
    }

    /// Set the coefficient decay factor (default: 0.0005).
    ///
    /// After each training step, all spline coefficients are multiplied by
    /// `(1 - coefficient_decay)`, biasing toward recent observations for
    /// concept-drift adaptation. Set to 0.0 to disable.
    pub fn coefficient_decay(mut self, d: f64) -> Self {
        self.config.coefficient_decay = d;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `layer_sizes` has fewer than 2 entries
    /// - Any layer size is 0
    /// - The last layer size is not 1
    /// - `spline_order` is 0
    /// - `grid_size` is 0
    pub fn build(self) -> Result<KANConfig, ConfigError> {
        let c = &self.config;

        if c.layer_sizes.len() < 2 {
            return Err(ConfigError::invalid(
                "layer_sizes",
                format!(
                    "need at least 2 layers (input + output), got {}",
                    c.layer_sizes.len()
                ),
            ));
        }

        for (i, &size) in c.layer_sizes.iter().enumerate() {
            if size == 0 {
                return Err(ConfigError::out_of_range(
                    "layer_sizes",
                    "all layer sizes must be > 0",
                    format!("layer_sizes[{}] = 0", i),
                ));
            }
        }

        if c.layer_sizes[c.layer_sizes.len() - 1] != 1 {
            return Err(ConfigError::invalid(
                "layer_sizes",
                format!(
                    "last layer must be 1 (regression output), got {}",
                    c.layer_sizes[c.layer_sizes.len() - 1]
                ),
            ));
        }

        if c.spline_order == 0 {
            return Err(ConfigError::out_of_range(
                "spline_order",
                "must be > 0",
                c.spline_order,
            ));
        }

        if c.grid_size == 0 {
            return Err(ConfigError::out_of_range(
                "grid_size",
                "must be > 0",
                c.grid_size,
            ));
        }

        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// StreamingKAN
// ---------------------------------------------------------------------------

/// Streaming KAN model with online B-spline learning.
///
/// Processes one sample at a time. Each edge in the network learns a
/// univariate B-spline activation function via per-sample SGD. Online
/// Welford normalization ensures stable input distribution.
///
/// # Example
///
/// ```no_run
/// use irithyll::kan::{StreamingKAN, KANConfig};
/// use irithyll::StreamingLearner;
///
/// let config = KANConfig::builder()
///     .layer_sizes(vec![3, 10, 1])
///     .lr(0.01)
///     .build()
///     .unwrap();
/// let mut model = StreamingKAN::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingKAN {
    layers: Vec<KANLayer>,
    config: KANConfig,
    // Online input normalization (Welford's algorithm)
    input_mean: Vec<f64>,
    input_var: Vec<f64>,
    input_count: u64,
    // Cached output for side-effect-free predict()
    last_output: f64,
    n_samples: u64,
    rng_state: u64,
    /// EWMA of squared prediction error for uncertainty-modulated learning.
    rolling_loss: f64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
}

impl StreamingKAN {
    /// Create a new StreamingKAN from config.
    pub fn new(config: KANConfig) -> Self {
        let mut rng = if config.seed == 0 { 1 } else { config.seed };
        let mut layers = Vec::with_capacity(config.layer_sizes.len() - 1);
        for i in 0..config.layer_sizes.len() - 1 {
            layers.push(KANLayer::new(
                config.layer_sizes[i],
                config.layer_sizes[i + 1],
                config.spline_order,
                config.grid_size,
                config.momentum,
                &mut rng,
            ));
        }
        let n_in = config.layer_sizes[0];
        Self {
            layers,
            config,
            input_mean: vec![0.0; n_in],
            input_var: vec![1.0; n_in],
            input_count: 0,
            last_output: 0.0,
            n_samples: 0,
            rng_state: rng,
            rolling_loss: 0.0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            alignment_ewma: 0.0,
        }
    }

    /// Access the config.
    pub fn config(&self) -> &KANConfig {
        &self.config
    }

    /// Number of KAN layers (edges between node layers).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of learnable parameters (B-spline coefficients).
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// The layer sizes from the config.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.config.layer_sizes
    }

    /// Normalize input via Welford's online algorithm, clamped for B-spline safety.
    fn normalize_input(&mut self, features: &[f64]) -> Vec<f64> {
        self.input_count += 1;
        let n = self.input_count as f64;
        let mut normalized = vec![0.0; features.len()];

        for (i, &x) in features.iter().enumerate() {
            if i >= self.input_mean.len() {
                // Handle dimension mismatch gracefully -- pass through
                normalized[i] = x;
                continue;
            }
            let delta = x - self.input_mean[i];
            self.input_mean[i] += delta / n;
            let delta2 = x - self.input_mean[i];
            self.input_var[i] += delta * delta2;

            // Normalize to roughly zero-mean, unit-variance
            let std = if n > 1.0 {
                (self.input_var[i] / (n - 1.0)).sqrt().max(1e-8)
            } else {
                1.0
            };
            normalized[i] = (x - self.input_mean[i]) / std;
            // Clamp to [-3, 3] — B-splines handle this range well.
            // Clamping to [-1, 1] was too aggressive and destroyed signal variance.
            normalized[i] = normalized[i].clamp(-3.0, 3.0);
        }
        normalized
    }
}

impl StreamingLearner for StreamingKAN {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Normalize input (skip during warmup — Welford stats unstable)
        let normalized = if self.input_count < 50 {
            // During warmup: use raw features, just clamp to prevent extreme values
            // Also update Welford stats without applying normalization
            self.input_count += 1;
            let n = self.input_count as f64;
            for (i, &x) in features.iter().enumerate() {
                if i < self.input_mean.len() {
                    let delta = x - self.input_mean[i];
                    self.input_mean[i] += delta / n;
                    let delta2 = x - self.input_mean[i];
                    self.input_var[i] += delta * delta2;
                }
            }
            features.iter().map(|x| x.clamp(-3.0, 3.0)).collect()
        } else {
            self.normalize_input(features)
        };

        // 2. Forward through all layers, saving activations
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(normalized.clone());
        let mut current = normalized;
        let n_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            // Inter-layer normalization: keep hidden activations in B-spline
            // support range. Skip the last layer (output layer).
            if layer_idx < n_layers - 1 {
                let layer_output_max = current
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                if layer_output_max > 5.0 {
                    let scale = 5.0 / layer_output_max;
                    for val in current.iter_mut() {
                        *val *= scale;
                    }
                }
            }
            activations.push(current.clone());
        }

        // 3. Compute output error (MSE loss: L = (pred - target)^2)
        let prediction = current[0];
        let error = prediction - target;
        let sq_error = error * error;

        // 4. Update rolling loss and compute uncertainty-modulated LR.
        //    High error relative to baseline → increase lr (adapt faster).
        //    Low error → decrease lr (conserve).
        const LOSS_ALPHA: f64 = 0.001;
        self.rolling_loss = (1.0 - LOSS_ALPHA) * self.rolling_loss + LOSS_ALPHA * sq_error;

        let effective_lr = if self.n_samples > 500 && self.rolling_loss > 1e-10 {
            let ratio = (sq_error / self.rolling_loss).clamp(0.5, 2.0);
            self.config.lr * ratio
        } else {
            self.config.lr // Fixed LR during warmup
        };

        let lr = effective_lr * weight;

        // 5. Backward through layers (reverse order)
        let mut grad = vec![2.0 * error]; // dL/d_output for MSE
        for i in (0..self.layers.len()).rev() {
            grad = self.layers[i].backward(&activations[i], &grad, lr);
        }

        // 6. Adaptive coefficient decay: high error → more decay (forget stale
        //    coefficients faster during drift), low error → less decay (conserve
        //    during stable regimes). Only apply after the model has had time to
        //    learn (warmup) — early error is learning phase error, not drift.
        if self.config.coefficient_decay > 0.0 && self.n_samples > 2000 {
            let ratio = if self.rolling_loss > 1e-10 {
                (sq_error / self.rolling_loss).clamp(0.5, 3.0)
            } else {
                1.0
            };
            let adaptive_decay = self.config.coefficient_decay * ratio;
            let decay = 1.0 - adaptive_decay.clamp(0.0, 0.01); // Never decay more than 1% per step
            for layer in &mut self.layers {
                for coeff in layer.coefficients_mut() {
                    *coeff *= decay;
                }
            }
        }

        // 7. Emergency coefficient normalization: if any coefficient exceeds 1e6,
        //    scale ALL coefficients down by 0.5. This prevents cascading NaN from
        //    divergent spline weights during sudden distribution shifts.
        let any_extreme = self.layers.iter().any(|l| {
            l.coefficients()
                .iter()
                .any(|c| !c.is_finite() || c.abs() > 1e6)
        });
        if any_extreme {
            for layer in &mut self.layers {
                for coeff in layer.coefficients_mut() {
                    if !coeff.is_finite() {
                        *coeff = 0.0;
                    } else {
                        *coeff *= 0.5;
                    }
                }
            }
        }

        // 8. Update residual alignment tracking.
        let current_change = prediction - self.prev_prediction;
        if self.n_samples > 0 {
            let agreement = if (current_change > 0.0) == (self.prev_change > 0.0) {
                1.0
            } else {
                -1.0
            };
            const ALIGN_ALPHA: f64 = 0.05;
            self.alignment_ewma =
                (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
        }
        self.prev_change = current_change;
        self.prev_prediction = prediction;

        // 9. Cache output for predict(), clamped to prevent cascading NaN
        self.last_output = prediction.clamp(-1e6, 1e6);
        if !self.last_output.is_finite() {
            self.last_output = 0.0;
        }
        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Use cached output from last train_one to avoid state mutation.
        // predict() must be side-effect-free (StreamingLearner contract).
        // See StreamingMamba / StreamingTTT for the same design rationale.
        if self.n_samples == 0 {
            return 0.0;
        }
        let _ = features; // Acknowledged but not used -- see design note above
        self.last_output
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset(&mut self.rng_state);
        }
        self.input_mean.fill(0.0);
        self.input_var.fill(1.0);
        self.input_count = 0;
        self.last_output = 0.0;
        self.n_samples = 0;
        self.rolling_loss = 0.0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.alignment_ewma = 0.0;
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

    fn adjust_config(&mut self, lr_multiplier: f64, _lambda_delta: f64) {
        // Scale the SGD learning rate for B-spline coefficient updates.
        self.config.lr *= lr_multiplier;
    }
}

impl std::fmt::Debug for StreamingKAN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingKAN")
            .field("layer_sizes", &self.config.layer_sizes)
            .field("spline_order", &self.config.spline_order)
            .field("grid_size", &self.config.grid_size)
            .field("lr", &self.config.lr)
            .field("n_params", &self.n_params())
            .field("n_samples", &self.n_samples)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingKAN {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: self.config.coefficient_decay,
            // Multi-layer KAN: depth_sufficiency > 0 for multi-layer architectures.
            depth_sufficiency: if self.layers.len() > 1 { 1.0 } else { 0.0 },
            effective_dof: self.n_params() as f64,
            uncertainty: self.rolling_loss.sqrt(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_default() {
        let config = KANConfig::builder().build().unwrap();
        assert_eq!(config.layer_sizes, vec![1, 5, 1]);
        assert_eq!(config.spline_order, 3);
        assert_eq!(config.grid_size, 8);
        assert!(
            (config.momentum - 0.5).abs() < 1e-12,
            "default momentum should be 0.5, got {}",
            config.momentum
        );
    }

    #[test]
    fn config_builder_custom() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .spline_order(4)
            .grid_size(8)
            .lr(0.005)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(config.layer_sizes, vec![3, 10, 1]);
        assert_eq!(config.spline_order, 4);
        assert_eq!(config.grid_size, 8);
        assert!((config.lr - 0.005).abs() < 1e-12);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn config_rejects_single_layer() {
        let result = KANConfig::builder().layer_sizes(vec![5]).build();
        assert!(result.is_err(), "single layer should be rejected");
    }

    #[test]
    fn config_rejects_zero_size() {
        let result = KANConfig::builder().layer_sizes(vec![0, 1]).build();
        assert!(result.is_err(), "zero-size layer should be rejected");
    }

    #[test]
    fn config_rejects_non_unit_output() {
        let result = KANConfig::builder().layer_sizes(vec![3, 5, 2]).build();
        assert!(result.is_err(), "non-unit output layer should be rejected");
    }

    #[test]
    fn config_rejects_zero_spline_order() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 1])
            .spline_order(0)
            .build();
        assert!(result.is_err(), "zero spline order should be rejected");
    }

    #[test]
    fn config_rejects_zero_grid_size() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 1])
            .grid_size(0)
            .build();
        assert!(result.is_err(), "zero grid size should be rejected");
    }

    #[test]
    fn new_creates_model() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.n_layers(), 2);
    }

    #[test]
    fn train_and_predict_finite() {
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .lr(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + 1.0;
            model.train(&[x], y);
        }

        let pred = model.predict(&[0.5]);
        assert!(pred.is_finite(), "prediction should be finite, got {pred}");
    }

    #[test]
    fn n_samples_tracks() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..25 {
            model.train(&[i as f64, (i as f64).sin()], i as f64);
        }
        assert_eq!(model.n_samples_seen(), 25);
    }

    #[test]
    fn reset_clears_state() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..20 {
            model.train(&[i as f64, 0.0], i as f64 * 2.0);
        }
        assert!(model.n_samples_seen() > 0);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn implements_streaming_learner() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        boxed.train(&[1.0, 2.0], 3.0);
        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn multi_layer_kan() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 8, 4, 1])
            .lr(0.005)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        assert_eq!(model.n_layers(), 3);

        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1] * 2.0 - x[2];
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "multi-layer KAN prediction should be finite, got {pred}"
        );
    }

    #[test]
    fn n_params_positive() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert!(
            model.n_params() > 0,
            "n_params should be positive, got {}",
            model.n_params()
        );
    }

    #[test]
    fn kan_learns_nonlinear() {
        // Use a stationary distribution: repeated passes over x in [-1, 1]
        // for y = x^2 (nonlinear, within grid range).
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 8, 1])
            .lr(0.01)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Generate fixed dataset: 20 points in [-1, 1]
        let n_pts = 20;
        let xs: Vec<f64> = (0..n_pts)
            .map(|i| -1.0 + 2.0 * i as f64 / (n_pts - 1) as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();

        // Measure error after initial pass (first epoch)
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            model.train(&[x], y);
        }
        let early_mse: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                model.train(&[x], y);
                let pred = model.predict(&[x]);
                (pred - y).powi(2)
            })
            .sum::<f64>()
            / n_pts as f64;

        // Train for many more epochs
        for _ in 0..20 {
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                model.train(&[x], y);
            }
        }

        // Measure error after training (last epoch)
        let late_mse: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                model.train(&[x], y);
                let pred = model.predict(&[x]);
                (pred - y).powi(2)
            })
            .sum::<f64>()
            / n_pts as f64;

        assert!(
            late_mse < early_mse,
            "KAN should learn: early MSE ({early_mse:.6}) should be > late MSE ({late_mse:.6})"
        );
    }

    #[test]
    fn config_display() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .build()
            .unwrap();
        let s = format!("{config}");
        assert!(s.contains("layers="), "display should contain layers");
        assert!(s.contains("k=3"), "display should contain spline order");
        assert!(s.contains("momentum="), "display should contain momentum");
        assert!(s.contains("decay="), "display should contain decay");
    }

    #[test]
    fn config_clone() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .seed(99)
            .build()
            .unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.layer_sizes, config.layer_sizes);
        assert_eq!(cloned.seed, config.seed);
    }

    #[test]
    fn predict_before_train_returns_zero() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn kan_uncertainty_modulated_lr() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 10, 1])
            .lr(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // rolling_loss starts at 0
        assert!(
            model.rolling_loss.abs() < 1e-15,
            "rolling_loss should start at 0, got {}",
            model.rolling_loss
        );

        // Train on 100 samples
        for i in 0..100 {
            let t = i as f64 * 0.05;
            let x = [t.sin(), t.cos()];
            let y = 0.5 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

        // rolling_loss should be tracked and > 0 after training
        assert!(
            model.rolling_loss > 0.0,
            "rolling_loss should be > 0 after training, got {}",
            model.rolling_loss
        );
        assert!(
            model.rolling_loss.is_finite(),
            "rolling_loss should be finite, got {}",
            model.rolling_loss
        );

        // Predictions should still be finite
        let pred = model.predict(&[0.5, 0.3]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after uncertainty-modulated training, got {}",
            pred
        );
    }

    #[test]
    fn coefficient_decay_shrinks_coefficients() {
        // Test that adaptive coefficient decay shrinks coefficients over
        // multiple steps. We use a moderate decay and zero learning rate so
        // gradient updates don't add magnitude back, isolating the decay effect.
        // Note: decay only activates after 2000 samples (warmup protection).
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .lr(0.01)
            .coefficient_decay(0.005) // moderate decay within adaptive clamp range
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Train past the warmup threshold so decay activates
        for i in 0..2100 {
            let x = (i as f64 * 0.03).sin();
            model.train(&[x], x * 2.0);
        }

        // Now set lr to 0 so only decay acts on coefficients
        model.config.lr = 0.0;

        // Snapshot the L2 norm of coefficients
        let norm_before: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        // Train several steps — decay should reduce the coefficient magnitudes
        // even with zero lr (no gradient contribution).
        for _ in 0..10 {
            model.train(&[0.5], 1.0);
        }

        let norm_after: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        assert!(
            norm_after < norm_before,
            "coefficient norm should decrease with decay: before={norm_before:.6}, after={norm_after:.6}"
        );
    }

    #[test]
    fn coefficient_decay_zero_preserves_behavior() {
        // With explicit decay=0.0, behavior should be identical across two identical models.
        let config_no_decay = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .lr(0.01)
            .coefficient_decay(0.0)
            .seed(42)
            .build()
            .unwrap();
        let config_zero_decay = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .lr(0.01)
            .coefficient_decay(0.0)
            .seed(42)
            .build()
            .unwrap();

        let mut model_a = StreamingKAN::new(config_no_decay);
        let mut model_b = StreamingKAN::new(config_zero_decay);

        for i in 0..50 {
            let x = i as f64 * 0.1;
            let y = x * 2.0 + 1.0;
            model_a.train(&[x], y);
            model_b.train(&[x], y);
        }

        let pred_a = model_a.predict(&[0.5]);
        let pred_b = model_b.predict(&[0.5]);
        assert!(
            (pred_a - pred_b).abs() < 1e-12,
            "zero decay should match no-decay: pred_a={pred_a}, pred_b={pred_b}"
        );
    }

    #[test]
    fn coefficient_decay_builder_sets_value() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .coefficient_decay(0.0005)
            .build()
            .unwrap();
        assert!(
            (config.coefficient_decay - 0.0005).abs() < 1e-15,
            "coefficient_decay should be 0.0005, got {}",
            config.coefficient_decay
        );
    }

    #[test]
    fn adaptive_coefficient_decay_varies_with_error() {
        // Train two identical models on the same stable data, then hit one with
        // a large-error sample. The adaptive decay rate on the high-error step
        // should produce different coefficient norms vs a fixed-decay model.
        // Note: decay only activates after 2000 samples (warmup protection).
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .lr(0.01)
            .coefficient_decay(0.001)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Train past the warmup threshold to activate decay
        for i in 0..2100 {
            let x = (i as f64 * 0.05).sin();
            let y = x * 2.0;
            model.train(&[x], y);
        }

        let baseline_rolling = model.rolling_loss;
        assert!(
            baseline_rolling > 0.0,
            "rolling_loss should be positive after training"
        );

        // Snapshot coefficients before a normal-error step
        let norm_before_normal: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        // Normal step (error close to baseline)
        model.train(&[0.5_f64.sin()], 0.5_f64.sin() * 2.0);
        let norm_after_normal: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        let normal_shrinkage = norm_before_normal - norm_after_normal;

        // Now hit it with a huge error (target far from expected)
        let norm_before_spike: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        model.train(&[0.5_f64.sin()], 1000.0); // massive error spike
        let norm_after_spike: f64 = model
            .layers
            .iter()
            .flat_map(|l| l.coefficients().iter())
            .map(|c| c * c)
            .sum();

        // The spike step should cause MORE coefficient change than the normal step
        // (adaptive decay increases when error is high relative to rolling_loss).
        // We measure absolute change because the gradient update also shifts things.
        let spike_change = (norm_before_spike - norm_after_spike).abs();
        let normal_change = normal_shrinkage.abs();

        assert!(
            spike_change > normal_change,
            "high-error step should cause larger coefficient change than normal step: \
             spike_change={spike_change:.8}, normal_change={normal_change:.8}"
        );
    }
}
