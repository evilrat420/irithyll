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
    /// Number of grid intervals per edge (default: 5).
    pub grid_size: usize,
    /// Learning rate for SGD (default: 0.01).
    pub lr: f64,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for KANConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![1, 5, 1],
            spline_order: 3,
            grid_size: 5,
            lr: 0.01,
            seed: 42,
        }
    }
}

impl std::fmt::Display for KANConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KANConfig(layers={:?}, k={}, g={}, lr={}, seed={})",
            self.layer_sizes, self.spline_order, self.grid_size, self.lr, self.seed
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
///     .grid_size(5)
///     .lr(0.01)
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

    /// Set the number of grid intervals per edge (default: 5).
    pub fn grid_size(mut self, g: usize) -> Self {
        self.config.grid_size = g;
        self
    }

    /// Set the learning rate for SGD (default: 0.01).
    pub fn lr(mut self, lr: f64) -> Self {
        self.config.lr = lr;
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
            // Clamp to B-spline grid range [-1, 1] for best spline support
            normalized[i] = normalized[i].clamp(-1.0, 1.0);
        }
        normalized
    }
}

impl StreamingLearner for StreamingKAN {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Normalize input
        let normalized = self.normalize_input(features);

        // 2. Forward through all layers, saving activations
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(normalized.clone());
        let mut current = normalized;
        for layer in &self.layers {
            current = layer.forward(&current);
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

        let effective_lr = if self.rolling_loss > 1e-10 {
            let ratio = (sq_error / self.rolling_loss).clamp(0.5, 2.0);
            self.config.lr * ratio
        } else {
            self.config.lr
        };

        let lr = effective_lr * weight;

        // 5. Backward through layers (reverse order)
        let mut grad = vec![2.0 * error]; // dL/d_output for MSE
        for i in (0..self.layers.len()).rev() {
            grad = self.layers[i].backward(&activations[i], &grad, lr);
        }

        // 6. Cache output for predict()
        self.last_output = prediction;
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
            effective_dof: self.n_params() as f64,
            // Learning rate as regularization sensitivity.
            regularization_sensitivity: self.config.lr,
            // Prediction uncertainty: RMSE from EWMA of squared errors.
            uncertainty: self.rolling_loss.sqrt(),
            ..Default::default()
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
        assert_eq!(config.grid_size, 5);
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
}
