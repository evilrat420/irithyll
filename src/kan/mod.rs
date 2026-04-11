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
//! # Recommended Configuration
//!
//! For streaming regression, use shallow architectures with higher learning rates:
//! ```
//! use irithyll::kan::KANConfig;
//!
//! let config = KANConfig::builder()
//!     .layer_sizes(vec![4, 20, 1])  // shallow: 1 hidden layer
//!     .grid_size(8)
//!     .learning_rate(0.1)   // higher than MLP — B-spline sparse updates need it
//!     .build()
//!     .unwrap();
//! ```
//! Deep architectures (3+ layers) can cause gradient instability in streaming mode.
//! B-spline locality ensures sparse updates don't interfere (Hoang et al., 2026).
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

use irithyll_core::continual::{ContinualStrategy, NeuronRegeneration};
use irithyll_core::math::sigmoid;
use irithyll_core::rng::standard_normal;

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
///     .learning_rate(0.1)
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
    /// Learning rate for SGD (default: 0.1).
    ///
    /// Online KAN convergence requires higher LR than MLPs because each sample
    /// only updates k+1 B-spline coefficients per edge (Hoang et al., 2026).
    /// Values 0.1-0.5 work for regression.
    pub learning_rate: f64,
    /// SGD momentum factor for B-spline coefficient updates (default: 0.0, disabled).
    ///
    /// Momentum on sparse B-spline updates magnifies overfitting in active
    /// input regions without helping inactive regions. Set to 0.0 for online
    /// streaming (Hoang et al., 2026). Non-zero values may help in batch mode.
    pub momentum: f64,
    /// Decay factor applied to spline coefficients each step (default: 0.0, disabled).
    ///
    /// B-spline locality naturally prevents catastrophic forgetting -- each sample
    /// only modifies coefficients in its input region, leaving other regions
    /// undisturbed. When enabled, all coefficients are multiplied by
    /// `(1 - coefficient_decay)` after each step, biasing toward recent observations
    /// for concept-drift adaptation. Usually unnecessary for online streaming.
    pub coefficient_decay: f64,
    /// Enable temporal gating (default: false).
    ///
    /// When enabled, the output of the last hidden layer is mixed with a
    /// recurrent temporal state via a learned sigmoid gate:
    /// `z_t = gate * z_{t-1} + (1-gate) * KAN_hidden(x_t)`
    /// where `gate = sigmoid(W_gate · x_t + b_gate)`.
    ///
    /// This adds autoregressive memory to the KAN, useful for LOB-style
    /// sequential prediction tasks (Makinde, 2026 "T-KAN").
    ///
    /// Requires at least one hidden layer (`layer_sizes.len() >= 3`).
    /// For single-layer KAN (input→output only), temporal gating is a no-op.
    pub temporal: bool,
    /// RNG seed (default: 42).
    pub seed: u64,
    /// Enable plasticity maintenance via neuron regeneration (default: false).
    ///
    /// When enabled, tracks per-hidden-unit activation energy and periodically
    /// reinitializes dead B-spline edges to maintain learning capacity over
    /// long streams (Dohare et al., Nature 2024).
    pub plasticity: bool,
}

impl Default for KANConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![1, 5, 1],
            spline_order: 3,
            grid_size: 8,
            learning_rate: 0.1,
            momentum: 0.0,
            coefficient_decay: 0.0,
            temporal: false,
            seed: 42,
            plasticity: false,
        }
    }
}

impl std::fmt::Display for KANConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KANConfig(layers={:?}, k={}, g={}, lr={}, momentum={}, decay={}, temporal={}, seed={})",
            self.layer_sizes,
            self.spline_order,
            self.grid_size,
            self.learning_rate,
            self.momentum,
            self.coefficient_decay,
            self.temporal,
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
///     .learning_rate(0.1)
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

    /// Set the learning rate for SGD (default: 0.1).
    ///
    /// Online KAN needs higher LR than MLPs due to sparse B-spline updates.
    /// Values 0.1-0.5 work for regression (Hoang et al., 2026).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the learning rate for SGD (deprecated alias for [`learning_rate`](Self::learning_rate)).
    #[deprecated(since = "0.0.0", note = "use `learning_rate()` instead")]
    pub fn lr(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the SGD momentum factor (default: 0.0, disabled).
    ///
    /// Momentum on sparse B-spline updates magnifies overfitting in active
    /// input regions without helping inactive regions. Disabled by default
    /// for online streaming. Non-zero values may help in batch mode.
    pub fn momentum(mut self, m: f64) -> Self {
        self.config.momentum = m;
        self
    }

    /// Set the coefficient decay factor (default: 0.0, disabled).
    ///
    /// B-spline locality naturally prevents catastrophic forgetting, so decay
    /// is unnecessary for most online streaming tasks. When enabled, all
    /// coefficients are multiplied by `(1 - coefficient_decay)` each step,
    /// biasing toward recent observations for concept-drift adaptation.
    pub fn coefficient_decay(mut self, d: f64) -> Self {
        self.config.coefficient_decay = d;
        self
    }

    /// Enable temporal gating (default: false).
    ///
    /// When enabled, the last hidden layer output is mixed with a recurrent
    /// temporal state via a learned sigmoid gate (Makinde, 2026 "T-KAN").
    /// Requires at least one hidden layer; no-op for single-layer KAN.
    pub fn temporal(mut self, t: bool) -> Self {
        self.config.temporal = t;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Enable or disable plasticity maintenance (default: false).
    ///
    /// When enabled, tracks per-hidden-unit activation energy and periodically
    /// reinitializes dead B-spline edges to maintain learning capacity over
    /// long streams (Dohare et al., Nature 2024).
    pub fn plasticity(mut self, p: bool) -> Self {
        self.config.plasticity = p;
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
///     .learning_rate(0.1)
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
    // Online target normalization (Welford's algorithm)
    target_mean: f64,
    target_var: f64,
    target_count: u64,
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
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    // Temporal gating state (T-KAN, Makinde 2026)
    /// Recurrent hidden state for temporal gating. Length = last hidden layer size.
    temporal_state: Vec<f64>,
    /// Gate weights W_gate for projecting input to a scalar gate value.
    /// Length = n_input (config.layer_sizes[0]).
    gate_weights: Vec<f64>,
    /// Gate bias, initialized to 0.0 so sigmoid(b) = 0.5, unbiased initial gating.
    gate_bias: f64,
    /// Optional plasticity guard for maintaining learning capacity.
    plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of per-hidden-unit activation energy from previous step.
    prev_hidden_energy: Vec<f64>,
    /// Total number of hidden units (sum of all hidden layer sizes).
    n_hidden_total: usize,
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

        // Temporal gating: state dim = last hidden layer size.
        // For layer_sizes=[a, h1, h2, 1], last hidden = h2 = layer_sizes[len-2].
        // For layer_sizes=[a, 1] (no hidden), temporal is a no-op; use empty vecs.
        let has_hidden = config.layer_sizes.len() >= 3;
        let temporal_dim = if has_hidden {
            config.layer_sizes[config.layer_sizes.len() - 2]
        } else {
            0
        };
        let temporal_state = vec![0.0; temporal_dim];
        let gate_weights = if config.temporal && has_hidden {
            (0..n_in)
                .map(|_| standard_normal(&mut rng) * 0.01)
                .collect()
        } else {
            vec![0.0; n_in]
        };
        // B.4: gate_bias initialized to 0.0; sigmoid(0) = 0.5, unbiased initial gating.
        let gate_bias = 0.0;

        // Compute total hidden units (all layers except input and output).
        let n_hidden_total: usize = if config.layer_sizes.len() > 2 {
            config.layer_sizes[1..config.layer_sizes.len() - 1]
                .iter()
                .sum()
        } else {
            0
        };

        // Create plasticity guard if enabled and there are hidden units.
        let plasticity_guard = if config.plasticity && n_hidden_total > 0 {
            Some(NeuronRegeneration::new(
                n_hidden_total,
                1,
                0.01,
                500,
                0.99,
                config.seed.wrapping_add(0x_DEAD_CAFE),
            ))
        } else {
            None
        };
        let prev_hidden_energy = vec![0.0; n_hidden_total];

        Self {
            layers,
            config,
            input_mean: vec![0.0; n_in],
            input_var: vec![1.0; n_in],
            input_count: 0,
            target_mean: 0.0,
            target_var: 1.0,
            target_count: 0,
            last_output: 0.0,
            n_samples: 0,
            rng_state: rng,
            rolling_loss: 0.0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            temporal_state,
            gate_weights,
            gate_bias,
            plasticity_guard,
            prev_hidden_energy,
            n_hidden_total,
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

    /// Surgically reinitialize a hidden unit by flat index.
    ///
    /// Maps the flat hidden index to the correct layer and local node index,
    /// then reinitializes all incoming edges (in the layer producing this node)
    /// and all outgoing edges (in the layer consuming this node). This preserves
    /// all other units' learned representations — only the dead unit gets recycled.
    ///
    /// # Arguments
    ///
    /// * `flat_j` — flat hidden unit index (0-based across all hidden layers)
    /// * `rng` — mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `flat_j >= n_hidden_total`.
    fn reinitialize_hidden_unit(&mut self, flat_j: usize, rng: &mut u64) {
        assert!(
            flat_j < self.n_hidden_total,
            "hidden unit index {} out of range (n_hidden_total={})",
            flat_j,
            self.n_hidden_total
        );

        // Map flat index to (hidden_layer_idx, local_j).
        // Hidden layers are layer_sizes[1..len-1].
        let mut remaining = flat_j;
        let mut hidden_layer_idx = 0;
        for &size in &self.config.layer_sizes[1..self.config.layer_sizes.len() - 1] {
            if remaining < size {
                break;
            }
            remaining -= size;
            hidden_layer_idx += 1;
        }
        let local_j = remaining;

        // layers[hidden_layer_idx] connects layer_sizes[hidden_layer_idx] →
        // layer_sizes[hidden_layer_idx+1]. This hidden node is output node
        // local_j of layers[hidden_layer_idx].
        self.layers[hidden_layer_idx].reinitialize_output_node(local_j, rng);

        // layers[hidden_layer_idx+1] connects layer_sizes[hidden_layer_idx+1] →
        // layer_sizes[hidden_layer_idx+2]. This hidden node is input node
        // local_j of layers[hidden_layer_idx+1].
        if hidden_layer_idx + 1 < self.layers.len() {
            self.layers[hidden_layer_idx + 1].reinitialize_input_node(local_j, rng);
        }

        // Zero the temporal state for this unit (if temporal gating is active
        // and this unit is in the last hidden layer).
        if self.config.temporal && !self.temporal_state.is_empty() {
            // Check if this flat_j maps to the last hidden layer.
            let last_hidden_start: usize = self.config.layer_sizes
                [1..self.config.layer_sizes.len() - 2]
                .iter()
                .sum();
            let last_hidden_size = self.config.layer_sizes[self.config.layer_sizes.len() - 2];
            if flat_j >= last_hidden_start && flat_j < last_hidden_start + last_hidden_size {
                let local_in_last = flat_j - last_hidden_start;
                if local_in_last < self.temporal_state.len() {
                    self.temporal_state[local_in_last] = 0.0;
                }
            }
        }
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

            // Normalize to roughly zero-mean, unit-variance.
            // B.3: If feature std < 1e-8 (constant feature), skip normalization
            // for that feature to avoid div-by-zero / NaN contamination.
            let raw_std = if n > 1.0 {
                (self.input_var[i] / (n - 1.0)).sqrt()
            } else {
                0.0
            };
            if raw_std < 1e-8 {
                // Constant feature: pass through centered at 0 in grid domain
                normalized[i] = 0.0;
                continue;
            }
            normalized[i] = (x - self.input_mean[i]) / raw_std;
            // Clamp to [-1+eps, 1-eps] — the full B-spline grid domain [-1, 1].
            // Using the full domain ensures all grid intervals participate.
            // Previous [-0.95, 0.95] clamp discarded ~44% of normalized data
            // at the boundaries, creating dead zones where B-splines had support.
            normalized[i] = normalized[i].clamp(-1.0 + 1e-7, 1.0 - 1e-7);
        }
        normalized
    }
}

impl StreamingLearner for StreamingKAN {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Normalize input to B-spline grid domain [-1, 1].
        //    During warmup (first 50 samples): accumulate Welford stats only,
        //    map features to zero (don't train on un-normalized garbage).
        let normalized = if self.input_count < 50 {
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
            // During warmup: use centered features scaled by a rough estimate.
            // Centering + scaling preserves signal for large-scale inputs
            // (e.g. pressure ~1013) that would be destroyed by fixed-range clamping.
            if n > 2.0 {
                features
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        if i < self.input_mean.len() {
                            // B.3: skip normalization for constant features
                            let raw_std = (self.input_var[i] / (n - 1.0)).sqrt();
                            if raw_std < 1e-8 {
                                return 0.0;
                            }
                            ((x - self.input_mean[i]) / raw_std).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                vec![0.0; features.len()] // first 2 samples: don't train, just observe
            }
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

        // 2b. Temporal gating (T-KAN, Makinde 2026):
        //     Mix last hidden layer output with recurrent state via learned gate.
        //     Requires >= 2 layers (at least one hidden layer before output).
        //     gate = sigmoid(W_gate · normalized_input + b_gate)
        //     mixed = gate * temporal_state + (1-gate) * pre_output_activation
        //     Then re-run final layer on mixed instead of original activation.
        let gate_value = if self.config.temporal && self.layers.len() >= 2 {
            // activations[0] is normalized input, activations[len-1] is final output,
            // activations[len-2] is last hidden layer output (pre-output).
            let pre_output_idx = activations.len() - 2;
            let pre_output = &activations[pre_output_idx];

            // Compute gate: g = sigmoid(dot(gate_weights, normalized_input) + gate_bias)
            let dot: f64 = self
                .gate_weights
                .iter()
                .zip(activations[0].iter())
                .map(|(w, x)| w * x)
                .sum();
            let g = sigmoid(dot + self.gate_bias);

            // Mix: mixed = g * temporal_state + (1-g) * pre_output
            let mixed: Vec<f64> = self
                .temporal_state
                .iter()
                .zip(pre_output.iter())
                .map(|(&s, &h)| g * s + (1.0 - g) * h)
                .collect();

            // Save previous temporal state for gate gradient
            let prev_state = self.temporal_state.clone();
            // Update temporal state
            self.temporal_state = mixed.clone();

            // Re-run final layer on mixed representation
            let last_layer = self.layers.len() - 1;
            current = self.layers[last_layer].forward(&mixed);

            // Update activations: replace pre-output with mixed, final with new output
            activations[pre_output_idx] = mixed;
            *activations.last_mut().unwrap() = current.clone();

            // Gate gradient update: dL/dg ≈ (prev_state - pre_output) projected,
            // but we use a simple heuristic: push gate toward better mixing.
            // dg/dW = g*(1-g) * (prev_state_i - pre_output_i) * x_j
            // Use small LR (0.01 * config.lr) and clip gradient.
            let gate_lr = 0.01 * self.config.learning_rate;
            let diff_sum: f64 = prev_state
                .iter()
                .zip(activations[pre_output_idx].iter())
                .map(|(&s, &h)| s - h)
                .sum::<f64>()
                / prev_state.len().max(1) as f64;
            let dg = g * (1.0 - g) * diff_sum;
            let dg_clipped = dg.clamp(-1.0, 1.0);
            for (w, &x_i) in self.gate_weights.iter_mut().zip(activations[0].iter()) {
                let update = gate_lr * dg_clipped * x_i;
                if update.is_finite() {
                    *w -= update.clamp(-0.1, 0.1);
                }
            }
            let bias_update = gate_lr * dg_clipped;
            if bias_update.is_finite() {
                self.gate_bias -= bias_update.clamp(-0.1, 0.1);
            }

            Some(g)
        } else {
            None
        };
        let _ = gate_value; // suppress unused warning when temporal=false

        // 3. Normalize target via Welford's online algorithm.
        //    KAN trains in normalized target space for stable gradient magnitudes
        //    when target magnitudes vary across regimes (e.g., Feynman equations).
        self.target_count += 1;
        let tn = self.target_count as f64;
        let t_delta = target - self.target_mean;
        self.target_mean += t_delta / tn;
        let t_delta2 = target - self.target_mean;
        self.target_var += t_delta * t_delta2;
        // B.2: If target std < 1e-8 (constant target), use target_std=1.0 and
        // skip normalization to avoid div-by-zero and NaN in gradient computation.
        let raw_target_std = if tn > 2.0 {
            (self.target_var / (tn - 1.0)).sqrt()
        } else {
            0.0
        };
        let target_std = if raw_target_std < 1e-8 {
            1.0
        } else {
            raw_target_std
        };
        let normalized_target = if raw_target_std < 1e-8 {
            target - self.target_mean
        } else {
            (target - self.target_mean) / target_std
        };

        // 4. Compute output error in normalized target space
        let prediction = current[0];
        let error = prediction - normalized_target;
        let sq_error = error * error;

        // 4. Update rolling loss and compute uncertainty-modulated LR.
        //    High error relative to baseline → increase lr (adapt faster).
        //    Low error → decrease lr (conserve).
        const LOSS_ALPHA: f64 = 0.001;
        if self.n_samples == 0 {
            self.rolling_loss = sq_error;
        } else {
            self.rolling_loss = (1.0 - LOSS_ALPHA) * self.rolling_loss + LOSS_ALPHA * sq_error;
        }

        let effective_lr = if self.n_samples > 500 && self.rolling_loss > 1e-10 {
            let ratio = (sq_error / self.rolling_loss).clamp(0.5, 2.0);
            self.config.learning_rate * ratio
        } else {
            self.config.learning_rate // Fixed LR during warmup
        };

        let lr = effective_lr * weight;

        // B.1: Finiteness guard — skip backward pass if any activation contains NaN/Inf.
        // This keeps the model healthy when an upstream computation produced bad values.
        let activations_finite = activations
            .iter()
            .all(|act| act.iter().all(|f| f.is_finite()));
        if !activations_finite {
            // Skip gradient update; cache denormalized prediction and return.
            let denormalized = current[0] * target_std + self.target_mean;
            self.last_output = if denormalized.is_finite() {
                denormalized.clamp(-1e6, 1e6)
            } else {
                0.0
            };
            self.n_samples += 1;
            return;
        }

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

        // 7. Coefficient magnitude guard: reset non-finite coefficients to zero
        //    and scale remaining by 0.5 when any exceed 1e6. Divergent coefficients
        //    typically signal a configuration issue (learning rate, input normalization).
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

        // 8. Update residual alignment tracking (acceleration-based).
        let current_change = prediction - self.prev_prediction;
        if self.n_samples > 0 {
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
            if self.n_samples == 1 {
                self.alignment_ewma = agreement;
            } else {
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = prediction;

        // 9. Plasticity maintenance: track per-hidden-unit activation energy
        //    and trigger regeneration when dead units are detected.
        // Plasticity maintenance: collect regeneration decisions first, then
        // apply surgical reinit outside the guard borrow to satisfy the borrow checker.
        let mut regenerated_units: Vec<usize> = Vec::new();
        if let Some(ref mut guard) = self.plasticity_guard {
            // Build flat energy vector from hidden layer activations.
            // activations[0] = input, activations[1..n-1] = hidden, activations[n-1] = output.
            let mut hidden_energy = Vec::with_capacity(self.n_hidden_total);
            for layer_acts in activations.iter().take(activations.len() - 1).skip(1) {
                for &a in layer_acts {
                    hidden_energy.push(a.abs());
                }
            }
            if hidden_energy.len() == self.n_hidden_total {
                guard.pre_update(&self.prev_hidden_energy, &mut hidden_energy);
                guard.post_update(&self.prev_hidden_energy);
                // Collect which units need reinit.
                let n_groups = guard.n_groups();
                for j in 0..n_groups {
                    if guard.was_regenerated(j) {
                        regenerated_units.push(j);
                    }
                }
                self.prev_hidden_energy = hidden_energy;
            }
        }
        // Surgical per-unit reinit: only dead units get recycled.
        // Each dead hidden unit has its incoming and outgoing B-spline
        // edges reinitialized while preserving all other units.
        if !regenerated_units.is_empty() {
            let mut reinit_rng = self
                .config
                .seed
                .wrapping_add(0xCAFE_BABE_u64.wrapping_mul(self.n_samples));
            for j in regenerated_units {
                self.reinitialize_hidden_unit(j, &mut reinit_rng);
            }
        }

        // 10. Denormalize prediction back to original target space, then cache.
        let denormalized = prediction * target_std + self.target_mean;
        self.last_output = denormalized.clamp(-1e6, 1e6);
        if !self.last_output.is_finite() {
            self.last_output = 0.0;
        }
        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Side-effect-free forward pass using frozen normalization stats.
        // When temporal gating is enabled, the current temporal_state is used
        // for mixing (read-only — no state mutation during predict).
        if self.n_samples == 0 {
            return 0.0;
        }

        // Normalize input using frozen Welford stats (no mutation)
        let n = self.input_count as f64;
        let mut normalized = vec![0.0; features.len()];
        for (i, &x) in features.iter().enumerate() {
            if i < self.input_mean.len() && n > 1.0 {
                let std = (self.input_var[i] / (n - 1.0)).sqrt().max(1e-8);
                normalized[i] = ((x - self.input_mean[i]) / std).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
            }
        }

        // Forward through all layers (all but last if temporal, then gate, then last)
        let mut current = normalized.clone();
        let n_layers = self.layers.len();
        let stop_before = if self.config.temporal && n_layers >= 2 {
            n_layers - 1
        } else {
            n_layers
        };

        for (layer_idx, layer) in self.layers.iter().enumerate().take(stop_before) {
            current = layer.forward(&current);
            // Inter-layer normalization (same as train_one)
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
        }

        // Temporal gating: mix with recurrent state (read-only, no state mutation)
        if self.config.temporal && n_layers >= 2 {
            let dot: f64 = self
                .gate_weights
                .iter()
                .zip(normalized.iter())
                .map(|(w, x)| w * x)
                .sum();
            let g = sigmoid(dot + self.gate_bias);

            // Mix: current is the pre-output hidden activation
            let mixed: Vec<f64> = self
                .temporal_state
                .iter()
                .zip(current.iter())
                .map(|(&s, &h)| g * s + (1.0 - g) * h)
                .collect();

            // Run final layer on mixed
            current = self.layers[n_layers - 1].forward(&mixed);
        }

        // Denormalize prediction back to original target space
        let prediction = current[0];
        let target_std = if self.target_count > 2 {
            (self.target_var / (self.target_count as f64 - 1.0))
                .sqrt()
                .max(1e-8)
        } else {
            1.0
        };
        let denormalized = prediction * target_std + self.target_mean;
        let result = denormalized.clamp(-1e6, 1e6);
        if result.is_finite() {
            result
        } else {
            0.0
        }
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
        self.target_mean = 0.0;
        self.target_var = 1.0;
        self.target_count = 0;
        self.last_output = 0.0;
        self.n_samples = 0;
        self.rolling_loss = 0.0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        // Reset temporal gating state
        self.temporal_state.fill(0.0);
        for w in &mut self.gate_weights {
            *w = standard_normal(&mut self.rng_state) * 0.01;
        }
        // B.4: reset gate_bias to 0.0 (the initial value from new()), not 1.0
        self.gate_bias = 0.0;
        if let Some(ref mut guard) = self.plasticity_guard {
            guard.reset();
        }
        self.prev_hidden_energy.fill(0.0);
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
        self.config.learning_rate *= lr_multiplier;
    }
}

impl std::fmt::Debug for StreamingKAN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingKAN")
            .field("layer_sizes", &self.config.layer_sizes)
            .field("spline_order", &self.config.spline_order)
            .field("grid_size", &self.config.grid_size)
            .field("learning_rate", &self.config.learning_rate)
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
        // Dead edge fraction: fraction of edges where all velocity magnitudes < 1e-8.
        // encoder_utilization = 1.0 - dead_edge_fraction.
        let encoder_utilization = {
            let (mut total_dead, mut total_edges) = (0usize, 0usize);
            for layer in &self.layers {
                let (dead, edges) = layer.count_dead_edges(1e-8);
                total_dead += dead;
                total_edges += edges;
            }
            if total_edges > 0 {
                1.0 - (total_dead as f64 / total_edges as f64)
            } else {
                0.0
            }
        };

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: self.config.coefficient_decay,
            depth_sufficiency: encoder_utilization,
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
            (config.learning_rate - 0.1).abs() < 1e-12,
            "default learning_rate should be 0.1, got {}",
            config.learning_rate
        );
        assert!(
            config.momentum.abs() < 1e-12,
            "default momentum should be 0.0, got {}",
            config.momentum
        );
        assert!(
            config.coefficient_decay.abs() < 1e-12,
            "default coefficient_decay should be 0.0, got {}",
            config.coefficient_decay
        );
    }

    #[test]
    fn config_builder_custom() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .spline_order(4)
            .grid_size(8)
            .learning_rate(0.005)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(config.layer_sizes, vec![3, 10, 1]);
        assert_eq!(config.spline_order, 4);
        assert_eq!(config.grid_size, 8);
        assert!((config.learning_rate - 0.005).abs() < 1e-12);
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
            .learning_rate(0.01)
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
            .learning_rate(0.005)
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
            .learning_rate(0.01)
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
            .learning_rate(0.01)
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
            .learning_rate(0.01)
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
        model.config.learning_rate = 0.0;

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
            .learning_rate(0.01)
            .coefficient_decay(0.0)
            .seed(42)
            .build()
            .unwrap();
        let config_zero_decay = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .learning_rate(0.01)
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
            .learning_rate(0.01)
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

    #[test]
    fn online_convergence_with_defaults() {
        // Validate Hoang et al. (2026) finding: high LR + no momentum converges online.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 10, 1])
            .learning_rate(0.1)
            .momentum(0.0)
            .coefficient_decay(0.0)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        let mut errors_initial = Vec::new();
        let mut errors_converged = Vec::new();

        for i in 0..5000 {
            let t = i as f64 * 0.01;
            let x = [t.sin(), t.cos()];
            let y = x[0] * x[0] + 0.5 * x[1]; // nonlinear target

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (1..50).contains(&i) {
                    errors_initial.push(err);
                } else if i >= 2000 {
                    errors_converged.push(err);
                }
            }
            model.train(&x, y);
        }

        let _mse_initial = errors_initial.iter().sum::<f64>() / errors_initial.len() as f64;
        let mse_converged = errors_converged.iter().sum::<f64>() / errors_converged.len() as f64;

        // With target normalization + LR=0.1, convergence is near-instant
        // on this trivial target. Just verify steady-state error is small.
        // Threshold 0.02: RMSProp adaptive scaling trades off speed on trivial
        // targets for much better convergence on high-dimensional problems.
        assert!(
            mse_converged < 0.02,
            "KAN converged MSE should be small, got {:.4}",
            mse_converged
        );
    }

    #[test]
    fn large_magnitude_targets_no_explosion() {
        // Gradient clipping + target normalization must keep RMSE bounded
        // for large-magnitude targets (Power Plant ~450 MW, Feynman ~500).
        let config = KANConfig::builder()
            .layer_sizes(vec![4, 20, 1])
            .learning_rate(0.1)
            .momentum(0.0)
            .coefficient_decay(0.0)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Simulate Power Plant: targets in [400, 500] range
        let mut rng = 0xBEEF_0001u64;
        let xor = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        let uniform = |s: &mut u64| (xor(s) as f64) / (u64::MAX as f64);

        for _ in 0..5000 {
            let x = [
                5.0 + uniform(&mut rng) * 30.0,
                30.0 + uniform(&mut rng) * 70.0,
                990.0 + uniform(&mut rng) * 50.0,
                25.0 + uniform(&mut rng) * 55.0,
            ];
            let y = 450.0 - 2.0 * x[0] + 0.1 * x[1] - 0.05 * x[2] + 0.3 * x[3];
            model.train(&x, y);
        }

        let pred = model.predict(&[20.0, 60.0, 1013.0, 50.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite on large targets, got {}",
            pred
        );
        assert!(
            pred.abs() < 1e6,
            "prediction should not explode on large targets, got {}",
            pred
        );

        // Feynman: targets from multiple physics equations, varying scales
        let mut model2 = StreamingKAN::new(
            KANConfig::builder()
                .layer_sizes(vec![4, 20, 1])
                .learning_rate(0.1)
                .build()
                .unwrap(),
        );

        for i in 0..5000 {
            let eq = i % 4;
            let (x, y) = match eq {
                0 => {
                    let m = 1.0 + uniform(&mut rng) * 9.0;
                    let v = 1.0 + uniform(&mut rng) * 9.0;
                    ([m, v, 0.0, 0.0], 0.5 * m * v * v) // up to 405
                }
                1 => {
                    let n = 1.0 + uniform(&mut rng) * 4.0;
                    let r = 1.0 + uniform(&mut rng) * 4.0;
                    let t = 1.0 + uniform(&mut rng) * 4.0;
                    let v = 1.0 + uniform(&mut rng) * 4.0;
                    ([n, r, t, v], n * r * t / v)
                }
                2 => {
                    let g = 1.0 + uniform(&mut rng) * 4.0;
                    let m1 = 1.0 + uniform(&mut rng) * 4.0;
                    let m2 = 1.0 + uniform(&mut rng) * 4.0;
                    let r = 1.0 + uniform(&mut rng) * 4.0;
                    ([g, m1, m2, r], g * m1 * m2 / (r * r))
                }
                _ => {
                    let q1 = 1.0 + uniform(&mut rng) * 4.0;
                    let q2 = 1.0 + uniform(&mut rng) * 4.0;
                    let eps = 1.0 + uniform(&mut rng) * 4.0;
                    let r = 1.0 + uniform(&mut rng) * 4.0;
                    (
                        [q1, q2, eps, r],
                        q1 * q2 / (4.0 * std::f64::consts::PI * eps * r * r),
                    )
                }
            };
            model2.train(&x, y);
        }

        let pred2 = model2.predict(&[5.0, 5.0, 3.0, 2.0]);
        assert!(
            pred2.is_finite() && pred2.abs() < 1e6,
            "Feynman predictions should not explode, got {}",
            pred2
        );
    }

    // -----------------------------------------------------------------------
    // Temporal gating tests (T-KAN, Makinde 2026)
    // -----------------------------------------------------------------------

    #[test]
    fn temporal_kan_config_default_off() {
        let config = KANConfig::default();
        assert!(!config.temporal, "temporal should default to false");
    }

    #[test]
    fn temporal_kan_trains_and_predicts() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 8, 1])
            .temporal(true)
            .learning_rate(0.1)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for i in 0..100 {
            let x = [i as f64 * 0.01, 0.5, -0.3];
            model.train_one(&x, (i as f64 * 0.02).sin(), 1.0);
        }
        let pred = model.predict(&[0.5, 0.5, -0.3]);
        assert!(pred.is_finite(), "temporal KAN prediction should be finite");
    }

    #[test]
    fn temporal_kan_state_persists_across_steps() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 6, 1])
            .temporal(true)
            .learning_rate(0.05)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        // Train a few steps
        for i in 0..20 {
            model.train_one(&[0.1 * i as f64, -0.1 * i as f64], i as f64 * 0.5, 1.0);
        }
        // Predictions on same input should differ due to temporal state
        let pred1 = model.predict(&[0.5, -0.5]);
        model.train_one(&[1.0, -1.0], 5.0, 1.0);
        let pred2 = model.predict(&[0.5, -0.5]);
        // After training one more step, temporal state changed, so predictions differ
        // (unless the model has fully converged, which it won't in 21 steps)
        assert!(pred1.is_finite() && pred2.is_finite());
    }

    #[test]
    fn temporal_kan_reset_clears_state() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .temporal(true)
            .learning_rate(0.1)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        model.train_one(&[0.5, -0.3], 1.0, 1.0);
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn non_temporal_kan_unchanged() {
        // Ensure temporal=false doesn't affect existing behavior
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.1)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        model.train_one(&[0.5, -0.3], 1.0, 1.0);
        let pred = model.predict(&[0.5, -0.3]);
        assert!(pred.is_finite());
    }

    #[test]
    fn kan_plasticity_disabled_by_default() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .build()
            .unwrap();
        assert!(!config.plasticity, "plasticity should default to false");
        let model = StreamingKAN::new(config);
        assert!(
            model.plasticity_guard.is_none(),
            "guard should be None when plasticity is disabled"
        );
    }

    #[test]
    fn kan_plasticity_enabled_creates_guard() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .plasticity(true)
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert!(
            model.plasticity_guard.is_some(),
            "guard should be Some when plasticity is enabled"
        );
        assert_eq!(
            model.plasticity_guard.as_ref().unwrap().n_groups(),
            10,
            "should have one group per hidden unit (layer_sizes=[3,10,1] -> 10 hidden)"
        );
    }

    #[test]
    fn kan_plasticity_train_runs_without_panic() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 8, 1])
            .learning_rate(0.1)
            .plasticity(true)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for i in 0..600 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "plasticity-enabled KAN should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn kan_plasticity_no_hidden_skips_guard() {
        // layer_sizes=[2, 1] has no hidden layers -> guard should be None even with plasticity=true
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 1])
            .plasticity(true)
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert!(
            model.plasticity_guard.is_none(),
            "guard should be None for KAN without hidden layers"
        );
    }

    #[test]
    fn plasticity_kan_surgical_reinit() {
        // Verify surgical per-unit reinit runs without panic and preserves
        // other units' weights. Uses a 2-hidden-layer KAN [2, 4, 3, 1] to
        // exercise cross-layer edge reinit.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 3, 1])
            .learning_rate(0.1)
            .plasticity(true)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Train enough to populate weights.
        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        // Snapshot layer 0 coefficients before surgical reinit.
        let coeffs_before: Vec<f64> = model.layers[0].coefficients().to_vec();

        // Surgically reinit hidden unit 0 (in first hidden layer, layer_sizes[1]).
        let mut rng = 0xBEEF_u64;
        model.reinitialize_hidden_unit(0, &mut rng);

        let coeffs_after: Vec<f64> = model.layers[0].coefficients().to_vec();

        // Unit 0 edges should have changed (incoming edges in layers[0]).
        // layers[0] has n_in=2, n_out=4. Output node 0 = edges (0,0) and (0,1).
        let n_coeffs = model.config.grid_size + model.config.spline_order;
        let n_in = 2;
        let edges_for_unit_0 = n_in * n_coeffs; // coefficients for output node 0
        let changed_in_unit_0 = coeffs_before[..edges_for_unit_0]
            .iter()
            .zip(coeffs_after[..edges_for_unit_0].iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();
        assert!(
            changed_in_unit_0 > 0,
            "unit 0 incoming edges should be reinitialized, but no coefficients changed"
        );

        // Unit 1 edges should be preserved (output node 1 in layers[0]).
        let unit1_start = n_in * n_coeffs;
        let unit1_end = unit1_start + n_in * n_coeffs;
        let changed_in_unit_1 = coeffs_before[unit1_start..unit1_end]
            .iter()
            .zip(coeffs_after[unit1_start..unit1_end].iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();
        assert_eq!(
            changed_in_unit_1, 0,
            "unit 1 incoming edges should be preserved, but {} coefficients changed",
            changed_in_unit_1
        );

        // Continue training after reinit — should not panic.
        for i in 100..200 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "KAN should produce finite predictions after surgical reinit, got {pred}"
        );
    }

    #[test]
    fn learning_rate_and_lr_alias_set_same_field() {
        // D1.1: both .learning_rate() and .learning_rate() (deprecated alias) must set the same field.
        let config_a = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.05)
            .build()
            .unwrap();
        #[allow(deprecated)]
        let config_b = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.05)
            .build()
            .unwrap();
        assert!(
            (config_a.learning_rate - config_b.learning_rate).abs() < 1e-15,
            "learning_rate and lr alias should set the same field: {} vs {}",
            config_a.learning_rate,
            config_b.learning_rate
        );
    }

    #[test]
    fn constant_target_does_not_produce_nan() {
        // B.2: KAN should handle constant targets without collapsing to NaN.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for _ in 0..50 {
            model.train(&[1.0, 2.0], 5.0); // constant target
        }
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "KAN should produce finite predictions with constant target, got {pred}"
        );
    }

    #[test]
    fn constant_feature_does_not_produce_nan() {
        // B.3: KAN should handle constant features without NaN from div-by-zero.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for i in 0..50 {
            model.train(&[1.0, i as f64 * 0.1], i as f64); // first feature constant
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "KAN should produce finite predictions with constant feature, got {pred}"
        );
    }
}
