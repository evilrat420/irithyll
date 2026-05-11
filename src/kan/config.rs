//! Configuration for StreamingKAN.

use crate::common::PlasticityConfig;
use crate::error::ConfigError;

/// Temporal gate mode for StreamingKAN.
///
/// Controls how the model incorporates recurrent memory across time steps.
/// Requires at least one hidden layer (`layer_sizes.len() >= 3`).
/// For single-layer KAN (input→output only), all modes behave as `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GateMode {
    /// No temporal gating (default). Pure feed-forward KAN, no recurrent state.
    #[default]
    None,
    /// Single-gate residual mixer (from Makinde 2026 "T-KAN", simplified).
    ///
    /// A scalar sigmoid gate mixes the last hidden layer output with a
    /// recurrent state: `z_t = g * z_{t-1} + (1-g) * KAN_hidden(x_t)`,
    /// where `g = sigmoid(W_gate · x_t + b_gate)`.
    ///
    /// This is the default temporal mode — cheaper than `LstmFull` and stable
    /// on long streams. The gate gradient is computed via the proper chain rule
    /// through the final KAN layer (not a heuristic).
    ResidualMix,
    /// Full 4-gate LSTM-KAN (Makinde 2026, Eq. 7–12).
    ///
    /// Implements the complete LSTM cell with KAN-gated heads:
    /// - Input gate `i_t = sigmoid(W_i · [x_t, h_{t-1}] + b_i)`
    /// - Forget gate `f_t = sigmoid(W_f · [x_t, h_{t-1}] + b_f)`
    /// - Candidate `g_t = tanh(W_g · [x_t, h_{t-1}] + b_g)`
    /// - Output gate `o_t = sigmoid(W_o · [x_t, h_{t-1}] + b_o)`
    /// - Cell state `c_t = f_t * c_{t-1} + i_t * g_t`
    /// - Hidden state `h_t = o_t * tanh(c_t)`
    ///
    /// Substantially richer memory than `ResidualMix` at ~4× parameter cost.
    /// Uses truncated BPTT with window = 1 (online-compatible).
    LstmFull,
}

/// Configuration for [`StreamingKAN`](super::StreamingKAN).
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
    /// Temporal gate mode (default: `GateMode::None`).
    ///
    /// Controls how the model incorporates recurrent memory. See [`GateMode`]
    /// for the available modes and their trade-offs.
    ///
    /// Requires at least one hidden layer (`layer_sizes.len() >= 3`).
    /// For single-layer KAN (input→output only), all gating is a no-op.
    ///
    /// The old `temporal: bool` field is a deprecated alias:
    /// `temporal: true` → `gate_mode: GateMode::ResidualMix`.
    pub gate_mode: GateMode,
    /// RNG seed (default: 42).
    pub seed: u64,
    /// Optional plasticity configuration for neuron regeneration (default: None).
    ///
    /// When `Some`, tracks per-hidden-unit activation energy and periodically
    /// reinitializes dead B-spline edges to maintain learning capacity over
    /// long streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub plasticity: Option<PlasticityConfig>,
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
            gate_mode: GateMode::None,
            seed: 42,
            plasticity: None,
        }
    }
}

impl std::fmt::Display for KANConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KANConfig(layers={:?}, k={}, g={}, lr={}, momentum={}, decay={}, gate_mode={:?}, seed={})",
            self.layer_sizes,
            self.spline_order,
            self.grid_size,
            self.learning_rate,
            self.momentum,
            self.coefficient_decay,
            self.gate_mode,
            self.seed
        )
    }
}

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

    /// Set the temporal gate mode (default: `GateMode::None`).
    ///
    /// See [`GateMode`] for available modes. `GateMode::ResidualMix` is the
    /// lightweight single-gate option (formerly `temporal: true`).
    /// `GateMode::LstmFull` adds a full 4-gate LSTM-KAN cell.
    pub fn gate_mode(mut self, mode: GateMode) -> Self {
        self.config.gate_mode = mode;
        self
    }

    /// Enable or disable temporal gating (deprecated convenience alias).
    ///
    /// `true` sets `gate_mode = GateMode::ResidualMix`.
    /// `false` sets `gate_mode = GateMode::None`.
    ///
    /// Prefer [`gate_mode`](Self::gate_mode) for new code.
    #[deprecated(
        since = "10.0.0",
        note = "Use `.gate_mode(GateMode::ResidualMix)` or `.gate_mode(GateMode::None)` instead"
    )]
    pub fn temporal(mut self, t: bool) -> Self {
        self.config.gate_mode = if t {
            GateMode::ResidualMix
        } else {
            GateMode::None
        };
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Set the plasticity configuration (default: None = disabled).
    ///
    /// When `Some`, tracks per-hidden-unit activation energy and periodically
    /// reinitializes dead B-spline edges to maintain learning capacity over
    /// long streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub fn plasticity(mut self, p: Option<PlasticityConfig>) -> Self {
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
    /// - `learning_rate` is not > 0
    /// - `momentum` is not in [0, 1)
    /// - `coefficient_decay` is not in [0, 1)
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

        if c.learning_rate <= 0.0 {
            return Err(ConfigError::out_of_range(
                "learning_rate",
                "must be > 0",
                c.learning_rate,
            ));
        }

        if c.momentum < 0.0 || c.momentum >= 1.0 {
            return Err(ConfigError::out_of_range(
                "momentum",
                "must be in [0, 1)",
                c.momentum,
            ));
        }

        if c.coefficient_decay < 0.0 || c.coefficient_decay >= 1.0 {
            return Err(ConfigError::out_of_range(
                "coefficient_decay",
                "must be in [0, 1)",
                c.coefficient_decay,
            ));
        }

        Ok(self.config)
    }
}

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
    fn config_rejects_zero_learning_rate() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .learning_rate(0.0)
            .build();
        assert!(result.is_err(), "learning_rate=0 must be rejected");
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .learning_rate(-0.1)
            .build();
        assert!(result.is_err(), "negative learning_rate must be rejected");
    }

    #[test]
    fn config_rejects_invalid_momentum() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .momentum(1.0)
            .build();
        assert!(result.is_err(), "momentum=1 must be rejected");
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .momentum(-0.1)
            .build();
        assert!(result.is_err(), "negative momentum must be rejected");
    }

    #[test]
    fn config_rejects_invalid_coefficient_decay() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .coefficient_decay(1.0)
            .build();
        assert!(result.is_err(), "coefficient_decay=1 must be rejected");
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .coefficient_decay(-0.1)
            .build();
        assert!(
            result.is_err(),
            "negative coefficient_decay must be rejected"
        );
    }

    #[test]
    fn config_accepts_zero_coefficient_decay() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .coefficient_decay(0.0)
            .build();
        assert!(
            result.is_ok(),
            "coefficient_decay=0 (disabled) should be valid"
        );
    }

    #[test]
    fn config_accepts_zero_momentum() {
        let result = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .momentum(0.0)
            .build();
        assert!(result.is_ok(), "momentum=0 (disabled) should be valid");
    }

    #[test]
    fn plasticity_disabled_by_default() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .build()
            .unwrap();
        assert!(
            config.plasticity.is_none(),
            "plasticity should default to None"
        );
    }

    #[test]
    fn plasticity_enabled_via_config() {
        use crate::common::PlasticityConfig;
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 5, 1])
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        assert!(config.plasticity.is_some());
    }
}
