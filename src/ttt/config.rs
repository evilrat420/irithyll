//! TTT configuration and builder.

use crate::common::PlasticityConfig;
use crate::error::ConfigError;

/// Configuration for [`crate::ttt::StreamingTTT`].
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::ttt::TTTConfig;
///
/// let config = TTTConfig::builder()
///     .d_model(32)
///     .learning_rate(0.1)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TTTConfig {
    /// Model dimension (output of TTT layer).
    pub d_model: usize,
    /// Inner learning rate for fast weight updates (default: 0.1).
    pub learning_rate: f64,
    /// Weight decay / forgetting factor, Titans-style (default: 0.001).
    ///
    /// At 0.001, fast weights lose 50% of old information in ~693 steps —
    /// enough retention for regimes of ~500-2000 samples while still
    /// allowing gradual forgetting of stale structure.
    pub alpha: f64,
    /// Momentum coefficient, Titans-style (default: 0.0 = none).
    pub momentum: f64,
    /// RLS forgetting factor for readout (default: 0.998).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0).
    pub delta_rls: f64,
    /// Mini-batch size for fast weight updates (default: 1).
    ///
    /// Default is 1 (single-sample streaming) for maximum responsiveness to
    /// regime changes. Fast weights update every sample, giving immediate
    /// adaptation at regime boundaries. Set to higher values (e.g. 16) if
    /// gradient noise is a concern on very stable data.
    pub batch_size: usize,
    /// Warmup samples before RLS training starts (default: 10).
    pub warmup: usize,
    /// Use Nesterov-accelerated momentum (default: false).
    ///
    /// Only activates when `momentum > 0.0`. When enabled, the momentum
    /// update computes gradients at a look-ahead position for faster
    /// convergence (Behrouz et al., 2025 — Titans).
    pub nesterov: bool,
    /// Steps to linearly ramp alpha (weight decay) from 0 (default: 0 = disabled).
    ///
    /// When > 0, `effective_alpha = alpha * min(1.0, step / alpha_warmup)`.
    /// Prevents weight decay from fighting learning during early adaptation.
    pub alpha_warmup: usize,
    /// Enable surprise-gated learning rate modulation (default: false).
    ///
    /// When enabled, the inner learning rate eta is scaled by prediction
    /// surprise: routine predictions get lower LR, surprising inputs get
    /// full LR. Applied after uncertainty modulation.
    pub surprise_gated: bool,
    /// RNG seed (default: 42).
    pub seed: u64,
    /// Optional plasticity configuration for neuron regeneration (default: None).
    ///
    /// When `Some`, tracks per-hidden-unit TTT output energy and periodically
    /// reinitializes dead units to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub plasticity: Option<PlasticityConfig>,
    /// Depth of the fast-weight memory MLP (default: 1 = linear, i.e. standard TTT-Linear).
    ///
    /// - `1` (default): single-layer linear fast weights. Liu et al. (2026) showed this is
    ///   equivalent to online linear regression (RLS) when projections are frozen.
    /// - `2`: two-layer MLP memory module `z = W2·σ(W1·k)` with GELU activations.
    ///   Behrouz et al. (2025, Titans §3.4) proved depth L_M ≥ 2 escapes the RLS
    ///   degeneracy and supports strictly richer function classes.
    ///
    /// Values > 2 are not currently supported and will be clamped to 2 at construction.
    /// The hidden dimension is set to `d_model` (matched to the state dimension, per §3.4).
    pub deep_memory_layers: usize,
}

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            learning_rate: 0.1,
            alpha: 0.001,
            momentum: 0.0,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            batch_size: 1,
            warmup: 10,
            nesterov: false,
            alpha_warmup: 0,
            surprise_gated: false,
            seed: 42,
            plasticity: None,
            deep_memory_layers: 1,
        }
    }
}

impl std::fmt::Display for TTTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TTTConfig(d_model={}, learning_rate={}, alpha={}, momentum={}, ff={}, batch_size={}, warmup={}, nesterov={}, alpha_warmup={}, surprise_gated={}, deep_memory_layers={}, seed={})",
            self.d_model,
            self.learning_rate,
            self.alpha,
            self.momentum,
            self.forgetting_factor,
            self.batch_size,
            self.warmup,
            self.nesterov,
            self.alpha_warmup,
            self.surprise_gated,
            self.deep_memory_layers,
            self.seed
        )
    }
}

/// Builder for [`TTTConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::ttt::TTTConfig;
///
/// let config = TTTConfig::builder()
///     .d_model(64)
///     .learning_rate(0.005)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_model, 64);
/// ```
pub struct TTTConfigBuilder {
    config: TTTConfig,
}

impl TTTConfig {
    /// Create a new builder with default values.
    pub fn builder() -> TTTConfigBuilder {
        TTTConfigBuilder {
            config: TTTConfig::default(),
        }
    }
}

impl TTTConfigBuilder {
    /// Set the model dimension (default: 32).
    pub fn d_model(mut self, d: usize) -> Self {
        self.config.d_model = d;
        self
    }

    /// Set the inner learning rate for fast weight updates (default: 0.1).
    pub fn learning_rate(mut self, e: f64) -> Self {
        self.config.learning_rate = e;
        self
    }

    /// Set the weight decay coefficient, Titans-style (default: 0.001).
    pub fn alpha(mut self, a: f64) -> Self {
        self.config.alpha = a;
        self
    }

    /// Set the momentum coefficient, Titans-style (default: 0.0 = none).
    pub fn momentum(mut self, m: f64) -> Self {
        self.config.momentum = m;
        self
    }

    /// Set the RLS forgetting factor for the readout (default: 0.998).
    pub fn forgetting_factor(mut self, f: f64) -> Self {
        self.config.forgetting_factor = f;
        self
    }

    /// Set the initial P matrix diagonal for RLS (default: 100.0).
    pub fn delta_rls(mut self, d: f64) -> Self {
        self.config.delta_rls = d;
        self
    }

    /// Set the mini-batch size for fast weight updates (default: 1).
    ///
    /// Default is 1 for single-sample streaming (maximum adaptation speed).
    /// Set higher (e.g. 16) for smoother gradients on very stable data.
    pub fn batch_size(mut self, b: usize) -> Self {
        self.config.batch_size = b;
        self
    }

    /// Set the warmup period in samples (default: 10).
    pub fn warmup(mut self, w: usize) -> Self {
        self.config.warmup = w;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Enable Nesterov-accelerated momentum (default: false).
    ///
    /// Only activates when `momentum > 0.0`. When enabled, gradients are
    /// computed at a look-ahead position for faster convergence.
    pub fn nesterov(mut self, n: bool) -> Self {
        self.config.nesterov = n;
        self
    }

    /// Set the alpha (weight decay) warmup steps (default: 0 = disabled).
    ///
    /// When > 0, alpha linearly ramps from 0 over this many steps.
    pub fn alpha_warmup(mut self, w: usize) -> Self {
        self.config.alpha_warmup = w;
        self
    }

    /// Enable surprise-gated learning rate modulation (default: false).
    ///
    /// Scales inner LR by prediction surprise: routine predictions get
    /// lower LR, surprising inputs get full LR.
    pub fn surprise_gated(mut self, sg: bool) -> Self {
        self.config.surprise_gated = sg;
        self
    }

    /// Set the plasticity configuration (default: None = disabled).
    ///
    /// When `Some`, tracks per-hidden-unit TTT output energy and periodically
    /// reinitializes dead units to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub fn plasticity(mut self, p: Option<PlasticityConfig>) -> Self {
        self.config.plasticity = p;
        self
    }

    /// Set the fast-weight memory depth (default: 1 = linear / TTT-Linear).
    ///
    /// - `1`: standard TTT-Linear (single-layer, equivalent to online RLS when
    ///   projections are frozen; Liu et al. 2026).
    /// - `2`: 2-layer MLP memory `W2·GELU(W1·k)` with hidden dim equal to
    ///   `d_model` (Behrouz et al. 2025, Titans §3.4). Strictly more expressive.
    ///
    /// Values > 2 are clamped to 2 at construction time.
    pub fn deep_memory_layers(mut self, layers: usize) -> Self {
        self.config.deep_memory_layers = layers;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `d_model` is 0
    /// - `batch_size` is 0
    /// - `learning_rate` is not > 0
    /// - `alpha` (weight decay) is not >= 0
    /// - `momentum` is not in [0, 1)
    /// - `forgetting_factor` is not in (0, 1]
    /// - `delta_rls` is not > 0
    pub fn build(self) -> Result<TTTConfig, ConfigError> {
        let c = &self.config;
        if c.d_model == 0 {
            return Err(ConfigError::out_of_range(
                "d_model",
                "must be > 0",
                c.d_model,
            ));
        }
        if c.batch_size == 0 {
            return Err(ConfigError::out_of_range(
                "batch_size",
                "must be > 0",
                c.batch_size,
            ));
        }
        if c.learning_rate <= 0.0 {
            return Err(ConfigError::out_of_range(
                "learning_rate",
                "must be > 0",
                c.learning_rate,
            ));
        }
        if c.alpha < 0.0 {
            return Err(ConfigError::out_of_range("alpha", "must be >= 0", c.alpha));
        }
        if c.momentum < 0.0 || c.momentum >= 1.0 {
            return Err(ConfigError::out_of_range(
                "momentum",
                "must be in [0, 1)",
                c.momentum,
            ));
        }
        if c.forgetting_factor <= 0.0 || c.forgetting_factor > 1.0 {
            return Err(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                c.forgetting_factor,
            ));
        }
        if c.delta_rls <= 0.0 {
            return Err(ConfigError::out_of_range(
                "delta_rls",
                "must be > 0",
                c.delta_rls,
            ));
        }
        if c.deep_memory_layers == 0 {
            return Err(ConfigError::out_of_range(
                "deep_memory_layers",
                "must be >= 1 (1 = linear, 2 = MLP; values > 2 are clamped to 2)",
                c.deep_memory_layers,
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
        let config = TTTConfig::builder().build().unwrap();
        assert_eq!(config.d_model, 32);
    }

    #[test]
    fn config_builder_custom() {
        let config = TTTConfig::builder()
            .d_model(64)
            .learning_rate(0.005)
            .alpha(0.001)
            .momentum(0.9)
            .warmup(20)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(config.d_model, 64);
    }

    #[test]
    fn config_rejects_invalid() {
        assert!(TTTConfig::builder().d_model(0).build().is_err());
        assert!(TTTConfig::builder().batch_size(0).build().is_err());
    }

    #[test]
    fn config_display() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        let s = format!("{config}");
        assert!(s.contains("d_model=16"), "display should contain d_model");
        assert!(
            s.contains("learning_rate="),
            "display should contain learning_rate"
        );
    }

    #[test]
    fn config_clone() {
        let config = TTTConfig::builder().d_model(64).seed(99).build().unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.d_model, config.d_model);
        assert_eq!(cloned.seed, config.seed);
    }

    #[test]
    fn config_rejects_zero_batch_size() {
        assert!(
            TTTConfig::builder().batch_size(0).build().is_err(),
            "batch_size=0 should be rejected"
        );
    }

    #[test]
    fn config_batch_size_default() {
        let config = TTTConfig::builder().build().unwrap();
        assert_eq!(config.batch_size, 1, "default batch_size should be 1");
    }

    #[test]
    fn config_display_includes_batch_size() {
        let config = TTTConfig::builder().batch_size(8).build().unwrap();
        let s = format!("{config}");
        assert!(
            s.contains("batch_size=8"),
            "display should contain batch_size, got: {s}"
        );
    }

    #[test]
    fn ttt_nesterov_config() {
        let config = TTTConfig::builder()
            .d_model(16)
            .nesterov(true)
            .momentum(0.9)
            .build()
            .unwrap();
        assert!(config.nesterov);
        assert!((config.momentum - 0.9).abs() < 1e-10);
    }

    #[test]
    fn ttt_alpha_warmup() {
        let config = TTTConfig::builder()
            .d_model(16)
            .alpha(0.01)
            .alpha_warmup(50)
            .learning_rate(0.05)
            .build()
            .unwrap();
        assert_eq!(config.alpha_warmup, 50);
    }

    #[test]
    fn ttt_surprise_gated_config() {
        let config = TTTConfig::builder()
            .d_model(16)
            .surprise_gated(true)
            .build()
            .unwrap();
        assert!(config.surprise_gated);
    }

    #[test]
    fn ttt_defaults_unchanged() {
        let config = TTTConfig::default();
        assert!(!config.nesterov);
        assert_eq!(config.alpha_warmup, 0);
        assert!(!config.surprise_gated);
    }

    #[test]
    fn ttt_plasticity_disabled_by_default() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        assert!(
            config.plasticity.is_none(),
            "plasticity should default to None"
        );
    }

    #[test]
    fn ttt_plasticity_enabled() {
        use crate::common::PlasticityConfig;
        let config = TTTConfig::builder()
            .d_model(16)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        assert!(config.plasticity.is_some());
    }

    #[test]
    fn ttt_rejects_zero_learning_rate() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .learning_rate(0.0)
                .build()
                .is_err(),
            "learning_rate=0 must be rejected"
        );
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .learning_rate(-0.1)
                .build()
                .is_err(),
            "negative learning_rate must be rejected"
        );
    }

    #[test]
    fn ttt_rejects_negative_alpha() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .alpha(-0.1)
                .build()
                .is_err(),
            "negative alpha must be rejected"
        );
    }

    #[test]
    fn ttt_rejects_invalid_momentum() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .momentum(1.0)
                .build()
                .is_err(),
            "momentum=1 must be rejected"
        );
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .momentum(-0.1)
                .build()
                .is_err(),
            "negative momentum must be rejected"
        );
    }

    #[test]
    fn ttt_rejects_invalid_forgetting_factor() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .forgetting_factor(0.0)
                .build()
                .is_err(),
            "forgetting_factor=0 must be rejected"
        );
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .forgetting_factor(1.01)
                .build()
                .is_err(),
            "forgetting_factor>1 must be rejected"
        );
    }

    #[test]
    fn ttt_rejects_invalid_delta_rls() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .delta_rls(0.0)
                .build()
                .is_err(),
            "delta_rls=0 must be rejected"
        );
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .delta_rls(-1.0)
                .build()
                .is_err(),
            "delta_rls<0 must be rejected"
        );
    }

    #[test]
    fn deep_memory_layers_default_is_one() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        assert_eq!(
            config.deep_memory_layers, 1,
            "deep_memory_layers should default to 1 (linear)"
        );
    }

    #[test]
    fn deep_memory_layers_two_accepted() {
        let config = TTTConfig::builder()
            .d_model(16)
            .deep_memory_layers(2)
            .build()
            .unwrap();
        assert_eq!(config.deep_memory_layers, 2);
    }

    #[test]
    fn deep_memory_layers_zero_rejected() {
        assert!(
            TTTConfig::builder()
                .d_model(16)
                .deep_memory_layers(0)
                .build()
                .is_err(),
            "deep_memory_layers=0 must be rejected"
        );
    }

    #[test]
    fn config_display_includes_deep_memory_layers() {
        let config = TTTConfig::builder()
            .d_model(16)
            .deep_memory_layers(2)
            .build()
            .unwrap();
        let s = format!("{config}");
        assert!(
            s.contains("deep_memory_layers=2"),
            "display should contain deep_memory_layers=2, got: {s}"
        );
    }
}
