//! Configuration and builder for [`StreamingMamba`](super::StreamingMamba).
//!
//! [`MambaConfig`] holds all hyperparameters for the streaming Mamba model.
//! Use [`MambaConfigBuilder`] (via [`MambaConfig::builder()`]) for validated
//! construction with sensible defaults.
//!
//! # Defaults
//!
//! | Parameter | Default | Description |
//! |-----------|---------|-------------|
//! | `d_in` | (required) | Input feature dimension |
//! | `n_state` | 32 | Hidden state dimension per channel |
//! | `forgetting_factor` | 0.998 | RLS exponential forgetting |
//! | `delta_rls` | 100.0 | Initial P matrix diagonal for RLS |
//! | `seed` | 42 | PRNG seed for SSM weight initialization |
//! | `warmup` | 10 | Samples before RLS predictions are trusted |

use std::fmt;

use crate::common::PlasticityConfig;
use crate::error::ConfigError;

/// Mamba architecture version.
///
/// ## Version progression
///
/// | Variant | Discretization | State type | Input mixing | Paper |
/// |---|---|---|---|---|
/// | `V1` | ZOH | Real scalar | Per-channel | Gu & Dao, 2023 |
/// | `V3` | Tustin (bilinear) | Complex diagonal | Grouped avg (MIMO-lite) | Mamba-3 precursor |
/// | `V3Exp` | Exp-trapezoidal (3-term, λ_t) | Complex diagonal | Grouped avg | Lahoti et al., ICLR 2026 |
/// | `V3Mimo` | Exp-trapezoidal (3-term, λ_t) | Complex matrix H ∈ R^{N×P} | True rank-R outer product | Lahoti et al., ICLR 2026 §3.3 |
/// | `BlockDiagonal` | Euler | Real block-diagonal | Dense per-block | Dubinin et al., 2026 |
///
/// ## Deprecation path
///
/// `V3` (Tustin MIMO-lite) is preserved for backward compatibility.
/// New code should prefer `V3Exp` (paper-spec discretization) or
/// `V3Mimo { rank }` (paper-spec MIMO). `V3` will be marked deprecated in v10.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MambaVersion {
    /// Mamba-1: per-channel scalar processing, real states, ZOH discretization.
    V1,
    /// Mamba-3 MIMO-lite: grouped channels, complex states, Tustin (bilinear) discretization.
    ///
    /// This is the **original** irithyll V3 cell — preserved for backward compatibility.
    /// It is **not** paper-spec Mamba-3: it uses Tustin bilinear (2-term) rather than
    /// exponential-trapezoidal (3-term) and averages group input rather than true MIMO.
    ///
    /// Prefer `V3Exp` or `V3Mimo` for new code.
    V3,
    /// Mamba-3 paper-spec: exp-trapezoidal 3-term recurrence + data-dependent λ_t.
    ///
    /// Implements §3.1 of Lahoti et al. (arXiv:2603.15569, ICLR 2026).
    /// Uses `exp_trapezoidal_complex` discretization (distinct from Tustin):
    ///
    /// ```text
    /// h_t = α_t · h_{t-1} + β_t · B_{t-1}·x_{t-1} + γ_t · B_t·x_t
    /// λ_t = sigmoid(W_λ · x_t + b_λ)   // data-dependent convex weight
    /// ```
    ///
    /// Optional BCNorm (§3.2) enabled when `use_bcnorm: true`.
    /// Stability: |α| = exp(Δ·Re(A)) < 1 for Re(A)<0, any positive Δ.
    V3Exp {
        /// Enable BCNorm on B_t and C_t projections (Lahoti et al. §3.2).
        ///
        /// Recommended for large-magnitude inputs. Paper default: enabled.
        use_bcnorm: bool,
    },
    /// Mamba-3 true rank-R MIMO: matrix-valued state H ∈ R^{N×P}.
    ///
    /// Implements §3.3 of Lahoti et al. (arXiv:2603.15569, ICLR 2026).
    /// Unlike `V3` (which averages group input), this variant maintains a
    /// per-channel matrix state updated via rank-R outer product:
    ///
    /// ```text
    /// H_t = α_t · H_{t-1} + β_t · prev_BX + γ_t · B_t · x_t^T
    /// y_t = C_t^T · H_t   (per-channel output, shape P)
    /// ```
    ///
    /// Rank R is an inference efficiency parameter (§3.3): R=1 is standard
    /// outer product; R=2/4 increases arithmetic intensity for memory-bound
    /// hardware. For CPU streaming, R=1 is recommended.
    ///
    /// Also uses exp-trapezoidal 3-term discretization and optional BCNorm.
    V3Mimo {
        /// MIMO rank (1 = outer product, 2/4 for richer mixing, paper typical: 1-4).
        rank: usize,
        /// Enable BCNorm on B_t and C_t projections.
        use_bcnorm: bool,
    },
    /// BD-LRU: Block-diagonal linear recurrence with dense m×m blocks.
    ///
    /// Groups `d_in` channels into `d_in / block_size` blocks, each with a
    /// dense `block_size × block_size` A matrix for cross-channel state mixing
    /// within each block. Row-wise L1 normalization ensures stability.
    ///
    /// Based on Dubinin, Orvieto & Effenberger (2026).
    BlockDiagonal {
        /// Size of each block (must divide `d_in`, typically 2-8).
        block_size: usize,
    },
}

/// Configuration for a [`StreamingMamba`](super::StreamingMamba) model.
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::ssm::MambaConfig;
///
/// let config = MambaConfig::builder()
///     .d_in(8)
///     .n_state(16)
///     .forgetting_factor(0.998)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Input/output feature dimension (required, >= 1).
    pub d_in: usize,
    /// Hidden state dimension per channel (default: 32, >= 1).
    pub n_state: usize,
    /// RLS forgetting factor (default: 0.998, in (0, 1]).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0, > 0).
    pub delta_rls: f64,
    /// Random seed for SSM weight initialization (default: 42).
    pub seed: u64,
    /// Number of warmup samples before predictions are trusted (default: 10, >= 0).
    pub warmup: usize,
    /// Mamba architecture version (default: V1).
    pub version: MambaVersion,
    /// Number of MIMO groups (default: 1, used for V3/V3Exp/V3Mimo).
    ///
    /// When `version` is V3/V3Exp/V3Mimo and `n_groups` was set to 0 at build
    /// time, it is auto-derived as `d_in / 4` clamped to `[1, d_in]`.
    /// When `version` is V1 or BlockDiagonal, this field is ignored.
    pub n_groups: usize,
    /// Block size for BlockDiagonal version (default: 4, only used for BlockDiagonal).
    ///
    /// Must divide `d_in` evenly. Typical values: 2, 4, 8.
    /// When `version` is V1/V3/V3Exp/V3Mimo, this field is ignored (stored as 1).
    pub block_size: usize,
    /// Optional plasticity configuration for neuron regeneration (default: None).
    ///
    /// When `Some`, tracks per-channel SSM state energy and periodically
    /// reinitializes dead channels to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub plasticity: Option<PlasticityConfig>,
}

impl MambaConfig {
    /// Create a new builder with default values.
    ///
    /// Only `d_in` is required; all other parameters have sensible defaults.
    pub fn builder() -> MambaConfigBuilder {
        MambaConfigBuilder::default()
    }
}

impl fmt::Display for MambaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            MambaVersion::V1 => write!(
                f,
                "MambaConfig(v1, d_in={}, n_state={}, ff={}, delta={}, seed={}, warmup={})",
                self.d_in, self.n_state, self.forgetting_factor, self.delta_rls, self.seed,
                self.warmup
            ),
            MambaVersion::V3 => write!(
                f,
                "MambaConfig(v3, d_in={}, n_state={}, n_groups={}, ff={}, delta={}, seed={}, warmup={})",
                self.d_in, self.n_state, self.n_groups, self.forgetting_factor, self.delta_rls,
                self.seed, self.warmup
            ),
            MambaVersion::V3Exp { use_bcnorm } => write!(
                f,
                "MambaConfig(v3exp, d_in={}, n_state={}, n_groups={}, bcnorm={}, ff={}, delta={}, seed={}, warmup={})",
                self.d_in, self.n_state, self.n_groups, use_bcnorm,
                self.forgetting_factor, self.delta_rls, self.seed, self.warmup
            ),
            MambaVersion::V3Mimo { rank, use_bcnorm } => write!(
                f,
                "MambaConfig(v3mimo, d_in={}, n_state={}, n_groups={}, rank={}, bcnorm={}, ff={}, delta={}, seed={}, warmup={})",
                self.d_in, self.n_state, self.n_groups, rank, use_bcnorm,
                self.forgetting_factor, self.delta_rls, self.seed, self.warmup
            ),
            MambaVersion::BlockDiagonal { block_size } => write!(
                f,
                "MambaConfig(bd, d_in={}, n_state={}, block_size={}, ff={}, delta={}, seed={}, warmup={})",
                self.d_in, self.n_state, block_size, self.forgetting_factor, self.delta_rls,
                self.seed, self.warmup
            ),
        }
    }
}

/// Builder for [`MambaConfig`] with validation.
///
/// # Required Parameters
///
/// - `d_in` -- must be set before calling `build()`
///
/// # Example
///
/// ```
/// use irithyll::ssm::MambaConfig;
///
/// let config = MambaConfig::builder()
///     .d_in(4)
///     .n_state(32)
///     .seed(123)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_in, 4);
/// assert_eq!(config.n_state, 32);
/// ```
#[derive(Debug)]
pub struct MambaConfigBuilder {
    d_in: Option<usize>,
    n_state: usize,
    forgetting_factor: f64,
    delta_rls: f64,
    seed: u64,
    warmup: usize,
    version: MambaVersion,
    n_groups: usize,
    block_size: usize,
    rank: usize,
    plasticity: Option<PlasticityConfig>,
}

impl Default for MambaConfigBuilder {
    fn default() -> Self {
        Self {
            d_in: None,
            n_state: 32,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            seed: 42,
            warmup: 10,
            version: MambaVersion::V1,
            n_groups: 1,
            block_size: 4,
            rank: 1,
            plasticity: None,
        }
    }
}

impl MambaConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input feature dimension (required, >= 1).
    pub fn d_in(mut self, d_in: usize) -> Self {
        self.d_in = Some(d_in);
        self
    }

    /// Set the hidden state dimension per channel (default: 32, >= 1).
    pub fn n_state(mut self, n_state: usize) -> Self {
        self.n_state = n_state;
        self
    }

    /// Set the RLS forgetting factor (default: 0.998, must be in (0, 1]).
    pub fn forgetting_factor(mut self, ff: f64) -> Self {
        self.forgetting_factor = ff;
        self
    }

    /// Set the initial P matrix diagonal for RLS (default: 100.0, must be > 0).
    pub fn delta_rls(mut self, delta: f64) -> Self {
        self.delta_rls = delta;
        self
    }

    /// Set the random seed for SSM weight initialization (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the warmup period in samples (default: 10).
    pub fn warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set the Mamba architecture version (default: V1).
    pub fn version(mut self, version: MambaVersion) -> Self {
        self.version = version;
        self
    }

    /// Set the number of MIMO groups (default: 1, only used for V3).
    ///
    /// Pass 0 for auto-derivation: `d_in / 4` clamped to `[1, d_in]`.
    pub fn n_groups(mut self, n_groups: usize) -> Self {
        self.n_groups = n_groups;
        self
    }

    /// Set the block size for BlockDiagonal version (default: 4).
    ///
    /// Must divide `d_in` evenly. Typical values: 2, 4, 8.
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set the MIMO rank for `V3Mimo` variant (default: 1).
    ///
    /// - `rank=1`: standard rank-1 outer product `B_t · x_t^T`. Recommended for CPU.
    /// - `rank=2/4`: richer per-channel mixing. Increases parameter count by R.
    ///
    /// Only used when `version` is `V3Mimo { rank, .. }`. Ignored for other variants.
    /// Must be >= 1.
    ///
    /// # References
    ///
    /// Lahoti et al. arXiv:2603.15569, ICLR 2026, §3.3.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set the plasticity configuration (default: None = disabled).
    ///
    /// When `Some`, tracks per-channel SSM state energy and periodically
    /// reinitializes dead channels to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024). Use [`PlasticityConfig::default()`]
    /// for paper-recommended defaults.
    pub fn plasticity(mut self, p: Option<PlasticityConfig>) -> Self {
        self.plasticity = p;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `d_in` was not set or is 0
    /// - `n_state` is 0
    /// - `forgetting_factor` is not in (0, 1]
    /// - `delta_rls` is not positive
    pub fn build(self) -> Result<MambaConfig, ConfigError> {
        let d_in = self.d_in.ok_or_else(|| {
            ConfigError::invalid("d_in", "d_in must be set (input feature dimension)")
        })?;
        if d_in < 1 {
            return Err(ConfigError::out_of_range("d_in", "must be >= 1", d_in));
        }
        if self.n_state < 1 {
            return Err(ConfigError::out_of_range(
                "n_state",
                "must be >= 1",
                self.n_state,
            ));
        }
        if self.forgetting_factor <= 0.0 || self.forgetting_factor > 1.0 {
            return Err(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                self.forgetting_factor,
            ));
        }
        if self.delta_rls <= 0.0 {
            return Err(ConfigError::out_of_range(
                "delta_rls",
                "must be > 0",
                self.delta_rls,
            ));
        }

        // Helper: auto-derive n_groups (shared logic for V3/V3Exp/V3Mimo)
        let derive_n_groups =
            |requested: usize, version_name: &'static str| -> Result<usize, ConfigError> {
                let g = if requested == 0 {
                    let target = (d_in / 4).max(1);
                    (1..=target).rev().find(|&g| d_in % g == 0).unwrap_or(1)
                } else {
                    requested
                };
                if g < 1 {
                    return Err(ConfigError::out_of_range("n_groups", version_name, g));
                }
                if d_in % g != 0 {
                    return Err(ConfigError::invalid(
                        "n_groups",
                        format!(
                            "n_groups ({}) must divide d_in ({}) evenly for {}",
                            g, d_in, version_name
                        ),
                    ));
                }
                Ok(g)
            };

        // Version-specific validation for n_groups and block_size.
        let (n_groups, block_size, version) = match self.version {
            MambaVersion::V1 => {
                // V1 ignores n_groups and block_size; store 1 for consistency.
                (1, 1, MambaVersion::V1)
            }
            MambaVersion::V3 => {
                let g = derive_n_groups(self.n_groups, "V3")?;
                (g, 1, MambaVersion::V3)
            }
            MambaVersion::V3Exp { use_bcnorm } => {
                let g = derive_n_groups(self.n_groups, "V3Exp")?;
                (g, 1, MambaVersion::V3Exp { use_bcnorm })
            }
            MambaVersion::V3Mimo {
                rank: _,
                use_bcnorm,
            } => {
                let g = derive_n_groups(self.n_groups, "V3Mimo")?;
                let r = self.rank;
                if r < 1 {
                    return Err(ConfigError::out_of_range(
                        "rank",
                        "must be >= 1 for V3Mimo (rank=1 is standard outer product)",
                        r,
                    ));
                }
                if r > 16 {
                    return Err(ConfigError::out_of_range(
                        "rank",
                        "must be <= 16 for V3Mimo (parameter count scales with rank)",
                        r,
                    ));
                }
                (
                    g,
                    1,
                    MambaVersion::V3Mimo {
                        rank: r,
                        use_bcnorm,
                    },
                )
            }
            MambaVersion::BlockDiagonal { block_size: _ } => {
                let bs = self.block_size;
                if bs < 2 {
                    return Err(ConfigError::out_of_range(
                        "block_size",
                        "must be >= 2 for BlockDiagonal (use V1 for block_size=1)",
                        bs,
                    ));
                }
                if bs > 16 {
                    return Err(ConfigError::out_of_range(
                        "block_size",
                        "must be <= 16 for BlockDiagonal (dense matmul cost is O(m^2))",
                        bs,
                    ));
                }
                if d_in % bs != 0 {
                    return Err(ConfigError::invalid(
                        "block_size",
                        format!(
                            "block_size ({}) must divide d_in ({}) evenly for BlockDiagonal",
                            bs, d_in
                        ),
                    ));
                }
                (1, bs, MambaVersion::BlockDiagonal { block_size: bs })
            }
        };

        Ok(MambaConfig {
            d_in,
            n_state: self.n_state,
            forgetting_factor: self.forgetting_factor,
            delta_rls: self.delta_rls,
            seed: self.seed,
            warmup: self.warmup,
            version,
            n_groups,
            block_size,
            plasticity: self.plasticity.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let config = MambaConfig::builder().d_in(4).build().unwrap();
        assert_eq!(config.d_in, 4);
        assert_eq!(config.n_state, 32);
        assert!((config.forgetting_factor - 0.998).abs() < 1e-12);
        assert!((config.delta_rls - 100.0).abs() < 1e-12);
        assert_eq!(config.seed, 42);
        assert_eq!(config.warmup, 10);
        assert_eq!(
            config.version,
            MambaVersion::V1,
            "default version should be V1"
        );
        assert_eq!(config.n_groups, 1, "V1 n_groups should always be 1");
    }

    #[test]
    fn builder_custom_values() {
        let config = MambaConfig::builder()
            .d_in(8)
            .n_state(32)
            .forgetting_factor(0.99)
            .delta_rls(50.0)
            .seed(123)
            .warmup(5)
            .build()
            .unwrap();
        assert_eq!(config.d_in, 8);
        assert_eq!(config.n_state, 32);
        assert!((config.forgetting_factor - 0.99).abs() < 1e-12);
        assert!((config.delta_rls - 50.0).abs() < 1e-12);
        assert_eq!(config.seed, 123);
        assert_eq!(config.warmup, 5);
    }

    #[test]
    fn builder_missing_d_in() {
        let result = MambaConfig::builder().build();
        assert!(result.is_err(), "should fail without d_in");
    }

    #[test]
    fn builder_invalid_n_state() {
        let result = MambaConfig::builder().d_in(4).n_state(0).build();
        assert!(result.is_err(), "n_state=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_zero() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(0.0)
            .build();
        assert!(result.is_err(), "ff=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_negative() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(-0.5)
            .build();
        assert!(result.is_err(), "ff=-0.5 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_over_one() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(1.01)
            .build();
        assert!(result.is_err(), "ff=1.01 should be invalid");
    }

    #[test]
    fn builder_forgetting_factor_one_valid() {
        let config = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(1.0)
            .build()
            .unwrap();
        assert!((config.forgetting_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn builder_invalid_delta_rls() {
        let result = MambaConfig::builder().d_in(4).delta_rls(0.0).build();
        assert!(result.is_err(), "delta_rls=0 should be invalid");
        let result = MambaConfig::builder().d_in(4).delta_rls(-1.0).build();
        assert!(result.is_err(), "delta_rls=-1 should be invalid");
    }

    #[test]
    fn display_format() {
        let config = MambaConfig::builder().d_in(4).build().unwrap();
        let s = format!("{}", config);
        assert!(s.contains("d_in=4"), "display should contain d_in");
        assert!(s.contains("n_state=32"), "display should contain n_state");
    }

    #[test]
    fn config_clone() {
        let config = MambaConfig::builder().d_in(4).seed(99).build().unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.d_in, config.d_in);
        assert_eq!(cloned.seed, config.seed);
    }

    #[test]
    fn mamba_version_default_is_v1() {
        let config = MambaConfig::builder().d_in(8).build().unwrap();
        assert_eq!(
            config.version,
            MambaVersion::V1,
            "default version should be V1"
        );
        assert_eq!(config.n_groups, 1, "V1 should have n_groups=1");
    }

    #[test]
    fn v3_explicit_n_groups() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3)
            .n_groups(2)
            .build()
            .unwrap();
        assert_eq!(config.version, MambaVersion::V3);
        assert_eq!(config.n_groups, 2, "should use explicit n_groups=2");
    }

    #[test]
    fn v3_auto_derive_n_groups() {
        // d_in=16, auto-derive: 16/4 = 4.
        let config = MambaConfig::builder()
            .d_in(16)
            .version(MambaVersion::V3)
            .n_groups(0)
            .build()
            .unwrap();
        assert_eq!(
            config.n_groups, 4,
            "auto-derived n_groups should be d_in/4 = 4"
        );
    }

    #[test]
    fn v3_auto_derive_n_groups_small_d_in() {
        // d_in=2, auto-derive: 2/4 = 0 -> clamped to 1.
        let config = MambaConfig::builder()
            .d_in(2)
            .version(MambaVersion::V3)
            .n_groups(0)
            .build()
            .unwrap();
        assert_eq!(
            config.n_groups, 1,
            "auto-derived n_groups should clamp to 1 for small d_in"
        );
    }

    #[test]
    fn v3_n_groups_must_divide_d_in() {
        let result = MambaConfig::builder()
            .d_in(7)
            .version(MambaVersion::V3)
            .n_groups(3)
            .build();
        assert!(result.is_err(), "n_groups=3 should not divide d_in=7");
    }

    #[test]
    fn v1_ignores_n_groups() {
        // Even if n_groups is set, V1 should ignore it and store 1.
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V1)
            .n_groups(4)
            .build()
            .unwrap();
        assert_eq!(config.n_groups, 1, "V1 should ignore n_groups and store 1");
    }

    #[test]
    fn display_format_v3() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3)
            .n_groups(2)
            .build()
            .unwrap();
        let s = format!("{}", config);
        assert!(s.contains("v3"), "V3 display should contain 'v3'");
        assert!(
            s.contains("n_groups=2"),
            "V3 display should contain n_groups"
        );
    }

    // ---- BlockDiagonal tests ----

    #[test]
    fn bd_basic_config() {
        let config = MambaConfig::builder()
            .d_in(8)
            .n_state(16)
            .version(MambaVersion::BlockDiagonal { block_size: 4 })
            .block_size(4)
            .build()
            .unwrap();
        assert_eq!(
            config.version,
            MambaVersion::BlockDiagonal { block_size: 4 }
        );
        assert_eq!(config.block_size, 4);
        assert_eq!(config.d_in, 8);
    }

    #[test]
    fn bd_block_size_must_divide_d_in() {
        let result = MambaConfig::builder()
            .d_in(7)
            .version(MambaVersion::BlockDiagonal { block_size: 4 })
            .block_size(4)
            .build();
        assert!(result.is_err(), "block_size=4 should not divide d_in=7");
    }

    #[test]
    fn bd_block_size_too_small() {
        let result = MambaConfig::builder()
            .d_in(4)
            .version(MambaVersion::BlockDiagonal { block_size: 1 })
            .block_size(1)
            .build();
        assert!(result.is_err(), "block_size=1 should be invalid (use V1)");
    }

    #[test]
    fn bd_block_size_too_large() {
        let result = MambaConfig::builder()
            .d_in(32)
            .version(MambaVersion::BlockDiagonal { block_size: 32 })
            .block_size(32)
            .build();
        assert!(result.is_err(), "block_size=32 should exceed maximum 16");
    }

    #[test]
    fn bd_display_format() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::BlockDiagonal { block_size: 4 })
            .block_size(4)
            .build()
            .unwrap();
        let s = format!("{}", config);
        assert!(s.contains("bd"), "BD display should contain 'bd'");
        assert!(
            s.contains("block_size=4"),
            "BD display should contain block_size"
        );
    }

    #[test]
    fn bd_various_block_sizes() {
        for bs in [2, 4, 8] {
            let config = MambaConfig::builder()
                .d_in(8)
                .version(MambaVersion::BlockDiagonal { block_size: bs })
                .block_size(bs)
                .build()
                .unwrap();
            assert_eq!(config.block_size, bs, "block_size should be {}", bs);
        }
    }

    // ---- V3Exp tests ----

    #[test]
    fn v3exp_basic_config() {
        let config = MambaConfig::builder()
            .d_in(8)
            .n_state(16)
            .version(MambaVersion::V3Exp { use_bcnorm: false })
            .n_groups(2)
            .build()
            .unwrap();
        assert_eq!(config.version, MambaVersion::V3Exp { use_bcnorm: false });
        assert_eq!(config.n_groups, 2);
        assert_eq!(config.d_in, 8);
    }

    #[test]
    fn v3exp_with_bcnorm() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Exp { use_bcnorm: true })
            .n_groups(2)
            .build()
            .unwrap();
        assert_eq!(config.version, MambaVersion::V3Exp { use_bcnorm: true });
    }

    #[test]
    fn v3exp_auto_derive_n_groups() {
        let config = MambaConfig::builder()
            .d_in(16)
            .version(MambaVersion::V3Exp { use_bcnorm: false })
            .n_groups(0)
            .build()
            .unwrap();
        assert_eq!(config.n_groups, 4, "auto-derived n_groups=d_in/4=4");
    }

    #[test]
    fn v3exp_n_groups_must_divide_d_in() {
        let result = MambaConfig::builder()
            .d_in(7)
            .version(MambaVersion::V3Exp { use_bcnorm: false })
            .n_groups(3)
            .build();
        assert!(
            result.is_err(),
            "n_groups=3 must not divide d_in=7 for V3Exp"
        );
    }

    #[test]
    fn v3exp_display_format() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Exp { use_bcnorm: false })
            .n_groups(2)
            .build()
            .unwrap();
        let s = format!("{}", config);
        assert!(s.contains("v3exp"), "V3Exp display should contain 'v3exp'");
        assert!(
            s.contains("n_groups=2"),
            "V3Exp display should contain n_groups"
        );
        assert!(
            s.contains("bcnorm="),
            "V3Exp display should contain bcnorm flag"
        );
    }

    // ---- V3Mimo tests ----

    #[test]
    fn v3mimo_basic_config() {
        let config = MambaConfig::builder()
            .d_in(8)
            .n_state(16)
            .version(MambaVersion::V3Mimo {
                rank: 1,
                use_bcnorm: false,
            })
            .n_groups(2)
            .rank(1)
            .build()
            .unwrap();
        assert_eq!(
            config.version,
            MambaVersion::V3Mimo {
                rank: 1,
                use_bcnorm: false
            }
        );
        assert_eq!(config.n_groups, 2);
    }

    #[test]
    fn v3mimo_rank_4() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Mimo {
                rank: 4,
                use_bcnorm: true,
            })
            .n_groups(2)
            .rank(4)
            .build()
            .unwrap();
        assert_eq!(
            config.version,
            MambaVersion::V3Mimo {
                rank: 4,
                use_bcnorm: true
            }
        );
    }

    #[test]
    fn v3mimo_rank_too_large() {
        let result = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Mimo {
                rank: 1,
                use_bcnorm: false,
            })
            .rank(32) // > 16
            .n_groups(2)
            .build();
        assert!(result.is_err(), "rank=32 should be invalid (> 16)");
    }

    #[test]
    fn v3mimo_rank_zero_invalid() {
        let result = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Mimo {
                rank: 1,
                use_bcnorm: false,
            })
            .rank(0)
            .n_groups(2)
            .build();
        assert!(result.is_err(), "rank=0 should be invalid");
    }

    #[test]
    fn v3mimo_display_format() {
        let config = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Mimo {
                rank: 2,
                use_bcnorm: false,
            })
            .n_groups(2)
            .rank(2)
            .build()
            .unwrap();
        let s = format!("{}", config);
        assert!(
            s.contains("v3mimo"),
            "V3Mimo display should contain 'v3mimo'"
        );
        assert!(s.contains("rank=2"), "V3Mimo display should contain rank");
        assert!(
            s.contains("n_groups=2"),
            "V3Mimo display should contain n_groups"
        );
    }

    #[test]
    fn v3mimo_readout_dim_smaller_than_v3exp() {
        // V3Mimo uses the simpler `d_in + n_groups` readout (gated output +
        // per-group Frobenius energy). V3Exp surfaces additional features
        // unique to its complex-diagonal cell:
        //   1. Per-component `Re(h)`, `Im(h)`, `|h|`, `|h|^2` — the
        //      load-bearing complex-state surface for parity-class tasks
        //      (the `|h|^2` feature carries up-to-degree-4 in input bits).
        //   2. A tanh random-feature lift over raw input bits — a Random
        //      Feature Network that approximates kernel ridge regression
        //      with Gaussian RBF in the limit (Rahimi & Recht 2008), giving
        //      the linear RLS readout universal-approximation capability
        //      for high-degree polynomial targets like multi-bit XOR parity.
        //
        // V3Mimo readout stays linear because V3Mimo's true rank-R matrix
        // state already provides quadratic cross-channel information
        // through its outer-product update; a separate lift is not the
        // architectural claim being defended.
        //
        // This test asserts the dimensional asymmetry, not equality. If a
        // future change wants V3Mimo to receive the same lift, update the
        // matcher accordingly.
        use crate::StreamingMamba;
        let config_exp = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Exp { use_bcnorm: false })
            .n_groups(2)
            .build()
            .unwrap();
        let config_mimo = MambaConfig::builder()
            .d_in(8)
            .version(MambaVersion::V3Mimo {
                rank: 1,
                use_bcnorm: false,
            })
            .n_groups(2)
            .rank(1)
            .build()
            .unwrap();
        let m_exp = StreamingMamba::new(config_exp);
        let m_mimo = StreamingMamba::new(config_mimo);
        assert!(
            m_exp.last_features().len() > m_mimo.last_features().len(),
            "V3Exp readout dim ({}) should exceed V3Mimo readout dim ({}) — \
             V3Exp surfaces complex-state features and a random tanh lift \
             that V3Mimo does not.",
            m_exp.last_features().len(),
            m_mimo.last_features().len()
        );
        // V3Mimo dim must equal d_in + n_groups (gated + per-group Frobenius).
        assert_eq!(
            m_mimo.last_features().len(),
            10,
            "V3Mimo readout dim should be d_in+n_groups = 10"
        );
    }
}
