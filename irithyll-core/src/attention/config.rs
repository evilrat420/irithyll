//! Configuration types for the unified streaming linear attention engine.
//!
//! [`AttentionMode`] selects which architecture variant to use, while
//! [`AttentionConfig`] holds the full configuration including dimensions,
//! head count, and PRNG seed.

/// Gate dimensionality for GLA-family variants.
///
/// The original GLA paper (Yang et al., 2024) specifies a vector gate
/// `α_t ∈ (0,1)^{d_k}` — one decay scalar per key dimension. The simplified
/// scalar gate (`Scalar`) uses a single sigmoid output shared across all key
/// dimensions, which reduces parameter count at the cost of expressivity.
///
/// `Vector` is the mathematically canonical form from the paper.
/// `Scalar` is retained for backward compatibility with models trained before
/// the vector-gate upgrade.
#[derive(Clone, Debug, PartialEq)]
pub enum GateMode {
    /// Single scalar gate shared across all key dimensions (legacy default).
    Scalar,
    /// Per-key-dimension gate vector `α_t ∈ (0,1)^{d_k}` (paper-canonical).
    Vector,
}

/// Beta parameterization mode for GatedDeltaNet.
///
/// Controls whether the delta-rule mixing strength `β_t` is a static
/// scalar or a data-dependent value computed per token.
///
/// # Paper reference
///
/// Yang et al. ICLR 2025 (arXiv:2412.06464) specifies the canonical form
/// as per-token `β_t = sigmoid(W_β · x_t + b_β)`, making the delta-rule
/// mixing strength data-dependent. The static scalar `β_scale` is a
/// degenerate case retained for backward compatibility.
#[derive(Clone, Debug, PartialEq)]
pub enum GatedDeltaMode {
    /// Static scalar `β_scale` shared across all tokens (legacy default).
    ///
    /// Equivalent to the original `beta_scale: f64` parameter. Back-compatible
    /// with all checkpoints trained before this field was introduced.
    Static,
    /// Per-token `β_t = sigmoid(W_β · x_t)` — paper-canonical form.
    ///
    /// Adds a learned `W_β ∈ R^{d_model}` projection per head. The mixing
    /// strength becomes data-dependent, allowing the model to selectively
    /// apply strong or weak delta-rule updates depending on the input.
    PerToken,
}

/// Selects the attention architecture variant.
///
/// Each variant corresponds to a different parameterization of the unified
/// structured linear attention recurrence `S_t = decay_t * S_{t-1} + update_t`.
///
/// # Variants
///
/// - **RetNet** -- Fixed exponential decay (Sun et al., 2023)
/// - **Hawk** -- Gated scalar recurrence with vector state (De et al., 2024)
/// - **GLA** -- Gated Linear Attention with scalar gate (Yang et al., 2024, legacy)
/// - **GLAVector** -- GLA with per-key-dimension vector gate (paper-canonical form)
/// - **DeltaNet** -- Delta rule error-corrective update (Schlag et al., 2021)
/// - **GatedDeltaNet** -- GLA + delta rule combined (Yang et al., 2024, NVIDIA)
/// - **RWKV** -- Exponential decay with dynamic w (Peng et al., 2024)
/// - **MLSTM** -- xLSTM matrix memory with forget+input gates (Beck et al., 2024)
/// - **DeltaProduct** -- Product of n_h Householder delta rules (Siems et al., NeurIPS 2025)
/// - **RWKV7** -- Vector-gated delta rule with DPLR transitions (Peng et al., 2025)
/// - **HGRN2** -- Lower-bounded gated linear RNN with state expansion (Qin et al., ICML 2024)
/// - **LogLinear** -- O(log T) hierarchical Fenwick state (Han Guo et al., ICLR 2026)
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AttentionMode {
    /// Fixed exponential decay: `S = gamma * S + k * v^T`.
    RetNet {
        /// Decay factor in (0, 1). Typical: 0.9 to 0.999.
        gamma: f64,
    },
    /// Gated scalar recurrence with vector state (Griffin/Hawk).
    Hawk,
    /// Gated Linear Attention with scalar data-dependent sigmoid gate (legacy).
    ///
    /// Uses a single sigmoid scalar `σ(w^T x) ∈ (0,1)` shared across all key
    /// dimensions. Preserved for backward compatibility with pre-v10 checkpoints.
    /// For new code, prefer [`GLAVector`](AttentionMode::GLAVector).
    GLA,
    /// Gated Linear Attention with per-key-dimension vector gate (paper-canonical).
    ///
    /// Implements the gate from Yang et al. 2024 exactly: `α_t ∈ (0,1)^{d_k}`
    /// with one independent sigmoid per key dimension, giving the model
    /// per-dimension memory control. The gate projection is `W_α ∈ R^{d_k × d_model}`.
    ///
    /// This is the recommended GLA variant for new models. Produces different
    /// (and generally richer) dynamics than the scalar-gate `GLA` variant.
    GLAVector,
    /// Delta rule: error-corrective associative memory update.
    DeltaNet,
    /// Gated delta rule: GLA gate + delta error correction (Yang et al., ICLR 2025).
    ///
    /// Combines GLA's data-dependent gating with DeltaNet's error-corrective
    /// delta rule, plus learnable beta scaling and L2-normalized keys.
    ///
    /// # Beta parameterization
    ///
    /// - [`GatedDeltaMode::Static`] (default): `beta_scale` scalar applies uniformly.
    /// - [`GatedDeltaMode::PerToken`]: paper-canonical form; `β_t = sigmoid(W_β · x_t)`
    ///   is computed per token per head via a learned projection. `beta_scale` is
    ///   ignored in this mode.
    GatedDeltaNet {
        /// Learnable beta scaling factor (default: 1.0).
        ///
        /// Used only when `gate_mode_delta` is [`GatedDeltaMode::Static`].
        /// Controls how aggressively the error-corrective update is applied.
        beta_scale: f64,
        /// Beta parameterization mode (default: [`GatedDeltaMode::Static`]).
        ///
        /// Set to [`GatedDeltaMode::PerToken`] for the paper-canonical
        /// data-dependent mixing strength (Yang et al. ICLR 2025).
        gate_mode_delta: GatedDeltaMode,
    },
    /// RWKV-style exponential decay with learned dynamic w.
    RWKV {
        /// Base decay rate before input-dependent modulation.
        initial_decay: f64,
    },
    /// xLSTM matrix memory with separate forget and input gates.
    MLSTM,
    /// Product of n_h Householder delta rules (Siems et al., NeurIPS 2025).
    ///
    /// Applies `n_compositions` sequential delta rule steps per token, each with
    /// its own key, value, and beta. The product of generalized Householder
    /// transformations gives a spectrally bounded (norm ≤ 1) transition matrix.
    /// Includes a scalar forget gate.
    ///
    /// # Reflections flag
    ///
    /// When `reflections` is `false` (default), each `β_{t,j} ∈ (0, 1)` via
    /// a plain sigmoid. When `true`, `β_{t,j} ∈ (0, 2)` via `2 · sigmoid(·)`,
    /// which enables full Householder reflections (negative eigenvalues) for
    /// stronger state-tracking capability (Siems et al. NeurIPS 2025,
    /// arXiv:2502.10297, §4).
    DeltaProduct {
        /// Number of Householder compositions per token (typically 2-4).
        n_compositions: usize,
        /// Enable full Householder reflections (default: `false`).
        ///
        /// When `false`: `β ∈ (0, 1)` — standard delta rule range.
        /// When `true`: `β ∈ (0, 2)` — full reflections, enables negative
        /// eigenvalues for parity / state-tracking tasks.
        reflections: bool,
    },
    /// RWKV-7 vector-gated delta rule with DPLR transitions (Peng et al., 2025).
    ///
    /// Uses per-dimension vector decay, vector in-context learning rate (ICLR),
    /// and decoupled removal/replacement keys. The transition matrix is
    /// diagonal-plus-low-rank (DPLR), enabling state tracking beyond TC^0.
    RWKV7,
    /// HGRN2: gated linear RNN with outer-product state expansion (Qin et al., ICML 2024).
    ///
    /// Uses a lower-bounded forget gate for minimum memory retention.
    /// The state update is outer-product based (like GLA) but with the
    /// bounded gate ensuring the model never completely forgets.
    /// `alpha_t = lower_bound + (1 - lower_bound) * sigmoid(raw_t)` per dimension.
    HGRN2 {
        /// Lower bound for forget gate (default: 0.9, range 0.0..1.0).
        /// Higher values ensure stronger memory retention.
        lower_bound: f64,
    },
    /// Log-Linear Attention (Han Guo et al., ICLR 2026, arXiv:2506.04761).
    ///
    /// Replaces the single fixed-size recurrent state of the inner
    /// linear-attention rule with an O(log T) hierarchy of states
    /// organized by a Fenwick-tree decomposition of the prefix.
    /// Compute per token grows as `log T`; total compute is
    /// `O(T log T)`. State memory is fixed at
    /// `max_levels * d_k * d_v * n_heads`.
    ///
    /// # Inner mode (paper §3.2)
    ///
    /// Wraps any non-recursive `AttentionMode` as the leaf update
    /// rule. Recommended: `GatedDeltaNet` for strongest associative
    /// recall, `GLA` for stability. The wrapper handles per-token
    /// key normalization automatically for delta-rule families.
    ///
    /// # `max_levels` (paper §3, R1 §3.5)
    ///
    /// Hard cap on hierarchy depth.
    /// `max_levels = ⌊log₂(T_max)⌋ + 1`, where `T_max` is the
    /// expected maximum stream length. Default 32 covers streams
    /// up to ~4 G tokens. State is padded to `max_levels`
    /// — NOT popcount-sized — for shape stability across stream
    /// length (paper §3.4 Option B / R1 §3.4). This is the paper-
    /// mandated stability choice: streaming consumers require
    /// constant-shape state vectors.
    ///
    /// # `lambda_init` (paper §3.3, R1 §5.3)
    ///
    /// Static bias added to per-level λ logits before
    /// softplus-softmax mixing. Default `1/max_levels` produces
    /// a uniform mixture across all levels — the principled choice
    /// when no information about which levels are useful is
    /// available (no backprop in the streaming setting).
    LogLinear {
        /// Inner update rule (must NOT be `LogLinear`).
        ///
        /// `Box`-ed to keep the enum size small and enable
        /// recursive-shape-checking in factory code.
        inner: alloc::boxed::Box<AttentionMode>,
        /// Hard cap on Fenwick depth. Memory is
        /// `max_levels × d_k × d_v` per head. Default: 32.
        max_levels: usize,
        /// Initial λ bias before softplus-softmax mixing.
        /// Default: `1/max_levels` (uniform mixture).
        lambda_init: f64,
    },
}

/// Full configuration for a multi-head streaming attention layer.
///
/// # Defaults
///
/// - `d_model`: 16
/// - `n_heads`: 4
/// - `d_key`: `d_model / n_heads` (4)
/// - `d_value`: `d_model / n_heads` (4)
/// - `mode`: `RetNet { gamma: 0.95 }`
/// - `seed`: 42
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    /// Input/output dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head key dimension.
    pub d_key: usize,
    /// Per-head value dimension.
    pub d_value: usize,
    /// Architecture variant.
    pub mode: AttentionMode,
    /// PRNG seed for deterministic weight initialization.
    pub seed: u64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 16,
            n_heads: 4,
            d_key: 4,
            d_value: 4,
            mode: AttentionMode::RetNet { gamma: 0.95 },
            seed: 42,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_dimensions_consistent() {
        let cfg = AttentionConfig::default();
        assert_eq!(
            cfg.d_key,
            cfg.d_model / cfg.n_heads,
            "d_key should equal d_model / n_heads"
        );
        assert_eq!(
            cfg.d_value,
            cfg.d_model / cfg.n_heads,
            "d_value should equal d_model / n_heads"
        );
    }

    #[test]
    fn default_mode_is_retnet() {
        let cfg = AttentionConfig::default();
        match cfg.mode {
            AttentionMode::RetNet { gamma } => {
                assert!(
                    (gamma - 0.95).abs() < 1e-12,
                    "default gamma should be 0.95, got {}",
                    gamma
                );
            }
            _ => panic!("default mode should be RetNet"),
        }
    }

    #[test]
    fn config_clone_is_independent() {
        let cfg1 = AttentionConfig::default();
        let mut cfg2 = cfg1.clone();
        cfg2.d_model = 32;
        assert_eq!(cfg1.d_model, 16, "clone should be independent");
        assert_eq!(cfg2.d_model, 32, "cloned config should have new value");
    }

    #[test]
    fn all_modes_constructible() {
        let modes = [
            AttentionMode::RetNet { gamma: 0.9 },
            AttentionMode::Hawk,
            AttentionMode::GLA,
            AttentionMode::GLAVector,
            AttentionMode::DeltaNet,
            AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            },
            AttentionMode::RWKV { initial_decay: 0.5 },
            AttentionMode::MLSTM,
            AttentionMode::DeltaProduct {
                n_compositions: 3,
                reflections: false,
            },
            AttentionMode::RWKV7,
            AttentionMode::HGRN2 { lower_bound: 0.9 },
            AttentionMode::LogLinear {
                inner: alloc::boxed::Box::new(AttentionMode::GLA),
                max_levels: 32,
                lambda_init: 1.0 / 32.0,
            },
        ];
        assert_eq!(modes.len(), 12, "should have exactly 12 modes");
    }

    /// `AttentionMode::LogLinear` is constructible and Box-ed inner is
    /// preserved through Clone (required for builder paths).
    #[test]
    fn log_linear_attention_mode_variant_constructible() {
        let mode = AttentionMode::LogLinear {
            inner: alloc::boxed::Box::new(AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            }),
            max_levels: 32,
            lambda_init: 1.0 / 32.0,
        };
        let cloned = mode.clone();
        let dbg = alloc::format!("{:?}", cloned);
        assert!(
            dbg.contains("LogLinear"),
            "Debug output must name LogLinear, got {dbg}"
        );
        assert!(
            dbg.contains("GatedDeltaNet"),
            "Debug output must include inner mode name, got {dbg}"
        );
    }

    #[test]
    fn config_debug_format_contains_mode() {
        let cfg = AttentionConfig {
            mode: AttentionMode::Hawk,
            ..AttentionConfig::default()
        };
        let debug = alloc::format!("{:?}", cfg);
        assert!(
            debug.contains("Hawk"),
            "debug format should contain mode name, got: {}",
            debug
        );
    }

    #[test]
    fn custom_config_preserves_values() {
        let cfg = AttentionConfig {
            d_model: 64,
            n_heads: 8,
            d_key: 8,
            d_value: 8,
            mode: AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            },
            seed: 1234,
        };
        assert_eq!(cfg.d_model, 64, "d_model should be 64");
        assert_eq!(cfg.n_heads, 8, "n_heads should be 8");
        assert_eq!(cfg.seed, 1234, "seed should be 1234");
    }

    /// Per-token beta mode is distinct from static mode (enum distinguishable).
    #[test]
    fn gated_delta_mode_variants_distinguishable() {
        assert_ne!(
            GatedDeltaMode::Static,
            GatedDeltaMode::PerToken,
            "Static and PerToken must be distinct variants"
        );
    }

    /// DeltaProduct with reflections=true is constructible.
    #[test]
    fn delta_product_reflections_flag_constructible() {
        let mode = AttentionMode::DeltaProduct {
            n_compositions: 2,
            reflections: true,
        };
        let debug = alloc::format!("{:?}", mode);
        assert!(
            debug.contains("DeltaProduct"),
            "debug should contain DeltaProduct, got: {}",
            debug
        );
    }
}
