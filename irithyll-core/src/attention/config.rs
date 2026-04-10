//! Configuration types for the unified streaming linear attention engine.
//!
//! [`AttentionMode`] selects which architecture variant to use, while
//! [`AttentionConfig`] holds the full configuration including dimensions,
//! head count, and PRNG seed.

/// Selects the attention architecture variant.
///
/// Each variant corresponds to a different parameterization of the unified
/// structured linear attention recurrence `S_t = decay_t * S_{t-1} + update_t`.
///
/// # Variants
///
/// - **RetNet** -- Fixed exponential decay (Sun et al., 2023)
/// - **Hawk** -- Gated scalar recurrence with vector state (De et al., 2024)
/// - **GLA** -- Gated Linear Attention with data-dependent gate (Yang et al., 2024)
/// - **DeltaNet** -- Delta rule error-corrective update (Schlag et al., 2021)
/// - **GatedDeltaNet** -- GLA + delta rule combined (Yang et al., 2024, NVIDIA)
/// - **RWKV** -- Exponential decay with dynamic w (Peng et al., 2024)
/// - **MLSTM** -- xLSTM matrix memory with forget+input gates (Beck et al., 2024)
/// - **DeltaProduct** -- Product of n_h Householder delta rules (Siems et al., NeurIPS 2025)
/// - **RWKV7** -- Vector-gated delta rule with DPLR transitions (Peng et al., 2025)
/// - **HGRN2** -- Lower-bounded gated linear RNN with state expansion (Qin et al., ICML 2024)
#[derive(Clone, Debug)]
pub enum AttentionMode {
    /// Fixed exponential decay: `S = gamma * S + k * v^T`.
    RetNet {
        /// Decay factor in (0, 1). Typical: 0.9 to 0.999.
        gamma: f64,
    },
    /// Gated scalar recurrence with vector state (Griffin/Hawk).
    Hawk,
    /// Gated Linear Attention with data-dependent sigmoid gate.
    GLA,
    /// Delta rule: error-corrective associative memory update.
    DeltaNet,
    /// Gated delta rule: GLA gate + delta error correction (Yang et al., ICLR 2025).
    ///
    /// Combines GLA's data-dependent gating with DeltaNet's error-corrective
    /// delta rule, plus learnable beta scaling and L2-normalized keys.
    GatedDeltaNet {
        /// Learnable beta scaling factor (default: 1.0).
        /// Controls how aggressively the error-corrective update is applied.
        beta_scale: f64,
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
    /// Beta range [0, 2] enables full reflections. Includes a scalar forget gate.
    DeltaProduct {
        /// Number of Householder compositions per token (typically 2-4).
        n_compositions: usize,
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
            AttentionMode::DeltaNet,
            AttentionMode::GatedDeltaNet { beta_scale: 1.0 },
            AttentionMode::RWKV { initial_decay: 0.5 },
            AttentionMode::MLSTM,
            AttentionMode::DeltaProduct { n_compositions: 3 },
            AttentionMode::RWKV7,
            AttentionMode::HGRN2 { lower_bound: 0.9 },
        ];
        assert_eq!(modes.len(), 10, "should have exactly 10 modes");
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
            mode: AttentionMode::GatedDeltaNet { beta_scale: 1.0 },
            seed: 1234,
        };
        assert_eq!(cfg.d_model, 64, "d_model should be 64");
        assert_eq!(cfg.n_heads, 8, "n_heads should be 8");
        assert_eq!(cfg.seed, 1234, "seed should be 1234");
    }
}
