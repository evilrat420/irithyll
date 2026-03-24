//! Continual learning strategies for streaming neural models.
//!
//! Provides drift-aware strategies that compose with irithyll's existing
//! drift detectors (ADWIN, DDM, Page-Hinkley) to prevent catastrophic
//! forgetting and maintain plasticity in long-running streams.
//!
//! # Strategies
//!
//! - [`StreamingEWC`] -- Elastic Weight Consolidation with streaming Fisher updates
//! - [`DriftMask`] -- Drift-triggered parameter isolation (streaming PackNet)
//! - [`NeuronRegeneration`] -- Continual Backpropagation (Dohare et al., Nature 2024)
//!
//! All implement [`ContinualStrategy`], which hooks into the gradient update
//! loop via `pre_update` / `post_update` and reacts to drift signals via
//! `on_drift`.

pub mod ewc;
pub mod parameter_isolation;
pub mod regeneration;

pub use ewc::StreamingEWC;
pub use parameter_isolation::DriftMask;
pub use regeneration::NeuronRegeneration;

use crate::drift::DriftSignal;

/// Trait for continual learning strategies.
///
/// Strategies modify gradients before weight updates and respond to
/// drift signals from irithyll's drift detectors.
pub trait ContinualStrategy: Send + Sync {
    /// Modify gradients before weight update. Called before each parameter update.
    /// `params` = current parameter values, `gradients` = computed gradients (modified in-place).
    fn pre_update(&mut self, params: &[f64], gradients: &mut [f64]);

    /// Called after weight update with the new parameter values.
    fn post_update(&mut self, params: &[f64]);

    /// React to a drift signal. E.g., update anchor point, recompute masks.
    fn on_drift(&mut self, params: &[f64], signal: DriftSignal);

    /// Number of parameters this strategy covers.
    fn n_params(&self) -> usize;

    /// Reset to initial state.
    fn reset(&mut self);
}
