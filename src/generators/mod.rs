//! Canonical synthetic stream generators for benchmarking streaming ML algorithms.
//!
//! This module provides reusable, deterministic data stream generators implementing
//! well-known benchmark problems from the streaming ML literature. Each generator
//! produces samples one at a time via the [`StreamGenerator`] trait, making them
//! ideal for prequential (test-then-train) evaluation.
//!
//! # Generators
//!
//! | Generator | Task | Drift | Reference |
//! |-----------|------|-------|-----------|
//! | [`SEA`] | Binary classification | Abrupt | Street & Kim, KDD 2001 |
//! | [`Agrawal`] | Binary classification | Abrupt | Agrawal et al., 1993 |
//! | [`Hyperplane`] | Binary classification | Gradual | Hulten et al., KDD 2001 |
//! | [`LED`] | 10-class classification | Abrupt | Breiman et al., 1984 |
//! | [`Waveform`] | 3-class classification | None | Breiman et al., 1984 |
//! | [`RandomRBF`] | Multiclass classification | Gradual | Bifet et al., MOA 2010 |
//! | [`Friedman`] | Regression | Gradual | Friedman, 1991 |
//! | [`MackeyGlass`] | Regression | None | Mackey & Glass, 1977 |
//! | [`Lorenz`] | Regression | None | Lorenz, 1963 |
//!
//! # Example
//!
//! ```
//! use irithyll::generators::{StreamGenerator, Hyperplane, TaskType};
//!
//! let mut gen = Hyperplane::new(42, 10, 3, 0.01, 0.05);
//! for _ in 0..1000 {
//!     let (features, target) = gen.next_sample();
//!     assert_eq!(features.len(), 10);
//!     assert!(target == 0.0 || target == 1.0);
//! }
//! ```

pub mod agrawal;
pub mod friedman;
pub mod hyperplane;
pub mod led;
pub mod lorenz;
pub mod mackey_glass;
pub mod rbf;
pub mod sea;
pub mod waveform;

pub use agrawal::Agrawal;
pub use friedman::Friedman;
pub use hyperplane::Hyperplane;
pub use led::LED;
pub use lorenz::Lorenz;
pub use mackey_glass::MackeyGlass;
pub use rbf::RandomRBF;
pub use sea::SEA;
pub use waveform::Waveform;

// ---------------------------------------------------------------------------
// TaskType enum
// ---------------------------------------------------------------------------

/// Describes the learning task for a stream generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    /// Continuous target prediction.
    Regression,
    /// Two-class (0/1) classification.
    BinaryClassification,
    /// Multi-class classification with `n_classes` classes (0..n_classes-1).
    MulticlassClassification {
        /// Number of distinct classes.
        n_classes: usize,
    },
}

// ---------------------------------------------------------------------------
// StreamGenerator trait
// ---------------------------------------------------------------------------

/// A synthetic data stream that produces samples one at a time.
///
/// All generators are deterministic given a seed, produce finite `f64` values,
/// and track concept drift events.
pub trait StreamGenerator {
    /// Generate the next sample as `(features, target)`.
    fn next_sample(&mut self) -> (Vec<f64>, f64);

    /// Number of features per sample.
    fn n_features(&self) -> usize;

    /// Whether this is regression, binary, or multiclass classification.
    fn task_type(&self) -> TaskType;

    /// Returns `true` if a concept drift occurred during the most recent
    /// call to [`next_sample`](StreamGenerator::next_sample).
    fn drift_occurred(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Xorshift64 PRNG (no external dependencies)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG for deterministic stream generation.
///
/// Identical to the one used in `benches/real_world_bench.rs`.
#[derive(Debug, Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    /// Create a new PRNG from a seed. Avoids the zero state.
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    /// Raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }

    /// Approximate standard normal via Box-Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }

    /// Uniform integer in `[0, n)`.
    pub fn uniform_int(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    /// Returns `true` with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}
