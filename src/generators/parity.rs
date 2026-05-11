//! XOR parity stream generator.
//!
//! Generates a binary classification stream where the label is the XOR parity
//! of a selected subset of input bits. This tests whether a model can track
//! the long-range parity of specific binary channels — a task that is
//! provably hard for models without sufficient state capacity or complex
//! eigenvalue support (e.g., V3Exp with complex SSM).
//!
//! # Protocol
//!
//! At each step:
//! 1. Sample a random binary vector of length `n_bits`.
//! 2. Compute the label as XOR of bits at indices `parity_bits`.
//! 3. Output `(binary_features, label)` where features are `f64` 0.0/1.0.
//!
//! The stream is i.i.d. (no temporal dependency in the inputs), but
//! the label rule can be made harder by giving the model only a subset
//! of bits and hiding which subset determines parity.

use super::{Rng, StreamGenerator, TaskType};

/// XOR parity stream generator.
///
/// # Parameters
/// - `seed`: PRNG seed for reproducibility
/// - `n_bits`: total number of binary input features (default: 8)
/// - `parity_bits`: indices of the bits that contribute to parity (default: all bits)
///
/// # Output
/// - Features: `n_bits` values, each 0.0 or 1.0
/// - Target: 0.0 or 1.0 (XOR parity of the selected bits)
///
/// # Why parity is hard
///
/// XOR parity over k bits requires tracking 2^k equivalence classes. For a
/// linear model with no nonlinearity this is impossible regardless of the number
/// of features. For SSMs, only those with complex or sign-alternating eigenvalues
/// can track XOR parity over time (Abbe et al., 2023; Goel et al., 2022).
#[derive(Debug, Clone)]
pub struct ParityStream {
    rng: Rng,
    /// Total number of binary input features.
    n_bits: usize,
    /// Sorted indices of bits that feed into the parity computation.
    parity_indices: Vec<usize>,
}

impl ParityStream {
    /// Default number of bits.
    pub const DEFAULT_N_BITS: usize = 8;

    /// Create a parity stream.
    ///
    /// - `seed`: PRNG seed.
    /// - `n_bits`: total number of binary input features.
    /// - `parity_count`: number of bits (from the first `parity_count` indices) that
    ///   contribute to the XOR parity label. Must be `<= n_bits`.
    ///
    /// # Panics
    ///
    /// Panics if `parity_count == 0` or `parity_count > n_bits`.
    pub fn new(seed: u64, n_bits: usize, parity_count: usize) -> Self {
        assert!(parity_count > 0, "parity_count must be > 0");
        assert!(
            parity_count <= n_bits,
            "parity_count ({}) must be <= n_bits ({})",
            parity_count,
            n_bits
        );
        let parity_bits: Vec<usize> = (0..parity_count).collect();
        Self::with_config(seed, n_bits, parity_bits)
    }

    /// Create a parity stream with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `n_bits == 0`, `parity_bits` is empty, or any index in
    /// `parity_bits` is `>= n_bits`.
    pub fn with_config(seed: u64, n_bits: usize, parity_bits: Vec<usize>) -> Self {
        assert!(n_bits > 0, "n_bits must be > 0");
        assert!(!parity_bits.is_empty(), "parity_bits must not be empty");
        for &idx in &parity_bits {
            assert!(
                idx < n_bits,
                "parity_bits index {} out of range for n_bits={}",
                idx,
                n_bits
            );
        }

        let mut sorted = parity_bits;
        sorted.sort_unstable();
        sorted.dedup();

        Self {
            rng: Rng::new(seed),
            n_bits,
            parity_indices: sorted,
        }
    }

    /// Number of input bits.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Indices contributing to parity.
    pub fn parity_indices(&self) -> &[usize] {
        &self.parity_indices
    }

    /// Compute parity label from a bit vector.
    ///
    /// Returns 1.0 if an odd number of selected bits are 1, else 0.0.
    pub fn compute_parity(bits: &[f64], parity_indices: &[usize]) -> f64 {
        let xor = parity_indices
            .iter()
            .fold(0u8, |acc, &i| acc ^ (bits[i] as u8));
        xor as f64
    }
}

impl StreamGenerator for ParityStream {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Random binary feature vector.
        let features: Vec<f64> = (0..self.n_bits)
            .map(|_| if self.rng.bernoulli(0.5) { 1.0 } else { 0.0 })
            .collect();

        let label = Self::compute_parity(&features, &self.parity_indices);

        (features, label)
    }

    fn n_features(&self) -> usize {
        self.n_bits
    }

    fn task_type(&self) -> TaskType {
        TaskType::BinaryClassification
    }

    fn drift_occurred(&self) -> bool {
        false // i.i.d. stream, no concept drift.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_produces_correct_n_features() {
        let n = ParityStream::DEFAULT_N_BITS;
        let mut gen = ParityStream::new(42, n, n);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            n,
            "features should have {} dims, got {}",
            n,
            features.len()
        );
    }

    #[test]
    fn parity_task_type_is_binary_classification() {
        let gen = ParityStream::new(42, 8, 8);
        assert_eq!(gen.task_type(), TaskType::BinaryClassification);
    }

    #[test]
    fn parity_labels_are_binary() {
        let mut gen = ParityStream::new(77, 8, 8);
        for _ in 0..500 {
            let (_, target) = gen.next_sample();
            assert!(
                target == 0.0 || target == 1.0,
                "parity label should be 0.0 or 1.0, got {}",
                target
            );
        }
    }

    #[test]
    fn parity_no_drift() {
        let mut gen = ParityStream::new(42, 8, 8);
        for _ in 0..500 {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "parity stream should not drift");
        }
    }

    #[test]
    fn parity_deterministic_with_same_seed() {
        let mut gen1 = ParityStream::new(42, 8, 8);
        let mut gen2 = ParityStream::new(42, 8, 8);
        for _ in 0..500 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn parity_label_matches_xor() {
        // Verify that the label is indeed the XOR parity of the selected bits.
        let mut gen = ParityStream::with_config(123, 4, vec![0, 2]);
        for _ in 0..200 {
            let (features, label) = gen.next_sample();
            let expected = ParityStream::compute_parity(&features, &[0, 2]);
            assert!(
                (label - expected).abs() < 1e-12,
                "label {} should match XOR parity {} for features {:?}",
                label,
                expected,
                features
            );
        }
    }

    #[test]
    fn parity_balanced_classes() {
        // i.i.d. uniform bits → parity should be ~50% 0 and ~50% 1.
        let mut gen = ParityStream::new(55, 8, 8);
        let mut ones = 0usize;
        let n = 2000;
        for _ in 0..n {
            let (_, t) = gen.next_sample();
            if t > 0.5 {
                ones += 1;
            }
        }
        let ratio = ones as f64 / n as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "parity classes should be ~50/50, got ratio={}",
            ratio
        );
    }

    #[test]
    fn parity_features_are_binary() {
        let mut gen = ParityStream::new(1, 8, 8);
        for _ in 0..200 {
            let (features, _) = gen.next_sample();
            for &f in &features {
                assert!(
                    f == 0.0 || f == 1.0,
                    "all features should be 0.0 or 1.0, got {}",
                    f
                );
            }
        }
    }

    #[test]
    fn parity_subset_parity_correct() {
        // Only bits 1 and 3 contribute: XOR(features[1], features[3]).
        let mut gen = ParityStream::with_config(7, 6, vec![1, 3]);
        for _ in 0..100 {
            let (features, label) = gen.next_sample();
            let xor = (features[1] as u8) ^ (features[3] as u8);
            assert!(
                (label - xor as f64).abs() < 1e-12,
                "subset parity label {} should match XOR({},{})={}",
                label,
                features[1],
                features[3],
                xor
            );
        }
    }

    #[test]
    fn parity_custom_n_bits() {
        let mut gen = ParityStream::with_config(42, 16, vec![0, 7, 15]);
        assert_eq!(gen.n_features(), 16);
        let (features, _) = gen.next_sample();
        assert_eq!(features.len(), 16);

        // Also test new() with custom count.
        let gen2 = ParityStream::new(42, 16, 4);
        assert_eq!(gen2.n_features(), 16);
        assert_eq!(gen2.parity_indices(), &[0, 1, 2, 3]);
    }
}
