//! Feature hashing (hashing trick) for dimensionality reduction.
//!
//! [`FeatureHasher`] maps a d-dimensional input to a fixed `n_buckets`-dimensional
//! output using the hashing trick (Weinberger et al., 2009). Each input feature
//! index is hashed to a bucket, and a separate sign hash determines +/-1 to
//! preserve inner products in expectation.
//!
//! The transform is completely stateless -- there is nothing to learn, so
//! `update_and_transform` and `transform` produce identical results, and
//! `reset` is a no-op.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::FeatureHasher;
//! use irithyll::pipeline::StreamingPreprocessor;
//!
//! let hasher = FeatureHasher::new(8);
//! assert_eq!(hasher.n_buckets(), 8);
//!
//! let out = hasher.transform(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! assert_eq!(out.len(), 8);
//! ```

use crate::pipeline::StreamingPreprocessor;

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

/// Multiply-xorshift hash for mapping feature indices to buckets/signs.
///
/// Two different seeds yield two independent hash functions, one for the
/// bucket assignment and one for the sign selection.
#[inline]
fn hash_index(i: usize, seed: u64) -> u64 {
    let mut h = seed ^ (i as u64);
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= h >> 32;
    h
}

/// Seed used for the bucket hash.
const BUCKET_SEED: u64 = 0x9e3779b97f4a7c15; // golden-ratio derived
/// Seed used for the sign hash.
const SIGN_SEED: u64 = 0x6c62272e07bb0142; // different constant

// ---------------------------------------------------------------------------
// FeatureHasher
// ---------------------------------------------------------------------------

/// Feature hashing (hashing trick) preprocessor.
///
/// Maps an arbitrary-length input feature vector to a fixed-size output of
/// `n_buckets` dimensions. For each input feature at index `i`:
///
/// 1. A **bucket** is chosen: `bucket = hash(i, BUCKET_SEED) % n_buckets`.
/// 2. A **sign** is chosen: `sign = if hash(i, SIGN_SEED) & 1 == 0 { +1 } else { -1 }`.
/// 3. The output accumulates: `output[bucket] += sign * features[i]`.
///
/// The sign-flipping ensures that the inner product of two hashed vectors is
/// an unbiased estimator of the original inner product (Weinberger et al., 2009).
///
/// Because the transform is purely deterministic and stateless, both
/// [`update_and_transform`](StreamingPreprocessor::update_and_transform) and
/// [`transform`](StreamingPreprocessor::transform) behave identically, and
/// [`reset`](StreamingPreprocessor::reset) is a no-op.
#[derive(Clone, Debug)]
pub struct FeatureHasher {
    /// Number of output buckets (output dimensionality).
    n_buckets: usize,
}

impl FeatureHasher {
    /// Create a new feature hasher with the given number of output buckets.
    ///
    /// # Panics
    ///
    /// Panics if `n_buckets` is zero.
    ///
    /// ```
    /// use irithyll::preprocessing::FeatureHasher;
    /// let hasher = FeatureHasher::new(16);
    /// assert_eq!(hasher.n_buckets(), 16);
    /// ```
    pub fn new(n_buckets: usize) -> Self {
        assert!(n_buckets > 0, "FeatureHasher: n_buckets must be > 0");
        Self { n_buckets }
    }

    /// Number of output buckets (output dimensionality).
    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }

    /// Core hashing logic shared by both trait methods.
    fn hash_features(&self, features: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.n_buckets];
        for (i, &val) in features.iter().enumerate() {
            let bucket = (hash_index(i, BUCKET_SEED) % self.n_buckets as u64) as usize;
            let sign = if hash_index(i, SIGN_SEED) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            output[bucket] += sign * val;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl StreamingPreprocessor for FeatureHasher {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        // Stateless -- identical to transform.
        self.hash_features(features)
    }

    fn transform(&self, features: &[f64]) -> Vec<f64> {
        self.hash_features(features)
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.n_buckets)
    }

    fn reset(&mut self) {
        // No-op: the hasher is entirely stateless.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dimension_matches_n_buckets() {
        for &n in &[1, 4, 16, 128, 1024] {
            let hasher = FeatureHasher::new(n);
            assert_eq!(
                hasher.output_dim(),
                Some(n),
                "output_dim should be Some({n}) for n_buckets={n}"
            );

            let input = vec![1.0; 50];
            let out = hasher.transform(&input);
            assert_eq!(
                out.len(),
                n,
                "output length should be {n} for n_buckets={n}, got {}",
                out.len()
            );
        }
    }

    #[test]
    fn deterministic_hashing() {
        let hasher = FeatureHasher::new(32);
        let input = vec![1.0, -2.5, 3.3, 0.0, 7.7];

        let out_a = hasher.transform(&input);
        let out_b = hasher.transform(&input);

        assert_eq!(
            out_a, out_b,
            "same input must always produce identical output"
        );
    }

    #[test]
    fn different_inputs_different_outputs() {
        let hasher = FeatureHasher::new(16);

        let out_a = hasher.transform(&[1.0, 0.0, 0.0]);
        let out_b = hasher.transform(&[0.0, 0.0, 1.0]);

        assert_ne!(
            out_a, out_b,
            "[1,0,0] and [0,0,1] should produce different hashed outputs"
        );
    }

    #[test]
    fn sign_preserves_inner_product_approx() {
        // With a large number of buckets (>> input dim), the inner product of
        // hashed vectors should approximate the inner product of the originals.
        //
        // We use vectors with a substantial inner product so that relative
        // error is a meaningful measure. The hashing trick is an unbiased
        // estimator -- collisions add noise, but with buckets >> dim the
        // variance is low.
        let n_buckets = 8192;
        let hasher = FeatureHasher::new(n_buckets);

        // Construct two vectors with a large, well-defined inner product.
        let dim = 50;
        let a: Vec<f64> = (0..dim).map(|i| i as f64 + 1.0).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.5).collect();

        let true_ip: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        // true_ip = sum_{i=1}^{50} i * (i * 0.5) = 0.5 * sum(i^2) = 0.5 * 42925 = 21462.5

        let ha = hasher.transform(&a);
        let hb = hasher.transform(&b);
        let hashed_ip: f64 = ha.iter().zip(hb.iter()).map(|(x, y)| x * y).sum();

        let rel_error = ((hashed_ip - true_ip) / true_ip).abs();

        assert!(
            rel_error < 0.15,
            "hashed inner product ({hashed_ip:.6}) should approximate true inner product \
             ({true_ip:.6}), relative error = {rel_error:.6}"
        );
    }

    #[test]
    fn reset_is_noop() {
        let mut hasher = FeatureHasher::new(16);
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let before = hasher.transform(&input);
        hasher.reset();
        let after = hasher.transform(&input);

        assert_eq!(
            before, after,
            "reset should not change transform output (hasher is stateless)"
        );
    }

    #[test]
    fn single_feature_maps_to_one_bucket() {
        let hasher = FeatureHasher::new(32);
        let out = hasher.transform(&[5.0]);

        let nonzero_count = out.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nonzero_count, 1,
            "a single input feature should map to exactly one bucket, \
             but {nonzero_count} buckets were nonzero"
        );

        // The nonzero value should be +5.0 or -5.0 (sign hash).
        let nonzero_val = out.iter().find(|&&v| v != 0.0).unwrap();
        assert!(
            (*nonzero_val - 5.0).abs() < 1e-12 || (*nonzero_val + 5.0).abs() < 1e-12,
            "single feature value 5.0 should map to +5.0 or -5.0, got {nonzero_val}"
        );
    }

    #[test]
    fn update_and_transform_equals_transform() {
        let mut hasher = FeatureHasher::new(64);
        let input = vec![0.5, -1.2, 3.7, 2.1, -0.8, 4.4];

        let out_transform = hasher.transform(&input);
        let out_update = hasher.update_and_transform(&input);

        assert_eq!(
            out_transform, out_update,
            "update_and_transform and transform must return identical output (stateless hasher)"
        );
    }

    #[test]
    #[should_panic(expected = "n_buckets must be > 0")]
    fn zero_buckets_panics() {
        FeatureHasher::new(0);
    }

    #[test]
    fn empty_input_returns_zero_vector() {
        let hasher = FeatureHasher::new(8);
        let out = hasher.transform(&[]);
        assert_eq!(out.len(), 8, "output length should match n_buckets");
        assert!(
            out.iter().all(|&v| v == 0.0),
            "hashing an empty feature vector should produce all zeros"
        );
    }
}
