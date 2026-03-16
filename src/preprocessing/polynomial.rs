//! Degree-2 polynomial feature generation for interaction modeling.
//!
//! [`PolynomialFeatures`] expands an input feature vector into degree-2
//! polynomial terms: original features, pairwise interactions, and squared
//! terms. The transform is completely stateless -- there is nothing to learn,
//! so `update_and_transform` and `transform` produce identical results.
//!
//! For input `[x0, x1, x2]` the full output is:
//! `[x0, x1, x2, x0*x0, x0*x1, x0*x2, x1*x1, x1*x2, x2*x2]`
//!
//! With `interaction_only` enabled, squared terms are excluded:
//! `[x0, x1, x2, x0*x1, x0*x2, x1*x2]`
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::PolynomialFeatures;
//! use irithyll::pipeline::StreamingPreprocessor;
//!
//! let poly = PolynomialFeatures::new();
//! let out = poly.transform(&[1.0, 2.0, 3.0]);
//! assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]);
//! ```

use crate::pipeline::StreamingPreprocessor;

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

/// Degree-2 polynomial feature generator.
///
/// Generates original features plus all degree-2 combinations (squared terms
/// and pairwise interactions). When [`interaction_only`](Self::interaction_only)
/// is constructed, squared terms (x_i * x_i) are excluded, leaving only
/// cross-interactions (x_i * x_j where i < j).
///
/// Because the transform is purely deterministic and stateless, both
/// [`update_and_transform`](StreamingPreprocessor::update_and_transform) and
/// [`transform`](StreamingPreprocessor::transform) behave identically, and
/// [`reset`](StreamingPreprocessor::reset) clears only the cached input
/// dimensionality.
#[derive(Clone, Debug)]
pub struct PolynomialFeatures {
    /// If true, only include cross-interactions (no x_i^2 terms).
    interaction_only: bool,
    /// Cached input dimensionality, set on first `update_and_transform` call.
    n_input_features: Option<usize>,
}

impl PolynomialFeatures {
    /// Create a polynomial feature generator with full degree-2 expansion.
    ///
    /// Includes original features, squared terms, and pairwise interactions.
    ///
    /// ```
    /// use irithyll::preprocessing::PolynomialFeatures;
    /// use irithyll::pipeline::StreamingPreprocessor;
    ///
    /// let poly = PolynomialFeatures::new();
    /// let out = poly.transform(&[2.0, 3.0]);
    /// // [a, b, a², ab, b²] = [2, 3, 4, 6, 9]
    /// assert_eq!(out, vec![2.0, 3.0, 4.0, 6.0, 9.0]);
    /// ```
    pub fn new() -> Self {
        Self {
            interaction_only: false,
            n_input_features: None,
        }
    }

    /// Create a polynomial feature generator that only includes cross-interactions.
    ///
    /// Squared terms (x_i * x_i) are excluded; only x_i * x_j where i < j
    /// are generated.
    ///
    /// ```
    /// use irithyll::preprocessing::PolynomialFeatures;
    /// use irithyll::pipeline::StreamingPreprocessor;
    ///
    /// let poly = PolynomialFeatures::interaction_only();
    /// let out = poly.transform(&[2.0, 3.0]);
    /// // [a, b, ab] = [2, 3, 6]
    /// assert_eq!(out, vec![2.0, 3.0, 6.0]);
    /// ```
    pub fn interaction_only() -> Self {
        Self {
            interaction_only: true,
            n_input_features: None,
        }
    }

    /// Returns `true` if this generator excludes squared terms.
    pub fn is_interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Compute the output dimensionality for a given input dimensionality.
    fn compute_output_dim(&self, d: usize) -> usize {
        if self.interaction_only {
            // d original + d*(d-1)/2 cross terms
            d + d * d.saturating_sub(1) / 2
        } else {
            // d original + d*(d+1)/2 interaction/squared terms
            d + d * (d + 1) / 2
        }
    }

    /// Core feature generation logic.
    ///
    /// Produces: original features, then degree-2 combinations.
    /// When `interaction_only` is false, combinations include (i, j) where i <= j.
    /// When `interaction_only` is true, combinations include (i, j) where i < j.
    fn generate(&self, features: &[f64]) -> Vec<f64> {
        let d = features.len();
        let capacity = self.compute_output_dim(d);
        let mut out = Vec::with_capacity(capacity);

        // Original features first.
        out.extend_from_slice(features);

        // Degree-2 terms.
        for i in 0..d {
            let start = if self.interaction_only { i + 1 } else { i };
            for j in start..d {
                out.push(features[i] * features[j]);
            }
        }

        out
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl StreamingPreprocessor for PolynomialFeatures {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        self.n_input_features = Some(features.len());
        self.generate(features)
    }

    fn transform(&self, features: &[f64]) -> Vec<f64> {
        self.generate(features)
    }

    fn output_dim(&self) -> Option<usize> {
        self.n_input_features.map(|d| self.compute_output_dim(d))
    }

    fn reset(&mut self) {
        self.n_input_features = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree2_full_output() {
        let poly = PolynomialFeatures::new();
        let out = poly.transform(&[1.0, 2.0, 3.0]);
        // [x0, x1, x2, x0*x0, x0*x1, x0*x2, x1*x1, x1*x2, x2*x2]
        // [1, 2, 3, 1, 2, 3, 4, 6, 9]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn interaction_only_output() {
        let poly = PolynomialFeatures::interaction_only();
        let out = poly.transform(&[1.0, 2.0, 3.0]);
        // [x0, x1, x2, x0*x1, x0*x2, x1*x2]
        // [1, 2, 3, 2, 3, 6]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn single_feature_full() {
        let poly = PolynomialFeatures::new();
        let out = poly.transform(&[5.0]);
        // [x0, x0*x0] = [5, 25]
        assert_eq!(out, vec![5.0, 25.0]);
    }

    #[test]
    fn single_feature_interaction_only() {
        let poly = PolynomialFeatures::interaction_only();
        let out = poly.transform(&[5.0]);
        // No cross terms possible with a single feature.
        // [x0] = [5]
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn output_dim_correct() {
        // Full mode: d + d*(d+1)/2
        let mut poly_full = PolynomialFeatures::new();
        for d in 1usize..=5 {
            let input: Vec<f64> = (1..=d).map(|x| x as f64).collect();
            let out = poly_full.update_and_transform(&input);
            let expected = d + d * (d + 1) / 2;
            assert_eq!(
                poly_full.output_dim(),
                Some(expected),
                "full mode: d={d}, expected output_dim={expected}"
            );
            assert_eq!(out.len(), expected);
            poly_full.reset();
        }

        // Interaction-only mode: d + d*(d-1)/2
        let mut poly_int = PolynomialFeatures::interaction_only();
        for d in 1usize..=5 {
            let input: Vec<f64> = (1..=d).map(|x| x as f64).collect();
            let out = poly_int.update_and_transform(&input);
            let expected = d + d * d.saturating_sub(1) / 2;
            assert_eq!(
                poly_int.output_dim(),
                Some(expected),
                "interaction_only mode: d={d}, expected output_dim={expected}"
            );
            assert_eq!(out.len(), expected);
            poly_int.reset();
        }
    }

    #[test]
    fn stateless_deterministic() {
        let poly = PolynomialFeatures::new();
        let input = [3.0, -1.0, 2.5, 0.0];

        let out1 = poly.transform(&input);
        let out2 = poly.transform(&input);
        let out3 = poly.transform(&input);

        assert_eq!(out1, out2);
        assert_eq!(out2, out3);
    }

    #[test]
    fn update_and_transform_equals_transform() {
        let mut poly = PolynomialFeatures::new();
        let input = [1.5, -2.0, 0.5];

        let out_transform = poly.transform(&input);
        let out_update = poly.update_and_transform(&input);

        assert_eq!(
            out_transform, out_update,
            "transform and update_and_transform must produce identical output"
        );
    }

    #[test]
    fn reset_clears_dim() {
        let mut poly = PolynomialFeatures::new();
        assert_eq!(poly.output_dim(), None);

        poly.update_and_transform(&[1.0, 2.0]);
        assert_eq!(poly.output_dim(), Some(5));

        poly.reset();
        assert_eq!(poly.output_dim(), None);
    }

    #[test]
    fn empty_input() {
        let poly = PolynomialFeatures::new();
        let out = poly.transform(&[]);
        let empty: Vec<f64> = vec![];
        assert_eq!(out, empty);

        let poly_int = PolynomialFeatures::interaction_only();
        let out_int = poly_int.transform(&[]);
        assert_eq!(out_int, empty);
    }

    #[test]
    fn is_interaction_only_accessor() {
        let poly_full = PolynomialFeatures::new();
        assert!(!poly_full.is_interaction_only());

        let poly_int = PolynomialFeatures::interaction_only();
        assert!(poly_int.is_interaction_only());
    }

    #[test]
    fn two_features_full() {
        let poly = PolynomialFeatures::new();
        let out = poly.transform(&[2.0, 3.0]);
        // [a, b, a², ab, b²] = [2, 3, 4, 6, 9]
        assert_eq!(out, vec![2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn two_features_interaction_only() {
        let poly = PolynomialFeatures::interaction_only();
        let out = poly.transform(&[2.0, 3.0]);
        // [a, b, ab] = [2, 3, 6]
        assert_eq!(out, vec![2.0, 3.0, 6.0]);
    }
}
