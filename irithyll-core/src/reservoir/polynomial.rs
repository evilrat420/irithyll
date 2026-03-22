//! Polynomial monomial generator for nonlinear feature expansion.
//!
//! [`HighDegreePolynomial`] generates all monomials of a given degree from an
//! input feature vector, used by NG-RC to create nonlinear observation features.
//!
//! For degree 2, the monomials for `[x0, x1, x2]` are:
//! `[x0*x0, x0*x1, x0*x2, x1*x1, x1*x2, x2*x2]`
//!
//! The original linear features are **not** included in the output — they are
//! handled separately by the caller (NG-RC concatenates `O_lin` and `O_nonlin`).

use alloc::vec::Vec;

/// Generates polynomial monomials (degree >= 2) for a feature vector.
///
/// Only produces monomials of exactly degree `p`. For degree 2 this is all
/// products `x_i * x_j` where `i <= j`. For degree 3, all `x_i * x_j * x_k`
/// where `i <= j <= k`.
///
/// # Examples
///
/// ```
/// use irithyll_core::reservoir::HighDegreePolynomial;
///
/// let poly = HighDegreePolynomial::new(2);
/// let monomials = poly.generate(&[1.0, 2.0, 3.0]);
/// // [1*1, 1*2, 1*3, 2*2, 2*3, 3*3] = [1, 2, 3, 4, 6, 9]
/// assert_eq!(monomials.len(), 6);
/// ```
pub struct HighDegreePolynomial {
    degree: usize,
}

impl HighDegreePolynomial {
    /// Create a polynomial monomial generator for the given degree.
    ///
    /// # Panics
    ///
    /// Panics if `degree < 2`.
    pub fn new(degree: usize) -> Self {
        assert!(degree >= 2, "HighDegreePolynomial degree must be >= 2");
        Self { degree }
    }

    /// The polynomial degree.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Generate all monomials of the configured degree from the input features.
    ///
    /// Returns only the monomials — does **not** include the original features.
    /// The monomials are ordered lexicographically by index tuple.
    pub fn generate(&self, features: &[f64]) -> Vec<f64> {
        let n = features.len();
        let mut result = Vec::with_capacity(self.output_dim(n));
        let mut indices = Vec::with_capacity(self.degree);
        self.gen_recursive(features, n, 0, &mut indices, &mut result);
        result
    }

    /// Compute the number of monomials produced for the given input dimension.
    ///
    /// This is the multiset coefficient C(n + p - 1, p) where n = input_dim
    /// and p = degree.
    pub fn output_dim(&self, input_dim: usize) -> usize {
        // C(n + p - 1, p) = (n+p-1)! / (p! * (n-1)!)
        // Compute via iterative multiplication to avoid large factorials.
        if input_dim == 0 {
            return 0;
        }
        let p = self.degree;
        let mut result: usize = 1;
        for i in 0..p {
            result = result * (input_dim + i) / (i + 1);
        }
        result
    }

    /// Recursive helper: generates all monomials with indices >= `start`.
    fn gen_recursive(
        &self,
        features: &[f64],
        n: usize,
        start: usize,
        indices: &mut Vec<usize>,
        result: &mut Vec<f64>,
    ) {
        if indices.len() == self.degree {
            // Compute the product of features at the selected indices.
            let mut product = 1.0;
            for &idx in indices.iter() {
                product *= features[idx];
            }
            result.push(product);
            return;
        }
        for i in start..n {
            indices.push(i);
            self.gen_recursive(features, n, i, indices, result);
            indices.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree2_three_features() {
        let poly = HighDegreePolynomial::new(2);
        let result = poly.generate(&[1.0, 2.0, 3.0]);
        // i<=j: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
        // = [1*1, 1*2, 1*3, 2*2, 2*3, 3*3]
        // = [1, 2, 3, 4, 6, 9]
        assert_eq!(result.len(), 6);
        let expected = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "monomial[{}]: expected {}, got {}",
                i,
                exp,
                got,
            );
        }
    }

    #[test]
    fn degree2_two_features() {
        let poly = HighDegreePolynomial::new(2);
        let result = poly.generate(&[3.0, 5.0]);
        // (0,0),(0,1),(1,1) = [9, 15, 25]
        assert_eq!(result.len(), 3);
        assert!((result[0] - 9.0).abs() < 1e-12);
        assert!((result[1] - 15.0).abs() < 1e-12);
        assert!((result[2] - 25.0).abs() < 1e-12);
    }

    #[test]
    fn degree3_two_features() {
        let poly = HighDegreePolynomial::new(3);
        let result = poly.generate(&[2.0, 3.0]);
        // i<=j<=k: (0,0,0),(0,0,1),(0,1,1),(1,1,1)
        // = [8, 12, 18, 27]
        assert_eq!(result.len(), 4);
        let expected = [8.0, 12.0, 18.0, 27.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "monomial[{}]: expected {}, got {}",
                i,
                exp,
                got,
            );
        }
    }

    #[test]
    fn output_dim_matches_generated() {
        let poly = HighDegreePolynomial::new(2);
        for n in 1..=8 {
            let features: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let result = poly.generate(&features);
            let expected_dim = poly.output_dim(n);
            assert_eq!(
                result.len(),
                expected_dim,
                "output_dim({}) = {} but generated {}",
                n,
                expected_dim,
                result.len(),
            );
        }
    }

    #[test]
    fn output_dim_degree3() {
        let poly = HighDegreePolynomial::new(3);
        // C(n+2, 3):
        // n=1: C(3,3) = 1
        // n=2: C(4,3) = 4
        // n=3: C(5,3) = 10
        assert_eq!(poly.output_dim(1), 1);
        assert_eq!(poly.output_dim(2), 4);
        assert_eq!(poly.output_dim(3), 10);
    }

    #[test]
    fn single_feature_degree2() {
        let poly = HighDegreePolynomial::new(2);
        let result = poly.generate(&[7.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 49.0).abs() < 1e-12);
    }

    #[test]
    fn empty_features_returns_empty() {
        let poly = HighDegreePolynomial::new(2);
        let result = poly.generate(&[]);
        assert!(result.is_empty());
        assert_eq!(poly.output_dim(0), 0);
    }

    #[test]
    #[should_panic(expected = "degree must be >= 2")]
    fn degree_one_panics() {
        let _ = HighDegreePolynomial::new(1);
    }
}
