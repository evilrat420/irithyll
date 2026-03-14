//! Candid Covariance-free Incremental PCA (CCIPCA) for streaming dimensionality reduction.
//!
//! [`CCIPCA`] implements the algorithm of Weng, Zhang & Hwang (2003) which
//! incrementally estimates the leading eigenvectors of the data covariance
//! matrix in O(kd) time per sample, where k = `n_components` and
//! d = `n_features`. No covariance matrix is ever formed or stored.
//!
//! The `amnestic` parameter controls forgetting: with `amnestic = 0.0` all
//! samples are weighted equally; higher values bias toward more recent data,
//! useful for non-stationary streams.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::CCIPCA;
//!
//! let mut pca = CCIPCA::new(2);
//!
//! // Feed high-dimensional samples one at a time.
//! for i in 0..200 {
//!     let x = [i as f64, (i as f64) * 0.1, (i as f64) * 0.01];
//!     pca.update(&x);
//! }
//!
//! // Project a new observation onto the top-2 principal components.
//! let reduced = pca.transform(&[100.0, 10.0, 1.0]);
//! assert_eq!(reduced.len(), 2);
//! ```

/// Candid Covariance-free Incremental PCA (CCIPCA).
///
/// Maintains `n_components` eigenvector estimates that are updated with each
/// new sample using a covariance-free rank-one update rule. The `amnestic`
/// weight controls how quickly old data is forgotten (0.0 = equal weighting).
///
/// Dimensionality is determined lazily from the first call to
/// [`update`](Self::update), matching the pattern of
/// [`IncrementalNormalizer`](crate::preprocessing::IncrementalNormalizer).
#[derive(Clone, Debug)]
pub struct CCIPCA {
    /// Number of principal components to estimate (k).
    n_components: usize,
    /// k eigenvectors, each of length d (n_features).
    components: Vec<Vec<f64>>,
    /// k eigenvalue estimates (norms of the un-normalised eigenvectors).
    eigenvalues: Vec<f64>,
    /// Running mean via Welford's algorithm.
    mean: Vec<f64>,
    /// Number of samples incorporated.
    count: u64,
    /// Feature dimensionality, `None` until the first sample.
    n_features: Option<usize>,
    /// Forgetting weight. 0.0 = equal weight for all history.
    amnestic: f64,
}

impl CCIPCA {
    /// Create a CCIPCA estimator with `n_components` components and no forgetting.
    ///
    /// ```
    /// use irithyll::preprocessing::CCIPCA;
    /// let pca = CCIPCA::new(3);
    /// assert_eq!(pca.n_components(), 3);
    /// assert_eq!(pca.count(), 0);
    /// ```
    pub fn new(n_components: usize) -> Self {
        Self::with_amnestic(n_components, 0.0)
    }

    /// Create a CCIPCA estimator with a custom amnestic (forgetting) weight.
    ///
    /// Higher `amnestic` values give more weight to recent samples. A value
    /// of 0.0 weights all samples equally.
    ///
    /// ```
    /// use irithyll::preprocessing::CCIPCA;
    /// let pca = CCIPCA::with_amnestic(2, 1.5);
    /// assert_eq!(pca.n_components(), 2);
    /// ```
    pub fn with_amnestic(n_components: usize, amnestic: f64) -> Self {
        Self {
            n_components,
            components: Vec::new(),
            eigenvalues: Vec::new(),
            mean: Vec::new(),
            count: 0,
            n_features: None,
            amnestic,
        }
    }

    /// Incorporate a new sample and update eigencomponent estimates.
    ///
    /// On the first call, the feature dimensionality is established from
    /// `features.len()`. Subsequent calls must provide the same length.
    ///
    /// The algorithm performs:
    /// 1. Welford mean update.
    /// 2. Center the sample.
    /// 3. Rank-one update of each eigenvector with re-orthogonalisation.
    /// 4. Sort components by descending eigenvalue.
    ///
    /// # Panics
    ///
    /// Panics if `features.len()` does not match the established dimensionality.
    pub fn update(&mut self, features: &[f64]) {
        let _d = self.init_or_check(features.len());

        // --- Welford mean update ---
        self.count += 1;
        let n = self.count;
        let n_f = n as f64;

        for (j, &fj) in features.iter().enumerate() {
            let delta = fj - self.mean[j];
            self.mean[j] += delta / n_f;
        }

        // --- Center the sample ---
        let mut x_c: Vec<f64> = features
            .iter()
            .zip(self.mean.iter())
            .map(|(&f, &m)| f - m)
            .collect();

        // --- Update each component ---
        for i in 0..self.n_components {
            if n <= (i as u64) + 1 {
                // Not enough samples yet for this component -- initialise.
                self.components[i] = x_c.clone();
                self.eigenvalues[i] = norm(&x_c);
                if self.eigenvalues[i] > 0.0 {
                    for v in &mut self.components[i] {
                        *v /= self.eigenvalues[i];
                    }
                }
            } else {
                // Weighted update of eigenvector.
                let w = (n_f - 1.0 - self.amnestic) / n_f;
                let proj = dot(&x_c, &self.components[i]);
                let scale = (1.0 + self.amnestic) / n_f * proj;

                let v_i: Vec<f64> = self.components[i]
                    .iter()
                    .zip(x_c.iter())
                    .map(|(&ci, &xj)| w * self.eigenvalues[i] * ci + scale * xj)
                    .collect();

                let norm_v = norm(&v_i);
                self.eigenvalues[i] = norm_v;
                if norm_v > 0.0 {
                    let inv = 1.0 / norm_v;
                    for (cj, &vj) in self.components[i].iter_mut().zip(v_i.iter()) {
                        *cj = vj * inv;
                    }
                }
            }

            // Re-orthogonalise: subtract the projection of x_c onto component i.
            let proj = dot(&x_c, &self.components[i]);
            for (xj, &cj) in x_c.iter_mut().zip(self.components[i].iter()) {
                *xj -= proj * cj;
            }
        }

        // --- Sort by eigenvalue descending (keep components aligned) ---
        self.sort_components();
    }

    /// Project `features` onto the current principal components.
    ///
    /// Centers the input using the running mean and computes the dot product
    /// with each eigenvector. Returns a `Vec` of length `n_components`.
    ///
    /// # Panics
    ///
    /// Panics if `features.len()` does not match the established dimensionality,
    /// or if no samples have been incorporated yet.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        let _d = self.check_features_len(features.len());
        assert!(
            self.count > 0,
            "CCIPCA: cannot transform before any updates"
        );

        let mut out = Vec::with_capacity(self.n_components);
        for component in &self.components {
            let proj: f64 = features
                .iter()
                .zip(self.mean.iter())
                .zip(component.iter())
                .map(|((&f, &m), &c)| (f - m) * c)
                .sum();
            out.push(proj);
        }
        out
    }

    /// Update eigencomponents with `features`, then project onto them.
    ///
    /// Equivalent to calling [`update`](Self::update) followed by
    /// [`transform`](Self::transform).
    pub fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        self.update(features);
        self.transform(features)
    }

    /// Reset all state, preserving `n_components` and `amnestic` settings.
    ///
    /// After reset, the next [`update`](Self::update) call will re-initialise
    /// the feature dimensionality from the input.
    pub fn reset(&mut self) {
        self.components.clear();
        self.eigenvalues.clear();
        self.mean.clear();
        self.count = 0;
        self.n_features = None;
    }

    /// Number of principal components being estimated.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Feature dimensionality, or `None` if not yet initialised.
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Number of samples incorporated so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Current eigenvalue estimates (sorted descending).
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Explained variance ratio for each component.
    ///
    /// Each element is the component's eigenvalue divided by the sum of all
    /// eigenvalues. Returns an empty `Vec` if no samples have been incorporated.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.eigenvalues.iter().sum();
        if total <= 0.0 {
            return vec![0.0; self.eigenvalues.len()];
        }
        self.eigenvalues.iter().map(|&ev| ev / total).collect()
    }

    // --- Internal helpers ---

    /// Lazy-initialise buffers on first sample, or check dimension on subsequent calls.
    fn init_or_check(&mut self, len: usize) -> usize {
        match self.n_features {
            None => {
                let d = len;
                self.n_features = Some(d);
                self.mean = vec![0.0; d];
                self.components = vec![vec![0.0; d]; self.n_components];
                self.eigenvalues = vec![0.0; self.n_components];
                d
            }
            Some(d) => {
                assert_eq!(len, d, "CCIPCA: expected {} features, got {}", d, len);
                d
            }
        }
    }

    /// Validate feature length against established dimensionality.
    fn check_features_len(&self, len: usize) -> usize {
        let d = self
            .n_features
            .expect("CCIPCA: n_features not set (call update first)");
        assert_eq!(len, d, "CCIPCA: expected {} features, got {}", d, len);
        d
    }

    /// Sort components by eigenvalue in descending order.
    fn sort_components(&mut self) {
        // Simple insertion sort -- n_components is typically very small.
        let k = self.n_components;
        for i in 1..k {
            let mut j = i;
            while j > 0 && self.eigenvalues[j] > self.eigenvalues[j - 1] {
                self.eigenvalues.swap(j, j - 1);
                self.components.swap(j, j - 1);
                j -= 1;
            }
        }
    }
}

impl Default for CCIPCA {
    /// Default CCIPCA with 2 components and no forgetting.
    fn default() -> Self {
        Self::new(2)
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl crate::pipeline::StreamingPreprocessor for CCIPCA {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        CCIPCA::update_and_transform(self, features)
    }

    fn transform(&self, features: &[f64]) -> Vec<f64> {
        CCIPCA::transform(self, features)
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.n_components)
    }

    fn reset(&mut self) {
        CCIPCA::reset(self);
    }
}

// ---------------------------------------------------------------------------
// Vector utilities (no external dependency needed for these basics)
// ---------------------------------------------------------------------------

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Euclidean norm of a slice.
#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::StreamingPreprocessor;

    const EPS: f64 = 1e-6;

    #[test]
    fn construction_and_initial_state() {
        let pca = CCIPCA::new(3);
        assert_eq!(pca.n_components(), 3);
        assert_eq!(pca.n_features(), None);
        assert_eq!(pca.count(), 0);
        assert!(pca.eigenvalues().is_empty());

        let pca2 = CCIPCA::with_amnestic(5, 2.0);
        assert_eq!(pca2.n_components(), 5);
        assert_eq!(pca2.count(), 0);

        let pca3 = CCIPCA::default();
        assert_eq!(pca3.n_components(), 2);
    }

    #[test]
    fn lazy_init_sets_n_features() {
        let mut pca = CCIPCA::new(2);
        assert_eq!(pca.n_features(), None);
        pca.update(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(pca.n_features(), Some(4));
        assert_eq!(pca.count(), 1);
    }

    #[test]
    fn known_principal_components() {
        // Feature 0 has high variance (range [-10, 10]),
        // Feature 1 has low variance (range [-1, 1]).
        // The first principal component should point approximately along axis 0.
        let mut pca = CCIPCA::new(2);

        for i in 0..500 {
            // Deterministic pseudo-data: feature 0 in [-10, 10], feature 1 in [-1, 1].
            let x0 = ((i * 7 + 13) % 200) as f64 / 10.0 - 10.0;
            let x1 = ((i * 3 + 7) % 200) as f64 / 100.0 - 1.0;
            pca.update(&[x0, x1]);
        }

        assert_eq!(pca.count(), 500);

        // First component should be approximately aligned with axis 0.
        // Check that the absolute value of the first component's x-coordinate
        // is close to 1, and the y-coordinate is close to 0.
        let c0 = &pca.components[0];
        assert!(
            c0[0].abs() > 0.9,
            "first component x should be ~1.0, got {}",
            c0[0]
        );
        assert!(
            c0[1].abs() < 0.3,
            "first component y should be ~0.0, got {}",
            c0[1]
        );

        // Eigenvalue of first component should be larger than second.
        assert!(
            pca.eigenvalues()[0] > pca.eigenvalues()[1],
            "eigenvalue[0] = {}, eigenvalue[1] = {}",
            pca.eigenvalues()[0],
            pca.eigenvalues()[1]
        );

        // Explained variance ratio should sum to 1.
        let ratios = pca.explained_variance_ratio();
        let total: f64 = ratios.iter().sum();
        assert!(
            (total - 1.0).abs() < EPS,
            "explained variance ratios sum to {}, expected 1.0",
            total
        );
    }

    #[test]
    fn components_are_orthogonal() {
        let mut pca = CCIPCA::new(3);

        // Feed 3D data with distinct variance along each axis.
        for i in 0..1000 {
            let x0 = ((i * 7 + 13) % 400) as f64 / 20.0 - 10.0; // range [-10, 10]
            let x1 = ((i * 3 + 7) % 200) as f64 / 50.0 - 2.0; // range [-2, 2]
            let x2 = ((i * 11 + 3) % 100) as f64 / 100.0 - 0.5; // range [-0.5, 0.5]
            pca.update(&[x0, x1, x2]);
        }

        // Check pairwise orthogonality: dot(c_i, c_j) should be close to 0.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let d = dot(&pca.components[i], &pca.components[j]);
                assert!(
                    d.abs() < 0.1,
                    "dot(component[{}], component[{}]) = {}, expected ~0",
                    i,
                    j,
                    d
                );
            }
        }

        // Each component should be approximately unit-length.
        for i in 0..3 {
            let n = norm(&pca.components[i]);
            assert!(
                (n - 1.0).abs() < EPS,
                "||component[{}]|| = {}, expected ~1.0",
                i,
                n
            );
        }
    }

    #[test]
    fn dimension_reduction() {
        let mut pca = CCIPCA::new(2);

        // Feed 5D data.
        for i in 0..100 {
            let x = [
                i as f64,
                (i as f64) * 0.5,
                (i as f64) * 0.1,
                (i as f64) * 0.01,
                (i as f64) * 0.001,
            ];
            pca.update(&x);
        }

        // Transform should produce exactly n_components outputs.
        let reduced = pca.transform(&[50.0, 25.0, 5.0, 0.5, 0.05]);
        assert_eq!(reduced.len(), 2);
        assert!(reduced[0].is_finite());
        assert!(reduced[1].is_finite());

        // update_and_transform should also produce n_components outputs.
        let reduced2 = pca.update_and_transform(&[51.0, 25.5, 5.1, 0.51, 0.051]);
        assert_eq!(reduced2.len(), 2);
    }

    #[test]
    fn streaming_preprocessor_trait() {
        let mut pca = CCIPCA::new(2);

        // Feed some data to initialise.
        for i in 0..50 {
            pca.update(&[i as f64, (i as f64) * 2.0, (i as f64) * 3.0]);
        }

        // Use as Box<dyn StreamingPreprocessor>.
        let mut boxed: Box<dyn StreamingPreprocessor> = Box::new(pca);

        assert_eq!(boxed.output_dim(), Some(2));

        let out = boxed.update_and_transform(&[25.0, 50.0, 75.0]);
        assert_eq!(out.len(), 2);

        let out2 = boxed.transform(&[25.0, 50.0, 75.0]);
        assert_eq!(out2.len(), 2);

        boxed.reset();
        assert_eq!(boxed.output_dim(), Some(2));
    }

    #[test]
    fn reset_clears_state() {
        let mut pca = CCIPCA::with_amnestic(3, 1.0);

        for i in 0..100 {
            pca.update(&[i as f64, (i as f64) * 0.5]);
        }

        assert_eq!(pca.count(), 100);
        assert_eq!(pca.n_features(), Some(2));

        pca.reset();

        assert_eq!(pca.count(), 0);
        assert_eq!(pca.n_features(), None);
        assert!(pca.eigenvalues().is_empty());
        // n_components and amnestic are preserved.
        assert_eq!(pca.n_components(), 3);

        // Can re-initialise with a different dimensionality after reset.
        pca.update(&[1.0, 2.0, 3.0]);
        assert_eq!(pca.n_features(), Some(3));
        assert_eq!(pca.count(), 1);
    }
}
