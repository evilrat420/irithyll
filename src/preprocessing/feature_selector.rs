//! EWMA-based online feature selector with dynamic importance masking.
//!
//! [`OnlineFeatureSelector`] tracks per-feature importance scores via an
//! Exponentially Weighted Moving Average (EWMA). After a configurable warmup
//! period, a mask is recomputed on each update: only the top `keep_fraction`
//! of features remain active; the rest are zeroed out.
//!
//! This is useful for streaming models that receive high-dimensional feature
//! vectors but want to suppress noise features in real time without a full
//! batch feature-selection pass.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::OnlineFeatureSelector;
//!
//! let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.3, 5);
//!
//! // During warmup all features pass through.
//! for _ in 0..5 {
//!     sel.update_importances(&[0.9, 0.1, 0.8, 0.2]);
//! }
//!
//! // After warmup the mask activates -- only top 50% survive.
//! sel.update_importances(&[0.9, 0.1, 0.8, 0.2]);
//! let masked = sel.mask_features(&[10.0, 20.0, 30.0, 40.0]);
//!
//! // Features 0 and 2 are important; 1 and 3 are zeroed.
//! assert!(masked[0] > 0.0);
//! assert_eq!(masked[1], 0.0);
//! assert!(masked[2] > 0.0);
//! assert_eq!(masked[3], 0.0);
//! ```

/// Online feature selector using EWMA importance tracking.
///
/// After each call to [`update_importances`](Self::update_importances), the
/// internal EWMA is updated and -- once past the warmup period -- a boolean
/// mask is recomputed. Features whose smoothed importance falls below the
/// threshold are suppressed (zeroed) by [`mask_features`](Self::mask_features).
///
/// The threshold is chosen so that exactly `ceil(n_features * keep_fraction)`
/// features remain active. Ties at the boundary are broken in favour of
/// lower-indexed features.
#[derive(Clone, Debug)]
pub struct OnlineFeatureSelector {
    /// Per-feature EWMA of importance scores.
    importance_ewma: Vec<f64>,
    /// EWMA smoothing factor in `(0, 1]`. Higher means more weight on recent.
    alpha: f64,
    /// Fraction of features to keep active, in `(0, 1]`.
    keep_fraction: f64,
    /// Number of updates before the mask activates.
    warmup: u64,
    /// Per-feature active/inactive mask. All `true` during warmup.
    mask: Vec<bool>,
    /// Total calls to [`update_importances`](Self::update_importances).
    n_updates: u64,
}

impl OnlineFeatureSelector {
    /// Create a new feature selector.
    ///
    /// # Arguments
    ///
    /// * `n_features` -- Dimensionality of the feature vector.
    /// * `keep_fraction` -- Fraction of features to keep (e.g. `0.5` keeps 50%).
    ///   Clamped to `(0.0, 1.0]`.
    /// * `alpha` -- EWMA smoothing factor. Closer to 1.0 reacts faster to
    ///   recent importance scores; closer to 0.0 is more stable.
    /// * `warmup` -- Number of [`update_importances`](Self::update_importances)
    ///   calls before the mask activates. During warmup all features pass.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::preprocessing::OnlineFeatureSelector;
    /// let sel = OnlineFeatureSelector::new(10, 0.3, 0.1, 20);
    /// assert_eq!(sel.n_features(), 10);
    /// assert_eq!(sel.active_count(), 10); // all active before warmup
    /// ```
    pub fn new(n_features: usize, keep_fraction: f64, alpha: f64, warmup: u64) -> Self {
        let keep_fraction = keep_fraction.clamp(f64::MIN_POSITIVE, 1.0);
        let alpha = alpha.clamp(f64::MIN_POSITIVE, 1.0);

        Self {
            importance_ewma: vec![0.0; n_features],
            alpha,
            keep_fraction,
            warmup,
            mask: vec![true; n_features],
            n_updates: 0,
        }
    }

    /// Feed new importance scores and recompute the mask (if past warmup).
    ///
    /// The EWMA for each feature is updated as:
    ///
    /// ```text
    /// ewma_i = alpha * importance_i + (1 - alpha) * ewma_i
    /// ```
    ///
    /// If `n_updates >= warmup` **after** this update, the mask is recomputed
    /// by sorting importances and keeping the top `ceil(n * keep_fraction)`.
    ///
    /// # Panics
    ///
    /// Panics if `importances.len() != n_features`.
    pub fn update_importances(&mut self, importances: &[f64]) {
        let n = self.importance_ewma.len();
        assert_eq!(
            importances.len(),
            n,
            "OnlineFeatureSelector: expected {} importances, got {}",
            n,
            importances.len()
        );

        // Update EWMA.
        for (ewma, &imp) in self.importance_ewma.iter_mut().zip(importances.iter()) {
            *ewma = self.alpha * imp + (1.0 - self.alpha) * *ewma;
        }

        self.n_updates += 1;

        // Recompute mask once warmup is complete.
        if self.n_updates >= self.warmup {
            self.recompute_mask();
        }
    }

    /// Apply the current mask to a feature vector, zeroing inactive features.
    ///
    /// Returns a new `Vec<f64>` with the same length as the input. Active
    /// features are passed through; inactive features become `0.0`.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != n_features`.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::preprocessing::OnlineFeatureSelector;
    /// let sel = OnlineFeatureSelector::new(3, 1.0, 0.5, 0);
    /// let masked = sel.mask_features(&[1.0, 2.0, 3.0]);
    /// assert_eq!(masked, vec![1.0, 2.0, 3.0]); // all kept before any updates
    /// ```
    pub fn mask_features(&self, features: &[f64]) -> Vec<f64> {
        let n = self.importance_ewma.len();
        assert_eq!(
            features.len(),
            n,
            "OnlineFeatureSelector: expected {} features, got {}",
            n,
            features.len()
        );

        let mut out = Vec::with_capacity(n);
        for (&m, &f) in self.mask.iter().zip(features.iter()) {
            out.push(if m { f } else { 0.0 });
        }
        out
    }

    /// Apply the current mask in place, zeroing inactive features.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != n_features`.
    pub fn mask_features_in_place(&self, features: &mut [f64]) {
        let n = self.importance_ewma.len();
        assert_eq!(
            features.len(),
            n,
            "OnlineFeatureSelector: expected {} features, got {}",
            n,
            features.len()
        );

        for (f, &m) in features.iter_mut().zip(self.mask.iter()) {
            if !m {
                *f = 0.0;
            }
        }
    }

    /// Current per-feature active mask. `true` means the feature passes.
    pub fn active_features(&self) -> &[bool] {
        &self.mask
    }

    /// Number of currently active (unmasked) features.
    pub fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }

    /// Total number of features this selector was configured for.
    pub fn n_features(&self) -> usize {
        self.importance_ewma.len()
    }

    /// Reset all EWMA scores, masks, and update counter.
    ///
    /// After reset, all features are active and warmup restarts.
    pub fn reset(&mut self) {
        self.importance_ewma.fill(0.0);
        self.mask.fill(true);
        self.n_updates = 0;
    }

    /// Recompute the boolean mask from current EWMA scores.
    ///
    /// Strategy: keep the top `ceil(n * keep_fraction)` features by EWMA.
    /// Sort a copy of EWMA values descending, pick the threshold at position
    /// `k - 1` (the k-th largest), and activate features >= threshold.
    ///
    /// When ties exist at the boundary, features with lower indices are
    /// preferred to maintain determinism.
    fn recompute_mask(&mut self) {
        let n = self.importance_ewma.len();
        if n == 0 {
            return;
        }

        let k = (n as f64 * self.keep_fraction).ceil() as usize;
        let k = k.clamp(1, n);

        // Sort importances descending to find the k-th largest.
        let mut sorted: Vec<f64> = self.importance_ewma.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];

        // First pass: activate everything strictly above threshold.
        let mut active_count = 0;
        for i in 0..n {
            if self.importance_ewma[i] > threshold {
                self.mask[i] = true;
                active_count += 1;
            } else {
                self.mask[i] = false;
            }
        }

        // Second pass: fill remaining slots from features exactly at threshold
        // (lower indices first for determinism).
        for i in 0..n {
            if active_count >= k {
                break;
            }
            if !self.mask[i] && (self.importance_ewma[i] - threshold).abs() < f64::EPSILON {
                self.mask[i] = true;
                active_count += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_active_during_warmup() {
        let mut sel = OnlineFeatureSelector::new(4, 0.25, 0.5, 10);

        // Feed importances that would eliminate features if past warmup.
        for _ in 0..9 {
            sel.update_importances(&[1.0, 0.0, 0.0, 0.0]);
        }

        // Still in warmup (9 < 10) -- all features should be active.
        assert_eq!(sel.active_count(), 4);
        assert!(sel.active_features().iter().all(|&b| b));
    }

    #[test]
    fn masking_activates_after_warmup() {
        let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.9, 3);

        // Warmup: 3 updates.
        for _ in 0..3 {
            sel.update_importances(&[0.9, 0.1, 0.8, 0.05]);
        }

        // Past warmup -- mask should now be active.
        // keep_fraction = 0.5 => ceil(4 * 0.5) = 2 features kept.
        assert_eq!(sel.active_count(), 2);

        // Features 0 and 2 should be active (highest importance).
        assert!(sel.active_features()[0]);
        assert!(!sel.active_features()[1]);
        assert!(sel.active_features()[2]);
        assert!(!sel.active_features()[3]);
    }

    #[test]
    fn keep_fraction_determines_active_count() {
        // Keep 75% of 8 features = ceil(6.0) = 6.
        let mut sel = OnlineFeatureSelector::new(8, 0.75, 0.9, 1);
        sel.update_importances(&[0.8, 0.7, 0.1, 0.05, 0.6, 0.5, 0.9, 0.4]);
        assert_eq!(sel.active_count(), 6);
    }

    #[test]
    fn mask_features_zeros_inactive() {
        let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.9, 1);
        sel.update_importances(&[1.0, 0.0, 1.0, 0.0]);

        let masked = sel.mask_features(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(masked[0], 10.0);
        assert_eq!(masked[1], 0.0);
        assert_eq!(masked[2], 30.0);
        assert_eq!(masked[3], 0.0);
    }

    #[test]
    fn importance_ewma_update_smooths_correctly() {
        let mut sel = OnlineFeatureSelector::new(1, 1.0, 0.5, 100);

        // With alpha = 0.5:
        // After [1.0]: ewma = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        sel.update_importances(&[1.0]);
        assert!((sel.importance_ewma[0] - 0.5).abs() < 1e-12);

        // After [1.0]: ewma = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        sel.update_importances(&[1.0]);
        assert!((sel.importance_ewma[0] - 0.75).abs() < 1e-12);

        // After [0.0]: ewma = 0.5 * 0.0 + 0.5 * 0.75 = 0.375
        sel.update_importances(&[0.0]);
        assert!((sel.importance_ewma[0] - 0.375).abs() < 1e-12);
    }

    #[test]
    fn full_keep_no_mask() {
        let mut sel = OnlineFeatureSelector::new(5, 1.0, 0.5, 1);
        sel.update_importances(&[0.1, 0.2, 0.3, 0.4, 0.5]);

        // keep_fraction = 1.0 => all features active.
        assert_eq!(sel.active_count(), 5);
        assert!(sel.active_features().iter().all(|&b| b));
    }

    #[test]
    fn active_count_matches_mask() {
        let mut sel = OnlineFeatureSelector::new(6, 0.5, 0.9, 1);
        sel.update_importances(&[0.9, 0.1, 0.8, 0.05, 0.7, 0.02]);

        // ceil(6 * 0.5) = 3 features.
        let count = sel.active_count();
        let mask_count = sel.active_features().iter().filter(|&&b| b).count();
        assert_eq!(count, mask_count);
        assert_eq!(count, 3);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.9, 2);

        // Advance past warmup and establish a mask.
        sel.update_importances(&[1.0, 0.0, 1.0, 0.0]);
        sel.update_importances(&[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(sel.active_count(), 2);

        sel.reset();
        assert_eq!(sel.active_count(), 4); // all active after reset
        assert!(sel.active_features().iter().all(|&b| b));
        assert!(sel.importance_ewma.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn mask_features_in_place_matches_mask_features() {
        let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.9, 1);
        sel.update_importances(&[0.9, 0.1, 0.8, 0.05]);

        let input = [10.0, 20.0, 30.0, 40.0];
        let alloc = sel.mask_features(&input);

        let mut inplace = input;
        sel.mask_features_in_place(&mut inplace);

        for (a, b) in alloc.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    #[should_panic(expected = "expected 4 importances, got 2")]
    fn panics_on_importance_length_mismatch() {
        let mut sel = OnlineFeatureSelector::new(4, 0.5, 0.5, 1);
        sel.update_importances(&[1.0, 2.0]);
    }
}
