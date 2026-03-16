//! Streaming AUC-ROC (Area Under the Receiver Operating Characteristic curve).
//!
//! Approximates AUC over a fixed-size sliding window of recent predictions using
//! the Wilcoxon-Mann-Whitney U statistic. When the window is full, the oldest
//! observation is evicted before a new one is added.
//!
//! # Algorithm
//!
//! AUC-ROC is equivalent to the probability that a randomly chosen positive
//! example receives a higher predicted score than a randomly chosen negative
//! example. The Wilcoxon-Mann-Whitney formulation computes this directly:
//!
//! ```text
//! U = sum over all (pos, neg) pairs of:
//!       1.0  if pos_score > neg_score
//!       0.5  if pos_score == neg_score  (tie)
//!       0.0  if pos_score < neg_score
//!
//! AUC = U / (n_pos * n_neg)
//! ```
//!
//! This is O(n^2) where n is the number of samples in the window. For typical
//! window sizes (up to ~10K) this is fast enough. If you need larger windows,
//! consider a sorted-insertion approach.

use std::collections::VecDeque;

/// Streaming approximation of AUC-ROC over a fixed-size sliding window.
///
/// Maintains a circular buffer of `(predicted_score, is_positive)` pairs. When
/// the buffer reaches capacity, the oldest entry is evicted. AUC is computed
/// on demand via the Wilcoxon-Mann-Whitney U statistic over the current window.
///
/// # Example
///
/// ```
/// use irithyll::metrics::auc::StreamingAUC;
///
/// let mut auc = StreamingAUC::new(1000);
///
/// // Perfect classifier: positives scored higher than negatives
/// auc.update(0.9, true);
/// auc.update(0.8, true);
/// auc.update(0.2, false);
/// auc.update(0.1, false);
///
/// assert_eq!(auc.auc(), Some(1.0));
/// assert_eq!(auc.window_count(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingAUC {
    window_size: usize,
    buffer: VecDeque<(f64, bool)>,
    n_samples_total: u64,
}

impl StreamingAUC {
    /// Create a new streaming AUC tracker with the given window size.
    ///
    /// Larger windows give more accurate AUC estimates but use more memory
    /// and take longer to compute (the `auc()` method is O(n^2) over the
    /// window). Window sizes up to ~10K are practical.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size + 1),
            n_samples_total: 0,
        }
    }

    /// Update with a new prediction.
    ///
    /// - `score`: the model's predicted probability or confidence for the
    ///   positive class (higher = more likely positive).
    /// - `is_positive`: whether the true label is positive (class 1).
    ///
    /// If the window is full, the oldest observation is evicted first.
    pub fn update(&mut self, score: f64, is_positive: bool) {
        if self.buffer.len() == self.window_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back((score, is_positive));
        self.n_samples_total += 1;
    }

    /// Compute AUC-ROC over the current window using the Wilcoxon-Mann-Whitney
    /// U statistic.
    ///
    /// For every (positive, negative) pair in the window, counts 1.0 if the
    /// positive score exceeds the negative score, 0.5 for ties, and 0.0
    /// otherwise. The final AUC is:
    ///
    /// ```text
    /// AUC = U / (n_pos * n_neg)
    /// ```
    ///
    /// # Returns
    ///
    /// - `None` if the buffer is empty, or if all samples belong to the same
    ///   class (AUC is undefined without both positive and negative examples).
    /// - `Some(1.0)` for perfect separation (every positive scored higher than
    ///   every negative).
    /// - `Some(0.0)` for perfectly inverted predictions (every negative scored
    ///   higher than every positive).
    /// - Values near `0.5` indicate random-quality predictions.
    ///
    /// # Complexity
    ///
    /// O(n^2) where n = `window_count()`. This is acceptable for window sizes
    /// up to roughly 10,000 samples.
    pub fn auc(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            return None;
        }

        // Separate positive and negative scores
        let mut pos_scores: Vec<f64> = Vec::new();
        let mut neg_scores: Vec<f64> = Vec::new();

        for &(score, is_positive) in &self.buffer {
            if is_positive {
                pos_scores.push(score);
            } else {
                neg_scores.push(score);
            }
        }

        let n_pos = pos_scores.len();
        let n_neg = neg_scores.len();

        if n_pos == 0 || n_neg == 0 {
            return None;
        }

        // Wilcoxon-Mann-Whitney U statistic
        let mut u = 0.0_f64;
        for &ps in &pos_scores {
            for &ns in &neg_scores {
                if ps > ns {
                    u += 1.0;
                } else if (ps - ns).abs() < f64::EPSILON {
                    u += 0.5;
                }
            }
        }

        Some(u / (n_pos as f64 * n_neg as f64))
    }

    /// Number of samples currently in the window.
    pub fn window_count(&self) -> usize {
        self.buffer.len()
    }

    /// Total number of samples seen since creation (including evicted ones).
    pub fn n_samples(&self) -> u64 {
        self.n_samples_total
    }

    /// Number of positive samples in the current window.
    pub fn n_positive(&self) -> usize {
        self.buffer.iter().filter(|(_, is_pos)| *is_pos).count()
    }

    /// Number of negative samples in the current window.
    pub fn n_negative(&self) -> usize {
        self.buffer.iter().filter(|(_, is_pos)| !*is_pos).count()
    }

    /// Whether the window is completely filled.
    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.window_size
    }

    /// The configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Reset all state: clear the buffer and zero the total sample count.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.n_samples_total = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn perfect_classifier_auc_is_one() {
        let mut auc = StreamingAUC::new(100);
        // All positives scored higher than all negatives
        auc.update(0.95, true);
        auc.update(0.90, true);
        auc.update(0.85, true);
        auc.update(0.20, false);
        auc.update(0.15, false);
        auc.update(0.10, false);
        assert_eq!(auc.auc(), Some(1.0));
    }

    #[test]
    fn perfectly_wrong_classifier_auc_is_zero() {
        let mut auc = StreamingAUC::new(100);
        // All negatives scored higher than all positives
        auc.update(0.10, true);
        auc.update(0.15, true);
        auc.update(0.85, false);
        auc.update(0.90, false);
        assert_eq!(auc.auc(), Some(0.0));
    }

    #[test]
    fn random_classifier_auc_near_half() {
        let mut auc = StreamingAUC::new(100);
        // Interleaved scores: pos at 0.1, 0.3, 0.5, 0.7, 0.9
        //                     neg at 0.2, 0.4, 0.6, 0.8
        // Pairs (pos, neg) where pos > neg:
        //   0.1 > {}: 0 wins
        //   0.3 > {0.2}: 1 win
        //   0.5 > {0.2, 0.4}: 2 wins
        //   0.7 > {0.2, 0.4, 0.6}: 3 wins
        //   0.9 > {0.2, 0.4, 0.6, 0.8}: 4 wins
        // Total U = 0 + 1 + 2 + 3 + 4 = 10
        // n_pos * n_neg = 5 * 4 = 20
        // AUC = 10/20 = 0.5
        auc.update(0.1, true);
        auc.update(0.2, false);
        auc.update(0.3, true);
        auc.update(0.4, false);
        auc.update(0.5, true);
        auc.update(0.6, false);
        auc.update(0.7, true);
        auc.update(0.8, false);
        auc.update(0.9, true);

        assert!(approx_eq(auc.auc().unwrap(), 0.5));
    }

    #[test]
    fn known_small_example_exact_auc() {
        // Hand-computed example:
        // Positives: [0.8, 0.6]  Negatives: [0.4, 0.3, 0.7]
        //
        // Pairs:
        //   0.8 vs 0.4 -> win (1.0)
        //   0.8 vs 0.3 -> win (1.0)
        //   0.8 vs 0.7 -> win (1.0)
        //   0.6 vs 0.4 -> win (1.0)
        //   0.6 vs 0.3 -> win (1.0)
        //   0.6 vs 0.7 -> loss (0.0)
        //
        // U = 5.0, n_pos * n_neg = 2 * 3 = 6
        // AUC = 5/6 = 0.8333...
        let mut auc = StreamingAUC::new(100);
        auc.update(0.8, true);
        auc.update(0.4, false);
        auc.update(0.6, true);
        auc.update(0.3, false);
        auc.update(0.7, false);

        let expected = 5.0 / 6.0;
        assert!(approx_eq(auc.auc().unwrap(), expected));
    }

    #[test]
    fn ties_handled_correctly() {
        // Positive: [0.5], Negative: [0.5]
        // Tie => 0.5
        // AUC = 0.5 / (1*1) = 0.5
        let mut auc = StreamingAUC::new(100);
        auc.update(0.5, true);
        auc.update(0.5, false);
        assert!(approx_eq(auc.auc().unwrap(), 0.5));
    }

    #[test]
    fn ties_mixed_with_clear_pairs() {
        // Positives: [0.7, 0.5]  Negatives: [0.5, 0.3]
        //
        // Pairs:
        //   0.7 vs 0.5 -> win (1.0)
        //   0.7 vs 0.3 -> win (1.0)
        //   0.5 vs 0.5 -> tie (0.5)
        //   0.5 vs 0.3 -> win (1.0)
        //
        // U = 3.5, n_pos * n_neg = 2 * 2 = 4
        // AUC = 3.5 / 4 = 0.875
        let mut auc = StreamingAUC::new(100);
        auc.update(0.7, true);
        auc.update(0.5, false);
        auc.update(0.5, true);
        auc.update(0.3, false);

        assert!(approx_eq(auc.auc().unwrap(), 0.875));
    }

    #[test]
    fn empty_buffer_returns_none() {
        let auc = StreamingAUC::new(100);
        assert_eq!(auc.auc(), None);
    }

    #[test]
    fn single_sample_returns_none() {
        let mut auc = StreamingAUC::new(100);
        auc.update(0.9, true);
        assert_eq!(auc.auc(), None);
    }

    #[test]
    fn all_positive_returns_none() {
        let mut auc = StreamingAUC::new(100);
        auc.update(0.9, true);
        auc.update(0.8, true);
        auc.update(0.7, true);
        assert_eq!(auc.auc(), None);
    }

    #[test]
    fn all_negative_returns_none() {
        let mut auc = StreamingAUC::new(100);
        auc.update(0.3, false);
        auc.update(0.2, false);
        assert_eq!(auc.auc(), None);
    }

    #[test]
    fn window_eviction_drops_old_samples() {
        let mut auc = StreamingAUC::new(3);
        auc.update(0.9, true); // will be evicted
        auc.update(0.1, false); // will be evicted
        auc.update(0.8, true); // will be evicted
        assert_eq!(auc.window_count(), 3);
        assert!(auc.is_full());

        // Now push 3 more: all negatives scored higher than positives
        auc.update(0.1, true);
        auc.update(0.8, false);
        auc.update(0.9, false);

        // Window is now: [(0.1, true), (0.8, false), (0.9, false)]
        // The old perfect separation is gone
        assert_eq!(auc.window_count(), 3);
        assert_eq!(auc.n_samples(), 6);
        assert_eq!(auc.n_positive(), 1);
        assert_eq!(auc.n_negative(), 2);
        // AUC = 0/(1*2) = 0.0 (positive at 0.1 loses to both negatives)
        assert_eq!(auc.auc(), Some(0.0));
    }

    #[test]
    fn n_samples_tracks_total_including_evicted() {
        let mut auc = StreamingAUC::new(2);
        auc.update(0.9, true);
        auc.update(0.1, false);
        assert_eq!(auc.n_samples(), 2);
        assert_eq!(auc.window_count(), 2);

        auc.update(0.5, true);
        assert_eq!(auc.n_samples(), 3);
        assert_eq!(auc.window_count(), 2);

        auc.update(0.4, false);
        assert_eq!(auc.n_samples(), 4);
        assert_eq!(auc.window_count(), 2);
    }

    #[test]
    fn reset_clears_everything() {
        let mut auc = StreamingAUC::new(100);
        auc.update(0.9, true);
        auc.update(0.1, false);
        auc.update(0.8, true);

        auc.reset();
        assert_eq!(auc.window_count(), 0);
        assert_eq!(auc.n_samples(), 0);
        assert_eq!(auc.n_positive(), 0);
        assert_eq!(auc.n_negative(), 0);
        assert!(!auc.is_full());
        assert_eq!(auc.auc(), None);
    }

    #[test]
    fn window_size_accessor() {
        let auc = StreamingAUC::new(42);
        assert_eq!(auc.window_size(), 42);
    }

    #[test]
    fn is_full_transitions() {
        let mut auc = StreamingAUC::new(2);
        assert!(!auc.is_full());
        auc.update(0.5, true);
        assert!(!auc.is_full());
        auc.update(0.5, false);
        assert!(auc.is_full());
        // Still full after eviction + insert
        auc.update(0.7, true);
        assert!(auc.is_full());
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn zero_window_panics() {
        StreamingAUC::new(0);
    }

    #[test]
    fn positive_negative_counts() {
        let mut auc = StreamingAUC::new(100);
        auc.update(0.9, true);
        auc.update(0.8, true);
        auc.update(0.3, false);
        assert_eq!(auc.n_positive(), 2);
        assert_eq!(auc.n_negative(), 1);
        assert_eq!(auc.window_count(), 3);
    }

    #[test]
    fn window_eviction_updates_class_counts() {
        let mut auc = StreamingAUC::new(2);
        auc.update(0.9, true);
        auc.update(0.1, false);
        assert_eq!(auc.n_positive(), 1);
        assert_eq!(auc.n_negative(), 1);

        // Evict (0.9, true), add (0.3, false)
        auc.update(0.3, false);
        assert_eq!(auc.n_positive(), 0);
        assert_eq!(auc.n_negative(), 2);
        assert_eq!(auc.auc(), None); // all negative
    }
}
