//! Online Kappa statistics for concept-drift-aware classification evaluation.
//!
//! Three streaming Kappa metrics are provided, each comparing a model's
//! accuracy against a different baseline classifier:
//!
//! - [`CohenKappa`] — compares against chance agreement (random classifier).
//! - [`KappaM`] — compares against a majority-class classifier.
//! - [`KappaT`] — compares against a no-change (temporal) classifier.
//!
//! All metrics are maintained incrementally with O(1) per-update cost
//! (amortized for [`CohenKappa`] when new classes appear). No past samples
//! are stored.

// ---------------------------------------------------------------------------
// CohenKappa
// ---------------------------------------------------------------------------

/// Incrementally computed Cohen's Kappa statistic.
///
/// Measures agreement between predictions and true labels, adjusted for
/// the probability of chance agreement. A dynamically sized confusion
/// matrix grows automatically as new class labels are observed.
///
/// # Formula
///
/// ```text
/// κ = (p_o - p_e) / (1 - p_e)
/// ```
///
/// where
/// - `p_o` = observed agreement (fraction of samples where prediction == target)
/// - `p_e` = expected (chance) agreement, computed from the marginal
///   distributions of the confusion matrix:
///   `p_e = Σ_k (row_sum_k * col_sum_k) / n²`
///
/// # Value range
///
/// - `κ = 1.0` — perfect agreement
/// - `κ = 0.0` — agreement equal to chance
/// - `κ < 0.0` — agreement worse than chance
///
/// # Example
///
/// ```
/// use irithyll::CohenKappa;
///
/// let mut k = CohenKappa::new();
/// k.update(0, 0);
/// k.update(1, 1);
/// assert_eq!(k.kappa(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct CohenKappa {
    /// Confusion matrix: `matrix[true_label][predicted_label]`.
    matrix: Vec<Vec<u64>>,
    /// Total number of samples observed.
    n: u64,
}

impl CohenKappa {
    /// Create a new, empty Cohen's Kappa tracker.
    pub fn new() -> Self {
        Self {
            matrix: Vec::new(),
            n: 0,
        }
    }

    /// Ensure the confusion matrix is large enough to hold `size` classes.
    fn grow(&mut self, size: usize) {
        let old = self.matrix.len();
        if size <= old {
            return;
        }
        // Extend existing rows.
        for row in &mut self.matrix {
            row.resize(size, 0);
        }
        // Add new rows.
        for _ in old..size {
            self.matrix.push(vec![0; size]);
        }
    }

    /// Update the confusion matrix with a single observation.
    ///
    /// - `true_label`: the ground-truth class index.
    /// - `predicted_label`: the model's predicted class index.
    ///
    /// The confusion matrix grows automatically if either label exceeds
    /// the current number of tracked classes.
    pub fn update(&mut self, true_label: usize, predicted_label: usize) {
        let required = true_label.max(predicted_label) + 1;
        self.grow(required);
        self.matrix[true_label][predicted_label] += 1;
        self.n += 1;
    }

    /// Cohen's Kappa statistic.
    ///
    /// Returns 0.0 when no samples have been observed or when `p_e == 1.0`
    /// (degenerate case where chance agreement is certain).
    pub fn kappa(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        let n = self.n as f64;
        let k = self.matrix.len();

        // Observed agreement: fraction of diagonal entries.
        let mut correct = 0u64;
        for i in 0..k {
            correct += self.matrix[i][i];
        }
        let p_o = correct as f64 / n;

        // Expected (chance) agreement from marginal distributions.
        let mut p_e = 0.0;
        for i in 0..k {
            let row_sum: u64 = self.matrix[i].iter().sum();
            let col_sum: u64 = self.matrix.iter().map(|row| row[i]).sum();
            p_e += (row_sum as f64) * (col_sum as f64);
        }
        p_e /= n * n;

        if (1.0 - p_e).abs() < f64::EPSILON {
            return 0.0;
        }

        (p_o - p_e) / (1.0 - p_e)
    }

    /// Number of samples observed so far.
    pub fn n_samples(&self) -> u64 {
        self.n
    }

    /// Reset all state to the initial empty condition.
    pub fn reset(&mut self) {
        self.matrix.clear();
        self.n = 0;
    }
}

impl Default for CohenKappa {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// KappaM
// ---------------------------------------------------------------------------

/// Incrementally computed Kappa-M statistic.
///
/// Compares the model's accuracy against a majority-class baseline
/// classifier that always predicts the most frequent true label seen so
/// far.
///
/// # Formula
///
/// ```text
/// κ_m = (p_model - p_majority) / (1 - p_majority)
/// ```
///
/// where
/// - `p_model` = model accuracy (fraction correctly predicted)
/// - `p_majority` = accuracy of always predicting the majority class
///   = `max_class_count / n`
///
/// # Value range
///
/// - `κ_m = 1.0` — perfect model
/// - `κ_m = 0.0` — model performs no better than always guessing the majority class
/// - `κ_m < 0.0` — model performs worse than majority-class guessing
///
/// # Example
///
/// ```
/// use irithyll::KappaM;
///
/// let mut k = KappaM::new();
/// k.update(0, 0);
/// k.update(0, 0);
/// k.update(1, 1);
/// assert!(k.kappa_m() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct KappaM {
    /// Per-class counts of true labels. `class_counts[c]` = number of
    /// samples whose true label is `c`.
    class_counts: Vec<u64>,
    /// Total number of samples observed.
    n: u64,
    /// Number of correctly classified samples.
    n_correct: u64,
}

impl KappaM {
    /// Create a new, empty Kappa-M tracker.
    pub fn new() -> Self {
        Self {
            class_counts: Vec::new(),
            n: 0,
            n_correct: 0,
        }
    }

    /// Update metrics with a single observation.
    ///
    /// - `true_label`: the ground-truth class index.
    /// - `predicted_label`: the model's predicted class index.
    pub fn update(&mut self, true_label: usize, predicted_label: usize) {
        // Grow class counts if needed.
        if true_label >= self.class_counts.len() {
            self.class_counts.resize(true_label + 1, 0);
        }
        self.class_counts[true_label] += 1;
        self.n += 1;

        if true_label == predicted_label {
            self.n_correct += 1;
        }
    }

    /// The current majority class (most frequent true label), or `None`
    /// if no samples have been observed.
    ///
    /// Ties are broken by the lowest class index.
    pub fn majority_class(&self) -> Option<usize> {
        if self.n == 0 {
            return None;
        }
        self.class_counts
            .iter()
            .enumerate()
            .max_by(|&(i_a, &count_a), &(i_b, &count_b)| count_a.cmp(&count_b).then(i_b.cmp(&i_a)))
            .map(|(idx, _)| idx)
    }

    /// Kappa-M statistic.
    ///
    /// Returns 0.0 when no samples have been observed or when `p_majority == 1.0`
    /// (only one class has ever appeared, making the baseline perfect).
    pub fn kappa_m(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        let n = self.n as f64;
        let p_model = self.n_correct as f64 / n;

        // Majority-class baseline accuracy = count of most frequent class / n.
        let max_count = self.class_counts.iter().copied().max().unwrap_or(0);
        let p_majority = max_count as f64 / n;

        if (1.0 - p_majority).abs() < f64::EPSILON {
            return 0.0;
        }

        (p_model - p_majority) / (1.0 - p_majority)
    }

    /// Number of samples observed so far.
    pub fn n_samples(&self) -> u64 {
        self.n
    }

    /// Reset all state to the initial empty condition.
    pub fn reset(&mut self) {
        self.class_counts.clear();
        self.n = 0;
        self.n_correct = 0;
    }
}

impl Default for KappaM {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// KappaT
// ---------------------------------------------------------------------------

/// Incrementally computed Kappa-T (temporal Kappa) statistic.
///
/// Compares the model's accuracy against a no-change baseline classifier
/// that always predicts the previous sample's true label. The first
/// sample is skipped for no-change evaluation since there is no prior
/// label.
///
/// # Formula
///
/// ```text
/// κ_t = (p_model - p_nochange) / (1 - p_nochange)
/// ```
///
/// where
/// - `p_model` = model accuracy (fraction correctly predicted)
/// - `p_nochange` = accuracy of the no-change classifier that predicts
///   `y_{t-1}` for sample `t`
///
/// # Value range
///
/// - `κ_t = 1.0` — perfect model
/// - `κ_t = 0.0` — model performs no better than always predicting the last label
/// - `κ_t < 0.0` — model performs worse than the no-change baseline
///
/// # Example
///
/// ```
/// use irithyll::KappaT;
///
/// let mut k = KappaT::new();
/// k.update(0, 0);
/// k.update(1, 1);
/// k.update(1, 1);
/// assert!(k.kappa_t() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct KappaT {
    /// Total number of samples observed (including the first).
    n: u64,
    /// Number of correctly classified samples by the model.
    n_correct: u64,
    /// Number of samples (from the second onward) where the no-change
    /// classifier would have been correct (i.e. `y_t == y_{t-1}`).
    n_nochange_correct: u64,
    /// The true label of the previous sample, or `None` before the
    /// first sample.
    prev_true_label: Option<usize>,
}

impl KappaT {
    /// Create a new, empty Kappa-T tracker.
    pub fn new() -> Self {
        Self {
            n: 0,
            n_correct: 0,
            n_nochange_correct: 0,
            prev_true_label: None,
        }
    }

    /// Update metrics with a single observation.
    ///
    /// - `true_label`: the ground-truth class index.
    /// - `predicted_label`: the model's predicted class index.
    ///
    /// The no-change baseline is evaluated from the second sample onward.
    pub fn update(&mut self, true_label: usize, predicted_label: usize) {
        self.n += 1;

        if true_label == predicted_label {
            self.n_correct += 1;
        }

        // No-change classifier evaluation (from 2nd sample onward).
        if let Some(prev) = self.prev_true_label {
            if true_label == prev {
                self.n_nochange_correct += 1;
            }
        }

        self.prev_true_label = Some(true_label);
    }

    /// Kappa-T statistic.
    ///
    /// Returns 0.0 when fewer than 2 samples have been observed (the
    /// no-change classifier cannot be evaluated) or when
    /// `p_nochange == 1.0` (the baseline is perfect, making comparison
    /// meaningless).
    pub fn kappa_t(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }

        let n = self.n as f64;
        let p_model = self.n_correct as f64 / n;

        // The no-change classifier is evaluated on samples 2..n (total n-1).
        let n_comparable = (self.n - 1) as f64;
        let p_nochange = self.n_nochange_correct as f64 / n_comparable;

        if (1.0 - p_nochange).abs() < f64::EPSILON {
            return 0.0;
        }

        (p_model - p_nochange) / (1.0 - p_nochange)
    }

    /// Number of samples observed so far.
    pub fn n_samples(&self) -> u64 {
        self.n
    }

    /// Reset all state to the initial empty condition.
    pub fn reset(&mut self) {
        self.n = 0;
        self.n_correct = 0;
        self.n_nochange_correct = 0;
        self.prev_true_label = None;
    }
}

impl Default for KappaT {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // -----------------------------------------------------------------------
    // CohenKappa tests
    // -----------------------------------------------------------------------

    #[test]
    fn cohen_empty_returns_zero() {
        let k = CohenKappa::new();
        assert_eq!(k.kappa(), 0.0);
        assert_eq!(k.n_samples(), 0);
    }

    #[test]
    fn cohen_perfect_classifier() {
        let mut k = CohenKappa::new();
        for _ in 0..50 {
            k.update(0, 0);
        }
        for _ in 0..50 {
            k.update(1, 1);
        }
        assert!(approx_eq(k.kappa(), 1.0));
        assert_eq!(k.n_samples(), 100);
    }

    #[test]
    fn cohen_perfect_multiclass() {
        let mut k = CohenKappa::new();
        for class in 0..5 {
            for _ in 0..20 {
                k.update(class, class);
            }
        }
        assert!(approx_eq(k.kappa(), 1.0));
    }

    #[test]
    fn cohen_known_confusion_matrix() {
        // Known example:
        //   Confusion matrix:
        //       pred 0  pred 1
        //   t 0 [  20,    5 ]
        //   t 1 [  10,   15 ]
        //
        //   n = 50
        //   p_o = (20 + 15) / 50 = 0.70
        //   row_sums = [25, 25], col_sums = [30, 20]
        //   p_e = (25*30 + 25*20) / 50^2 = (750 + 500) / 2500 = 0.50
        //   kappa = (0.70 - 0.50) / (1.0 - 0.50) = 0.40
        let mut k = CohenKappa::new();
        for _ in 0..20 {
            k.update(0, 0);
        }
        for _ in 0..5 {
            k.update(0, 1);
        }
        for _ in 0..10 {
            k.update(1, 0);
        }
        for _ in 0..15 {
            k.update(1, 1);
        }
        assert!(approx_eq(k.kappa(), 0.4));
    }

    #[test]
    fn cohen_chance_classifier() {
        // Simulate a classifier that assigns labels independently of truth.
        // With balanced classes and uniform random predictions, kappa ~ 0.
        //
        // Construct exact confusion matrix:
        //       pred 0  pred 1
        //   t 0 [  25,   25 ]
        //   t 1 [  25,   25 ]
        //
        //   n = 100, p_o = 50/100 = 0.50
        //   row_sums = [50, 50], col_sums = [50, 50]
        //   p_e = (50*50 + 50*50) / 10000 = 0.50
        //   kappa = (0.50 - 0.50) / (1.0 - 0.50) = 0.0
        let mut k = CohenKappa::new();
        for _ in 0..25 {
            k.update(0, 0);
        }
        for _ in 0..25 {
            k.update(0, 1);
        }
        for _ in 0..25 {
            k.update(1, 0);
        }
        for _ in 0..25 {
            k.update(1, 1);
        }
        assert!(approx_eq(k.kappa(), 0.0));
    }

    #[test]
    fn cohen_worse_than_chance() {
        // Every prediction is wrong (always predict the opposite class).
        //       pred 0  pred 1
        //   t 0 [   0,   50 ]
        //   t 1 [  50,    0 ]
        //
        //   n = 100, p_o = 0/100 = 0.0
        //   row_sums = [50, 50], col_sums = [50, 50]
        //   p_e = (50*50 + 50*50) / 10000 = 0.50
        //   kappa = (0.0 - 0.50) / (1.0 - 0.50) = -1.0
        let mut k = CohenKappa::new();
        for _ in 0..50 {
            k.update(0, 1);
        }
        for _ in 0..50 {
            k.update(1, 0);
        }
        assert!(approx_eq(k.kappa(), -1.0));
    }

    #[test]
    fn cohen_single_class() {
        // Only one class ever appears. p_e = 1.0 => kappa = 0.0 by convention.
        let mut k = CohenKappa::new();
        for _ in 0..100 {
            k.update(0, 0);
        }
        assert_eq!(k.kappa(), 0.0);
    }

    #[test]
    fn cohen_single_sample() {
        let mut k = CohenKappa::new();
        k.update(0, 0);
        // p_o = 1.0, p_e = 1.0 => 0.0 by degenerate guard
        assert_eq!(k.kappa(), 0.0);
        assert_eq!(k.n_samples(), 1);
    }

    #[test]
    fn cohen_auto_grow() {
        // Classes appear in non-sequential order.
        let mut k = CohenKappa::new();
        k.update(5, 5);
        k.update(2, 2);
        k.update(0, 0);
        assert_eq!(k.n_samples(), 3);
        // Matrix should be at least 6x6.
        assert!(k.matrix.len() >= 6);
    }

    #[test]
    fn cohen_reset() {
        let mut k = CohenKappa::new();
        k.update(0, 0);
        k.update(1, 1);
        k.reset();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa(), 0.0);
        assert!(k.matrix.is_empty());
    }

    #[test]
    fn cohen_default_is_empty() {
        let k = CohenKappa::default();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa(), 0.0);
    }

    // -----------------------------------------------------------------------
    // KappaM tests
    // -----------------------------------------------------------------------

    #[test]
    fn kappam_empty_returns_zero() {
        let k = KappaM::new();
        assert_eq!(k.kappa_m(), 0.0);
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.majority_class(), None);
    }

    #[test]
    fn kappam_perfect_classifier() {
        // Perfect model on imbalanced data: majority = class 0 (60/100).
        // p_model = 1.0, p_majority = 0.6
        // kappa_m = (1.0 - 0.6) / (1.0 - 0.6) = 1.0
        let mut k = KappaM::new();
        for _ in 0..60 {
            k.update(0, 0);
        }
        for _ in 0..40 {
            k.update(1, 1);
        }
        assert!(approx_eq(k.kappa_m(), 1.0));
    }

    #[test]
    fn kappam_majority_class_tracking() {
        let mut k = KappaM::new();
        k.update(0, 0);
        k.update(0, 0);
        k.update(1, 1);
        assert_eq!(k.majority_class(), Some(0));

        k.update(1, 1);
        k.update(1, 1);
        assert_eq!(k.majority_class(), Some(1));
    }

    #[test]
    fn kappam_majority_class_tie_breaks_lowest() {
        let mut k = KappaM::new();
        k.update(0, 0);
        k.update(1, 1);
        // Tie: class 0 and class 1 both have count 1. Lowest index wins.
        assert_eq!(k.majority_class(), Some(0));
    }

    #[test]
    fn kappam_acts_like_majority_baseline() {
        // Model that always predicts class 0 on data where class 0 is majority.
        // This should produce kappa_m = 0.0 (model == baseline).
        let mut k = KappaM::new();
        for _ in 0..60 {
            k.update(0, 0); // correct
        }
        for _ in 0..40 {
            k.update(1, 0); // wrong (always predicts 0)
        }
        // p_model = 60/100 = 0.6, p_majority = 60/100 = 0.6
        // kappa_m = 0.0
        assert!(approx_eq(k.kappa_m(), 0.0));
    }

    #[test]
    fn kappam_worse_than_majority() {
        // Model that always predicts the minority class.
        let mut k = KappaM::new();
        for _ in 0..80 {
            k.update(0, 1); // wrong
        }
        for _ in 0..20 {
            k.update(1, 1); // correct
        }
        // p_model = 20/100 = 0.2, p_majority = 80/100 = 0.8
        // kappa_m = (0.2 - 0.8) / (1.0 - 0.8) = -0.6 / 0.2 = -3.0
        assert!(approx_eq(k.kappa_m(), -3.0));
    }

    #[test]
    fn kappam_single_class_returns_zero() {
        // Only class 0 ever appears. p_majority = 1.0 => degenerate => 0.0
        let mut k = KappaM::new();
        for _ in 0..50 {
            k.update(0, 0);
        }
        assert_eq!(k.kappa_m(), 0.0);
    }

    #[test]
    fn kappam_single_sample() {
        let mut k = KappaM::new();
        k.update(0, 0);
        // p_majority = 1.0 => 0.0
        assert_eq!(k.kappa_m(), 0.0);
        assert_eq!(k.n_samples(), 1);
    }

    #[test]
    fn kappam_reset() {
        let mut k = KappaM::new();
        k.update(0, 0);
        k.update(1, 1);
        k.reset();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa_m(), 0.0);
        assert_eq!(k.majority_class(), None);
    }

    #[test]
    fn kappam_default_is_empty() {
        let k = KappaM::default();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa_m(), 0.0);
    }

    // -----------------------------------------------------------------------
    // KappaT tests
    // -----------------------------------------------------------------------

    #[test]
    fn kappat_empty_returns_zero() {
        let k = KappaT::new();
        assert_eq!(k.kappa_t(), 0.0);
        assert_eq!(k.n_samples(), 0);
    }

    #[test]
    fn kappat_single_sample_returns_zero() {
        let mut k = KappaT::new();
        k.update(0, 0);
        // Need at least 2 samples to evaluate no-change baseline.
        assert_eq!(k.kappa_t(), 0.0);
        assert_eq!(k.n_samples(), 1);
    }

    #[test]
    fn kappat_perfect_on_changing_stream() {
        // Stream: 0, 1, 0, 1, 0, 1 (alternating).
        // Model always predicts correctly.
        // No-change classifier always wrong (labels always change).
        //
        // n = 6, model correct = 6, p_model = 1.0
        // no-change evaluated on samples 2..6 (5 samples), all wrong => p_nochange = 0.0
        // kappa_t = (1.0 - 0.0) / (1.0 - 0.0) = 1.0
        let mut k = KappaT::new();
        let labels = [0, 1, 0, 1, 0, 1];
        for &l in &labels {
            k.update(l, l);
        }
        assert!(approx_eq(k.kappa_t(), 1.0));
    }

    #[test]
    fn kappat_no_change_baseline_is_perfect() {
        // Stream where true labels never change: 0, 0, 0, 0, 0.
        // No-change classifier is always correct => p_nochange = 1.0.
        // This is a degenerate case => kappa_t = 0.0.
        let mut k = KappaT::new();
        for _ in 0..5 {
            k.update(0, 0);
        }
        assert_eq!(k.kappa_t(), 0.0);
    }

    #[test]
    fn kappat_model_equals_nochange() {
        // Stream: 0, 0, 1, 1, 0, 0.
        // Model that mimics no-change (always predicts previous true label).
        //
        // Sample 1: true=0, pred=?  (no prev, say pred=0 — correct)
        // Sample 2: true=0, pred=0 (prev true=0) — correct, nochange correct
        // Sample 3: true=1, pred=0 (prev true=0) — wrong, nochange wrong
        // Sample 4: true=1, pred=1 (prev true=1) — correct, nochange correct
        // Sample 5: true=0, pred=1 (prev true=1) — wrong, nochange wrong
        // Sample 6: true=0, pred=0 (prev true=0) — correct, nochange correct
        //
        // n=6, model correct=4, p_model = 4/6
        // no-change: 5 evaluated, 3 correct, p_nochange = 3/5 = 0.6
        // kappa_t = (4/6 - 3/5) / (1 - 3/5) = (0.6667 - 0.6) / 0.4 = 0.1667
        let mut k = KappaT::new();
        let trues = [0, 0, 1, 1, 0, 0];
        // Model predicts previous true label (for the first, guess 0).
        let preds = [0, 0, 0, 1, 1, 0];
        for i in 0..6 {
            k.update(trues[i], preds[i]);
        }
        let expected = (4.0 / 6.0 - 3.0 / 5.0) / (1.0 - 3.0 / 5.0);
        assert!(approx_eq(k.kappa_t(), expected));
    }

    #[test]
    fn kappat_worse_than_nochange() {
        // Stream: 0, 0, 0, 0, 0  (constant).
        // No-change classifier: perfect on samples 2-5 (4/4).
        // Model always predicts 1: wrong on every sample.
        //
        // n=5, model correct=0, p_model=0.0
        // no-change: 4 evaluated, 4 correct, p_nochange=1.0
        // Degenerate => 0.0
        let mut k = KappaT::new();
        for _ in 0..5 {
            k.update(0, 1);
        }
        assert_eq!(k.kappa_t(), 0.0);
    }

    #[test]
    fn kappat_computed_example() {
        // Stream: true = [0, 1, 1, 0, 1], pred = [0, 0, 1, 1, 1]
        //
        // Model correct: samples 1(0==0), 3(1==1), 5(1==1) => 3/5
        // No-change eval (samples 2-5):
        //   sample 2: true=1, prev=0 => nochange wrong
        //   sample 3: true=1, prev=1 => nochange correct
        //   sample 4: true=0, prev=1 => nochange wrong
        //   sample 5: true=1, prev=0 => nochange wrong
        //   => 1/4 correct, p_nochange = 0.25
        //
        // p_model = 3/5 = 0.6
        // kappa_t = (0.6 - 0.25) / (1.0 - 0.25) = 0.35 / 0.75 = 0.4667
        let mut k = KappaT::new();
        let trues = [0, 1, 1, 0, 1];
        let preds = [0, 0, 1, 1, 1];
        for i in 0..5 {
            k.update(trues[i], preds[i]);
        }
        let expected = (0.6 - 0.25) / (1.0 - 0.25);
        assert!(approx_eq(k.kappa_t(), expected));
    }

    #[test]
    fn kappat_reset() {
        let mut k = KappaT::new();
        k.update(0, 0);
        k.update(1, 1);
        k.reset();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa_t(), 0.0);
    }

    #[test]
    fn kappat_default_is_empty() {
        let k = KappaT::default();
        assert_eq!(k.n_samples(), 0);
        assert_eq!(k.kappa_t(), 0.0);
    }

    #[test]
    fn kappat_two_samples() {
        // Minimal case where kappa_t is meaningful.
        // true = [0, 1], pred = [0, 1]
        // model correct = 2/2 = 1.0
        // no-change: 1 eval, true=1 vs prev=0 => wrong, p_nochange = 0.0
        // kappa_t = (1.0 - 0.0) / (1.0 - 0.0) = 1.0
        let mut k = KappaT::new();
        k.update(0, 0);
        k.update(1, 1);
        assert!(approx_eq(k.kappa_t(), 1.0));
    }
}
