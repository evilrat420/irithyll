//! [`StreamingMetric`] trait and composable metric instances.
//!
//! Provides a uniform interface for streaming metrics that update in O(1) per
//! sample and compose via the `+` operator (river-style).
//!
//! # Design
//!
//! Every metric type:
//! - Implements [`StreamingMetric`] (trait-object safe).
//! - Updates in O(1) state per sample — no past data stored.
//! - Derives all formulas from mathematical first principles; no magic constants.
//!
//! Metrics compose into a [`MetricUnion`] via `std::ops::Add`:
//!
//! ```
//! use irithyll::metrics::streaming_metric::{MAE, MSE, StreamingMetric};
//!
//! let mut m = MAE::new() + MSE::new();
//! m.update(3.0, 2.5);
//! println!("{}: {}", m.a.name(), m.a.get());
//! println!("{}: {}", m.b.name(), m.b.get());
//! ```
//!
//! # Instances
//!
//! | Type | Formula | O(1)? |
//! |------|---------|-------|
//! | [`MAE`] | mean |y - ŷ| | Yes |
//! | [`MSE`] | mean (y - ŷ)² | Yes |
//! | [`RMSE`] | √MSE | Yes |
//! | [`R2`] | 1 - SSres/SStot (Welford SStot) | Yes |
//! | [`Pinball<TAU>`] | quantile loss: max(τ(y-ŷ), (τ-1)(y-ŷ)) | Yes |
//! | [`LogLoss`] | cross-entropy loss | Yes |
//! | [`Accuracy`] | fraction correct | Yes |

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Uniform interface for streaming (one-sample-at-a-time) metrics.
///
/// Object-safe: all methods take `&self` / `&mut self` with concrete types.
/// Compose multiple instances with `a + b` to produce a [`MetricUnion<A, B>`].
///
/// # Contract
///
/// - `update` runs in O(1) time and O(1) space (no past samples stored).
/// - `get` returns the current metric value.
/// - `name` returns a `'static str` suitable for display.
/// - `reset` restores to the initial empty state.
pub trait StreamingMetric: Send + Sync {
    /// Incorporate a single (prediction, actual) pair into the running state.
    fn update(&mut self, pred: f64, actual: f64);

    /// Return the current metric value.
    fn get(&self) -> f64;

    /// Short display name for this metric.
    fn name(&self) -> &'static str;

    /// Reset to empty initial state.
    fn reset(&mut self);

    /// Whether higher values indicate better model performance.
    ///
    /// Used by evaluators to decide promotion direction.
    fn higher_is_better(&self) -> bool {
        false
    }
}

// Object-safety assertion — compile-time check.
fn _assert_object_safe(_: Box<dyn StreamingMetric>) {}

// ---------------------------------------------------------------------------
// MetricUnion — composition via + operator
// ---------------------------------------------------------------------------

/// A pair of streaming metrics that both receive the same updates.
///
/// Created by the `+` operator on two [`StreamingMetric`] implementors.
///
/// ```
/// use irithyll::metrics::streaming_metric::{MAE, RMSE, StreamingMetric};
///
/// let mut m = MAE::new() + RMSE::new();
/// m.update(3.0, 2.0);  // error = 1.0
/// assert!((m.a.get() - 1.0).abs() < 1e-10, "MAE should be 1.0");
/// assert!((m.b.get() - 1.0).abs() < 1e-10, "RMSE should be 1.0");
/// ```
pub struct MetricUnion<A, B> {
    /// Left metric of the union.
    pub a: A,
    /// Right metric of the union.
    pub b: B,
}

impl<A: StreamingMetric, B: StreamingMetric> StreamingMetric for MetricUnion<A, B> {
    fn update(&mut self, pred: f64, actual: f64) {
        self.a.update(pred, actual);
        self.b.update(pred, actual);
    }

    fn get(&self) -> f64 {
        // MetricUnion doesn't reduce to a single value in a meaningful way;
        // callers should access `.a.get()` / `.b.get()` directly.
        // We return the left metric's value for compatibility with code that
        // calls .get() on the union without inspecting both sides.
        self.a.get()
    }

    fn name(&self) -> &'static str {
        self.a.name()
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
    }

    fn higher_is_better(&self) -> bool {
        self.a.higher_is_better()
    }
}

impl<A: StreamingMetric, C: StreamingMetric, B: StreamingMetric> std::ops::Add<C>
    for MetricUnion<A, B>
{
    type Output = MetricUnion<MetricUnion<A, B>, C>;

    fn add(self, rhs: C) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// MAE — Mean Absolute Error
// ---------------------------------------------------------------------------

/// Streaming Mean Absolute Error: mean |y - ŷ|.
///
/// Updates in O(1) via a running sum of absolute errors.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{MAE, StreamingMetric};
///
/// let mut m = MAE::new();
/// m.update(2.0, 3.0); // |2.0 - 3.0| = 1.0
/// m.update(5.0, 3.0); // |5.0 - 3.0| = 2.0
/// assert!((m.get() - 1.5).abs() < 1e-10, "MAE = 1.5");
/// ```
#[derive(Debug, Clone, Default)]
pub struct MAE {
    count: u64,
    sum_abs_error: f64,
}

impl MAE {
    /// Create a new MAE tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingMetric for MAE {
    fn update(&mut self, pred: f64, actual: f64) {
        self.count += 1;
        self.sum_abs_error += (actual - pred).abs();
    }

    fn get(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.count as f64
    }

    fn name(&self) -> &'static str {
        "MAE"
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum_abs_error = 0.0;
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for MAE {
    type Output = MetricUnion<MAE, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// MSE — Mean Squared Error
// ---------------------------------------------------------------------------

/// Streaming Mean Squared Error: mean (y - ŷ)².
///
/// Updates in O(1) via a running sum of squared errors.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{MSE, StreamingMetric};
///
/// let mut m = MSE::new();
/// m.update(2.0, 4.0); // (2 - 4)^2 = 4
/// m.update(4.0, 4.0); // (4 - 4)^2 = 0
/// assert!((m.get() - 2.0).abs() < 1e-10, "MSE = 2.0");
/// ```
#[derive(Debug, Clone, Default)]
pub struct MSE {
    count: u64,
    sum_sq_error: f64,
}

impl MSE {
    /// Create a new MSE tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingMetric for MSE {
    fn update(&mut self, pred: f64, actual: f64) {
        self.count += 1;
        let e = actual - pred;
        self.sum_sq_error += e * e;
    }

    fn get(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.count as f64
    }

    fn name(&self) -> &'static str {
        "MSE"
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum_sq_error = 0.0;
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for MSE {
    type Output = MetricUnion<MSE, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// RMSE — Root Mean Squared Error
// ---------------------------------------------------------------------------

/// Streaming Root Mean Squared Error: √(mean (y - ŷ)²).
///
/// Internally maintains a running MSE; `get()` returns its square root.
/// O(1) state and O(1) update.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{RMSE, StreamingMetric};
///
/// let mut m = RMSE::new();
/// m.update(1.0, 2.0); // (1 - 2)^2 = 1 => RMSE = 1.0
/// assert!((m.get() - 1.0).abs() < 1e-10, "RMSE = 1.0");
/// ```
#[derive(Debug, Clone, Default)]
pub struct RMSE {
    inner: MSE,
}

impl RMSE {
    /// Create a new RMSE tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingMetric for RMSE {
    fn update(&mut self, pred: f64, actual: f64) {
        self.inner.update(pred, actual);
    }

    fn get(&self) -> f64 {
        self.inner.get().sqrt()
    }

    fn name(&self) -> &'static str {
        "RMSE"
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for RMSE {
    type Output = MetricUnion<RMSE, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// R2 — Coefficient of Determination
// ---------------------------------------------------------------------------

/// Streaming R² (coefficient of determination): 1 - SSres / SStot.
///
/// SStot is the running sum of squared deviations of the actual values
/// from their running mean. Maintained via Welford's online algorithm,
/// which is numerically stable and O(1) per update.
///
/// Formula: `R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²`
///
/// Returns 0.0 if fewer than 2 samples have been observed, or if the
/// target variance is zero (constant target stream).
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{R2, StreamingMetric};
///
/// let mut m = R2::new();
/// // Perfect predictions => R² = 1.0
/// m.update(1.0, 1.0);
/// m.update(2.0, 2.0);
/// m.update(3.0, 3.0);
/// assert!((m.get() - 1.0).abs() < 1e-10, "R2 = 1.0 for perfect predictions");
/// ```
#[derive(Debug, Clone, Default)]
pub struct R2 {
    count: u64,
    sum_sq_error: f64,
    // Welford state for target variance (SStot without /n factor):
    // target_m2 = Σ (y_i - ȳ_i)², updated incrementally.
    target_mean: f64,
    target_m2: f64,
}

impl R2 {
    /// Create a new R2 tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingMetric for R2 {
    fn update(&mut self, pred: f64, actual: f64) {
        self.count += 1;

        // SSres accumulation
        let e = actual - pred;
        self.sum_sq_error += e * e;

        // Welford online update for SStot denominator
        let delta = actual - self.target_mean;
        self.target_mean += delta / self.count as f64;
        let delta2 = actual - self.target_mean;
        self.target_m2 += delta * delta2;
    }

    fn get(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        if self.target_m2 == 0.0 {
            // Constant target stream: R² undefined, return 0.
            return 0.0;
        }
        1.0 - self.sum_sq_error / self.target_m2
    }

    fn name(&self) -> &'static str {
        "R2"
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum_sq_error = 0.0;
        self.target_mean = 0.0;
        self.target_m2 = 0.0;
    }

    fn higher_is_better(&self) -> bool {
        true
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for R2 {
    type Output = MetricUnion<R2, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// Pinball — Quantile Loss
// ---------------------------------------------------------------------------

/// Streaming quantile (pinball) loss at a given quantile level τ.
///
/// Formula: `L_τ(y, ŷ) = max(τ(y - ŷ), (τ - 1)(y - ŷ))`
///
/// Equivalently: `(y - ŷ) * (τ - 1_{y < ŷ})` where 1_{y < ŷ} is the indicator.
///
/// τ is the quantile parameter in (0, 1). τ = 0.5 reduces to scaled MAE.
/// τ is a structural parameter, not a tuning knob — derive it from the
/// decision problem (e.g. τ = cost_underpredict / (cost_underpredict + cost_overpredict)).
///
/// O(1) state: maintains running sum and count.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{Pinball, StreamingMetric};
///
/// // At τ=0.5, pinball = 0.5 * |y - ŷ| (= MAE/2)
/// let mut m = Pinball::new(0.5);
/// m.update(2.0, 3.0); // y=2, ŷ=3: y < ŷ, loss = (1-0.5)*(3-2) = 0.5
/// assert!((m.get() - 0.5).abs() < 1e-10, "Pinball at tau=0.5 = 0.5");
/// ```
#[derive(Debug, Clone)]
pub struct Pinball {
    tau: f64,
    count: u64,
    sum_loss: f64,
}

impl Pinball {
    /// Create a Pinball metric at quantile level `tau` ∈ (0, 1).
    ///
    /// # Panics
    ///
    /// Panics if `tau` is not in the open interval (0, 1).
    pub fn new(tau: f64) -> Self {
        assert!(
            tau > 0.0 && tau < 1.0,
            "Pinball tau must be in (0, 1), got {tau}"
        );
        Self {
            tau,
            count: 0,
            sum_loss: 0.0,
        }
    }

    /// The quantile level τ this metric tracks.
    pub fn tau(&self) -> f64 {
        self.tau
    }
}

impl StreamingMetric for Pinball {
    fn update(&mut self, pred: f64, actual: f64) {
        self.count += 1;
        let residual = actual - pred;
        // pinball: max(τ * residual, (τ - 1) * residual)
        // equivalently: τ * residual if residual >= 0 else (τ - 1) * residual
        let loss = if residual >= 0.0 {
            self.tau * residual
        } else {
            (self.tau - 1.0) * residual
        };
        self.sum_loss += loss;
    }

    fn get(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_loss / self.count as f64
    }

    fn name(&self) -> &'static str {
        "Pinball"
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum_loss = 0.0;
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for Pinball {
    type Output = MetricUnion<Pinball, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// LogLoss — Binary Cross-Entropy
// ---------------------------------------------------------------------------

/// Streaming binary cross-entropy (log loss).
///
/// Formula: `L = -(y ln p + (1-y) ln(1-p))`
///
/// where y ∈ {0, 1} is the target label and p ∈ [0, 1] is the predicted
/// probability. Probabilities are clipped to [1e-15, 1-1e-15] to prevent
/// log(0).
///
/// `update(pred, actual)` interprets `pred` as predicted probability p,
/// `actual` as 0.0 or 1.0 label y.
///
/// O(1) state: running sum and count.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{LogLoss, StreamingMetric};
///
/// let mut m = LogLoss::new();
/// // Predicting 0.5 for all => log loss = ln(2) ≈ 0.693
/// m.update(0.5, 1.0);
/// m.update(0.5, 0.0);
/// let expected = 2.0_f64.ln();
/// assert!((m.get() - expected).abs() < 1e-10, "LogLoss at p=0.5 = ln(2)");
/// ```
#[derive(Debug, Clone, Default)]
pub struct LogLoss {
    count: u64,
    sum_loss: f64,
}

impl LogLoss {
    /// Create a new LogLoss tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

const LOGLOSS_CLIP_MIN: f64 = 1e-15;
const LOGLOSS_CLIP_MAX: f64 = 1.0 - 1e-15;

impl StreamingMetric for LogLoss {
    fn update(&mut self, pred: f64, actual: f64) {
        self.count += 1;
        let p = pred.clamp(LOGLOSS_CLIP_MIN, LOGLOSS_CLIP_MAX);
        let y = if actual > 0.5 { 1.0_f64 } else { 0.0_f64 };
        let loss = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        self.sum_loss += loss;
    }

    fn get(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_loss / self.count as f64
    }

    fn name(&self) -> &'static str {
        "LogLoss"
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum_loss = 0.0;
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for LogLoss {
    type Output = MetricUnion<LogLoss, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
    }
}

// ---------------------------------------------------------------------------
// Accuracy — Classification Accuracy
// ---------------------------------------------------------------------------

/// Streaming classification accuracy: fraction of correct predictions.
///
/// `update(pred, actual)` treats `pred` and `actual` as class labels; a
/// prediction is "correct" when `pred.round() as i64 == actual.round() as i64`.
///
/// O(1) state: running correct count and total count.
///
/// # Example
///
/// ```
/// use irithyll::metrics::streaming_metric::{Accuracy, StreamingMetric};
///
/// let mut m = Accuracy::new();
/// m.update(1.0, 1.0); // correct
/// m.update(0.0, 1.0); // wrong
/// assert!((m.get() - 0.5).abs() < 1e-10, "Accuracy = 0.5");
/// ```
#[derive(Debug, Clone, Default)]
pub struct Accuracy {
    n_total: u64,
    n_correct: u64,
}

impl Accuracy {
    /// Create a new Accuracy tracker (zero initial state).
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingMetric for Accuracy {
    fn update(&mut self, pred: f64, actual: f64) {
        self.n_total += 1;
        if pred.round() as i64 == actual.round() as i64 {
            self.n_correct += 1;
        }
    }

    fn get(&self) -> f64 {
        if self.n_total == 0 {
            return 0.0;
        }
        self.n_correct as f64 / self.n_total as f64
    }

    fn name(&self) -> &'static str {
        "Accuracy"
    }

    fn reset(&mut self) {
        self.n_total = 0;
        self.n_correct = 0;
    }

    fn higher_is_better(&self) -> bool {
        true
    }
}

impl<B: StreamingMetric> std::ops::Add<B> for Accuracy {
    type Output = MetricUnion<Accuracy, B>;
    fn add(self, rhs: B) -> Self::Output {
        MetricUnion { a: self, b: rhs }
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

    // ---------- MAE ----------

    #[test]
    fn mae_streaming_matches_offline_computation() {
        let data = [(3.0_f64, 2.5_f64), (1.0, 1.5), (5.0, 4.0), (2.0, 2.0)];
        let mut m = MAE::new();
        for &(actual, pred) in &data {
            m.update(pred, actual);
        }
        // offline: mean(|3-2.5|, |1-1.5|, |5-4|, |2-2|) = mean(0.5, 0.5, 1.0, 0.0) = 0.5
        assert!(approx_eq(m.get(), 0.5), "MAE expected 0.5, got {}", m.get());
    }

    #[test]
    fn mae_empty_returns_zero() {
        let m = MAE::new();
        assert_eq!(m.get(), 0.0, "empty MAE must be 0.0");
    }

    #[test]
    fn mae_reset_clears_state() {
        let mut m = MAE::new();
        m.update(1.0, 2.0);
        m.reset();
        assert_eq!(m.get(), 0.0, "MAE after reset must be 0.0");
    }

    // ---------- MSE ----------

    #[test]
    fn mse_streaming_matches_offline_computation() {
        // errors: (3-2.5)^2=0.25, (1-1.5)^2=0.25, (5-4)^2=1.0, (2-2)^2=0 => MSE = 1.5/4 = 0.375
        let data = [(3.0_f64, 2.5_f64), (1.0, 1.5), (5.0, 4.0), (2.0, 2.0)];
        let mut m = MSE::new();
        for &(actual, pred) in &data {
            m.update(pred, actual);
        }
        assert!(
            approx_eq(m.get(), 0.375),
            "MSE expected 0.375, got {}",
            m.get()
        );
    }

    #[test]
    fn mse_empty_returns_zero() {
        assert_eq!(MSE::new().get(), 0.0);
    }

    // ---------- RMSE ----------

    #[test]
    fn rmse_is_sqrt_of_mse() {
        let data = [(3.0_f64, 2.5_f64), (1.0, 1.5), (5.0, 4.0)];
        let mut rmse = RMSE::new();
        let mut mse = MSE::new();
        for &(actual, pred) in &data {
            rmse.update(pred, actual);
            mse.update(pred, actual);
        }
        let expected = mse.get().sqrt();
        assert!(
            approx_eq(rmse.get(), expected),
            "RMSE = sqrt(MSE): expected {expected}, got {}",
            rmse.get()
        );
    }

    // ---------- R2 ----------

    #[test]
    fn r_squared_streaming_uses_welford_sstot() {
        // Perfect predictions: R2 = 1.0
        let mut m = R2::new();
        m.update(1.0, 1.0);
        m.update(2.0, 2.0);
        m.update(3.0, 3.0);
        m.update(4.0, 4.0);
        assert!(
            approx_eq(m.get(), 1.0),
            "R2 = 1.0 for perfect predictions, got {}",
            m.get()
        );
    }

    #[test]
    fn r2_negative_for_bad_predictions() {
        let mut m = R2::new();
        // targets: 1, 2, 3; predictions all 10 — much worse than mean predictor
        m.update(10.0, 1.0);
        m.update(10.0, 2.0);
        m.update(10.0, 3.0);
        assert!(
            m.get() < 0.0,
            "R2 must be negative for terrible predictions"
        );
    }

    #[test]
    fn r2_cold_start_returns_zero() {
        let mut m = R2::new();
        // Single sample: R2 undefined, must return 0.0
        m.update(1.0, 1.0);
        assert_eq!(m.get(), 0.0, "R2 with <2 samples must be 0.0");
    }

    // ---------- Pinball ----------

    #[test]
    fn pinball_loss_correct_at_tau_05() {
        // At τ=0.5, pinball = 0.5 * |y - ŷ| (half MAE)
        let mut m = Pinball::new(0.5);
        // residual = actual - pred = 3 - 2 = 1 (positive): loss = 0.5 * 1 = 0.5
        m.update(2.0, 3.0);
        assert!(
            approx_eq(m.get(), 0.5),
            "Pinball(0.5) on residual=1 expected 0.5, got {}",
            m.get()
        );
    }

    #[test]
    fn pinball_asymmetry_at_tau_09() {
        // τ=0.9: overestimate is penalised less than underestimate
        // pred=5, actual=3 => residual = 3-5 = -2 (negative): loss = (0.9-1)*(-2) = (-0.1)*(-2) = 0.2
        let mut m_over = Pinball::new(0.9);
        m_over.update(5.0, 3.0);

        // pred=3, actual=5 => residual = 5-3 = 2 (positive): loss = 0.9*2 = 1.8
        let mut m_under = Pinball::new(0.9);
        m_under.update(3.0, 5.0);

        assert!(
            m_over.get() < m_under.get(),
            "overestimate loss ({}) < underestimate loss ({}) at tau=0.9",
            m_over.get(),
            m_under.get()
        );
        assert!(approx_eq(m_over.get(), 0.2));
        assert!(approx_eq(m_under.get(), 1.8));
    }

    #[test]
    fn pinball_empty_returns_zero() {
        assert_eq!(Pinball::new(0.5).get(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Pinball tau must be in (0, 1)")]
    fn pinball_rejects_tau_out_of_range() {
        let _ = Pinball::new(1.0);
    }

    // ---------- LogLoss ----------

    #[test]
    fn logloss_at_half_equals_ln2() {
        // predicting 0.5 for all => loss = ln(2) per sample
        let mut m = LogLoss::new();
        m.update(0.5, 1.0);
        m.update(0.5, 0.0);
        let expected = 2.0_f64.ln();
        assert!(
            approx_eq(m.get(), expected),
            "LogLoss at p=0.5 expected ln(2)={expected}, got {}",
            m.get()
        );
    }

    #[test]
    fn logloss_clamps_extremes() {
        let mut m = LogLoss::new();
        m.update(0.0, 1.0); // would be ln(0) without clamp
        m.update(1.0, 0.0); // would be ln(0) without clamp
        assert!(
            m.get().is_finite(),
            "LogLoss must not be inf/NaN at extremes"
        );
    }

    // ---------- Accuracy ----------

    #[test]
    fn accuracy_classification_metric_correct() {
        let mut m = Accuracy::new();
        m.update(1.0, 1.0); // correct
        m.update(0.0, 0.0); // correct
        m.update(1.0, 0.0); // wrong
        m.update(0.0, 1.0); // wrong
        assert!(
            approx_eq(m.get(), 0.5),
            "Accuracy expected 0.5, got {}",
            m.get()
        );
    }

    #[test]
    fn accuracy_handles_multiclass_labels() {
        let mut m = Accuracy::new();
        m.update(2.0, 2.0); // correct
        m.update(1.0, 2.0); // wrong
        m.update(0.0, 0.0); // correct
        assert!(
            approx_eq(m.get(), 2.0 / 3.0),
            "Accuracy with 3-class expected {}, got {}",
            2.0 / 3.0,
            m.get()
        );
    }

    // ---------- MetricUnion ----------

    #[test]
    fn composed_metrics_emit_both_values() {
        let mut m = MAE::new() + MSE::new();
        // errors: 1.0, 2.0
        m.update(2.0, 3.0); // |3-2|=1, (3-2)^2=1
        m.update(1.0, 3.0); // |3-1|=2, (3-1)^2=4
                            // MAE = (1+2)/2 = 1.5, MSE = (1+4)/2 = 2.5
        assert!(
            approx_eq(m.a.get(), 1.5),
            "Union.a (MAE) expected 1.5, got {}",
            m.a.get()
        );
        assert!(
            approx_eq(m.b.get(), 2.5),
            "Union.b (MSE) expected 2.5, got {}",
            m.b.get()
        );
    }

    #[test]
    fn triple_union_receives_all_updates() {
        let mut m = MAE::new() + MSE::new() + Accuracy::new();
        m.update(1.0, 1.0); // error=0, correct
        m.update(2.0, 3.0); // error=1, wrong
                            // MAE = 0.5, MSE = 0.5, Accuracy = 0.5
                            // m.a = MetricUnion(MAE, MSE), m.b = Accuracy
        assert!(
            approx_eq(m.a.a.get(), 0.5),
            "MAE in triple union: expected 0.5, got {}",
            m.a.a.get()
        );
        assert!(
            approx_eq(m.b.get(), 0.5),
            "Accuracy in triple union: expected 0.5, got {}",
            m.b.get()
        );
    }

    #[test]
    fn union_reset_resets_both() {
        let mut m = MAE::new() + MSE::new();
        m.update(1.0, 2.0);
        m.reset();
        assert_eq!(m.a.get(), 0.0, "MAE must be 0 after union reset");
        assert_eq!(m.b.get(), 0.0, "MSE must be 0 after union reset");
    }

    #[test]
    fn higher_is_better_flags() {
        assert!(!MAE::new().higher_is_better(), "MAE: lower is better");
        assert!(!MSE::new().higher_is_better(), "MSE: lower is better");
        assert!(!RMSE::new().higher_is_better(), "RMSE: lower is better");
        assert!(R2::new().higher_is_better(), "R2: higher is better");
        assert!(
            Accuracy::new().higher_is_better(),
            "Accuracy: higher is better"
        );
    }
}
