//! Learning rate scheduling for streaming gradient boosted trees.
//!
//! In standard (batch) gradient boosting the learning rate is fixed for the
//! entire training run. Streaming ensembles see data indefinitely, so a
//! fixed rate must balance early convergence against long-term stability.
//! Learning rate schedulers resolve this tension by adapting the rate over
//! the lifetime of the model.
//!
//! # Provided schedulers
//!
//! | Scheduler | Strategy |
//! |-----------|----------|
//! | [`ConstantLR`] | Fixed rate -- baseline behaviour, equivalent to no scheduling. |
//! | [`LinearDecayLR`] | Linearly interpolates from `initial_lr` to `final_lr` over a fixed number of steps, then holds `final_lr`. |
//! | [`ExponentialDecayLR`] | Multiplicative decay by `gamma` each step, floored at `1e-8` to avoid numerical zero. |
//! | [`CosineAnnealingLR`] | Periodic cosine wave between `max_lr` and `min_lr`, useful for warm-restart style exploration. |
//! | [`PlateauLR`] | Monitors the loss and reduces the rate by `factor` when improvement stalls for `patience` steps. |
//!
//! # Custom schedulers
//!
//! Implement the [`LRScheduler`] trait to build your own schedule:
//!
//! ```
//! use irithyll::ensemble::lr_schedule::LRScheduler;
//!
//! #[derive(Clone, Debug)]
//! struct HalvingLR { lr: f64 }
//!
//! impl LRScheduler for HalvingLR {
//!     fn learning_rate(&mut self, step: u64, _loss: f64) -> f64 {
//!         let lr = self.lr;
//!         self.lr *= 0.5_f64.max(1e-8);
//!         lr
//!     }
//!     fn reset(&mut self) { self.lr = 1.0; }
//! }
//! ```

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A learning rate scheduler for streaming gradient boosted trees.
///
/// The ensemble calls [`learning_rate`](LRScheduler::learning_rate) once per
/// boosting round, passing the monotonically increasing `step` counter and the
/// most recent loss value. Implementations may use either or both to decide the
/// rate.
///
/// All schedulers must be `Send + Sync` so they can live inside async ensemble
/// wrappers and be shared across threads.
pub trait LRScheduler: Send + Sync {
    /// Return the learning rate for the given `step` and `current_loss`.
    ///
    /// # Arguments
    ///
    /// * `step` -- Zero-based step counter. Incremented by the caller before
    ///   each invocation (0 on the first call, 1 on the second, ...).
    /// * `current_loss` -- The most recent loss value observed by the ensemble.
    ///   Schedulers that do not use loss feedback (everything except
    ///   [`PlateauLR`]) may ignore this argument.
    fn learning_rate(&mut self, step: u64, current_loss: f64) -> f64;

    /// Reset the scheduler to its initial state.
    ///
    /// Called when the ensemble is reset (e.g., after a concept-drift event
    /// triggers a full model rebuild).
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// 1. ConstantLR
// ---------------------------------------------------------------------------

/// Always returns the same learning rate.
///
/// This is the simplest scheduler and reproduces the behaviour of a plain
/// fixed-rate ensemble. It exists so that code paths expecting a `dyn
/// LRScheduler` can use a constant rate without special-casing.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::lr_schedule::{LRScheduler, ConstantLR};
///
/// let mut sched = ConstantLR::new(0.05);
/// assert!((sched.learning_rate(0, 1.0) - 0.05).abs() < f64::EPSILON);
/// assert!((sched.learning_rate(1000, 0.1) - 0.05).abs() < f64::EPSILON);
/// ```
#[derive(Clone, Debug)]
pub struct ConstantLR {
    /// The fixed learning rate.
    lr: f64,
}

impl ConstantLR {
    /// Create a constant-rate scheduler.
    ///
    /// # Arguments
    ///
    /// * `lr` -- The learning rate returned on every call.
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    #[inline]
    fn learning_rate(&mut self, _step: u64, _current_loss: f64) -> f64 {
        self.lr
    }

    fn reset(&mut self) {
        // Nothing to reset -- the rate is stateless.
    }
}

// ---------------------------------------------------------------------------
// 2. LinearDecayLR
// ---------------------------------------------------------------------------

/// Linearly interpolates the learning rate from `initial_lr` to `final_lr`
/// over `decay_steps`, then holds `final_lr` forever.
///
/// The formula is:
///
/// ```text
/// lr = initial_lr - (initial_lr - final_lr) * min(step / decay_steps, 1.0)
/// ```
///
/// This gives a smooth ramp-down that reaches `final_lr` at exactly
/// `step == decay_steps` and clamps there for all subsequent steps.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::lr_schedule::{LRScheduler, LinearDecayLR};
///
/// let mut sched = LinearDecayLR::new(0.1, 0.01, 100);
/// // At step 0 we get the initial rate.
/// assert!((sched.learning_rate(0, 0.0) - 0.1).abs() < 1e-12);
/// // At step 50 we're halfway.
/// assert!((sched.learning_rate(50, 0.0) - 0.055).abs() < 1e-12);
/// // At step 100 we've reached the final rate.
/// assert!((sched.learning_rate(100, 0.0) - 0.01).abs() < 1e-12);
/// // Beyond decay_steps the rate stays clamped.
/// assert!((sched.learning_rate(200, 0.0) - 0.01).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct LinearDecayLR {
    /// Starting learning rate.
    initial_lr: f64,
    /// Terminal learning rate (held after `decay_steps`).
    final_lr: f64,
    /// Number of steps over which the linear ramp is applied.
    decay_steps: u64,
}

impl LinearDecayLR {
    /// Create a linear-decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` -- Rate at step 0.
    /// * `final_lr` -- Rate from step `decay_steps` onward.
    /// * `decay_steps` -- Length of the linear ramp in steps.
    pub fn new(initial_lr: f64, final_lr: f64, decay_steps: u64) -> Self {
        Self {
            initial_lr,
            final_lr,
            decay_steps,
        }
    }
}

impl LRScheduler for LinearDecayLR {
    #[inline]
    fn learning_rate(&mut self, step: u64, _current_loss: f64) -> f64 {
        let t = if self.decay_steps == 0 {
            1.0
        } else {
            (step as f64 / self.decay_steps as f64).min(1.0)
        };
        self.initial_lr - (self.initial_lr - self.final_lr) * t
    }

    fn reset(&mut self) {
        // Stateless -- nothing to reset.
    }
}

// ---------------------------------------------------------------------------
// 3. ExponentialDecayLR
// ---------------------------------------------------------------------------

/// Multiplicative exponential decay: `lr = initial_lr * gamma^step`.
///
/// The rate is floored at `1e-8` to prevent numerical underflow that would
/// effectively freeze learning. For a half-life of *h* steps, set
/// `gamma = 0.5^(1/h)`.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::lr_schedule::{LRScheduler, ExponentialDecayLR};
///
/// let mut sched = ExponentialDecayLR::new(1.0, 0.9);
/// assert!((sched.learning_rate(0, 0.0) - 1.0).abs() < 1e-12);
/// assert!((sched.learning_rate(1, 0.0) - 0.9).abs() < 1e-12);
/// assert!((sched.learning_rate(2, 0.0) - 0.81).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct ExponentialDecayLR {
    /// Learning rate at step 0.
    initial_lr: f64,
    /// Per-step multiplicative factor (typically in (0, 1)).
    gamma: f64,
}

impl ExponentialDecayLR {
    /// Create an exponential-decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` -- Rate at step 0.
    /// * `gamma` -- Multiplicative decay factor applied each step.
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRScheduler for ExponentialDecayLR {
    #[inline]
    fn learning_rate(&mut self, step: u64, _current_loss: f64) -> f64 {
        (self.initial_lr * self.gamma.powi(step as i32)).max(1e-8)
    }

    fn reset(&mut self) {
        // Stateless -- nothing to reset.
    }
}

// ---------------------------------------------------------------------------
// 4. CosineAnnealingLR
// ---------------------------------------------------------------------------

/// Cosine annealing with periodic warm restarts.
///
/// The learning rate follows a cosine curve between `max_lr` and `min_lr`,
/// repeating every `period` steps:
///
/// ```text
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * (step % period) / period))
/// ```
///
/// At the start of each period (`step % period == 0`) the rate jumps back to
/// `max_lr`, providing a "warm restart" that can help the ensemble escape
/// local plateaus in a streaming setting.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::lr_schedule::{LRScheduler, CosineAnnealingLR};
///
/// let mut sched = CosineAnnealingLR::new(0.1, 0.01, 100);
/// // Period start → max_lr.
/// assert!((sched.learning_rate(0, 0.0) - 0.1).abs() < 1e-12);
/// // Midpoint → min_lr.
/// assert!((sched.learning_rate(50, 0.0) - 0.01).abs() < 1e-12);
/// // Full period → back to max_lr.
/// assert!((sched.learning_rate(100, 0.0) - 0.1).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    /// Peak learning rate (at the start of each period).
    max_lr: f64,
    /// Trough learning rate (at the midpoint of each period).
    min_lr: f64,
    /// Number of steps per cosine cycle.
    period: u64,
}

impl CosineAnnealingLR {
    /// Create a cosine-annealing scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` -- Rate at the start (and end) of each cosine period.
    /// * `min_lr` -- Rate at the midpoint of each period.
    /// * `period` -- Length of one cosine cycle in steps.
    pub fn new(max_lr: f64, min_lr: f64, period: u64) -> Self {
        Self {
            max_lr,
            min_lr,
            period,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    #[inline]
    fn learning_rate(&mut self, step: u64, _current_loss: f64) -> f64 {
        let phase = if self.period == 0 {
            0.0
        } else {
            (step % self.period) as f64 / self.period as f64
        };
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (2.0 * PI * phase).cos())
    }

    fn reset(&mut self) {
        // Stateless -- nothing to reset.
    }
}

// ---------------------------------------------------------------------------
// 5. PlateauLR
// ---------------------------------------------------------------------------

/// Reduce learning rate when the loss plateaus.
///
/// Monitors `current_loss` and reduces the rate by `factor` whenever the loss
/// has not improved for `patience` consecutive steps. "Improved" means the
/// new loss is strictly less than the best-seen loss.
///
/// The rate is floored at `min_lr` to prevent it from vanishing entirely.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::lr_schedule::{LRScheduler, PlateauLR};
///
/// let mut sched = PlateauLR::new(0.1, 0.5, 3, 0.001);
///
/// // Improving loss -- rate stays at 0.1.
/// assert!((sched.learning_rate(0, 1.0) - 0.1).abs() < 1e-12);
/// assert!((sched.learning_rate(1, 0.9) - 0.1).abs() < 1e-12);
///
/// // Stagnating loss for patience=3 steps.
/// assert!((sched.learning_rate(2, 0.95) - 0.1).abs() < 1e-12);
/// assert!((sched.learning_rate(3, 0.95) - 0.1).abs() < 1e-12);
/// assert!((sched.learning_rate(4, 0.95) - 0.1).abs() < 1e-12);
///
/// // Patience exhausted -- rate drops to 0.1 * 0.5 = 0.05.
/// assert!((sched.learning_rate(5, 0.95) - 0.05).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct PlateauLR {
    /// Starting learning rate.
    initial_lr: f64,
    /// Multiplicative factor applied when patience is exhausted (0 < factor < 1).
    factor: f64,
    /// Number of non-improving steps before a reduction.
    patience: u64,
    /// Minimum learning rate floor.
    min_lr: f64,

    // -- internal state --
    /// Best loss observed so far.
    best_loss: f64,
    /// Number of consecutive steps without improvement.
    steps_without_improvement: u64,
    /// The current (possibly reduced) learning rate.
    current_lr: f64,
}

impl PlateauLR {
    /// Create a plateau-aware scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` -- Starting learning rate.
    /// * `factor` -- Multiplicative reduction factor (e.g., 0.5 halves the rate).
    /// * `patience` -- Number of non-improving steps to tolerate before reducing.
    /// * `min_lr` -- Floor below which the rate will not be reduced.
    pub fn new(initial_lr: f64, factor: f64, patience: u64, min_lr: f64) -> Self {
        Self {
            initial_lr,
            factor,
            patience,
            min_lr,
            best_loss: f64::INFINITY,
            steps_without_improvement: 0,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for PlateauLR {
    fn learning_rate(&mut self, _step: u64, current_loss: f64) -> f64 {
        if current_loss < self.best_loss {
            // Improvement -- record and reset counter.
            self.best_loss = current_loss;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;

            if self.steps_without_improvement > self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.steps_without_improvement = 0;
            }
        }

        self.current_lr
    }

    fn reset(&mut self) {
        self.best_loss = f64::INFINITY;
        self.steps_without_improvement = 0;
        self.current_lr = self.initial_lr;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ConstantLR --------------------------------------------------------

    /// Constant scheduler always returns the configured rate regardless of
    /// step or loss.
    #[test]
    fn test_constant_lr() {
        let mut sched = ConstantLR::new(0.05);

        for step in 0..100 {
            let lr = sched.learning_rate(step, 999.0);
            assert!(
                (lr - 0.05).abs() < f64::EPSILON,
                "ConstantLR should always return 0.05, got {} at step {}",
                lr,
                step,
            );
        }
    }

    // -- LinearDecayLR -----------------------------------------------------

    /// Linear decay interpolates correctly between initial and final.
    #[test]
    fn test_linear_decay() {
        let mut sched = LinearDecayLR::new(0.1, 0.01, 100);

        let lr0 = sched.learning_rate(0, 0.0);
        assert!(
            (lr0 - 0.1).abs() < 1e-12,
            "step 0 should be initial_lr (0.1), got {}",
            lr0,
        );

        let lr50 = sched.learning_rate(50, 0.0);
        let expected_50 = 0.1 - (0.1 - 0.01) * 0.5;
        assert!(
            (lr50 - expected_50).abs() < 1e-12,
            "step 50 should be {}, got {}",
            expected_50,
            lr50,
        );

        let lr100 = sched.learning_rate(100, 0.0);
        assert!(
            (lr100 - 0.01).abs() < 1e-12,
            "step 100 should be final_lr (0.01), got {}",
            lr100,
        );
    }

    /// Linear decay clamps at final_lr for steps beyond decay_steps.
    #[test]
    fn test_linear_decay_clamps() {
        let mut sched = LinearDecayLR::new(0.1, 0.01, 50);

        let lr_before = sched.learning_rate(50, 0.0);
        let lr_after = sched.learning_rate(200, 0.0);
        assert!(
            (lr_before - 0.01).abs() < 1e-12,
            "at decay_steps should be final_lr, got {}",
            lr_before,
        );
        assert!(
            (lr_after - 0.01).abs() < 1e-12,
            "beyond decay_steps should still be final_lr, got {}",
            lr_after,
        );
    }

    // -- ExponentialDecayLR ------------------------------------------------

    /// Exponential decay follows gamma^step correctly.
    #[test]
    fn test_exponential_decay() {
        let mut sched = ExponentialDecayLR::new(1.0, 0.9);

        let lr0 = sched.learning_rate(0, 0.0);
        assert!(
            (lr0 - 1.0).abs() < 1e-12,
            "step 0 should be initial_lr (1.0), got {}",
            lr0,
        );

        let lr1 = sched.learning_rate(1, 0.0);
        assert!(
            (lr1 - 0.9).abs() < 1e-12,
            "step 1 should be 0.9, got {}",
            lr1,
        );

        let lr2 = sched.learning_rate(2, 0.0);
        assert!(
            (lr2 - 0.81).abs() < 1e-12,
            "step 2 should be 0.81, got {}",
            lr2,
        );

        let lr10 = sched.learning_rate(10, 0.0);
        let expected_10 = 0.9_f64.powi(10);
        assert!(
            (lr10 - expected_10).abs() < 1e-10,
            "step 10 should be {}, got {}",
            expected_10,
            lr10,
        );
    }

    /// Exponential decay floors at 1e-8, never reaching zero.
    #[test]
    fn test_exponential_floor() {
        let mut sched = ExponentialDecayLR::new(1.0, 0.01);

        // After enough steps, gamma^step would be astronomically small.
        let lr = sched.learning_rate(10_000, 0.0);
        assert!(
            lr >= 1e-8,
            "exponential decay should floor at 1e-8, got {}",
            lr,
        );
        assert!(
            (lr - 1e-8).abs() < 1e-15,
            "at extreme steps the rate should equal the floor, got {}",
            lr,
        );
    }

    // -- CosineAnnealingLR -------------------------------------------------

    /// Cosine annealing hits max at period boundaries and min at midpoints.
    #[test]
    fn test_cosine_annealing() {
        let mut sched = CosineAnnealingLR::new(0.1, 0.01, 100);

        // At step 0 → max_lr.
        let lr0 = sched.learning_rate(0, 0.0);
        assert!(
            (lr0 - 0.1).abs() < 1e-12,
            "period start should be max_lr (0.1), got {}",
            lr0,
        );

        // At step 50 → min_lr (cos(pi) = -1).
        let lr50 = sched.learning_rate(50, 0.0);
        assert!(
            (lr50 - 0.01).abs() < 1e-12,
            "period midpoint should be min_lr (0.01), got {}",
            lr50,
        );

        // At step 25 → halfway between max and min.
        let lr25 = sched.learning_rate(25, 0.0);
        let expected_25 = 0.01 + 0.5 * (0.1 - 0.01) * (1.0 + (2.0 * PI * 0.25).cos());
        assert!(
            (lr25 - expected_25).abs() < 1e-12,
            "quarter-period should be {}, got {}",
            expected_25,
            lr25,
        );
    }

    /// Cosine annealing wraps correctly at period boundaries.
    #[test]
    fn test_cosine_boundaries() {
        let mut sched = CosineAnnealingLR::new(0.1, 0.01, 100);

        let at_boundary = sched.learning_rate(100, 0.0);
        assert!(
            (at_boundary - 0.1).abs() < 1e-12,
            "step==period should wrap to max_lr, got {}",
            at_boundary,
        );

        let second_mid = sched.learning_rate(150, 0.0);
        assert!(
            (second_mid - 0.01).abs() < 1e-12,
            "second period midpoint should be min_lr, got {}",
            second_mid,
        );
    }

    // -- PlateauLR ---------------------------------------------------------

    /// Plateau scheduler reduces the rate after patience non-improving steps.
    #[test]
    fn test_plateau_reduces() {
        let mut sched = PlateauLR::new(0.1, 0.5, 3, 0.001);

        // First call sets best_loss = 1.0.
        let lr = sched.learning_rate(0, 1.0);
        assert!(
            (lr - 0.1).abs() < 1e-12,
            "initial rate should be 0.1, got {}",
            lr
        );

        // Three non-improving steps (patience = 3).
        sched.learning_rate(1, 1.0); // counter: 1
        sched.learning_rate(2, 1.0); // counter: 2
        sched.learning_rate(3, 1.0); // counter: 3

        // Fourth non-improving step exceeds patience → reduce.
        let lr_reduced = sched.learning_rate(4, 1.0);
        assert!(
            (lr_reduced - 0.05).abs() < 1e-12,
            "after patience exceeded, rate should be 0.1*0.5 = 0.05, got {}",
            lr_reduced,
        );
    }

    /// Plateau scheduler resets its counter when loss improves.
    #[test]
    fn test_plateau_improvement_resets() {
        let mut sched = PlateauLR::new(0.1, 0.5, 2, 0.001);

        // Establish baseline.
        sched.learning_rate(0, 1.0);

        // Two non-improving steps (counter: 1, 2).
        sched.learning_rate(1, 1.5);
        sched.learning_rate(2, 1.5);

        // Now improve -- counter resets to 0.
        sched.learning_rate(3, 0.5);

        // One non-improving step after improvement (counter: 1).
        sched.learning_rate(4, 0.6);

        // Second non-improving step (counter: 2, still <= patience=2).
        let lr = sched.learning_rate(5, 0.6);
        assert!(
            (lr - 0.1).abs() < 1e-12,
            "improvement should have reset counter; rate should be 0.1, got {}",
            lr,
        );
    }

    /// Plateau scheduler never drops below min_lr.
    #[test]
    fn test_plateau_min_lr() {
        let mut sched = PlateauLR::new(0.1, 0.1, 0, 0.05);

        // With patience=0, every non-improving step triggers a reduction.
        sched.learning_rate(0, 1.0); // sets best_loss = 1.0

        // Non-improving: counter goes to 1 which exceeds patience (0).
        sched.learning_rate(1, 1.0); // reduce: 0.1 * 0.1 = 0.01 → clamped to 0.05

        let lr = sched.learning_rate(2, 1.0);
        assert!(
            lr >= 0.05 - 1e-12,
            "rate should never drop below min_lr (0.05), got {}",
            lr,
        );
    }

    /// Plateau reset restores the scheduler to its initial state.
    #[test]
    fn test_plateau_reset() {
        let mut sched = PlateauLR::new(0.1, 0.5, 1, 0.001);

        // Drive the rate down.
        sched.learning_rate(0, 1.0);
        sched.learning_rate(1, 1.0);
        sched.learning_rate(2, 1.0);

        // The rate should have been reduced at least once.
        let lr_before_reset = sched.current_lr;
        assert!(
            lr_before_reset < 0.1,
            "rate should have decreased before reset, got {}",
            lr_before_reset,
        );

        sched.reset();

        let lr_after = sched.learning_rate(0, 10.0);
        assert!(
            (lr_after - 0.1).abs() < 1e-12,
            "after reset, rate should be back to initial_lr (0.1), got {}",
            lr_after,
        );
    }

    // -- Cross-cutting property tests --------------------------------------

    /// Every scheduler must return a strictly positive learning rate for any
    /// non-negative step and finite loss value.
    #[test]
    fn test_all_positive() {
        let mut schedulers: Vec<Box<dyn LRScheduler>> = vec![
            Box::new(ConstantLR::new(0.05)),
            Box::new(LinearDecayLR::new(0.1, 0.001, 100)),
            Box::new(ExponentialDecayLR::new(1.0, 0.99)),
            Box::new(CosineAnnealingLR::new(0.1, 0.001, 50)),
            Box::new(PlateauLR::new(0.1, 0.5, 5, 0.001)),
        ];

        for (i, sched) in schedulers.iter_mut().enumerate() {
            for step in 0..500 {
                let lr = sched.learning_rate(step, 1.0);
                assert!(
                    lr > 0.0,
                    "scheduler {} returned non-positive lr {} at step {}",
                    i,
                    lr,
                    step,
                );
                assert!(
                    lr.is_finite(),
                    "scheduler {} returned non-finite lr at step {}",
                    i,
                    step,
                );
            }
        }
    }
}
