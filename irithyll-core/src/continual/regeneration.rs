//! Continual Backpropagation: neuron regeneration (Dohare et al., Nature 2024).
//!
//! Tracks per-neuron utility and periodically reinitializes the least useful
//! neurons, preventing loss of plasticity in long-running streaming models.
//!
//! # The Mechanism
//!
//! Deep networks gradually lose plasticity (the ability to learn new things)
//! during continual learning. Continual backpropagation counters this by:
//!
//! 1. Tracking **utility** of each neuron as an EWMA of |activation|
//!    (approximated from gradient magnitude)
//! 2. Periodically (every `regen_interval` samples) identifying the bottom
//!    `regen_fraction` least useful neurons
//! 3. Reinitializing their parameters with fresh random values
//! 4. Resetting their utility tracker to the population mean
//!
//! This keeps the network "fresh" — low-utility neurons get recycled into
//! potentially useful ones instead of becoming dead weight.
//!
//! # References
//!
//! - Dohare et al., "Loss of Plasticity in Deep Continual Learning",
//!   Nature 632, 768–774, 2024.

use alloc::vec::Vec;

use super::ContinualStrategy;
use crate::drift::DriftSignal;
use crate::math::{abs, ceil};

/// Neuron regeneration strategy for maintaining plasticity.
///
/// Tracks utility of each "neuron" (conceptually a group of parameters)
/// and periodically reinitializes the least useful ones with fresh random values.
///
/// # How It Works
///
/// 1. Each neuron has a utility score = EWMA of |activation| (approximated
///    from gradient magnitude across the group)
/// 2. Every `regen_interval` samples, the bottom `regen_fraction` neurons
///    get regenerated
/// 3. Regenerated neurons have their parameters set to small random values
/// 4. Their utility is reset to the mean utility (giving them a fair chance)
///
/// # Parameter Grouping
///
/// The strategy treats parameters in groups of `group_size`. Each group
/// represents one "neuron" (e.g., all incoming weights to a hidden unit).
/// Utility is tracked per group, and regeneration reinitializes entire groups.
///
/// # Example
///
/// ```
/// use irithyll_core::continual::{NeuronRegeneration, ContinualStrategy};
///
/// // 100 params, groups of 10 (10 neurons), regenerate 10% every 50 steps
/// let mut regen = NeuronRegeneration::new(100, 10, 0.10, 50, 0.99, 42);
///
/// let mut params = vec![0.5; 100];
/// let mut grads = vec![0.01; 100];
///
/// // Run 50 updates to trigger regeneration
/// for _ in 0..50 {
///     regen.pre_update(&params, &mut grads);
///     regen.post_update(&params);
///     // Apply reinitialization for any regenerated neurons
///     regen.force_regenerate(&mut params);
/// }
/// ```
pub struct NeuronRegeneration {
    /// Per-neuron utility (EWMA of |activation|), one per group.
    utility: Vec<f64>,
    /// EWMA decay for utility tracking.
    utility_alpha: f64,
    /// Fraction of neurons to regenerate each cycle (0.0..1.0).
    regen_fraction: f64,
    /// Regenerate every N samples.
    regen_interval: u64,
    /// Number of parameters per neuron group.
    group_size: usize,
    /// Total parameter count.
    n_params: usize,
    /// Update counter (incremented each `pre_update` call).
    n_updates: u64,
    /// xorshift64 PRNG state for reinitialization.
    rng_state: u64,
    /// Which groups were regenerated in the most recent cycle.
    regenerated_mask: Vec<bool>,
}

/// Generate a pseudorandom u64 using xorshift64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a small random value in approximately [-0.01, 0.01].
#[inline]
fn small_random(state: &mut u64) -> f64 {
    let bits = xorshift64(state);
    (bits as f64 / u64::MAX as f64 - 0.5) * 0.02
}

impl NeuronRegeneration {
    /// Create a new neuron regeneration strategy.
    ///
    /// # Arguments
    ///
    /// * `n_params` — total number of parameters
    /// * `group_size` — parameters per neuron group (must be >= 1)
    /// * `regen_fraction` — fraction of neurons to regenerate per cycle (0.0..1.0)
    /// * `regen_interval` — number of update steps between regeneration cycles
    /// * `utility_alpha` — EWMA decay for utility tracking (higher = more smoothing)
    /// * `seed` — PRNG seed for reinitialization
    ///
    /// # Panics
    ///
    /// Panics if `group_size` is 0 or if `n_params` is 0.
    pub fn new(
        n_params: usize,
        group_size: usize,
        regen_fraction: f64,
        regen_interval: u64,
        utility_alpha: f64,
        seed: u64,
    ) -> Self {
        assert!(group_size > 0, "group_size must be >= 1");
        assert!(n_params > 0, "n_params must be >= 1");
        let n_groups = n_params.div_ceil(group_size);
        Self {
            utility: alloc::vec![0.0; n_groups],
            utility_alpha,
            regen_fraction: regen_fraction.clamp(0.0, 1.0),
            regen_interval: regen_interval.max(1),
            group_size,
            n_params,
            n_updates: 0,
            rng_state: if seed == 0 { 1 } else { seed },
            regenerated_mask: alloc::vec![false; n_groups],
        }
    }

    /// Create with sensible defaults: group_size=1, fraction=0.01, interval=1000, alpha=0.99.
    pub fn with_defaults(n_params: usize) -> Self {
        Self::new(n_params, 1, 0.01, 1000, 0.99, 0xDEAD_BEEF)
    }

    /// Current utility per group.
    pub fn utility(&self) -> &[f64] {
        &self.utility
    }

    /// Number of neuron groups.
    pub fn n_groups(&self) -> usize {
        self.utility.len()
    }

    /// Whether a given group was regenerated in the most recent cycle.
    ///
    /// Returns `false` if `group_idx` is out of range.
    pub fn was_regenerated(&self, group_idx: usize) -> bool {
        self.regenerated_mask
            .get(group_idx)
            .copied()
            .unwrap_or(false)
    }

    /// Manually trigger regeneration now, reinitializing the bottom fraction
    /// of neurons by utility directly in `params`.
    pub fn force_regenerate(&mut self, params: &mut [f64]) {
        self.perform_regeneration(params);
    }

    /// Number of update steps performed so far.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the utility threshold below which neurons get regenerated.
    /// Returns the value at the `regen_fraction` percentile of utility scores.
    fn utility_threshold(&self) -> f64 {
        let n = self.utility.len();
        if n == 0 {
            return 0.0;
        }

        // Number of groups to regenerate (at least 1 if fraction > 0 and we
        // have groups).
        let k = (ceil(n as f64 * self.regen_fraction) as usize)
            .max(1)
            .min(n);

        // Partial selection: find the k-th smallest utility.
        // For moderate n (typical hidden layer sizes), a simple sorted copy is fine.
        let mut sorted: Vec<f64> = self.utility.clone();
        // Simple insertion sort — n_groups is typically small (< 1000).
        for i in 1..sorted.len() {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && sorted[j - 1] > key {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
        }

        sorted[k - 1]
    }

    /// Compute mean utility across all groups.
    fn mean_utility(&self) -> f64 {
        if self.utility.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.utility.iter().sum();
        sum / self.utility.len() as f64
    }

    /// Perform regeneration: mark bottom-fraction groups and reinitialize their
    /// parameters with small random values.
    fn perform_regeneration(&mut self, params: &mut [f64]) {
        let n_groups = self.n_groups();
        if n_groups == 0 {
            return;
        }

        let threshold = self.utility_threshold();
        let mean_util = self.mean_utility();

        // Count how many we plan to regenerate (respect regen_fraction ceiling).
        let target_count = (ceil(n_groups as f64 * self.regen_fraction) as usize)
            .max(1)
            .min(n_groups);
        let mut regen_count = 0;

        // Clear previous mask.
        for m in self.regenerated_mask.iter_mut() {
            *m = false;
        }

        for g in 0..n_groups {
            if regen_count >= target_count {
                break;
            }
            if self.utility[g] <= threshold {
                self.regenerated_mask[g] = true;
                regen_count += 1;

                // Reinitialize parameters for this group.
                let start = g * self.group_size;
                let end = (start + self.group_size).min(params.len());
                for p in &mut params[start..end] {
                    *p = small_random(&mut self.rng_state);
                }

                // Reset utility to mean (gives regenerated neurons a fair chance).
                self.utility[g] = mean_util;
            }
        }
    }

    /// Mark bottom-fraction groups for regeneration and zero their gradients
    /// (they will be reinitialized in `post_update`, not updated via gradient).
    fn mark_for_regeneration(&mut self, gradients: &mut [f64]) {
        let n_groups = self.n_groups();
        if n_groups == 0 {
            return;
        }

        let threshold = self.utility_threshold();
        let mean_util = self.mean_utility();

        let target_count = (ceil(n_groups as f64 * self.regen_fraction) as usize)
            .max(1)
            .min(n_groups);
        let mut regen_count = 0;

        // Clear previous mask.
        for m in self.regenerated_mask.iter_mut() {
            *m = false;
        }

        for g in 0..n_groups {
            if regen_count >= target_count {
                break;
            }
            if self.utility[g] <= threshold {
                self.regenerated_mask[g] = true;
                regen_count += 1;

                // Zero gradients for this group — they will be reinitialized, not
                // updated via gradient descent.
                let start = g * self.group_size;
                let end = (start + self.group_size).min(gradients.len());
                for grad in &mut gradients[start..end] {
                    *grad = 0.0;
                }

                // Reset utility to mean.
                self.utility[g] = mean_util;
            }
        }
    }
}

impl ContinualStrategy for NeuronRegeneration {
    fn pre_update(&mut self, params: &[f64], gradients: &mut [f64]) {
        let _ = params; // utility derived from gradients, not current params
        let n_groups = self.n_groups();

        // Step 1: Update utility EWMA for each group from gradient magnitudes.
        for g in 0..n_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(gradients.len());
            let count = end - start;
            if count == 0 {
                continue;
            }

            // Group utility = mean |gradient| across the group's parameters.
            let mut sum = 0.0;
            for &grad in &gradients[start..end] {
                sum += abs(grad);
            }
            let group_mag = sum / count as f64;

            // EWMA update.
            self.utility[g] =
                self.utility_alpha * self.utility[g] + (1.0 - self.utility_alpha) * group_mag;
        }

        // Step 2: Increment counter.
        self.n_updates += 1;

        // Step 3: If at regeneration interval, mark bottom fraction for regen.
        if self.n_updates % self.regen_interval == 0 {
            self.mark_for_regeneration(gradients);
        }
    }

    fn post_update(&mut self, _params: &[f64]) {
        // If regeneration happened, reinitialize marked groups.
        let any_regenerated = self.regenerated_mask.iter().any(|&m| m);
        if !any_regenerated {
            return;
        }

        // The trait gives &[f64] (immutable), so we cannot reinitialize params
        // here directly. Instead we advance the PRNG for consistency and clear
        // the mask. Callers must use `force_regenerate(&mut params)` after
        // `post_update` to apply the actual reinitialization.

        let n_groups = self.n_groups();
        for g in 0..n_groups {
            if self.regenerated_mask[g] {
                // Advance PRNG state for each parameter in the group, so the
                // random sequence stays consistent regardless of call pattern.
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.n_params);
                for _ in start..end {
                    let _ = xorshift64(&mut self.rng_state);
                }
            }
        }

        // Clear the mask after processing.
        for m in self.regenerated_mask.iter_mut() {
            *m = false;
        }
    }

    fn on_drift(&mut self, _params: &[f64], signal: DriftSignal) {
        match signal {
            DriftSignal::Drift => {
                // Force immediate regeneration marking regardless of interval.
                // Caller should follow with `force_regenerate` for param mutation.
                let n_groups = self.n_groups();
                if n_groups == 0 {
                    return;
                }

                let threshold = self.utility_threshold();
                let mean_util = self.mean_utility();
                let target_count = (ceil(n_groups as f64 * self.regen_fraction) as usize)
                    .max(1)
                    .min(n_groups);
                let mut regen_count = 0;

                for m in self.regenerated_mask.iter_mut() {
                    *m = false;
                }

                for g in 0..n_groups {
                    if regen_count >= target_count {
                        break;
                    }
                    if self.utility[g] <= threshold {
                        self.regenerated_mask[g] = true;
                        regen_count += 1;
                        self.utility[g] = mean_util;
                    }
                }
            }
            DriftSignal::Warning | DriftSignal::Stable => {
                // No action on warning or stable signals.
            }
        }
    }

    fn n_params(&self) -> usize {
        self.n_params
    }

    fn reset(&mut self) {
        for u in self.utility.iter_mut() {
            *u = 0.0;
        }
        for m in self.regenerated_mask.iter_mut() {
            *m = false;
        }
        self.n_updates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn initially_zero_utility() {
        let regen = NeuronRegeneration::with_defaults(10);
        assert_eq!(
            regen.n_groups(),
            10,
            "should have 10 groups with group_size=1"
        );
        for &u in regen.utility() {
            assert!(
                u == 0.0,
                "all utility values should start at 0.0, got {}",
                u
            );
        }
    }

    #[test]
    fn utility_tracks_gradient_magnitude() {
        let mut regen = NeuronRegeneration::new(4, 1, 0.01, 10_000, 0.5, 42);
        let params = vec![1.0; 4];
        // Feed large gradients to group 0, small to others.
        let mut grads = vec![0.0; 4];
        grads[0] = 10.0;
        grads[1] = 0.001;
        grads[2] = 0.001;
        grads[3] = 0.001;

        for _ in 0..20 {
            regen.pre_update(&params, &mut grads);
        }

        let u = regen.utility();
        assert!(
            u[0] > u[1],
            "group 0 utility ({}) should exceed group 1 ({}), larger gradients mean higher utility",
            u[0],
            u[1]
        );
        assert!(
            u[0] > u[2],
            "group 0 utility ({}) should exceed group 2 ({})",
            u[0],
            u[2]
        );
    }

    #[test]
    fn regeneration_triggers_at_interval() {
        let interval = 5;
        let mut regen = NeuronRegeneration::new(10, 1, 0.5, interval, 0.99, 42);
        let params = vec![1.0; 10];
        let mut grads = vec![0.01; 10];

        // Before interval: no regeneration.
        for _ in 0..(interval - 1) {
            regen.pre_update(&params, &mut grads);
            assert!(
                !regen.regenerated_mask.iter().any(|&m| m),
                "no regeneration should happen before regen_interval is reached"
            );
        }

        // At interval: regeneration should trigger.
        regen.pre_update(&params, &mut grads);
        assert!(
            regen.regenerated_mask.iter().any(|&m| m),
            "regeneration should trigger at step {}",
            interval
        );
    }

    #[test]
    fn bottom_fraction_gets_regenerated() {
        // 10 groups, regenerate 20% = 2 groups.
        let mut regen = NeuronRegeneration::new(10, 1, 0.20, 5, 0.5, 42);
        let params = vec![1.0; 10];

        // Give groups 0..8 high gradients, groups 8..10 near zero.
        let mut grads = vec![5.0; 10];
        grads[8] = 0.0001;
        grads[9] = 0.0001;

        // Build up utility difference over several steps (not hitting interval yet).
        for _ in 0..4 {
            regen.pre_update(&params, &mut grads);
        }

        // Step 5 triggers regeneration.
        regen.pre_update(&params, &mut grads);

        // Groups 8 and 9 should be regenerated (lowest utility).
        assert!(
            regen.was_regenerated(8),
            "group 8 (low utility) should be regenerated"
        );
        assert!(
            regen.was_regenerated(9),
            "group 9 (low utility) should be regenerated"
        );
        // High-utility groups should not be regenerated.
        assert!(
            !regen.was_regenerated(0),
            "group 0 (high utility) should not be regenerated"
        );
    }

    #[test]
    fn regenerated_params_are_reinitialized() {
        let mut regen = NeuronRegeneration::new(4, 2, 0.5, 1, 0.5, 42);
        let mut params = vec![100.0; 4];

        // Make group 0 low utility, group 1 high utility.
        let mut grads = vec![0.0; 4];
        grads[0] = 0.0;
        grads[1] = 0.0;
        grads[2] = 10.0;
        grads[3] = 10.0;

        regen.pre_update(&params, &mut grads);

        // Use force_regenerate to actually mutate params.
        regen.force_regenerate(&mut params);

        // Group 0 params should be reinitialized to small values, not 100.0.
        assert!(
            abs(params[0]) < 1.0,
            "regenerated param[0] should be small random, got {}",
            params[0]
        );
        assert!(
            abs(params[1]) < 1.0,
            "regenerated param[1] should be small random, got {}",
            params[1]
        );
    }

    #[test]
    fn non_regenerated_params_unchanged() {
        let mut regen = NeuronRegeneration::new(4, 2, 0.5, 1, 0.5, 42);
        let mut params = vec![100.0; 4];

        // Make group 0 low utility, group 1 high utility.
        let mut grads = vec![0.0; 4];
        grads[0] = 0.0;
        grads[1] = 0.0;
        grads[2] = 10.0;
        grads[3] = 10.0;

        regen.pre_update(&params, &mut grads);

        // Save group 1 values before regeneration.
        let before_2 = params[2];
        let before_3 = params[3];

        regen.force_regenerate(&mut params);

        // Group 1 params should be unchanged.
        assert_eq!(
            params[2], before_2,
            "non-regenerated param[2] should be unchanged"
        );
        assert_eq!(
            params[3], before_3,
            "non-regenerated param[3] should be unchanged"
        );
    }

    #[test]
    fn drift_forces_immediate_regeneration() {
        let mut regen = NeuronRegeneration::new(10, 1, 0.20, 1_000_000, 0.5, 42);
        let params = vec![1.0; 10];

        // Build up some utility variance.
        for _ in 0..10 {
            let mut grads = vec![5.0; 10];
            grads[8] = 0.0001;
            grads[9] = 0.0001;
            regen.pre_update(&params, &mut grads);
        }

        // Normally no regeneration (interval is 1 million).
        assert!(
            !regen.regenerated_mask.iter().any(|&m| m),
            "should not have regenerated yet (interval is 1_000_000)"
        );

        // Drift signal should force regeneration marking.
        regen.on_drift(&params, DriftSignal::Drift);
        assert!(
            regen.regenerated_mask.iter().any(|&m| m),
            "drift signal should force immediate regeneration marking"
        );
        assert!(
            regen.was_regenerated(8) || regen.was_regenerated(9),
            "low-utility groups should be marked for regeneration on drift"
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut regen = NeuronRegeneration::new(10, 1, 0.5, 2, 0.5, 42);
        let params = vec![1.0; 10];
        let mut grads = vec![1.0; 10];

        // Accumulate some state.
        regen.pre_update(&params, &mut grads);
        regen.pre_update(&params, &mut grads);
        assert!(
            regen.n_updates() > 0,
            "should have incremented update counter"
        );

        regen.reset();

        assert_eq!(regen.n_updates(), 0, "counter should be zeroed after reset");
        for &u in regen.utility() {
            assert!(u == 0.0, "utility should be 0.0 after reset, got {}", u);
        }
        for g in 0..regen.n_groups() {
            assert!(
                !regen.was_regenerated(g),
                "regenerated mask should be cleared after reset"
            );
        }
    }

    #[test]
    fn group_size_groups_params_correctly() {
        // 12 params, groups of 4 = 3 neuron groups. Fraction 0.33 -> ceil(3*0.33)=1 group.
        let mut regen = NeuronRegeneration::new(12, 4, 0.33, 5, 0.5, 42);
        assert_eq!(regen.n_groups(), 3, "12 params / 4 group_size = 3 groups");

        let params = vec![1.0; 12];

        // Give group 0 (params 0..4) zero gradients, group 1 and 2 high gradients.
        let mut grads = vec![0.0; 12];
        for g in grads.iter_mut().skip(4) {
            *g = 5.0;
        }

        // Run to regeneration trigger.
        for _ in 0..5 {
            regen.pre_update(&params, &mut grads);
        }

        // Group 0 should be regenerated (lowest utility), groups 1 & 2 should not.
        assert!(
            regen.was_regenerated(0),
            "group 0 (zero gradients) should be regenerated"
        );
        assert!(
            !regen.was_regenerated(1),
            "group 1 (high gradients) should not be regenerated"
        );
        assert!(
            !regen.was_regenerated(2),
            "group 2 (high gradients) should not be regenerated"
        );

        // Verify the actual param reinitialization covers all 4 params in the group.
        let mut params_mut = vec![100.0; 12];
        regen.force_regenerate(&mut params_mut);
        for (i, &p) in params_mut.iter().enumerate().take(4) {
            assert!(
                abs(p) < 1.0,
                "all params in regenerated group should be reinitialized, param[{}] = {}",
                i,
                p
            );
        }
        for (i, &p) in params_mut.iter().enumerate().skip(4) {
            assert_eq!(
                p, 100.0,
                "params outside regenerated group should be untouched, param[{}] = {}",
                i, p
            );
        }
    }

    #[test]
    fn high_utility_neurons_preserved() {
        // 5 groups, regenerate 20% = 1 group.
        let mut regen = NeuronRegeneration::new(5, 1, 0.20, 3, 0.5, 42);
        let params = vec![1.0; 5];

        // Group 0 gets massive gradients (high utility), all others get tiny.
        let mut grads = vec![0.001; 5];
        grads[0] = 100.0;

        for _ in 0..3 {
            regen.pre_update(&params, &mut grads);
        }

        // Group 0 should never be regenerated (it has the highest utility).
        assert!(
            !regen.was_regenerated(0),
            "high-utility group 0 should be preserved, not regenerated"
        );
        // At least one of the low-utility groups should be regenerated.
        let any_low_regenerated = (1..5).any(|g| regen.was_regenerated(g));
        assert!(
            any_low_regenerated,
            "at least one low-utility group should be regenerated"
        );
    }

    #[test]
    fn warning_and_stable_signals_are_noop() {
        let mut regen = NeuronRegeneration::new(10, 1, 0.5, 1_000_000, 0.5, 42);
        let params = vec![1.0; 10];
        let mut grads = vec![1.0; 10];
        grads[9] = 0.0;

        // Build some utility.
        for _ in 0..5 {
            regen.pre_update(&params, &mut grads);
        }

        regen.on_drift(&params, DriftSignal::Warning);
        assert!(
            !regen.regenerated_mask.iter().any(|&m| m),
            "Warning signal should not trigger regeneration"
        );

        regen.on_drift(&params, DriftSignal::Stable);
        assert!(
            !regen.regenerated_mask.iter().any(|&m| m),
            "Stable signal should not trigger regeneration"
        );
    }

    #[test]
    fn xorshift_produces_distinct_values() {
        let mut state: u64 = 42;
        let mut vals: Vec<f64> = Vec::new();
        for _ in 0..100 {
            vals.push(small_random(&mut state));
        }
        // All values should be in [-0.01, 0.01].
        for &v in &vals {
            assert!(
                (-0.011..=0.011).contains(&v),
                "small_random should produce values in ~[-0.01, 0.01], got {}",
                v
            );
        }
        // Values should not all be identical.
        let first = vals[0];
        let all_same = vals.iter().all(|&v| (v - first).abs() < 1e-15);
        assert!(!all_same, "PRNG should produce distinct values");
    }
}
