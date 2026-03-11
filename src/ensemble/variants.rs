//! SGBT computational variants (Gunasekara et al., 2024).
//!
//! Three variants control how many times each tree trains on a given instance:
//!
//! - **Standard**: every tree processes every instance exactly once.
//! - **Skip (SGBT-SK)**: each tree randomly skips 1/k of instances, reducing
//!   computational cost at a small accuracy trade-off.
//! - **MultipleIterations (SGBT-MI)**: each tree trains `ceil(|hessian| * multiplier)`
//!   times per instance, emphasising hard-to-fit regions of the loss surface.

/// SGBT computational variant selection.
///
/// Controls how many times a given boosting step trains its tree on each
/// incoming instance. The default is [`Standard`](SGBTVariant::Standard),
/// which processes every instance exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SGBTVariant {
    /// Standard SGBT: train each tree on every instance exactly once.
    #[default]
    Standard,

    /// SGBT-SK: randomly skip training for approximately 1/k of instances.
    ///
    /// The parameter `k` controls the skip rate. For example, `k = 10` means
    /// each instance has a ~10% chance of being skipped per boosting step.
    /// This reduces computational cost at the expense of slightly lower
    /// accuracy on rapidly shifting distributions.
    Skip {
        /// Skip denominator. Must be >= 2. An instance is skipped with
        /// probability 1/k.
        k: usize,
    },

    /// SGBT-MI: train each tree multiple times per instance.
    ///
    /// The training count is `ceil(|hessian| * multiplier)`, clamped to a
    /// minimum of 1. This causes the tree to invest more splits on instances
    /// where the loss surface has high curvature (large hessian), effectively
    /// performing importance-weighted boosting.
    MultipleIterations {
        /// Hessian multiplier. Must be > 0. Typical values: 1.0 -- 10.0.
        multiplier: f64,
    },
}

impl std::fmt::Display for SGBTVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::Skip { k } => write!(f, "Skip(k={})", k),
            Self::MultipleIterations { multiplier } => {
                write!(f, "MultipleIterations(multiplier={})", multiplier)
            }
        }
    }
}

impl SGBTVariant {
    /// Determine how many times to train a tree on this instance.
    ///
    /// Returns 0 if the instance should be skipped (SGBT-SK), 1 for standard
    /// processing, or > 1 for multiple iterations (SGBT-MI).
    ///
    /// # Arguments
    ///
    /// * `hessian` -- the second derivative of the loss for this instance.
    ///   Only used by [`MultipleIterations`](SGBTVariant::MultipleIterations);
    ///   ignored by the other variants.
    /// * `rng_state` -- mutable reference to a 64-bit xorshift state used for
    ///   skip randomisation. The caller must seed this to a non-zero value.
    ///   Only mutated by [`Skip`](SGBTVariant::Skip).
    pub fn train_count(&self, hessian: f64, rng_state: &mut u64) -> usize {
        match self {
            SGBTVariant::Standard => 1,

            SGBTVariant::Skip { k } => {
                // Xorshift64 step for fast, deterministic pseudo-random skip.
                *rng_state ^= *rng_state << 13;
                *rng_state ^= *rng_state >> 7;
                *rng_state ^= *rng_state << 17;

                if (*rng_state % (*k as u64)) == 0 {
                    0 // skip this instance
                } else {
                    1
                }
            }

            SGBTVariant::MultipleIterations { multiplier } => {
                let count = (hessian.abs() * multiplier).ceil() as usize;
                count.max(1) // always train at least once
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Standard always returns 1.
    #[test]
    fn standard_always_one() {
        let variant = SGBTVariant::Standard;
        let mut rng: u64 = 12345;

        // Various hessians -- result must always be 1.
        for &h in &[0.0, 0.5, 1.0, -2.0, 100.0, f64::MIN_POSITIVE] {
            assert_eq!(variant.train_count(h, &mut rng), 1);
        }
    }

    // 2. Skip returns 0 roughly 1/k of the time (statistical test, k=10).
    #[test]
    fn skip_skips_roughly_one_in_k() {
        let variant = SGBTVariant::Skip { k: 10 };
        let mut rng: u64 = 0xDEAD_BEEF_CAFE_1234; // non-zero seed
        let trials = 10_000;

        let mut skip_count = 0usize;
        for _ in 0..trials {
            if variant.train_count(0.0, &mut rng) == 0 {
                skip_count += 1;
            }
        }

        // Expected: ~1000 skips out of 10000 (10%).
        // Allow generous range: 5% -- 15% to avoid flaky tests.
        let skip_rate = skip_count as f64 / trials as f64;
        assert!(
            skip_rate > 0.05 && skip_rate < 0.15,
            "expected ~10% skip rate, got {:.2}% ({} / {})",
            skip_rate * 100.0,
            skip_count,
            trials
        );
    }

    // 3. Skip never skips when k=1 would always skip (degenerate case).
    //    Actually k=1 means skip with probability 1/1 = 100%. We verify
    //    this behaves correctly: every call returns 0.
    #[test]
    fn skip_k_one_always_skips() {
        let variant = SGBTVariant::Skip { k: 1 };
        let mut rng: u64 = 42;

        for _ in 0..100 {
            assert_eq!(variant.train_count(0.0, &mut rng), 0);
        }
    }

    // 4. MultipleIterations: hessian=0.5, multiplier=10 -> ceil(5.0) = 5.
    #[test]
    fn mi_hessian_half_multiplier_ten() {
        let variant = SGBTVariant::MultipleIterations { multiplier: 10.0 };
        let mut rng: u64 = 1;

        let count = variant.train_count(0.5, &mut rng);
        assert_eq!(count, 5, "ceil(0.5 * 10) = 5");
    }

    // 5. MultipleIterations: negative hessian uses abs().
    #[test]
    fn mi_negative_hessian_uses_abs() {
        let variant = SGBTVariant::MultipleIterations { multiplier: 10.0 };
        let mut rng: u64 = 1;

        let count = variant.train_count(-0.5, &mut rng);
        assert_eq!(count, 5, "ceil(|-0.5| * 10) = 5");
    }

    // 6. MultipleIterations: very small hessian clamps to 1.
    #[test]
    fn mi_always_at_least_one() {
        let variant = SGBTVariant::MultipleIterations { multiplier: 1.0 };
        let mut rng: u64 = 1;

        // hessian=0.0 => ceil(0.0 * 1.0) = 0, but clamped to 1.
        assert_eq!(variant.train_count(0.0, &mut rng), 1);

        // hessian very tiny => ceil(tiny) = 1.
        assert_eq!(variant.train_count(1e-20, &mut rng), 1);
    }

    // 7. MultipleIterations: large hessian.
    #[test]
    fn mi_large_hessian() {
        let variant = SGBTVariant::MultipleIterations { multiplier: 2.0 };
        let mut rng: u64 = 1;

        // hessian=3.7, multiplier=2.0 => ceil(7.4) = 8.
        assert_eq!(variant.train_count(3.7, &mut rng), 8);
    }

    // 8. Default variant is Standard.
    #[test]
    fn default_is_standard() {
        let variant = SGBTVariant::default();
        assert_eq!(variant, SGBTVariant::Standard);
    }

    // 9. Equality / PartialEq works correctly.
    #[test]
    fn partial_eq() {
        assert_eq!(SGBTVariant::Standard, SGBTVariant::Standard);
        assert_ne!(SGBTVariant::Standard, SGBTVariant::Skip { k: 10 });
        assert_eq!(SGBTVariant::Skip { k: 5 }, SGBTVariant::Skip { k: 5 });
        assert_ne!(SGBTVariant::Skip { k: 5 }, SGBTVariant::Skip { k: 10 });
        assert_eq!(
            SGBTVariant::MultipleIterations { multiplier: 2.0 },
            SGBTVariant::MultipleIterations { multiplier: 2.0 }
        );
    }

    // 10. Clone preserves variant data.
    #[test]
    fn clone_preserves_data() {
        let original = SGBTVariant::MultipleIterations { multiplier: 7.5 };
        let cloned = original;
        assert_eq!(original, cloned);
    }

    // 11. Skip produces varied results across calls (not always the same).
    #[test]
    fn skip_is_non_degenerate() {
        let variant = SGBTVariant::Skip { k: 2 };
        let mut rng: u64 = 0xABCD_1234_5678_EF90;

        let mut saw_zero = false;
        let mut saw_one = false;
        for _ in 0..100 {
            match variant.train_count(0.0, &mut rng) {
                0 => saw_zero = true,
                1 => saw_one = true,
                _ => panic!("Skip should only return 0 or 1"),
            }
            if saw_zero && saw_one {
                break;
            }
        }
        assert!(saw_zero && saw_one, "k=2 should produce both 0 and 1");
    }
}
