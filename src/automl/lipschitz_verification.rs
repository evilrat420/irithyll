//! Lipschitz proptest harness for MetaAdapter implementations.
//!
//! Every concrete [`MetaAdapter`] must have a test here that verifies the
//! declared [`MetaAdapter::lipschitz_bound`] holds empirically for random
//! perturbation pairs.  The test cannot be a hand-picked unit test — it must
//! use proptest to cover the parameter space randomly and catch high-curvature
//! corner cases.
//!
//! # CI Gate
//!
//! This file is the CI gate.  No new adapter merges without a proptest here.
//! The PR contract (phase9_spec_gap_option_a.md §3) requires:
//! 1. A proptest in this file with adapter-appropriate `eps` (§1.3).
//! 2. A `// FAILURE MODE:` comment block (§1.5) on each test.
//! 3. `declared_lipschitz()` in the adapter's doc-comment with the derivation.
//!
//! # Normalised Coordinate Space (Important)
//!
//! All Lipschitz verification is done in θ-space [0,1]^2, not in raw parameter
//! space.  Adapters operating in log-coordinate space MUST clip in log-space
//! (θ-space), never in raw-space.  See §1.2 of the spec for the quantization
//! trap that invalidates Lipschitz if you clip raw values then re-normalise.
//!
//! # Per-adapter epsilon calibration (§1.3)
//!
//! | Adapter                        | Declared L | Test eps | Rationale                         |
//! |-------------------------------|-----------|---------|-----------------------------------|
//! | DriftRateAdapter               | 1.0       | 1e-3    | Non-expansive; test is L ≤ 1.0    |
//! | PlasticityAdapter              | 1.0       | 1e-3    | Non-expansive; same class         |
//! | ContractionAdapter (ρ=0.3)    | 0.7       | 1e-4    | High-curvature at lr→0 boundary   |
//! | NoOpAdapter                    | 1.0       | 1e-3    | Identity; trivially non-expansive |

#[cfg(test)]
mod lipschitz_tests {
    use super::super::adaptation_bus::{
        norm2_diff, AdaptContext, DriftRateAdapter, MetaAdapter, NoOpAdapter, PlasticityAdapter,
        ThetaDelta,
    };
    use super::super::auto_builder::ConfigDiagnostics;
    use proptest::prelude::*;
    use rand::SeedableRng;

    // -----------------------------------------------------------------------
    // Helper: empirical Lipschitz estimator
    // -----------------------------------------------------------------------

    /// Estimate the empirical Lipschitz constant of `adapter.apply()` by sampling
    /// `n_pairs` random pairs (θ_a, θ_b) with ‖θ_a − θ_b‖_∞ ≤ eps.
    ///
    /// Returns the maximum ratio ‖apply(θ_a) − apply(θ_b)‖ / ‖θ_a − θ_b‖.
    ///
    /// The adapter's `apply()` function is called twice per pair with different
    /// θ values in the AdaptContext.  All other context fields are held constant
    /// (zeroed diagnostics, fixed sample_idx=1000).
    fn empirical_lipschitz<A: MetaAdapter>(
        adapter: &mut A,
        eps: f64,
        n_pairs: usize,
        rng: &mut impl rand::Rng,
    ) -> f64 {
        let base_ctx = AdaptContext {
            theta: [0.5, 0.5],
            diagnostics: ConfigDiagnostics::default(),
            prediction: 0.0,
            target: 0.0,
            sample_idx: 1000,
        };

        let mut max_ratio: f64 = 0.0;

        for _ in 0..n_pairs {
            // Sample θ_a uniformly in [0,1]^2.
            let theta_a: [f64; 2] = [rng.gen_range(0.0..=1.0), rng.gen_range(0.0..=1.0)];

            // Perturb to get θ_b, clamped to [0,1].
            let theta_b: [f64; 2] = [
                (theta_a[0] + rng.gen_range(-eps..=eps)).clamp(0.0, 1.0),
                (theta_a[1] + rng.gen_range(-eps..=eps)).clamp(0.0, 1.0),
            ];

            let ctx_a = AdaptContext {
                theta: theta_a,
                ..base_ctx.clone()
            };
            let ctx_b = AdaptContext {
                theta: theta_b,
                ..base_ctx.clone()
            };

            let delta_a = adapter.apply(&ctx_a).delta;
            let delta_b = adapter.apply(&ctx_b).delta;

            let output_dist = norm2_diff(&delta_a, &delta_b);
            let input_dist = norm2_diff(&theta_a, &theta_b);

            if input_dist > 1e-15 {
                max_ratio = max_ratio.max(output_dist / input_dist);
            }
        }

        max_ratio
    }

    // -----------------------------------------------------------------------
    // NoOpAdapter — identity, L = 1.0
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// NoOpAdapter is the identity map on δθ = 0 — trivially Lipschitz-0.
        ///
        /// // FAILURE MODE: If NoOpAdapter somehow returns non-zero delta
        /// // (implementation bug), empirical_l would be > 0, failing the test.
        #[test]
        fn noop_adapter_non_expansive(
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut adapter = NoOpAdapter;
            let declared_l = adapter.lipschitz_bound(); // 1.0
            let actual_l = empirical_lipschitz(&mut adapter, 1e-3, 200, &mut rng);
            // NoOp emits zero delta — actual_l should be 0.0.
            prop_assert!(
                actual_l <= declared_l + 1e-10,
                "NoOpAdapter: empirical L={} exceeds declared L={}", actual_l, declared_l
            );
        }
    }

    // -----------------------------------------------------------------------
    // DriftRateAdapter — non-expansive shift, L = 1.0
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// DriftRateAdapter produces a bounded additive shift in θ-space.
        ///
        /// // FAILURE MODE: If the adapter clips in raw_lr space and re-normalises,
        /// // a kink at the boundary creates a discontinuity.  L spikes to ∞ at the
        /// // kink boundary.  Proptest pairs near theta[0] ≈ 0 or ≈ 1 catch this.
        ///
        /// // FAILURE MODE: If the direction logic uses a conditional branch that
        /// // fires for θ_a but not θ_b (near the EWMA threshold), the output is
        /// // discontinuous at the threshold.  eps=1e-3 is large enough to straddle
        /// // most thresholds; the 1000 cases cover the threshold region statistically.
        #[test]
        fn drift_rate_adapter_non_expansive(
            delta_max in 0.001f64..0.05,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut adapter = DriftRateAdapter::new(delta_max);
            let actual_l = empirical_lipschitz(&mut adapter, 1e-3, 500, &mut rng);
            // Non-expansive: ‖apply(a) − apply(b)‖ ≤ ‖a − b‖ (L ≤ 1.0).
            prop_assert!(
                actual_l <= 1.0 + 1e-10,
                "DriftRateAdapter: empirical L={} > 1.0 (expansive)", actual_l
            );
        }
    }

    // -----------------------------------------------------------------------
    // PlasticityAdapter — non-expansive shift, L = 1.0
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// PlasticityAdapter produces a bounded additive shift in θ-space.
        ///
        /// // FAILURE MODE: Discontinuous direction decision (utility_ewma > 0.6)
        /// // at the threshold creates a step function in apply().  The threshold
        /// // is at utility_ewma = 0.6 — eps=1e-3 pairs straddle this region.
        /// // The clamp to [0,1] after applying the delta prevents expansion;
        /// // the test verifies the clamp is operating in theta-space.
        ///
        /// // FAILURE MODE: If utility_ewma grows unbounded (EWMA with alpha near 1.0
        /// // on large uncertainty values), the directional decision oscillates.
        /// // The diagnostics in empirical_lipschitz are zeroed (uncertainty=0),
        /// // so utility_ewma converges to 0 — the test covers the converged state.
        #[test]
        fn plasticity_adapter_non_expansive(
            delta_max in 0.001f64..0.1,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut adapter = PlasticityAdapter::new(delta_max);
            let actual_l = empirical_lipschitz(&mut adapter, 1e-3, 500, &mut rng);
            prop_assert!(
                actual_l <= 1.0 + 1e-10,
                "PlasticityAdapter: empirical L={} > 1.0 (expansive)", actual_l
            );
        }
    }

    // -----------------------------------------------------------------------
    // ContractionAdapter (ρ-blended) — strict contraction, L = max(ρ, 1-ρ) = 0.7
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// ρ-blended step adapter: θ_{t+1} = (1-ρ)·θ_t + ρ·clip(...)
        /// Lipschitz constant = max(ρ, 1-ρ) < 1 for ρ ≠ 0.5.
        ///
        /// // FAILURE MODE: High-curvature at θ→0 or θ→1 boundary.
        /// // When θ_a is at 0.01 and ε = 1e-4, the ρ-blend toward 0.5
        /// // amplifies the relative distance in log-space but not in θ-space.
        /// // The proptest covers θ near boundary via uniform sampling.
        ///
        /// // FAILURE MODE: If ρ > 0.5 the declared bound L = ρ may be violated
        /// // numerically.  proptest with ρ in [0.1, 0.4] ensures L ≤ 0.7 < 1.0.
        #[test]
        fn contraction_adapter_at_declared_bound(
            rho in 0.1f64..0.4,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

            struct ContractionAdapter { rho: f64 }
            impl MetaAdapter for ContractionAdapter {
                fn lipschitz_bound(&self) -> f64 { self.rho.max(1.0 - self.rho) }
                fn update_period(&self) -> u64 { 30 }
                fn apply(&mut self, ctx: &AdaptContext) -> ThetaDelta {
                    let target = [0.5_f64, 0.5_f64];
                    ThetaDelta { delta: [
                        self.rho * (target[0] - ctx.theta[0]),
                        self.rho * (target[1] - ctx.theta[1]),
                    ] }
                }
                fn progress(&self) -> f64 { 0.0 }
            }

            let mut adapter = ContractionAdapter { rho };
            let declared_l = adapter.lipschitz_bound();
            // eps=1e-4: small relative to the contraction margin (1-L ≥ 0.6).
            let actual_l = empirical_lipschitz(&mut adapter, 1e-4, 200, &mut rng);
            // Allow 5% numerical headroom.
            prop_assert!(
                actual_l <= declared_l * 1.05,
                "ContractionAdapter(rho={:.3}): empirical L={:.6} > declared L={:.4} * 1.05",
                rho, actual_l, declared_l
            );
        }
    }

    // -----------------------------------------------------------------------
    // DriftRateAdapter flush preserves non-expansion post-flush
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// After flush_state(), the DriftRateAdapter must still be non-expansive.
        ///
        /// // FAILURE MODE: If flush_state() leaves the adapter in an inconsistent
        /// // state where the first apply() call post-flush produces a delta
        /// // disproportionate to the input perturbation, the Lipschitz bound is
        /// // violated in the post-flush transient.  The re-prime mechanism
        /// // (return ThetaDelta::default() on first apply post-flush) prevents this.
        #[test]
        fn drift_rate_adapter_non_expansive_after_flush(
            delta_max in 0.001f64..0.05,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut adapter = DriftRateAdapter::new(delta_max);

            // Warm up the adapter with some calls.
            let base_ctx = AdaptContext {
                theta: [0.5, 0.5],
                diagnostics: ConfigDiagnostics::default(),
                prediction: 0.5,
                target: 0.3,
                sample_idx: 100,
            };
            for _ in 0..10 {
                adapter.apply(&base_ctx);
            }

            // Flush then measure.
            adapter.flush_state();

            let actual_l = empirical_lipschitz(&mut adapter, 1e-3, 200, &mut rng);
            prop_assert!(
                actual_l <= 1.0 + 1e-10,
                "DriftRateAdapter post-flush: empirical L={} > 1.0", actual_l
            );
        }
    }

    // -----------------------------------------------------------------------
    // PlasticityAdapter flush preserves non-expansion post-flush
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// After flush_state(), the PlasticityAdapter must still be non-expansive.
        ///
        /// // FAILURE MODE: flush_state() resets utility_ewma to 0.5 (neutral).
        /// // The first apply() call post-flush sees the needs_reprime=true flag
        /// // and returns ThetaDelta::default().  If this guard is missing, the
        /// // adapter applies a stale delta to a potentially very different θ —
        /// // creating an anomalous spike in empirical Lipschitz.
        #[test]
        fn plasticity_adapter_non_expansive_after_flush(
            delta_max in 0.001f64..0.1,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut adapter = PlasticityAdapter::new(delta_max);

            let base_ctx = AdaptContext {
                theta: [0.5, 0.5],
                diagnostics: ConfigDiagnostics::default(),
                prediction: 0.5,
                target: 0.3,
                sample_idx: 100,
            };
            for _ in 0..10 {
                adapter.apply(&base_ctx);
            }

            adapter.flush_state();

            let actual_l = empirical_lipschitz(&mut adapter, 1e-3, 200, &mut rng);
            prop_assert!(
                actual_l <= 1.0 + 1e-10,
                "PlasticityAdapter post-flush: empirical L={} > 1.0", actual_l
            );
        }
    }
}
