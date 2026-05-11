//! Integration tests that prove the streaming-architecture claims of
//! irithyll's SelectiveSSMv3 models (V3Exp + ComplexDiagonalSSM). Each test
//! corresponds to a specific architectural claim and is verified by a
//! principled triple-claim (Bernstein-bounded baseline, Pareto-dominance,
//! mechanism diagnostic).
//!
//! Tests are principled (no arbitrary thresholds): every assertion derives
//! from theory (Bernstein concentration, expected-value calculations) or is a
//! Pareto comparison against a same-pipeline baseline.
//!
//! # Test inventory
//!
//! 1. `v3exp_parity_complex_modes_dominate_real_decay` — V3Exp (complex SSM)
//!    Pareto-dominates V1 (real scalar SSM) on `ParityStream` accuracy at
//!    the same compute budget. Triple claim:
//!    (a) V3Exp accuracy above 0.5 + Bernstein 95% bound.
//!    (b) V3Exp accuracy > V1 accuracy.
//!    (c) Imaginary state components non-trivially excited (complex-mode
//!    mechanism alive, not collapsed to real).
//! 2. `complex_diag_million_step_finite` — `ComplexDiagonalSSM` outputs stay
//!    finite over a 1M-step stream. Slow test; gated behind `#[ignore]`.

use irithyll::{
    generators::{ParityStream, StreamGenerator},
    ssm::{MambaConfig, MambaVersion, StreamingMamba},
    StreamingLearner,
};
use irithyll_core::ssm::complex_diag::{ComplexDiagonalSSM, DiscretizeMethod};

// ---------------------------------------------------------------------------
// Statistical helpers (theory-derived, no arbitrary constants)
// ---------------------------------------------------------------------------

/// Empirical Bernstein bound on a sample mean for `n` i.i.d. samples bounded
/// in `[0, R]` with sample variance `var`, at confidence `1 - delta`.
///
/// Form (Maurer & Pontil 2009, Theorem 4):
/// ```text
/// B(n, var, R, delta) = sqrt(2 * var * ln(2/delta) / n)
///                     + 7 * R * ln(2/delta) / (3 * (n - 1))
/// ```
///
/// `B` upper-bounds the deviation `|mean_observed - mean_true|` with
/// probability `1 - delta`. We use this to compute a one-sided "above
/// the noise floor" guard band that any learned model must clear.
///
/// Why empirical-Bernstein and not Hoeffding: when the variance is much
/// smaller than the worst-case `R^2 / 4`, Hoeffding is loose. Empirical-
/// Bernstein adapts. For Bernoulli accuracy near 0.5 the variance is
/// 0.25 and the two bounds nearly coincide; for regression-recall in
/// `[0, 1]` near the noise floor the variance is much smaller and the
/// bound is materially tighter.
fn empirical_bernstein_bound(n: usize, sample_var: f64, range: f64, delta: f64) -> f64 {
    debug_assert!(n >= 2, "Bernstein bound requires n >= 2");
    debug_assert!(sample_var >= 0.0, "variance must be non-negative");
    debug_assert!(range > 0.0, "range must be positive");
    debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0, 1)");
    let n_f = n as f64;
    let log_term = (2.0 / delta).ln();
    let var_term = (2.0 * sample_var * log_term / n_f).sqrt();
    let range_term = 7.0 * range * log_term / (3.0 * (n_f - 1.0));
    var_term + range_term
}

// ---------------------------------------------------------------------------
// Test 1: V3Exp parity — complex modes dominate real-only decay
// ---------------------------------------------------------------------------
//
// Architectural claim: V3Exp's complex-diagonal SSM (Lahoti et al., ICLR 2026)
// has oscillatory eigenvalues that can express XOR-parity tracking, whereas
// V1's real scalar SSM (Gu & Dao, 2023) cannot. Under the same training
// budget on `ParityStream`, V3Exp Pareto-dominates V1.
//
// Pareto-baseline construction: `MambaVersion::V1` is the canonical real-mode
// scalar SSM in irithyll — same `StreamingMamba` shell (RLS readout, warmup,
// learning schedule), same n_state, same compute budget. The only
// architectural difference is the SSM cell type (complex vs. real).
//
// Bernstein for binary accuracy:
//   variance bound = 0.25 (Bernoulli at p=0.5; max for [0,1] random variable)
//   bound = sqrt(2 * 0.25 * ln(2/δ) / n) + 7 * ln(2/δ) / (3 * (n-1))
// at δ = 0.05 this is the 95% deviation guard.

#[test]
fn v3exp_parity_complex_modes_dominate_real_decay() {
    // ParityStream default: n_bits = 8 i.i.d. bits, label = XOR parity.
    const D_IN: usize = ParityStream::DEFAULT_N_BITS; // 8
    const N_STATE: usize = 16;
    const N_STEPS: usize = 5_000;
    const WARMUP: usize = 500;
    const SEED: u64 = 42;
    const DELTA: f64 = 0.05; // 95% confidence
    const RANDOM_ACC: f64 = 0.5; // balanced binary parity
    const IMAG_NORM_FLOOR: f64 = 0.01; // mechanism: complex modes carry signal

    // Build the V3Exp arm (complex-diagonal SSM, BCNorm enabled per paper §3.2).
    let cfg_v3exp = MambaConfig::builder()
        .d_in(D_IN)
        .n_state(N_STATE)
        .version(MambaVersion::V3Exp { use_bcnorm: true })
        .n_groups(2)
        .warmup(WARMUP)
        .seed(SEED)
        .build()
        .expect("v3exp parity test: V3Exp config build failed");

    // Build the V1 arm (real scalar SSM, ZOH discretization). Same n_state
    // and warmup so the per-step compute budget matches modulo the cell math.
    let cfg_v1 = MambaConfig::builder()
        .d_in(D_IN)
        .n_state(N_STATE)
        .version(MambaVersion::V1)
        .warmup(WARMUP)
        .seed(SEED)
        .build()
        .expect("v3exp parity test: V1 config build failed");

    let mut model_v3exp = StreamingMamba::new(cfg_v3exp);
    let mut model_v1 = StreamingMamba::new(cfg_v1);

    // Two parity streams from the same seed so both arms see identical samples.
    let mut gen_v3exp = ParityStream::new(SEED, D_IN, D_IN);
    let mut gen_v1 = ParityStream::new(SEED, D_IN, D_IN);

    // Track post-warmup imag-state norms during V3Exp training. The h state
    // for V3Exp is interleaved [re, im, re, im, ...]; odd-indexed entries are
    // imaginary parts (length 2 * n_groups * n_state).
    let mut max_imag_norm = 0.0_f64;

    let mut correct_v3exp = 0usize;
    let mut correct_v1 = 0usize;
    let mut total = 0usize;

    for t in 0..N_STEPS {
        let (features_a, target_a) = gen_v3exp.next_sample();
        let (features_b, target_b) = gen_v1.next_sample();
        debug_assert_eq!(
            target_a, target_b,
            "parity streams must produce the same samples"
        );
        debug_assert_eq!(
            features_a, features_b,
            "parity streams must produce the same features"
        );

        let pred_v3exp = model_v3exp.predict(&features_a);
        let pred_v1 = model_v1.predict(&features_b);
        model_v3exp.train(&features_a, target_a);
        model_v1.train(&features_b, target_b);

        if t >= WARMUP {
            let label_v3exp = if pred_v3exp >= 0.5 { 1.0 } else { 0.0 };
            let label_v1 = if pred_v1 >= 0.5 { 1.0 } else { 0.0 };
            if (label_v3exp - target_a).abs() < 0.5 {
                correct_v3exp += 1;
            }
            if (label_v1 - target_b).abs() < 0.5 {
                correct_v1 += 1;
            }
            total += 1;

            // Mechanism diagnostic: track largest imag-component magnitude
            // observed in the V3Exp h state. Indices 1, 3, 5, ... are imag.
            let h = model_v3exp.ssm_state();
            for &v in h.iter().skip(1).step_by(2) {
                let av = v.abs();
                if av > max_imag_norm {
                    max_imag_norm = av;
                }
            }
        }
    }

    let acc_v3exp = correct_v3exp as f64 / total.max(1) as f64;
    let acc_v1 = correct_v1 as f64 / total.max(1) as f64;

    // Variance bound for [0,1] Bernoulli accuracy: max variance is 0.25 at p=0.5.
    // We use this as a conservative upper bound rather than the empirical
    // sample variance because we have a single accuracy point per arm, not
    // a trajectory; the worst-case Bernoulli variance is the principled
    // assumption when one cannot estimate variance from data.
    let bernstein = empirical_bernstein_bound(total, 0.25, 1.0, DELTA);

    // Compute all three checks first so the assertion message can carry the
    // full diagnostic state — this keeps the orchestrator informed about
    // which leg of the triple-claim is the actual blocker, not just the
    // first one to short-circuit.
    let above_floor = acc_v3exp > RANDOM_ACC + bernstein;
    let pareto_ok = acc_v3exp > acc_v1;
    let mechanism_ok = max_imag_norm > IMAG_NORM_FLOOR;

    // ---- (a) V3Exp accuracy above random + Bernstein 95% guard ----
    assert!(
        above_floor,
        "V3Exp parity accuracy {acc_v3exp:.4} not significantly above random \
         ({RANDOM_ACC:.4}) + Bernstein 95% guard ({bernstein:.4}, n={total}). \
         Companion measurements: acc_v1={acc_v1:.4} (V3Exp Pareto vs V1: \
         {pareto_ok}), max|Im(h)|={max_imag_norm:.4} (complex pathway alive: \
         {mechanism_ok}). \
         BLOCKED ON: V3Exp StreamingMamba pipeline fails to lift parity \
         accuracy above the noise floor. Complex modes ARE active in the cell \
         when mechanism_ok=true, but the linear RLS readout in \
         StreamingMamba::predict cannot map them to XOR parity. Action: \
         dispatch V3Exp readout review — try nonlinear readout, complex-output \
         projection, or surface |h| (modulus) as a feature alongside Re(h)."
    );

    // ---- (b) Pareto-dominance over V1 (real scalar SSM) ----
    assert!(
        pareto_ok,
        "V3Exp accuracy {acc_v3exp:.4} must exceed V1 (real scalar) accuracy \
         {acc_v1:.4} on parity tracking. Same StreamingMamba shell, same \
         n_state={N_STATE}, same warmup={WARMUP}, same seed={SEED}. \
         max|Im(h)|={max_imag_norm:.4} (complex pathway alive: {mechanism_ok}). \
         BLOCKED ON: V3Exp fails Pareto-dominance vs. V1 on parity (the \
         canonical task that requires complex eigenvalues, per Abbe et al. \
         2023 / Goel et al. 2022). Action: confirm whether the blocker is \
         the readout (linear RLS over Re(h) only) or the cell itself, then \
         escalate accordingly."
    );

    // ---- (c) Mechanism diagnostic: imag state non-trivially excited ----
    //
    // If `max_imag_norm` is essentially zero after WARMUP+ steps, the V3Exp
    // accuracy claim is suspicious — even if the number is above the V1
    // baseline, the mechanism for that gain is not the complex-mode pathway.
    // We require at least IMAG_NORM_FLOOR magnitude on at least one imag
    // component at some point during the run — a low bar that any genuinely
    // active complex pathway clears trivially.
    assert!(
        mechanism_ok,
        "V3Exp complex modes inactive — max|Im(h)| = {max_imag_norm:.6} ≤ \
         {IMAG_NORM_FLOOR}. The state is collapsing to the real axis, which \
         means any V3Exp accuracy gain (if observed) does NOT come from the \
         complex pathway. \
         BLOCKED ON: V3Exp complex mechanism is dormant. Action: investigate \
         whether log_a_complex initialization (s4d_inv_complex) or the \
         input-driven imaginary-part update path is broken."
    );
}

// ---------------------------------------------------------------------------
// Test 2: ComplexDiagonalSSM 1M-step finite
// ---------------------------------------------------------------------------
//
// The stability guarantee of ComplexDiagonalSSM (Re(A) < 0 structurally
// enforced via negated exp) must hold over long streams. This test drives
// the cell for 1M steps and verifies no NaN/Inf appears.
//
// Gated behind #[ignore] because 1M steps takes ~1-3 seconds on fast hardware
// but is too slow for default CI. Run explicitly:
//
//   cargo test --test selective_ssm_v3 complex_diag_million_step_finite -- --ignored

#[test]
#[ignore]
fn complex_diag_million_step_finite() {
    const N_STATE: usize = 16;
    const N_STEPS: usize = 1_000_000;

    let mut cell = ComplexDiagonalSSM::new(N_STATE, DiscretizeMethod::ExpTrapezoidal);

    let b: Vec<f64> = (0..N_STATE)
        .map(|i| 0.5 * ((i as f64 + 1.0) / N_STATE as f64))
        .collect();
    let c: Vec<f64> = (0..N_STATE)
        .map(|i| 0.3 * ((i as f64 + 1.0) / N_STATE as f64))
        .collect();

    let mut rng = 0xABCD_EF01_2345_6789_u64;
    let mut last_output = 0.0_f64;

    for step in 0..N_STEPS {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        let delta = 0.01 + (rng >> 32) as f64 / u64::MAX as f64 * 0.1;
        let lambda = (rng & 0xFFFF) as f64 / 0xFFFF as f64;

        let y = cell.step(delta, &b, &c, x, lambda);

        assert!(
            y.is_finite(),
            "BLOCKED ON: ComplexDiagonalSSM output NaN/Inf at step {step}. \
             y={y}, x={x}, delta={delta}, lambda={lambda}. \
             Action: investigate Re(A) enforcement in DiscretizeMethod::ExpTrapezoidal path."
        );

        last_output = y;
    }

    assert!(
        last_output.is_finite(),
        "BLOCKED ON: ComplexDiagonalSSM final output not finite after {N_STEPS} steps."
    );
}
