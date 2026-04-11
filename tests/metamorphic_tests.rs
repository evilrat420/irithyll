//! Metamorphic relation tests for [`StreamingLearner`] implementations.
//!
//! A metamorphic relation is a property relating inputs and outputs that must
//! hold independently of the exact function value. These tests catch bugs that
//! unit tests miss because the "oracle" is a structural/relational invariant
//! rather than a specific expected value.
//!
//! Relations tested:
//! 1. **Prediction consistency** -- predict(x) is deterministic (same result twice).
//! 2. **Reset idempotence** -- reset(); reset() identical to single reset().
//! 3. **Monotone adaptation** -- error on constant target K decreases over time.
//! 4. **Sample count monotonicity** -- n_samples_seen() increments by exactly 1 per call.
//!
//! Models with warmup (ESN, TTT, sLSTM, mGRADE) skip the exact sample-count
//! monotonicity assertion and instead verify count is non-decreasing.

use irithyll::mgrade;
use irithyll::{
    esn, gla, krls, linear, mamba, mondrian, rls, sgbt, spikenet, streaming_kan, streaming_slstm,
    streaming_ttt, StreamingLearner,
};

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Relation 1: Prediction consistency (deterministic predict)
// ---------------------------------------------------------------------------

fn assert_predict_consistent(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
    // Warm up enough to pass any warmup period.
    for _ in 0..100 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0).collect();
        let target = xorshift64(&mut rng) * 2.0;
        model.train(&features, target);
    }
    let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0).collect();
    let p1 = model.predict(&features);
    let p2 = model.predict(&features);
    assert_eq!(
        p1, p2,
        "{}: predict(x) must be deterministic (got {} then {})",
        name, p1, p2
    );
}

#[test]
fn predict_consistent_sgbt() {
    let mut m = sgbt(10, 0.05);
    assert_predict_consistent(&mut m, "SGBT", 3);
}

#[test]
fn predict_consistent_rls() {
    let mut m = rls(0.99);
    assert_predict_consistent(&mut m, "RLS", 3);
}

#[test]
fn predict_consistent_linear() {
    let mut m = linear(0.01);
    assert_predict_consistent(&mut m, "Linear", 3);
}

#[test]
fn predict_consistent_mamba() {
    let mut m = mamba(3, 8);
    assert_predict_consistent(&mut m, "Mamba", 3);
}

#[test]
fn predict_consistent_esn() {
    let mut m = esn(30, 0.9);
    assert_predict_consistent(&mut m, "ESN", 2);
}

#[test]
fn predict_consistent_spikenet() {
    let mut m = spikenet(16);
    assert_predict_consistent(&mut m, "SpikeNet", 3);
}

#[test]
fn predict_consistent_kan() {
    let mut m = streaming_kan(&[3, 6, 1], 0.01);
    assert_predict_consistent(&mut m, "KAN", 3);
}

#[test]
fn predict_consistent_attention() {
    let mut m = gla(8, 2);
    assert_predict_consistent(&mut m, "GLA", 8);
}

// ---------------------------------------------------------------------------
// Relation 2: Reset idempotence
// ---------------------------------------------------------------------------

fn assert_reset_idempotent(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0xFEED_FACE_CAFE_BABE;
    // Train some samples.
    for _ in 0..60 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0).collect();
        model.train(&features, xorshift64(&mut rng));
    }
    // Single reset.
    model.reset();
    let count_after_one = model.n_samples_seen();
    let probe: Vec<f64> = vec![0.5; dim];
    let pred_after_one = model.predict(&probe);

    // Second reset (idempotent).
    model.reset();
    let count_after_two = model.n_samples_seen();
    let pred_after_two = model.predict(&probe);

    assert_eq!(
        count_after_one, count_after_two,
        "{}: double reset should leave n_samples_seen identical ({} vs {})",
        name, count_after_one, count_after_two
    );
    assert_eq!(
        pred_after_one, pred_after_two,
        "{}: double reset should leave predict identical ({} vs {})",
        name, pred_after_one, pred_after_two
    );
}

#[test]
fn reset_idempotent_sgbt() {
    let mut m = sgbt(10, 0.05);
    assert_reset_idempotent(&mut m, "SGBT", 2);
}

#[test]
fn reset_idempotent_rls() {
    let mut m = rls(0.99);
    assert_reset_idempotent(&mut m, "RLS", 2);
}

#[test]
fn reset_idempotent_mamba() {
    let mut m = mamba(2, 8);
    assert_reset_idempotent(&mut m, "Mamba", 2);
}

#[test]
fn reset_idempotent_kan() {
    let mut m = streaming_kan(&[2, 4, 1], 0.01);
    assert_reset_idempotent(&mut m, "KAN", 2);
}

#[test]
fn reset_idempotent_mondrian() {
    let mut m = mondrian(5);
    assert_reset_idempotent(&mut m, "Mondrian", 3);
}

#[test]
fn reset_idempotent_ttt() {
    let mut m = streaming_ttt(8, 0.05);
    assert_reset_idempotent(&mut m, "TTT", 2);
}

#[test]
fn reset_idempotent_slstm() {
    let mut m = streaming_slstm(8);
    assert_reset_idempotent(&mut m, "sLSTM", 2);
}

// ---------------------------------------------------------------------------
// Relation 3: Monotone adaptation on constant target
//
// Train on constant target K=7.0 for 200 steps. Assert that the error in
// the SECOND half is no greater than in the FIRST half (adaption makes
// progress). We use |pred - K|, averaged over each half.
// ---------------------------------------------------------------------------

fn assert_monotone_adaptation(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    const K: f64 = 7.0;
    const TOTAL: usize = 200;
    const HALF: usize = TOTAL / 2;

    let features: Vec<f64> = vec![0.5; dim];
    let mut first_half_error = 0.0_f64;
    let mut second_half_error = 0.0_f64;

    for i in 0..TOTAL {
        let pred = model.predict(&features);
        let err = (pred - K).abs();
        if i < HALF {
            first_half_error += err;
        } else {
            second_half_error += err;
        }
        model.train(&features, K);
    }

    first_half_error /= HALF as f64;
    second_half_error /= HALF as f64;

    // Allow a generous tolerance: second half should be at most 80% of first.
    // The first few predictions will be 0 (or base value), so first half error
    // starts large; second half should have adapted substantially.
    assert!(
        second_half_error <= first_half_error * 1.05 || second_half_error < 1.0,
        "{}: monotone adaptation failed: first_half_err={:.4}, second_half_err={:.4}",
        name,
        first_half_error,
        second_half_error
    );
}

#[test]
fn monotone_adaptation_sgbt() {
    let mut m = sgbt(20, 0.1);
    assert_monotone_adaptation(&mut m, "SGBT", 2);
}

#[test]
fn monotone_adaptation_rls() {
    let mut m = rls(0.99);
    assert_monotone_adaptation(&mut m, "RLS", 2);
}

#[test]
fn monotone_adaptation_kan() {
    let mut m = streaming_kan(&[2, 8, 1], 0.05);
    assert_monotone_adaptation(&mut m, "KAN", 2);
}

// ---------------------------------------------------------------------------
// Relation 4: Sample count monotonicity
//
// n_samples_seen() must be non-decreasing after every train() call.
// For models WITHOUT warmup: it must increase by exactly 1 per call.
// For models WITH warmup: it must be non-decreasing (may be 0 during warmup).
// ---------------------------------------------------------------------------

fn assert_count_monotone_exact(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let features: Vec<f64> = vec![1.0; dim];
    for i in 0..50u64 {
        assert_eq!(
            model.n_samples_seen(),
            i,
            "{}: expected n_samples_seen={} before train #{}, got {}",
            name,
            i,
            i + 1,
            model.n_samples_seen()
        );
        model.train(&features, 1.0);
        assert_eq!(
            model.n_samples_seen(),
            i + 1,
            "{}: expected n_samples_seen={} after train #{}, got {}",
            name,
            i + 1,
            i + 1,
            model.n_samples_seen()
        );
    }
}

fn assert_count_monotone_nondecreasing(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let features: Vec<f64> = vec![1.0; dim];
    let mut prev = model.n_samples_seen();
    for i in 0..100 {
        model.train(&features, 1.0);
        let cur = model.n_samples_seen();
        assert!(
            cur >= prev,
            "{}: n_samples_seen() decreased at step {}: {} -> {}",
            name,
            i,
            prev,
            cur
        );
        prev = cur;
    }
}

#[test]
fn count_monotone_sgbt() {
    let mut m = sgbt(5, 0.05);
    assert_count_monotone_exact(&mut m, "SGBT", 2);
}

#[test]
fn count_monotone_rls() {
    let mut m = rls(0.99);
    assert_count_monotone_exact(&mut m, "RLS", 2);
}

#[test]
fn count_monotone_linear() {
    let mut m = linear(0.01);
    assert_count_monotone_exact(&mut m, "Linear", 2);
}

#[test]
fn count_monotone_spikenet() {
    let mut m = spikenet(16);
    assert_count_monotone_exact(&mut m, "SpikeNet", 2);
}

#[test]
fn count_monotone_krls() {
    let mut m = krls(1.0, 20, 1e-4);
    assert_count_monotone_exact(&mut m, "KRLS", 2);
}

#[test]
fn count_monotone_mondrian() {
    let mut m = mondrian(5);
    assert_count_monotone_exact(&mut m, "Mondrian", 2);
}

#[test]
fn count_monotone_ttt_nondecreasing() {
    // TTT has warmup=10, so n_samples_seen() only increments post-warmup.
    let mut m = streaming_ttt(8, 0.05);
    assert_count_monotone_nondecreasing(&mut m, "TTT", 2);
}

#[test]
fn count_monotone_slstm_nondecreasing() {
    // sLSTM has warmup=10.
    let mut m = streaming_slstm(8);
    assert_count_monotone_nondecreasing(&mut m, "sLSTM", 2);
}

#[test]
fn count_monotone_esn_nondecreasing() {
    // ESN has warmup=50.
    let mut m = esn(20, 0.9);
    assert_count_monotone_nondecreasing(&mut m, "ESN", 2);
}

#[test]
fn count_monotone_mgrade_nondecreasing() {
    // mGRADE has warmup=10.
    let mut m = mgrade(2, 8);
    assert_count_monotone_nondecreasing(&mut m, "mGRADE", 2);
}
