//! Generic compliance harness for the [`StreamingLearner`] trait.
//!
//! Every model family has one `#[test]` function that calls `check_compliance`,
//! which drives the model through a fixed 6-phase protocol and asserts
//! invariants that every conforming implementation must satisfy.
//!
//! Models with warmup periods (ESN, TTT, sLSTM, mGRADE) are tested with
//! enough samples to pass their warmup, and use a relaxed sample-count check
//! that only asserts at least some samples were recorded.

use irithyll::mgrade;
use irithyll::{
    esn, gaussian_nb, gla, krls, linear, mamba, mondrian, rls, sgbt, spikenet, streaming_kan,
    streaming_slstm, streaming_ttt, StreamingLearner,
};

// ---------------------------------------------------------------------------
// Inline XORShift64 RNG -- no external deps required
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Core compliance protocol
// ---------------------------------------------------------------------------

/// Run the 6-phase compliance protocol on `model`.
///
/// * `name`   -- human-readable model name for assert messages
/// * `dim`    -- number of features (must match model's expectation)
/// * `n_train`-- samples to train in Phase 2 (must exceed warmup for all models)
/// * `has_warmup` -- if true, skip the n_samples_seen == n_train assertion
///   (models with warmup period report only post-warmup sample counts)
fn check_compliance(
    model: &mut dyn StreamingLearner,
    name: &str,
    dim: usize,
    n_train: usize,
    has_warmup: bool,
) {
    // ------------------------------------------------------------------
    // Phase 1: Fresh model starts at 0
    // ------------------------------------------------------------------
    assert_eq!(
        model.n_samples_seen(),
        0,
        "{}: expected 0 samples on fresh model, got {}",
        name,
        model.n_samples_seen()
    );

    // ------------------------------------------------------------------
    // Phase 2: Train n_train samples, verify finite predictions throughout
    // ------------------------------------------------------------------
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_1234;
    for i in 0..n_train {
        let features: Vec<f64> = (0..dim)
            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
            .collect();
        let target = xorshift64(&mut rng) * 20.0 - 10.0;
        model.train(&features, target);

        // Only check prediction finiteness after model has had time to warm up.
        // For warmup-based models, the first few predictions are 0.0 (valid).
        let pred = model.predict(&features);
        assert!(
            pred.is_finite(),
            "{}: non-finite prediction after {} samples, got {}",
            name,
            i + 1,
            pred
        );
    }

    // Sample count: exact for non-warmup models; at least 1 for warmup models.
    if has_warmup {
        assert!(
            model.n_samples_seen() > 0,
            "{}: expected > 0 post-warmup samples after {} trains, got {}",
            name,
            n_train,
            model.n_samples_seen()
        );
    } else {
        assert_eq!(
            model.n_samples_seen(),
            n_train as u64,
            "{}: expected {} samples, got {}",
            name,
            n_train,
            model.n_samples_seen()
        );
    }

    // ------------------------------------------------------------------
    // Phase 3: Reset brings count back to 0
    // ------------------------------------------------------------------
    model.reset();
    assert_eq!(
        model.n_samples_seen(),
        0,
        "{}: expected 0 after reset(), got {}",
        name,
        model.n_samples_seen()
    );

    // ------------------------------------------------------------------
    // Phase 4: Reset is idempotent
    // ------------------------------------------------------------------
    model.reset();
    assert_eq!(
        model.n_samples_seen(),
        0,
        "{}: expected 0 after double reset(), got {}",
        name,
        model.n_samples_seen()
    );

    // ------------------------------------------------------------------
    // Phase 5: Train one sample after reset -- prediction must be finite
    // ------------------------------------------------------------------
    let warmup_features: Vec<f64> = vec![1.0; dim];

    // For warmup models we need to burn through warmup first.
    if has_warmup {
        // Use a generous warmup budget (100 samples covers all models).
        for _ in 0..100 {
            model.train(&warmup_features, 5.0);
        }
    } else {
        model.train(&warmup_features, 5.0);
    }

    let pred = model.predict(&warmup_features);
    assert!(
        pred.is_finite(),
        "{}: non-finite prediction after reset+retrain, got {}",
        name,
        pred
    );

    // ------------------------------------------------------------------
    // Phase 6: predict is deterministic (two calls same result)
    // ------------------------------------------------------------------
    let p1 = model.predict(&warmup_features);
    let p2 = model.predict(&warmup_features);
    assert_eq!(
        p1, p2,
        "{}: predict should be deterministic, got {} then {}",
        name, p1, p2
    );
}

// ---------------------------------------------------------------------------
// One test per model family
// ---------------------------------------------------------------------------

#[test]
fn compliance_sgbt() {
    let mut model = sgbt(20, 0.05);
    check_compliance(&mut model, "SGBT", 2, 100, false);
}

#[test]
fn compliance_rls() {
    let mut model = rls(0.99);
    check_compliance(&mut model, "RLS", 4, 100, false);
}

#[test]
fn compliance_krls() {
    // KRLS with budget=50 so it stays tractable.
    let mut model = krls(1.0, 50, 1e-4);
    check_compliance(&mut model, "KRLS", 3, 100, false);
}

#[test]
fn compliance_linear() {
    let mut model = linear(0.01);
    check_compliance(&mut model, "Linear", 4, 100, false);
}

#[test]
fn compliance_esn() {
    // ESN default warmup = 50. Use 150 train samples to ensure some
    // post-warmup samples are counted.
    let mut model = esn(30, 0.9);
    check_compliance(&mut model, "ESN", 2, 150, true);
}

#[test]
fn compliance_mamba() {
    let mut model = mamba(3, 16);
    check_compliance(&mut model, "Mamba", 3, 100, false);
}

#[test]
fn compliance_kan() {
    let mut model = streaming_kan(&[4, 8, 1], 0.01);
    check_compliance(&mut model, "KAN", 4, 100, false);
}

#[test]
fn compliance_ttt() {
    // TTT default warmup = 10. Use 100 train samples; has_warmup = true.
    let mut model = streaming_ttt(16, 0.05);
    check_compliance(&mut model, "TTT", 2, 100, true);
}

#[test]
fn compliance_spikenet() {
    // SpikeNet counts every sample, no warmup gating.
    let mut model = spikenet(32);
    check_compliance(&mut model, "SpikeNet", 2, 100, false);
}

#[test]
fn compliance_mgrade() {
    // mGRADE warmup = 10.
    let mut model = mgrade(3, 16);
    check_compliance(&mut model, "mGRADE", 3, 100, true);
}

#[test]
fn compliance_slstm() {
    // sLSTM warmup = 10.
    let mut model = streaming_slstm(16);
    check_compliance(&mut model, "sLSTM", 2, 100, true);
}

#[test]
fn compliance_attention_gla() {
    // GLA counts every sample (no separate warmup gating).
    let mut model = gla(8, 2);
    check_compliance(&mut model, "GLA", 8, 100, false);
}

#[test]
fn compliance_mondrian() {
    let mut model = mondrian(5);
    check_compliance(&mut model, "Mondrian", 4, 100, false);
}

#[test]
fn compliance_gaussian_nb() {
    // GaussianNB targets are used as class labels.
    // Use 0.0 / 1.0 targets to keep it sensible.
    let mut model = gaussian_nb();

    // Phase 1: fresh model at 0.
    assert_eq!(
        model.n_samples_seen(),
        0,
        "GaussianNB: expected 0 samples on fresh model"
    );

    // Phase 2: train with binary targets.
    let mut rng: u64 = 0xBEEF_CAFE_DEAD_1234;
    for i in 0..100 {
        let features: Vec<f64> = (0..4).map(|_| xorshift64(&mut rng) * 10.0 - 5.0).collect();
        let target = if xorshift64(&mut rng) > 0.5 { 1.0 } else { 0.0 };
        model.train(&features, target);
        let pred = model.predict(&features);
        assert!(
            pred.is_finite(),
            "GaussianNB: non-finite prediction after {} samples, got {}",
            i + 1,
            pred
        );
    }
    assert_eq!(
        model.n_samples_seen(),
        100,
        "GaussianNB: expected 100 samples"
    );

    // Phase 3: reset.
    model.reset();
    assert_eq!(
        model.n_samples_seen(),
        0,
        "GaussianNB: expected 0 after reset"
    );

    // Phase 4: double reset.
    model.reset();
    assert_eq!(
        model.n_samples_seen(),
        0,
        "GaussianNB: expected 0 after double reset"
    );

    // Phase 5 & 6: train one, check finite and deterministic.
    let features = vec![0.5, 1.0, -0.5, 2.0];
    model.train(&features, 1.0);
    let pred = model.predict(&features);
    assert!(
        pred.is_finite(),
        "GaussianNB: non-finite prediction after reset+retrain, got {}",
        pred
    );
    let p1 = model.predict(&features);
    let p2 = model.predict(&features);
    assert_eq!(
        p1, p2,
        "GaussianNB: predict should be deterministic, got {} then {}",
        p1, p2
    );
}
