//! Numerical stability tests with deterministic adversarial input patterns.
//!
//! Each pattern is run on 8 model families. Tests are grouped by *pattern*
//! (not by model) to avoid bloat and make failures easy to triage.
//!
//! Patterns:
//! 1. Extreme magnitudes       -- features [1e15, 1e-15]
//! 2. Repeated identical       -- same (features, target) 1000 times
//! 3. Alternating extremes     -- targets alternate +1e6 and -1e6
//! 4. All-zero features        -- [0.0, 0.0]
//! 5. Near-zero features       -- [1e-300, 1e-300]
//!
//! All assertions are `.is_finite()`. NaN or Inf is a failure.

use irithyll::mgrade;
use irithyll::{
    esn, mamba, rls, sgbt, spikenet, streaming_kan, streaming_slstm, streaming_ttt,
    StreamingLearner,
};

// ---------------------------------------------------------------------------
// Helper: run a closure that drives a boxed model, then assert finite predict.
// ---------------------------------------------------------------------------

fn run_pattern(
    model: &mut dyn StreamingLearner,
    name: &str,
    pattern: &str,
    f: impl Fn(&mut dyn StreamingLearner),
    check_features: &[f64],
) {
    f(model);
    let pred = model.predict(check_features);
    assert!(
        pred.is_finite(),
        "{} / {}: prediction not finite after pattern, got {}",
        name,
        pattern,
        pred
    );
}

// ---------------------------------------------------------------------------
// Helper: build all 8 models as boxed trait objects
// ---------------------------------------------------------------------------

fn all_models() -> Vec<(String, Box<dyn StreamingLearner>)> {
    vec![
        ("SGBT".into(), Box::new(sgbt(10, 0.05))),
        ("RLS".into(), Box::new(rls(0.99))),
        ("Mamba".into(), Box::new(mamba(2, 8))),
        ("sLSTM".into(), Box::new(streaming_slstm(8))),
        ("KAN".into(), Box::new(streaming_kan(&[2, 4, 1], 0.01))),
        ("TTT".into(), Box::new(streaming_ttt(8, 0.05))),
        ("ESN".into(), Box::new(esn(20, 0.9))),
        ("mGRADE".into(), Box::new(mgrade(2, 8))),
        ("SpikeNet".into(), Box::new(spikenet(16))),
    ]
}

// ---------------------------------------------------------------------------
// Pattern 1: Extreme magnitudes
// Train 50 samples with features [1e15, 1e-15], target 1.0.
// Prediction must be finite.
// ---------------------------------------------------------------------------

fn run_extreme_magnitudes(model: &mut dyn StreamingLearner, name: &str) {
    let features = vec![1e15_f64, 1e-15_f64];
    run_pattern(
        model,
        name,
        "extreme_magnitudes",
        |m| {
            for _ in 0..50 {
                m.train(&[1e15, 1e-15], 1.0);
            }
        },
        &features,
    );
}

#[test]
fn stability_extreme_magnitudes() {
    for (name, mut model) in all_models() {
        run_extreme_magnitudes(model.as_mut(), &name);
    }
}

// ---------------------------------------------------------------------------
// Pattern 2: Repeated identical samples
// Same (features, target) repeated 1000 times. Prediction must be finite.
// ---------------------------------------------------------------------------

fn run_repeated_identical(model: &mut dyn StreamingLearner, name: &str) {
    let features = vec![0.5_f64, -0.3_f64];
    let target = 2.0_f64;
    run_pattern(
        model,
        name,
        "repeated_identical",
        |m| {
            for _ in 0..1000 {
                m.train(&[0.5, -0.3], target);
            }
        },
        &features,
    );
}

#[test]
fn stability_repeated_identical() {
    for (name, mut model) in all_models() {
        run_repeated_identical(model.as_mut(), &name);
    }
}

// ---------------------------------------------------------------------------
// Pattern 3: Alternating extreme targets
// Features fixed [0.1, 0.2]. Targets alternate +1e6 and -1e6.
// 50 samples. Prediction must be finite.
// ---------------------------------------------------------------------------

fn run_alternating_extremes(model: &mut dyn StreamingLearner, name: &str) {
    let features = vec![0.1_f64, 0.2_f64];
    run_pattern(
        model,
        name,
        "alternating_extremes",
        |m| {
            for i in 0..50 {
                let target = if i % 2 == 0 { 1e6 } else { -1e6 };
                m.train(&[0.1, 0.2], target);
            }
        },
        &features,
    );
}

#[test]
fn stability_alternating_extremes() {
    for (name, mut model) in all_models() {
        run_alternating_extremes(model.as_mut(), &name);
    }
}

// ---------------------------------------------------------------------------
// Pattern 4: All-zero features
// [0.0, 0.0] for 50 samples, target 3.0. Prediction must be finite.
// ---------------------------------------------------------------------------

fn run_all_zero_features(model: &mut dyn StreamingLearner, name: &str) {
    let features = vec![0.0_f64, 0.0_f64];
    run_pattern(
        model,
        name,
        "all_zero_features",
        |m| {
            for _ in 0..50 {
                m.train(&[0.0, 0.0], 3.0);
            }
        },
        &features,
    );
}

#[test]
fn stability_all_zero_features() {
    for (name, mut model) in all_models() {
        run_all_zero_features(model.as_mut(), &name);
    }
}

// ---------------------------------------------------------------------------
// Pattern 5: Near-zero features (subnormal territory)
// [1e-300, 1e-300] for 50 samples, target 0.5. Prediction must be finite.
// ---------------------------------------------------------------------------

fn run_near_zero_features(model: &mut dyn StreamingLearner, name: &str) {
    let features = vec![1e-300_f64, 1e-300_f64];
    run_pattern(
        model,
        name,
        "near_zero_features",
        |m| {
            for _ in 0..50 {
                m.train(&[1e-300, 1e-300], 0.5);
            }
        },
        &features,
    );
}

#[test]
fn stability_near_zero_features() {
    for (name, mut model) in all_models() {
        run_near_zero_features(model.as_mut(), &name);
    }
}

// ---------------------------------------------------------------------------
// Extra: Verify prediction after reset + re-exposure to extreme patterns
// ---------------------------------------------------------------------------

#[test]
fn stability_reset_then_extreme() {
    let mut model = rls(0.99);
    // Train on extremes.
    for _ in 0..50 {
        model.train(&[1e15, 1e-15], 1.0);
    }
    // Reset.
    model.reset();
    assert_eq!(
        model.n_samples_seen(),
        0,
        "RLS: expected 0 samples after reset"
    );
    // Re-train on extremes.
    for _ in 0..50 {
        model.train(&[1e15, 1e-15], 1.0);
    }
    let pred = model.predict(&[1e15, 1e-15]);
    assert!(
        pred.is_finite(),
        "RLS: prediction not finite after reset+extreme re-train, got {}",
        pred
    );
}

#[test]
fn stability_sgbt_mixed_extreme_and_normal() {
    // Train 50 normal samples, then 50 extreme target samples, then 50 normal.
    let mut model = sgbt(10, 0.05);
    let mut rng: u64 = 0xABCD_EF01_2345_6789;
    #[inline]
    fn xs(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64) / (u64::MAX as f64) * 4.0 - 2.0
    }
    // Normal.
    for _ in 0..50 {
        model.train(&[xs(&mut rng), xs(&mut rng)], xs(&mut rng));
    }
    // Extreme targets.
    for i in 0..50 {
        let t = if i % 2 == 0 { 1e6 } else { -1e6 };
        model.train(&[xs(&mut rng), xs(&mut rng)], t);
    }
    // Normal again.
    for _ in 0..50 {
        model.train(&[xs(&mut rng), xs(&mut rng)], xs(&mut rng));
    }
    let pred = model.predict(&[0.0, 0.0]);
    assert!(
        pred.is_finite(),
        "SGBT mixed extreme/normal: prediction not finite, got {}",
        pred
    );
}
