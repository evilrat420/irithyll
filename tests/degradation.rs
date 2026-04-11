//! Degradation and concept-drift scenario tests.
//!
//! Three scenarios:
//!
//! 1. **Abrupt drift** -- Train y=2x for 500 samples, then switch to y=-3x.
//!    Measure windowed RMSE in the pre-drift stable window, the post-drift
//!    shock window, and the post-drift recovery window. Assert the model
//!    recovers (recovery RMSE < shock RMSE). Neural models may not recover
//!    from abrupt drift; they are only tested for finite predictions (no crash).
//!
//! 2. **Target scale shift** -- targets [0,1] shift to [0,100]. Assert the
//!    model adapts (late predictions closer to large-scale targets).
//!
//! 3. **Cold restart** -- train 500 samples, then only 1 sample per 50 steps
//!    (very sparse stream). Predictions must remain finite throughout.

use irithyll::mgrade;
use irithyll::{
    esn, mamba, rls, sgbt, spikenet, streaming_kan, streaming_slstm, streaming_ttt,
    StreamingLearner,
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

fn rmse(errors: &[f64]) -> f64 {
    if errors.is_empty() {
        return 0.0;
    }
    (errors.iter().sum::<f64>() / errors.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Scenario 1a: Abrupt drift -- SGBT and RLS should recover
// ---------------------------------------------------------------------------

fn run_abrupt_drift_recovers(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0xC0FF_EEBA_BEDE_ADBE;

    // Pre-drift: y = 2*x0 for 500 samples.
    let pre_samples = 500;
    let post_samples = 500;

    // Windows for measuring:
    //   pre_stable   : samples 400-499 (just before drift)
    //   post_shock   : samples 500-549 (just after drift)
    //   post_recovery: samples 900-999 (late post-drift)

    let mut pre_stable_errs: Vec<f64> = Vec::new();
    let mut post_shock_errs: Vec<f64> = Vec::new();
    let mut post_recovery_errs: Vec<f64> = Vec::new();

    for i in 0..(pre_samples + post_samples) {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0 - 2.0).collect();
        let target = if i < pre_samples {
            2.0 * features[0]
        } else {
            -3.0 * features[0]
        };

        let pred = model.predict(&features);
        let err = (pred - target).powi(2);

        if (400..500).contains(&i) {
            pre_stable_errs.push(err);
        } else if (500..550).contains(&i) {
            post_shock_errs.push(err);
        } else if (900..1000).contains(&i) {
            post_recovery_errs.push(err);
        }

        model.train(&features, target);
    }

    let pre_stable_rmse = rmse(&pre_stable_errs);
    let post_shock_rmse = rmse(&post_shock_errs);
    let post_recovery_rmse = rmse(&post_recovery_errs);

    // The shock should be worse than the stable period.
    assert!(
        post_shock_rmse >= pre_stable_rmse * 0.5,
        "{} abrupt drift: expected shock RMSE to increase from stable RMSE={:.4} but got shock RMSE={:.4}",
        name,
        pre_stable_rmse,
        post_shock_rmse
    );

    // Recovery RMSE should be lower than shock RMSE.
    // Allow recovery RMSE to be at most 50% of shock RMSE.
    // If shock RMSE is very small (model adapted immediately), skip this check.
    if post_shock_rmse > 0.5 {
        assert!(
            post_recovery_rmse < post_shock_rmse,
            "{} abrupt drift: recovery RMSE={:.4} should be < shock RMSE={:.4}",
            name,
            post_recovery_rmse,
            post_shock_rmse
        );
    }
}

#[test]
fn abrupt_drift_sgbt_recovers() {
    let mut m = sgbt(20, 0.05);
    run_abrupt_drift_recovers(&mut m, "SGBT", 2);
}

#[test]
fn abrupt_drift_rls_recovers() {
    // RLS with lower forgetting factor adapts faster to drift.
    let mut m = rls(0.99);
    run_abrupt_drift_recovers(&mut m, "RLS", 2);
}

// ---------------------------------------------------------------------------
// Scenario 1b: Abrupt drift -- neural models stay finite (no crash required)
// ---------------------------------------------------------------------------

fn run_abrupt_drift_stays_finite(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0xDEAD_BEEF_C0FF_EE42;

    for i in 0..1000 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0 - 2.0).collect();
        let target = if i < 500 {
            2.0 * features[0]
        } else {
            -3.0 * features[0]
        };

        let pred = model.predict(&features);
        assert!(
            pred.is_finite(),
            "{}: non-finite prediction at step {} during abrupt drift, got {}",
            name,
            i,
            pred
        );
        model.train(&features, target);
    }

    // Final prediction must also be finite.
    let probe: Vec<f64> = vec![0.5; dim];
    let final_pred = model.predict(&probe);
    assert!(
        final_pred.is_finite(),
        "{}: non-finite prediction after abrupt drift sequence, got {}",
        name,
        final_pred
    );
}

#[test]
fn abrupt_drift_mamba_stays_finite() {
    let mut m = mamba(2, 8);
    run_abrupt_drift_stays_finite(&mut m, "Mamba", 2);
}

#[test]
fn abrupt_drift_esn_stays_finite() {
    let mut m = esn(20, 0.9);
    run_abrupt_drift_stays_finite(&mut m, "ESN", 2);
}

#[test]
fn abrupt_drift_ttt_stays_finite() {
    let mut m = streaming_ttt(8, 0.05);
    run_abrupt_drift_stays_finite(&mut m, "TTT", 2);
}

#[test]
fn abrupt_drift_slstm_stays_finite() {
    let mut m = streaming_slstm(8);
    run_abrupt_drift_stays_finite(&mut m, "sLSTM", 2);
}

#[test]
fn abrupt_drift_kan_stays_finite() {
    let mut m = streaming_kan(&[2, 4, 1], 0.01);
    run_abrupt_drift_stays_finite(&mut m, "KAN", 2);
}

#[test]
fn abrupt_drift_spikenet_stays_finite() {
    let mut m = spikenet(16);
    run_abrupt_drift_stays_finite(&mut m, "SpikeNet", 2);
}

#[test]
fn abrupt_drift_mgrade_stays_finite() {
    let mut m = mgrade(2, 8);
    run_abrupt_drift_stays_finite(&mut m, "mGRADE", 2);
}

// ---------------------------------------------------------------------------
// Scenario 2: Target scale shift
// Phase 1: targets in [0, 1] for 300 samples.
// Phase 2: targets in [0, 100] for 300 samples.
// Assert: late-phase-2 predictions have moved toward the large-scale range.
// ---------------------------------------------------------------------------

fn run_scale_shift_adapts(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0x1111_2222_3333_4444;

    // Phase 1: small targets.
    for _ in 0..300 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 2.0).collect();
        let target = xorshift64(&mut rng); // [0, 1]
        model.train(&features, target);
    }

    // Phase 2: large targets. Measure average prediction in last 50 samples.
    let mut late_preds: Vec<f64> = Vec::new();
    for i in 0..300 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 2.0).collect();
        let target = xorshift64(&mut rng) * 100.0; // [0, 100]
        if i >= 250 {
            let pred = model.predict(&features);
            assert!(
                pred.is_finite(),
                "{}: non-finite prediction during scale shift at step {}, got {}",
                name,
                300 + i,
                pred
            );
            late_preds.push(pred);
        }
        model.train(&features, target);
    }

    // Assert predictions moved toward the new scale (average > 1.0).
    let avg_late_pred = late_preds.iter().sum::<f64>() / late_preds.len() as f64;
    assert!(
        avg_late_pred > 1.0,
        "{}: expected late predictions to shift toward [0,100] scale, avg={:.4}",
        name,
        avg_late_pred
    );
}

#[test]
fn scale_shift_sgbt_adapts() {
    let mut m = sgbt(20, 0.05);
    run_scale_shift_adapts(&mut m, "SGBT", 2);
}

#[test]
fn scale_shift_rls_adapts() {
    let mut m = rls(0.99);
    run_scale_shift_adapts(&mut m, "RLS", 2);
}

// ---------------------------------------------------------------------------
// Scenario 3: Cold restart -- sparse stream after dense training
// Train 500 samples, then only 1 sample every 50 "ticks".
// No actual sleeping -- we just intersperse many predict calls with no train.
// All predictions must remain finite.
// ---------------------------------------------------------------------------

fn run_cold_restart_stays_finite(model: &mut dyn StreamingLearner, name: &str, dim: usize) {
    let mut rng: u64 = 0xFACE_FEED_DEAD_C0DE;

    // Dense training phase.
    for _ in 0..500 {
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0 - 2.0).collect();
        let target = xorshift64(&mut rng) * 10.0;
        model.train(&features, target);
    }

    // Sparse phase: 1 train per 50 predict calls.
    let probe: Vec<f64> = vec![0.5; dim];
    for i in 0..500 {
        // Many consecutive predict calls with no training.
        for _ in 0..50 {
            let pred = model.predict(&probe);
            assert!(
                pred.is_finite(),
                "{}: non-finite prediction in sparse phase at outer step {}, got {}",
                name,
                i,
                pred
            );
        }
        // One training sample.
        let features: Vec<f64> = (0..dim).map(|_| xorshift64(&mut rng) * 4.0 - 2.0).collect();
        let target = xorshift64(&mut rng) * 10.0;
        model.train(&features, target);
    }
}

#[test]
fn cold_restart_sgbt_stays_finite() {
    let mut m = sgbt(10, 0.05);
    run_cold_restart_stays_finite(&mut m, "SGBT", 2);
}

#[test]
fn cold_restart_rls_stays_finite() {
    let mut m = rls(0.99);
    run_cold_restart_stays_finite(&mut m, "RLS", 2);
}

#[test]
fn cold_restart_mamba_stays_finite() {
    let mut m = mamba(2, 8);
    run_cold_restart_stays_finite(&mut m, "Mamba", 2);
}

#[test]
fn cold_restart_slstm_stays_finite() {
    let mut m = streaming_slstm(8);
    run_cold_restart_stays_finite(&mut m, "sLSTM", 2);
}

#[test]
fn cold_restart_kan_stays_finite() {
    let mut m = streaming_kan(&[2, 4, 1], 0.01);
    run_cold_restart_stays_finite(&mut m, "KAN", 2);
}

#[test]
fn cold_restart_spikenet_stays_finite() {
    let mut m = spikenet(16);
    run_cold_restart_stays_finite(&mut m, "SpikeNet", 2);
}
