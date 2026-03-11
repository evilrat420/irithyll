//! Integration tests for drift detection and tree replacement.

use irithyll::drift::adwin::Adwin;
use irithyll::drift::pht::PageHinkleyTest;
use irithyll::drift::{DriftDetector, DriftSignal};
use irithyll::ensemble::config::DriftDetectorType;
use irithyll::{SGBTConfig, Sample, SGBT};

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// 1. PHT detects abrupt drift
// ---------------------------------------------------------------------------

#[test]
fn pht_detects_abrupt_drift() {
    let mut pht = PageHinkleyTest::new();

    // Phase 1: 500 values around 0.1
    for _ in 0..500 {
        let signal = pht.update(0.1);
        assert_ne!(
            signal,
            DriftSignal::Drift,
            "should not drift during stable phase"
        );
    }

    // Phase 2: abrupt shift to 10.0
    let mut drift_detected = false;
    for i in 0..500 {
        if pht.update(10.0) == DriftSignal::Drift {
            drift_detected = true;
            // Drift should be detected relatively quickly
            assert!(
                i < 200,
                "PHT should detect abrupt drift quickly, took {} samples",
                i
            );
            break;
        }
    }

    assert!(
        drift_detected,
        "PHT must detect abrupt shift from 0.1 to 10.0"
    );
}

// ---------------------------------------------------------------------------
// 2. ADWIN detects abrupt drift
// ---------------------------------------------------------------------------

#[test]
fn adwin_detects_abrupt_drift() {
    let mut adwin = Adwin::with_delta(0.01);

    // Phase 1: 500 values of 0.1
    for _ in 0..500 {
        let signal = adwin.update(0.1);
        assert_ne!(
            signal,
            DriftSignal::Drift,
            "should not drift during stable phase"
        );
    }

    // Phase 2: abrupt shift to 10.0
    let mut drift_detected = false;
    for i in 0..500 {
        if adwin.update(10.0) == DriftSignal::Drift {
            drift_detected = true;
            assert!(
                i < 200,
                "ADWIN should detect abrupt drift quickly, took {} samples",
                i
            );
            break;
        }
    }

    assert!(
        drift_detected,
        "ADWIN must detect abrupt shift from 0.1 to 10.0"
    );
}

// ---------------------------------------------------------------------------
// 3. No false alarm on stationary stream
// ---------------------------------------------------------------------------

#[test]
fn no_false_alarm_stationary() {
    let mut pht = PageHinkleyTest::new();

    // Feed 5000 constant values
    for i in 0..5000 {
        let signal = pht.update(1.0);
        assert_ne!(
            signal,
            DriftSignal::Drift,
            "constant stream should never trigger Drift (sample {})",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// 4. SGBT adapts after drift (concept switch)
// ---------------------------------------------------------------------------

#[test]
fn sgbt_adapts_after_drift() {
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .drift_detector(DriftDetectorType::PageHinkley {
            delta: 0.005,
            lambda: 50.0,
        })
        .build()
        .unwrap();

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xBEEF_CAFE_1234_5678;

    // Phase 1: y = 2*x for 500 samples
    for _ in 0..500 {
        let x = xorshift64(&mut rng) * 10.0 - 5.0;
        let target = 2.0 * x;
        model.train_one(&Sample::new(vec![x], target));
    }

    // Phase 2: concept switch to y = -3*x for 500 samples
    // Collect errors right after the switch (first 50 samples)
    let mut early_switch_errors = Vec::new();
    // Collect errors late in the new concept (last 100 samples)
    let mut late_switch_errors = Vec::new();

    for i in 0..500 {
        let x = xorshift64(&mut rng) * 10.0 - 5.0;
        let target = -3.0 * x;

        let pred = model.predict(&[x]);
        let err = (pred - target).powi(2);

        if i < 50 {
            early_switch_errors.push(err);
        }
        if i >= 400 {
            late_switch_errors.push(err);
        }

        model.train_one(&Sample::new(vec![x], target));
    }

    let early_rmse =
        (early_switch_errors.iter().sum::<f64>() / early_switch_errors.len() as f64).sqrt();
    let late_rmse =
        (late_switch_errors.iter().sum::<f64>() / late_switch_errors.len() as f64).sqrt();

    assert!(
        late_rmse < early_rmse,
        "Model should adapt to new concept: early_rmse={:.4}, late_rmse={:.4}",
        early_rmse,
        late_rmse
    );
}

// ---------------------------------------------------------------------------
// 5. clone_fresh resets to clean state
// ---------------------------------------------------------------------------

#[test]
fn drift_detector_clone_fresh_resets() {
    let mut pht = PageHinkleyTest::with_params(0.01, 100.0);

    // Feed data until we get a warning (or just feed a lot)
    for _ in 0..500 {
        pht.update(5.0);
    }

    // The detector has accumulated state
    assert!(
        pht.estimated_mean() != 0.0,
        "detector should have non-zero estimated mean after feeding data"
    );

    // clone_fresh should return a detector with clean state
    let fresh = pht.clone_fresh();
    assert!(
        (fresh.estimated_mean() - 0.0).abs() < 1e-12,
        "fresh clone should have estimated_mean = 0, got {}",
        fresh.estimated_mean()
    );

    // Verify the fresh clone is functional
    let mut fresh = pht.clone_fresh();
    let signal = fresh.update(1.0);
    assert_eq!(
        signal,
        DriftSignal::Stable,
        "single sample on fresh detector should be Stable"
    );
}

// ---------------------------------------------------------------------------
// 6. ADWIN no false alarm on constant stream
// ---------------------------------------------------------------------------

#[test]
fn adwin_no_false_alarm_constant() {
    let mut adwin = Adwin::with_delta(0.002);

    let mut drift_count = 0;
    for _ in 0..5000 {
        if adwin.update(1.0) == DriftSignal::Drift {
            drift_count += 1;
        }
    }

    assert!(
        drift_count <= 1,
        "constant stream should produce at most 1 false alarm, got {}",
        drift_count
    );
}
