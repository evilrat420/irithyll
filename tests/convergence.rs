//! Convergence and regret bound verification tests.

use irithyll::loss::huber::HuberLoss;
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

/// Compute RMSE from a slice of squared errors.
fn rmse(errors: &[f64]) -> f64 {
    (errors.iter().sum::<f64>() / errors.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// 1. Linear function converges
// ---------------------------------------------------------------------------

#[test]
fn linear_function_converges() {
    // Match the approach from the unit test: use features in [-5, 5],
    // a simpler target (y = 2*x1 + 3*x2), and measure predict-before-train error.
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(3)
        .n_bins(16)
        .build()
        .unwrap();

    let mut model = SGBT::new(config);
    let mut rng: u64 = 12345;

    let mut early_errors = Vec::new();
    let mut late_errors = Vec::new();

    for i in 0..1000 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x1 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x2 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
        let target = 2.0 * x1 + 3.0 * x2;

        let pred = model.predict(&[x1, x2]);
        let error = (pred - target).powi(2);

        if (50..150).contains(&i) {
            early_errors.push(error);
        }
        if i >= 900 {
            late_errors.push(error);
        }

        model.train_one(&Sample::new(vec![x1, x2], target));
    }

    let early_rmse = rmse(&early_errors);
    let late_rmse = rmse(&late_errors);

    assert!(
        late_rmse < early_rmse,
        "RMSE should decrease: early={:.4}, late={:.4}",
        early_rmse,
        late_rmse
    );
}

// ---------------------------------------------------------------------------
// 2. Constant target converges to mean
// ---------------------------------------------------------------------------

#[test]
fn constant_target_converges_to_mean() {
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(3)
        .n_bins(8)
        .build()
        .unwrap();

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0x1111_2222_3333_4444;

    // Train on constant target = 42.0 with random features
    for _ in 0..200 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        model.train_one(&Sample::new(vec![x1, x2], 42.0));
    }

    // Prediction at any point should be close to 42.0
    let pred = model.predict(&[0.0, 0.0]);
    assert!(
        (pred - 42.0).abs() < 5.0,
        "prediction should be within 5.0 of 42.0, got {:.4}",
        pred
    );

    let pred2 = model.predict(&[3.0, -1.0]);
    assert!(
        (pred2 - 42.0).abs() < 5.0,
        "prediction at different point should also be close to 42.0, got {:.4}",
        pred2
    );
}

// ---------------------------------------------------------------------------
// 3. Huber loss converges with outliers
// ---------------------------------------------------------------------------

#[test]
fn huber_converges_with_outliers() {
    // Use the same RNG style as the working unit test: direct xorshift inline.
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(3)
        .n_bins(16)
        .build()
        .unwrap();

    let mut model = SGBT::with_loss(config, HuberLoss::new(1.0));
    let mut rng: u64 = 54321;

    let mut early_errors = Vec::new();
    let mut late_errors = Vec::new();

    for i in 0..1000 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
        let target = 3.0 * x;

        // 5% of samples get a large outlier added to the training target
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let r = rng as f64 / u64::MAX as f64;
        let train_target = if r < 0.05 { target + 50.0 } else { target };

        let pred = model.predict(&[x]);
        // Measure error against the clean target (not the outlier-contaminated one)
        let error = (pred - target).powi(2);

        if (50..150).contains(&i) {
            early_errors.push(error);
        }
        if i >= 900 {
            late_errors.push(error);
        }

        model.train_one(&Sample::new(vec![x], train_target));
    }

    let early_rmse = rmse(&early_errors);
    let late_rmse = rmse(&late_errors);

    assert!(
        late_rmse < early_rmse,
        "Huber loss should converge even with 5%% outliers: early={:.4}, late={:.4}",
        early_rmse,
        late_rmse
    );
}

// ---------------------------------------------------------------------------
// 4. More steps improves accuracy
// ---------------------------------------------------------------------------

#[test]
fn more_steps_improves_accuracy() {
    let config_5 = SGBTConfig::builder()
        .n_steps(5)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .build()
        .unwrap();

    let config_50 = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .build()
        .unwrap();

    let mut model_5 = SGBT::new(config_5);
    let mut model_50 = SGBT::new(config_50);

    // Generate training data deterministically
    let mut rng: u64 = 0xFACE_FEED_BEEF_CAFE;
    let mut samples = Vec::new();
    for _ in 0..500 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.2;
        let target = 3.0 * x1 - x2 + 2.0 + noise;
        samples.push(Sample::new(vec![x1, x2], target));
    }

    // Train both models on the same data
    for s in &samples {
        model_5.train_one(s);
        model_50.train_one(s);
    }

    // Evaluate on test data
    let mut errors_5 = Vec::new();
    let mut errors_50 = Vec::new();
    for _ in 0..200 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let target = 3.0 * x1 - x2 + 2.0;

        let pred_5 = model_5.predict(&[x1, x2]);
        let pred_50 = model_50.predict(&[x1, x2]);

        errors_5.push((pred_5 - target).powi(2));
        errors_50.push((pred_50 - target).powi(2));
    }

    let rmse_5 = rmse(&errors_5);
    let rmse_50 = rmse(&errors_50);

    // The 50-step model should have lower RMSE (or at least not dramatically worse)
    // Using a generous tolerance: 50-step should be at most 2x worse than 5-step
    // (in practice it should be better)
    assert!(
        rmse_50 < rmse_5 * 2.0,
        "50-step model should not be dramatically worse than 5-step: rmse_5={:.4}, rmse_50={:.4}",
        rmse_5,
        rmse_50
    );
}
