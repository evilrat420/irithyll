//! Integration tests for SGBT ensemble.

use irithyll::ensemble::multiclass::MulticlassSGBT;
use irithyll::ensemble::variants::SGBTVariant;
use irithyll::loss::logistic::LogisticLoss;
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
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> SGBTConfig {
    SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .build()
        .unwrap()
}

/// Compute RMSE from a slice of squared errors.
fn rmse(errors: &[f64]) -> f64 {
    (errors.iter().sum::<f64>() / errors.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// 1. Regression convergence
// ---------------------------------------------------------------------------

#[test]
fn sgbt_regression_converges() {
    let config = default_config();
    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xCAFE_BABE_1234_5678;

    let mut early_errors = Vec::new();
    let mut late_errors = Vec::new();

    for i in 0..1000 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.5;
        let target = 2.0 * x1 + 3.0 * x2 + noise;

        // Predict before training
        let pred = model.predict(&[x1, x2]);
        let err_sq = (pred - target).powi(2);

        if (50..150).contains(&i) {
            early_errors.push(err_sq);
        }
        if i >= 900 {
            late_errors.push(err_sq);
        }

        model.train_one(&Sample::new(vec![x1, x2], target));
    }

    let early_rmse = rmse(&early_errors);
    let late_rmse = rmse(&late_errors);

    assert!(
        late_rmse < early_rmse,
        "RMSE should decrease over time: early={:.4}, late={:.4}",
        early_rmse,
        late_rmse
    );
}

// ---------------------------------------------------------------------------
// 2. Binary classification accuracy
// ---------------------------------------------------------------------------

#[test]
fn sgbt_classification_accuracy() {
    // Use a single feature for a clean 1D separation problem.
    // Class 0: feature in [-5, -1], Class 1: feature in [1, 5].
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .build()
        .unwrap();

    let mut model = SGBT::with_loss(config, LogisticLoss);
    let mut rng: u64 = 12345;

    for _ in 0..2000 {
        // Class 0: x in [-5, -1]
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x = -5.0 + (rng as f64 / u64::MAX as f64) * 4.0; // [-5, -1]
        model.train_one(&Sample::new(vec![x], 0.0));

        // Class 1: x in [1, 5]
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let x = 1.0 + (rng as f64 / u64::MAX as f64) * 4.0; // [1, 5]
        model.train_one(&Sample::new(vec![x], 1.0));
    }

    // The raw prediction for class 1 region should be higher than for class 0 region.
    let raw_class1 = model.predict(&[3.0]);
    let raw_class0 = model.predict(&[-3.0]);

    // At minimum, the model should assign a higher raw score to the class 1 region.
    assert!(
        raw_class1 > raw_class0,
        "Raw prediction for class 1 region ({:.4}) should exceed class 0 region ({:.4})",
        raw_class1,
        raw_class0
    );

    // The transformed predictions (sigmoid) should reflect this ordering.
    let prob_class1 = model.predict_proba(&[3.0]);
    let prob_class0 = model.predict_proba(&[-3.0]);
    assert!(
        prob_class1 > prob_class0,
        "P(class1|x=3) = {:.4} should > P(class1|x=-3) = {:.4}",
        prob_class1,
        prob_class0
    );
}

// ---------------------------------------------------------------------------
// 3. Batch == Sequential
// ---------------------------------------------------------------------------

#[test]
fn sgbt_batch_equals_sequential() {
    let config = default_config();
    let mut model_seq = SGBT::new(config.clone());
    let mut model_batch = SGBT::new(config);

    let samples: Vec<Sample> = (0..100)
        .map(|i| {
            let x = i as f64 * 0.1;
            Sample::new(vec![x, x * 2.0, x * 0.5], x * 3.0 + 1.0)
        })
        .collect();

    for s in &samples {
        model_seq.train_one(s);
    }
    model_batch.train_batch(&samples);

    // Check predictions at several test points
    let test_points: Vec<Vec<f64>> = vec![
        vec![0.5, 1.0, 0.25],
        vec![3.0, 6.0, 1.5],
        vec![7.0, 14.0, 3.5],
    ];

    for features in &test_points {
        let pred_seq = model_seq.predict(features);
        let pred_batch = model_batch.predict(features);
        assert!(
            (pred_seq - pred_batch).abs() < 1e-10,
            "Predictions should be identical: seq={}, batch={} for features {:?}",
            pred_seq,
            pred_batch,
            features
        );
    }

    assert_eq!(model_seq.n_samples_seen(), model_batch.n_samples_seen());
}

// ---------------------------------------------------------------------------
// 4. Reset restores initial state
// ---------------------------------------------------------------------------

#[test]
fn sgbt_reset_restores_initial_state() {
    let mut model = SGBT::new(default_config());

    for i in 0..200 {
        let x = i as f64 * 0.1;
        model.train_one(&Sample::new(vec![x, x * 2.0], x * 3.0));
    }

    assert!(model.n_samples_seen() == 200);
    assert!(model.is_initialized());

    model.reset();

    assert_eq!(
        model.n_samples_seen(),
        0,
        "samples_seen should be 0 after reset"
    );
    assert!(
        !model.is_initialized(),
        "model should not be initialized after reset"
    );

    let pred = model.predict(&[1.0, 2.0]);
    assert!(
        pred.abs() < 1e-12,
        "prediction should be ~0 after reset, got {}",
        pred
    );
}

// ---------------------------------------------------------------------------
// 5. Skip variant still converges
// ---------------------------------------------------------------------------

#[test]
fn sgbt_with_skip_variant() {
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .variant(SGBTVariant::Skip { k: 5 })
        .build()
        .unwrap();

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xABCD_EF01_2345_6789;

    let mut early_errors = Vec::new();
    let mut late_errors = Vec::new();

    for i in 0..500 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let target = 2.0 * x1 + 3.0 * x2;

        let pred = model.predict(&[x1, x2]);
        let err_sq = (pred - target).powi(2);

        if (50..150).contains(&i) {
            early_errors.push(err_sq);
        }
        if i >= 400 {
            late_errors.push(err_sq);
        }

        model.train_one(&Sample::new(vec![x1, x2], target));
    }

    let early_rmse = rmse(&early_errors);
    let late_rmse = rmse(&late_errors);

    assert!(
        late_rmse < early_rmse,
        "Skip variant should still converge: early={:.4}, late={:.4}",
        early_rmse,
        late_rmse
    );
}

// ---------------------------------------------------------------------------
// 6. Multiclass basic
// ---------------------------------------------------------------------------

#[test]
fn sgbt_multiclass_basic() {
    let config = SGBTConfig::builder()
        .n_steps(10)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(3)
        .n_bins(8)
        .build()
        .unwrap();

    let mut model = MulticlassSGBT::new(config, 3).unwrap();
    let mut rng: u64 = 0x1234_5678_ABCD_EF00;

    for i in 0..300 {
        let class = (i % 3) as f64;
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        model.train_one(&Sample::new(vec![x1, x2], class));
    }

    // Verify probabilities sum to ~1.0
    let proba = model.predict_proba(&[1.0, 2.0]);
    assert_eq!(proba.len(), 3, "should return 3 probabilities");

    let sum: f64 = proba.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probabilities should sum to ~1.0, got {:.8}",
        sum
    );

    // Each probability should be non-negative
    for (i, &p) in proba.iter().enumerate() {
        assert!(
            p >= 0.0,
            "probability for class {} should be non-negative, got {}",
            i,
            p
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Prediction is finite after training
// ---------------------------------------------------------------------------

#[test]
fn sgbt_predictions_always_finite() {
    let mut model = SGBT::new(default_config());
    let mut rng: u64 = 0xFEED_FACE_DEAD_BEEF;

    for _ in 0..500 {
        let x1 = xorshift64(&mut rng) * 100.0 - 50.0;
        let x2 = xorshift64(&mut rng) * 100.0 - 50.0;
        let target = x1.sin() + x2.cos();
        model.train_one(&Sample::new(vec![x1, x2], target));

        let pred = model.predict(&[x1, x2]);
        assert!(pred.is_finite(), "prediction must be finite, got {}", pred);
    }
}

// ---------------------------------------------------------------------------
// 8. n_steps and total_leaves are consistent
// ---------------------------------------------------------------------------

#[test]
fn sgbt_metadata_consistency() {
    let config = SGBTConfig::builder()
        .n_steps(15)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(3)
        .n_bins(8)
        .build()
        .unwrap();

    let mut model = SGBT::new(config);

    assert_eq!(model.n_steps(), 15);
    // Before training, each step has 1 leaf (root)
    assert_eq!(model.total_leaves(), 15);

    // Train enough samples to trigger some splits
    let mut rng: u64 = 0x9999_8888_7777_6666;
    for _ in 0..300 {
        let x = xorshift64(&mut rng) * 10.0;
        model.train_one(&Sample::new(vec![x], x * 2.0));
    }

    // After training, total leaves should be >= n_steps (trees may have split)
    assert!(
        model.total_leaves() >= model.n_steps(),
        "total_leaves ({}) should be >= n_steps ({})",
        model.total_leaves(),
        model.n_steps()
    );
}
