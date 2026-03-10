//! Binary classification example.
//!
//! Demonstrates streaming binary classification using logistic loss.
//! Two clusters in 2D space: class 0 centered at (-2, -2),
//! class 1 centered at (2, 2), with Gaussian-like noise.

use irithyll::{SGBTConfig, SGBT, Sample, ClassificationMetrics};
use irithyll::loss::logistic::LogisticLoss;

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Approximate standard normal sample using Box-Muller (one output).
/// Uses two uniform draws from xorshift64.
fn randn(state: &mut u64) -> f64 {
    let u1 = xorshift64(state).max(1e-15); // avoid log(0)
    let u2 = xorshift64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn main() {
    println!("=== Irithyll: Binary Classification ===");
    println!("Class 0: center (-2, -2)  |  Class 1: center (2, 2)\n");

    // 1. Configure with logistic loss
    let config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    let mut model = SGBT::with_loss(config, Box::new(LogisticLoss));

    println!("Config: n_steps=30, lr=0.1, logistic loss");

    // 2. Stream 1000 samples, track accuracy
    let mut rng: u64 = 0xBEEF_FACE_1234_5678;
    let mut metrics = ClassificationMetrics::new();
    let n_samples = 1000;

    println!("\n--- Training ---");
    for i in 0..n_samples {
        // Alternate classes: even=0, odd=1
        let label = if xorshift64(&mut rng) < 0.5 { 0.0 } else { 1.0 };
        let (cx, cy) = if label == 0.0 { (-2.0, -2.0) } else { (2.0, 2.0) };

        // Add Gaussian-like noise (stddev ~ 1.0)
        let x1 = cx + randn(&mut rng);
        let x2 = cy + randn(&mut rng);

        // Predict BEFORE training (prequential)
        let prob = model.predict_proba(&[x1, x2]);
        let predicted_class = if prob >= 0.5 { 1 } else { 0 };
        let true_class = label as usize;

        metrics.update(true_class, predicted_class, prob);

        // Train
        model.train_one(&Sample::new(vec![x1, x2], label));

        // Report every 200 samples
        if (i + 1) % 200 == 0 {
            println!(
                "  Samples: {:>5} | Accuracy: {:.2}% | Precision: {:.4} | Recall: {:.4} | F1: {:.4} | LogLoss: {:.4}",
                i + 1,
                metrics.accuracy() * 100.0,
                metrics.precision(),
                metrics.recall(),
                metrics.f1(),
                metrics.log_loss(),
            );
        }
    }

    // 3. Final summary
    println!("\n--- Final Metrics ---");
    println!("  Total samples: {}", metrics.n_samples());
    println!("  Accuracy:      {:.2}%", metrics.accuracy() * 100.0);
    println!("  Precision:     {:.4}", metrics.precision());
    println!("  Recall:        {:.4}", metrics.recall());
    println!("  F1 Score:      {:.4}", metrics.f1());
    println!("  Log Loss:      {:.4}", metrics.log_loss());

    // 4. Test predictions on clear-cut points
    println!("\n--- Test Predictions ---");
    println!("  {:>6} {:>6} | {:>8} {:>10}",
        "x1", "x2", "prob", "class");
    println!("  {}", "-".repeat(38));

    let test_points: [(f64, f64, &str); 6] = [
        (-3.0, -3.0, "expect 0"),
        (-1.0, -1.0, "expect 0"),
        ( 0.0,  0.0, "boundary"),
        ( 1.0,  1.0, "expect 1"),
        ( 3.0,  3.0, "expect 1"),
        ( 5.0,  5.0, "expect 1"),
    ];

    for (x1, x2, note) in &test_points {
        let prob = model.predict_proba(&[*x1, *x2]);
        let class = if prob >= 0.5 { 1 } else { 0 };
        println!(
            "  {:>6.1} {:>6.1} | {:>8.4} {:>5}     ({})",
            x1, x2, prob, class, note
        );
    }

    println!("\n[DONE] Classification example complete.");
}
