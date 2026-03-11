//! Basic streaming regression example.
//!
//! Demonstrates training an SGBT model on `y = 2*x1 + 3*x2 + noise`,
//! tracking RMSE over time, and making predictions.

use irithyll::{RegressionMetrics, SGBTConfig, Sample, SGBT};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: Basic Streaming Regression ===");
    println!("Target function: y = 2*x1 + 3*x2 + noise\n");

    // 1. Configure the model
    let config = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    println!("Config: n_steps=50, lr=0.1, grace_period=20, max_depth=4, n_bins=32");

    // 2. Create the model
    let mut model = SGBT::new(config);

    // 3. Stream 2000 samples, printing RMSE every 500
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let mut metrics = RegressionMetrics::new();
    let n_samples = 2000;

    println!("\n--- Training ---");
    for i in 0..n_samples {
        // Generate features in [-5, 5]
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        // Noise in [-0.5, 0.5]
        let noise = xorshift64(&mut rng) - 0.5;
        let target = 2.0 * x1 + 3.0 * x2 + noise;

        // Predict BEFORE training (prequential evaluation)
        let prediction = model.predict(&[x1, x2]);
        metrics.update(target, prediction);

        // Train on this sample
        model.train_one(&Sample::new(vec![x1, x2], target));

        // Report every 500 samples
        if (i + 1) % 500 == 0 {
            println!(
                "  Samples: {:>5} | RMSE: {:.4} | MAE: {:.4} | R2: {:.4}",
                i + 1,
                metrics.rmse(),
                metrics.mae(),
                metrics.r_squared(),
            );
        }
    }

    // 4. Final model statistics
    println!("\n--- Model Stats ---");
    println!("  Samples seen:    {}", model.n_samples_seen());
    println!("  Total leaves:    {}", model.total_leaves());
    println!("  Base prediction: {:.4}", model.base_prediction());
    println!("  Initialized:     {}", model.is_initialized());

    // 5. Make predictions and compare to true values
    println!("\n--- Test Predictions ---");
    println!(
        "  {:>6} {:>6} | {:>10} {:>10} {:>10}",
        "x1", "x2", "true_y", "predicted", "error"
    );
    println!("  {}", "-".repeat(52));

    let test_points: [(f64, f64); 5] = [
        (1.0, 1.0),
        (-2.0, 3.0),
        (0.0, 0.0),
        (4.0, -1.0),
        (-3.0, -2.0),
    ];

    for (x1, x2) in &test_points {
        let true_y = 2.0 * x1 + 3.0 * x2;
        let pred = model.predict(&[*x1, *x2]);
        let error = (pred - true_y).abs();
        println!(
            "  {:>6.1} {:>6.1} | {:>10.4} {:>10.4} {:>10.4}",
            x1, x2, true_y, pred, error
        );
    }

    println!("\n[DONE] Basic regression example complete.");
}
