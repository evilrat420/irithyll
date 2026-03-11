//! Model checkpointing and restore example.
//!
//! Demonstrates saving a trained SGBT model to JSON, restoring it, and
//! verifying that predictions match. This is essential for production
//! deployments where model state must survive process restarts.

use irithyll::serde_support::{load_model, save_model};
use irithyll::{RegressionMetrics, SGBTConfig, Sample, SGBT};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: Model Checkpointing ===\n");

    // 1. Train a model
    let config = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_train = 1000;

    println!(
        "Training on {} samples (y = 2*x1 + 3*x2 + noise)...",
        n_train
    );
    for _ in 0..n_train {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = xorshift64(&mut rng) - 0.5;
        let target = 2.0 * x1 + 3.0 * x2 + noise;
        model.train_one(&Sample::new(vec![x1, x2], target));
    }

    println!("  Samples seen: {}", model.n_samples_seen());
    println!("  Total leaves: {}", model.total_leaves());

    // 2. Collect predictions before saving
    let test_points: Vec<[f64; 2]> = vec![
        [1.0, 1.0],
        [-2.0, 3.0],
        [0.0, 0.0],
        [4.0, -1.0],
        [-3.0, -2.0],
    ];
    let original_preds: Vec<f64> = test_points.iter().map(|p| model.predict(p)).collect();

    // 3. Save model to JSON
    println!("\nSaving model to JSON...");
    let json = save_model(&model).expect("serialization failed");
    let json_bytes = json.len();
    println!(
        "  JSON size: {} bytes ({:.1} KB)",
        json_bytes,
        json_bytes as f64 / 1024.0
    );

    // 4. Restore model from JSON
    println!("\nRestoring model from JSON...");
    let restored = load_model(&json).expect("deserialization failed");

    println!("  Samples seen: {}", restored.n_samples_seen());
    println!("  Total leaves: {}", restored.total_leaves());

    // 5. Verify predictions match
    println!("\n--- Prediction Comparison ---");
    println!(
        "  {:>6} {:>6} | {:>12} {:>12} {:>10}",
        "x1", "x2", "original", "restored", "diff"
    );
    println!("  {}", "-".repeat(56));

    let mut max_diff: f64 = 0.0;
    for (i, point) in test_points.iter().enumerate() {
        let restored_pred = restored.predict(point);
        let diff = (original_preds[i] - restored_pred).abs();
        max_diff = max_diff.max(diff);
        println!(
            "  {:>6.1} {:>6.1} | {:>12.6} {:>12.6} {:>10.2e}",
            point[0], point[1], original_preds[i], restored_pred, diff
        );
    }

    if max_diff < 1e-10 {
        println!("\n  [OK] Predictions match (max diff: {:.2e})", max_diff);
    } else {
        println!(
            "\n  [WARN] Prediction mismatch detected (max diff: {:.2e})",
            max_diff
        );
    }

    // 6. Continue training the restored model
    println!("\nContinuing training on restored model (500 more samples)...");
    let mut metrics = RegressionMetrics::new();

    for _ in 0..500 {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = xorshift64(&mut rng) - 0.5;
        let target = 2.0 * x1 + 3.0 * x2 + noise;

        let pred = restored.predict(&[x1, x2]);
        metrics.update(target, pred);
    }

    println!("  Post-restore RMSE: {:.4}", metrics.rmse());
    println!("  Total samples seen: {}", restored.n_samples_seen());

    // 7. Production pattern: write to file
    println!("\n--- Production Checkpoint Pattern ---");
    println!("  // Save checkpoint to disk:");
    println!("  //   let json = save_model(&model)?;");
    println!("  //   std::fs::write(\"model.json\", &json)?;");
    println!("  //");
    println!("  // Restore on startup:");
    println!("  //   let json = std::fs::read_to_string(\"model.json\")?;");
    println!("  //   let model = load_model(&json)?;");

    println!("\n[DONE] Model checkpointing example complete.");
}
