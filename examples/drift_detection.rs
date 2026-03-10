//! Concept drift detection and adaptation example.
//!
//! Demonstrates how SGBT handles abrupt concept drift:
//! - Phase 1 (samples 0-999):    y = 2*x + 1
//! - Phase 2 (samples 1000-1999): y = -3*x + 5
//!
//! The model's RMSE spikes at the drift point, then recovers as the
//! ensemble replaces drifted trees via the Page-Hinkley detector.

use irithyll::{SGBTConfig, SGBT, Sample, RegressionMetrics};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: Concept Drift Detection ===");
    println!("Phase 1 (0-999):    y = 2*x + 1");
    println!("Phase 2 (1000-1999): y = -3*x + 5");
    println!("Drift detector: Page-Hinkley (default)\n");

    // Configure with default drift detector (PageHinkley)
    let config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xDEAD_BEEF_1337_7331;

    let n_samples = 2000;
    let drift_point = 1000;
    let window_size = 200;

    // Track RMSE in windows of 200 samples
    let mut window_metrics = RegressionMetrics::new();
    let mut window_results: Vec<(usize, f64)> = Vec::new();

    println!("--- Training with windowed RMSE (window={}) ---", window_size);

    for i in 0..n_samples {
        // Generate feature in [-5, 5]
        let x = xorshift64(&mut rng) * 10.0 - 5.0;
        // Small noise
        let noise = (xorshift64(&mut rng) - 0.5) * 0.2;

        // Target depends on phase
        let target = if i < drift_point {
            2.0 * x + 1.0 + noise
        } else {
            -3.0 * x + 5.0 + noise
        };

        // Prequential: predict before training
        let prediction = model.predict(&[x]);
        window_metrics.update(target, prediction);

        // Train
        model.train_one(&Sample::new(vec![x], target));

        // Report at window boundaries
        if (i + 1) % window_size == 0 {
            let rmse = window_metrics.rmse();
            let window_end = i + 1;
            let phase = if window_end <= drift_point { "Phase 1" } else { "Phase 2" };
            let marker = if window_end > drift_point
                && window_end <= (drift_point + window_size)
            {
                " <-- DRIFT"
            } else {
                ""
            };

            println!(
                "  Samples {:>5}-{:>5} | RMSE: {:>8.4} | {} {}",
                window_end - window_size,
                window_end,
                rmse,
                phase,
                marker,
            );

            window_results.push((window_end, rmse));
            window_metrics.reset();
        }
    }

    // Analysis: compare pre-drift, drift-window, and post-recovery RMSE
    println!("\n--- Drift Analysis ---");

    // Pre-drift: average of windows entirely in Phase 1
    let pre_drift_rmses: Vec<f64> = window_results
        .iter()
        .filter(|(end, _)| *end <= drift_point)
        .map(|(_, rmse)| *rmse)
        .collect();
    let pre_drift_avg = pre_drift_rmses.iter().sum::<f64>()
        / pre_drift_rmses.len().max(1) as f64;

    // Drift window: first window after drift point
    let drift_window_rmse = window_results
        .iter()
        .find(|(end, _)| *end > drift_point && *end <= drift_point + window_size)
        .map(|(_, rmse)| *rmse)
        .unwrap_or(0.0);

    // Post-recovery: last 2 windows
    let post_recovery_rmses: Vec<f64> = window_results
        .iter()
        .rev()
        .take(2)
        .map(|(_, rmse)| *rmse)
        .collect();
    let post_recovery_avg = post_recovery_rmses.iter().sum::<f64>()
        / post_recovery_rmses.len().max(1) as f64;

    println!("  Pre-drift avg RMSE:    {:.4}", pre_drift_avg);
    println!("  Drift-window RMSE:     {:.4}", drift_window_rmse);
    println!("  Post-recovery avg RMSE: {:.4}", post_recovery_avg);

    if drift_window_rmse > pre_drift_avg {
        println!("\n  [OK] RMSE spiked at drift point (expected behavior)");
    }
    if post_recovery_avg < drift_window_rmse {
        println!("  [OK] RMSE recovered after drift (adaptation working)");
    }

    // Final model stats
    println!("\n--- Model Stats ---");
    println!("  Samples seen: {}", model.n_samples_seen());
    println!("  Total leaves: {}", model.total_leaves());

    // Verify the model has adapted to the new concept
    println!("\n--- Post-Drift Predictions (should follow y = -3*x + 5) ---");
    let test_xs = [-3.0, -1.0, 0.0, 1.0, 3.0];
    println!("  {:>6} | {:>10} {:>10} {:>10}", "x", "true_y", "predicted", "error");
    println!("  {}", "-".repeat(45));

    for x in &test_xs {
        let true_y = -3.0 * x + 5.0;
        let pred = model.predict(&[*x]);
        let error = (pred - true_y).abs();
        println!(
            "  {:>6.1} | {:>10.4} {:>10.4} {:>10.4}",
            x, true_y, pred, error
        );
    }

    println!("\n[DONE] Drift detection example complete.");
}
