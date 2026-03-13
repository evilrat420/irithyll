//! Composable preprocessing pipelines.
//!
//! Demonstrates the `pipe()` builder for chaining a normalizer with an SGBT
//! learner. The pipeline handles feature standardization automatically,
//! enabling the model to learn effectively even when input features have
//! wildly different scales (one in [0, 1000], the other in [0, 0.01]).
//!
//! Run: `cargo run --example pipeline_composition`

use irithyll::{normalizer, pipe, sgbt, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: Pipeline Composition ===");
    println!("Normalizer -> SGBT pipeline with multi-scale features\n");

    // 1. Build pipeline: normalizer -> SGBT (50 steps, lr=0.01)
    let mut pipeline = pipe(normalizer()).learner(sgbt(50, 0.1));

    println!("Pipeline: IncrementalNormalizer -> SGBT(n_steps=50, lr=0.1)\n");

    // 2. Feature design: wildly different scales
    //    Feature 0: range [0, 1000]   (large scale)
    //    Feature 1: range [0, 0.01]   (tiny scale)
    //    Target:    y = 0.3*f0 + 5000*f1 + noise
    //    Both features contribute roughly equally (~150 each on average)
    println!("--- Feature Scales ---");
    println!("  Feature 0: range [0, 1000]   (large scale)");
    println!("  Feature 1: range [0, 0.01]   (tiny scale)");
    println!("  Target:    y = 0.3*f0 + 5000*f1 + noise\n");

    // 3. Stream 2000 samples (SGBT needs enough data for tree splits)
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_samples = 2000;

    println!("--- Training ({} samples) ---", n_samples);
    for i in 0..n_samples {
        let f0 = xorshift64(&mut rng) * 1000.0;
        let f1 = xorshift64(&mut rng) * 0.01;
        let noise = (xorshift64(&mut rng) - 0.5) * 5.0;
        let target = 0.3 * f0 + 5000.0 * f1 + noise;

        pipeline.train(&[f0, f1], target);

        if (i + 1) % 500 == 0 {
            // Show a prediction at a midrange point to track learning progress
            let test_pred = pipeline.predict(&[500.0, 0.005]);
            let test_true = 0.3 * 500.0 + 5000.0 * 0.005;
            println!(
                "  Samples: {:>4} | Pred at (500, 0.005): {:>8.2} (true: {:.2})",
                i + 1,
                test_pred,
                test_true,
            );
        }
    }

    // 4. Test predictions at several points
    println!("\n--- Test Predictions ---");
    println!(
        "  {:>8} {:>8} | {:>10} {:>10} {:>10}",
        "f0", "f1", "true_y", "predicted", "error"
    );
    println!("  {}", "-".repeat(56));

    let test_points: [(f64, f64); 5] = [
        (100.0, 0.001),
        (500.0, 0.005),
        (800.0, 0.008),
        (250.0, 0.003),
        (0.0, 0.01),
    ];

    let mut total_error = 0.0;
    for (f0, f1) in &test_points {
        let true_y = 0.3 * f0 + 5000.0 * f1;
        let pred = pipeline.predict(&[*f0, *f1]);
        let error = (pred - true_y).abs();
        total_error += error;
        println!(
            "  {:>8.1} {:>8.4} | {:>10.2} {:>10.2} {:>10.2}",
            f0, f1, true_y, pred, error
        );
    }

    println!("  {}", "-".repeat(56));
    println!(
        "  Mean absolute error: {:.2}",
        total_error / test_points.len() as f64
    );

    // 5. Show that normalization handled the scale difference
    println!("\n--- Scale Handling ---");
    println!("  Features span 5 orders of magnitude (1000 vs 0.01)");
    println!("  Pipeline normalizes to zero-mean, unit-variance before SGBT");
    println!("  Samples seen: {}", pipeline.n_samples_seen());

    println!("\n[DONE] Pipeline composition example complete.");
}
