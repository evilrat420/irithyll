//! Prediction confidence intervals with Recursive Least Squares.
//!
//! Demonstrates RLS uncertainty quantification on y = 2x + 1 with noise.
//! As more data arrives, the P matrix shrinks and prediction intervals
//! narrow. Reports the 95% confidence interval at x=5.0 every 50 samples,
//! showing how uncertainty decreases with more observations.
//!
//! Run: `cargo run --example rls_confidence`

use irithyll::RecursiveLeastSquares;
use irithyll::{rls, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: RLS Prediction Confidence Intervals ===");
    println!("Target function: y = 2*x + 1 + noise\n");

    // 1. Create RLS with forgetting factor 0.99
    //    Using bias trick: features = [x, 1.0] so weights learn [slope, intercept]
    let mut model: RecursiveLeastSquares = rls(0.99);

    println!("Config: forgetting_factor=0.99, bias trick for intercept");
    println!("Test point: x=5.0, true y=11.0\n");

    // 2. Stream training data, reporting 95% CI every 50 samples
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_samples = 500;
    let test_x = 5.0;
    let test_features = [test_x, 1.0]; // bias trick
    let z = 1.96; // ~95% CI
    let true_y = 2.0 * test_x + 1.0;

    println!(
        "--- 95% Confidence Intervals at x={} (every 50 samples) ---",
        test_x
    );
    println!(
        "  {:>8} | {:>10} {:>10} {:>10} {:>10}",
        "samples", "pred", "CI_lo", "CI_hi", "CI_width"
    );
    println!("  {}", "-".repeat(58));

    let mut prev_width = f64::INFINITY;
    let mut narrowing_count = 0;

    for i in 0..n_samples {
        // Generate x uniformly in [0, 10]
        let x = xorshift64(&mut rng) * 10.0;
        // Noise in [-0.5, 0.5]
        let noise = (xorshift64(&mut rng) - 0.5) * 1.0;
        let y = 2.0 * x + 1.0 + noise;

        model.train(&[x, 1.0], y);

        // Report every 50 samples
        if (i + 1) % 50 == 0 {
            let (pred, lo, hi) = model.predict_interval(&test_features, z);
            let width = hi - lo;

            println!(
                "  {:>8} | {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                i + 1,
                pred,
                lo,
                hi,
                width
            );

            // Track whether intervals are narrowing
            if width < prev_width {
                narrowing_count += 1;
            }
            prev_width = width;
        }
    }

    // 3. Show that intervals narrowed over time
    println!("\n--- Interval Narrowing ---");
    let total_reports = n_samples / 50;
    println!(
        "  Intervals narrowed in {}/{} consecutive reports",
        narrowing_count, total_reports
    );
    if narrowing_count > total_reports / 2 {
        println!("  [OK] Confidence intervals tightened as data accumulated");
    }

    // 4. Show final learned weights
    println!("\n--- Learned Weights ---");
    let weights = model.weights();
    println!("  Slope:     {:.6}  (true: 2.0)", weights[0]);
    println!("  Intercept: {:.6}  (true: 1.0)", weights[1]);
    println!("  Noise var: {:.6}", model.noise_variance());
    println!("  True y:    {:.1} at x={}", true_y, test_x);

    println!("\n[DONE] RLS confidence intervals example complete.");
}
