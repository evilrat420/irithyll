//! Non-linear regression with Kernel Recursive Least Squares (KRLS).
//!
//! Demonstrates learning y = sin(x) over [0, 2*PI] using KRLS with an RBF
//! kernel. Shows dictionary growth during training, ALD sparsification
//! (dictionary much smaller than training set), and prediction accuracy
//! at known trigonometric points (pi/4, pi/2, pi, 3*pi/2).
//!
//! Run: `cargo run --example krls_nonlinear`

use irithyll::{krls, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: KRLS Non-linear Regression ===");
    println!("Target function: y = sin(x), x in [0, 2*PI]\n");

    // 1. Create KRLS with RBF kernel
    //    gamma=1.0, budget=100, ald_threshold=1e-4
    let gamma = 1.0;
    let budget = 100;
    let ald_threshold = 1e-4;
    let mut model = krls(gamma, budget, ald_threshold);

    println!(
        "Config: RBF gamma={}, budget={}, ALD threshold={}\n",
        gamma, budget, ald_threshold
    );

    // 2. Stream 200 training samples of sin(x) over [0, 2*PI] with slight noise
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_samples = 200;
    let two_pi = 2.0 * std::f64::consts::PI;

    println!("--- Training ({} samples over [0, 2*PI]) ---", n_samples);
    for i in 0..n_samples {
        let x = (i as f64) * two_pi / (n_samples as f64);
        let noise = (xorshift64(&mut rng) - 0.5) * 0.05;
        let y = x.sin() + noise;

        model.train(&[x], y);

        // Report dictionary growth at intervals
        if (i + 1) % 50 == 0 {
            println!(
                "  Samples: {:>4} | Dict size: {:>3} / {}",
                i + 1,
                model.dict_size(),
                budget,
            );
        }
    }

    // 3. Dictionary statistics -- ALD sparsification
    println!("\n--- ALD Sparsification ---");
    println!("  Training samples:  {}", n_samples);
    println!("  Dictionary size:   {}", model.dict_size());
    println!("  Budget limit:      {}", budget);
    println!(
        "  Compression:       {:.1}% of samples retained in dictionary",
        (model.dict_size() as f64 / n_samples as f64) * 100.0
    );
    assert!(
        model.dict_size() < n_samples,
        "dictionary should be smaller than training set"
    );
    println!("  [OK] Dictionary is much smaller than training set");

    // 4. Predictions at known trigonometric points
    let pi = std::f64::consts::PI;
    let test_points: [(f64, &str); 4] = [
        (pi / 4.0, "PI/4"),
        (pi / 2.0, "PI/2"),
        (pi, "PI"),
        (3.0 * pi / 2.0, "3*PI/2"),
    ];

    println!("\n--- Predictions at Known Points ---");
    println!(
        "  {:>8} | {:>10} {:>10} {:>10}",
        "x", "sin(x)", "predicted", "error"
    );
    println!("  {}", "-".repeat(46));

    let mut total_error = 0.0;
    for &(x, label) in &test_points {
        let actual = x.sin();
        let pred = model.predict(&[x]);
        let error = (pred - actual).abs();
        total_error += error;
        println!(
            "  {:>8} | {:>10.6} {:>10.6} {:>10.6}",
            label, actual, pred, error
        );
    }

    println!("  {}", "-".repeat(46));
    println!(
        "  Mean absolute error: {:.6}",
        total_error / test_points.len() as f64
    );

    println!("\n[DONE] KRLS non-linear regression example complete.");
}
