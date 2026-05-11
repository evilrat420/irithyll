//! KAN learns nonlinear functions via B-spline edge activations.
//!
//! Trains a StreamingKAN on a compositional function y = sin(x1*x2) + x3^2
//! and prints convergence as spline coefficients adapt online.
//!
//! Run: cargo run --example streaming_kan

use irithyll::{streaming_kan, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    eprintln!("=== Irithyll: StreamingKAN (B-Spline Online Learning) ===");
    eprintln!("Target: y = sin(x1*x2) + x3^2\n");

    // 3 inputs -> 12 hidden -> 1 output, lr=0.05
    let mut model = streaming_kan(&[3, 12, 1], 0.05);

    let mut rng: u64 = 0x1234_5678_ABCD_EF01;
    let n_samples = 3000;

    let window = 500;
    let mut window_sse = 0.0;
    let mut window_count = 0u64;

    eprintln!(
        "Architecture: [3, 12, 1], lr=0.05, {} params",
        model.n_params()
    );
    eprintln!("--- Training ({} samples) ---", n_samples);

    for i in 0..n_samples {
        let x1 = (xorshift64(&mut rng) - 0.5) * 4.0;
        let x2 = (xorshift64(&mut rng) - 0.5) * 4.0;
        let x3 = (xorshift64(&mut rng) - 0.5) * 4.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.05;
        let y = (x1 * x2).sin() + x3 * x3 + noise;

        // Prequential evaluation
        let pred = model.predict(&[x1, x2, x3]);
        let err = y - pred;
        window_sse += err * err;
        window_count += 1;

        model.train(&[x1, x2, x3], y);

        if (i + 1) % window == 0 {
            let rmse = (window_sse / window_count as f64).sqrt();
            eprintln!("  Samples: {:>5} | Window RMSE: {:.4}", i + 1, rmse,);
            window_sse = 0.0;
            window_count = 0;
        }
    }

    // Test predictions
    eprintln!("\n--- Test Predictions ---");
    let tests: [(f64, f64, f64); 3] = [(0.5, 1.0, 0.3), (-1.0, 0.5, 1.0), (0.0, 0.0, 2.0)];
    for (x1, x2, x3) in &tests {
        let expected = (x1 * x2).sin() + x3 * x3;
        let pred = model.predict(&[*x1, *x2, *x3]);
        eprintln!(
            "  [{:.1}, {:.1}, {:.1}] -> pred: {:.4}, true: {:.4}, err: {:.4}",
            x1,
            x2,
            x3,
            pred,
            expected,
            (pred - expected).abs(),
        );
    }

    eprintln!("\n--- Model Info ---");
    eprintln!("  Parameters:  {}", model.n_params());
    eprintln!("  Layers:      {}", model.n_layers());
    eprintln!("  Samples seen: {}", model.n_samples_seen());

    eprintln!("\n[DONE] StreamingKAN example complete.");
}
