//! TTT adapts to changing data distributions via fast weights.
//!
//! Generates a stream with a regime shift at sample 1000: the target function
//! changes abruptly. StreamingTTT's prediction-directed fast weight updates
//! allow it to adapt quickly at the boundary.
//!
//! Run: cargo run --example streaming_ttt

use irithyll::{streaming_ttt, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    eprintln!("=== Irithyll: StreamingTTT (Test-Time Training) ===");
    eprintln!("Regime shift at sample 1000: fast weights adapt to new distribution.\n");

    // d_model=16, eta=0.05 (inner learning rate for fast weight updates)
    let mut model = streaming_ttt(16, 0.05);

    let mut rng: u64 = 0xBAAD_F00D_CAFE_1234;
    let n_samples = 2000;
    let regime_shift = 1000;

    // Track RMSE in windows of 200 samples
    let window = 200;
    let mut window_sse = 0.0;
    let mut window_count = 0u64;

    eprintln!(
        "--- Training ({} samples, regime shift at {}) ---",
        n_samples, regime_shift
    );
    for i in 0..n_samples {
        let x1 = (xorshift64(&mut rng) - 0.5) * 2.0;
        let x2 = (xorshift64(&mut rng) - 0.5) * 2.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.1;

        // Regime 1: y = 2*x1 + x2
        // Regime 2: y = -x1 + 3*x2  (abrupt shift)
        let y = if i < regime_shift {
            2.0 * x1 + x2 + noise
        } else {
            -x1 + 3.0 * x2 + noise
        };

        // Prequential evaluation
        let pred = model.predict(&[x1, x2]);
        let err = y - pred;
        window_sse += err * err;
        window_count += 1;

        model.train(&[x1, x2], y);

        if (i + 1) % window == 0 {
            let rmse = (window_sse / window_count as f64).sqrt();
            let regime = if i < regime_shift { "A" } else { "B" };
            eprintln!(
                "  Samples: {:>5} [Regime {}] | Window RMSE: {:.4}",
                i + 1,
                regime,
                rmse,
            );
            // Reset window
            window_sse = 0.0;
            window_count = 0;
        }
    }

    eprintln!("\n--- Model Info ---");
    eprintln!("  Samples seen: {}", model.n_samples_seen());

    eprintln!("\n[DONE] StreamingTTT regime adaptation example complete.");
}
