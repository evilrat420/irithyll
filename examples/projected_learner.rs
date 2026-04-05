//! Dimensionality reduction with PAST-based ProjectedLearner.
//!
//! Creates high-dimensional synthetic data where only 5 of 20 features carry
//! signal. Wraps an RLS model in a ProjectedLearner (rank=8) and compares
//! projected vs raw RMSE to show that projection discards noise dimensions.
//!
//! Run: cargo run --example projected_learner

use irithyll::projection::{ProjectedLearner, ProjectionConfig};
use irithyll::{rls, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    eprintln!("=== Irithyll: ProjectedLearner (Online Projection Learning) ===");
    eprintln!("20 features, only 5 relevant. Rank-8 projection vs raw RLS.\n");

    let (d_in, rank, n_samples) = (20, 8, 2000);

    // Projected model: RLS wrapped in PAST-based projection
    let config = ProjectionConfig::builder()
        .rank(rank)
        .warmup(100)
        .supervised_lr(0.01)
        .build()
        .unwrap();
    let mut projected = ProjectedLearner::from_learner(rls(0.99), d_in, config);

    // Raw model: plain RLS on all 20 features
    let mut raw = rls(0.99);
    let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
    let (mut proj_sse, mut raw_sse, mut count) = (0.0, 0.0, 0u64);

    eprintln!("--- Training ({} samples) ---", n_samples);
    for i in 0..n_samples {
        // Generate 20 features: dims 0-4 are signal, dims 5-19 are noise
        let x: Vec<f64> = (0..d_in)
            .map(|_| (xorshift64(&mut rng) - 0.5) * 2.0)
            .collect();

        // Target depends only on first 5 features
        let noise = (xorshift64(&mut rng) - 0.5) * 0.1;
        let y = 3.0 * x[0] + 2.0 * x[1] - 1.5 * x[2] + 0.8 * x[3] + x[4] + noise;

        // Prequential evaluation
        let proj_pred = projected.predict(&x);
        let raw_pred = raw.predict(&x);

        let proj_err = y - proj_pred;
        let raw_err = y - raw_pred;
        proj_sse += proj_err * proj_err;
        raw_sse += raw_err * raw_err;
        count += 1;

        projected.train(&x, y);
        raw.train(&x, y);

        if (i + 1) % 500 == 0 {
            let proj_rmse = (proj_sse / count as f64).sqrt();
            let raw_rmse = (raw_sse / count as f64).sqrt();
            eprintln!(
                "  Samples: {:>5} | Projected RMSE: {:.4} | Raw RMSE: {:.4}",
                i + 1,
                proj_rmse,
                raw_rmse,
            );
        }
    }

    // 3. Final comparison
    let proj_rmse = (proj_sse / count as f64).sqrt();
    let raw_rmse = (raw_sse / count as f64).sqrt();
    eprintln!("\n--- Results ---");
    eprintln!("  Projected RMSE (rank={}): {:.4}", rank, proj_rmse);
    eprintln!("  Raw RLS RMSE (dim={}):    {:.4}", d_in, raw_rmse);
    eprintln!("  Warmup complete: {}", projected.warmup_complete());
    eprintln!("  Samples seen:    {}", projected.n_samples_seen());

    eprintln!("\n[DONE] ProjectedLearner example complete.");
}
