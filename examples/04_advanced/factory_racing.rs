//! Automated model selection: races SGBT vs ESN vs KAN on streaming data.
//!
//! The AutoTuner runs successive halving tournaments across multiple factories,
//! using a bandit to allocate candidates. The champion model always provides
//! predictions while challengers compete in the background.
//!
//! Run: cargo run --example factory_racing

use irithyll::{automl::Factory, StreamingLearner};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    eprintln!("=== Irithyll: AutoML Factory Racing ===");
    eprintln!("Racing SGBT vs ESN vs KAN on Friedman-like stream\n");

    // 1. Build an AutoTuner with three competing factories
    let n_features = 5;
    let mut tuner = irithyll::automl::AutoTuner::builder()
        .factory(Factory::sgbt(n_features))
        .add_factory(Factory::esn())
        .add_factory(Factory::kan(n_features))
        .n_initial(6)
        .round_budget(100)
        .build()
        .expect("valid AutoTuner config");

    eprintln!("Factories: {:?}", tuner.factory_names());
    eprintln!("Racing with 6 candidates per tournament, 100-sample rounds\n");

    // 2. Stream 3000 samples from a Friedman-like function:
    //    y = 10*sin(pi*x0*x1) + 20*(x2 - 0.5)^2 + 10*x3 + 5*x4 + noise
    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_samples = 3000;
    let mut sum_sq_err = 0.0;
    let mut count = 0u64;

    eprintln!("--- Training ({} samples) ---", n_samples);
    for i in 0..n_samples {
        let x: Vec<f64> = (0..n_features).map(|_| xorshift64(&mut rng)).collect();
        let noise = (xorshift64(&mut rng) - 0.5) * 0.2;
        let y = 10.0 * (std::f64::consts::PI * x[0] * x[1]).sin()
            + 20.0 * (x[2] - 0.5).powi(2)
            + 10.0 * x[3]
            + 5.0 * x[4]
            + noise;

        // Prequential: predict before training
        let pred = tuner.predict(&x);
        let err = y - pred;
        sum_sq_err += err * err;
        count += 1;

        tuner.train(&x, y);

        if (i + 1) % 500 == 0 {
            let rmse = (sum_sq_err / count as f64).sqrt();
            eprintln!(
                "  Samples: {:>5} | RMSE: {:.4} | Promotions: {} | Tournaments: {}",
                i + 1,
                rmse,
                tuner.promotions(),
                tuner.tournaments_completed(),
            );
        }
    }

    // 3. Final results
    let rmse = (sum_sq_err / count as f64).sqrt();
    eprintln!("\n--- Results ---");
    eprintln!("  Final RMSE:    {:.4}", rmse);
    eprintln!("  Promotions:    {}", tuner.promotions());
    eprintln!("  Tournaments:   {}", tuner.tournaments_completed());
    eprintln!("  Samples seen:  {}", tuner.n_samples_seen());

    eprintln!("\n[DONE] Factory racing example complete.");
}
