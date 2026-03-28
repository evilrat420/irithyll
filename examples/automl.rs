//! Streaming AutoML: auto-tune hyperparameters via tournament racing.
use irithyll::{auto_tune, automl::Factory, StreamingLearner};

fn main() {
    let mut tuner = auto_tune(Factory::sgbt(3));

    println!("Auto-tuning SGBT with tournament racing...");

    for i in 0..1000 {
        let x = [
            i as f64 * 0.01,
            (i as f64 * 0.1).sin(),
            (i as f64 * 0.05).cos(),
        ];
        let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
        tuner.train(&x, y);
    }

    let pred = tuner.predict(&[5.0, 500.0_f64.sin(), 250.0_f64.cos()]);
    println!("Prediction: {:.4}", pred);
    println!("Promotions: {}", tuner.promotions());
    println!("Tournaments completed: {}", tuner.tournaments_completed());
    println!("Samples seen: {}", tuner.n_samples_seen());
}
