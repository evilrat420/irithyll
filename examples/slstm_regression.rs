//! sLSTM streaming regression example.
//!
//! Demonstrates the stabilized LSTM with exponential gating for
//! online time series regression.

use irithyll::learner::StreamingLearner;
use irithyll::lstm::{SLSTMConfig, StreamingsLSTM};

fn main() {
    let config = SLSTMConfig::builder()
        .d_model(4)
        .forgetting_factor(0.998)
        .warmup(20)
        .build()
        .unwrap();

    let mut model = StreamingsLSTM::new(config);

    // Generate a simple streaming signal
    for i in 0..500 {
        let t = i as f64 * 0.02;
        let features = [t.sin(), t.cos(), (2.0 * t).sin(), 1.0];
        let target = 0.5 * t.sin() + 0.3 * t.cos();

        let pred = model.predict(&features);
        model.train(&features, target);

        if i % 100 == 0 {
            let error = (pred - target).abs();
            println!(
                "Step {:>4}: pred={:.4}, target={:.4}, error={:.4}",
                i, pred, target, error
            );
        }
    }

    println!("Samples seen: {}", model.n_samples_seen());
}
