//! RWKV-7 streaming attention example.
//!
//! Demonstrates the RWKV-7 vector-gated delta rule attention
//! for online streaming regression.

use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};
use irithyll::learner::StreamingLearner;

fn main() {
    let config = StreamingAttentionConfig::builder()
        .d_model(8)
        .n_heads(2)
        .mode(AttentionMode::RWKV7)
        .build()
        .unwrap();

    let mut model = StreamingAttentionModel::new(config);

    for i in 0..500 {
        let t = i as f64 * 0.02;
        let mut features = [0.0; 8];
        for (j, f) in features.iter_mut().enumerate() {
            *f = ((j as f64 * 0.5 + 1.0) * t).sin();
        }
        let target = (t).sin() + 0.3 * (2.0 * t).sin();

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
