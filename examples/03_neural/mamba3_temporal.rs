//! Mamba-3 MIMO SSM for temporal regression.
//!
//! Demonstrates the Mamba-3 architecture with MIMO groups and
//! complex states for capturing oscillatory patterns.

use irithyll::learner::StreamingLearner;
use irithyll::ssm::{MambaConfig, MambaVersion, StreamingMamba};

fn main() {
    let config = MambaConfig::builder()
        .d_in(8)
        .n_state(16)
        .version(MambaVersion::V3)
        .n_groups(2)
        .warmup(20)
        .build()
        .unwrap();

    let mut model = StreamingMamba::new(config);

    for i in 0..500 {
        let t = i as f64 * 0.01;
        let mut features = [0.0; 8];
        for (j, f) in features.iter_mut().enumerate() {
            *f = ((j as f64 + 1.0) * t).sin();
        }
        let target = (t).sin() + 0.5 * (3.0 * t).cos();

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
