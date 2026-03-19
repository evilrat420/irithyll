//! Generates pre-exported packed model binaries for embedded benchmarks.
//!
//! Run with: cargo test --test generate_bench_model -- --ignored

use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::{SGBTConfig, Sample, SGBT};

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

#[test]
#[ignore]
fn generate_bench_models() {
    let n_steps = 50;
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.01)
        .grace_period(20)
        .max_depth(max_depth)
        .n_bins(32)
        .build()
        .unwrap();

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xBEEF_CAFE;

    for _ in 0..n_samples {
        let features: Vec<f64> = (0..n_features)
            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
            .collect();
        let target = features
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, &f)| acc + (i as f64 + 1.0) * f);
        model.train_one(&Sample::new(features, target));
    }

    // Export f32 packed
    let packed_f32 = export_packed(&model, n_features);
    let f32_path = "irithyll-core/test_data/bench_model_50t.bin";
    std::fs::write(f32_path, &packed_f32).unwrap();
    eprintln!("[OK] Wrote {} ({} bytes)", f32_path, packed_f32.len());

    // Export i16 quantized
    let packed_i16 = export_packed_i16(&model, n_features);
    let i16_path = "irithyll-core/test_data/bench_model_50t_i16.bin";
    std::fs::write(i16_path, &packed_i16).unwrap();
    eprintln!("[OK] Wrote {} ({} bytes)", i16_path, packed_i16.len());

    // Verify they load
    let view = irithyll_core::EnsembleView::from_bytes(&packed_f32).unwrap();
    eprintln!(
        "f32 model: {} trees, {} features, {} nodes",
        view.n_trees(),
        view.n_features(),
        view.total_nodes()
    );

    let view_i16 = irithyll_core::QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();
    eprintln!(
        "i16 model: {} trees, {} features, {} nodes",
        view_i16.n_trees(),
        view_i16.n_features(),
        view_i16.total_nodes()
    );
}
