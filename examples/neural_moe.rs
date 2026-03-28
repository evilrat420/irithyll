//! Neural Mixture of Experts: heterogeneous experts competing via learned routing.
use irithyll::{linear, moe::NeuralMoE, rls, sgbt, StreamingLearner};

fn main() {
    let mut moe = NeuralMoE::builder()
        .expert(sgbt(50, 0.01))
        .expert(sgbt(100, 0.005))
        .expert(linear(0.01))
        .expert(rls(0.99))
        .top_k(2)
        .build();

    println!(
        "Training NeuralMoE with {} experts, top_k={}...",
        moe.n_experts(),
        moe.top_k()
    );

    for i in 0..500 {
        let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
        let y = x[0] * 3.0 + x[1] * 2.0 + 1.0;
        moe.train(&x, y);
    }

    let pred = moe.predict(&[2.5, (250.0_f64 * 0.1).sin()]);
    println!("Prediction: {:.4}", pred);
    println!("Expert utilization: {:?}", moe.utilization());
    println!("Samples seen: {}", moe.n_samples_seen());
}
