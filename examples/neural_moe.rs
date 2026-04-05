//! Mixture of experts with ESN + Mamba + RLS experts.
//!
//! Polymorphic NeuralMoE where each expert is a different streaming model type.
//! The softmax router learns online which expert performs best for each input,
//! with top-k sparse routing and automatic load balancing.
//!
//! Run: cargo run --example neural_moe

use irithyll::{esn, mamba, moe::NeuralMoE, rls, StreamingLearner};

/// Mackey-Glass step: x' = x + beta*x_tau/(1+x_tau^n) - gamma*x.
fn mg_step(hist: &[f64], tau: usize) -> f64 {
    let x = *hist.last().unwrap();
    let x_d = if hist.len() > tau {
        hist[hist.len() - 1 - tau]
    } else {
        hist[0]
    };
    x + 0.2 * x_d / (1.0 + x_d.powi(10)) - 0.1 * x
}

fn main() {
    eprintln!("=== Irithyll: NeuralMoE (Polymorphic Mixture of Experts) ===");
    eprintln!("Experts: ESN (reservoir) + Mamba (SSM) + RLS (linear)\n");

    let d_in = 4; // Embedding dimension (4 lagged values)

    // Build MoE with heterogeneous experts
    let mut moe = NeuralMoE::builder()
        .expert_with_warmup(esn(50, 0.9), 50) // ESN: 50 neurons, needs warmup
        .expert_with_warmup(mamba(d_in, 16), 20) // Mamba: 16 states, needs warmup
        .expert(rls(0.99)) // RLS: no warmup needed
        .top_k(2)
        .router_lr(0.01)
        .build();

    eprintln!("  Experts: {}, Top-k: {}", moe.n_experts(), moe.top_k());

    // Generate Mackey-Glass time series
    let n_total = 2000;
    let tau = 17;
    let mut history = vec![1.2; tau + 1];

    // Pre-generate enough history
    for _ in 0..200 {
        let next = mg_step(&history, tau);
        history.push(next);
    }

    let window = 500;
    let mut window_sse = 0.0;
    let mut window_count = 0u64;

    eprintln!(
        "\n--- Training on Mackey-Glass (tau={}, {} samples) ---",
        tau, n_total
    );
    for i in 0..n_total {
        // Extend series
        let next = mg_step(&history, tau);
        history.push(next);

        // Create embedding: [x(t-3), x(t-2), x(t-1), x(t)]
        let idx = history.len() - 1;
        let features = [
            history[idx - 3],
            history[idx - 2],
            history[idx - 1],
            history[idx],
        ];
        // Predict next value
        let target = mg_step(&history, tau);

        let pred = moe.predict(&features);
        let err = target - pred;
        window_sse += err * err;
        window_count += 1;

        moe.train(&features, target);

        if (i + 1) % window == 0 {
            let rmse = (window_sse / window_count as f64).sqrt();
            let util = moe.utilization();
            eprintln!(
                "  Samples: {:>5} | RMSE: {:.6} | Util: [{:.2}, {:.2}, {:.2}]",
                i + 1,
                rmse,
                util[0],
                util[1],
                util[2],
            );
            window_sse = 0.0;
            window_count = 0;
        }
    }

    // Final expert selection stats
    let util = moe.utilization();
    eprintln!("\n--- Expert Stats ---");
    eprintln!("  ESN utilization:   {:.4}", util[0]);
    eprintln!("  Mamba utilization: {:.4}", util[1]);
    eprintln!("  RLS utilization:   {:.4}", util[2]);
    eprintln!("  Samples seen:      {}", moe.n_samples_seen());

    eprintln!("\n[DONE] NeuralMoE example complete.");
}
