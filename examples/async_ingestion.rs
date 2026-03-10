//! Async streaming ingestion example.
//!
//! Demonstrates the tokio-native async training API:
//! - Create an AsyncSGBT runner
//! - Obtain sender and predictor handles
//! - Spawn the training loop on a tokio task
//! - Feed samples via the sender from the main task
//! - Drop the sender to signal completion
//! - Await training and show final predictions

use irithyll::{SGBTConfig, Sample};
use irithyll::stream::AsyncSGBT;

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

#[tokio::main]
async fn main() {
    println!("=== Irithyll: Async Streaming Ingestion ===");
    println!("Target function: y = 4*x1 - 2*x2 + 1\n");

    // 1. Create configuration
    let config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    // 2. Create the async runner
    let mut runner = AsyncSGBT::new(config);

    // 3. Get sender and predictor handles BEFORE spawning
    let sender = runner.sender();
    let predictor = runner.predictor();

    println!("  Model initialized: {}", predictor.is_initialized());
    println!("  Samples seen:      {}", predictor.n_samples_seen());

    // 4. Spawn the training loop on a tokio task
    let train_handle = tokio::spawn(async move {
        runner.run().await
    });

    println!("\n--- Sending 500 samples ---");

    // 5. Send 500 samples via the sender
    let mut rng: u64 = 0xABCD_EF01_2345_6789;
    let n_samples = 500;

    for i in 0..n_samples {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.2;
        let target = 4.0 * x1 - 2.0 * x2 + 1.0 + noise;

        sender
            .send(Sample::new(vec![x1, x2], target))
            .await
            .expect("channel should be open");

        // Report progress periodically
        if (i + 1) % 100 == 0 {
            println!(
                "  Sent {:>4} samples | Model has trained on: {}",
                i + 1,
                predictor.n_samples_seen(),
            );
        }
    }

    println!("\n--- All samples sent, dropping sender ---");

    // 6. Drop sender to signal completion
    drop(sender);

    // 7. Await the training task
    let result = train_handle.await.expect("task should not panic");
    result.expect("training should succeed");

    println!("  Training loop completed.");
    println!("  Final samples seen: {}", predictor.n_samples_seen());
    println!("  Model initialized:  {}", predictor.is_initialized());

    // 8. Show final predictions from predictor
    println!("\n--- Final Predictions (y = 4*x1 - 2*x2 + 1) ---");
    println!("  {:>6} {:>6} | {:>10} {:>10} {:>10}",
        "x1", "x2", "true_y", "predicted", "error");
    println!("  {}", "-".repeat(52));

    let test_points: [(f64, f64); 5] = [
        (1.0, 1.0),
        (-2.0, 3.0),
        (0.0, 0.0),
        (3.0, -1.0),
        (-1.0, -2.0),
    ];

    for (x1, x2) in &test_points {
        let true_y = 4.0 * x1 - 2.0 * x2 + 1.0;
        let pred = predictor.predict(&[*x1, *x2]);
        let error = (pred - true_y).abs();
        println!(
            "  {:>6.1} {:>6.1} | {:>10.4} {:>10.4} {:>10.4}",
            x1, x2, true_y, pred, error
        );
    }

    println!("\n[DONE] Async ingestion example complete.");
}
