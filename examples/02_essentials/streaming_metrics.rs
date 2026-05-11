//! Streaming metrics evaluation example.
//!
//! Demonstrates prequential (test-then-train) evaluation with windowed
//! metrics, showing how RMSE and R-squared evolve as the model trains.
//! This is the standard evaluation protocol for streaming ML models.

use irithyll::loss::logistic::LogisticLoss;
use irithyll::{ClassificationMetrics, RegressionMetrics, SGBTConfig, Sample, SGBT};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: Streaming Metrics Evaluation ===\n");

    // -----------------------------------------------------------------------
    // Part 1: Regression with windowed RMSE
    // -----------------------------------------------------------------------
    println!("--- Part 1: Regression (y = 2*x1 + 3*x2 + noise) ---\n");

    let config = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xDEAD_BEEF_1337_7331;
    let n_samples = 3000;
    let window_size = 500;

    // Cumulative and windowed metrics
    let mut cumulative = RegressionMetrics::new();
    let mut windowed = RegressionMetrics::new();

    println!(
        "  {:>8} | {:>10} {:>10} {:>10} | {:>10} {:>10}",
        "samples", "win_RMSE", "win_MAE", "win_R2", "cum_RMSE", "cum_R2"
    );
    println!("  {}", "-".repeat(70));

    for i in 0..n_samples {
        let x1 = xorshift64(&mut rng) * 10.0 - 5.0;
        let x2 = xorshift64(&mut rng) * 10.0 - 5.0;
        let noise = (xorshift64(&mut rng) - 0.5) * 0.5;
        let target = 2.0 * x1 + 3.0 * x2 + noise;

        // Prequential: predict BEFORE training
        let prediction = model.predict(&[x1, x2]);
        cumulative.update(target, prediction);
        windowed.update(target, prediction);

        model.train_one(&Sample::new(vec![x1, x2], target));

        if (i + 1) % window_size == 0 {
            println!(
                "  {:>8} | {:>10.4} {:>10.4} {:>10.4} | {:>10.4} {:>10.4}",
                i + 1,
                windowed.rmse(),
                windowed.mae(),
                windowed.r_squared(),
                cumulative.rmse(),
                cumulative.r_squared(),
            );
            windowed.reset();
        }
    }

    println!(
        "\n  Final cumulative: RMSE={:.4}, MAE={:.4}, MSE={:.4}, R2={:.4}",
        cumulative.rmse(),
        cumulative.mae(),
        cumulative.mse(),
        cumulative.r_squared()
    );

    // -----------------------------------------------------------------------
    // Part 2: Classification with accuracy/precision/recall
    // -----------------------------------------------------------------------
    println!("\n--- Part 2: Binary Classification ---\n");

    let cls_config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    let mut cls_model = SGBT::with_loss(cls_config, LogisticLoss);
    let mut cls_cumulative = ClassificationMetrics::new();
    let mut cls_windowed = ClassificationMetrics::new();
    let n_cls_samples = 2000;
    let cls_window = 500;

    println!(
        "  {:>8} | {:>8} {:>8} {:>8} {:>8} | {:>8}",
        "samples", "win_acc", "win_prec", "win_rec", "win_f1", "cum_acc"
    );
    println!("  {}", "-".repeat(62));

    for i in 0..n_cls_samples {
        // Generate two clusters: class 0 near (-2,-2), class 1 near (2,2)
        let class = if xorshift64(&mut rng) < 0.5 { 0 } else { 1 };
        let center = if class == 0 { -2.0 } else { 2.0 };
        let x1 = center + (xorshift64(&mut rng) - 0.5) * 2.0;
        let x2 = center + (xorshift64(&mut rng) - 0.5) * 2.0;

        // Prequential evaluation
        let prob = cls_model.predict_proba(&[x1, x2]);
        let predicted = if prob >= 0.5 { 1 } else { 0 };

        cls_cumulative.update(class, predicted, prob);
        cls_windowed.update(class, predicted, prob);

        cls_model.train_one(&Sample::new(vec![x1, x2], class as f64));

        if (i + 1) % cls_window == 0 {
            println!(
                "  {:>8} | {:>8.4} {:>8.4} {:>8.4} {:>8.4} | {:>8.4}",
                i + 1,
                cls_windowed.accuracy(),
                cls_windowed.precision(),
                cls_windowed.recall(),
                cls_windowed.f1(),
                cls_cumulative.accuracy(),
            );
            cls_windowed.reset();
        }
    }

    println!(
        "\n  Final cumulative: Acc={:.4}, Prec={:.4}, Rec={:.4}, F1={:.4}, LogLoss={:.4}",
        cls_cumulative.accuracy(),
        cls_cumulative.precision(),
        cls_cumulative.recall(),
        cls_cumulative.f1(),
        cls_cumulative.log_loss(),
    );

    println!("\n[DONE] Streaming metrics example complete.");
}
