//! Custom loss function example: Quantile (Pinball) Loss.
//!
//! Demonstrates implementing the `irithyll::Loss` trait for a custom
//! quantile regression loss at tau=0.9, then training an SGBT model
//! that targets the 90th percentile rather than the mean.

use irithyll::{Loss, RegressionMetrics, SGBTConfig, Sample, SGBT};

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Custom Quantile (Pinball) Loss
// ---------------------------------------------------------------------------

/// Quantile loss (pinball loss) for quantile regression.
///
/// For quantile `tau`:
///   L(y, f) = tau * max(y - f, 0) + (1 - tau) * max(f - y, 0)
///
/// The gradient pushes predictions toward the tau-th quantile of the
/// conditional distribution of y given x.
struct QuantileLoss {
    /// Target quantile in (0, 1). E.g. 0.9 for the 90th percentile.
    tau: f64,
}

impl QuantileLoss {
    fn new(tau: f64) -> Self {
        assert!(tau > 0.0 && tau < 1.0, "tau must be in (0, 1)");
        Self { tau }
    }
}

impl Loss for QuantileLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        if target > prediction {
            -self.tau
        } else {
            1.0 - self.tau
        }
    }

    fn hessian(&self, _target: f64, _prediction: f64) -> f64 {
        // Constant hessian (similar to absolute loss).
        // Using 1.0 for stable gradient boosting updates.
        1.0
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let diff = target - prediction;
        if diff > 0.0 {
            self.tau * diff
        } else {
            (1.0 - self.tau) * (-diff)
        }
    }

    fn predict_transform(&self, raw: f64) -> f64 {
        // Identity transform for regression.
        raw
    }

    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        // Use the empirical quantile as the initial prediction.
        let mut sorted = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f64 * self.tau) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

fn main() {
    println!("=== Irithyll: Custom Loss Function (Quantile Regression) ===");
    println!("Quantile tau=0.9 (targeting 90th percentile)");
    println!("Target function: y = 3*x + heavy-tailed noise\n");

    // 1. Configure the model
    let config = SGBTConfig::builder()
        .n_steps(40)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .expect("valid config");

    // 2. Create SGBT with custom quantile loss
    let quantile_loss = QuantileLoss::new(0.9);
    let mut model = SGBT::with_loss(config.clone(), Box::new(quantile_loss));

    // Also create a standard (mean) regression model for comparison
    let mut mean_model = SGBT::new(config);

    // 3. Generate training data with heavy-tailed noise
    let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
    let n_samples = 2000;
    let mut metrics_quantile = RegressionMetrics::new();
    let mut metrics_mean = RegressionMetrics::new();

    println!("--- Training both models on {} samples ---\n", n_samples);

    for i in 0..n_samples {
        let x = xorshift64(&mut rng) * 10.0 - 5.0;

        // Heavy-tailed noise: mostly small, occasionally large positive spikes
        let u = xorshift64(&mut rng);
        let noise = if u < 0.8 {
            // 80% of the time: small noise in [-1, 1]
            (xorshift64(&mut rng) - 0.5) * 2.0
        } else if u < 0.95 {
            // 15% of the time: moderate positive noise [2, 8]
            xorshift64(&mut rng) * 6.0 + 2.0
        } else {
            // 5% of the time: large positive spike [10, 30]
            xorshift64(&mut rng) * 20.0 + 10.0
        };

        let target = 3.0 * x + noise;

        // Prequential evaluation
        let pred_q = model.predict(&[x]);
        let pred_m = mean_model.predict(&[x]);
        metrics_quantile.update(target, pred_q);
        metrics_mean.update(target, pred_m);

        // Train both models
        let sample = Sample::new(vec![x], target);
        model.train_one(&sample);
        mean_model.train_one(&sample);

        if (i + 1) % 500 == 0 {
            println!(
                "  Samples: {:>5} | Quantile RMSE: {:>8.4} | Mean RMSE: {:>8.4}",
                i + 1,
                metrics_quantile.rmse(),
                metrics_mean.rmse(),
            );
        }
    }

    // 4. Compare predictions: quantile should be ABOVE the mean model
    println!("\n--- Comparing Predictions ---");
    println!("  The quantile model (tau=0.9) should predict ABOVE the mean model,");
    println!("  targeting the 90th percentile of the noise distribution.\n");

    println!(
        "  {:>6} | {:>10} {:>10} {:>10} {:>10}",
        "x", "y=3*x", "mean_pred", "q90_pred", "q90-mean"
    );
    println!("  {}", "-".repeat(56));

    let test_xs = [-4.0, -2.0, 0.0, 2.0, 4.0];
    let mut q90_above_count = 0;

    for x in &test_xs {
        let baseline = 3.0 * x;
        let pred_mean = mean_model.predict(&[*x]);
        let pred_q90 = model.predict(&[*x]);
        let diff = pred_q90 - pred_mean;

        if pred_q90 > pred_mean {
            q90_above_count += 1;
        }

        println!(
            "  {:>6.1} | {:>10.4} {:>10.4} {:>10.4} {:>+10.4}",
            x, baseline, pred_mean, pred_q90, diff
        );
    }

    // 5. Empirical verification: generate test data and check percentiles
    println!("\n--- Empirical Percentile Check ---");
    println!("  For each x, generate 200 noisy samples and check what fraction");
    println!("  falls below the quantile model prediction (should be ~90%).\n");

    let check_xs = [-3.0, 0.0, 3.0];
    for x in &check_xs {
        let q90_pred = model.predict(&[*x]);
        let mut below_count = 0;
        let n_test = 200;

        for _ in 0..n_test {
            let u = xorshift64(&mut rng);
            let noise = if u < 0.8 {
                (xorshift64(&mut rng) - 0.5) * 2.0
            } else if u < 0.95 {
                xorshift64(&mut rng) * 6.0 + 2.0
            } else {
                xorshift64(&mut rng) * 20.0 + 10.0
            };
            let y = 3.0 * x + noise;
            if y <= q90_pred {
                below_count += 1;
            }
        }

        let empirical_pct = below_count as f64 / n_test as f64 * 100.0;
        println!(
            "  x={:>5.1}: q90_pred={:>8.4}, {:.0}% of test data below prediction",
            x, q90_pred, empirical_pct,
        );
    }

    // Summary
    println!("\n--- Summary ---");
    println!(
        "  Quantile model predicted above mean model in {}/{} test points",
        q90_above_count,
        test_xs.len(),
    );
    if q90_above_count >= 3 {
        println!("  [OK] Quantile loss is shifting predictions toward the upper tail");
    } else {
        println!("  [NOTE] Model may need more training data to fully separate from mean");
    }

    println!("\n[DONE] Custom loss example complete.");
}
