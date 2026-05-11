#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::*;
    use crate::ensemble::config::SGBTConfig;
    use crate::learner::StreamingLearner;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap()
    }

    #[test]
    fn fresh_model_predicts_zero() {
        let model = DistributionalSGBT::new(test_config());
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.mu.abs() < 1e-12);
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn sigma_always_positive() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }
        for i in 0..20 {
            let x = i as f64 * 0.5;
            let pred = model.predict(&[x, x * 0.5]);
            assert!(
                pred.sigma > 0.0,
                "sigma must be positive, got {}",
                pred.sigma
            );
            assert!(pred.sigma.is_finite(), "sigma must be finite");
        }
    }

    #[test]
    fn constant_target_has_small_sigma() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], 5.0));
        }
        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn noisy_target_has_finite_predictions() {
        let mut model = DistributionalSGBT::new(test_config());
        let mut rng: u64 = 42;
        for i in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 1000) as f64 / 500.0 - 1.0;
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + noise));
        }
        let pred = model.predict(&[5.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn predict_interval_bounds_correct() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        let (lo, hi) = model.predict_interval(&[5.0], 1.96);
        let pred = model.predict(&[5.0]);
        assert!(lo < pred.mu, "lower bound should be < mu");
        assert!(hi > pred.mu, "upper bound should be > mu");
        assert!((hi - lo - 2.0 * 1.96 * pred.sigma).abs() < 1e-10);
    }

    #[test]
    fn batch_prediction_matches_individual() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], x));
        }
        let features = vec![vec![1.0, 2.0], vec![3.0, 6.0], vec![5.0, 10.0]];
        let batch = model.predict_batch(&features);
        for (feat, batch_pred) in features.iter().zip(batch.iter()) {
            let individual = model.predict(feat);
            assert!((batch_pred.mu - individual.mu).abs() < 1e-12);
            assert!((batch_pred.sigma - individual.sigma).abs() < 1e-12);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        assert!(model.n_samples_seen() > 0);
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
    }

    #[test]
    fn gaussian_prediction_lower_upper() {
        let pred = GaussianPrediction {
            mu: 10.0,
            sigma: 2.0,
            log_sigma: crate::math::ln(2.0),
            honest_sigma: 0.0,
        };
        assert!((pred.lower(1.96) - (10.0 - 1.96 * 2.0)).abs() < 1e-10);
        assert!((pred.upper(1.96) - (10.0 + 1.96 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn train_batch_works() {
        let mut model = DistributionalSGBT::new(test_config());
        let samples: Vec<(Vec<f64>, f64)> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (vec![x], x * 2.0)
            })
            .collect();
        model.train_batch(&samples);
        assert_eq!(model.n_samples_seen(), 100);
    }

    #[test]
    fn debug_format_works() {
        let model = DistributionalSGBT::new(test_config());
        let debug = format!("{:?}", model);
        assert!(debug.contains("DistributionalSGBT"));
    }

    #[test]
    fn n_trees_counts_both_ensembles() {
        let model = DistributionalSGBT::new(test_config());
        assert!(model.n_trees() >= 20);
    }

    fn modulated_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap()
    }

    #[test]
    fn sigma_modulated_initializes_rolling_mean() {
        let mut model = DistributionalSGBT::new(modulated_config());
        assert!(model.is_uncertainty_modulated());
        assert!((model.rolling_sigma_mean() - 1.0).abs() < 1e-12);
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }
        assert!(model.rolling_sigma_mean() > 0.0);
        assert!(model.rolling_sigma_mean().is_finite());
    }

    #[test]
    fn predict_distributional_returns_sigma_ratio() {
        let mut model = DistributionalSGBT::new(modulated_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + 1.0));
        }
        let (mu, sigma, sigma_ratio) = model.predict_distributional(&[5.0]);
        assert!(mu.is_finite());
        assert!(sigma > 0.0);
        assert!(
            (0.1..=10.0).contains(&sigma_ratio),
            "sigma_ratio={}",
            sigma_ratio
        );
    }

    #[test]
    fn predict_distributional_without_modulation_returns_one() {
        let mut model = DistributionalSGBT::new(test_config());
        assert!(!model.is_uncertainty_modulated());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        let (_mu, _sigma, sigma_ratio) = model.predict_distributional(&[5.0]);
        assert!(
            (sigma_ratio - 1.0).abs() < 1e-12,
            "should be 1.0 when disabled"
        );
    }

    #[test]
    fn modulated_model_sigma_finite_under_varying_noise() {
        let mut model = DistributionalSGBT::new(modulated_config());
        let mut rng: u64 = 123;
        for i in 0..500 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 1000) as f64 / 100.0 - 5.0;
            let x = i as f64 * 0.1;
            let scale = if i < 250 { 1.0 } else { 5.0 };
            model.train_one(&(vec![x], x * 2.0 + noise * scale));
        }
        let pred = model.predict(&[10.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
        assert!(model.rolling_sigma_mean().is_finite());
    }

    #[test]
    fn reset_clears_rolling_sigma_mean() {
        let mut model = DistributionalSGBT::new(modulated_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        let sigma_before = model.rolling_sigma_mean();
        assert!(sigma_before > 0.0);
        model.reset();
        assert!((model.rolling_sigma_mean() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_learner_returns_mu() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            StreamingLearner::train_one(&mut model, &[x], x * 2.0 + 1.0, 1.0);
        }
        let pred = StreamingLearner::predict(&model, &[5.0]);
        let gaussian = DistributionalSGBT::predict(&model, &[5.0]);
        assert!(
            (pred - gaussian.mu).abs() < 1e-12,
            "StreamingLearner::predict should return mu"
        );
    }

    fn trained_model() -> DistributionalSGBT {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..500 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5, (i % 3) as f64], x * 2.0 + 1.0));
        }
        model
    }

    #[test]
    fn diagnostics_returns_correct_tree_count() {
        let model = trained_model();
        let diag = model.diagnostics();
        assert_eq!(diag.trees.len(), 20, "should have 2*n_steps trees");
    }

    #[test]
    fn diagnostics_trees_have_leaves() {
        let model = trained_model();
        let diag = model.diagnostics();
        for (i, tree) in diag.trees.iter().enumerate() {
            assert!(tree.n_leaves >= 1, "tree {i} should have at least 1 leaf");
        }
        let total_samples: u64 = diag.trees.iter().map(|t| t.samples_seen).sum();
        assert!(
            total_samples > 0,
            "at least some trees should have seen samples"
        );
    }

    #[test]
    fn diagnostics_leaf_weight_stats_finite() {
        let model = trained_model();
        let diag = model.diagnostics();
        for (i, tree) in diag.trees.iter().enumerate() {
            let (min, max, mean, std) = tree.leaf_weight_stats;
            assert!(min.is_finite(), "tree {i} min not finite");
            assert!(max.is_finite(), "tree {i} max not finite");
            assert!(mean.is_finite(), "tree {i} mean not finite");
            assert!(std.is_finite(), "tree {i} std not finite");
            assert!(min <= max, "tree {i} min > max");
        }
    }

    #[test]
    fn diagnostics_base_predictions_match() {
        let model = trained_model();
        let diag = model.diagnostics();
        assert!(
            (diag.location_base - model.predict(&[0.0, 0.0, 0.0]).mu).abs() < 100.0,
            "location_base should be plausible"
        );
    }

    #[test]
    fn predict_decomposed_reconstructs_prediction() {
        let model = trained_model();
        let features = [5.0, 2.5, 1.0];
        let pred = model.predict(&features);
        let decomp = model.predict_decomposed(&features);
        assert!(
            (decomp.mu() - pred.mu).abs() < 1e-10,
            "decomposed mu ({}) != predict mu ({})",
            decomp.mu(),
            pred.mu
        );
        assert!(
            (decomp.sigma() - pred.sigma).abs() < 1e-10,
            "decomposed sigma ({}) != predict sigma ({})",
            decomp.sigma(),
            pred.sigma
        );
    }
}
