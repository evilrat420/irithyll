//! Tests for distributional SGBT (empirical σ and tree-chain modes).

use super::*;
use crate::ensemble::config::ScaleMode;
use crate::learner::StreamingLearner;
use crate::SGBTConfig;

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
fn predict_interval_bounds_correct() {
    let mut model = DistributionalSGBT::new(test_config());
    for i in 0..200 {
        let x = i as f64 * 0.1;
        model.train_one(&(vec![x], x * 2.0));
    }
    let (lo, hi) = model.predict_interval(&[5.0], 1.96);
    let pred = model.predict(&[5.0]);
    assert!(lo < pred.mu);
    assert!(hi > pred.mu);
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
fn sigma_modulated_initializes_rolling_mean() {
    let mut model = DistributionalSGBT::new(modulated_config());
    assert!(model.is_uncertainty_modulated());
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
    assert!((0.1..=10.0).contains(&sigma_ratio));
}

#[test]
fn streaming_learner_returns_mu() {
    let mut model = DistributionalSGBT::new(test_config());
    for i in 0..200 {
        let x = i as f64 * 0.1;
        StreamingLearner::train(&mut model, &[x], x * 2.0 + 1.0);
    }
    let pred = StreamingLearner::predict(&model, &[5.0]);
    let gaussian = DistributionalSGBT::predict(&model, &[5.0]);
    assert!((pred - gaussian.mu).abs() < 1e-12);
}

#[test]
fn diagnostics_returns_correct_tree_count() {
    let model = trained_model();
    let diag = model.diagnostics();
    assert_eq!(diag.trees.len(), 20);
}

#[test]
fn diagnostics_trees_have_leaves() {
    let model = trained_model();
    let diag = model.diagnostics();
    for (i, tree) in diag.trees.iter().enumerate() {
        assert!(tree.n_leaves >= 1, "tree {i} should have at least 1 leaf");
    }
}

#[test]
fn predict_decomposed_reconstructs_prediction() {
    let model = trained_model();
    let features = [5.0, 2.5, 1.0];
    let pred = model.predict(&features);
    let decomp = model.predict_decomposed(&features);
    assert!((decomp.mu() - pred.mu).abs() < 1e-10);
    assert!((decomp.sigma() - pred.sigma).abs() < 1e-10);
}

#[test]
fn feature_importances_work() {
    let model = trained_model();
    let imp = model.feature_importances();
    for (i, &v) in imp.iter().enumerate() {
        assert!(v >= 0.0, "importance {i} should be non-negative, got {v}");
        assert!(v.is_finite(), "importance {i} should be finite");
    }
    let sum: f64 = imp.iter().sum();
    if sum > 0.0 {
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

#[test]
fn empirical_sigma_default_mode() {
    let model = DistributionalSGBT::new(test_config());
    assert_eq!(model.scale_mode(), ScaleMode::Empirical);
}

#[test]
fn tree_chain_mode_trains_scale_trees() {
    let config = SGBTConfig::builder()
        .n_steps(10)
        .learning_rate(0.1)
        .grace_period(10)
        .max_depth(4)
        .n_bins(16)
        .initial_target_count(10)
        .scale_mode(ScaleMode::TreeChain)
        .build()
        .unwrap();
    let mut model = DistributionalSGBT::new(config);
    assert_eq!(model.scale_mode(), ScaleMode::TreeChain);

    for i in 0..500 {
        let x = i as f64 * 0.1;
        model.train_one(&(vec![x, x * 0.5, (i % 3) as f64], x * 2.0 + 1.0));
    }

    let pred = model.predict(&[5.0, 2.5, 1.0]);
    assert!(pred.mu.is_finite());
    assert!(pred.sigma > 0.0);
    assert!(pred.sigma.is_finite());
}

#[test]
fn predict_smooth_returns_finite() {
    let config = SGBTConfig::builder()
        .n_steps(3)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(16)
        .initial_target_count(10)
        .build()
        .unwrap();
    let mut model = DistributionalSGBT::new(config);

    for i in 0..200 {
        let x = (i as f64) * 0.1;
        let features = vec![x, x.sin()];
        model.train_one(&(features, 2.0 * x + 1.0));
    }

    let pred = model.predict_smooth(&[1.0, 1.0_f64.sin()], 0.5);
    assert!(pred.mu.is_finite());
    assert!(pred.sigma.is_finite());
    assert!(pred.sigma > 0.0);
}

#[test]
fn honest_sigma_in_gaussian_prediction() {
    let config = SGBTConfig::builder()
        .n_steps(5)
        .learning_rate(0.1)
        .max_depth(3)
        .grace_period(2)
        .initial_target_count(10)
        .build()
        .unwrap();
    let mut model = DistributionalSGBT::new(config);
    for i in 0..100 {
        let x = i as f64 * 0.1;
        model.train_one(&(vec![x], x * 2.0));
    }

    let pred = model.predict(&[5.0]);
    assert!(pred.honest_sigma.is_finite());
    assert!(pred.honest_sigma >= 0.0);
}

#[test]
fn packed_cache_disabled_by_default() {
    let model = DistributionalSGBT::new(test_config());
    assert!(!model.has_packed_cache());
    assert_eq!(model.config().packed_refresh_interval, 0);
}

#[test]
fn diagnostic_source_impl() {
    use crate::automl::DiagnosticSource;

    let mut model = DistributionalSGBT::new(test_config());

    for i in 0..200 {
        let x = (i as f64) * 0.1;
        model.train_one(&(vec![x, x * 0.3], x.sin()));
    }

    let diag = model.config_diagnostics();
    assert!(diag.is_some());
    let diag = diag.unwrap();

    assert!(diag.effective_dof > 0.0);
    assert!(diag.uncertainty.is_finite());
}

#[test]
fn predict_interpolated_returns_finite() {
    let config = SGBTConfig::builder()
        .n_steps(10)
        .learning_rate(0.1)
        .grace_period(20)
        .max_depth(4)
        .n_bins(16)
        .initial_target_count(10)
        .build()
        .unwrap();
    let mut model = DistributionalSGBT::new(config);
    for i in 0..200 {
        let x = i as f64 * 0.1;
        model.train_one(&(vec![x, x.sin()], x.cos()));
    }

    let pred = model.predict_interpolated(&[1.0, 0.5]);
    assert!(pred.mu.is_finite());
    assert!(pred.sigma > 0.0);
}

#[cfg(test)]
#[cfg(feature = "serde-json")]
mod serde_tests {
    use super::super::*;
    use crate::SGBTConfig;

    fn make_trained_distributional() -> DistributionalSGBT {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x.sin()));
        }
        model
    }

    #[test]
    fn json_round_trip_preserves_predictions() {
        let model = make_trained_distributional();
        let state = model.to_distributional_state();
        let json = crate::serde_support::save_distributional_model(&state).unwrap();
        let loaded_state = crate::serde_support::load_distributional_model(&json).unwrap();
        let restored = DistributionalSGBT::from_distributional_state(loaded_state);

        let test_points = [0.5, 1.0, 2.0, 3.0];
        for &x in &test_points {
            let orig = model.predict(&[x]);
            let rest = restored.predict(&[x]);
            assert!((orig.mu - rest.mu).abs() < 1e-10);
            assert!((orig.sigma - rest.sigma).abs() < 1e-10);
        }
    }

    #[test]
    fn state_preserves_rolling_sigma_mean() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x.sin()));
        }
        let state = model.to_distributional_state();
        assert!(state.uncertainty_modulated_lr);
        assert!(state.rolling_sigma_mean >= 0.0);

        let restored = DistributionalSGBT::from_distributional_state(state);
        assert_eq!(model.n_samples_seen(), restored.n_samples_seen());
    }
}
