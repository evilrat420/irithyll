//! Property-based tests using proptest.

use proptest::prelude::*;

use irithyll::loss::logistic::LogisticLoss;
use irithyll::loss::softmax::SoftmaxLoss;
use irithyll::loss::squared::SquaredLoss;
use irithyll::loss::Loss;
use irithyll::tree::builder::TreeConfig;
use irithyll::tree::hoeffding::HoeffdingTree;
use irithyll::tree::StreamingTree;
use irithyll::Sample;

// ---------------------------------------------------------------------------
// 1. SquaredLoss gradient is always finite
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn loss_gradient_finite(
        target in -1000.0..1000.0_f64,
        prediction in -1000.0..1000.0_f64,
    ) {
        let loss = SquaredLoss;
        let g = loss.gradient(target, prediction);
        prop_assert!(g.is_finite(), "gradient should be finite, got {}", g);
    }
}

// ---------------------------------------------------------------------------
// 2. SquaredLoss hessian is non-negative (convex)
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn loss_hessian_nonneg(
        target in -1000.0..1000.0_f64,
        prediction in -1000.0..1000.0_f64,
    ) {
        let loss = SquaredLoss;
        let h = loss.hessian(target, prediction);
        prop_assert!(h >= 0.0, "hessian should be >= 0, got {}", h);
    }
}

// ---------------------------------------------------------------------------
// 3. LogisticLoss gradient is bounded in [-1, 1]
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn logistic_gradient_bounded(
        target in prop::sample::select(vec![0.0_f64, 1.0]),
        prediction in -1000.0..1000.0_f64,
    ) {
        let loss = LogisticLoss;
        let g = loss.gradient(target, prediction);
        prop_assert!(
            (-1.0..=1.0).contains(&g),
            "logistic gradient should be in [-1, 1], got {}",
            g
        );
    }
}

// ---------------------------------------------------------------------------
// 4. LogisticLoss predict_transform (sigmoid) always in [0, 1]
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn logistic_sigmoid_in_01(
        raw in -100.0..100.0_f64,
    ) {
        let loss = LogisticLoss;
        let p = loss.predict_transform(raw);
        prop_assert!(
            (0.0..=1.0).contains(&p),
            "sigmoid should be in [0, 1], got {} for raw={}",
            p, raw
        );
    }
}

// ---------------------------------------------------------------------------
// 5. HoeffdingTree predictions are always finite
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]
    #[test]
    fn tree_prediction_finite(
        seed in 1u64..100_000u64,
    ) {
        let config = TreeConfig::new()
            .max_depth(3)
            .n_bins(8)
            .grace_period(5)
            .lambda(1.0);

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state = seed;

        // Train with random samples
        for _ in 0..100 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let x1 = (rng_state as f64) / (u64::MAX as f64) * 10.0 - 5.0;

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let x2 = (rng_state as f64) / (u64::MAX as f64) * 10.0 - 5.0;

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let gradient = (rng_state as f64) / (u64::MAX as f64) * 4.0 - 2.0;

            let hessian = 1.0; // constant for squared loss

            tree.train_one(&[x1, x2], gradient, hessian);
        }

        // Predictions at arbitrary points must be finite
        let pred = tree.predict(&[0.0, 0.0]);
        prop_assert!(pred.is_finite(), "tree prediction must be finite, got {}", pred);

        let pred2 = tree.predict(&[5.0, -5.0]);
        prop_assert!(pred2.is_finite(), "tree prediction must be finite, got {}", pred2);
    }
}

// ---------------------------------------------------------------------------
// 6. SoftmaxLoss gradient + hessian produce finite values
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn softmax_sums_to_one(
        target in prop::sample::select(vec![0.0_f64, 1.0]),
        prediction in -100.0..100.0_f64,
    ) {
        let loss = SoftmaxLoss { n_classes: 3 };
        let g = loss.gradient(target, prediction);
        let h = loss.hessian(target, prediction);

        prop_assert!(g.is_finite(), "softmax gradient should be finite, got {}", g);
        prop_assert!(h.is_finite(), "softmax hessian should be finite, got {}", h);
        prop_assert!(h > 0.0, "softmax hessian should be positive, got {}", h);
    }
}

// ---------------------------------------------------------------------------
// 7. Sample::new always has weight=1.0
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn sample_weight_positive(
        target in -1000.0..1000.0_f64,
        x1 in -100.0..100.0_f64,
        x2 in -100.0..100.0_f64,
    ) {
        let sample = Sample::new(vec![x1, x2], target);
        prop_assert!(
            (sample.weight - 1.0).abs() < f64::EPSILON,
            "Sample::new should have weight=1.0, got {}",
            sample.weight
        );
    }
}

// ---------------------------------------------------------------------------
// 8. LogisticLoss hessian is always positive
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn logistic_hessian_positive(
        target in prop::sample::select(vec![0.0_f64, 1.0]),
        prediction in -500.0..500.0_f64,
    ) {
        let loss = LogisticLoss;
        let h = loss.hessian(target, prediction);
        prop_assert!(
            h > 0.0,
            "logistic hessian should be positive, got {} for target={}, prediction={}",
            h, target, prediction
        );
    }
}
