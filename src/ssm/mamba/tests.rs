//! Tests for StreamingMamba.

use super::*;
use crate::learner::StreamingLearner;
use crate::ssm::mamba_config::{MambaConfig, MambaVersion};
use irithyll_core::math::silu;

fn default_config(d_in: usize) -> MambaConfig {
    MambaConfig::builder().d_in(d_in).build().unwrap()
}

#[test]
fn new_creates_fresh_model() {
    let model = StreamingMamba::new(default_config(3));
    assert_eq!(model.n_samples_seen(), 0);
    assert_eq!(
        model.last_features().len(),
        6,
        "last_features should have 2*d_in=6 dimensions, got {}",
        model.last_features().len()
    );
    assert_eq!(
        model.gate_weights.len(),
        3 * 3,
        "gate_weights should be d_in * d_in = 9"
    );
    assert!(
        model.gate_weights.iter().any(|&w| w.abs() > 1e-15),
        "gate_weights should be non-zero after Xavier init"
    );
    assert_eq!(
        model.gate_bias.len(),
        3,
        "gate_bias should have d_in = 3 elements"
    );
    assert!(
        model.gate_bias.iter().all(|&b| b.abs() < 1e-15),
        "gate_bias should be zero-initialized"
    );
}

#[test]
fn train_one_increments_samples() {
    let mut model = StreamingMamba::new(default_config(2));
    model.train_one(&[1.0, 2.0], 3.0, 1.0);
    assert_eq!(model.n_samples_seen(), 1);
    model.train_one(&[4.0, 5.0], 6.0, 1.0);
    assert_eq!(model.n_samples_seen(), 2);
}

#[test]
fn predict_before_training_returns_zero() {
    let model = StreamingMamba::new(default_config(3));
    let pred = model.predict(&[1.0, 2.0, 3.0]);
    assert!(
        pred.abs() < 1e-15,
        "prediction before training should be 0.0, got {}",
        pred
    );
}

#[test]
fn predict_after_training_is_finite() {
    let mut model = StreamingMamba::new(default_config(2));
    model.train(&[1.0, 2.0], 3.0);
    let pred = model.predict(&[1.0, 2.0]);
    assert!(
        pred.is_finite(),
        "prediction should be finite, got {}",
        pred
    );
}

#[test]
fn reset_clears_everything() {
    let mut model = StreamingMamba::new(default_config(2));
    model.train(&[1.0, 2.0], 3.0);
    model.train(&[4.0, 5.0], 6.0);
    assert_eq!(model.n_samples_seen(), 2);

    model.reset();
    assert_eq!(model.n_samples_seen(), 0);
    for &f in model.last_features() {
        assert!(
            f.abs() < 1e-15,
            "last_features should be zeroed after reset"
        );
    }
    for &h in model.ssm_state() {
        assert!(h.abs() < 1e-15, "SSM state should be zeroed after reset");
    }
}

#[test]
fn gating_filters_ssm_output() {
    let config = MambaConfig::builder()
        .d_in(3)
        .n_state(8)
        .seed(42)
        .build()
        .unwrap();

    let mut model_gated = StreamingMamba::new(config.clone());
    let mut model_zeroed = StreamingMamba::new(config);
    for w in model_zeroed.gate_weights.iter_mut() {
        *w = 0.0;
    }

    let x = [1.0, 0.5, -0.3];
    let y = 2.0;

    model_gated.train(&x, y);
    model_zeroed.train(&x, y);

    let pred_gated = model_gated.predict(&x);
    let pred_zeroed = model_zeroed.predict(&x);

    assert!(
        pred_gated.is_finite(),
        "gated prediction should be finite, got {}",
        pred_gated
    );
    assert!(
        pred_zeroed.is_finite(),
        "zeroed-gate prediction should be finite, got {}",
        pred_zeroed
    );

    assert!(
        (pred_gated - pred_zeroed).abs() > 1e-15,
        "gated and zeroed-gate predictions should differ: gated={}, zeroed={}",
        pred_gated,
        pred_zeroed
    );
}

#[test]
fn silu_activation_correctness() {
    assert!((silu(0.0)).abs() < 1e-15, "SiLU(0) should be 0");
    let large = silu(10.0);
    assert!(
        (large - 10.0).abs() < 0.01,
        "SiLU(10) should be close to 10, got {}",
        large
    );
    let neg = silu(-10.0);
    assert!(
        neg.abs() < 0.01,
        "SiLU(-10) should be close to 0, got {}",
        neg
    );
    let mid = silu(-1.0);
    assert!(mid < 0.0, "SiLU(-1) should be negative, got {}", mid);
}

#[test]
fn gate_weights_deterministic() {
    let m1 = StreamingMamba::new(default_config(4));
    let m2 = StreamingMamba::new(default_config(4));
    assert_eq!(
        m1.gate_weights, m2.gate_weights,
        "same seed should produce identical gate weights"
    );
    assert_eq!(
        m1.gate_bias, m2.gate_bias,
        "same seed should produce identical gate bias"
    );
}

#[test]
fn reset_restores_gate_weights() {
    let mut model = StreamingMamba::new(default_config(3));
    let original_weights = model.gate_weights.clone();

    for w in model.gate_weights.iter_mut() {
        *w += 1.0;
    }
    assert_ne!(model.gate_weights, original_weights);

    model.reset();
    assert_eq!(
        model.gate_weights, original_weights,
        "gate weights should be restored to initial values after reset"
    );
}

#[test]
fn train_convenience_uses_unit_weight() {
    let mut model1 = StreamingMamba::new(default_config(2));
    let mut model2 = StreamingMamba::new(default_config(2));

    model1.train(&[1.0, 2.0], 3.0);
    model2.train_one(&[1.0, 2.0], 3.0, 1.0);

    assert_eq!(model1.n_samples_seen(), model2.n_samples_seen());
    let p1 = model1.predict(&[1.0, 2.0]);
    let p2 = model2.predict(&[1.0, 2.0]);
    assert!(
        (p1 - p2).abs() < 1e-12,
        "train() and train_one(w=1) should be equivalent: {} vs {}",
        p1,
        p2
    );
}

#[test]
fn convergence_on_linear_target() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(16)
        .seed(123)
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..500 {
        let x1 = (i as f64) * 0.01;
        let x2 = ((i as f64) * 0.01).sin();
        let x3 = ((i as f64) * 0.01).cos();
        let x4 = (i as f64) * 0.001;
        let target = 2.0 * x1 - 1.5 * x2 + 0.8 * x3 + x4;
        model.train(&[x1, x2, x3, x4], target);
    }

    let final_rmse: f64 = (0..100)
        .map(|i| {
            let t = 5.0 + (i as f64) * 0.01;
            let x = [t, t.sin(), t.cos(), t * 0.001];
            let target = 2.0 * x[0] - 1.5 * x[1] + 0.8 * x[2] + x[3];
            let pred = model.predict(&x);
            (pred - target).powi(2)
        })
        .sum::<f64>()
        .sqrt()
        / 10.0;

    assert!(
        final_rmse < 1.0,
        "RMSE on linear task should be < 1.0, got {}",
        final_rmse
    );
}

#[test]
fn v3_variant_readout_dim() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .version(MambaVersion::V3)
        .build()
        .unwrap();
    let model = StreamingMamba::new(config);

    assert_eq!(
        model.last_features().len(),
        6,
        "V3 d_in=4, n_groups=2 should have readout dim = 4 + 2 = 6, got {}",
        model.last_features().len()
    );
}

#[test]
fn v3_variant_trains_successfully() {
    let config = MambaConfig::builder()
        .d_in(3)
        .n_state(8)
        .n_groups(3)
        .version(MambaVersion::V3)
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..50 {
        let t = i as f64 * 0.1;
        model.train(&[t.sin(), t.cos(), t * 0.5], t.sin());
    }

    let pred = model.predict(&[0.5, 0.5, 0.05]);
    assert!(
        pred.is_finite(),
        "V3 prediction should be finite, got {}",
        pred
    );
}

#[test]
fn block_diagonal_nan_guard_with_large_features() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(32)
        .version(MambaVersion::BlockDiagonal { block_size: 2 })
        .block_size(2)
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..200 {
        let at = 14.96 + (i as f64 % 26.0);
        let ap = 992.89 + (i as f64 % 24.0);
        let rh = 25.36 + (i as f64 % 67.0);
        let pe = 420.26 + (i as f64 % 75.0);
        let x = [at, ap, rh, pe];
        let target = pe;

        model.train(&x, target);

        for (i, &s) in model.ssm_state().iter().enumerate() {
            assert!(
                s.is_finite(),
                "BD SSM state[{i}] became non-finite with Power Plant scale features"
            );
        }
    }

    let test_x = [25.0, 1010.0, 60.0, 450.0];
    let pred = model.predict(&test_x);
    assert!(
        pred.is_finite(),
        "BD predict must be finite on Power Plant-scale features, got {pred}"
    );
}

#[test]
fn mamba_bd_nan_guard_resets_state_not_panic() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(32)
        .version(MambaVersion::BlockDiagonal { block_size: 2 })
        .block_size(2)
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..50 {
        let t = i as f64 * 0.1;
        model.train(&[t.sin(), t.cos(), t * 0.5, 1.0], t.sin());
    }

    model.train(&[25.0, 1013.0, 72.0, 460.0], 460.0);

    let pred = model.predict(&[25.0, 1013.0, 72.0, 460.0]);
    assert!(
        pred.is_finite(),
        "prediction should be finite after large-magnitude step with NaN guard, got {pred}"
    );
}

#[test]
fn mamba_bd_4_features_matches_readout_dim() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(32)
        .version(MambaVersion::BlockDiagonal { block_size: 2 })
        .block_size(2)
        .build()
        .unwrap();
    let model = StreamingMamba::new(config);

    assert_eq!(
        model.last_features().len(),
        6,
        "BD d_in=4, block_size=2 should have readout dim = d_in + n_blocks = 6, got {}",
        model.last_features().len()
    );
}

/// V3Exp variant trains and predicts with finite outputs.
#[test]
fn v3exp_variant_trains_successfully() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .version(MambaVersion::V3Exp { use_bcnorm: false })
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..100 {
        let t = i as f64 * 0.1;
        model.train(&[t.sin(), t.cos(), t * 0.5, 1.0], t.sin());
    }

    let pred = model.predict(&[0.5, 0.5, 0.25, 1.0]);
    assert!(
        pred.is_finite(),
        "V3Exp prediction should be finite, got {}",
        pred
    );
}

/// V3Exp readout dim = d_in + n_groups + 4 * n_groups * n_state.
///
/// V3Exp readout dim = base + lift, where:
///
/// - base = `d_in + n_groups + 4 * n_groups * n_state` — gated output, per-group
///   Frobenius energy, plus per-component `Re(h)`, `Im(h)`, `|h|`, `|h|^2` for
///   the complex-diagonal cell.
/// - lift = `max(64, 32 * d_in)` — random tanh feature lift over raw input bits.
///   Random Feature Network (Rahimi & Recht 2008) approximating Gaussian-RBF
///   kernel ridge regression, mathematically necessary for high-degree
///   polynomial targets like multi-bit XOR parity that V3Exp's BASE features
///   alone (degree-4 max in input bits) cannot express.
#[test]
fn v3exp_readout_dim() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .version(MambaVersion::V3Exp { use_bcnorm: false })
        .build()
        .unwrap();
    let model = StreamingMamba::new(config);
    // base: d_in + n_groups + 4 * n_groups * n_state = 4 + 2 + 4*2*8 = 70
    // lift: max(64, 32 * d_in) = max(64, 128) = 128
    // total: 198
    let expected_base: usize = 4 + 2 + 4 * 2 * 8;
    // Lift width = max(64, 32 * d_in). Compute via runtime helper to avoid
    // const-folded clippy noise on small values where 32 * d_in dominates.
    let expected_lift = StreamingMamba::n_lift_for_config(model.config());
    let expected = expected_base + expected_lift;
    assert_eq!(
        model.last_features().len(),
        expected,
        "V3Exp readout dim should be base ({}) + lift ({}) = {}, got {}",
        expected_base,
        expected_lift,
        expected,
        model.last_features().len()
    );
}

/// V3Exp with BCNorm trains without NaN.
#[test]
fn v3exp_with_bcnorm_trains_successfully() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .version(MambaVersion::V3Exp { use_bcnorm: true })
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..50 {
        let t = i as f64 * 0.1;
        // Large-magnitude inputs to test BCNorm stabilization
        model.train(&[t * 10.0, -t * 5.0, t.sin(), 1.0], t * 2.0);
    }

    let pred = model.predict(&[5.0, -2.5, 0.5, 1.0]);
    assert!(
        pred.is_finite(),
        "V3Exp+BCNorm prediction should be finite, got {}",
        pred
    );
}

/// V3Mimo trains and predicts with finite outputs.
#[test]
fn v3mimo_variant_trains_successfully() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .rank(1)
        .version(MambaVersion::V3Mimo {
            rank: 1,
            use_bcnorm: false,
        })
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    for i in 0..100 {
        let t = i as f64 * 0.1;
        model.train(&[t.sin(), t.cos(), t * 0.5, 1.0], t.sin());
    }

    let pred = model.predict(&[0.5, 0.5, 0.25, 1.0]);
    assert!(
        pred.is_finite(),
        "V3Mimo prediction should be finite, got {}",
        pred
    );
}

/// V3Mimo readout dim = d_in + n_groups (same as V3Exp).
#[test]
fn v3mimo_readout_dim() {
    let config = MambaConfig::builder()
        .d_in(4)
        .n_state(8)
        .n_groups(2)
        .rank(1)
        .version(MambaVersion::V3Mimo {
            rank: 1,
            use_bcnorm: false,
        })
        .build()
        .unwrap();
    let model = StreamingMamba::new(config);
    assert_eq!(
        model.last_features().len(),
        6, // d_in=4 + n_groups=2
        "V3Mimo readout dim should be d_in+n_groups=6, got {}",
        model.last_features().len()
    );
}

/// Option D correctness: predict(x_t) must strongly correlate with x_t, not x_{t-1}.
///
/// A label-leak model (state advanced before readout training) would learn to predict
/// y_t from the SSM output that already contains x_t info, but predict() uses the
/// cached pre-update SSM output. With Option D, both paths use the same pre-update
/// features, so the model genuinely learns from the current input.
///
/// Strategy: train on y_t = x_t[0] * 2.0 for many steps. Then verify that
/// predict(x_a) differs meaningfully from predict(x_b) for distinct x_a, x_b —
/// if predict correlated with the PRIOR step's input instead, it would fail to
/// distinguish inputs that differ only in the current step.
#[test]
fn mamba_predict_reads_current_input() {
    let config = MambaConfig::builder()
        .d_in(2)
        .n_state(8)
        .seed(99)
        .build()
        .unwrap();
    let mut model = StreamingMamba::new(config);

    // Train on a stream where y_t = x_t[0] * 2.0
    for i in 0..200 {
        let x0 = (i as f64) * 0.05;
        model.train_one(&[x0, 0.0], x0 * 2.0, 1.0);
    }

    // predict(x_a) should differ from predict(x_b) for different x_a, x_b
    let pred_a = model.predict(&[1.0, 0.0]);
    let pred_b = model.predict(&[5.0, 0.0]);

    assert!(
        pred_a.is_finite() && pred_b.is_finite(),
        "both predictions must be finite: pred_a={pred_a}, pred_b={pred_b}"
    );
    assert!(
        (pred_a - pred_b).abs() > 0.1,
        "predict must respond to current input: pred_a={pred_a} (x=1.0), pred_b={pred_b} (x=5.0), diff={}",
        (pred_a - pred_b).abs()
    );
}
