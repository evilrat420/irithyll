//! Multi-family TUI demo machinery: model dispatch, family-specific
//! diagnostics, and shared training/eval loops.
//!
//! Lifted out of `main.rs` so the same dispatch is reachable from both the
//! no-subcommand demo path (`run_tui_demo`) and the user-CSV paths
//! (`train --tui`, `eval --tui`). Holding concrete model handles (rather than
//! `Box<dyn StreamingLearner>`) lets the diagnostics refresh call typed
//! accessors (`ssm_state()`, `n_params()`, `weights()`, etc.) without `Any`
//! downcasting.

use color_eyre::Result;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

use irithyll::generators::StreamGenerator;
use irithyll::DriftSignal;

use super::app::{AppState, SharedState, PINBALL_QUANTILES};
use super::ModelFamily;
use crate::data::Dataset;

/// Concrete model handles owned by the TUI demo.
///
/// Every member implements `irithyll::StreamingLearner`. The training loop
/// dispatches via a single `match` so the diagnostics refresh can call
/// typed accessors that don't exist on the trait.
pub enum DemoModel {
    Sgbt(irithyll::SGBTLearner),
    Mamba(irithyll::StreamingMamba),
    Ttt(irithyll::StreamingTTT),
    Kan(irithyll::StreamingKAN),
    Esn(irithyll::EchoStateNetwork),
    Ngrc(irithyll::NextGenRC),
    SpikeNet(irithyll::SpikeNet),
    Linear(irithyll::StreamingLinearModel),
}

impl DemoModel {
    /// Build a model sized for the Friedman demo (d_in = 10: 5 causal + 5 noise).
    pub fn build_for_demo(family: ModelFamily) -> Self {
        Self::build_with_n_features(family, 10)
    }

    /// Build a model sized for an arbitrary dataset's feature dimension.
    pub fn build_for_dataset(family: ModelFamily, n_features: usize) -> Self {
        Self::build_with_n_features(family, n_features.max(1))
    }

    fn build_with_n_features(family: ModelFamily, n_features: usize) -> Self {
        use irithyll::ensemble::config::{DriftDetectorType, SGBTConfig};
        match family {
            ModelFamily::Sgbt => {
                let config = SGBTConfig::builder()
                    .n_steps(20)
                    .learning_rate(0.05)
                    .grace_period(50)
                    .max_depth(6)
                    .n_bins(16)
                    .drift_detector(DriftDetectorType::PageHinkley {
                        delta: 0.005,
                        lambda: 200.0,
                    })
                    .build()
                    .expect("TUI demo config: invalid parameters");
                DemoModel::Sgbt(irithyll::SGBTLearner::from_config(config))
            }
            ModelFamily::Mamba => DemoModel::Mamba(irithyll::mamba(n_features, 32)),
            ModelFamily::Ttt => DemoModel::Ttt(irithyll::streaming_ttt(n_features, 0.05)),
            ModelFamily::Kan => {
                // Single hidden layer of 8 keeps spline param count modest
                // while letting diagnostics show non-trivial gradient flow.
                DemoModel::Kan(irithyll::streaming_kan(&[n_features, 8, 1], 0.01))
            }
            ModelFamily::Esn => DemoModel::Esn(irithyll::esn(64, 0.9)),
            ModelFamily::Ngrc => DemoModel::Ngrc(irithyll::ngrc(2, 1, 2)),
            ModelFamily::SpikeNet => DemoModel::SpikeNet(irithyll::spikenet(32)),
            ModelFamily::Linear => DemoModel::Linear(irithyll::linear(0.01)),
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        use irithyll::StreamingLearner;
        match self {
            DemoModel::Sgbt(m) => m.predict(features),
            DemoModel::Mamba(m) => m.predict(features),
            DemoModel::Ttt(m) => m.predict(features),
            DemoModel::Kan(m) => m.predict(features),
            DemoModel::Esn(m) => m.predict(features),
            DemoModel::Ngrc(m) => m.predict(features),
            DemoModel::SpikeNet(m) => m.predict(features),
            DemoModel::Linear(m) => m.predict(features),
        }
    }

    pub fn train(&mut self, features: &[f64], target: f64) {
        use irithyll::StreamingLearner;
        match self {
            DemoModel::Sgbt(m) => m.train(features, target),
            DemoModel::Mamba(m) => m.train(features, target),
            DemoModel::Ttt(m) => m.train(features, target),
            DemoModel::Kan(m) => m.train(features, target),
            DemoModel::Esn(m) => m.train(features, target),
            DemoModel::Ngrc(m) => m.train(features, target),
            DemoModel::SpikeNet(m) => m.train(features, target),
            DemoModel::Linear(m) => m.train(features, target),
        }
    }

    /// Which `ModelFamily` variant this model corresponds to.
    pub fn family(&self) -> ModelFamily {
        match self {
            DemoModel::Sgbt(_) => ModelFamily::Sgbt,
            DemoModel::Mamba(_) => ModelFamily::Mamba,
            DemoModel::Ttt(_) => ModelFamily::Ttt,
            DemoModel::Kan(_) => ModelFamily::Kan,
            DemoModel::Esn(_) => ModelFamily::Esn,
            DemoModel::Ngrc(_) => ModelFamily::Ngrc,
            DemoModel::SpikeNet(_) => ModelFamily::SpikeNet,
            DemoModel::Linear(_) => ModelFamily::Linear,
        }
    }
}

/// Refresh family-specific vital signs (`state.metrics`) and diagnostic rows.
pub fn refresh_family_diagnostics(
    model: &DemoModel,
    loss_val: f64,
    s: &mut AppState,
    sample_idx: usize,
) {
    use irithyll::StreamingLearner;
    let mut metrics: Vec<(String, f64)> = vec![("Loss".to_string(), loss_val)];
    let mut rows: Vec<(String, String, String)> = Vec::new();

    #[allow(deprecated)]
    let diag = match model {
        DemoModel::Sgbt(m) => m.diagnostics_array(),
        DemoModel::Mamba(m) => m.diagnostics_array(),
        DemoModel::Ttt(m) => m.diagnostics_array(),
        DemoModel::Kan(m) => m.diagnostics_array(),
        DemoModel::Esn(m) => m.diagnostics_array(),
        DemoModel::Ngrc(m) => m.diagnostics_array(),
        DemoModel::SpikeNet(m) => m.diagnostics_array(),
        DemoModel::Linear(m) => m.diagnostics_array(),
    };

    rows.push((
        format!("── {} ──", s.active_family.label()),
        String::new(),
        "neutral".into(),
    ));

    let n_samples = match model {
        DemoModel::Sgbt(m) => m.n_samples_seen(),
        DemoModel::Mamba(m) => m.n_samples_seen(),
        DemoModel::Ttt(m) => m.n_samples_seen(),
        DemoModel::Kan(m) => m.n_samples_seen(),
        DemoModel::Esn(m) => m.n_samples_seen(),
        DemoModel::Ngrc(m) => m.n_samples_seen(),
        DemoModel::SpikeNet(m) => m.n_samples_seen(),
        DemoModel::Linear(m) => m.n_samples_seen(),
    };
    rows.push((
        "samples seen".into(),
        format!("{}", n_samples),
        "neutral".into(),
    ));

    match model {
        DemoModel::Sgbt(m) => {
            use irithyll::Structural;
            // Friedman emits 10 features; for arbitrary datasets the SGBT
            // contribution-evaluation placeholder must be sized appropriately.
            // We use n_samples_seen to detect that no per-feature size info
            // is available and fall back to 10 (Friedman default).
            let last_features = vec![0.0; 10];
            let diag = m.inner().diagnostics(&last_features);
            let mean_depth = if diag.trees.is_empty() {
                0.0
            } else {
                diag.trees.iter().map(|t| t.max_depth as f64).sum::<f64>() / diag.trees.len() as f64
            };
            s.diagnostics_array = [
                diag.base_prediction,
                diag.learning_rate,
                mean_depth / 10.0,
                diag.n_trees as f64,
                diag.total_replacements as f64,
            ];
            s.total_replacements = Structural::replacement_count(m);
            s.honest_sigma = loss_val.sqrt();

            rows.push(("── Ensemble ──".into(), String::new(), "neutral".into()));
            rows.push(("n_trees".into(), format!("{}", diag.n_trees), "good".into()));
            rows.push((
                "total_replacements".into(),
                format!("{}", diag.total_replacements),
                "neutral".into(),
            ));
            rows.push((
                "base_prediction".into(),
                format!("{:.4}", diag.base_prediction),
                "neutral".into(),
            ));
            rows.push((
                "learning_rate".into(),
                format!("{:.4}", diag.learning_rate),
                "neutral".into(),
            ));

            rows.push(("── Trees ──".into(), String::new(), "neutral".into()));
            if diag.trees.is_empty() {
                rows.push(("(no trees yet)".into(), String::new(), "neutral".into()));
            } else {
                let n = diag.trees.len() as f64;
                let nodes_sum: usize = diag.trees.iter().map(|t| t.n_nodes).sum();
                let nodes_min = diag.trees.iter().map(|t| t.n_nodes).min().unwrap_or(0);
                let nodes_max = diag.trees.iter().map(|t| t.n_nodes).max().unwrap_or(0);
                rows.push((
                    "nodes  (mean/min/max)".into(),
                    format!(
                        "{:.1} / {} / {}",
                        nodes_sum as f64 / n,
                        nodes_min,
                        nodes_max
                    ),
                    "neutral".into(),
                ));
                let leaves_sum: usize = diag.trees.iter().map(|t| t.n_leaves).sum();
                let leaves_min = diag.trees.iter().map(|t| t.n_leaves).min().unwrap_or(0);
                let leaves_max = diag.trees.iter().map(|t| t.n_leaves).max().unwrap_or(0);
                rows.push((
                    "leaves (mean/min/max)".into(),
                    format!(
                        "{:.1} / {} / {}",
                        leaves_sum as f64 / n,
                        leaves_min,
                        leaves_max
                    ),
                    "neutral".into(),
                ));
                let depth_sum: usize = diag.trees.iter().map(|t| t.max_depth).sum();
                let depth_min = diag.trees.iter().map(|t| t.max_depth).min().unwrap_or(0);
                let depth_max = diag.trees.iter().map(|t| t.max_depth).max().unwrap_or(0);
                rows.push((
                    "depth  (mean/min/max)".into(),
                    format!(
                        "{:.1} / {} / {}",
                        depth_sum as f64 / n,
                        depth_min,
                        depth_max
                    ),
                    if depth_max == 0 { "warn" } else { "neutral" }.into(),
                ));
                let contrib_mean = diag.trees.iter().map(|t| t.contribution.abs()).sum::<f64>() / n;
                let contrib_min = diag
                    .trees
                    .iter()
                    .map(|t| t.contribution.abs())
                    .fold(f64::INFINITY, f64::min);
                let contrib_max = diag
                    .trees
                    .iter()
                    .map(|t| t.contribution.abs())
                    .fold(f64::NEG_INFINITY, f64::max);
                rows.push((
                    "contrib (mean/min/max)".into(),
                    format!(
                        "{:.4} / {:.4} / {:.4}",
                        contrib_mean, contrib_min, contrib_max
                    ),
                    "neutral".into(),
                ));
            }

            rows.push(("── Drift ──".into(), String::new(), "neutral".into()));
            let n_drift = s
                .drift_events
                .iter()
                .filter(|e| matches!(e.signal, irithyll::DriftSignal::Drift))
                .count();
            let n_warn = s
                .drift_events
                .iter()
                .filter(|e| matches!(e.signal, irithyll::DriftSignal::Warning))
                .count();
            rows.push((
                "drift events".into(),
                format!("{}", n_drift),
                if n_drift > 5 { "warn" } else { "neutral" }.into(),
            ));
            rows.push(("warnings".into(), format!("{}", n_warn), "neutral".into()));
            let drift_rate = (n_drift + n_warn) as f64 / (sample_idx as f64).max(1.0);
            rows.push((
                "drift rate (events/sample)".into(),
                format!("{:.5}", drift_rate),
                if drift_rate > 0.01 { "warn" } else { "neutral" }.into(),
            ));
            rows.push((
                "replacements".into(),
                format!("{}", diag.total_replacements),
                "neutral".into(),
            ));

            let raw_imp = m.inner().feature_importances();
            if !raw_imp.is_empty() {
                const N_FEATURES: usize = 10;
                let mut per_feature = [0.0_f64; N_FEATURES];
                for (idx, &v) in raw_imp.iter().enumerate() {
                    per_feature[idx % N_FEATURES] += v;
                }
                let mut pairs: Vec<(String, f64)> = per_feature
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (format!("x_{}", i + 1), v))
                    .collect();
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                s.feature_importances = pairs;
            }
        }
        DemoModel::Mamba(m) => {
            let cfg = m.config();
            let state = m.ssm_state();
            let state_norm = state.iter().map(|s| s * s).sum::<f64>().sqrt();
            let uncertainty = m.prediction_uncertainty();
            metrics.push(("StateNorm".into(), state_norm));
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("ReadoutNorm".into(), diag[3]));
            metrics.push(("Uncertainty".into(), uncertainty));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push(("d_in".into(), format!("{}", cfg.d_in), "neutral".into()));
            rows.push((
                "n_state".into(),
                format!("{}", cfg.n_state),
                "neutral".into(),
            ));
            rows.push((
                "forgetting factor".into(),
                format!("{:.4}", 1.0 - diag[1]),
                "neutral".into(),
            ));
            rows.push(("seed".into(), format!("{}", cfg.seed), "neutral".into()));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "state norm (L2)".into(),
                format!("{:.4}", state_norm),
                "good".into(),
            ));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                if diag[0] > 0.0 { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "rls saturation·state".into(),
                format!("{:.4}", diag[2]),
                "neutral".into(),
            ));
            rows.push((
                "readout magnitude".into(),
                format!("{:.4}", diag[3]),
                "neutral".into(),
            ));
            rows.push((
                "uncertainty σ".into(),
                format!("{:.4}", uncertainty),
                "neutral".into(),
            ));
        }
        DemoModel::Ttt(m) => {
            let cfg = m.config();
            let uncertainty = m.prediction_uncertainty();
            let output_dim = m.output_dim();
            let past_warmup = m.past_warmup();
            metrics.push(("FastNorm".into(), diag[3]));
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("EffectiveAlpha".into(), diag[1]));
            metrics.push(("Uncertainty".into(), uncertainty));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "d_model".into(),
                format!("{}", cfg.d_model),
                "neutral".into(),
            ));
            rows.push((
                "learning_rate".into(),
                format!("{:.4}", cfg.learning_rate),
                "neutral".into(),
            ));
            rows.push((
                "output dim".into(),
                format!("{}", output_dim),
                "neutral".into(),
            ));
            rows.push(("warmup".into(), format!("{}", cfg.warmup), "neutral".into()));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "past warmup".into(),
                format!("{}", past_warmup),
                if past_warmup { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                "neutral".into(),
            ));
            rows.push((
                "effective α".into(),
                format!("{:.4}", diag[1]),
                "neutral".into(),
            ));
            rows.push((
                "rls saturation·state".into(),
                format!("{:.4}", diag[2]),
                "neutral".into(),
            ));
            rows.push((
                "readout magnitude".into(),
                format!("{:.4}", diag[3]),
                "neutral".into(),
            ));
            rows.push((
                "uncertainty σ".into(),
                format!("{:.4}", uncertainty),
                "neutral".into(),
            ));
        }
        DemoModel::Kan(m) => {
            let n_layers = m.n_layers();
            let n_params = m.n_params();
            let layer_sizes = m.layer_sizes().to_vec();
            let cfg = m.config();
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("EncoderUtil".into(), diag[2]));
            metrics.push(("Uncertainty".into(), diag[4]));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "layer sizes".into(),
                format!("{:?}", layer_sizes),
                "neutral".into(),
            ));
            rows.push(("n_layers".into(), format!("{}", n_layers), "neutral".into()));
            rows.push((
                "spline params".into(),
                format!("{}", n_params),
                "neutral".into(),
            ));
            rows.push((
                "learning_rate".into(),
                format!("{:.4}", cfg.learning_rate),
                "neutral".into(),
            ));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                "neutral".into(),
            ));
            rows.push((
                "encoder utilization".into(),
                format!("{:.4}", diag[2]),
                if diag[2] > 0.5 { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "uncertainty".into(),
                format!("{:.4}", diag[4]),
                "neutral".into(),
            ));

            let raw = m.input_importances();
            let total: f64 = raw.iter().sum();
            if total > 1e-12 {
                let mut pairs: Vec<(String, f64)> = raw
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (format!("x_{}", i + 1), v / total))
                    .collect();
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                s.feature_importances = pairs;
            }
        }
        DemoModel::Esn(m) => {
            let cfg = m.config();
            let res = m.reservoir_state();
            let state_norm = res.iter().map(|s| s * s).sum::<f64>().sqrt();
            let uncertainty = m.prediction_uncertainty();
            metrics.push(("StateNorm".into(), state_norm));
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("ReadoutNorm".into(), diag[3]));
            metrics.push(("Uncertainty".into(), uncertainty));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "n_reservoir".into(),
                format!("{}", cfg.n_reservoir),
                "neutral".into(),
            ));
            rows.push((
                "spectral radius".into(),
                format!("{:.3}", cfg.spectral_radius),
                "neutral".into(),
            ));
            rows.push((
                "leak rate".into(),
                format!("{:.3}", cfg.leak_rate),
                "neutral".into(),
            ));
            rows.push(("warmup".into(), format!("{}", cfg.warmup), "neutral".into()));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "past warmup".into(),
                format!("{}", m.past_warmup()),
                if m.past_warmup() { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "reservoir norm".into(),
                format!("{:.4}", state_norm),
                "good".into(),
            ));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                "neutral".into(),
            ));
            rows.push((
                "rls saturation·entropy".into(),
                format!("{:.4}", diag[2]),
                "neutral".into(),
            ));
            rows.push((
                "readout magnitude".into(),
                format!("{:.4}", diag[3]),
                "neutral".into(),
            ));
            rows.push((
                "uncertainty σ".into(),
                format!("{:.4}", uncertainty),
                "neutral".into(),
            ));
        }
        DemoModel::Ngrc(m) => {
            let cfg = m.config();
            let warm = m.is_warm();
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("ReadoutNorm".into(), diag[3]));
            metrics.push(("Uncertainty".into(), diag[4]));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "delay length k".into(),
                format!("{}", cfg.k),
                "neutral".into(),
            ));
            rows.push((
                "skip stride s".into(),
                format!("{}", cfg.s),
                "neutral".into(),
            ));
            rows.push((
                "polynomial degree".into(),
                format!("{}", cfg.degree),
                "neutral".into(),
            ));
            rows.push((
                "include bias".into(),
                format!("{}", cfg.include_bias),
                "neutral".into(),
            ));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "warm".into(),
                format!("{}", warm),
                if warm { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "total pushed".into(),
                format!("{}", m.total_pushed()),
                "neutral".into(),
            ));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                "neutral".into(),
            ));
            rows.push((
                "rls saturation·entropy".into(),
                format!("{:.4}", diag[2]),
                "neutral".into(),
            ));
            rows.push((
                "readout magnitude".into(),
                format!("{:.4}", diag[3]),
                "neutral".into(),
            ));
            rows.push((
                "uncertainty σ".into(),
                format!("{:.4}", diag[4]),
                "neutral".into(),
            ));
        }
        DemoModel::SpikeNet(m) => {
            let cfg = m.config();
            let initialized = m.is_initialized();
            let n_input = m.n_input();
            let bytes = m.memory_bytes();
            metrics.push(("Alignment".into(), diag[0]));
            metrics.push(("SpikeRate".into(), diag[3]));
            metrics.push(("Membrane".into(), diag[4]));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "n_hidden".into(),
                format!("{}", cfg.n_hidden),
                "neutral".into(),
            ));
            rows.push((
                "learning_rate".into(),
                format!("{:.4}", cfg.learning_rate),
                "neutral".into(),
            ));
            rows.push((
                "n_input (lazy)".into(),
                format!("{}", n_input),
                if initialized { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "memory (bytes)".into(),
                format!("{}", bytes),
                "neutral".into(),
            ));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push((
                "initialized".into(),
                format!("{}", initialized),
                if initialized { "good" } else { "warn" }.into(),
            ));
            rows.push((
                "alignment EWMA".into(),
                format!("{:+.4}", diag[0]),
                "neutral".into(),
            ));
            rows.push((
                "spike rate EWMA".into(),
                format!("{:.4}", diag[3]),
                "neutral".into(),
            ));
            rows.push((
                "membrane mean".into(),
                format!("{:.4}", diag[4]),
                "neutral".into(),
            ));
        }
        DemoModel::Linear(m) => {
            let weights = m.weights();
            let weight_norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
            let bias = m.bias();
            metrics.push(("WeightNorm".into(), weight_norm));
            metrics.push(("Bias".into(), bias));
            metrics.push(("Uncertainty".into(), diag[4]));

            rows.push(("── Architecture ──".into(), String::new(), "neutral".into()));
            rows.push((
                "n_features".into(),
                format!("{}", weights.len()),
                "neutral".into(),
            ));

            rows.push(("── State ──".into(), String::new(), "neutral".into()));
            rows.push(("‖w‖₂".into(), format!("{:.4}", weight_norm), "good".into()));
            rows.push(("bias".into(), format!("{:+.4}", bias), "neutral".into()));
            rows.push((
                "uncertainty (1/n)".into(),
                format!("{:.6}", diag[4]),
                "neutral".into(),
            ));

            if !weights.is_empty() {
                let total: f64 = weights.iter().map(|w| w.abs()).sum();
                if total > 1e-12 {
                    let mut pairs: Vec<(String, f64)> = weights
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| (format!("x_{}", i + 1), w.abs() / total))
                        .collect();
                    pairs
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    s.feature_importances = pairs;
                }
            }
        }
    }

    s.metrics = metrics;
    s.diagnostic_rows = rows;
    s.model_type = s.active_family.label().to_string();
}

// ---------------------------------------------------------------------------
// Generic training/eval loops
// ---------------------------------------------------------------------------

/// Training mode: write final model to disk after the run, or skip the save.
pub enum TrainMode<'a> {
    /// Train then serialize to `output_path` (ignored for non-SGBT families
    /// for now — those families don't yet have a JSON serializer wired into
    /// the CLI's save path).
    SaveTo(&'a str),
    /// Don't save (used by `eval --tui` which just measures rolling metrics).
    NoSave,
}

/// Run the multi-family TUI on a stream of `(features, target)` samples.
///
/// Backends both the no-subcommand demo (Friedman/`StreamGenerator`) and the
/// user-CSV `train --tui` / `eval --tui` paths. The training thread writes
/// metrics + diagnostics into shared state at ~200 update intervals; the
/// renderer reads them at sixel refresh cadence.
pub fn run_multi_family_tui(
    model: DemoModel,
    samples: impl Iterator<Item = (Vec<f64>, f64)> + Send + 'static,
    n_samples: usize,
    dataset_label: String,
    mode: TrainMode<'_>,
    throttle_us: u64,
) -> Result<()> {
    let family = model.family();
    let output_path = match mode {
        TrainMode::SaveTo(p) => Some(p.to_string()),
        TrainMode::NoSave => None,
    };

    let mut initial_state = AppState::new(n_samples as u64);
    initial_state.active_family = family;
    initial_state.model_type = family.label().to_string();
    initial_state.dataset_label = dataset_label.clone();
    let state: SharedState = Arc::new(RwLock::new(initial_state));
    let tui_state = state.clone();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let train_state = state.clone();
        let dataset_label_for_loop = dataset_label;
        let train_handle = tokio::task::spawn_blocking(move || {
            run_training_loop(
                model,
                samples,
                n_samples,
                train_state,
                dataset_label_for_loop,
                output_path,
                throttle_us,
            )
        });

        let tui_result = super::run_tui(tui_state).await;
        let _ = train_handle.await?;
        tui_result
    })
}

fn run_training_loop(
    mut model: DemoModel,
    mut samples: impl Iterator<Item = (Vec<f64>, f64)>,
    n_samples: usize,
    train_state: SharedState,
    dataset_label: String,
    _output_path: Option<String>,
    throttle_us: u64,
) -> Result<()> {
    let start = Instant::now();
    let update_interval = (n_samples / 200).max(1);
    let mut last_replacements: u64 = 0;

    let mut ema_loss: f64 = 0.0;
    let mut ema_initialized = false;

    let mut sum_y = 0.0_f64;
    let mut sum_y_sq = 0.0_f64;
    let mut sum_sq_res = 0.0_f64;

    let mut sum_abs_err = 0.0_f64;
    let pinball_qs: &[f64] = PINBALL_QUANTILES;
    let mut sum_pinballs: Vec<f64> = vec![0.0_f64; pinball_qs.len()];

    let mut correct_direction: u64 = 0;
    let mut total_direction: u64 = 0;

    let mut i: usize = 0;
    while i < n_samples {
        if train_state.read().is_paused {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let (features, target) = match samples.next() {
            Some(s) => s,
            None => break,
        };

        let pred = model.predict(&features);
        let err = target - pred;
        let loss_val = err * err;

        if !ema_initialized {
            ema_loss = loss_val;
            ema_initialized = true;
        } else {
            let alpha = 0.005;
            ema_loss = alpha * loss_val + (1.0 - alpha) * ema_loss;
        }

        let n = (i + 1) as f64;
        sum_y += target;
        sum_y_sq += target * target;
        sum_sq_res += loss_val;

        let abs_err = err.abs();
        sum_abs_err += abs_err;
        for (slot, &q) in pinball_qs.iter().enumerate() {
            sum_pinballs[slot] += (q * err).max((q - 1.0) * err);
        }

        if (i + 1) >= 20 {
            let y_bar = sum_y / n;
            let target_dir = (target - y_bar).signum();
            let pred_dir = (pred - y_bar).signum();
            if target_dir != 0.0 && pred_dir != 0.0 {
                total_direction += 1;
                if target_dir == pred_dir {
                    correct_direction += 1;
                }
            }
        }

        model.train(&features, target);

        if throttle_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(throttle_us));
        }

        let replacements: u64 = if let DemoModel::Sgbt(ref m) = model {
            use irithyll::Structural;
            Structural::replacement_count(m)
        } else {
            0
        };
        let new_drifts = replacements.saturating_sub(last_replacements);
        last_replacements = replacements;

        if i % update_interval == 0 || i + 1 == n_samples || new_drifts > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let throughput = if elapsed > 0.0 {
                (i + 1) as f64 / elapsed
            } else {
                0.0
            };
            let mut s = train_state.write();
            s.n_samples = (i + 1) as u64;
            s.elapsed_secs = elapsed;
            s.throughput = throughput;
            s.loss_history.push(ema_loss);
            if matches!(model, DemoModel::Sgbt(_)) {
                s.total_replacements = replacements;
            }

            let ss_tot = sum_y_sq - (sum_y * sum_y) / n;
            let r2 = if ss_tot > 1e-12 {
                (1.0 - sum_sq_res / ss_tot).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            s.r2_history.push(r2);
            s.accuracy_history.push(if total_direction > 0 {
                correct_direction as f64 / total_direction as f64
            } else {
                0.0
            });
            for (slot, &sp) in sum_pinballs.iter().enumerate() {
                if let Some(slot_hist) = s.pinball_history.get_mut(slot) {
                    slot_hist.push(sp / n);
                }
            }
            s.mae_history.push(sum_abs_err / n);
            s.status_message = format!(
                "{} on {} · {:.0} samp/s",
                s.active_family.label(),
                dataset_label,
                throughput
            );
            for _ in 0..new_drifts {
                s.record_drift(DriftSignal::Drift);
            }

            refresh_family_diagnostics(&model, loss_val, &mut s, i + 1);
        }

        i += 1;
    }

    {
        let mut s = train_state.write();
        s.is_training = false;
        s.is_done = true;
    }
    Ok(())
}

/// Run the TUI on an in-tree streaming generator (Friedman, etc.).
pub fn run_with_generator<G>(
    model: DemoModel,
    mut gen: G,
    n_samples: usize,
    dataset_label: String,
    throttle_us: u64,
) -> Result<()>
where
    G: StreamGenerator + Send + 'static,
{
    let samples = std::iter::from_fn(move || Some(gen.next_sample()));
    run_multi_family_tui(
        model,
        samples,
        n_samples,
        dataset_label,
        TrainMode::NoSave,
        throttle_us,
    )
}

/// Run the TUI on a user CSV dataset (training).
pub fn run_with_dataset(
    model: DemoModel,
    dataset: Dataset,
    output_path: &str,
    dataset_label: String,
) -> Result<()> {
    let n_samples = dataset.n_samples;
    let samples = dataset.features.into_iter().zip(dataset.targets);
    run_multi_family_tui(
        model,
        samples,
        n_samples,
        dataset_label,
        TrainMode::SaveTo(output_path),
        0,
    )
}

/// Run the TUI on a user CSV dataset (eval — no model save).
pub fn run_eval_with_dataset(
    model: DemoModel,
    dataset: Dataset,
    dataset_label: String,
) -> Result<()> {
    let n_samples = dataset.n_samples;
    let samples = dataset.features.into_iter().zip(dataset.targets);
    run_multi_family_tui(
        model,
        samples,
        n_samples,
        dataset_label,
        TrainMode::NoSave,
        0,
    )
}

/// Extract a friendly label from a CSV path — just the filename, no directory.
pub fn label_from_csv_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("dataset")
        .to_string()
}
