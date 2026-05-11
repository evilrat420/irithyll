//! Diagnostics for SGBT ensembles.
//!
//! Provides [`TreeDiagnostics`], [`EnsembleDiagnostics`], and
//! [`DistributionalDiagnostics`] for inspecting tree structure, feature
//! importance, per-tree contributions, and ensemble health.

use alloc::vec;
use alloc::vec::Vec;

use core::cmp::Ordering;
use core::fmt;

/// Diagnostics for a single tree in the ensemble.
#[derive(Debug, Clone)]
pub struct TreeDiagnostics {
    /// Total nodes (internal + leaf).
    pub n_nodes: usize,
    /// Number of leaf nodes.
    pub n_leaves: usize,
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Samples this tree has been trained on.
    pub n_samples: u64,
    /// Number of times this tree slot has been replaced.
    pub n_replacements: u64,
    /// This tree's contribution to the current prediction (`lr * tree.predict(x)`).
    pub contribution: f64,
}

/// Diagnostics for an SGBT ensemble.
#[derive(Debug, Clone)]
pub struct EnsembleDiagnostics {
    /// Per-tree diagnostics.
    pub trees: Vec<TreeDiagnostics>,
    /// Feature importance: fraction of splits per feature across all trees.
    /// Sums to 1.0. Indexed by feature index.
    pub feature_importance: Vec<f64>,
    /// Total number of tree replacements across all slots.
    pub total_replacements: u64,
    /// Number of active trees (n_steps).
    pub n_trees: usize,
    /// Base prediction (intercept).
    pub base_prediction: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Total samples the ensemble has seen.
    pub n_samples: u64,
}

/// Diagnostics for a [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
#[derive(Debug, Clone)]
pub struct DistributionalDiagnostics {
    /// Location (mu) ensemble diagnostics.
    pub location: EnsembleDiagnostics,
    /// Scale (sigma) ensemble diagnostics (if tree-chain mode).
    pub scale: Option<EnsembleDiagnostics>,
    /// Standard deviation of per-tree contributions to the current prediction,
    /// used as a model-derived uncertainty estimate.
    pub honest_sigma: f64,
    /// Exponential moving average of `honest_sigma`, providing a stable
    /// baseline for detecting sudden changes in prediction variance.
    pub rolling_honest_sigma_mean: f64,
    /// Effective minimum-training-samples threshold when `adaptive_mts` is
    /// enabled; `None` if adaptive MTS is not active.
    pub effective_mts: Option<u64>,
}

// ---------------------------------------------------------------------------
// Display impls
// ---------------------------------------------------------------------------

impl fmt::Display for TreeDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "nodes={}, leaves={}, depth={}, samples={}, replacements={}, contribution={:.6}",
            self.n_nodes,
            self.n_leaves,
            self.max_depth,
            self.n_samples,
            self.n_replacements,
            self.contribution,
        )
    }
}

impl fmt::Display for EnsembleDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = writeln!(f, "=== Ensemble Diagnostics ===");
        let _ = writeln!(
            f,
            "Trees: {}, Base: {:.4}, LR: {:.4}, Samples: {}",
            self.n_trees, self.base_prediction, self.learning_rate, self.n_samples,
        );
        let _ = writeln!(f, "Total replacements: {}", self.total_replacements);

        // Feature importance (top 10)
        let mut importance: Vec<(usize, f64)> = self
            .feature_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let top_n = importance.len().min(10);
        let _ = writeln!(f, "Feature importance (top {top_n}):");
        for &(feat, imp) in importance.iter().take(top_n) {
            let _ = writeln!(f, "  feature[{feat}]: {imp:.4}");
        }

        // Tree summary
        if !self.trees.is_empty() {
            let avg_depth = self.trees.iter().map(|t| t.max_depth).sum::<usize>() as f64
                / self.trees.len() as f64;
            let avg_nodes = self.trees.iter().map(|t| t.n_nodes).sum::<usize>() as f64
                / self.trees.len() as f64;
            let _ = writeln!(f, "Avg depth: {avg_depth:.1}, Avg nodes: {avg_nodes:.1}");
        }

        Ok(())
    }
}

impl fmt::Display for DistributionalDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = writeln!(f, "=== Distributional Diagnostics ===");
        let _ = writeln!(f, "--- Location ---");
        let _ = write!(f, "{}", self.location);
        if let Some(ref scale) = self.scale {
            let _ = writeln!(f, "--- Scale ---");
            let _ = write!(f, "{}", scale);
        }
        let _ = writeln!(f, "honest_sigma: {:.6}", self.honest_sigma);
        let _ = writeln!(
            f,
            "rolling_honest_sigma_mean: {:.6}",
            self.rolling_honest_sigma_mean,
        );
        if let Some(mts) = self.effective_mts {
            let _ = writeln!(f, "effective_mts: {mts}");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Builder helpers (used by SGBT and DistributionalSGBT)
// ---------------------------------------------------------------------------

use crate::ensemble::step::BoostingStep;

/// Build per-tree diagnostics and aggregate feature importance from a slice of
/// boosting steps.
///
/// `features` is the feature vector used to compute per-tree contributions.
/// If `None`, contributions are set to 0.0.
#[allow(dead_code)]
pub(crate) fn build_ensemble_diagnostics(
    steps: &[BoostingStep],
    base_prediction: f64,
    learning_rate: f64,
    n_samples: u64,
    features: Option<&[f64]>,
) -> EnsembleDiagnostics {
    let mut trees = Vec::with_capacity(steps.len());
    let mut split_counts: Vec<u64> = Vec::new();

    for step in steps {
        let slot = step.slot();
        let tree = slot.active_tree();
        let arena = tree.arena();

        let n_nodes = arena.n_nodes();
        let n_leaves = arena.n_leaves();
        let max_depth = (0..arena.is_leaf.len())
            .filter(|&i| arena.is_leaf[i])
            .map(|i| arena.depth[i] as usize)
            .max()
            .unwrap_or(0);
        let n_tree_samples = step.n_samples_seen();
        let n_replacements = slot.replacements();

        let contribution = match features {
            Some(f) => learning_rate * step.predict(f),
            None => 0.0,
        };

        // Accumulate split counts per feature using thresholds.
        let thresholds = tree.collect_split_thresholds_per_feature();
        if !thresholds.is_empty() {
            if split_counts.len() < thresholds.len() {
                split_counts.resize(thresholds.len(), 0);
            }
            for (feat_idx, splits) in thresholds.iter().enumerate() {
                if feat_idx < split_counts.len() {
                    split_counts[feat_idx] += splits.len() as u64;
                }
            }
        }

        trees.push(TreeDiagnostics {
            n_nodes,
            n_leaves,
            max_depth,
            n_samples: n_tree_samples,
            n_replacements,
            contribution,
        });
    }

    // Normalize split counts to feature importance (sums to 1.0).
    let total: u64 = split_counts.iter().sum();
    let feature_importance = if total > 0 {
        split_counts
            .iter()
            .map(|&c| c as f64 / total as f64)
            .collect()
    } else {
        vec![0.0; split_counts.len()]
    };

    let total_replacements: u64 = trees.iter().map(|t| t.n_replacements).sum();

    EnsembleDiagnostics {
        n_trees: trees.len(),
        trees,
        feature_importance,
        total_replacements,
        base_prediction,
        learning_rate,
        n_samples,
    }
}
