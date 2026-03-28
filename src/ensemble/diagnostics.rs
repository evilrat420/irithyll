//! Diagnostics for SGBT ensembles.
//!
//! Provides [`TreeDiagnostics`], [`EnsembleDiagnostics`], and
//! [`DistributionalDiagnostics`] for inspecting tree structure, feature
//! importance, per-tree contributions, and ensemble health.

use std::cmp::Ordering;
use std::fmt;

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
    /// Current honest_sigma (tree contribution std dev).
    pub honest_sigma: f64,
    /// Rolling honest_sigma baseline.
    pub rolling_honest_sigma_mean: f64,
    /// Current effective MTS (if adaptive_mts enabled).
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
        writeln!(f, "=== Ensemble Diagnostics ===")?;
        writeln!(
            f,
            "Trees: {}, Base: {:.4}, LR: {:.4}, Samples: {}",
            self.n_trees, self.base_prediction, self.learning_rate, self.n_samples,
        )?;
        writeln!(f, "Total replacements: {}", self.total_replacements)?;

        // Feature importance (top 10)
        let mut importance: Vec<(usize, f64)> = self
            .feature_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let top_n = importance.len().min(10);
        writeln!(f, "Feature importance (top {top_n}):")?;
        for &(feat, imp) in importance.iter().take(top_n) {
            writeln!(f, "  feature[{feat}]: {imp:.4}")?;
        }

        // Tree summary
        if !self.trees.is_empty() {
            let avg_depth = self.trees.iter().map(|t| t.max_depth).sum::<usize>() as f64
                / self.trees.len() as f64;
            let avg_nodes = self.trees.iter().map(|t| t.n_nodes).sum::<usize>() as f64
                / self.trees.len() as f64;
            writeln!(f, "Avg depth: {avg_depth:.1}, Avg nodes: {avg_nodes:.1}")?;
        }

        Ok(())
    }
}

impl fmt::Display for DistributionalDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Distributional Diagnostics ===")?;
        writeln!(f, "--- Location ---")?;
        write!(f, "{}", self.location)?;
        if let Some(ref scale) = self.scale {
            writeln!(f, "--- Scale ---")?;
            write!(f, "{}", scale)?;
        }
        writeln!(f, "honest_sigma: {:.6}", self.honest_sigma)?;
        writeln!(
            f,
            "rolling_honest_sigma_mean: {:.6}",
            self.rolling_honest_sigma_mean,
        )?;
        if let Some(mts) = self.effective_mts {
            writeln!(f, "effective_mts: {mts}")?;
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

#[cfg(test)]
mod tests {
    use crate::ensemble::config::SGBTConfig;
    use crate::ensemble::SGBT;
    use crate::sample::Sample;

    fn default_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .build()
            .unwrap()
    }

    // ------------------------------------------------------------------
    // Test 1: basic diagnostics on a trained model
    // ------------------------------------------------------------------
    #[test]
    fn ensemble_diagnostics_basic() {
        let mut model = SGBT::new(default_config());
        let mut rng: u64 = 42;
        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let y = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x + 0.5 * y;
            model.train_one(&Sample::new(vec![x, y], target));
        }

        let diag = model.diagnostics(&[1.0, 2.0]);
        assert!(
            diag.n_trees > 0,
            "n_trees should be positive, got {}",
            diag.n_trees
        );
        assert_eq!(
            diag.n_trees,
            model.n_steps(),
            "n_trees should equal n_steps"
        );
        assert_eq!(diag.n_samples, 200, "n_samples should match training count");

        // Feature importance should sum to ~1.0 (or be empty if no splits)
        if !diag.feature_importance.is_empty() {
            let sum: f64 = diag.feature_importance.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "feature_importance should sum to 1.0, got {}",
                sum,
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 2: individual tree diagnostics have reasonable values
    // ------------------------------------------------------------------
    #[test]
    fn tree_diagnostics_populated() {
        let mut model = SGBT::new(default_config());
        for i in 0..200 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 0.5], x.sin()));
        }

        let diag = model.diagnostics(&[1.0, 0.5]);
        for (idx, tree) in diag.trees.iter().enumerate() {
            assert!(
                tree.n_nodes > 0,
                "tree[{}] n_nodes should be > 0, got {}",
                idx,
                tree.n_nodes,
            );
            assert!(
                tree.n_leaves > 0,
                "tree[{}] n_leaves should be > 0, got {}",
                idx,
                tree.n_leaves,
            );
            assert!(
                tree.n_leaves <= tree.n_nodes,
                "tree[{}] n_leaves ({}) should be <= n_nodes ({})",
                idx,
                tree.n_leaves,
                tree.n_nodes,
            );
            assert!(
                tree.contribution.is_finite(),
                "tree[{}] contribution should be finite",
                idx,
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 3: feature importance reflects predictive signal
    // ------------------------------------------------------------------
    #[test]
    fn feature_importance_nonzero() {
        let config = SGBTConfig::builder()
            .n_steps(20)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(4)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        // feature[0] is highly predictive, feature[1] is noise
        let mut rng: u64 = 99;
        for _ in 0..500 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng as f64 / u64::MAX as f64) * 0.01;
            let target = 3.0 * x;
            model.train_one(&Sample::new(vec![x, noise], target));
        }

        let diag = model.diagnostics(&[1.0, 0.0]);
        assert!(
            diag.feature_importance.len() >= 2,
            "should have at least 2 features in importance"
        );
        // feature[0] should dominate
        assert!(
            diag.feature_importance[0] > 0.0,
            "feature[0] importance should be > 0"
        );
    }

    // ------------------------------------------------------------------
    // Test 4: Display impl does not panic
    // ------------------------------------------------------------------
    #[test]
    fn diagnostics_display() {
        let mut model = SGBT::new(default_config());
        for i in 0..100 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0], x * 3.0));
        }

        let diag = model.diagnostics(&[1.0, 2.0]);
        let display = format!("{}", diag);
        assert!(
            display.contains("Ensemble Diagnostics"),
            "display should contain header"
        );
        assert!(
            display.contains("Trees:"),
            "display should contain tree count"
        );

        // TreeDiagnostics display
        if let Some(tree) = diag.trees.first() {
            let tree_display = format!("{}", tree);
            assert!(
                tree_display.contains("nodes="),
                "tree display should contain nodes"
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 5: diagnostics without features (None contributions)
    // ------------------------------------------------------------------
    #[test]
    fn diagnostics_no_features() {
        let mut model = SGBT::new(default_config());
        for i in 0..100 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0], x * 3.0));
        }

        let diag = model.diagnostics_overview();
        assert_eq!(diag.n_trees, model.n_steps());
        // All contributions should be 0.0
        for tree in &diag.trees {
            assert!(
                tree.contribution.abs() < 1e-15,
                "contribution without features should be 0.0"
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 6: distributional diagnostics
    // ------------------------------------------------------------------
    #[test]
    fn distributional_diagnostics_basic() {
        use crate::ensemble::distributional::DistributionalSGBT;

        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        for i in 0..200 {
            let x = (i as f64) * 0.1;
            model.train_one(&(vec![x, x * 0.3], x.sin()));
        }

        let diag = model.ensemble_diagnostics(&[1.0, 0.3]);
        assert_eq!(diag.location.n_trees, 5);
        assert!(
            diag.honest_sigma.is_finite(),
            "honest_sigma should be finite"
        );
        assert!(
            diag.rolling_honest_sigma_mean.is_finite(),
            "rolling_honest_sigma_mean should be finite"
        );

        // Display should not panic
        let display = format!("{}", diag);
        assert!(display.contains("Distributional Diagnostics"));
    }
}
