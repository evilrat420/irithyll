//! Tree-based factory: SGBT, Distributional, MulticlassSGBT.

use crate::automl::space::{int_range, linear_range, log_range, ParamMap, SearchSpace};
use crate::ensemble::config::SGBTConfig;
use crate::ensemble::distributional::DistributionalSGBT;
use crate::ensemble::multiclass::MulticlassSGBT;
use crate::learner::SGBTLearner;
use irithyll_core::learner::StreamingLearner;

use super::{Factory, FactoryError};

/// Build the canonical SGBT search space.
///
/// **Source for ranges:** `learning_rate` ∈ [0.001, 0.3] log-scaled is the
/// span used in canonical streaming gradient-boosting benchmarks (Beygelzimer
/// et al. 2015 "Online Gradient Boosting"); `n_steps` ∈ [10, 500] covers
/// both stumpy ensembles and deep boosting; `max_depth` ∈ [3, 10] covers
/// the LightGBM/XGBoost default span. Lambda is log-scaled across two orders
/// of magnitude per Friedman (2001) gradient-boosting analysis. Feature
/// subsample rate ∈ [0.3, 1.0] is the canonical SRM regularization range.
fn sgbt_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("learning_rate", log_range(0.001, 0.3))
        .param("n_steps", int_range(10, 500))
        .param("max_depth", int_range(3, 10))
        .param("n_bins", int_range(16, 256))
        .param("lambda", log_range(0.01, 10.0))
        .param("feature_subsample_rate", linear_range(0.3, 1.0))
        .param("grace_period", int_range(3, 200))
        .build()
        .expect("sgbt_search_space: builder produces a valid space by construction")
}

impl Factory {
    /// Create a factory for streaming gradient boosted trees.
    pub fn sgbt(n_features: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Sgbt,
            n_features,
            space: sgbt_search_space(),
            warmup: 0,
            complexity: 500,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for distributional SGBT (Gaussian output with mu + sigma).
    pub fn distributional(n_features: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Distributional,
            n_features,
            space: sgbt_search_space(),
            warmup: 0,
            complexity: 1000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for multi-class SGBT (one-vs-rest committee).
    pub fn multiclass_sgbt(n_features: usize, max_classes: usize) -> Self {
        let space = SearchSpace::builder()
            .param("learning_rate", log_range(0.001, 0.3))
            .param("n_steps", int_range(10, 500))
            .param("max_depth", int_range(3, 10))
            .param("n_bins", int_range(16, 256))
            .param("lambda", log_range(0.01, 10.0))
            .param("n_classes", int_range(2, max_classes.max(2) as i64))
            .build()
            .expect("multiclass_sgbt: builder produces a valid space by construction");
        Self {
            algorithm: super::Algorithm::MulticlassSgbt,
            n_features,
            space,
            warmup: 0,
            complexity: 1000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    pub(crate) fn create_tree(
        &self,
        params: &ParamMap,
    ) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        match self.algorithm {
            super::Algorithm::Sgbt => {
                let learning_rate = params.float("learning_rate")?;
                let n_steps = params.usize("n_steps")?;
                let max_depth = params.usize("max_depth")?;
                let n_bins = params.usize("n_bins")?;
                let lambda = params.float("lambda")?;
                let feature_subsample_rate = params.float("feature_subsample_rate")?;
                let grace_period = params.usize("grace_period")?;

                let mut builder = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .grace_period(grace_period)
                    .error_weight_alpha(0.01)
                    .shadow_warmup(100)
                    .accuracy_based_pruning(self.accuracy_based_pruning);
                if let Some(interval) = self.proactive_prune_interval {
                    builder = builder.proactive_prune_interval(interval);
                }
                if let Some(hl) = self.prune_half_life {
                    builder = builder.prune_half_life(hl);
                }
                let sgbt_config = builder.build()?;
                Ok(Box::new(SGBTLearner::from_config(sgbt_config)))
            }
            super::Algorithm::Distributional => {
                let learning_rate = params.float("learning_rate")?;
                let n_steps = params.usize("n_steps")?;
                let max_depth = params.usize("max_depth")?;
                let n_bins = params.usize("n_bins")?;
                let lambda = params.float("lambda")?;
                let feature_subsample_rate = params.float("feature_subsample_rate")?;
                let grace_period = params.usize("grace_period")?;

                let mut builder = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .grace_period(grace_period)
                    .error_weight_alpha(0.01)
                    .shadow_warmup(100)
                    .accuracy_based_pruning(self.accuracy_based_pruning);
                if let Some(interval) = self.proactive_prune_interval {
                    builder = builder.proactive_prune_interval(interval);
                }
                if let Some(hl) = self.prune_half_life {
                    builder = builder.prune_half_life(hl);
                }
                let sgbt_config = builder.build()?;
                Ok(Box::new(DistributionalSGBT::new(sgbt_config)))
            }
            super::Algorithm::MulticlassSgbt => {
                let learning_rate = params.float("learning_rate")?;
                let n_steps = params.usize("n_steps")?;
                let max_depth = params.usize("max_depth")?;
                let n_bins = params.usize("n_bins")?;
                let lambda = params.float("lambda")?;
                let n_classes = params.usize("n_classes")?.max(2);

                let sgbt_config = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .build()?;

                let model = MulticlassSGBT::new(sgbt_config, n_classes).map_err(|e| {
                    FactoryError::IncompatibleArm {
                        reason: format!("n_classes={} rejected: {}", n_classes, e),
                    }
                })?;
                Ok(Box::new(model))
            }
            _ => panic!("create_tree called on non-tree algorithm"),
        }
    }
}
