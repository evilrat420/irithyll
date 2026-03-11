//! Python bindings for irithyll streaming gradient boosted trees.
//!
//! Provides `StreamingGBT`, `StreamingGBTConfig`, `ShapExplanation`,
//! and `MultiTargetGBT` as Python classes via PyO3.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use irithyll::ensemble::config::SGBTConfig;
use irithyll::ensemble::multi_target::MultiTargetSGBT;
use irithyll::loss::LossType;
use irithyll::serde_support::{from_json, to_json_pretty, ModelState};
use irithyll::{DynSGBT, ShapValues};

// ---------------------------------------------------------------------------
// StreamingGBTConfig
// ---------------------------------------------------------------------------

/// Configuration for a streaming gradient boosted tree ensemble.
///
/// Builder-style API: each setter returns `self` for chaining.
///
/// Example::
///
///     config = StreamingGBTConfig()
///     config = config.n_steps(100).learning_rate(0.0125).max_depth(6)
///
#[pyclass(name = "StreamingGBTConfig")]
#[derive(Clone)]
struct PyConfig {
    n_steps: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    grace_period: usize,
    lambda: f64,
    gamma: f64,
    delta: f64,
    feature_subsample_rate: f64,
    loss: String,
    huber_delta: f64,
    n_classes: usize,
    feature_names: Option<Vec<String>>,
    max_tree_samples: Option<u64>,
    leaf_half_life: Option<usize>,
    seed: u64,
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new() -> Self {
        Self {
            n_steps: 100,
            learning_rate: 0.0125,
            max_depth: 6,
            n_bins: 64,
            grace_period: 200,
            lambda: 1.0,
            gamma: 0.0,
            delta: 1e-7,
            feature_subsample_rate: 1.0,
            loss: "squared".to_string(),
            huber_delta: 1.0,
            n_classes: 2,
            feature_names: None,
            max_tree_samples: None,
            leaf_half_life: None,
            seed: 42,
        }
    }

    fn n_steps(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.n_steps = value;
        slf
    }

    fn learning_rate(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.learning_rate = value;
        slf
    }

    fn max_depth(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.max_depth = value;
        slf
    }

    fn n_bins(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.n_bins = value;
        slf
    }

    fn grace_period(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.grace_period = value;
        slf
    }

    fn lambda(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.lambda = value;
        slf
    }

    fn gamma(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.gamma = value;
        slf
    }

    fn delta(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.delta = value;
        slf
    }

    fn feature_subsample_rate(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.feature_subsample_rate = value;
        slf
    }

    fn loss(mut slf: PyRefMut<'_, Self>, value: String) -> PyRefMut<'_, Self> {
        slf.loss = value;
        slf
    }

    fn huber_delta(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.huber_delta = value;
        slf
    }

    fn n_classes(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.n_classes = value;
        slf
    }

    fn feature_names(mut slf: PyRefMut<'_, Self>, value: Vec<String>) -> PyRefMut<'_, Self> {
        slf.feature_names = Some(value);
        slf
    }

    fn max_tree_samples(mut slf: PyRefMut<'_, Self>, value: u64) -> PyRefMut<'_, Self> {
        slf.max_tree_samples = Some(value);
        slf
    }

    fn leaf_half_life(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.leaf_half_life = Some(value);
        slf
    }

    fn seed(mut slf: PyRefMut<'_, Self>, value: u64) -> PyRefMut<'_, Self> {
        slf.seed = value;
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingGBTConfig(n_steps={}, lr={}, depth={}, loss='{}')",
            self.n_steps, self.learning_rate, self.max_depth, self.loss
        )
    }
}

impl PyConfig {
    fn build_sgbt_config(&self) -> PyResult<SGBTConfig> {
        let mut builder = SGBTConfig::builder()
            .n_steps(self.n_steps)
            .learning_rate(self.learning_rate)
            .max_depth(self.max_depth)
            .n_bins(self.n_bins)
            .grace_period(self.grace_period)
            .lambda(self.lambda)
            .gamma(self.gamma)
            .delta(self.delta)
            .feature_subsample_rate(self.feature_subsample_rate)
            .seed(self.seed);

        if let Some(ref names) = self.feature_names {
            builder = builder.feature_names(names.clone());
        }
        if let Some(max) = self.max_tree_samples {
            builder = builder.max_tree_samples(max);
        }
        if let Some(hl) = self.leaf_half_life {
            builder = builder.leaf_half_life(hl);
        }

        builder
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn loss_type(&self) -> PyResult<LossType> {
        match self.loss.as_str() {
            "squared" => Ok(LossType::Squared),
            "logistic" => Ok(LossType::Logistic),
            "huber" => Ok(LossType::Huber {
                delta: self.huber_delta,
            }),
            "softmax" => Ok(LossType::Softmax {
                n_classes: self.n_classes,
            }),
            other => Err(PyValueError::new_err(format!(
                "unknown loss '{}': expected 'squared', 'logistic', 'huber', or 'softmax'",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// ShapExplanation
// ---------------------------------------------------------------------------

/// SHAP explanation for a single prediction.
///
/// Attributes:
///     values: numpy array of per-feature SHAP contributions
///     base_value: expected model output (float)
///
/// Invariant: ``base_value + sum(values) ≈ model.predict(features)``
#[pyclass(name = "ShapExplanation")]
struct PyShapExplanation {
    inner: ShapValues,
}

#[pymethods]
impl PyShapExplanation {
    /// Per-feature SHAP contributions as a numpy array.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.values)
    }

    /// Expected model output (base prediction).
    #[getter]
    fn base_value(&self) -> f64 {
        self.inner.base_value
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapExplanation(base_value={:.6}, n_features={})",
            self.inner.base_value,
            self.inner.values.len()
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingGBT
// ---------------------------------------------------------------------------

/// Streaming Gradient Boosted Trees model.
///
/// Supports any loss function (squared, logistic, huber, softmax) via
/// dynamic dispatch. For Python usage this has negligible overhead.
///
/// Example::
///
///     config = StreamingGBTConfig().n_steps(50).learning_rate(0.05)
///     model = StreamingGBT(config)
///     model.train_one(np.array([1.0, 2.0]), 0.5)
///     pred = model.predict(np.array([1.0, 2.0]))
///
#[pyclass(name = "StreamingGBT")]
struct PyModel {
    inner: DynSGBT,
    loss_type: LossType,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(config: &PyConfig) -> PyResult<Self> {
        let sgbt_config = config.build_sgbt_config()?;
        let loss_type = config.loss_type()?;
        let loss = loss_type.clone().into_loss();
        let inner = DynSGBT::with_loss(sgbt_config, loss);
        Ok(Self { inner, loss_type })
    }

    /// Train on a single sample.
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///     target: target value (float)
    fn train_one(&mut self, py: Python<'_>, features: PyReadonlyArray1<f64>, target: f64) {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| {
            self.inner.train_one(&(slice, target));
        });
    }

    /// Predict the raw output for a feature vector.
    ///
    /// Returns:
    ///     float: raw prediction (before loss transform)
    fn predict(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> f64 {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| self.inner.predict(slice))
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic).
    fn predict_transformed(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> f64 {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| self.inner.predict_transformed(slice))
    }

    /// Compute SHAP explanations for a prediction.
    ///
    /// Returns:
    ///     ShapExplanation with per-feature contributions
    fn explain(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> PyShapExplanation {
        let slice = features.as_slice().unwrap();
        let shap = py.allow_threads(|| self.inner.explain(slice));
        PyShapExplanation { inner: shap }
    }

    /// Feature importances based on accumulated split gains.
    ///
    /// Returns:
    ///     numpy array of normalized importances (sum to 1.0)
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let importances = self.inner.feature_importances();
        PyArray1::from_vec(py, importances)
    }

    /// Number of boosting steps.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    /// Total leaves across all active trees.
    #[getter]
    fn total_leaves(&self) -> usize {
        self.inner.total_leaves()
    }

    /// Whether the base prediction has been initialized.
    #[getter]
    fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let state = self.inner.to_model_state_with(self.loss_type.clone());
        to_json_pretty(&state).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load from JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let state: ModelState =
            from_json(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let loss_type = state.loss_type.clone();
        let inner = DynSGBT::from_model_state(state);
        Ok(Self { inner, loss_type })
    }

    /// Save model to a JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load model from a JSON file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let json =
            std::fs::read_to_string(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Self::from_json(&json)
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingGBT(n_steps={}, samples={}, leaves={})",
            self.inner.n_steps(),
            self.inner.n_samples_seen(),
            self.inner.total_leaves()
        )
    }
}

// ---------------------------------------------------------------------------
// MultiTargetGBT
// ---------------------------------------------------------------------------

/// Multi-target regression model.
///
/// Wraps T independent streaming GBT models, one per target dimension.
///
/// Example::
///
///     config = StreamingGBTConfig().n_steps(20)
///     model = MultiTargetGBT(config, n_targets=3)
///     model.train_one(np.array([1.0, 2.0]), np.array([0.5, 1.0, 1.5]))
///     preds = model.predict(np.array([1.0, 2.0]))
///
#[pyclass(name = "MultiTargetGBT")]
struct PyMultiTarget {
    inner: MultiTargetSGBT,
}

#[pymethods]
impl PyMultiTarget {
    #[new]
    fn new(config: &PyConfig, n_targets: usize) -> PyResult<Self> {
        let sgbt_config = config.build_sgbt_config()?;
        let inner = MultiTargetSGBT::new(sgbt_config, n_targets)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Train on a single multi-target sample.
    fn train_one(
        &mut self,
        py: Python<'_>,
        features: PyReadonlyArray1<f64>,
        targets: PyReadonlyArray1<f64>,
    ) {
        let feat_slice = features.as_slice().unwrap();
        let tgt_slice = targets.as_slice().unwrap();
        py.allow_threads(|| {
            self.inner.train_one(feat_slice, tgt_slice);
        });
    }

    /// Predict all target values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let feat_slice = features.as_slice().unwrap();
        let preds = py.allow_threads(|| self.inner.predict(feat_slice));
        PyArray1::from_vec(py, preds)
    }

    /// Number of target dimensions.
    #[getter]
    fn n_targets(&self) -> usize {
        self.inner.n_targets()
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    /// Reset all target models.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiTargetGBT(n_targets={}, samples={})",
            self.inner.n_targets(),
            self.inner.n_samples_seen()
        )
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Irithyll: Streaming Gradient Boosted Trees for Python.
#[pymodule]
fn irithyll_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyShapExplanation>()?;
    m.add_class::<PyMultiTarget>()?;
    Ok(())
}
