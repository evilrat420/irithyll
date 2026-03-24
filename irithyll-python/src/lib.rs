//! Python bindings for irithyll streaming gradient boosted trees.
//!
//! Provides `StreamingGBT`, `StreamingGBTConfig`, `ShapExplanation`,
//! and `MultiTargetGBT` as Python classes via PyO3.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use irithyll::ensemble::config::SGBTConfig;
use irithyll::ensemble::distributional::DistributionalSGBT;
use irithyll::ensemble::multi_target::MultiTargetSGBT;
use irithyll::ensemble::multiclass::MulticlassSGBT;
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
    gradient_clip_sigma: Option<f64>,
    max_leaf_output: Option<f64>,
    adaptive_leaf_bound: Option<f64>,
    adaptive_depth: Option<f64>,
    min_hessian_sum: Option<f64>,
    split_reeval_interval: Option<usize>,
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
            gradient_clip_sigma: None,
            max_leaf_output: None,
            adaptive_leaf_bound: None,
            adaptive_depth: None,
            min_hessian_sum: None,
            split_reeval_interval: None,
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

    /// Gradient clipping threshold in units of gradient standard deviation.
    fn gradient_clip_sigma(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.gradient_clip_sigma = Some(value);
        slf
    }

    /// Maximum absolute value for leaf outputs (clamping).
    fn max_leaf_output(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.max_leaf_output = Some(value);
        slf
    }

    /// Per-leaf adaptive output bounds (EWMA-synchronized, self-calibrating).
    fn adaptive_leaf_bound(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.adaptive_leaf_bound = Some(value);
        slf
    }

    /// Per-split information criterion (Lunde-Kleppe-Skaug 2020).
    /// Typical: 7.5 (<=10 features), 9.0 (<=50), 11.0 (<=200).
    fn adaptive_depth(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.adaptive_depth = Some(value);
        slf
    }

    /// Minimum hessian sum required to keep a leaf (suppress thin leaves).
    fn min_hessian_sum(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.min_hessian_sum = Some(value);
        slf
    }

    /// Periodic split re-evaluation interval (in samples).
    fn split_reeval_interval(mut slf: PyRefMut<'_, Self>, value: usize) -> PyRefMut<'_, Self> {
        slf.split_reeval_interval = Some(value);
        slf
    }

    /// Convenience: create a model and fit it on the given data in one call.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///     y: 1D numpy array of shape (n_samples,)
    ///
    /// Returns:
    ///     StreamingGBT: trained model
    fn fit(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<PyModel> {
        let mut model = PyModel::new(self)?;
        model.fit(py, x, y)?;
        Ok(model)
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
        if let Some(v) = self.gradient_clip_sigma {
            builder = builder.gradient_clip_sigma(v);
        }
        if let Some(v) = self.max_leaf_output {
            builder = builder.max_leaf_output(v);
        }
        if let Some(v) = self.adaptive_leaf_bound {
            builder = builder.adaptive_leaf_bound(v);
        }
        if let Some(v) = self.adaptive_depth {
            builder = builder.adaptive_depth(v);
        }
        if let Some(v) = self.min_hessian_sum {
            builder = builder.min_hessian_sum(v);
        }
        if let Some(v) = self.split_reeval_interval {
            builder = builder.split_reeval_interval(v);
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

    /// Train on an entire dataset (numpy arrays).
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///     y: 1D numpy array of shape (n_samples,)
    ///
    /// Iterates rows internally with GIL released for maximum throughput.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let n_rows = x.shape()[0];
        let n_y = y.shape()[0];
        if n_rows != n_y {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but y has {} elements",
                n_rows, n_y
            )));
        }
        // Copy data out of numpy arrays so we can release the GIL.
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<f64> = y.as_slice()?.to_vec();
        py.allow_threads(|| {
            for i in 0..n_rows {
                self.inner.train_one(&(&x_owned[i][..], y_owned[i]));
            }
        });
        Ok(())
    }

    /// Incremental fit (sklearn convention alias for ``fit``).
    ///
    /// For streaming models ``fit`` and ``partial_fit`` are identical:
    /// both process samples incrementally without resetting state.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///     y: 1D numpy array of shape (n_samples,)
    fn partial_fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        self.fit(py, x, y)
    }

    /// Train from a Python iterator yielding ``(features, target)`` tuples.
    ///
    /// This processes samples one at a time, calling the model's ``train_one``
    /// for each ``(numpy_1d, float)`` pair yielded by *data*.
    ///
    /// Args:
    ///     data: Python iterator of ``(numpy_1d, float)`` tuples
    ///     callback: optional callable, invoked every *every_n* samples with
    ///         the current sample count as argument: ``callback(n_samples)``
    ///     every_n: callback frequency (default: 100)
    #[pyo3(signature = (data, callback=None, every_n=None))]
    fn train_stream(
        &mut self,
        _py: Python<'_>,
        data: &Bound<'_, PyAny>,
        callback: Option<&Bound<'_, PyAny>>,
        every_n: Option<usize>,
    ) -> PyResult<()> {
        let freq = every_n.unwrap_or(100);
        let mut count: usize = 0;

        let iter = data.try_iter()?;
        for item_result in iter {
            let item = item_result?;
            let tuple = item.downcast::<pyo3::types::PyTuple>().map_err(|_| {
                PyValueError::new_err("train_stream: each item must be a (features, target) tuple")
            })?;
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "train_stream: expected 2-element tuple, got {}",
                    tuple.len()
                )));
            }
            let features: Vec<f64> = tuple.get_item(0)?.extract()?;
            let target: f64 = tuple.get_item(1)?.extract()?;

            self.inner.train_one(&(&features[..], target));
            count += 1;

            if let Some(cb) = callback {
                if count % freq == 0 {
                    cb.call1((count,))?;
                }
            }
        }

        // Final callback if count is not already a multiple of freq.
        if let Some(cb) = callback {
            if count % freq != 0 {
                cb.call1((count,))?;
            }
        }

        Ok(())
    }

    /// Predict for multiple samples.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///
    /// Returns:
    ///     1D numpy array of predictions
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let n_rows = x.shape()[0];
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let preds = py.allow_threads(|| {
            x_owned
                .iter()
                .map(|row| self.inner.predict(row))
                .collect::<Vec<f64>>()
        });
        PyArray1::from_vec(py, preds)
    }

    /// Compute negative mean squared error on a dataset.
    ///
    /// Follows the sklearn convention where higher is better (hence
    /// negative MSE).
    ///
    /// Args:
    ///     X: 2D numpy array (n_samples, n_features)
    ///     y: 1D numpy array (n_samples,)
    ///
    /// Returns:
    ///     float: negative MSE (higher is better)
    fn score(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        let n_rows = x.shape()[0];
        let n_y = y.shape()[0];
        if n_rows != n_y {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but y has {} elements",
                n_rows, n_y
            )));
        }
        if n_rows == 0 {
            return Err(PyValueError::new_err("empty dataset"));
        }
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<f64> = y.as_slice()?.to_vec();
        let neg_mse = py.allow_threads(|| {
            let mse: f64 = x_owned
                .iter()
                .zip(y_owned.iter())
                .map(|(row, &target)| {
                    let pred = self.inner.predict(row);
                    (pred - target).powi(2)
                })
                .sum::<f64>()
                / n_rows as f64;
            -mse
        });
        Ok(neg_mse)
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

    /// Rich HTML representation for Jupyter notebooks.
    fn _repr_html_(&self) -> String {
        format!(
            "<div style='font-family: monospace; padding: 8px; border: 1px solid #444; border-radius: 4px;'>\
             <b>StreamingGBT</b><br/>\
             Steps: {}<br/>\
             Samples: {}<br/>\
             Leaves: {}\
             </div>",
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

    /// Train on an entire multi-target dataset.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///     Y: 2D numpy array of shape (n_samples, n_targets)
    ///
    /// Iterates rows internally with GIL released for maximum throughput.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let n_rows = x.shape()[0];
        let n_y_rows = y.shape()[0];
        let n_targets = y.shape()[1];
        if n_rows != n_y_rows {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but Y has {} rows",
                n_rows, n_y_rows
            )));
        }
        if n_targets != self.inner.n_targets() {
            return Err(PyValueError::new_err(format!(
                "Y has {} columns but model expects {} targets",
                n_targets,
                self.inner.n_targets()
            )));
        }
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| y.as_array().row(i).to_vec()).collect();
        py.allow_threads(|| {
            for i in 0..n_rows {
                self.inner.train_one(&x_owned[i], &y_owned[i]);
            }
        });
        Ok(())
    }

    /// Incremental fit (sklearn convention alias for ``fit``).
    fn partial_fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.fit(py, x, y)
    }

    /// Predict all target values for a single sample.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let feat_slice = features.as_slice().unwrap();
        let preds = py.allow_threads(|| self.inner.predict(feat_slice));
        PyArray1::from_vec(py, preds)
    }

    /// Predict for multiple samples.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///
    /// Returns:
    ///     2D numpy array of shape (n_samples, n_targets)
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n_rows = x.shape()[0];
        let n_targets = self.inner.n_targets();
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let preds = py.allow_threads(|| {
            x_owned
                .iter()
                .flat_map(|row| self.inner.predict(row))
                .collect::<Vec<f64>>()
        });
        PyArray2::from_vec2(
            py,
            &preds
                .chunks(n_targets)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
// DistributionalGBT
// ---------------------------------------------------------------------------

/// Distributional Streaming GBT -- outputs Gaussian N(mu, sigma^2).
///
/// Provides both point predictions (mu) and uncertainty estimates (sigma)
/// for probabilistic regression.
///
/// Example::
///
///     config = StreamingGBTConfig().n_steps(50).learning_rate(0.05)
///     model = DistributionalGBT(config)
///     model.train_one(np.array([1.0, 2.0]), 0.5)
///     mu, sigma = model.predict(np.array([1.0, 2.0]))
///
#[pyclass(name = "DistributionalGBT")]
struct PyDistributional {
    inner: DistributionalSGBT,
}

#[pymethods]
impl PyDistributional {
    #[new]
    fn new(config: &PyConfig) -> PyResult<Self> {
        let sgbt_config = config.build_sgbt_config()?;
        let inner = DistributionalSGBT::new(sgbt_config);
        Ok(Self { inner })
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

    /// Predict the full Gaussian output (mu, sigma).
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///
    /// Returns:
    ///     tuple: (mu, sigma) where mu is the mean and sigma is the std dev
    fn predict(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> (f64, f64) {
        let slice = features.as_slice().unwrap();
        let pred = py.allow_threads(|| self.inner.predict(slice));
        (pred.mu, pred.sigma)
    }

    /// Predict the mean (location parameter) only.
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///
    /// Returns:
    ///     float: predicted mean
    fn predict_mean(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> f64 {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| self.inner.predict_mu(slice))
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "DistributionalGBT(n_steps={}, samples={}, leaves={})",
            self.inner.n_steps(),
            self.inner.n_samples_seen(),
            self.inner.total_leaves()
        )
    }
}

// ---------------------------------------------------------------------------
// ClassifierGBT
// ---------------------------------------------------------------------------

/// Multi-class classification via one-vs-rest SGBT committees.
///
/// Maintains one SGBT per class, trained with softmax loss. Final predictions
/// are softmax-normalized across all committee outputs.
///
/// Example::
///
///     config = StreamingGBTConfig().n_steps(50).learning_rate(0.05)
///     model = ClassifierGBT(config, n_classes=3)
///     model.train_one(np.array([1.0, 2.0]), 0)
///     predicted_class = model.predict(np.array([1.0, 2.0]))
///     probabilities = model.predict_proba(np.array([1.0, 2.0]))
///
#[pyclass(name = "ClassifierGBT")]
struct PyClassifier {
    inner: MulticlassSGBT,
}

#[pymethods]
impl PyClassifier {
    #[new]
    fn new(config: &PyConfig, n_classes: usize) -> PyResult<Self> {
        let sgbt_config = config.build_sgbt_config()?;
        let inner = MulticlassSGBT::new(sgbt_config, n_classes)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Train on a single sample with a class label.
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///     class_label: integer class index (0-based)
    fn train_one(&mut self, py: Python<'_>, features: PyReadonlyArray1<f64>, class_label: usize) {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| {
            self.inner.train_one(&(slice, class_label as f64));
        });
    }

    /// Predict the most likely class.
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///
    /// Returns:
    ///     int: predicted class index
    fn predict(&self, py: Python<'_>, features: PyReadonlyArray1<f64>) -> usize {
        let slice = features.as_slice().unwrap();
        py.allow_threads(|| self.inner.predict(slice))
    }

    /// Predict class probabilities via softmax normalization.
    ///
    /// Args:
    ///     features: numpy array of feature values
    ///
    /// Returns:
    ///     numpy 1D array of probabilities (sums to ~1.0)
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let slice = features.as_slice().unwrap();
        let proba = py.allow_threads(|| self.inner.predict_proba(slice));
        PyArray1::from_vec(py, proba)
    }

    /// Train on an entire dataset (numpy arrays).
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///     y: 1D numpy array of class labels (integers, 0-based)
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let n_rows = x.shape()[0];
        let n_y = y.shape()[0];
        if n_rows != n_y {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but y has {} elements",
                n_rows, n_y
            )));
        }
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<f64> = y.as_slice()?.to_vec();
        py.allow_threads(|| {
            for i in 0..n_rows {
                self.inner.train_one(&(&x_owned[i][..], y_owned[i]));
            }
        });
        Ok(())
    }

    /// Predict classes for multiple samples.
    ///
    /// Args:
    ///     X: 2D numpy array of shape (n_samples, n_features)
    ///
    /// Returns:
    ///     1D numpy array of predicted class indices
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let n_rows = x.shape()[0];
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let preds = py.allow_threads(|| {
            x_owned
                .iter()
                .map(|row| self.inner.predict(row) as f64)
                .collect::<Vec<f64>>()
        });
        PyArray1::from_vec(py, preds)
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassifierGBT(n_classes={}, samples={})",
            self.inner.n_classes(),
            self.inner.n_samples_seen()
        )
    }
}

// ---------------------------------------------------------------------------
// PrequentialEvaluator
// ---------------------------------------------------------------------------

/// Prequential (test-then-train) evaluator for streaming models.
///
/// For each sample the evaluator first predicts, records the error against
/// the true target, and only then trains on that sample.  This gives an
/// honest online estimate of generalisation performance.
///
/// Example::
///
///     evaluator = PrequentialEvaluator(warmup=200)
///     metrics = evaluator.evaluate(model, X, y)
///     print(metrics["accuracy"], metrics["rmse"])
///
#[pyclass(name = "PrequentialEvaluator")]
struct PyEvaluator {
    warmup: usize,
}

#[pymethods]
impl PyEvaluator {
    #[new]
    #[pyo3(signature = (warmup=None))]
    fn new(warmup: Option<usize>) -> Self {
        Self {
            warmup: warmup.unwrap_or(100),
        }
    }

    /// Run prequential evaluation over an entire dataset.
    ///
    /// For each sample (after the warmup period):
    ///   1. predict
    ///   2. record accuracy / error metrics
    ///   3. train
    ///
    /// Samples in the warmup window are trained on but not evaluated.
    ///
    /// Args:
    ///     model: a StreamingGBT model
    ///     X: 2D numpy array (n_samples, n_features)
    ///     y: 1D numpy array (n_samples,)
    ///
    /// Returns:
    ///     dict with keys ``accuracy``, ``rmse``, ``mae``, ``n_samples``
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        model: &mut PyModel,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n_rows = x.shape()[0];
        let n_y = y.shape()[0];
        if n_rows != n_y {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but y has {} elements",
                n_rows, n_y
            )));
        }

        // Copy data out of numpy so we own it.
        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<f64> = y.as_slice()?.to_vec();

        let mut n_correct: u64 = 0;
        let mut n_total: u64 = 0;
        let mut sum_sq_error: f64 = 0.0;
        let mut sum_abs_error: f64 = 0.0;

        for i in 0..n_rows {
            if i >= self.warmup {
                let pred = model.inner.predict(&x_owned[i]);
                let target = y_owned[i];
                let error = pred - target;

                sum_sq_error += error * error;
                sum_abs_error += error.abs();
                n_total += 1;

                // Round both to nearest integer for classification accuracy.
                if pred.round() as i64 == target.round() as i64 {
                    n_correct += 1;
                }
            }
            model.inner.train_one(&(&x_owned[i][..], y_owned[i]));
        }

        let dict = PyDict::new(py);
        if n_total > 0 {
            let accuracy = n_correct as f64 / n_total as f64;
            let rmse = (sum_sq_error / n_total as f64).sqrt();
            let mae = sum_abs_error / n_total as f64;
            dict.set_item("accuracy", accuracy)?;
            dict.set_item("rmse", rmse)?;
            dict.set_item("mae", mae)?;
        } else {
            dict.set_item("accuracy", f64::NAN)?;
            dict.set_item("rmse", f64::NAN)?;
            dict.set_item("mae", f64::NAN)?;
        }
        dict.set_item("n_samples", n_total)?;
        Ok(dict)
    }

    /// Prequential evaluation with periodic metric snapshots.
    ///
    /// Same logic as ``evaluate`` but returns a list of metric dictionaries,
    /// one snapshot every *every_n* evaluated samples plus a final snapshot.
    ///
    /// Args:
    ///     model: a StreamingGBT model
    ///     X: 2D numpy array (n_samples, n_features)
    ///     y: 1D numpy array (n_samples,)
    ///     every_n: snapshot frequency in evaluated samples (default: 1000)
    ///
    /// Returns:
    ///     list[dict] — each dict has ``accuracy``, ``rmse``, ``mae``,
    ///     ``n_samples``
    #[pyo3(signature = (model, x, y, every_n=None))]
    fn evaluate_streaming<'py>(
        &self,
        py: Python<'py>,
        model: &mut PyModel,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        every_n: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let freq = every_n.unwrap_or(1000);
        let n_rows = x.shape()[0];
        let n_y = y.shape()[0];
        if n_rows != n_y {
            return Err(PyValueError::new_err(format!(
                "X has {} rows but y has {} elements",
                n_rows, n_y
            )));
        }

        let x_owned: Vec<Vec<f64>> = (0..n_rows).map(|i| x.as_array().row(i).to_vec()).collect();
        let y_owned: Vec<f64> = y.as_slice()?.to_vec();

        let mut n_correct: u64 = 0;
        let mut n_total: u64 = 0;
        let mut sum_sq_error: f64 = 0.0;
        let mut sum_abs_error: f64 = 0.0;
        let mut snapshots: Vec<Bound<'py, PyDict>> = Vec::new();
        let mut last_snapshot_at: u64 = 0;

        for i in 0..n_rows {
            if i >= self.warmup {
                let pred = model.inner.predict(&x_owned[i]);
                let target = y_owned[i];
                let error = pred - target;

                sum_sq_error += error * error;
                sum_abs_error += error.abs();
                n_total += 1;

                if pred.round() as i64 == target.round() as i64 {
                    n_correct += 1;
                }

                if n_total % freq as u64 == 0 {
                    let dict = PyDict::new(py);
                    let accuracy = n_correct as f64 / n_total as f64;
                    let rmse = (sum_sq_error / n_total as f64).sqrt();
                    let mae = sum_abs_error / n_total as f64;
                    dict.set_item("accuracy", accuracy)?;
                    dict.set_item("rmse", rmse)?;
                    dict.set_item("mae", mae)?;
                    dict.set_item("n_samples", n_total)?;
                    snapshots.push(dict);
                    last_snapshot_at = n_total;
                }
            }
            model.inner.train_one(&(&x_owned[i][..], y_owned[i]));
        }

        // Final snapshot if we haven't just taken one.
        if n_total > 0 && n_total != last_snapshot_at {
            let dict = PyDict::new(py);
            let accuracy = n_correct as f64 / n_total as f64;
            let rmse = (sum_sq_error / n_total as f64).sqrt();
            let mae = sum_abs_error / n_total as f64;
            dict.set_item("accuracy", accuracy)?;
            dict.set_item("rmse", rmse)?;
            dict.set_item("mae", mae)?;
            dict.set_item("n_samples", n_total)?;
            snapshots.push(dict);
        }

        PyList::new(py, snapshots)
    }

    fn __repr__(&self) -> String {
        format!("PrequentialEvaluator(warmup={})", self.warmup)
    }
}

// ---------------------------------------------------------------------------
// NextGenRC (NG-RC)
// ---------------------------------------------------------------------------

/// Next Generation Reservoir Computer.
///
/// A reservoir-free approach using time-delay embeddings and polynomial
/// features, trained online via RLS. Fully deterministic -- no random weights.
///
/// Example::
///
///     model = NextGenRC(k=3, s=1, degree=2)
///     model.train([1.0], 2.0)
///     model.train([2.0], 3.0)
///     model.train([3.0], 4.0)
///     pred = model.predict([4.0])
///
#[pyclass(name = "NextGenRC")]
struct PyNextGenRC {
    inner: irithyll::NextGenRC,
}

#[pymethods]
impl PyNextGenRC {
    #[new]
    #[pyo3(signature = (k=2, s=1, degree=2, forgetting_factor=0.999))]
    fn new(k: usize, s: usize, degree: usize, forgetting_factor: f64) -> PyResult<Self> {
        let config = irithyll::NGRCConfig::builder()
            .k(k)
            .s(s)
            .degree(degree)
            .forgetting_factor(forgetting_factor)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: irithyll::NextGenRC::new(config),
        })
    }

    /// Train on a single sample.
    fn train(&mut self, features: Vec<f64>, target: f64) {
        use irithyll::StreamingLearner;
        self.inner.train(&features, target);
    }

    /// Predict from a feature vector.
    fn predict(&self, features: Vec<f64>) -> f64 {
        use irithyll::StreamingLearner;
        self.inner.predict(&features)
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        use irithyll::StreamingLearner;
        self.inner.reset();
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        use irithyll::StreamingLearner;
        self.inner.n_samples_seen()
    }

    fn __repr__(&self) -> String {
        use irithyll::StreamingLearner;
        format!("NextGenRC(samples={})", self.inner.n_samples_seen())
    }
}

// ---------------------------------------------------------------------------
// EchoStateNetwork (ESN)
// ---------------------------------------------------------------------------

/// Echo State Network with cycle reservoir topology and RLS readout.
///
/// Example::
///
///     model = EchoStateNetwork(n_reservoir=50, spectral_radius=0.9)
///     for i in range(60):
///         model.train([i * 0.1], 0.0)
///     pred = model.predict([1.0])
///
#[pyclass(name = "EchoStateNetwork")]
struct PyEchoStateNetwork {
    inner: irithyll::EchoStateNetwork,
}

#[pymethods]
impl PyEchoStateNetwork {
    #[new]
    #[pyo3(signature = (n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42))]
    fn new(n_reservoir: usize, spectral_radius: f64, leak_rate: f64, seed: u64) -> PyResult<Self> {
        let config = irithyll::ESNConfig::builder()
            .n_reservoir(n_reservoir)
            .spectral_radius(spectral_radius)
            .leak_rate(leak_rate)
            .seed(seed)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: irithyll::EchoStateNetwork::new(config),
        })
    }

    /// Train on a single sample.
    fn train(&mut self, features: Vec<f64>, target: f64) {
        use irithyll::StreamingLearner;
        self.inner.train(&features, target);
    }

    /// Predict from a feature vector.
    fn predict(&self, features: Vec<f64>) -> f64 {
        use irithyll::StreamingLearner;
        self.inner.predict(&features)
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        use irithyll::StreamingLearner;
        self.inner.reset();
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        use irithyll::StreamingLearner;
        self.inner.n_samples_seen()
    }

    fn __repr__(&self) -> String {
        use irithyll::StreamingLearner;
        format!("EchoStateNetwork(samples={})", self.inner.n_samples_seen())
    }
}

// ---------------------------------------------------------------------------
// StreamingMamba (selective SSM)
// ---------------------------------------------------------------------------

/// Streaming Mamba model (selective state space model with RLS readout).
///
/// Example::
///
///     model = StreamingMamba(d_in=3, n_state=16)
///     model.train([1.0, 2.0, 3.0], 4.0)
///     pred = model.predict([1.0, 2.0, 3.0])
///
#[pyclass(name = "StreamingMamba")]
struct PyStreamingMamba {
    inner: irithyll::StreamingMamba,
}

#[pymethods]
impl PyStreamingMamba {
    #[new]
    #[pyo3(signature = (d_in, n_state=16, seed=42))]
    fn new(d_in: usize, n_state: usize, seed: u64) -> PyResult<Self> {
        let config = irithyll::MambaConfig::builder()
            .d_in(d_in)
            .n_state(n_state)
            .seed(seed)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: irithyll::StreamingMamba::new(config),
        })
    }

    /// Train on a single sample.
    fn train(&mut self, features: Vec<f64>, target: f64) {
        use irithyll::StreamingLearner;
        self.inner.train(&features, target);
    }

    /// Predict from a feature vector.
    fn predict(&self, features: Vec<f64>) -> f64 {
        use irithyll::StreamingLearner;
        self.inner.predict(&features)
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        use irithyll::StreamingLearner;
        self.inner.reset();
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        use irithyll::StreamingLearner;
        self.inner.n_samples_seen()
    }

    fn __repr__(&self) -> String {
        use irithyll::StreamingLearner;
        format!("StreamingMamba(samples={})", self.inner.n_samples_seen())
    }
}

// ---------------------------------------------------------------------------
// SpikeNet (spiking neural network with e-prop)
// ---------------------------------------------------------------------------

/// Spiking Neural Network with e-prop learning.
///
/// Example::
///
///     model = SpikeNet(n_hidden=32)
///     model.train([0.5, -0.3], 1.0)
///     pred = model.predict([0.5, -0.3])
///
#[pyclass(name = "SpikeNet")]
struct PySpikeNet {
    inner: irithyll::SpikeNet,
}

#[pymethods]
impl PySpikeNet {
    #[new]
    #[pyo3(signature = (n_hidden=64, learning_rate=0.001, seed=42))]
    fn new(n_hidden: usize, learning_rate: f64, seed: u64) -> PyResult<Self> {
        let config = irithyll::SpikeNetConfig::builder()
            .n_hidden(n_hidden)
            .learning_rate(learning_rate)
            .seed(seed)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: irithyll::SpikeNet::new(config),
        })
    }

    /// Train on a single sample.
    fn train(&mut self, features: Vec<f64>, target: f64) {
        use irithyll::StreamingLearner;
        self.inner.train(&features, target);
    }

    /// Predict from a feature vector.
    fn predict(&self, features: Vec<f64>) -> f64 {
        use irithyll::StreamingLearner;
        self.inner.predict(&features)
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        use irithyll::StreamingLearner;
        self.inner.reset();
    }

    /// Total samples trained.
    #[getter]
    fn n_samples_seen(&self) -> u64 {
        use irithyll::StreamingLearner;
        self.inner.n_samples_seen()
    }

    fn __repr__(&self) -> String {
        use irithyll::StreamingLearner;
        format!("SpikeNet(samples={})", self.inner.n_samples_seen())
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
    m.add_class::<PyDistributional>()?;
    m.add_class::<PyClassifier>()?;
    m.add_class::<PyEvaluator>()?;
    m.add_class::<PyNextGenRC>()?;
    m.add_class::<PyEchoStateNetwork>()?;
    m.add_class::<PyStreamingMamba>()?;
    m.add_class::<PySpikeNet>()?;
    Ok(())
}
