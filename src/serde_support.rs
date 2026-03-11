//! Model serialization and deserialization support.
//!
//! Provides JSON serialization (default feature) and optional bincode
//! serialization for model persistence. Currently exposes utility functions
//! for serializing/deserializing generic `Serialize`/`Deserialize` types.

use crate::error::{IrithyllError, Result};
use serde::{Deserialize, Serialize};

/// Serialize a value to a JSON string.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if serialization fails.
#[cfg(feature = "serde-json")]
pub fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|e| IrithyllError::Serialization(e.to_string()))
}

/// Serialize a value to a pretty-printed JSON string.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if serialization fails.
#[cfg(feature = "serde-json")]
pub fn to_json_pretty<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string_pretty(value).map_err(|e| IrithyllError::Serialization(e.to_string()))
}

/// Deserialize a value from a JSON string.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if deserialization fails.
#[cfg(feature = "serde-json")]
pub fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    serde_json::from_str(json).map_err(|e| IrithyllError::Serialization(e.to_string()))
}

/// Serialize a value to JSON bytes.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if serialization fails.
#[cfg(feature = "serde-json")]
pub fn to_json_bytes<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| IrithyllError::Serialization(e.to_string()))
}

/// Deserialize a value from JSON bytes.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if deserialization fails.
#[cfg(feature = "serde-json")]
pub fn from_json_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| IrithyllError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Model checkpoint/restore serialization
// ---------------------------------------------------------------------------

#[cfg(feature = "serde-json")]
use crate::ensemble::config::SGBTConfig;
#[cfg(feature = "serde-json")]
use crate::ensemble::SGBT;
#[cfg(feature = "serde-json")]
use crate::loss::huber::HuberLoss;
#[cfg(feature = "serde-json")]
use crate::loss::logistic::LogisticLoss;
#[cfg(feature = "serde-json")]
use crate::loss::softmax::SoftmaxLoss;
#[cfg(feature = "serde-json")]
use crate::loss::squared::SquaredLoss;
#[cfg(feature = "serde-json")]
use crate::loss::Loss;

/// Tag for reconstructing the correct loss function on deserialization.
///
/// When saving a model, the loss type is captured as a tag so the correct
/// loss function can be reconstructed on load without runtime type inspection.
#[cfg(feature = "serde-json")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossType {
    Squared,
    Logistic,
    Huber { delta: f64 },
    Softmax { n_classes: usize },
}

#[cfg(feature = "serde-json")]
impl LossType {
    /// Reconstruct the loss function from its serialized tag.
    pub fn into_loss(self) -> Box<dyn Loss> {
        match self {
            LossType::Squared => Box::new(SquaredLoss),
            LossType::Logistic => Box::new(LogisticLoss),
            LossType::Huber { delta } => Box::new(HuberLoss { delta }),
            LossType::Softmax { n_classes } => Box::new(SoftmaxLoss { n_classes }),
        }
    }
}

/// Serializable snapshot of the tree arena structure.
///
/// Captures the minimal state needed to reconstruct a tree for prediction:
/// node topology, split decisions, and leaf values. Histogram accumulators
/// are NOT serialized -- they rebuild naturally from continued training.
#[cfg(feature = "serde-json")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSnapshot {
    pub feature_idx: Vec<u32>,
    pub threshold: Vec<f64>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub leaf_value: Vec<f64>,
    pub is_leaf: Vec<bool>,
    pub depth: Vec<u16>,
    pub sample_count: Vec<u64>,
    pub n_features: Option<usize>,
    pub samples_seen: u64,
    pub rng_state: u64,
}

/// Serializable snapshot of a single boosting step.
#[cfg(feature = "serde-json")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepSnapshot {
    pub tree: TreeSnapshot,
    pub alternate_tree: Option<TreeSnapshot>,
}

/// Complete serializable state of an SGBT model.
///
/// Captures everything needed to reconstruct a trained model for prediction
/// and continued training. The loss function is stored as a [`LossType`] tag.
#[cfg(feature = "serde-json")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub config: SGBTConfig,
    pub loss_type: LossType,
    pub base_prediction: f64,
    pub base_initialized: bool,
    pub initial_targets: Vec<f64>,
    pub initial_target_count: usize,
    pub samples_seen: u64,
    pub rng_state: u64,
    pub steps: Vec<StepSnapshot>,
}

/// Save an SGBT model to a JSON string.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Arguments
///
/// * `model` - The SGBT model to serialize.
/// * `loss_type` - The loss type tag (must match the loss function used to create the model).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if serialization fails.
#[cfg(feature = "serde-json")]
pub fn save_model(model: &SGBT, loss_type: LossType) -> Result<String> {
    let state = model.to_model_state(loss_type);
    to_json_pretty(&state)
}

/// Load an SGBT model from a JSON string.
///
/// Requires the `serde-json` feature (enabled by default).
///
/// # Errors
///
/// Returns [`IrithyllError::Serialization`] if deserialization fails.
#[cfg(feature = "serde-json")]
pub fn load_model(json: &str) -> Result<SGBT> {
    let state: ModelState = from_json(json)?;
    Ok(SGBT::from_model_state(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    #[cfg(feature = "serde-json")]
    #[test]
    fn json_round_trip_sample() {
        let sample = Sample::new(vec![1.0, 2.0, 3.0], 4.0);
        let json = to_json(&sample).unwrap();
        let restored: Sample = from_json(&json).unwrap();
        assert_eq!(restored.features, sample.features);
        assert!((restored.target - sample.target).abs() < f64::EPSILON);
    }

    #[cfg(feature = "serde-json")]
    #[test]
    fn json_pretty_round_trip() {
        let sample = Sample::weighted(vec![1.0], 2.0, 0.5);
        let json = to_json_pretty(&sample).unwrap();
        assert!(json.contains('\n'));
        let restored: Sample = from_json(&json).unwrap();
        assert!((restored.weight - 0.5).abs() < f64::EPSILON);
    }

    #[cfg(feature = "serde-json")]
    #[test]
    fn json_bytes_round_trip() {
        let sample = Sample::new(vec![10.0, 20.0], 30.0);
        let bytes = to_json_bytes(&sample).unwrap();
        let restored: Sample = from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.features, sample.features);
    }

    #[cfg(feature = "serde-json")]
    #[test]
    fn json_invalid_input_returns_error() {
        let result = from_json::<Sample>("not valid json");
        assert!(result.is_err());
        match result.unwrap_err() {
            IrithyllError::Serialization(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("expected Serialization error, got {:?}", other),
        }
    }

    #[cfg(feature = "serde-json")]
    #[test]
    fn json_batch_samples() {
        let samples = vec![Sample::new(vec![1.0], 2.0), Sample::new(vec![3.0], 4.0)];
        let json = to_json(&samples).unwrap();
        let restored: Vec<Sample> = from_json(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }
}
