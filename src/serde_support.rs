//! Model serialization and deserialization support.
//!
//! Provides JSON serialization (default feature) and optional bincode
//! serialization for model persistence. Currently exposes utility functions
//! for serializing/deserializing generic `Serialize`/`Deserialize` types.

use crate::error::{IrithyllError, Result};

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
        let samples = vec![
            Sample::new(vec![1.0], 2.0),
            Sample::new(vec![3.0], 4.0),
        ];
        let json = to_json(&samples).unwrap();
        let restored: Vec<Sample> = from_json(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }
}
