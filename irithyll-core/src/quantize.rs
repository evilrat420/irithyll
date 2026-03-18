//! f64 → f32 quantization utilities for packed export.
//!
//! When converting a trained SGBT (f64 precision) to the packed format (f32),
//! thresholds and leaf values are quantized. This module provides validation
//! to ensure the precision loss is acceptable.

/// Maximum acceptable absolute difference between f64 and f32 representations.
pub const DEFAULT_TOLERANCE: f64 = 1e-5;

/// Quantize an f64 threshold to f32, returning the f32 value.
#[inline]
pub fn quantize_threshold(value: f64) -> f32 {
    value as f32
}

/// Quantize a leaf value with learning rate baked in.
///
/// Returns `lr * leaf_value` as f32.
#[inline]
pub fn quantize_leaf(leaf_value: f64, learning_rate: f64) -> f32 {
    (learning_rate * leaf_value) as f32
}

/// Check whether quantizing `value` to f32 stays within tolerance.
///
/// Returns `true` if `|value - (value as f32) as f64| <= tolerance`.
#[inline]
pub fn within_tolerance(value: f64, tolerance: f64) -> bool {
    let quantized = value as f32;
    let roundtrip = quantized as f64;
    let diff = (value - roundtrip).abs();
    diff <= tolerance
}

/// Compute the maximum absolute quantization error across a slice of f64 values.
pub fn max_quantization_error(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|&v| {
            let q = v as f32;
            (v - q as f64).abs()
        })
        .fold(0.0f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values_quantize_exactly() {
        // Small integers are exactly representable in f32
        assert!(within_tolerance(0.0, DEFAULT_TOLERANCE));
        assert!(within_tolerance(1.0, DEFAULT_TOLERANCE));
        assert!(within_tolerance(-1.0, DEFAULT_TOLERANCE));
        assert!(within_tolerance(0.5, DEFAULT_TOLERANCE));
    }

    #[test]
    fn typical_thresholds_within_tolerance() {
        // Typical tree thresholds are small-ish floats
        let thresholds = [0.001, 0.1, 1.5, 10.0, 100.0, -0.5, -50.0];
        for &t in &thresholds {
            assert!(
                within_tolerance(t, DEFAULT_TOLERANCE),
                "threshold {} should be within tolerance",
                t
            );
        }
    }

    #[test]
    fn quantize_leaf_bakes_in_lr() {
        let leaf = 2.0;
        let lr = 0.1;
        let q = quantize_leaf(leaf, lr);
        assert!((q - 0.2f32).abs() < 1e-7);
    }

    #[test]
    fn max_error_of_empty_slice() {
        assert_eq!(max_quantization_error(&[]), 0.0);
    }

    #[test]
    fn max_error_tracks_worst_case() {
        let values = [0.0, 1.0, 0.1]; // 0.1 has the worst f32 roundtrip
        let err = max_quantization_error(&values);
        assert!(err > 0.0);
        assert!(err < 1e-7); // 0.1 still very close in f32
    }
}
