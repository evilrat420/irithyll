//! Element-wise tanh bounding applied in-place.
//!
//! This is the canonical "bound a recurrent state before readout" operation.
//! It is a named primitive rather than an inline closure to make the bounding
//! *intent* explicit at call sites in attention and SSM models:
//!
//! ```text
//! tanh_inplace(&mut state);   // ← "this is a deliberate bounding step"
//! vs
//! state.iter_mut().for_each(|x| *x = x.tanh());  // ← intent unclear
//! ```
//!
//! Per the AGENTS.md streaming-ML constitution: anything feeding RLS readout
//! must be bounded. `tanh_inplace` is the recommended bounding primitive for
//! delta-family attention variants (DeltaNet, GatedDeltaNet, DeltaProduct,
//! RWKV-7) and Titans/TTT.

use crate::math;

/// Apply `x ← tanh(x)` to every element of `slice` in-place.
///
/// Maps unbounded recurrent state into `(−1, 1)` before passing to an RLS
/// readout, preventing weight explosion from unbounded feature input.
///
/// This function is trivial but its naming makes the architectural intent
/// visible at call sites: "this is a deliberate readout-bounding step."
#[inline]
pub fn tanh_inplace(slice: &mut [f64]) {
    for x in slice.iter_mut() {
        *x = math::tanh(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;

    /// tanh_inplace matches element-wise tanh applied per element.
    #[test]
    fn tanh_inplace_matches_elementwise() {
        let inputs = [-100.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 100.0];
        let mut slice = inputs;
        tanh_inplace(&mut slice);

        for (&original, &result) in inputs.iter().zip(slice.iter()) {
            let expected = math::tanh(original);
            assert!(
                (result - expected).abs() < 1e-15,
                "tanh_inplace({original}) = {result}, expected {expected}"
            );
        }
    }

    /// All outputs are bounded in [−1, 1] for any finite input.
    ///
    /// Note: tanh can return exactly ±1.0 for extreme inputs in IEEE-754
    /// floating point (e.g. tanh(-1e300) = -1.0), so the test uses ≥ / ≤
    /// rather than strict inequality.
    #[test]
    fn output_is_bounded_in_unit_interval() {
        let mut extremes = [f64::MIN_POSITIVE, -1e300, 0.0, 1e300, f64::MAX / 2.0];
        tanh_inplace(&mut extremes);
        for &x in &extremes {
            assert!(
                (-1.0..=1.0).contains(&x),
                "tanh output {x} must be in [-1, 1] (IEEE-754: exact ±1.0 is allowed at extremes)"
            );
        }
    }

    /// tanh_inplace on an empty slice is a no-op.
    #[test]
    fn empty_slice_is_noop() {
        let mut empty: [f64; 0] = [];
        tanh_inplace(&mut empty); // must not panic
    }

    /// tanh(0) = 0 exactly.
    #[test]
    fn zero_maps_to_zero() {
        let mut slice = [0.0f64];
        tanh_inplace(&mut slice);
        assert_eq!(slice[0], 0.0);
    }

    /// Applied once, values equal tanh of original.
    #[test]
    fn single_application_equals_tanh() {
        let originals = [0.5f64, -0.5, 2.0, -2.0];
        let mut slice = originals;
        tanh_inplace(&mut slice);
        for (&orig, &result) in originals.iter().zip(slice.iter()) {
            let expected = math::tanh(orig);
            assert!(
                (result - expected).abs() < 1e-15,
                "tanh_inplace({orig}) = {result}, expected {expected}"
            );
        }
    }
}
