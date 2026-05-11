//! Softplus-softmax mix: numerically stable smooth probability distribution.
//!
//! Used by Log-Linear Attention (Phase 11) for the bounded mix operation over
//! attention scores. Unlike vanilla softmax, softplus-based normalisation is
//! smoother near zero and avoids the hard step-function behaviour that causes
//! gradient underflow for low-magnitude logits.
//!
//! The operation is:
//! ```text
//! softplus_softmax_mix(x, τ)_i = softplus(x_i / τ) / Σ_j softplus(x_j / τ)
//! ```
//!
//! For large τ (temperature → ∞), softplus(x/τ) ≈ x/τ + ln(2), and the
//! denominator normalises to a nearly-uniform distribution — matching the
//! high-temperature softmax limit. This is verified by the test suite.

use crate::math;

/// Compute the softplus-softmax mix of `input` with temperature `τ` into `output`.
///
/// Each output element is:
/// ```text
/// output_i = softplus(input_i / τ) / Σ_j softplus(input_j / τ)
/// ```
///
/// The result is a valid probability distribution (all non-negative, sums to 1).
///
/// # Arguments
///
/// - `input` — raw logits, length `n`.
/// - `temperature` — τ > 0. Higher values produce a more uniform distribution;
///   lower values sharpen toward argmax.
/// - `output` — must have the same length as `input`.
///
/// # Panics
///
/// Panics in debug mode if `output.len() != input.len()` or `temperature <= 0`.
pub fn softplus_softmax_mix(input: &[f64], temperature: f64, output: &mut [f64]) {
    debug_assert_eq!(
        input.len(),
        output.len(),
        "softplus_softmax_mix: input and output must have the same length"
    );
    debug_assert!(
        temperature > 0.0,
        "softplus_softmax_mix: temperature must be positive, got {temperature}"
    );

    let inv_tau = 1.0 / temperature;
    let mut sum = 0.0;
    for (&xi, o) in input.iter().zip(output.iter_mut()) {
        let sp = math::softplus(xi * inv_tau);
        *o = sp;
        sum += sp;
    }

    // Normalize. sum > 0 is guaranteed because softplus(x) > 0 for all x.
    // When input is empty, we avoid division by zero by guarding.
    if sum > 0.0 {
        for o in output.iter_mut() {
            *o /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;

    /// Output must be non-negative and sum to 1.
    #[test]
    fn output_is_probability_distribution() {
        let input = [1.0, -1.0, 0.5, 2.0, -2.0];
        let mut output = [0.0f64; 5];
        softplus_softmax_mix(&input, 1.0, &mut output);

        for &o in &output {
            assert!(o >= 0.0, "all outputs must be non-negative, got {o}");
        }
        let total: f64 = output.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "outputs must sum to 1, got {total}"
        );
    }

    /// All-zero input should give uniform distribution.
    #[test]
    fn zero_input_gives_uniform_output() {
        let input = [0.0f64; 5];
        let mut output = [0.0f64; 5];
        softplus_softmax_mix(&input, 1.0, &mut output);

        let expected = 1.0 / 5.0_f64;
        for &o in &output {
            assert!(
                (o - expected).abs() < 1e-12,
                "uniform input should give uniform output 1/5={expected}, got {o}"
            );
        }
    }

    /// High temperature approaches a near-uniform distribution (softmax limit).
    #[test]
    fn high_temperature_approaches_uniform() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output_low_t = [0.0f64; 5];
        let mut output_high_t = [0.0f64; 5];
        softplus_softmax_mix(&input, 0.1, &mut output_low_t);
        softplus_softmax_mix(&input, 1000.0, &mut output_high_t);

        // High-T output should be much closer to uniform than low-T.
        let uniform = 1.0 / 5.0;
        let max_dev_high: f64 = output_high_t
            .iter()
            .map(|&o| (o - uniform).abs())
            .fold(0.0f64, f64::max);
        let max_dev_low: f64 = output_low_t
            .iter()
            .map(|&o| (o - uniform).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_dev_high < max_dev_low,
            "high temperature should produce more uniform output: max_dev_high={max_dev_high}, max_dev_low={max_dev_low}"
        );
        // High-T deviation from uniform < 0.01.
        assert!(
            max_dev_high < 0.01,
            "high-T output should be near uniform (max_dev={max_dev_high})"
        );
    }

    /// Single-element input gives output = 1.0.
    #[test]
    fn single_element_gives_one() {
        let input = [42.0];
        let mut output = [0.0f64; 1];
        softplus_softmax_mix(&input, 1.0, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-12,
            "single element must sum to 1"
        );
    }

    /// Larger input has larger output (monotone with respect to input).
    #[test]
    fn larger_input_has_larger_output() {
        let input = [1.0, 3.0, 2.0];
        let mut output = [0.0f64; 3];
        softplus_softmax_mix(&input, 1.0, &mut output);
        assert!(
            output[1] > output[2] && output[2] > output[0],
            "ordering should be preserved: output={output:?}"
        );
    }

    /// Negative temperature panics in debug mode.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "temperature must be positive")]
    fn negative_temperature_panics() {
        let input = [1.0, 2.0];
        let mut output = [0.0f64; 2];
        softplus_softmax_mix(&input, -1.0, &mut output);
    }

    /// softplus is always positive (used internally; test the primitive).
    #[test]
    fn softplus_is_positive_on_all_inputs() {
        for &x in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let sp = math::softplus(x);
            assert!(sp > 0.0, "softplus({x}) = {sp} must be > 0");
        }
    }
}
