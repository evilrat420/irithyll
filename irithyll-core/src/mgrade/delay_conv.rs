//! Learnable delay convolution -- 1D depthwise conv with learnable delay spacings.
//!
//! For each channel `d`:
//!
//! ```text
//! y[d] = sum_k( weight[d,k] * buffer[d, round(delay[d,k])] )
//! ```
//!
//! Where:
//! - `buffer` is a circular buffer of recent inputs (per channel)
//! - `delay[d,k]` are learnable real-valued spacings (initialized as 0, 1, ..., K-1)
//! - `weight[d,k]` are learnable convolution weights
//! - K is the kernel size (number of delay taps)
//!
//! Delays are real-valued; the current implementation rounds to the nearest
//! integer index for buffer lookup. Buffer wrapping uses modular arithmetic.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::rng::standard_normal;

/// 1D depthwise delay convolution with learnable delay spacings.
///
/// # Example
///
/// ```
/// use irithyll_core::mgrade::DelayConv1D;
///
/// let mut conv = DelayConv1D::new(3, 4, 42);
/// let input = [0.1, -0.2, 0.3];
/// let output = conv.forward(&input);
/// assert_eq!(output.len(), 3);
/// ```
pub struct DelayConv1D {
    /// Convolution weights: [d_in x kernel_size] row-major.
    weights: Vec<f64>,
    /// Learnable delays: [d_in x kernel_size], initialized as [0.0, 1.0, ..., K-1].
    delays: Vec<f64>,
    /// Circular buffer: [d_in x buffer_len] row-major.
    buffer: Vec<f64>,
    /// Write position in the circular buffer.
    buf_pos: usize,
    /// Number of input channels.
    d_in: usize,
    /// Number of delay taps per channel.
    kernel_size: usize,
    /// Length of the circular buffer per channel.
    buffer_len: usize,
}

impl DelayConv1D {
    /// Create a new delay convolution.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- number of input channels (depthwise)
    /// * `kernel_size` -- number of delay taps per channel
    /// * `seed` -- RNG seed for deterministic weight initialization
    pub fn new(d_in: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = seed;

        // Buffer length: accommodate max initial delay + margin
        let buffer_len = 2 * kernel_size.max(1);

        // Initialize weights from normal distribution, scale by 1/sqrt(K)
        let scale = 1.0 / math::sqrt(kernel_size as f64);
        let n_weights = d_in * kernel_size;
        let weights: Vec<f64> = (0..n_weights)
            .map(|_| standard_normal(&mut rng) * scale)
            .collect();

        // Initialize delays as [0.0, 1.0, ..., K-1] for each channel
        let delays: Vec<f64> = (0..d_in)
            .flat_map(|_| (0..kernel_size).map(|k| k as f64))
            .collect();

        let buffer = vec![0.0; d_in * buffer_len];

        Self {
            weights,
            delays,
            buffer,
            buf_pos: 0,
            d_in,
            kernel_size,
            buffer_len,
        }
    }

    /// Process one input timestep: write to circular buffer and compute
    /// the convolution output.
    ///
    /// # Arguments
    ///
    /// * `input` -- input vector of length `d_in`
    ///
    /// # Returns
    ///
    /// Convolution output of length `d_in`.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        // Write input to circular buffer at current position
        for (d, &val) in input.iter().enumerate().take(self.d_in) {
            self.buffer[d * self.buffer_len + self.buf_pos] = val;
        }

        // Compute output
        let output = self.compute_output();

        // Advance write position
        self.buf_pos = (self.buf_pos + 1) % self.buffer_len;

        output
    }

    /// Compute what the output would be with `input`, without mutating state.
    ///
    /// Writes `input` into a temporary copy of the buffer at the current
    /// write position, computes the convolution, then discards the copy.
    pub fn forward_predict(&self, input: &[f64]) -> Vec<f64> {
        // Temporary buffer with input written at current position
        let mut buf_copy = self.buffer.clone();
        for (d, &val) in input.iter().enumerate().take(self.d_in) {
            buf_copy[d * self.buffer_len + self.buf_pos] = val;
        }

        // Compute output from temporary buffer
        let mut output = vec![0.0; self.d_in];
        for (d, out_d) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..self.kernel_size {
                let delay = self.delays[d * self.kernel_size + k];
                let delay_int = crate::math::round(delay) as isize;
                let idx = ((self.buf_pos as isize - delay_int).rem_euclid(self.buffer_len as isize))
                    as usize;
                let w = self.weights[d * self.kernel_size + k];
                sum += w * buf_copy[d * self.buffer_len + idx];
            }
            *out_d = sum;
        }

        output
    }

    /// Reset the circular buffer to zeros, preserving weights and delays.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buf_pos = 0;
    }

    /// Number of input/output channels.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Number of delay taps per channel.
    #[inline]
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Internal helper: compute convolution output from current buffer state.
    fn compute_output(&self) -> Vec<f64> {
        let mut output = vec![0.0; self.d_in];
        for (d, out_d) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..self.kernel_size {
                let delay = self.delays[d * self.kernel_size + k];
                let delay_int = crate::math::round(delay) as isize;
                let idx = ((self.buf_pos as isize - delay_int).rem_euclid(self.buffer_len as isize))
                    as usize;
                let w = self.weights[d * self.kernel_size + k];
                sum += w * self.buffer[d * self.buffer_len + idx];
            }
            *out_d = sum;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_conv_new() {
        let conv = DelayConv1D::new(3, 4, 42);
        assert_eq!(conv.d_in(), 3, "d_in should match constructor arg");
        assert_eq!(
            conv.kernel_size(),
            4,
            "kernel_size should match constructor arg"
        );
        assert_eq!(
            conv.weights.len(),
            3 * 4,
            "weights should have d_in * kernel_size elements"
        );
        assert_eq!(
            conv.delays.len(),
            3 * 4,
            "delays should have d_in * kernel_size elements"
        );
    }

    #[test]
    fn delay_conv_delays_initialized_correctly() {
        let conv = DelayConv1D::new(2, 4, 42);
        // Each channel should have delays [0.0, 1.0, 2.0, 3.0]
        for d in 0..2 {
            for k in 0..4 {
                let expected = k as f64;
                let actual = conv.delays[d * 4 + k];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "delay[{d},{k}] should be {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn delay_conv_forward_output_length() {
        let mut conv = DelayConv1D::new(5, 3, 42);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let output = conv.forward(&input);
        assert_eq!(output.len(), 5, "output should have d_in elements");
    }

    #[test]
    fn delay_conv_forward_finite() {
        let mut conv = DelayConv1D::new(3, 4, 123);
        let input = [1.0, -0.5, 2.0];
        for _ in 0..10 {
            let output = conv.forward(&input);
            for (i, &val) in output.iter().enumerate() {
                assert!(val.is_finite(), "output[{}] = {} should be finite", i, val);
            }
        }
    }

    #[test]
    fn delay_conv_forward_predict_no_state_change() {
        let mut conv = DelayConv1D::new(3, 4, 42);
        let input = [1.0, 2.0, 3.0];
        conv.forward(&input);

        let buf_before = conv.buffer.clone();
        let pos_before = conv.buf_pos;

        let _pred = conv.forward_predict(&[0.5, -0.5, 1.5]);

        assert_eq!(
            conv.buffer, buf_before,
            "buffer should not change after forward_predict"
        );
        assert_eq!(
            conv.buf_pos, pos_before,
            "buf_pos should not change after forward_predict"
        );
    }

    #[test]
    fn delay_conv_reset() {
        let mut conv = DelayConv1D::new(3, 4, 42);
        for i in 0..10 {
            conv.forward(&[i as f64, (i as f64) * 0.5, -(i as f64)]);
        }

        let weights_before = conv.weights.clone();
        let delays_before = conv.delays.clone();

        conv.reset();

        assert!(
            conv.buffer.iter().all(|&v| v == 0.0),
            "buffer should be all zeros after reset"
        );
        assert_eq!(conv.buf_pos, 0, "buf_pos should be 0 after reset");
        assert_eq!(
            conv.weights, weights_before,
            "weights should be preserved after reset"
        );
        assert_eq!(
            conv.delays, delays_before,
            "delays should be preserved after reset"
        );
    }

    #[test]
    fn delay_conv_circular_buffer_wraps() {
        let mut conv = DelayConv1D::new(1, 2, 42);
        // buffer_len = 2 * max(2, 1) = 4

        // Fill buffer more than buffer_len times to test wrapping
        for i in 0..10 {
            conv.forward(&[i as f64]);
        }
        // Should not panic -- wrapping works
        let output = conv.forward(&[10.0]);
        assert!(
            output[0].is_finite(),
            "output should be finite after buffer wraps"
        );
    }

    #[test]
    fn delay_conv_forward_predict_matches_forward() {
        let mut conv = DelayConv1D::new(3, 4, 42);
        let x1 = [1.0, 2.0, 3.0];
        conv.forward(&x1);

        let x2 = [0.5, -0.5, 1.5];
        let pred = conv.forward_predict(&x2);
        let actual = conv.forward(&x2);

        for (i, (p, a)) in pred.iter().zip(actual.iter()).enumerate() {
            assert!(
                (p - a).abs() < 1e-12,
                "forward_predict[{i}]={p} should match forward[{i}]={a}"
            );
        }
    }
}
