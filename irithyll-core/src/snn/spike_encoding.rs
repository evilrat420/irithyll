//! Delta spike encoding for converting continuous features to spike trains.
//!
//! The delta encoder compares each feature's current value against its previous
//! value and emits UP/DOWN spikes when the absolute change exceeds a threshold.
//! Each raw feature maps to two encoding channels (positive and negative change),
//! so N raw features produce 2*N spike channels.
//!
//! # Encoding Rule
//!
//! ```text
//! spike_pos[i] = 1 if (x[i] - x_prev[i]) >  threshold[i]
//! spike_neg[i] = 1 if (x[i] - x_prev[i]) < -threshold[i]
//! ```
//!
//! The output buffer has layout: `[pos_0, neg_0, pos_1, neg_1, ..., pos_{N-1}, neg_{N-1}]`
//! where even indices are positive (UP) spikes and odd indices are negative (DOWN) spikes.

use alloc::vec;
use alloc::vec::Vec;

/// Delta spike encoder for fixed-point inputs.
///
/// Converts continuous-valued features (Q1.14 i16) into binary spike trains
/// by detecting temporal changes exceeding per-feature thresholds.
///
/// # Layout
///
/// For `n_features` input features, the output has `2 * n_features` channels:
/// - Channel `2*i` = positive spike (feature increased)
/// - Channel `2*i + 1` = negative spike (feature decreased)
pub struct DeltaEncoderFixed {
    /// Previous input values for delta computation.
    prev: Vec<i16>,
    /// Per-feature threshold for spike emission.
    threshold: Vec<i16>,
    /// Whether we have seen at least one input (first input produces no spikes).
    initialized: bool,
}

impl DeltaEncoderFixed {
    /// Create a new delta encoder with a uniform threshold for all features.
    ///
    /// # Arguments
    ///
    /// * `n_features` -- number of raw input features
    /// * `threshold` -- uniform spike threshold in Q1.14
    pub fn new(n_features: usize, threshold: i16) -> Self {
        Self {
            prev: vec![0; n_features],
            threshold: vec![threshold; n_features],
            initialized: false,
        }
    }

    /// Create a new delta encoder with per-feature thresholds.
    ///
    /// # Arguments
    ///
    /// * `thresholds` -- one threshold per input feature, in Q1.14
    pub fn new_per_feature(thresholds: Vec<i16>) -> Self {
        let n = thresholds.len();
        Self {
            prev: vec![0; n],
            threshold: thresholds,
            initialized: false,
        }
    }

    /// Number of raw input features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.prev.len()
    }

    /// Number of output spike channels (always `2 * n_features`).
    #[inline]
    pub fn n_output_channels(&self) -> usize {
        self.prev.len() * 2
    }

    /// Encode an input vector into a spike buffer.
    ///
    /// The `out_spikes` buffer must have length `>= 2 * n_features`. It will be
    /// filled with 0s and 1s: even indices for positive spikes, odd for negative.
    ///
    /// On the very first call, no spikes are emitted (the encoder needs a
    /// previous value to compute deltas). The previous values are stored for
    /// the next call.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != n_features` or `out_spikes.len() < 2 * n_features`.
    pub fn encode(&mut self, input: &[i16], out_spikes: &mut [u8]) {
        let n = self.prev.len();
        assert_eq!(
            input.len(),
            n,
            "input length {} does not match encoder n_features {}",
            input.len(),
            n
        );
        assert!(
            out_spikes.len() >= 2 * n,
            "out_spikes length {} too small for {} channels",
            out_spikes.len(),
            2 * n
        );

        if !self.initialized {
            // First call: store input, emit no spikes
            self.prev.copy_from_slice(input);
            for spike in out_spikes[..2 * n].iter_mut() {
                *spike = 0;
            }
            self.initialized = true;
            return;
        }

        for i in 0..n {
            let delta = input[i] as i32 - self.prev[i] as i32;
            let thr = self.threshold[i] as i32;

            // Positive spike: feature increased past threshold
            out_spikes[2 * i] = if delta > thr { 1 } else { 0 };
            // Negative spike: feature decreased past threshold
            out_spikes[2 * i + 1] = if delta < -thr { 1 } else { 0 };
        }

        // Update previous values
        self.prev.copy_from_slice(input);
    }

    /// Reset the encoder to its initial state.
    ///
    /// The next call to `encode` will produce no spikes (warmup step).
    pub fn reset(&mut self) {
        for v in self.prev.iter_mut() {
            *v = 0;
        }
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_produces_no_spikes() {
        let mut enc = DeltaEncoderFixed::new(3, 100);
        let input = [500_i16, -200, 1000];
        let mut spikes = vec![0u8; 6];
        enc.encode(&input, &mut spikes);
        assert!(
            spikes.iter().all(|&s| s == 0),
            "first call should produce no spikes"
        );
    }

    #[test]
    fn positive_spike_on_increase() {
        let mut enc = DeltaEncoderFixed::new(2, 100);
        let mut spikes = vec![0u8; 4];

        // First call: baseline
        enc.encode(&[0, 0], &mut spikes);

        // Second call: feature 0 increases by 200 (> threshold 100)
        enc.encode(&[200, 0], &mut spikes);
        assert_eq!(spikes[0], 1, "feature 0 should have positive spike");
        assert_eq!(spikes[1], 0, "feature 0 should not have negative spike");
        assert_eq!(spikes[2], 0, "feature 1 should have no positive spike");
        assert_eq!(spikes[3], 0, "feature 1 should have no negative spike");
    }

    #[test]
    fn negative_spike_on_decrease() {
        let mut enc = DeltaEncoderFixed::new(2, 100);
        let mut spikes = vec![0u8; 4];

        enc.encode(&[500, 500], &mut spikes);
        enc.encode(&[500, 200], &mut spikes); // feature 1 decreases by 300

        assert_eq!(spikes[0], 0, "feature 0 pos should be 0");
        assert_eq!(spikes[1], 0, "feature 0 neg should be 0");
        assert_eq!(spikes[2], 0, "feature 1 pos should be 0");
        assert_eq!(spikes[3], 1, "feature 1 should have negative spike");
    }

    #[test]
    fn no_spike_within_threshold() {
        let mut enc = DeltaEncoderFixed::new(1, 500);
        let mut spikes = vec![0u8; 2];

        enc.encode(&[1000], &mut spikes);
        // Change of 100 is within threshold of 500
        enc.encode(&[1100], &mut spikes);

        assert_eq!(spikes[0], 0, "should not spike for small increase");
        assert_eq!(spikes[1], 0, "should not spike for small change");
    }

    #[test]
    fn both_spikes_in_same_step() {
        let mut enc = DeltaEncoderFixed::new(2, 100);
        let mut spikes = vec![0u8; 4];

        enc.encode(&[0, 1000], &mut spikes);
        // Feature 0 increases, feature 1 decreases
        enc.encode(&[500, 500], &mut spikes);

        assert_eq!(spikes[0], 1, "feature 0 should have positive spike");
        assert_eq!(spikes[1], 0, "feature 0 should not have negative spike");
        assert_eq!(spikes[2], 0, "feature 1 should not have positive spike");
        assert_eq!(spikes[3], 1, "feature 1 should have negative spike");
    }

    #[test]
    fn per_feature_thresholds() {
        let thresholds = vec![100_i16, 1000];
        let mut enc = DeltaEncoderFixed::new_per_feature(thresholds);
        let mut spikes = vec![0u8; 4];

        enc.encode(&[0, 0], &mut spikes);
        // Feature 0: change 200 > threshold 100 -> spike
        // Feature 1: change 200 < threshold 1000 -> no spike
        enc.encode(&[200, 200], &mut spikes);

        assert_eq!(spikes[0], 1, "feature 0 should spike (200 > 100)");
        assert_eq!(spikes[2], 0, "feature 1 should not spike (200 < 1000)");
    }

    #[test]
    fn reset_clears_state() {
        let mut enc = DeltaEncoderFixed::new(2, 100);
        let mut spikes = vec![0u8; 4];

        enc.encode(&[1000, 2000], &mut spikes);
        enc.encode(&[2000, 3000], &mut spikes);
        assert!(
            spikes.iter().any(|&s| s == 1),
            "should have spikes before reset"
        );

        enc.reset();
        // After reset, first call should produce no spikes
        enc.encode(&[5000, 5000], &mut spikes);
        assert!(
            spikes.iter().all(|&s| s == 0),
            "after reset, first call should produce no spikes"
        );
    }

    #[test]
    fn sequential_encoding_uses_updated_prev() {
        let mut enc = DeltaEncoderFixed::new(1, 100);
        let mut spikes = vec![0u8; 2];

        enc.encode(&[0], &mut spikes);
        enc.encode(&[500], &mut spikes); // delta = 500
        assert_eq!(spikes[0], 1, "first increase should spike");

        enc.encode(&[600], &mut spikes); // delta = 100, NOT > threshold
        assert_eq!(spikes[0], 0, "small subsequent change should not spike");

        enc.encode(&[1200], &mut spikes); // delta = 600 from 600
        assert_eq!(spikes[0], 1, "large subsequent change should spike");
    }

    #[test]
    fn output_channel_count() {
        let enc = DeltaEncoderFixed::new(5, 100);
        assert_eq!(enc.n_features(), 5);
        assert_eq!(enc.n_output_channels(), 10);
    }
}
