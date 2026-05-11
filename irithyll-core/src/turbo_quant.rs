//! TurboQuant multi-mode weight quantization with randomized Hadamard rotation.
//!
//! Compresses weight vectors using one of three quantization modes:
//!
//! | Mode | Levels | Packing | Compression vs f64 | Typical error |
//! |------|--------|---------|--------------------|---------------|
//! | 8-bit | 256 | 4 per u32 | ~8x | ~0.4% |
//! | 3.5-bit | 11 | 7 per u32 (base-11) | ~14x | ~10% |
//! | 2.5-bit | 5 | 13 per u32 (base-5) | ~21x | ~20% |
//!
//! # Design
//!
//! - **Data-oblivious**: No calibration set required -- randomized rotation + min/max scaling
//! - **Online-compatible**: Quantize once after training, inference is pure integer
//! - **Embedded-friendly**: [`TurboQuantizedView`] is zero-copy from `&[u8]`
//! - **Zero-alloc predict**: [`predict_with_scratch`](TurboQuantized::predict_with_scratch)
//!   avoids allocation when given a caller-provided scratch buffer
//! - **Multi-type input**: [`quantize_f32`] and [`quantize_i16`] accept non-f64 weights
//! - **Hadamard rotation**: Applies `H * D * w` before quantization where `D` is a
//!   random sign-flip diagonal and `H` is the normalized Walsh-Hadamard matrix.
//!   This decorrelates weight dimensions so quantization error distributes uniformly.
//!
//! # Packing
//!
//! ```text
//! 8-bit:   4 values x 256 levels = byte packing, 4 per u32 (shift encoding)
//! 3.5-bit: 7 values x  11 levels = 11^7 = 19,487,171 states <= 2^25 (base-11)
//! 2.5-bit: 13 values x  5 levels = 5^13 = 1,220,703,125 states <= 2^31 (base-5)
//! ```
//!
//! # References
//!
//! Inspired by data-oblivious quantization (Google/NYU, ICLR 2026).

use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// QuantMode
// ---------------------------------------------------------------------------

/// Quantization bit depth. Controls the quality/compression tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantMode {
    /// 8-bit: 256 levels, 4 values per u32. Near-lossless (~0.4% max error).
    /// ~8x compression vs f64. Simple byte packing.
    Bits8,
    /// 3.5-bit: 11 levels, 7 values per u32. Aggressive (~10% max error).
    /// ~14x compression vs f64. Base-11 mixed-radix packing.
    Bits3_5,
    /// 2.5-bit: 5 levels, 13 values per u32. Ultra-aggressive (~20% max error).
    /// ~21x compression vs f64. Base-5 mixed-radix packing.
    Bits2_5,
}

impl QuantMode {
    /// Number of quantization levels for this mode.
    #[inline]
    fn n_levels(self) -> u32 {
        match self {
            QuantMode::Bits8 => N_LEVELS_8,
            QuantMode::Bits3_5 => N_LEVELS_3_5,
            QuantMode::Bits2_5 => N_LEVELS_2_5,
        }
    }

    /// Number of values packed per u32 word for this mode.
    #[inline]
    fn values_per_word(self) -> usize {
        match self {
            QuantMode::Bits8 => VALUES_PER_WORD_8,
            QuantMode::Bits3_5 => VALUES_PER_WORD_3_5,
            QuantMode::Bits2_5 => VALUES_PER_WORD_2_5,
        }
    }

    /// Encode mode as u32 for serialization.
    #[inline]
    fn to_u32(self) -> u32 {
        match self {
            QuantMode::Bits8 => 0,
            QuantMode::Bits3_5 => 1,
            QuantMode::Bits2_5 => 2,
        }
    }

    /// Decode mode from u32. Returns `None` for unknown values.
    #[inline]
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(QuantMode::Bits8),
            1 => Some(QuantMode::Bits3_5),
            2 => Some(QuantMode::Bits2_5),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 8-bit: 256 levels (0..=255).
const N_LEVELS_8: u32 = 256;
/// 8-bit: 4 values per u32.
const VALUES_PER_WORD_8: usize = 4;

/// 3.5-bit: 11 levels (0..=10).
const N_LEVELS_3_5: u32 = 11;
/// 3.5-bit: 7 values per u32.
const VALUES_PER_WORD_3_5: usize = 7;

/// 2.5-bit: 5 levels (0..=4).
const N_LEVELS_2_5: u32 = 5;
/// 2.5-bit: 13 values per u32.
const VALUES_PER_WORD_2_5: usize = 13;

// ---------------------------------------------------------------------------
// Packing: 8-bit (4 values per u32, byte shift encoding)
// ---------------------------------------------------------------------------

/// Pack up to 4 quantized u8 values into a single `u32` using byte shift encoding.
#[inline]
fn pack4_bytes(values: &[u8]) -> u32 {
    let mut packed: u32 = 0;
    for (i, &v) in values.iter().enumerate().take(4) {
        packed |= (v as u32) << (i * 8);
    }
    packed
}

/// Unpack a `u32` into up to 4 quantized u8 values.
#[inline]
fn unpack4_bytes(packed: u32, count: usize) -> [u8; 4] {
    let mut values = [0u8; 4];
    for (i, v) in values.iter_mut().enumerate().take(count) {
        *v = ((packed >> (i * 8)) & 0xFF) as u8;
    }
    values
}

// ---------------------------------------------------------------------------
// Packing: 3.5-bit (7 values per u32, base-11 mixed-radix)
// ---------------------------------------------------------------------------

/// Pack up to 7 quantized values (each in 0..=10) into a single `u32`.
///
/// Uses base-11 mixed-radix encoding: `v0 + 11*v1 + 11^2*v2 + ... + 11^6*v6`.
/// The maximum packed value is `11^7 - 1 = 19,487,170`, which fits in 25 bits.
///
/// `values` must have length <= 7, and each element must be in `0..=10`.
#[inline]
fn pack7(values: &[u8]) -> u32 {
    debug_assert!(values.len() <= 7);
    let mut packed: u32 = 0;
    for &v in values.iter().rev() {
        debug_assert!(v < N_LEVELS_3_5 as u8);
        packed = packed * N_LEVELS_3_5 + v as u32;
    }
    packed
}

/// Unpack a `u32` into up to 7 quantized values.
///
/// Extracts `count` values from the base-11 mixed-radix encoding.
/// Remaining slots in the returned array are zero-filled.
#[inline]
fn unpack7(packed: u32, count: usize) -> [u8; 7] {
    let mut values = [0u8; 7];
    let mut p = packed;
    for v in values.iter_mut().take(count) {
        *v = (p % N_LEVELS_3_5) as u8;
        p /= N_LEVELS_3_5;
    }
    values
}

// ---------------------------------------------------------------------------
// Packing: 2.5-bit (13 values per u32, base-5 mixed-radix)
// ---------------------------------------------------------------------------

/// Pack up to 13 quantized values (each in 0..=4) into a single `u32`.
///
/// Uses base-5 mixed-radix encoding. The maximum packed value is
/// `5^13 - 1 = 1,220,703,124`, which fits in 31 bits.
#[inline]
fn pack13(values: &[u8]) -> u32 {
    debug_assert!(values.len() <= 13);
    let mut packed: u32 = 0;
    for &v in values.iter().rev() {
        debug_assert!(v < N_LEVELS_2_5 as u8);
        packed = packed * N_LEVELS_2_5 + v as u32;
    }
    packed
}

/// Unpack a `u32` into up to 13 quantized values.
///
/// Extracts `count` values from the base-5 mixed-radix encoding.
/// Remaining slots in the returned array are zero-filled.
#[inline]
fn unpack13(packed: u32, count: usize) -> [u8; 13] {
    let mut values = [0u8; 13];
    let mut p = packed;
    for v in values.iter_mut().take(count) {
        *v = (p % N_LEVELS_2_5) as u8;
        p /= N_LEVELS_2_5;
    }
    values
}

// ---------------------------------------------------------------------------
// Generic mode-dispatched packing
// ---------------------------------------------------------------------------

/// Pack a slice of quantized values into a u32 word, dispatching on mode.
#[inline]
fn pack_word(values: &[u8], mode: QuantMode) -> u32 {
    match mode {
        QuantMode::Bits8 => pack4_bytes(values),
        QuantMode::Bits3_5 => pack7(values),
        QuantMode::Bits2_5 => pack13(values),
    }
}

// ---------------------------------------------------------------------------
// TurboQuantized (owned)
// ---------------------------------------------------------------------------

/// Quantized weight vector (owned).
///
/// Created by [`quantize`], [`quantize_weights`], [`quantize_f32`], or
/// [`quantize_i16`]. Supports inference via [`predict`](Self::predict)
/// and serialization via [`to_bytes`](Self::to_bytes).
///
/// Weights are stored in Hadamard-rotated space (`H * D * w`). During prediction,
/// the same rotation is applied to features so the dot product is preserved.
pub struct TurboQuantized {
    /// Packed u32 words. Format depends on `mode`:
    /// - Bits8: 4 byte-shift-encoded u8 values per u32
    /// - Bits3_5: 7 base-11 mixed-radix values per u32
    /// - Bits2_5: 13 base-5 mixed-radix values per u32
    packed: Vec<u32>,
    /// Number of original weights (last word may be partially filled).
    n_weights: usize,
    /// Scale factor for dequantization. Zero means all weights are identical.
    scale: f64,
    /// Offset: minimum weight value (in rotated space).
    offset: f64,
    /// Seed for the random sign-flip diagonal (needed to reproduce rotation).
    seed: u64,
    /// Power-of-2 padded length used for FWHT.
    padded_len: usize,
    /// Quantization mode.
    mode: QuantMode,
}

impl TurboQuantized {
    /// Dot product of quantized weights with a feature vector.
    ///
    /// Applies the same Hadamard rotation to `features`, then computes
    /// the dot product with the quantized rotated weights over the full
    /// padded length. Since `HD` is orthogonal, `w . x == (HD*w) . (HD*x)`.
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.n_weights == 0 {
            return 0.0;
        }
        // Rotate features with the same transform (pad to padded_len)
        let mut rotated_features = Vec::with_capacity(self.padded_len);
        let use_len = self.n_weights.min(features.len());
        rotated_features.extend_from_slice(&features[..use_len]);
        rotated_features.resize(self.padded_len, 0.0);
        apply_rotation(&mut rotated_features, self.seed);

        self.dot_with_rotated(&rotated_features)
    }

    /// Predict using a caller-provided scratch buffer for the Hadamard rotation.
    ///
    /// `scratch` must have length >= `padded_len`. This avoids allocation,
    /// making it suitable for embedded inference loops.
    pub fn predict_with_scratch(&self, features: &[f64], scratch: &mut [f64]) -> f64 {
        if self.n_weights == 0 {
            return 0.0;
        }
        assert!(
            scratch.len() >= self.padded_len,
            "scratch buffer too small: {} < {}",
            scratch.len(),
            self.padded_len
        );

        // Zero and fill scratch
        for v in scratch[..self.padded_len].iter_mut() {
            *v = 0.0;
        }
        let use_len = self.n_weights.min(features.len());
        scratch[..use_len].copy_from_slice(&features[..use_len]);

        // Rotate in-place
        apply_rotation(&mut scratch[..self.padded_len], self.seed);

        self.dot_with_rotated(&scratch[..self.padded_len])
    }

    /// Compute dot product of packed weights with already-rotated features.
    fn dot_with_rotated(&self, rotated_features: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut feat_idx = 0;
        let vpw = self.mode.values_per_word();

        for &word in self.packed.iter() {
            let remaining = self.padded_len - feat_idx;
            let count = remaining.min(vpw);
            // Inline unpack + dot to avoid temporary array overhead
            match self.mode {
                QuantMode::Bits8 => {
                    let values = unpack4_bytes(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
                QuantMode::Bits3_5 => {
                    let values = unpack7(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
                QuantMode::Bits2_5 => {
                    let values = unpack13(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
            }
            if feat_idx >= self.padded_len {
                break;
            }
        }
        sum
    }

    /// Dequantize all weights back to `f64` (approximate original space).
    ///
    /// Unpacks all `padded_len` rotated values, then applies the inverse
    /// Hadamard rotation to recover approximate original weights.
    pub fn dequantize(&self) -> Vec<f64> {
        let mut rotated = Vec::with_capacity(self.padded_len);
        let mut count_total = 0;
        let vpw = self.mode.values_per_word();

        for &word in self.packed.iter() {
            let remaining = self.padded_len - count_total;
            let count = remaining.min(vpw);
            match self.mode {
                QuantMode::Bits8 => {
                    let values = unpack4_bytes(word, count);
                    for &q in values.iter().take(count) {
                        rotated.push(q as f64 * self.scale + self.offset);
                        count_total += 1;
                    }
                }
                QuantMode::Bits3_5 => {
                    let values = unpack7(word, count);
                    for &q in values.iter().take(count) {
                        rotated.push(q as f64 * self.scale + self.offset);
                        count_total += 1;
                    }
                }
                QuantMode::Bits2_5 => {
                    let values = unpack13(word, count);
                    for &q in values.iter().take(count) {
                        rotated.push(q as f64 * self.scale + self.offset);
                        count_total += 1;
                    }
                }
            }
            if count_total >= self.padded_len {
                break;
            }
        }
        // Apply inverse rotation to recover original space
        apply_inverse_rotation(&mut rotated, self.seed);
        rotated.truncate(self.n_weights);
        rotated
    }

    /// Number of quantized weights.
    pub fn n_weights(&self) -> usize {
        self.n_weights
    }

    /// Power-of-2 padded length used for FWHT (needed for scratch allocation).
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Quantization mode used.
    pub fn mode(&self) -> QuantMode {
        self.mode
    }

    /// Compression ratio vs `f64` (original bytes / packed bytes).
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.n_weights * 8; // f64
        let packed_bytes = self.packed.len() * 4 + HEADER_SIZE;
        original_bytes as f64 / packed_bytes as f64
    }

    /// Serialize to bytes for embedded deployment.
    ///
    /// Format (36-byte header):
    /// ```text
    /// [n_weights: u32 LE]
    /// [mode: u32 LE]        // 0=Bits8, 1=Bits3_5, 2=Bits2_5
    /// [seed: u64 LE]
    /// [padded_len: u32 LE]
    /// [scale: f64 LE]
    /// [offset: f64 LE]
    /// [packed_words: u32 LE...]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE + self.packed.len() * 4);
        buf.extend_from_slice(&(self.n_weights as u32).to_le_bytes());
        buf.extend_from_slice(&self.mode.to_u32().to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());
        buf.extend_from_slice(&(self.padded_len as u32).to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.extend_from_slice(&self.offset.to_le_bytes());
        for &word in &self.packed {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// TurboQuantizedView (zero-copy)
// ---------------------------------------------------------------------------

/// Zero-copy view over a TurboQuant packed binary.
///
/// Constructed from `&[u8]` with no allocation -- suitable for embedded
/// targets where the binary is in flash/ROM. Note: `predict` does allocate
/// for the Hadamard rotation of the feature vector; use
/// [`predict_with_scratch`](Self::predict_with_scratch) for zero-alloc inference.
pub struct TurboQuantizedView<'a> {
    /// Raw bytes of packed u32 words.
    packed: &'a [u8],
    /// Number of original weights.
    n_weights: usize,
    /// Seed for the random sign-flip diagonal.
    seed: u64,
    /// Power-of-2 padded length used for FWHT.
    padded_len: usize,
    /// Scale factor for dequantization.
    scale: f64,
    /// Offset (minimum weight value in rotated space) for dequantization.
    offset: f64,
    /// Quantization mode.
    mode: QuantMode,
}

/// Header size in bytes: n_weights(4) + mode(4) + seed(8) + padded_len(4) + scale(8) + offset(8) = 36.
const HEADER_SIZE: usize = 36;

impl<'a> TurboQuantizedView<'a> {
    /// Parse a TurboQuant binary from raw bytes.
    ///
    /// Returns [`FormatError::Truncated`](crate::error::FormatError::Truncated)
    /// if the buffer is too short, has inconsistent length, or contains an
    /// unknown quantization mode.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, crate::error::FormatError> {
        if bytes.len() < HEADER_SIZE {
            return Err(crate::error::FormatError::Truncated);
        }
        let n_weights = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mode_raw = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let mode = QuantMode::from_u32(mode_raw).ok_or(crate::error::FormatError::Truncated)?;
        let seed = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let padded_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let scale = f64::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
        ]);
        let offset = f64::from_le_bytes([
            bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35],
        ]);

        // Packed data holds padded_len values
        let vpw = mode.values_per_word();
        let n_words = padded_len.div_ceil(vpw);
        let expected_len = HEADER_SIZE + n_words * 4;
        if bytes.len() < expected_len {
            return Err(crate::error::FormatError::Truncated);
        }

        Ok(Self {
            packed: &bytes[HEADER_SIZE..HEADER_SIZE + n_words * 4],
            n_weights,
            seed,
            padded_len,
            scale,
            offset,
            mode,
        })
    }

    /// Dot product of quantized weights with a feature vector.
    ///
    /// Applies the same Hadamard rotation to `features`, then computes
    /// the dot product with the quantized rotated weights over the full
    /// padded length.
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.n_weights == 0 {
            return 0.0;
        }
        // Rotate features with the same transform (pad to padded_len)
        let mut rotated_features = Vec::with_capacity(self.padded_len);
        let use_len = self.n_weights.min(features.len());
        rotated_features.extend_from_slice(&features[..use_len]);
        rotated_features.resize(self.padded_len, 0.0);
        apply_rotation(&mut rotated_features, self.seed);

        self.dot_with_rotated(&rotated_features)
    }

    /// Predict using a caller-provided scratch buffer for the Hadamard rotation.
    ///
    /// `scratch` must have length >= `padded_len`. This avoids allocation,
    /// making it suitable for embedded inference loops.
    pub fn predict_with_scratch(&self, features: &[f64], scratch: &mut [f64]) -> f64 {
        if self.n_weights == 0 {
            return 0.0;
        }
        assert!(
            scratch.len() >= self.padded_len,
            "scratch buffer too small: {} < {}",
            scratch.len(),
            self.padded_len
        );

        for v in scratch[..self.padded_len].iter_mut() {
            *v = 0.0;
        }
        let use_len = self.n_weights.min(features.len());
        scratch[..use_len].copy_from_slice(&features[..use_len]);
        apply_rotation(&mut scratch[..self.padded_len], self.seed);

        self.dot_with_rotated(&scratch[..self.padded_len])
    }

    /// Compute dot product of packed weights with already-rotated features.
    fn dot_with_rotated(&self, rotated_features: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut feat_idx = 0;
        let vpw = self.mode.values_per_word();
        let n_words = self.packed.len() / 4;

        for word_idx in 0..n_words {
            let off = word_idx * 4;
            let word = u32::from_le_bytes([
                self.packed[off],
                self.packed[off + 1],
                self.packed[off + 2],
                self.packed[off + 3],
            ]);
            let remaining = self.padded_len - feat_idx;
            let count = remaining.min(vpw);
            match self.mode {
                QuantMode::Bits8 => {
                    let values = unpack4_bytes(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
                QuantMode::Bits3_5 => {
                    let values = unpack7(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
                QuantMode::Bits2_5 => {
                    let values = unpack13(word, count);
                    for &q in values.iter().take(count) {
                        let w = q as f64 * self.scale + self.offset;
                        sum += w * rotated_features[feat_idx];
                        feat_idx += 1;
                    }
                }
            }
            if feat_idx >= self.padded_len {
                break;
            }
        }
        sum
    }

    /// Number of weights in this view.
    pub fn n_weights(&self) -> usize {
        self.n_weights
    }

    /// Power-of-2 padded length used for FWHT.
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Quantization mode.
    pub fn mode(&self) -> QuantMode {
        self.mode
    }
}

// ---------------------------------------------------------------------------
// Hadamard rotation internals
// ---------------------------------------------------------------------------

/// Default deterministic seed for Hadamard rotation.
const DEFAULT_SEED: u64 = 0xDEAD_BEEF;

/// Smallest power of 2 >= `n`. Returns 1 for `n == 0`.
#[inline]
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    // Bit trick: round up to next power of 2
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    #[cfg(target_pointer_width = "64")]
    {
        v |= v >> 32;
    }
    v + 1
}

/// In-place Fast Walsh-Hadamard Transform (normalized).
///
/// `x` must have power-of-2 length. After transform, `H` is orthogonal:
/// applying FWHT twice recovers the original vector (self-inverse).
fn fwht_inplace(x: &mut [f64]) {
    let n = x.len();
    debug_assert!(
        n > 0 && (n & (n - 1)) == 0,
        "FWHT requires power-of-2 length"
    );
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / crate::math::sqrt(n as f64);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Apply random sign flips (diagonal D matrix) to `x`.
///
/// `D` is self-inverse: applying the same sign flips twice recovers the original.
fn apply_sign_flip(x: &mut [f64], seed: u64) {
    let mut state = seed;
    for v in x.iter_mut() {
        let r = crate::rng::xorshift64(&mut state);
        if r & 1 == 0 {
            *v = -*v;
        }
    }
}

/// Apply the full Hadamard rotation `H * D * x`.
fn apply_rotation(buf: &mut [f64], seed: u64) {
    apply_sign_flip(buf, seed);
    fwht_inplace(buf);
}

/// Apply the inverse Hadamard rotation `D * H * x` to recover original space.
fn apply_inverse_rotation(buf: &mut [f64], seed: u64) {
    fwht_inplace(buf);
    apply_sign_flip(buf, seed);
}

// ---------------------------------------------------------------------------
// Public quantization API
// ---------------------------------------------------------------------------

/// Quantize a weight vector to 3.5-bit TurboQuant format.
///
/// Applies a randomized Hadamard rotation before quantization to decorrelate
/// weight dimensions, then compresses using an 11-level linear grid with
/// min/max scaling. Uses a deterministic default seed.
///
/// This is a backwards-compatible wrapper around [`quantize`] with
/// [`QuantMode::Bits3_5`] and the default seed.
///
/// # Example
///
/// ```
/// use irithyll_core::turbo_quant::quantize_weights;
///
/// let weights = vec![0.1, -0.5, 0.3, 0.0, -0.2, 0.4, 0.1, -0.1, 0.2];
/// let quantized = quantize_weights(&weights);
/// assert_eq!(quantized.n_weights(), 9);
///
/// // Predict with on-the-fly dequantization
/// let features = vec![1.0; 9];
/// let pred = quantized.predict(&features);
/// assert!(pred.is_finite());
///
/// // Roundtrip check
/// let original_dot: f64 = weights.iter().zip(features.iter()).map(|(w, f)| w * f).sum();
/// assert!((pred - original_dot).abs() < 0.5, "quantization error should be small");
/// ```
pub fn quantize_weights(weights: &[f64]) -> TurboQuantized {
    quantize(weights, QuantMode::Bits3_5, DEFAULT_SEED)
}

/// Quantize with an explicit seed for the Hadamard rotation (3.5-bit mode).
///
/// Backwards-compatible wrapper around [`quantize`].
pub fn quantize_weights_with_seed(weights: &[f64], seed: u64) -> TurboQuantized {
    quantize(weights, QuantMode::Bits3_5, seed)
}

/// Quantize a weight vector with explicit mode and seed.
///
/// Applies a randomized Hadamard rotation before quantization to decorrelate
/// weight dimensions, then compresses using a linear grid with min/max
/// scaling at the specified bit depth.
pub fn quantize(weights: &[f64], mode: QuantMode, seed: u64) -> TurboQuantized {
    if weights.is_empty() {
        return TurboQuantized {
            packed: vec![],
            n_weights: 0,
            scale: 0.0,
            offset: 0.0,
            seed,
            padded_len: 1,
            mode,
        };
    }

    // Apply Hadamard rotation: pad to power of 2, sign flip, FWHT
    let padded_len = next_power_of_two(weights.len());
    let mut rotated = Vec::with_capacity(padded_len);
    rotated.extend_from_slice(weights);
    rotated.resize(padded_len, 0.0);
    apply_rotation(&mut rotated, seed);

    // Quantize ALL padded_len rotated values (rotation spreads information
    // across the full padded vector, so truncating would lose data).
    let min_val = rotated.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = rotated.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let n_levels = mode.n_levels();
    let max_level = n_levels - 1;
    let scale = if range < 1e-15 {
        0.0
    } else {
        range / max_level as f64
    };

    // Quantize each rotated weight to [0, max_level]
    let quantized: Vec<u8> = rotated
        .iter()
        .map(|&w| {
            if scale < 1e-15 {
                (max_level / 2) as u8 // constant weights -> mid-grid
            } else {
                let q = crate::math::round((w - min_val) / scale);
                (q as u8).min(max_level as u8)
            }
        })
        .collect();

    // Pack into u32 words (all padded_len values)
    let vpw = mode.values_per_word();
    let n_words = padded_len.div_ceil(vpw);
    let mut packed = Vec::with_capacity(n_words);
    for chunk in quantized.chunks(vpw) {
        packed.push(pack_word(chunk, mode));
    }

    TurboQuantized {
        packed,
        n_weights: weights.len(),
        scale,
        offset: min_val,
        seed,
        padded_len,
        mode,
    }
}

/// Quantize f32 weights with explicit mode. Uses the default seed.
pub fn quantize_f32(weights: &[f32], mode: QuantMode) -> TurboQuantized {
    let f64_weights: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    quantize(&f64_weights, mode, DEFAULT_SEED)
}

/// Quantize i16 weights with a dequantization scale and explicit mode.
///
/// Each i16 value is converted to `f64` via `value as f64 * scale` before
/// quantization. Uses the default seed.
pub fn quantize_i16(weights: &[i16], scale: f64, mode: QuantMode) -> TurboQuantized {
    let f64_weights: Vec<f64> = weights.iter().map(|&w| w as f64 * scale).collect();
    quantize(&f64_weights, mode, DEFAULT_SEED)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Original tests (updated for new internals) ----

    #[test]
    fn pack_unpack_roundtrip() {
        let values = [0u8, 5, 10, 3, 7, 1, 9];
        let packed = pack7(&values);
        let unpacked = unpack7(packed, 7);
        assert_eq!(&unpacked, &values, "pack/unpack roundtrip failed");
    }

    #[test]
    fn pack_unpack_partial() {
        let values = [2u8, 8, 4];
        let packed = pack7(&values);
        let unpacked = unpack7(packed, 3);
        assert_eq!(&unpacked[..3], &values, "partial pack/unpack failed");
    }

    #[test]
    fn quantize_empty() {
        let q = quantize_weights(&[]);
        assert_eq!(q.n_weights(), 0);
        assert_eq!(q.predict(&[]), 0.0);
    }

    #[test]
    fn quantize_single_weight() {
        let q = quantize_weights(&[3.125]);
        assert_eq!(q.n_weights(), 1);
        let pred = q.predict(&[1.0]);
        assert!(
            (pred - 3.125).abs() < 0.5,
            "single weight should roundtrip reasonably, got {pred}"
        );
    }

    #[test]
    fn quantize_constant_weights() {
        let q = quantize_weights(&[2.5, 2.5, 2.5, 2.5]);
        let dq = q.dequantize();
        for (i, &w) in dq.iter().enumerate() {
            assert!(
                (w - 2.5).abs() < 0.05,
                "constant weights should dequantize closely, got {w} at [{i}]"
            );
        }
    }

    #[test]
    fn quantize_predict_accuracy() {
        let weights = vec![0.1, -0.5, 0.3, 0.0, -0.2, 0.4, 0.1, -0.1, 0.2];
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let exact: f64 = weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();
        let q = quantize_weights(&weights);
        let pred = q.predict(&features);
        let rel_err = if exact.abs() > 1e-10 {
            (pred - exact).abs() / exact.abs()
        } else {
            (pred - exact).abs()
        };
        assert!(
            rel_err < 0.25,
            "relative error should be < 25%, got {rel_err:.4} (exact={exact:.4}, pred={pred:.4})"
        );
    }

    #[test]
    fn quantize_dequantize_bounded_error() {
        let weights: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
        let q = quantize_weights(&weights);
        let dq = q.dequantize();
        let max_err = weights
            .iter()
            .zip(dq.iter())
            .map(|(w, d)| (w - d).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 0.25,
            "max dequantize error should be < 0.25, got {max_err}"
        );
    }

    #[test]
    fn to_bytes_from_bytes_roundtrip() {
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.2, 0.7, -0.9, 0.4, 0.6];
        let q = quantize_weights(&weights);
        let bytes = q.to_bytes();
        let view = TurboQuantizedView::from_bytes(&bytes).expect("valid bytes");
        assert_eq!(view.n_weights(), q.n_weights());
        let features = vec![1.0; 10];
        let pred_owned = q.predict(&features);
        let pred_view = view.predict(&features);
        assert!(
            (pred_owned - pred_view).abs() < 1e-15,
            "owned vs view predict mismatch: {pred_owned} vs {pred_view}"
        );
    }

    #[test]
    fn from_bytes_rejects_short() {
        assert!(TurboQuantizedView::from_bytes(&[0u8; 10]).is_err());
        assert!(TurboQuantizedView::from_bytes(&[0u8; 35]).is_err());
    }

    #[test]
    fn compression_ratio_reasonable() {
        let weights: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let q = quantize_weights(&weights);
        let ratio = q.compression_ratio();
        assert!(
            ratio > 3.0,
            "compression ratio should be > 3x for 100 weights, got {ratio:.2}"
        );
    }

    #[test]
    fn predict_large_vector() {
        let n = 1000;
        let weights: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin()).collect();
        let features: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.05).cos()).collect();
        let exact: f64 = weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();
        let q = quantize_weights(&weights);
        let pred = q.predict(&features);
        assert!(pred.is_finite(), "prediction should be finite");
        let abs_err = (pred - exact).abs();
        assert!(
            abs_err < exact.abs() * 0.5 + 5.0,
            "absolute error too large: {abs_err} for exact {exact}"
        );
    }

    #[test]
    fn next_power_of_two_correctness() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(7), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(1024), 1024);
        assert_eq!(next_power_of_two(1025), 2048);
    }

    #[test]
    fn fwht_roundtrip() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        for (i, (&a, &b)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "FWHT roundtrip failed at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn fwht_roundtrip_large() {
        let n = 64;
        let mut data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 - 3.0).collect();
        let original = data.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        for (i, (&a, &b)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "FWHT large roundtrip failed at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn sign_flip_is_self_inverse() {
        let seed = 42u64;
        let mut data = vec![1.0, -2.5, 3.7, 0.0, -1.1, 5.5, 2.2, -0.8];
        let original = data.clone();
        apply_sign_flip(&mut data, seed);
        apply_sign_flip(&mut data, seed);
        for (i, (&a, &b)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "sign flip self-inverse failed at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn full_rotation_roundtrip() {
        let seed = 0xCAFE_u64;
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut buf = original.clone();
        apply_rotation(&mut buf, seed);
        apply_inverse_rotation(&mut buf, seed);
        for (i, (&a, &b)) in buf.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "rotation roundtrip failed at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let seed = 0xBEEF_u64;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f64 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        apply_rotation(&mut rotated, seed);
        let norm_after: f64 = rotated.iter().map(|x| x * x).sum();
        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "rotation should preserve norm: {norm_before} vs {norm_after}"
        );
    }

    #[test]
    fn rotation_improves_correlated_weights() {
        let weights = vec![1.0, 1.01, 0.99, 1.02, 0.98, 1.01, 0.99, 1.0];
        let q = quantize_weights(&weights);
        let dq = q.dequantize();
        let max_err: f64 = weights
            .iter()
            .zip(dq.iter())
            .map(|(w, d)| (w - d).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 0.05,
            "rotation should improve correlated weight quantization, max_err={max_err}"
        );
    }

    #[test]
    fn quantize_with_seed_deterministic() {
        let weights = vec![0.1, -0.5, 0.3, 0.0, -0.2, 0.4, 0.1, -0.1];
        let features = vec![1.0; 8];
        let q1 = quantize_weights_with_seed(&weights, 123);
        let q2 = quantize_weights_with_seed(&weights, 123);
        let p1 = q1.predict(&features);
        let p2 = q2.predict(&features);
        assert!(
            (p1 - p2).abs() < 1e-15,
            "same seed should give identical results: {p1} vs {p2}"
        );
    }

    #[test]
    fn different_seeds_produce_different_quantizations() {
        let weights = vec![0.1, -0.5, 0.3, 0.0, -0.2, 0.4, 0.1, -0.1];
        let q1 = quantize_weights_with_seed(&weights, 111);
        let q2 = quantize_weights_with_seed(&weights, 222);
        assert_ne!(
            q1.packed, q2.packed,
            "different seeds should produce different packed data"
        );
    }

    #[test]
    fn to_bytes_from_bytes_preserves_seed_and_padded_len() {
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let q = quantize_weights_with_seed(&weights, 0xABCD);
        let bytes = q.to_bytes();
        let view = TurboQuantizedView::from_bytes(&bytes).expect("valid bytes");
        assert_eq!(view.seed, 0xABCD);
        assert_eq!(view.padded_len, q.padded_len);
        assert_eq!(view.n_weights(), q.n_weights());
    }

    // ---- New tests: 8-bit mode ----

    #[test]
    fn bits8_pack_unpack_roundtrip() {
        let values = [0u8, 127, 255, 42];
        let packed = pack4_bytes(&values);
        let unpacked = unpack4_bytes(packed, 4);
        assert_eq!(&unpacked, &values, "8-bit pack/unpack roundtrip failed");
    }

    #[test]
    fn bits8_near_lossless() {
        let weights: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) / 32.0).collect();
        let q = quantize(&weights, QuantMode::Bits8, DEFAULT_SEED);
        let dq = q.dequantize();
        let max_err = weights
            .iter()
            .zip(dq.iter())
            .map(|(w, d)| (w - d).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 0.02,
            "8-bit should be near-lossless, max_err={max_err}"
        );
    }

    #[test]
    fn bits8_predict_accuracy() {
        let weights: Vec<f64> = (0..32).map(|i| (i as f64).sin() * 0.5).collect();
        let features: Vec<f64> = (0..32).map(|i| (i as f64).cos() * 0.3).collect();
        let exact: f64 = weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();
        let q = quantize(&weights, QuantMode::Bits8, DEFAULT_SEED);
        let pred = q.predict(&features);
        let rel_err = (pred - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.10,
            "8-bit predict should have <10% relative error, got {rel_err:.4}"
        );
    }

    // ---- New tests: 2.5-bit mode ----

    #[test]
    fn bits2_5_packing_roundtrip() {
        let values = [0u8, 4, 2, 1, 3, 0, 4, 2, 1, 3, 0, 4, 2];
        let packed = pack13(&values);
        let unpacked = unpack13(packed, 13);
        assert_eq!(&unpacked, &values, "2.5-bit pack/unpack roundtrip failed");
    }

    #[test]
    fn bits2_5_quantize_and_predict() {
        let weights: Vec<f64> = (0..16).map(|i| (i as f64 - 8.0) / 8.0).collect();
        let features = vec![1.0; 16];
        let q = quantize(&weights, QuantMode::Bits2_5, DEFAULT_SEED);
        let pred = q.predict(&features);
        assert!(pred.is_finite(), "2.5-bit predict should be finite");
    }

    // ---- New tests: cross-mode serialization ----

    #[test]
    fn all_modes_serialize_roundtrip() {
        let weights = vec![0.1, -0.3, 0.5, 0.0, -0.2, 0.4, 0.3, -0.1];
        for mode in [QuantMode::Bits8, QuantMode::Bits3_5, QuantMode::Bits2_5] {
            let q = quantize(&weights, mode, 42);
            let bytes = q.to_bytes();
            let view = TurboQuantizedView::from_bytes(&bytes).expect("valid bytes");
            assert_eq!(view.n_weights(), q.n_weights());
            assert_eq!(view.mode(), mode);
            let features = vec![1.0; 8];
            let p1 = q.predict(&features);
            let p2 = view.predict(&features);
            assert!(
                (p1 - p2).abs() < 1e-15,
                "mode {mode:?}: owned={p1} vs view={p2}"
            );
        }
    }

    // ---- New tests: zero-alloc predict ----

    #[test]
    fn predict_with_scratch_matches_predict() {
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.2];
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = quantize(&weights, QuantMode::Bits3_5, DEFAULT_SEED);
        let pred = q.predict(&features);
        let mut scratch = vec![0.0; q.padded_len()];
        let pred_scratch = q.predict_with_scratch(&features, &mut scratch);
        assert!(
            (pred - pred_scratch).abs() < 1e-15,
            "scratch predict should match: {pred} vs {pred_scratch}"
        );
    }

    #[test]
    fn predict_with_scratch_view_matches_predict() {
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.2];
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = quantize(&weights, QuantMode::Bits8, DEFAULT_SEED);
        let bytes = q.to_bytes();
        let view = TurboQuantizedView::from_bytes(&bytes).expect("valid bytes");
        let pred = view.predict(&features);
        let mut scratch = vec![0.0; view.padded_len()];
        let pred_scratch = view.predict_with_scratch(&features, &mut scratch);
        assert!(
            (pred - pred_scratch).abs() < 1e-15,
            "view scratch predict should match: {pred} vs {pred_scratch}"
        );
    }

    // ---- New tests: multi-type input ----

    #[test]
    fn quantize_f32_works() {
        let weights = vec![0.5f32, -0.3, 0.8, -0.1];
        let q = quantize_f32(&weights, QuantMode::Bits8);
        assert_eq!(q.n_weights(), 4);
        let pred = q.predict(&[1.0, 1.0, 1.0, 1.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn quantize_i16_works() {
        let weights = vec![1000i16, -500, 2000, -1000];
        let scale = 1.0 / 32767.0;
        let q = quantize_i16(&weights, scale, QuantMode::Bits3_5);
        assert_eq!(q.n_weights(), 4);
    }

    // ---- New tests: mode-specific compression ratios ----

    #[test]
    fn bits8_compression_ratio() {
        let weights: Vec<f64> = (0..256).map(|i| i as f64 * 0.01).collect();
        let q = quantize(&weights, QuantMode::Bits8, DEFAULT_SEED);
        let ratio = q.compression_ratio();
        // 256 * 8 bytes = 2048 bytes original. 8-bit: 256/4 = 64 words + 36 header = 292 bytes. ~7x
        assert!(
            ratio > 5.0,
            "8-bit compression ratio should be > 5x, got {ratio:.2}"
        );
    }

    #[test]
    fn bits2_5_compression_ratio() {
        let weights: Vec<f64> = (0..256).map(|i| i as f64 * 0.01).collect();
        let q = quantize(&weights, QuantMode::Bits2_5, DEFAULT_SEED);
        let ratio = q.compression_ratio();
        // 2.5-bit: 256/13 = ~20 words + 36 header = ~116 bytes. ~17x
        assert!(
            ratio > 10.0,
            "2.5-bit compression ratio should be > 10x, got {ratio:.2}"
        );
    }

    // ---- New tests: edge cases ----

    #[test]
    fn quantize_empty_all_modes() {
        for mode in [QuantMode::Bits8, QuantMode::Bits3_5, QuantMode::Bits2_5] {
            let q = quantize(&[], mode, DEFAULT_SEED);
            assert_eq!(q.n_weights(), 0);
            assert_eq!(q.predict(&[]), 0.0);
        }
    }

    #[test]
    fn predict_with_scratch_empty() {
        let q = quantize(&[], QuantMode::Bits3_5, DEFAULT_SEED);
        let mut scratch = vec![0.0; 1];
        assert_eq!(q.predict_with_scratch(&[], &mut scratch), 0.0);
    }
}
