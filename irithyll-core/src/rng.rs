//! Deterministic xorshift64 PRNG for reproducible initialization.
//!
//! Provides both a struct-based API ([`Xorshift64`]) and free functions
//! ([`xorshift64`], [`xorshift64_f64`], [`standard_normal`]) operating
//! on a bare `u64` state. The struct is ergonomic for callers that own
//! an RNG, while the free functions are convenient for modules that
//! already thread a `&mut u64` seed through their APIs.
//!
//! # no_std
//!
//! Everything in this module is `no_std`-compatible and allocation-free.

use crate::math;

// ---------------------------------------------------------------------------
// Struct-based API
// ---------------------------------------------------------------------------

/// Xorshift64 pseudo-random number generator.
///
/// A fast, deterministic PRNG suitable for weight initialization. Not
/// cryptographically secure, but provides good statistical properties
/// for ML weight sampling.
///
/// # Example
///
/// ```
/// use irithyll_core::rng::Xorshift64;
///
/// let mut rng = Xorshift64(12345);
/// let val = rng.next_f64();   // uniform in [0, 1)
/// let normal = rng.next_normal(); // standard normal via Box-Muller
/// ```
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    /// Generate the next random u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        xorshift64(&mut self.0)
    }

    /// Generate a uniform random f64 in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        xorshift64_f64(&mut self.0)
    }

    /// Generate a standard normal random value via Box-Muller transform.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        standard_normal(&mut self.0)
    }
}

// ---------------------------------------------------------------------------
// Free-function API (operates on a bare &mut u64 state)
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state and return the new raw `u64`.
///
/// This is the core step shared by all RNG helpers in irithyll.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform f64 in `[0, 1)` from an xorshift64 state.
///
/// Uses the upper 53 bits for full mantissa precision.
#[inline]
pub fn xorshift64_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Standard normal sample via Box-Muller transform (returns one sample).
///
/// Uses [`xorshift64_f64`] internally and delegates to
/// [`crate::math`] for `sqrt`, `ln`, and `cos`.
pub fn standard_normal(state: &mut u64) -> f64 {
    let u1 = loop {
        let u = xorshift64_f64(state);
        if u > 0.0 {
            break u;
        }
    };
    let u2 = xorshift64_f64(state);
    math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * core::f64::consts::PI * u2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..1000 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    #[test]
    fn xorshift64_f64_in_unit_interval() {
        let mut s = 12345u64;
        for _ in 0..10_000 {
            let v = xorshift64_f64(&mut s);
            assert!(
                (0.0..1.0).contains(&v),
                "xorshift64_f64 should be in [0, 1), got {}",
                v
            );
        }
    }

    #[test]
    fn standard_normal_finite() {
        let mut s = 99u64;
        for _ in 0..5_000 {
            let z = standard_normal(&mut s);
            assert!(z.is_finite(), "standard_normal returned non-finite: {}", z);
        }
    }

    #[test]
    fn struct_delegates_to_free_fns() {
        let mut rng = Xorshift64(42);
        let mut state = 42u64;
        for _ in 0..100 {
            assert_eq!(rng.next_u64(), xorshift64(&mut state));
        }
    }
}
