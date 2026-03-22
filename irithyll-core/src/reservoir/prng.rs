//! Deterministic xorshift64 PRNG for reproducible reservoir initialization.
//!
//! [`Xorshift64Rng`] is a simple, fast, `no_std`-compatible pseudo-random number
//! generator. It produces uniform f64 values in `[0, 1)` and standard normal
//! samples via the Box-Muller transform.
//!
//! The generator is fully deterministic given a seed, ensuring that reservoir
//! weights are reproducible across platforms and runs.

use crate::math;

/// Xorshift64 pseudo-random number generator.
///
/// Period: 2^64 - 1 (the zero state is excluded). Given the same seed, the
/// sequence is identical across all platforms.
///
/// # Examples
///
/// ```
/// use irithyll_core::reservoir::Xorshift64Rng;
///
/// let mut rng = Xorshift64Rng::new(42);
/// let u = rng.next_f64();
/// assert!(u >= 0.0 && u < 1.0);
/// ```
pub struct Xorshift64Rng {
    state: u64,
    /// Cached second normal sample from Box-Muller (None if consumed).
    cached_normal: Option<f64>,
}

impl Xorshift64Rng {
    /// Create a new PRNG with the given seed.
    ///
    /// A seed of 0 is silently replaced with 1 to avoid the degenerate
    /// all-zeros fixed point of xorshift.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
            cached_normal: None,
        }
    }

    /// Advance the state and return the next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a uniform f64 in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for the mantissa (f64 has 52-bit mantissa + implicit 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a uniform f64 in `[-scale, scale]`.
    #[inline]
    pub fn next_uniform(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }

    /// Return a sample from the standard normal distribution N(0, 1)
    /// using the Box-Muller transform.
    ///
    /// Each call to Box-Muller produces two independent normal samples.
    /// The second is cached and returned on the next call, so on average
    /// this consumes one uniform sample per normal sample.
    pub fn next_normal(&mut self) -> f64 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }

        // Box-Muller: generate two independent N(0,1) from two U(0,1).
        let u1 = loop {
            let u = self.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.next_f64();

        let two_pi = 2.0 * core::f64::consts::PI;
        let r = math::sqrt(-2.0 * math::ln(u1));
        let theta = two_pi * u2;

        let z0 = r * math::cos(theta);
        let z1 = r * math::sin(theta);

        self.cached_normal = Some(z1);
        z0
    }

    /// Return a random sign: +1.0 or -1.0 with equal probability.
    #[inline]
    pub fn next_sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut rng1 = Xorshift64Rng::new(12345);
        let mut rng2 = Xorshift64Rng::new(12345);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn uniform_in_zero_one() {
        let mut rng = Xorshift64Rng::new(42);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "out of range: {}", u);
        }
    }

    #[test]
    fn uniform_symmetric_range() {
        let mut rng = Xorshift64Rng::new(99);
        let scale = 2.5;
        for _ in 0..5_000 {
            let v = rng.next_uniform(scale);
            assert!(
                (-scale..=scale).contains(&v),
                "value {} out of [-{}, {}]",
                v,
                scale,
                scale,
            );
        }
    }

    #[test]
    fn normal_mean_and_variance() {
        let mut rng = Xorshift64Rng::new(7);
        let n = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let z = rng.next_normal();
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(
            math::abs(mean) < 0.02,
            "normal mean too far from 0: {}",
            mean,
        );
        assert!(
            math::abs(var - 1.0) < 0.05,
            "normal variance too far from 1: {}",
            var,
        );
    }

    #[test]
    fn sign_produces_both_values() {
        let mut rng = Xorshift64Rng::new(1);
        let mut pos = 0;
        let mut neg = 0;
        for _ in 0..1000 {
            let s = rng.next_sign();
            if s > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
        }
        assert!(pos > 100, "too few positive signs: {}", pos);
        assert!(neg > 100, "too few negative signs: {}", neg);
    }

    #[test]
    fn seed_zero_does_not_degenerate() {
        let mut rng = Xorshift64Rng::new(0);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        // With seed 0 silently mapped to 1, the sequence should advance.
        assert_ne!(v1, 0);
        assert_ne!(v1, v2, "sequence should not be constant");
    }

    #[test]
    fn normal_uses_cache() {
        // Calling next_normal twice should use the cache on the second call.
        let mut rng = Xorshift64Rng::new(42);
        let z0 = rng.next_normal();
        assert!(rng.cached_normal.is_some(), "second value should be cached");
        let z1 = rng.next_normal();
        assert!(rng.cached_normal.is_none(), "cache should be consumed",);
        // z0 and z1 should be distinct (statistically near-certain).
        assert!(
            (z0 - z1).abs() > 1e-15,
            "two normal samples should differ: z0={}, z1={}",
            z0,
            z1,
        );
    }
}
