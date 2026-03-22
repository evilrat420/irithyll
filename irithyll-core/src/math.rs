//! Platform-agnostic f64 math operations.
//!
//! In `std` mode, these delegate to inherent f64 methods (zero overhead).
//! In `no_std` mode, these use `libm` (pure Rust software implementations).
//!
//! This module exists because f64 inherent methods (.sqrt(), .exp(), etc.)
//! are not available in no_std on MSRV 1.75.

// NOTE: We always use libm functions regardless of std feature.
// On std targets, LLVM will optimize these to the same native instructions.
// This avoids conditional compilation complexity and ensures identical behavior.

/// Absolute value.
#[inline]
pub fn abs(x: f64) -> f64 {
    libm::fabs(x)
}

/// Square root.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Natural exponential (e^x).
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm (ln).
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Base-2 logarithm.
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

/// Base-10 logarithm.
#[inline]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// Power: x^n (floating point exponent).
#[inline]
pub fn powf(x: f64, n: f64) -> f64 {
    libm::pow(x, n)
}

/// Power: x^n (integer exponent).
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    libm::pow(x, n as f64)
}

/// Sine.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine.
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Floor.
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

/// Ceil.
#[inline]
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

/// Round to nearest integer.
#[inline]
pub fn round(x: f64) -> f64 {
    libm::round(x)
}

/// Hyperbolic tangent.
#[inline]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// Softplus: ln(1 + exp(x)), numerically stable.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        libm::exp(x)
    } else {
        libm::log(1.0 + libm::exp(x))
    }
}

/// Logistic sigmoid: 1 / (1 + exp(-x)), numerically stable.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = libm::exp(-x);
        1.0 / (1.0 + e)
    } else {
        let e = libm::exp(x);
        e / (1.0 + e)
    }
}

/// Minimum of two f64 values (handles NaN: returns the non-NaN value).
#[inline]
pub fn fmin(x: f64, y: f64) -> f64 {
    libm::fmin(x, y)
}

/// Maximum of two f64 values (handles NaN: returns the non-NaN value).
#[inline]
pub fn fmax(x: f64, y: f64) -> f64 {
    libm::fmax(x, y)
}

/// Error function.
#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// f32 absolute value.
#[inline]
pub fn abs_f32(x: f32) -> f32 {
    libm::fabsf(x)
}

/// f32 square root.
#[inline]
pub fn sqrt_f32(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_of_4() {
        assert!((sqrt(4.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn exp_of_0() {
        assert!((exp(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ln_of_e() {
        assert!((ln(core::f64::consts::E) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn abs_negative() {
        assert_eq!(abs(-5.0), 5.0);
        assert_eq!(abs(5.0), 5.0);
        assert_eq!(abs(0.0), 0.0);
    }

    #[test]
    fn powf_squares() {
        assert!((powf(3.0, 2.0) - 9.0).abs() < 1e-15);
    }

    #[test]
    fn powi_cubes() {
        assert!((powi(2.0, 3) - 8.0).abs() < 1e-15);
    }

    #[test]
    fn sin_cos_identity() {
        let x = 1.0;
        let s = sin(x);
        let c = cos(x);
        assert!((s * s + c * c - 1.0).abs() < 1e-15);
    }

    #[test]
    fn floor_ceil_round() {
        assert_eq!(floor(2.7), 2.0);
        assert_eq!(ceil(2.3), 3.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(2.4), 2.0);
    }

    #[test]
    fn log2_of_8() {
        assert!((log2(8.0) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn tanh_of_0() {
        assert!((tanh(0.0)).abs() < 1e-15);
    }

    #[test]
    fn fmin_fmax() {
        assert_eq!(fmin(1.0, 2.0), 1.0);
        assert_eq!(fmax(1.0, 2.0), 2.0);
    }

    #[test]
    fn softplus_large_positive() {
        // For x >> 0, softplus(x) ~ x
        assert!((softplus(50.0) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn softplus_large_negative() {
        // For x << 0, softplus(x) ~ 0
        let result = softplus(-50.0);
        assert!((0.0..1e-20).contains(&result));
    }

    #[test]
    fn softplus_zero() {
        let expected = ln(2.0);
        assert!((softplus(0.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn softplus_always_positive() {
        for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!(softplus(x) > 0.0, "softplus({}) should be > 0", x);
        }
    }

    #[test]
    fn sigmoid_at_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_range() {
        for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!(
                (0.0..1.0).contains(&s),
                "sigmoid({}) = {} should be in (0, 1)",
                x,
                s
            );
        }
    }

    #[test]
    fn sigmoid_symmetry() {
        let x = 3.0;
        assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_extreme_values() {
        let s_pos = sigmoid(100.0);
        let s_neg = sigmoid(-100.0);
        assert!((0.0..=1.0).contains(&s_pos));
        assert!((0.0..=1.0).contains(&s_neg));
    }
}
