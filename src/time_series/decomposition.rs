//! Online seasonal decomposition for streaming time series.
//!
//! Provides a streaming STL-lite algorithm that decomposes a time series into
//! trend, seasonal, and residual components. Unlike batch STL, this processes
//! one observation at a time with O(period) memory.
//!
//! - **Trend** is estimated via an exponentially weighted moving average (EWMA).
//! - **Seasonal** factors are per-position running averages of the deseasonalized
//!   residual, updated with a separate EWMA.
//! - **Residual** is simply `observed - trend - seasonal`.

/// A single decomposed observation, splitting the original value into its
/// trend, seasonal, and residual components.
///
/// The identity `observed == trend + seasonal + residual` holds exactly
/// (up to floating-point rounding).
#[derive(Debug, Clone, Copy)]
pub struct DecomposedPoint {
    /// Original observed value.
    pub observed: f64,
    /// Estimated trend component.
    pub trend: f64,
    /// Estimated seasonal component.
    pub seasonal: f64,
    /// Residual: observed - trend - seasonal.
    pub residual: f64,
}

/// Builder for [`DecompositionConfig`].
///
/// Constructed via [`DecompositionConfig::builder`]. Call `.build()` to
/// validate and produce the final configuration.
///
/// # Example
///
/// ```
/// use irithyll::time_series::decomposition::DecompositionConfig;
///
/// let config = DecompositionConfig::builder(7)
///     .trend_alpha(0.2)
///     .seasonal_alpha(0.1)
///     .build()
///     .unwrap();
/// assert_eq!(config.period, 7);
/// ```
#[derive(Debug, Clone)]
pub struct DecompositionConfigBuilder {
    period: usize,
    trend_alpha: f64,
    seasonal_alpha: f64,
}

impl DecompositionConfigBuilder {
    /// Set the EWMA alpha for trend estimation.
    ///
    /// Must be in the open interval (0, 1). Smaller values produce a smoother
    /// trend that reacts more slowly to level shifts.
    pub fn trend_alpha(mut self, a: f64) -> Self {
        self.trend_alpha = a;
        self
    }

    /// Set the EWMA alpha for seasonal factor estimation.
    ///
    /// Must be in the open interval (0, 1). Smaller values produce more
    /// stable seasonal patterns that require many cycles to change.
    pub fn seasonal_alpha(mut self, a: f64) -> Self {
        self.seasonal_alpha = a;
        self
    }

    /// Validate and build the configuration.
    ///
    /// Returns an error if `period < 2` or either alpha is outside (0, 1).
    pub fn build(self) -> Result<DecompositionConfig, String> {
        if self.period < 2 {
            return Err(format!("period must be >= 2, got {}", self.period));
        }
        if self.trend_alpha <= 0.0 || self.trend_alpha >= 1.0 {
            return Err(format!(
                "trend_alpha must be in (0, 1), got {}",
                self.trend_alpha
            ));
        }
        if self.seasonal_alpha <= 0.0 || self.seasonal_alpha >= 1.0 {
            return Err(format!(
                "seasonal_alpha must be in (0, 1), got {}",
                self.seasonal_alpha
            ));
        }
        Ok(DecompositionConfig {
            period: self.period,
            trend_alpha: self.trend_alpha,
            seasonal_alpha: self.seasonal_alpha,
        })
    }
}

/// Configuration for [`StreamingDecomposition`].
///
/// Use [`DecompositionConfig::builder`] to construct with validation.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Seasonal period (number of positions in one full cycle).
    pub period: usize,
    /// EWMA alpha for trend estimation (0 < alpha < 1).
    /// Smaller values produce a smoother trend.
    pub trend_alpha: f64,
    /// EWMA alpha for seasonal factor estimation (0 < alpha < 1).
    /// Smaller values produce more stable seasonal patterns.
    pub seasonal_alpha: f64,
}

impl DecompositionConfig {
    /// Create a new builder with the given seasonal period.
    ///
    /// Defaults: `trend_alpha = 0.1`, `seasonal_alpha = 0.05`.
    pub fn builder(period: usize) -> DecompositionConfigBuilder {
        DecompositionConfigBuilder {
            period,
            trend_alpha: 0.1,
            seasonal_alpha: 0.05,
        }
    }
}

/// Online seasonal decomposition via EWMA trend and per-position seasonal factors.
///
/// Processes observations one at a time with O(period) memory. Each call to
/// [`update`](StreamingDecomposition::update) returns a [`DecomposedPoint`]
/// splitting the observation into trend, seasonal, and residual components.
///
/// # Algorithm
///
/// For each observation y at position `t mod period`:
///
/// 1. **Trend update:** `trend = alpha * y + (1 - alpha) * trend`
///    (first sample initializes trend to y)
/// 2. **Seasonal deviation:** `dev = y - trend`
/// 3. **Seasonal factor update:** `seasonal[pos] = s_alpha * dev + (1 - s_alpha) * seasonal[pos]`
/// 4. **Residual:** `y - trend - seasonal[pos]`
/// 5. **Advance position:** `pos = (pos + 1) % period`
///
/// After one full period of observations, `initialized` becomes true.
///
/// # Example
///
/// ```
/// use irithyll::time_series::decomposition::{DecompositionConfig, StreamingDecomposition};
///
/// let config = DecompositionConfig::builder(4)
///     .trend_alpha(0.3)
///     .seasonal_alpha(0.1)
///     .build()
///     .unwrap();
///
/// let mut decomp = StreamingDecomposition::new(config);
///
/// // Feed observations from a series with trend + seasonal pattern
/// let pattern = [10.0, -5.0, -5.0, 10.0];
/// for i in 0..20 {
///     let y = 100.0 + pattern[i % 4];
///     let point = decomp.update(y);
///     // Identity always holds:
///     assert!((point.observed - (point.trend + point.seasonal + point.residual)).abs() < 1e-10);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StreamingDecomposition {
    config: DecompositionConfig,
    /// Current EWMA trend estimate.
    trend: f64,
    /// Per-position seasonal factors (length = period).
    seasonal: Vec<f64>,
    /// Number of samples observed at each seasonal position (for warmup tracking).
    season_counts: Vec<u64>,
    /// Current position in the seasonal cycle (t mod period).
    position: usize,
    /// Total number of samples processed.
    n_samples: u64,
    /// Whether at least one full period has been observed.
    initialized: bool,
}

impl StreamingDecomposition {
    /// Create a new streaming decomposition from the given configuration.
    pub fn new(config: DecompositionConfig) -> Self {
        let period = config.period;
        Self {
            config,
            trend: 0.0,
            seasonal: vec![0.0; period],
            season_counts: vec![0; period],
            position: 0,
            n_samples: 0,
            initialized: false,
        }
    }

    /// Process one observation and return its decomposition.
    ///
    /// The returned [`DecomposedPoint`] satisfies the identity:
    /// `observed = trend + seasonal + residual` (up to floating-point rounding).
    pub fn update(&mut self, y: f64) -> DecomposedPoint {
        let pos = self.position;

        // 1. Update trend via EWMA (first sample initializes to y)
        if self.n_samples == 0 {
            self.trend = y;
        } else {
            self.trend = self.config.trend_alpha * y + (1.0 - self.config.trend_alpha) * self.trend;
        }

        // 2. Compute seasonal deviation
        let dev = y - self.trend;

        // 3. Update seasonal factor for current position via EWMA
        self.seasonal[pos] = self.config.seasonal_alpha * dev
            + (1.0 - self.config.seasonal_alpha) * self.seasonal[pos];

        // 4. Compute decomposition
        let trend_component = self.trend;
        let seasonal_component = self.seasonal[pos];
        let residual = y - trend_component - seasonal_component;

        // 5. Bookkeeping
        self.season_counts[pos] += 1;
        self.n_samples += 1;
        self.position = (pos + 1) % self.config.period;

        // Mark initialized once every position has been seen at least once
        if !self.initialized && self.n_samples >= self.config.period as u64 {
            self.initialized = true;
        }

        DecomposedPoint {
            observed: y,
            trend: trend_component,
            seasonal: seasonal_component,
            residual,
        }
    }

    /// Current trend estimate.
    pub fn trend(&self) -> f64 {
        self.trend
    }

    /// Current per-position seasonal factors.
    ///
    /// The returned slice has length equal to the configured period. Each
    /// element is the EWMA-smoothed seasonal deviation for that position.
    pub fn seasonal_factors(&self) -> &[f64] {
        &self.seasonal
    }

    /// Current position in the seasonal cycle (0-indexed).
    ///
    /// This is the position that will be used for the *next* observation.
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Total number of samples processed so far.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Whether at least one full seasonal period has been observed.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset all state to the initial empty condition.
    ///
    /// The configuration is preserved; only internal accumulators are cleared.
    pub fn reset(&mut self) {
        self.trend = 0.0;
        self.seasonal.fill(0.0);
        self.season_counts.fill(0);
        self.position = 0;
        self.n_samples = 0;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn default_config(period: usize) -> DecompositionConfig {
        DecompositionConfig::builder(period).build().unwrap()
    }

    #[test]
    fn constant_series_zero_seasonal() {
        let config = DecompositionConfig::builder(4)
            .trend_alpha(0.3)
            .seasonal_alpha(0.1)
            .build()
            .unwrap();
        let mut decomp = StreamingDecomposition::new(config);

        // Feed a constant series for several cycles
        for _ in 0..40 {
            let pt = decomp.update(50.0);

            // Seasonal should stay near zero (constant input has no seasonal variation)
            assert!(
                pt.seasonal.abs() < 1.0,
                "seasonal {} should be near zero for constant input",
                pt.seasonal
            );
        }

        // After many samples, seasonal factors should all be very close to zero
        for (i, &s) in decomp.seasonal_factors().iter().enumerate() {
            assert!(
                s.abs() < 0.5,
                "seasonal factor at position {} = {}, expected near zero",
                i,
                s
            );
        }
    }

    #[test]
    fn trend_tracks_level_shift() {
        let config = DecompositionConfig::builder(4)
            .trend_alpha(0.3)
            .seasonal_alpha(0.05)
            .build()
            .unwrap();
        let mut decomp = StreamingDecomposition::new(config);

        // Feed value 10 for 20 samples
        for _ in 0..20 {
            decomp.update(10.0);
        }
        let trend_at_10 = decomp.trend();
        assert!(
            approx_eq(trend_at_10, 10.0, 0.5),
            "trend should be near 10.0 after 20 samples, got {}",
            trend_at_10
        );

        // Feed value 20 for 20 samples
        for _ in 0..20 {
            decomp.update(20.0);
        }
        let trend_at_20 = decomp.trend();
        assert!(
            trend_at_20 > 15.0,
            "trend should have moved toward 20.0, got {}",
            trend_at_20
        );
        assert!(
            approx_eq(trend_at_20, 20.0, 1.0),
            "trend should be near 20.0 after level shift, got {}",
            trend_at_20
        );
    }

    #[test]
    fn seasonal_pattern_captured() {
        let config = DecompositionConfig::builder(4)
            .trend_alpha(0.05)
            .seasonal_alpha(0.15)
            .build()
            .unwrap();
        let mut decomp = StreamingDecomposition::new(config);

        let pattern = [10.0, -5.0, -5.0, 10.0];

        // Feed y = 100 + pattern[t % 4] for many cycles so the EWMA converges
        for i in 0..500 {
            let y = 100.0 + pattern[i % 4];
            decomp.update(y);
        }

        // Seasonal factors should approximate the true pattern
        let factors = decomp.seasonal_factors();
        let tol = 4.0; // EWMA converges slowly, generous tolerance
        for (i, &expected) in pattern.iter().enumerate() {
            assert!(
                approx_eq(factors[i], expected, tol),
                "seasonal factor at position {} = {}, expected near {}",
                i,
                factors[i],
                expected
            );
        }
    }

    #[test]
    fn decomposition_identity() {
        let config = DecompositionConfig::builder(5)
            .trend_alpha(0.2)
            .seasonal_alpha(0.1)
            .build()
            .unwrap();
        let mut decomp = StreamingDecomposition::new(config);

        // Feed a mix of values and verify the identity at every point
        let values = [
            3.0, 7.0, 1.5, 9.2, 4.8, 6.1, 2.3, 8.7, 0.5, 5.5, 3.3, 7.7, 1.1, 9.9, 4.4, 6.6, 2.2,
            8.8, 0.0, 5.0,
        ];
        for &y in &values {
            let pt = decomp.update(y);
            let reconstructed = pt.trend + pt.seasonal + pt.residual;
            assert!(
                approx_eq(pt.observed, reconstructed, EPS),
                "identity violated: observed={}, trend+seasonal+residual={}",
                pt.observed,
                reconstructed
            );
        }
    }

    #[test]
    fn position_cycles_correctly() {
        let period = 7;
        let config = default_config(period);
        let mut decomp = StreamingDecomposition::new(config);

        for i in 0..30 {
            assert_eq!(
                decomp.current_position(),
                i % period,
                "position mismatch at sample {}",
                i
            );
            decomp.update(1.0);
        }
        // After 30 samples, position should be 30 % 7 = 2
        assert_eq!(decomp.current_position(), 30 % period);
    }

    #[test]
    fn reset_clears_state() {
        let config = DecompositionConfig::builder(4)
            .trend_alpha(0.2)
            .seasonal_alpha(0.1)
            .build()
            .unwrap();
        let mut decomp = StreamingDecomposition::new(config);

        // Feed some data
        for i in 0..20 {
            decomp.update(i as f64);
        }
        assert!(decomp.n_samples_seen() > 0);
        assert!(decomp.is_initialized());

        // Reset and verify clean state
        decomp.reset();
        assert_eq!(decomp.n_samples_seen(), 0);
        assert_eq!(decomp.current_position(), 0);
        assert!(!decomp.is_initialized());
        assert_eq!(decomp.trend(), 0.0);
        for &s in decomp.seasonal_factors() {
            assert_eq!(s, 0.0, "seasonal factor should be zero after reset");
        }
    }

    #[test]
    fn config_validates() {
        // period < 2
        let err = DecompositionConfig::builder(1).build();
        assert!(err.is_err(), "period=1 should be rejected");
        assert!(
            err.unwrap_err().contains("period"),
            "error should mention period"
        );

        // period = 0
        let err = DecompositionConfig::builder(0).build();
        assert!(err.is_err(), "period=0 should be rejected");

        // trend_alpha out of range
        let err = DecompositionConfig::builder(4).trend_alpha(0.0).build();
        assert!(err.is_err(), "trend_alpha=0.0 should be rejected");

        let err = DecompositionConfig::builder(4).trend_alpha(1.0).build();
        assert!(err.is_err(), "trend_alpha=1.0 should be rejected");

        let err = DecompositionConfig::builder(4).trend_alpha(-0.5).build();
        assert!(err.is_err(), "trend_alpha=-0.5 should be rejected");

        // seasonal_alpha out of range
        let err = DecompositionConfig::builder(4).seasonal_alpha(0.0).build();
        assert!(err.is_err(), "seasonal_alpha=0.0 should be rejected");

        let err = DecompositionConfig::builder(4).seasonal_alpha(1.5).build();
        assert!(err.is_err(), "seasonal_alpha=1.5 should be rejected");

        // Valid config should succeed
        let ok = DecompositionConfig::builder(4)
            .trend_alpha(0.5)
            .seasonal_alpha(0.5)
            .build();
        assert!(ok.is_ok(), "valid config should build successfully");
    }

    #[test]
    fn first_sample_initializes_trend() {
        let config = default_config(4);
        let mut decomp = StreamingDecomposition::new(config);

        let pt = decomp.update(42.0);
        assert_eq!(
            pt.trend, 42.0,
            "first sample should initialize trend to the observed value"
        );
    }

    #[test]
    fn initialized_after_one_period() {
        let period = 5;
        let config = default_config(period);
        let mut decomp = StreamingDecomposition::new(config);

        for i in 0..period - 1 {
            decomp.update(i as f64);
            assert!(
                !decomp.is_initialized(),
                "should not be initialized after {} samples (period={})",
                i + 1,
                period
            );
        }

        decomp.update(99.0);
        assert!(
            decomp.is_initialized(),
            "should be initialized after {} samples (one full period)",
            period
        );
    }

    #[test]
    fn n_samples_increments() {
        let config = default_config(3);
        let mut decomp = StreamingDecomposition::new(config);

        for i in 0..10 {
            assert_eq!(decomp.n_samples_seen(), i as u64);
            decomp.update(1.0);
        }
        assert_eq!(decomp.n_samples_seen(), 10);
    }
}
