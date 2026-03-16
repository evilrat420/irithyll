//! Streaming Holt-Winters triple exponential smoothing.
//!
//! Online implementation supporting both additive and multiplicative
//! seasonality. State is O(m) where m is the seasonal period. No past
//! samples are stored after initialization.

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// Seasonality enum
// ---------------------------------------------------------------------------

/// Type of seasonal decomposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seasonality {
    /// Seasonal effects are added to the level+trend.
    Additive,
    /// Seasonal effects multiply the level+trend.
    Multiplicative,
}

// ---------------------------------------------------------------------------
// HoltWintersConfig
// ---------------------------------------------------------------------------

/// Configuration for the Holt-Winters exponential smoothing model.
///
/// Use [`HoltWintersConfig::builder`] to construct with validation.
///
/// # Example
///
/// ```
/// use irithyll::time_series::HoltWintersConfig;
///
/// let config = HoltWintersConfig::builder(12)
///     .alpha(0.2)
///     .beta(0.05)
///     .gamma(0.15)
///     .build()
///     .unwrap();
/// assert_eq!(config.period, 12);
/// ```
#[derive(Debug, Clone)]
pub struct HoltWintersConfig {
    /// Smoothing parameter for level (0 < alpha < 1).
    pub alpha: f64,
    /// Smoothing parameter for trend (0 < beta < 1).
    pub beta: f64,
    /// Smoothing parameter for seasonality (0 < gamma < 1).
    pub gamma: f64,
    /// Seasonal period (e.g., 12 for monthly, 7 for daily).
    pub period: usize,
    /// Seasonality type.
    pub seasonality: Seasonality,
}

impl HoltWintersConfig {
    /// Create a builder for `HoltWintersConfig` with the given seasonal period.
    pub fn builder(period: usize) -> HoltWintersConfigBuilder {
        HoltWintersConfigBuilder {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            period,
            seasonality: Seasonality::Additive,
        }
    }
}

// ---------------------------------------------------------------------------
// HoltWintersConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`HoltWintersConfig`] with parameter validation.
#[derive(Debug, Clone)]
pub struct HoltWintersConfigBuilder {
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    seasonality: Seasonality,
}

impl HoltWintersConfigBuilder {
    /// Set the level smoothing parameter (default 0.3).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the trend smoothing parameter (default 0.1).
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the seasonality smoothing parameter (default 0.1).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the seasonality type (default [`Seasonality::Additive`]).
    pub fn seasonality(mut self, seasonality: Seasonality) -> Self {
        self.seasonality = seasonality;
        self
    }

    /// Build the configuration, validating all parameters.
    ///
    /// Returns `Err` if any smoothing parameter is not in (0, 1) or period < 2.
    pub fn build(self) -> Result<HoltWintersConfig, String> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(format!("alpha must be in (0, 1), got {}", self.alpha));
        }
        if self.beta <= 0.0 || self.beta >= 1.0 {
            return Err(format!("beta must be in (0, 1), got {}", self.beta));
        }
        if self.gamma <= 0.0 || self.gamma >= 1.0 {
            return Err(format!("gamma must be in (0, 1), got {}", self.gamma));
        }
        if self.period < 2 {
            return Err(format!("period must be >= 2, got {}", self.period));
        }
        Ok(HoltWintersConfig {
            alpha: self.alpha,
            beta: self.beta,
            gamma: self.gamma,
            period: self.period,
            seasonality: self.seasonality,
        })
    }
}

// ---------------------------------------------------------------------------
// HoltWinters
// ---------------------------------------------------------------------------

/// Streaming Holt-Winters triple exponential smoothing.
///
/// Maintains level, trend, and seasonal components updated incrementally
/// with each observation. Supports both additive and multiplicative
/// seasonality modes.
///
/// The first `period` observations are buffered for initialization. Once a
/// full season is seen, level is set to the season mean, trend to 0, and
/// seasonal factors are estimated from the buffered values. The buffer is
/// then replayed through the update equations.
///
/// # Example
///
/// ```
/// use irithyll::time_series::{HoltWinters, HoltWintersConfig};
///
/// let config = HoltWintersConfig::builder(4)
///     .alpha(0.3)
///     .beta(0.1)
///     .gamma(0.1)
///     .build()
///     .unwrap();
///
/// let mut hw = HoltWinters::new(config);
///
/// // Feed a few seasons of data
/// for t in 0..20 {
///     let seasonal = [10.0, 20.0, 30.0, 15.0][t % 4];
///     hw.train_one(100.0 + seasonal);
/// }
///
/// // Forecast the next 4 steps
/// let forecast = hw.forecast(4);
/// assert_eq!(forecast.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct HoltWinters {
    config: HoltWintersConfig,
    level: f64,
    trend: f64,
    seasonal: Vec<f64>,
    season_idx: usize,
    n_samples: u64,
    initialized: bool,
    init_buffer: Vec<f64>,
}

impl HoltWinters {
    /// Create a new Holt-Winters model from the given configuration.
    pub fn new(config: HoltWintersConfig) -> Self {
        let period = config.period;
        let init_seasonal = match config.seasonality {
            Seasonality::Additive => vec![0.0; period],
            Seasonality::Multiplicative => vec![1.0; period],
        };
        Self {
            config,
            level: 0.0,
            trend: 0.0,
            seasonal: init_seasonal,
            season_idx: 0,
            n_samples: 0,
            initialized: false,
            init_buffer: Vec::with_capacity(period),
        }
    }

    /// Update the model with a single observation.
    ///
    /// During the initialization phase (first `period` observations), values
    /// are buffered. Once a full season is collected, the model is initialized
    /// and all buffered values are replayed.
    pub fn train_one(&mut self, y: f64) {
        self.n_samples += 1;

        if !self.initialized {
            self.init_buffer.push(y);
            if self.init_buffer.len() == self.config.period {
                self.initialize();
            }
            return;
        }

        self.update(y);
    }

    /// One-step-ahead forecast from the current state.
    ///
    /// Returns 0.0 if the model has not been initialized yet.
    pub fn predict_one(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.forecast_step(1)
    }

    /// Multi-step forecast from the current state.
    ///
    /// Returns a `Vec` of length `horizon` with forecasts for steps 1..=horizon.
    /// Returns an empty `Vec` if the model is not initialized or horizon is 0.
    pub fn forecast(&self, horizon: usize) -> Vec<f64> {
        if !self.initialized || horizon == 0 {
            return vec![0.0; horizon];
        }
        (1..=horizon).map(|h| self.forecast_step(h)).collect()
    }

    /// Current level component.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Current trend component.
    pub fn trend(&self) -> f64 {
        self.trend
    }

    /// Seasonal factors (one per period position).
    pub fn seasonal_factors(&self) -> &[f64] {
        &self.seasonal
    }

    /// Whether the model has been initialized (a full season has been seen).
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Total number of observations processed (including buffered ones).
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset the model to its initial untrained state.
    pub fn reset(&mut self) {
        let period = self.config.period;
        self.level = 0.0;
        self.trend = 0.0;
        self.seasonal = match self.config.seasonality {
            Seasonality::Additive => vec![0.0; period],
            Seasonality::Multiplicative => vec![1.0; period],
        };
        self.season_idx = 0;
        self.n_samples = 0;
        self.initialized = false;
        self.init_buffer.clear();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Initialize level, trend, and seasonal factors from the first season.
    fn initialize(&mut self) {
        let m = self.config.period;
        let buf = &self.init_buffer;

        // Level = mean of first season
        let mean: f64 = buf.iter().sum::<f64>() / m as f64;
        self.level = mean;

        // Trend = 0 (single-season initialization)
        self.trend = 0.0;

        // Seasonal factors from first season
        match self.config.seasonality {
            Seasonality::Additive => {
                for (i, &b) in buf.iter().enumerate().take(m) {
                    self.seasonal[i] = b - mean;
                }
            }
            Seasonality::Multiplicative => {
                for (i, &b) in buf.iter().enumerate().take(m) {
                    // Guard against zero mean to avoid division by zero.
                    if mean.abs() < f64::EPSILON {
                        self.seasonal[i] = 1.0;
                    } else {
                        self.seasonal[i] = b / mean;
                    }
                }
            }
        }

        self.initialized = true;
        self.season_idx = 0;

        // Replay the buffered observations through the update equations.
        // We clone the buffer since update() borrows &mut self.
        let replay: Vec<f64> = buf.clone();
        for &y in &replay {
            self.update(y);
        }
    }

    /// Update level, trend, and seasonal components with a single observation.
    fn update(&mut self, y: f64) {
        let m = self.config.period;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let gamma = self.config.gamma;

        let prev_level = self.level;
        let prev_trend = self.trend;
        let prev_seasonal = self.seasonal[self.season_idx];

        match self.config.seasonality {
            Seasonality::Additive => {
                // Level
                self.level =
                    alpha * (y - prev_seasonal) + (1.0 - alpha) * (prev_level + prev_trend);

                // Trend
                self.trend = beta * (self.level - prev_level) + (1.0 - beta) * prev_trend;

                // Seasonal
                self.seasonal[self.season_idx] =
                    gamma * (y - self.level) + (1.0 - gamma) * prev_seasonal;
            }
            Seasonality::Multiplicative => {
                // Guard against zero seasonal factor
                let safe_seasonal = if prev_seasonal.abs() < f64::EPSILON {
                    1.0
                } else {
                    prev_seasonal
                };

                // Level
                self.level =
                    alpha * (y / safe_seasonal) + (1.0 - alpha) * (prev_level + prev_trend);

                // Trend
                self.trend = beta * (self.level - prev_level) + (1.0 - beta) * prev_trend;

                // Seasonal — guard against zero level
                let safe_level = if self.level.abs() < f64::EPSILON {
                    1.0
                } else {
                    self.level
                };
                self.seasonal[self.season_idx] =
                    gamma * (y / safe_level) + (1.0 - gamma) * prev_seasonal;
            }
        }

        // Advance seasonal index
        self.season_idx = (self.season_idx + 1) % m;
    }

    /// Forecast h steps ahead from the current state.
    fn forecast_step(&self, h: usize) -> f64 {
        let m = self.config.period;
        // Seasonal index for h steps ahead:
        // s_{t-m+((h-1) mod m)+1}
        let idx = (self.season_idx + (h - 1) % m) % m;

        match self.config.seasonality {
            Seasonality::Additive => self.level + (h as f64) * self.trend + self.seasonal[idx],
            Seasonality::Multiplicative => {
                (self.level + (h as f64) * self.trend) * self.seasonal[idx]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for HoltWinters {
    fn train_one(&mut self, _features: &[f64], target: f64, _weight: f64) {
        HoltWinters::train_one(self, target);
    }

    fn predict(&self, _features: &[f64]) -> f64 {
        self.predict_one()
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        HoltWinters::reset(self);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-6;

    fn default_config(period: usize) -> HoltWintersConfig {
        HoltWintersConfig::builder(period)
            .alpha(0.3)
            .beta(0.1)
            .gamma(0.1)
            .build()
            .unwrap()
    }

    #[test]
    fn constant_series_converges() {
        let mut hw = HoltWinters::new(default_config(4));
        let val = 42.0;

        // Feed 100 observations of a constant
        for _ in 0..100 {
            hw.train_one(val);
        }

        assert!(
            hw.is_initialized(),
            "should be initialized after 100 samples"
        );
        assert!(
            (hw.level() - val).abs() < 1.0,
            "level should converge to {}, got {}",
            val,
            hw.level()
        );
        assert!(
            hw.trend().abs() < 1.0,
            "trend should converge to 0, got {}",
            hw.trend()
        );
    }

    #[test]
    fn linear_trend_captured() {
        let mut hw = HoltWinters::new(default_config(4));

        // Feed y = 2*t
        for t in 0..200 {
            hw.train_one(2.0 * t as f64);
        }

        assert!(hw.is_initialized());
        assert!(
            hw.trend() > 0.0,
            "trend should be positive for increasing series, got {}",
            hw.trend()
        );
    }

    #[test]
    fn additive_seasonal_captured() {
        let period = 12;
        let config = HoltWintersConfig::builder(period)
            .alpha(0.3)
            .beta(0.1)
            .gamma(0.3)
            .build()
            .unwrap();
        let mut hw = HoltWinters::new(config);

        // Feed y = 100 + 10*sin(2*pi*t/period)
        for t in 0..120 {
            let y = 100.0 + 10.0 * (2.0 * PI * t as f64 / period as f64).sin();
            hw.train_one(y);
        }

        assert!(hw.is_initialized());

        // Seasonal factors should be nonzero (not all zero)
        let factors = hw.seasonal_factors();
        let has_nonzero = factors.iter().any(|s| s.abs() > EPS);
        assert!(
            has_nonzero,
            "additive seasonal factors should be nonzero, got {:?}",
            factors
        );
    }

    #[test]
    fn multiplicative_seasonal_captured() {
        let period = 12;
        let config = HoltWintersConfig::builder(period)
            .alpha(0.3)
            .beta(0.1)
            .gamma(0.3)
            .seasonality(Seasonality::Multiplicative)
            .build()
            .unwrap();
        let mut hw = HoltWinters::new(config);

        // Feed y = 100 * (1 + 0.1*sin(2*pi*t/period))
        for t in 0..120 {
            let y = 100.0 * (1.0 + 0.1 * (2.0 * PI * t as f64 / period as f64).sin());
            hw.train_one(y);
        }

        assert!(hw.is_initialized());

        // Multiplicative factors should not all be 1.0
        let factors = hw.seasonal_factors();
        let has_deviation = factors.iter().any(|s| (s - 1.0).abs() > EPS);
        assert!(
            has_deviation,
            "multiplicative seasonal factors should deviate from 1.0, got {:?}",
            factors
        );
    }

    #[test]
    fn forecast_returns_correct_length() {
        let mut hw = HoltWinters::new(default_config(4));

        // Before init, forecast should return zeros of correct length
        let f0 = hw.forecast(5);
        assert_eq!(f0.len(), 5, "forecast length should match horizon");

        // Feed enough to init
        for t in 0..20 {
            hw.train_one(100.0 + (t % 4) as f64 * 10.0);
        }

        let f1 = hw.forecast(10);
        assert_eq!(f1.len(), 10, "forecast length should match horizon");

        let f_empty = hw.forecast(0);
        assert_eq!(f_empty.len(), 0, "forecast(0) should return empty vec");
    }

    #[test]
    fn forecast_uses_seasonal() {
        let period = 4;
        let config = HoltWintersConfig::builder(period)
            .alpha(0.3)
            .beta(0.01)
            .gamma(0.3)
            .build()
            .unwrap();
        let mut hw = HoltWinters::new(config);

        // Feed distinct seasonal pattern
        let pattern = [10.0, 20.0, 30.0, 15.0];
        for cycle in 0..50 {
            for &v in &pattern {
                hw.train_one(100.0 + v + cycle as f64 * 0.1);
            }
        }

        // Forecast one full period
        let fc = hw.forecast(period);
        assert_eq!(fc.len(), period);

        // Forecasted values should not all be the same (seasonality present)
        let all_same = fc.windows(2).all(|w| (w[0] - w[1]).abs() < EPS);
        assert!(!all_same, "forecast should show periodicity, got {:?}", fc);
    }

    #[test]
    fn initialization_buffers_first_period() {
        let period = 7;
        let mut hw = HoltWinters::new(default_config(period));

        // Feed period-1 samples -- should not be initialized yet
        for t in 0..period - 1 {
            hw.train_one(t as f64);
            assert!(
                !hw.is_initialized(),
                "should not be initialized after {} samples",
                t + 1
            );
        }

        // One more to complete the period
        hw.train_one((period - 1) as f64);
        assert!(
            hw.is_initialized(),
            "should be initialized after {} samples",
            period
        );
    }

    #[test]
    fn streaming_learner_trait() {
        let config = default_config(4);
        let mut hw = HoltWinters::new(config);

        // Use through the StreamingLearner interface
        let learner: &mut dyn StreamingLearner = &mut hw;

        // Train
        for t in 0..20 {
            learner.train_one(&[], 100.0 + (t % 4) as f64 * 10.0, 1.0);
        }

        assert_eq!(learner.n_samples_seen(), 20);

        // Predict (features ignored)
        let pred = learner.predict(&[]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
        assert!(
            pred > 0.0,
            "prediction should be positive for positive series, got {}",
            pred
        );

        // Reset
        learner.reset();
        assert_eq!(learner.n_samples_seen(), 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut hw = HoltWinters::new(default_config(4));

        // Train
        for t in 0..20 {
            hw.train_one(50.0 + t as f64);
        }

        assert!(hw.is_initialized());
        assert!(hw.n_samples_seen() > 0);

        // Reset
        hw.reset();

        assert!(
            !hw.is_initialized(),
            "should not be initialized after reset"
        );
        assert_eq!(hw.n_samples_seen(), 0, "n_samples should be 0 after reset");
        assert_eq!(hw.level(), 0.0, "level should be 0 after reset");
        assert_eq!(hw.trend(), 0.0, "trend should be 0 after reset");

        // Should be able to retrain
        for t in 0..10 {
            hw.train_one(t as f64 * 5.0);
        }
        assert!(hw.is_initialized());
    }

    #[test]
    fn config_validates() {
        // Valid config
        let ok = HoltWintersConfig::builder(4)
            .alpha(0.5)
            .beta(0.5)
            .gamma(0.5)
            .build();
        assert!(ok.is_ok(), "valid config should succeed");

        // Alpha out of range
        let err = HoltWintersConfig::builder(4).alpha(0.0).build();
        assert!(err.is_err(), "alpha=0 should fail");

        let err = HoltWintersConfig::builder(4).alpha(1.0).build();
        assert!(err.is_err(), "alpha=1 should fail");

        let err = HoltWintersConfig::builder(4).alpha(-0.1).build();
        assert!(err.is_err(), "alpha<0 should fail");

        let err = HoltWintersConfig::builder(4).alpha(1.5).build();
        assert!(err.is_err(), "alpha>1 should fail");

        // Beta out of range
        let err = HoltWintersConfig::builder(4).beta(0.0).build();
        assert!(err.is_err(), "beta=0 should fail");

        // Gamma out of range
        let err = HoltWintersConfig::builder(4).gamma(0.0).build();
        assert!(err.is_err(), "gamma=0 should fail");

        // Period too small
        let err = HoltWintersConfig::builder(1).build();
        assert!(err.is_err(), "period=1 should fail");

        let err = HoltWintersConfig::builder(0).build();
        assert!(err.is_err(), "period=0 should fail");
    }
}
