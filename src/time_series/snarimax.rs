//! Streaming Non-linear ARIMA with eXogenous inputs (SNARIMAX).
//!
//! A streaming time series model combining autoregressive (AR), moving average
//! (MA), seasonal AR/MA, and exogenous input components. All parameters are
//! updated incrementally via online stochastic gradient descent, requiring O(1)
//! memory per model regardless of the length of the observed time series.
//!
//! # Model
//!
//! The prediction at time *t* is:
//!
//! ```text
//! y_hat = intercept
//!       + sum_{i=1}^{p} phi_i * y_{t-i}         (AR)
//!       + sum_{j=1}^{q} theta_j * e_{t-j}        (MA)
//!       + sum_{k=1}^{sp} Phi_k * y_{t-k*s}       (seasonal AR)
//!       + sum_{l=1}^{sq} Theta_l * e_{t-l*s}     (seasonal MA)
//!       + sum_{m=1}^{M} beta_m * x_m              (exogenous)
//! ```
//!
//! where `e_{t} = y_{t} - y_hat_{t}` is the prediction error at time *t*, `s`
//! is the seasonal period, and `x_m` are external feature inputs.
//!
//! # SGD Updates
//!
//! After observing the true value `y_t`, the error `e_t = y_t - y_hat_t` is
//! computed and each coefficient is updated by gradient descent on the squared
//! error loss:
//!
//! ```text
//! phi_i   -= lr * (-2 * e_t * y_{t-i})
//! theta_j -= lr * (-2 * e_t * e_{t-j})
//! ```
//!
//! and analogously for seasonal and exogenous coefficients.
//!
//! # Buffer Management
//!
//! Past values and errors are stored in fixed-capacity circular buffers. The
//! buffer capacity is `max(p, q, sp * s, sq * s)`, and lags are accessed via
//! modular indexing: `buffer[(pos - lag) % capacity]`.
//!
//! # Usage
//!
//! ```
//! use irithyll::time_series::snarimax::{SNARIMAXConfig, SNARIMAX};
//!
//! let config = SNARIMAXConfig::builder()
//!     .p(2)
//!     .q(1)
//!     .learning_rate(0.001)
//!     .build();
//!
//! let mut model = SNARIMAX::new(config);
//!
//! // Stream observations one at a time
//! for t in 0..100 {
//!     let y = (t as f64) * 0.5;
//!     model.train_one(y, &[]);
//! }
//!
//! let forecast = model.forecast(5);
//! assert_eq!(forecast.len(), 5);
//! ```

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// SNARIMAXConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`SNARIMAX`] model.
///
/// Use [`SNARIMAXConfig::builder()`] to construct with fluent setter methods.
/// All parameters have sensible defaults (AR(1) with learning rate 0.01).
///
/// # Fields
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `p` | 1 | AR order (number of lagged values) |
/// | `q` | 0 | MA order (number of lagged errors) |
/// | `seasonal_period` | 0 | Seasonal period (0 = no seasonality) |
/// | `sp` | 0 | Seasonal AR order |
/// | `sq` | 0 | Seasonal MA order |
/// | `n_exogenous` | 0 | Number of exogenous features |
/// | `learning_rate` | 0.01 | Step size for SGD updates |
#[derive(Debug, Clone)]
pub struct SNARIMAXConfig {
    /// AR order (number of lagged values).
    pub p: usize,
    /// MA order (number of lagged errors).
    pub q: usize,
    /// Seasonal period (0 = no seasonality).
    pub seasonal_period: usize,
    /// Seasonal AR order.
    pub sp: usize,
    /// Seasonal MA order.
    pub sq: usize,
    /// Number of exogenous features (0 = none).
    pub n_exogenous: usize,
    /// Learning rate for SGD updates.
    pub learning_rate: f64,
}

impl SNARIMAXConfig {
    /// Create a new builder with default parameters.
    ///
    /// Defaults: `p=1, q=0, seasonal_period=0, sp=0, sq=0, n_exogenous=0, lr=0.01`.
    ///
    /// # Examples
    ///
    /// ```
    /// use irithyll::time_series::snarimax::SNARIMAXConfig;
    ///
    /// let config = SNARIMAXConfig::builder()
    ///     .p(3)
    ///     .q(2)
    ///     .learning_rate(0.005)
    ///     .build();
    /// assert_eq!(config.p, 3);
    /// assert_eq!(config.q, 2);
    /// ```
    pub fn builder() -> SNARIMAXConfigBuilder {
        SNARIMAXConfigBuilder {
            p: 1,
            q: 0,
            seasonal_period: 0,
            sp: 0,
            sq: 0,
            n_exogenous: 0,
            learning_rate: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// SNARIMAXConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`SNARIMAXConfig`].
///
/// Constructed via [`SNARIMAXConfig::builder()`]. All setters return `self` for
/// method chaining. Call [`.build()`](SNARIMAXConfigBuilder::build) to produce
/// the final configuration.
#[derive(Debug, Clone)]
pub struct SNARIMAXConfigBuilder {
    p: usize,
    q: usize,
    seasonal_period: usize,
    sp: usize,
    sq: usize,
    n_exogenous: usize,
    learning_rate: f64,
}

impl SNARIMAXConfigBuilder {
    /// Set the AR order (number of lagged values).
    pub fn p(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Set the MA order (number of lagged errors).
    pub fn q(mut self, q: usize) -> Self {
        self.q = q;
        self
    }

    /// Set the seasonal period (0 = no seasonality).
    pub fn seasonal_period(mut self, s: usize) -> Self {
        self.seasonal_period = s;
        self
    }

    /// Set the seasonal AR order.
    pub fn sp(mut self, sp: usize) -> Self {
        self.sp = sp;
        self
    }

    /// Set the seasonal MA order.
    pub fn sq(mut self, sq: usize) -> Self {
        self.sq = sq;
        self
    }

    /// Set the number of exogenous features.
    pub fn n_exogenous(mut self, n: usize) -> Self {
        self.n_exogenous = n;
        self
    }

    /// Set the learning rate for SGD updates.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Build the final [`SNARIMAXConfig`].
    pub fn build(self) -> SNARIMAXConfig {
        SNARIMAXConfig {
            p: self.p,
            q: self.q,
            seasonal_period: self.seasonal_period,
            sp: self.sp,
            sq: self.sq,
            n_exogenous: self.n_exogenous,
            learning_rate: self.learning_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// SNARIMAXCoefficients
// ---------------------------------------------------------------------------

/// Snapshot of all learned coefficients in a [`SNARIMAX`] model.
///
/// Returned by [`SNARIMAX::coefficients()`]. All vectors are cloned from the
/// model's internal state -- mutating them does not affect the model.
#[derive(Debug, Clone)]
pub struct SNARIMAXCoefficients {
    /// Intercept (bias) term.
    pub intercept: f64,
    /// AR coefficients (phi_1 .. phi_p).
    pub ar: Vec<f64>,
    /// MA coefficients (theta_1 .. theta_q).
    pub ma: Vec<f64>,
    /// Seasonal AR coefficients (Phi_1 .. Phi_sp).
    pub seasonal_ar: Vec<f64>,
    /// Seasonal MA coefficients (Theta_1 .. Theta_sq).
    pub seasonal_ma: Vec<f64>,
    /// Exogenous input coefficients (beta_1 .. beta_m).
    pub exogenous: Vec<f64>,
}

// ---------------------------------------------------------------------------
// SNARIMAX
// ---------------------------------------------------------------------------

/// Streaming Non-linear ARIMA with eXogenous inputs.
///
/// Combines autoregressive (AR), moving average (MA), seasonal AR/MA, and
/// exogenous input components into a single streaming time series model.
/// Parameters are updated via online SGD after each observation.
///
/// Past values and errors are stored in fixed-capacity circular buffers,
/// giving O(1) memory usage regardless of series length.
///
/// # Examples
///
/// ```
/// use irithyll::time_series::snarimax::{SNARIMAXConfig, SNARIMAX};
///
/// let config = SNARIMAXConfig::builder()
///     .p(1)
///     .learning_rate(0.01)
///     .build();
///
/// let mut model = SNARIMAX::new(config);
/// model.train_one(1.0, &[]);
/// model.train_one(2.0, &[]);
///
/// let pred = model.predict_one(&[]);
/// assert!(pred.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct SNARIMAX {
    config: SNARIMAXConfig,
    // Coefficients
    intercept: f64,
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    sar_coeffs: Vec<f64>,
    sma_coeffs: Vec<f64>,
    exo_coeffs: Vec<f64>,
    // Circular lag buffers
    y_buffer: Vec<f64>,
    e_buffer: Vec<f64>,
    buffer_pos: usize,
    n_samples: u64,
}

impl SNARIMAX {
    /// Create a new SNARIMAX model from the given configuration.
    ///
    /// All coefficients are initialized to zero. Circular buffers are allocated
    /// with capacity equal to the maximum lag needed by any component.
    ///
    /// # Arguments
    ///
    /// * `config` -- model configuration specifying AR/MA orders, seasonality,
    ///   exogenous feature count, and learning rate
    ///
    /// # Examples
    ///
    /// ```
    /// use irithyll::time_series::snarimax::{SNARIMAXConfig, SNARIMAX};
    ///
    /// let config = SNARIMAXConfig::builder().p(3).q(1).build();
    /// let model = SNARIMAX::new(config);
    /// assert_eq!(model.n_samples_seen(), 0);
    /// ```
    pub fn new(config: SNARIMAXConfig) -> Self {
        let buf_cap = Self::compute_buffer_capacity(&config);

        Self {
            ar_coeffs: vec![0.0; config.p],
            ma_coeffs: vec![0.0; config.q],
            sar_coeffs: vec![0.0; config.sp],
            sma_coeffs: vec![0.0; config.sq],
            exo_coeffs: vec![0.0; config.n_exogenous],
            intercept: 0.0,
            y_buffer: vec![0.0; buf_cap],
            e_buffer: vec![0.0; buf_cap],
            buffer_pos: 0,
            n_samples: 0,
            config,
        }
    }

    /// Train the model on a single observation.
    ///
    /// Computes the prediction for the current state, calculates the error
    /// `e_t = y - y_hat`, updates all coefficients via SGD, and pushes `y`
    /// and `e_t` into the circular buffers.
    ///
    /// # Arguments
    ///
    /// * `y` -- the observed value at time *t*
    /// * `exogenous` -- slice of exogenous feature values (length must match
    ///   `config.n_exogenous`, or be empty if `n_exogenous == 0`)
    ///
    /// # Panics
    ///
    /// Does not panic. If `exogenous` is shorter than `n_exogenous`, missing
    /// features are treated as zero. If longer, extra values are ignored.
    pub fn train_one(&mut self, y: f64, exogenous: &[f64]) {
        self.train_impl(y, exogenous);
    }

    /// Predict the next value using the current model state.
    ///
    /// Uses the values and errors currently in the circular buffers along with
    /// the provided exogenous features to compute a one-step-ahead prediction.
    ///
    /// # Arguments
    ///
    /// * `exogenous` -- slice of exogenous feature values (length should match
    ///   `config.n_exogenous`)
    pub fn predict_one(&self, exogenous: &[f64]) -> f64 {
        self.predict_from_buffers(exogenous)
    }

    /// Multi-step forecast without exogenous inputs.
    ///
    /// Generates `horizon` predictions into the future by feeding each
    /// predicted value back as the next lag input. Since future exogenous
    /// values are unknown, they are treated as zero. Future errors are also
    /// treated as zero (the model uses its own predictions as truth for
    /// recursive forecasting).
    ///
    /// Creates a temporary copy of the internal buffers to avoid mutating
    /// model state.
    ///
    /// # Arguments
    ///
    /// * `horizon` -- number of future steps to forecast
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `horizon` containing the forecasted values.
    ///
    /// # Examples
    ///
    /// ```
    /// use irithyll::time_series::snarimax::{SNARIMAXConfig, SNARIMAX};
    ///
    /// let config = SNARIMAXConfig::builder().p(1).build();
    /// let mut model = SNARIMAX::new(config);
    ///
    /// for i in 0..50 {
    ///     model.train_one(i as f64, &[]);
    /// }
    ///
    /// let forecast = model.forecast(10);
    /// assert_eq!(forecast.len(), 10);
    /// ```
    pub fn forecast(&self, horizon: usize) -> Vec<f64> {
        let mut results = Vec::with_capacity(horizon);

        // Clone buffers for temporary state
        let mut y_buf = self.y_buffer.clone();
        let mut e_buf = self.e_buffer.clone();
        let mut pos = self.buffer_pos;
        let cap = y_buf.len();

        if cap == 0 {
            // No lags configured -- prediction is just the intercept
            let pred = self.intercept;
            return vec![pred; horizon];
        }

        for _ in 0..horizon {
            let mut y_hat = self.intercept;

            // AR component
            for i in 0..self.config.p {
                let lag = i + 1;
                let idx = (pos + cap - lag) % cap;
                y_hat += self.ar_coeffs[i] * y_buf[idx];
            }

            // MA component
            for j in 0..self.config.q {
                let lag = j + 1;
                let idx = (pos + cap - lag) % cap;
                y_hat += self.ma_coeffs[j] * e_buf[idx];
            }

            // Seasonal AR component
            if self.config.seasonal_period > 0 {
                for k in 0..self.config.sp {
                    let lag = (k + 1) * self.config.seasonal_period;
                    if lag <= cap {
                        let idx = (pos + cap - lag) % cap;
                        y_hat += self.sar_coeffs[k] * y_buf[idx];
                    }
                }

                // Seasonal MA component
                for l in 0..self.config.sq {
                    let lag = (l + 1) * self.config.seasonal_period;
                    if lag <= cap {
                        let idx = (pos + cap - lag) % cap;
                        y_hat += self.sma_coeffs[l] * e_buf[idx];
                    }
                }
            }

            // Exogenous: zero (unknown future values)
            // No contribution to y_hat

            // Push predicted value into temp buffer (error = 0 for forecasts)
            y_buf[pos] = y_hat;
            e_buf[pos] = 0.0;
            pos = (pos + 1) % cap;

            results.push(y_hat);
        }

        results
    }

    /// Total number of observations trained on since creation or last reset.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Return a snapshot of all learned coefficients.
    ///
    /// The returned [`SNARIMAXCoefficients`] contains cloned copies of the
    /// intercept, AR, MA, seasonal AR/MA, and exogenous coefficient vectors.
    pub fn coefficients(&self) -> SNARIMAXCoefficients {
        SNARIMAXCoefficients {
            intercept: self.intercept,
            ar: self.ar_coeffs.clone(),
            ma: self.ma_coeffs.clone(),
            seasonal_ar: self.sar_coeffs.clone(),
            seasonal_ma: self.sma_coeffs.clone(),
            exogenous: self.exo_coeffs.clone(),
        }
    }

    /// Reset the model to its initial (untrained) state.
    ///
    /// All coefficients are set to zero and circular buffers are cleared.
    /// After calling `reset()`, the model behaves identically to a freshly
    /// constructed instance with the same configuration.
    pub fn reset(&mut self) {
        self.reset_impl();
    }

    /// Immutable access to the model configuration.
    #[inline]
    pub fn config(&self) -> &SNARIMAXConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute the required buffer capacity from a configuration.
    ///
    /// Capacity = max(p, q, sp * seasonal_period, sq * seasonal_period).
    /// Returns at least 1 to avoid zero-size buffer edge cases.
    fn compute_buffer_capacity(config: &SNARIMAXConfig) -> usize {
        let s = config.seasonal_period;
        let mut cap = config.p;
        if config.q > cap {
            cap = config.q;
        }
        if s > 0 {
            let sar_max = config.sp * s;
            let sma_max = config.sq * s;
            if sar_max > cap {
                cap = sar_max;
            }
            if sma_max > cap {
                cap = sma_max;
            }
        }
        // Minimum capacity of 1 to avoid modulo-by-zero
        if cap == 0 {
            1
        } else {
            cap
        }
    }

    /// Compute prediction from current buffer state.
    fn predict_from_buffers(&self, exogenous: &[f64]) -> f64 {
        let mut y_hat = self.intercept;

        // AR component: sum_{i=1}^{p} phi_i * y_{t-i}
        for i in 0..self.config.p {
            let lag = i + 1;
            y_hat += self.ar_coeffs[i] * self.get_y_lag(lag);
        }

        // MA component: sum_{j=1}^{q} theta_j * e_{t-j}
        for j in 0..self.config.q {
            let lag = j + 1;
            y_hat += self.ma_coeffs[j] * self.get_e_lag(lag);
        }

        // Seasonal AR: sum_{k=1}^{sp} Phi_k * y_{t-k*s}
        if self.config.seasonal_period > 0 {
            let cap = self.y_buffer.len();

            for k in 0..self.config.sp {
                let lag = (k + 1) * self.config.seasonal_period;
                if lag <= cap {
                    y_hat += self.sar_coeffs[k] * self.get_y_lag(lag);
                }
            }

            // Seasonal MA: sum_{l=1}^{sq} Theta_l * e_{t-l*s}
            for l in 0..self.config.sq {
                let lag = (l + 1) * self.config.seasonal_period;
                if lag <= cap {
                    y_hat += self.sma_coeffs[l] * self.get_e_lag(lag);
                }
            }
        }

        // Exogenous: sum_{m=1}^{M} beta_m * x_m
        for m in 0..self.config.n_exogenous {
            let x = if m < exogenous.len() {
                exogenous[m]
            } else {
                0.0
            };
            y_hat += self.exo_coeffs[m] * x;
        }

        y_hat
    }

    /// Get a lagged value from the y buffer.
    ///
    /// `lag` is 1-indexed: lag=1 means the most recently pushed value.
    #[inline]
    fn get_y_lag(&self, lag: usize) -> f64 {
        let cap = self.y_buffer.len();
        let idx = (self.buffer_pos + cap - lag) % cap;
        self.y_buffer[idx]
    }

    /// Get a lagged value from the error buffer.
    ///
    /// `lag` is 1-indexed: lag=1 means the most recently pushed error.
    #[inline]
    fn get_e_lag(&self, lag: usize) -> f64 {
        let cap = self.e_buffer.len();
        let idx = (self.buffer_pos + cap - lag) % cap;
        self.e_buffer[idx]
    }

    /// Push a value and error into the circular buffers, advancing the position.
    #[inline]
    fn push_to_buffers(&mut self, y: f64, error: f64) {
        let cap = self.y_buffer.len();
        let pos = self.buffer_pos;
        self.y_buffer[pos] = y;
        self.e_buffer[pos] = error;
        self.buffer_pos = (pos + 1) % cap;
    }

    /// Core training logic -- called by both `train_one` (inherent) and the
    /// `StreamingLearner` trait impl to avoid method name ambiguity.
    fn train_impl(&mut self, y: f64, exogenous: &[f64]) {
        // Step 1: Predict with current state
        let y_hat = self.predict_from_buffers(exogenous);

        // Step 2: Compute error with gradient clipping to prevent divergence
        let raw_error = y - y_hat;
        let error = raw_error.clamp(-1e6, 1e6);
        let lr = self.config.learning_rate;

        // Step 3: SGD coefficient updates
        // gradient of 0.5 * (y - y_hat)^2 w.r.t. each coefficient is -error * (input)
        // update rule: coeff += lr * error * input  (equivalent to coeff -= lr * (-error * input))

        // Intercept
        self.intercept += lr * error;

        // AR coefficients: gradient w.r.t. phi_i is -error * y_{t-i}
        for i in 0..self.config.p {
            let lag = i + 1;
            let y_lag = self.get_y_lag(lag);
            self.ar_coeffs[i] += lr * error * y_lag;
        }

        // MA coefficients: gradient w.r.t. theta_j is -error * e_{t-j}
        for j in 0..self.config.q {
            let lag = j + 1;
            let e_lag = self.get_e_lag(lag);
            self.ma_coeffs[j] += lr * error * e_lag;
        }

        // Seasonal AR coefficients: gradient w.r.t. Phi_k is -error * y_{t-k*s}
        if self.config.seasonal_period > 0 {
            for k in 0..self.config.sp {
                let lag = (k + 1) * self.config.seasonal_period;
                let y_lag = self.get_y_lag(lag);
                self.sar_coeffs[k] += lr * error * y_lag;
            }

            // Seasonal MA coefficients: gradient w.r.t. Theta_l is -error * e_{t-l*s}
            for l in 0..self.config.sq {
                let lag = (l + 1) * self.config.seasonal_period;
                let e_lag = self.get_e_lag(lag);
                self.sma_coeffs[l] += lr * error * e_lag;
            }
        }

        // Exogenous coefficients: gradient w.r.t. beta_m is -error * x_m
        for m in 0..self.config.n_exogenous {
            let x = if m < exogenous.len() {
                exogenous[m]
            } else {
                0.0
            };
            self.exo_coeffs[m] += lr * error * x;
        }

        // Step 4: Push y and error into circular buffers
        self.push_to_buffers(y, error);

        self.n_samples += 1;
    }

    /// Core reset logic -- called by both `reset` (inherent) and the
    /// `StreamingLearner` trait impl to avoid method name ambiguity.
    fn reset_impl(&mut self) {
        self.intercept = 0.0;
        self.ar_coeffs.iter_mut().for_each(|c| *c = 0.0);
        self.ma_coeffs.iter_mut().for_each(|c| *c = 0.0);
        self.sar_coeffs.iter_mut().for_each(|c| *c = 0.0);
        self.sma_coeffs.iter_mut().for_each(|c| *c = 0.0);
        self.exo_coeffs.iter_mut().for_each(|c| *c = 0.0);
        self.y_buffer.iter_mut().for_each(|v| *v = 0.0);
        self.e_buffer.iter_mut().for_each(|v| *v = 0.0);
        self.buffer_pos = 0;
        self.n_samples = 0;
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

/// [`StreamingLearner`] implementation for SNARIMAX.
///
/// The `features` parameter maps to exogenous inputs, and `target` maps to
/// the observed time series value. Sample weight is accepted but currently
/// unused (all observations are weighted equally in the SGD update).
impl StreamingLearner for SNARIMAX {
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        self.train_impl(target, features);
    }

    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        self.predict_one(features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.reset_impl();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 PRNG for reproducible noise generation.
    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            // Ensure non-zero state
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        /// Generate the next u64 value.
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        /// Generate a f64 in [-amplitude, +amplitude].
        fn next_f64(&mut self, amplitude: f64) -> f64 {
            let raw = self.next_u64();
            // Map to [0, 1) then to [-amplitude, amplitude)
            let unit = (raw as f64) / (u64::MAX as f64);
            (unit * 2.0 - 1.0) * amplitude
        }
    }

    #[test]
    fn config_builder_defaults() {
        let config = SNARIMAXConfig::builder().build();
        assert_eq!(config.p, 1, "default p should be 1");
        assert_eq!(config.q, 0, "default q should be 0");
        assert_eq!(
            config.seasonal_period, 0,
            "default seasonal_period should be 0"
        );
        assert_eq!(config.sp, 0, "default sp should be 0");
        assert_eq!(config.sq, 0, "default sq should be 0");
        assert_eq!(config.n_exogenous, 0, "default n_exogenous should be 0");
        assert!(
            (config.learning_rate - 0.01).abs() < 1e-12,
            "default learning_rate should be 0.01, got {}",
            config.learning_rate,
        );
    }

    #[test]
    fn simple_ar1_converges() {
        // Generate y_t = 0.8 * y_{t-1} + noise
        let true_phi = 0.8;
        let mut rng = Xorshift64::new(42);

        let config = SNARIMAXConfig::builder()
            .p(1)
            .q(0)
            .learning_rate(0.05)
            .build();

        let mut model = SNARIMAX::new(config);
        let mut y_prev = 0.0;

        for _ in 0..10_000 {
            let noise = rng.next_f64(0.1);
            let y = true_phi * y_prev + noise;
            model.train_one(y, &[]);
            y_prev = y;
        }

        let coeffs = model.coefficients();
        let learned_phi = coeffs.ar[0];
        // With gradient clipping + higher LR, verify convergence toward true phi
        assert!(
            learned_phi > 0.3 && learned_phi.is_finite(),
            "AR(1) coefficient should converge toward {}, got {}",
            true_phi,
            learned_phi,
        );
    }

    #[test]
    fn predict_one_uses_lags() {
        let config = SNARIMAXConfig::builder()
            .p(2)
            .q(0)
            .learning_rate(0.01)
            .build();

        let mut model = SNARIMAX::new(config);

        // Train on a simple sequence so coefficients become non-zero
        for i in 0..100 {
            let y = (i as f64) * 0.5;
            model.train_one(y, &[]);
        }

        // After training, predict_one should produce a finite, non-zero value
        // that depends on the buffered lags
        let pred = model.predict_one(&[]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
        assert!(
            pred.abs() > 1e-6,
            "prediction should be non-zero after training, got {}",
            pred,
        );

        // A second prediction (without new training) should return the same value
        let pred2 = model.predict_one(&[]);
        assert!(
            (pred - pred2).abs() < 1e-12,
            "consecutive predict_one calls should return the same value: {} vs {}",
            pred,
            pred2,
        );
    }

    #[test]
    fn forecast_multi_step() {
        let config = SNARIMAXConfig::builder()
            .p(2)
            .q(1)
            .learning_rate(0.001)
            .build();

        let mut model = SNARIMAX::new(config);

        // Train on a trend — gradient clipping prevents divergence
        for i in 0..200 {
            model.train_one(i as f64, &[]);
        }

        let horizon = 10;
        let forecast = model.forecast(horizon);
        assert_eq!(
            forecast.len(),
            horizon,
            "forecast should return exactly {} values, got {}",
            horizon,
            forecast.len(),
        );

        // All forecasted values should be finite
        for (i, &val) in forecast.iter().enumerate() {
            assert!(
                val.is_finite(),
                "forecast[{}] should be finite, got {}",
                i,
                val,
            );
        }

        // Forecasting should not mutate model state
        let n_before = model.n_samples_seen();
        let _ = model.forecast(5);
        assert_eq!(
            model.n_samples_seen(),
            n_before,
            "forecast should not change n_samples_seen",
        );
    }

    #[test]
    fn seasonal_component_works() {
        // Generate data with period=4 seasonality: y_t = seasonal[t % 4] + noise
        let seasonal_pattern = [10.0, 20.0, 30.0, 40.0];
        let period = seasonal_pattern.len();
        let mut rng = Xorshift64::new(123);

        let config = SNARIMAXConfig::builder()
            .p(0)
            .q(0)
            .seasonal_period(period)
            .sp(1)
            .sq(0)
            .learning_rate(0.001)
            .build();

        let mut model = SNARIMAX::new(config);

        // Train on many cycles
        for _cycle in 0..2000 {
            for sp_val in seasonal_pattern.iter().take(period) {
                let noise = rng.next_f64(0.5);
                let y = sp_val + noise;
                model.train_one(y, &[]);
            }
        }

        // The seasonal AR coefficient should be non-zero, indicating the model
        // has learned to use the seasonal lag
        let coeffs = model.coefficients();
        assert!(
            coeffs.seasonal_ar[0].abs() > 0.01,
            "seasonal AR coefficient should be non-zero, got {}",
            coeffs.seasonal_ar[0],
        );
    }

    #[test]
    fn exogenous_input() {
        // y_t = 3.0 * x_t + noise
        let mut rng = Xorshift64::new(999);

        let config = SNARIMAXConfig::builder()
            .p(0)
            .q(0)
            .n_exogenous(1)
            .learning_rate(0.001)
            .build();

        let mut model = SNARIMAX::new(config);

        for _ in 0..5000 {
            let x = rng.next_f64(5.0);
            let noise = rng.next_f64(0.1);
            let y = 3.0 * x + noise;
            model.train_one(y, &[x]);
        }

        let coeffs = model.coefficients();
        assert!(
            coeffs.exogenous[0].abs() > 0.1,
            "exogenous coefficient should be non-zero, got {}",
            coeffs.exogenous[0],
        );
        assert!(
            (coeffs.exogenous[0] - 3.0).abs() < 1.0,
            "exogenous coefficient should converge toward 3.0, got {}",
            coeffs.exogenous[0],
        );
    }

    #[test]
    fn streaming_learner_trait() {
        // Verify SNARIMAX works through the StreamingLearner trait interface.
        let config = SNARIMAXConfig::builder()
            .p(1)
            .n_exogenous(2)
            .learning_rate(0.01)
            .build();

        let model = SNARIMAX::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);

        // Train through trait
        boxed.train(&[1.0, 2.0], 5.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        boxed.train(&[3.0, 4.0], 10.0);
        assert_eq!(boxed.n_samples_seen(), 2);

        // Predict through trait
        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "trait prediction should be finite, got {}",
            pred
        );

        // Reset through trait
        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    #[test]
    fn reset_clears_state() {
        let config = SNARIMAXConfig::builder()
            .p(2)
            .q(1)
            .n_exogenous(1)
            .learning_rate(0.01)
            .build();

        let mut model = SNARIMAX::new(config);

        // Train some samples — gradient clipping keeps this stable
        for i in 0..100 {
            model.train_one(i as f64, &[i as f64 * 0.5]);
        }
        assert_eq!(model.n_samples_seen(), 100);

        // At least some coefficient should have moved from zero
        let coeffs_before = model.coefficients();
        let has_nonzero = coeffs_before.ar.iter().any(|c| c.abs() > 1e-15)
            || coeffs_before.intercept.abs() > 1e-15
            || coeffs_before.exogenous.iter().any(|c| c.abs() > 1e-15);
        assert!(
            has_nonzero,
            "coefficients should be non-zero after training"
        );

        // Reset
        model.reset();
        assert_eq!(
            model.n_samples_seen(),
            0,
            "n_samples should be 0 after reset"
        );

        // All coefficients should be zero
        let coeffs_after = model.coefficients();
        assert!(
            coeffs_after.intercept.abs() < 1e-12,
            "intercept should be zero after reset, got {}",
            coeffs_after.intercept,
        );
        for (i, c) in coeffs_after.ar.iter().enumerate() {
            assert!(
                c.abs() < 1e-12,
                "ar[{}] should be zero after reset, got {}",
                i,
                c,
            );
        }
        for (j, c) in coeffs_after.ma.iter().enumerate() {
            assert!(
                c.abs() < 1e-12,
                "ma[{}] should be zero after reset, got {}",
                j,
                c,
            );
        }
        for (m, c) in coeffs_after.exogenous.iter().enumerate() {
            assert!(
                c.abs() < 1e-12,
                "exo[{}] should be zero after reset, got {}",
                m,
                c,
            );
        }

        // Prediction after reset should be zero (all coefficients and buffers zeroed)
        let pred = model.predict_one(&[0.0]);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be zero, got {}",
            pred,
        );
    }
}
