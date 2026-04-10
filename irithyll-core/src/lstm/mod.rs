//! sLSTM (stabilized LSTM) cell with exponential gating and log-domain stabilization.
//!
//! Based on Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory".
//!
//! sLSTM differs from standard LSTM in three key ways:
//!
//! 1. **Exponential gating** -- forget and input gates use `exp()` instead of
//!    `sigmoid()`, allowing gate values > 1.
//! 2. **Log-domain stabilization** -- a stabilizer `m_t` prevents numerical
//!    overflow when composing exponential gates across time.
//! 3. **Normalizer state** -- tracks cumulative gate products for output
//!    normalization, keeping hidden activations bounded.

pub mod slstm_cell;

pub use slstm_cell::SLSTMCell;
