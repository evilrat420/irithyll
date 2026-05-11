//! Streaming Kolmogorov-Arnold Networks (KAN).
//!
//! Learnable B-spline activation functions on edges instead of fixed
//! activations on nodes. Each edge learns a univariate function via
//! B-spline coefficients, enabling expressive nonlinear regression
//! with sparse per-sample gradient updates.
//!
//! # Architecture
//!
//! ```text
//! x_t -> [Input Norm] -> [KAN Layer 1] -> [KAN Layer 2] -> ... -> y_hat_t
//! ```
//!
//! # Recommended Configuration
//!
//! For streaming regression, use shallow architectures with higher learning rates:
//! ```
//! use irithyll::kan::KANConfig;
//!
//! let config = KANConfig::builder()
//!     .layer_sizes(vec![4, 20, 1])  // shallow: 1 hidden layer
//!     .grid_size(8)
//!     .learning_rate(0.1)   // higher than MLP — B-spline sparse updates need it
//!     .build()
//!     .unwrap();
//! ```
//! Deep architectures (3+ layers) can cause gradient instability in streaming mode.
//! B-spline locality ensures sparse updates don't interfere (Hoang et al., 2026).
//!
//! # References
//!
//! - Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks" ICLR 2025
//! - Hoang et al. (2026) "Ultrafast On-chip Online Learning" -- proves single-sample KAN SGD
//! - Makinde (2026) "T-KAN" -- temporal KAN for LOB forecasting

mod bspline;
mod config;
mod layer;
mod model;

pub use config::{GateMode, KANConfig, KANConfigBuilder};
pub use model::StreamingKAN;
