//! Core types and inference engine for irithyll streaming ML models.
//!
//! `irithyll-core` provides loss functions, observation traits, a compact binary
//! format, and branch-free traversal for deploying trained SGBT models on
//! embedded targets (Cortex-M0+, 32KB flash).
//!
//! # Features
//!
//! - **Loss functions** — squared, logistic, Huber, softmax, expectile, quantile
//! - **Observation trait** — zero-copy training interface with `SampleRef`
//! - **12-byte packed nodes** — 5 nodes per 64-byte cache line
//! - **Zero-copy `EnsembleView`** — constructed from `&[u8]`, no allocation after validation
//! - **Branch-free traversal** — `cmov`/`csel` child selection, no pipeline stalls
//! - **`#![no_std]`** — zero mandatory dependencies, runs on bare metal
//!
//! # Usage
//!
//! ```ignore
//! use irithyll_core::{EnsembleView, FormatError};
//!
//! // Load packed binary (e.g. from flash, file, or network)
//! let packed_bytes: &[u8] = &[/* exported via irithyll::export_embedded() */];
//! let view = EnsembleView::from_bytes(packed_bytes)?;
//! let prediction = view.predict(&[1.0f32, 2.0, 3.0]);
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(all(test, not(feature = "alloc")))]
extern crate alloc;

pub mod drift;
pub mod error;
pub mod loss;
pub mod math;
pub mod packed;
pub mod packed_i16;
pub mod quantize;
pub mod sample;
pub mod traverse;
pub mod traverse_i16;
pub mod view;
pub mod view_i16;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod histogram;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod feature;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod learner;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod tree;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod ensemble;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod reservoir;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod snn;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod ssm;

// Convenience re-exports -- inference
pub use error::FormatError;
pub use packed::{EnsembleHeader, PackedNode, TreeEntry};
pub use packed_i16::{PackedNodeI16, QuantizedEnsembleHeader};
pub use view::EnsembleView;
pub use view_i16::QuantizedEnsembleView;

// Convenience re-exports -- training core types
pub use loss::{Loss, LossType};
#[cfg(feature = "alloc")]
pub use sample::Sample;
pub use sample::{Observation, SampleRef};

// Convenience re-exports -- drift detection
pub use drift::DriftSignal;
#[cfg(feature = "alloc")]
pub use drift::{DriftDetector, DriftDetectorState};

// Convenience re-exports -- config/error (requires alloc)
#[cfg(feature = "alloc")]
pub use error::{ConfigError, IrithyllError, Result};
