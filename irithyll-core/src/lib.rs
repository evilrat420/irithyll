//! Zero-alloc inference engine for irithyll streaming ML models.
//!
//! `irithyll-core` provides a compact binary format and branch-free traversal
//! for deploying trained SGBT models on embedded targets (Cortex-M0+, 32KB flash).
//!
//! # Features
//!
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

#[cfg(test)]
extern crate alloc;

pub mod error;
pub mod packed;
pub mod quantize;
pub mod traverse;
pub mod view;

// Convenience re-exports
pub use error::FormatError;
pub use packed::{EnsembleHeader, PackedNode, TreeEntry};
pub use view::EnsembleView;
