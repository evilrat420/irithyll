//! Core types and inference engine for irithyll streaming ML.
//!
//! `irithyll-core` is the `#![no_std]` foundation shared by the full irithyll
//! crate and embedded targets. It provides loss functions, the streaming
//! [`Observation`] trait, a compact 12-byte packed node format, and branch-free
//! ensemble traversal for deploying trained models on bare metal
//! (Cortex-M0+, 32 KB flash).
//!
//! The crate has a hard dependency boundary: only `libm` is mandatory. All
//! dynamic-allocation paths (histogram binning, tree construction, drift
//! detectors, neural architectures) gate on the `alloc` feature.
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `alloc` | Enables dynamic-allocation types: trees, ensembles, drift, neural |
//! | `std` | Enables `alloc` plus standard I/O (required for full `irithyll` crate) |
//! | `serde` | Derives `Serialize`/`Deserialize` on config and snapshot types |
//! | `kmeans-binning` | K-means histogram binning strategy (requires `alloc`) |
//! | `parallel` | Rayon-parallel tree training (requires `alloc`) |
//! | `simd` | AVX2 histogram accumulation acceleration (requires `std`) |
//! | `simd-avx2` | Explicit AVX2 SIMD intrinsics (requires `std`) |
//! | `embedded-bench` | Cortex-M semihosting bench helpers |
//! | `iai-bench` | iai-callgrind regression bench source |
//!
//! # Embedded Inference
//!
//! Train a model with the full `irithyll` crate, then export:
//!
//! ```ignore
//! use irithyll_core::{EnsembleView, FormatError};
//!
//! // packed_bytes is a &[u8] from irithyll::export_embedded()
//! let packed_bytes: &[u8] = &[];
//! let view = EnsembleView::from_bytes(packed_bytes)?;
//! let prediction = view.predict(&[1.0f32, 2.0, 3.0]);
//! # Ok::<(), FormatError>(())
//! ```
//!
//! The packed format stores each node in 12 bytes (5 nodes per cache line).
//! Traversal is branch-free: child selection uses `cmov`/`csel`-equivalent
//! conditionals that avoid pipeline stalls on Cortex-M.

#![no_std]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Note: irithyll-core contains legitimate `unsafe impl Send/Sync` for tree types
// and optional AVX2 SIMD intrinsics. The main irithyll crate forbids unsafe code.

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
pub mod rng;
pub mod sample;
pub mod simd;
pub mod streaming_primitives;
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

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod attention;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod continual;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod lstm;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod turbo_quant;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod mgrade;

// Convenience re-exports -- inference
pub use error::FormatError;
pub use packed::{EnsembleHeader, PackedNode, TreeEntry};
pub use packed_i16::{PackedNodeI16, QuantizedEnsembleHeader};
pub use view::EnsembleView;
pub use view_i16::QuantizedEnsembleView;

// Convenience re-exports -- training core types
pub use loss::{Loss, LossType};
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use sample::Sample;
pub use sample::{Observation, SampleRef};

// Convenience re-exports -- drift detection
pub use drift::DriftSignal;
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use drift::{DriftDetector, DriftDetectorState};

// Convenience re-exports -- config/error (requires alloc)
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use error::{ConfigError, IrithyllError, Result};
