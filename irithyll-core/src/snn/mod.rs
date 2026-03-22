//! Spiking Neural Networks with online e-prop learning.
//!
//! This module provides LIF (Leaky Integrate-and-Fire) neurons with integer
//! arithmetic (Q1.14 fixed-point), delta spike encoding, and the e-prop
//! three-factor learning rule for online training.
//!
//! # Architecture
//!
//! The SNN processes streaming data through three stages:
//!
//! 1. **Delta encoding** -- raw features are converted to spike trains via
//!    temporal differencing (UP/DOWN spike pairs per feature)
//! 2. **Recurrent spiking layer** -- LIF neurons with e-prop eligibility
//!    traces and feedback alignment for weight updates
//! 3. **Readout** -- non-spiking leaky integrators whose membrane potentials
//!    serve as continuous output predictions
//!
//! # Fixed-Point Arithmetic
//!
//! All neuron computations use Q1.14 fixed-point (i16 where 16384 = 1.0).
//! Multiply-accumulate operations use i32 intermediates to prevent overflow.
//! This enables deployment on integer-only hardware (Cortex-M0+, RISC-V).
//!
//! # Components
//!
//! - [`lif`] -- LIF neuron step and surrogate gradient functions
//! - [`spike_encoding`] -- Delta spike encoder for continuous-to-spike conversion
//! - [`eprop`] -- e-prop three-factor learning rule functions
//! - [`readout`] -- Non-spiking leaky integrator readout neuron
//! - [`network_fixed`] -- Complete SNN combining all components

pub mod eprop;
pub mod lif;
pub mod network_fixed;
pub mod readout;
pub mod spike_encoding;

pub use eprop::{
    compute_learning_signal_fixed, update_eligibility_fixed, update_output_weights_fixed,
    update_pre_trace_fixed, update_weights_fixed,
};
pub use lif::{lif_step, surrogate_gradient_pwl, Q14_HALF, Q14_ONE};
pub use network_fixed::SpikeNetFixed;
pub use readout::ReadoutNeuron;
pub use spike_encoding::DeltaEncoderFixed;
