//! mGRADE (Minimal Recurrent Gating with Delay Convolutions) core cells.
//!
//! Based on arXiv July 2025. Combines:
//!
//! 1. **minGRU** -- minimal gated recurrent unit with no recurrent candidate
//! 2. **Learnable delay convolution** -- 1D depthwise conv with learnable spacings

pub mod delay_conv;
pub mod min_gru;

pub use delay_conv::DelayConv1D;
pub use min_gru::MinGRUCell;
