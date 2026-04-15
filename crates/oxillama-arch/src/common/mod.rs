//! Common layer implementations shared across model architectures.
//!
//! These building blocks (RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, Linear) are used by
//! multiple model families and are implemented once here.

pub mod gelu;
pub mod layer_norm;
pub mod linear;
pub mod moe;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
