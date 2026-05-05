//! Phi-3.5-MoE model architecture implementation.
//!
//! Phi-3.5-MoE uses Phi-3 style attention (fused QKV, partial RoPE, GQA)
//! combined with a sparse Mixture-of-Experts FFN (top-2 from 16 experts).
//! Each expert uses a SwiGLU activation function.

pub mod config;
pub mod model;

pub use config::PhiMoeConfig;
pub use model::{load_phi_moe_from_gguf, PhiMoeArchitecture, PhiMoeLayer, PhiMoeModel};
