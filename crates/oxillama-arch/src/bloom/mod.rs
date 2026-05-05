//! BLOOM model architecture implementation.
//!
//! BigScience BLOOM uses:
//! - Pre-LayerNorm with bias (not RMSNorm)
//! - Fused QKV projection (`attn_qkv.weight / bias`)
//! - ALiBi positional biases (no RoPE)
//! - MHA (Multi-Head Attention, not GQA)
//! - Standard GELU FFN (not SwiGLU)
//! - Biases on all linear layers

pub mod config;
pub mod model;

pub use config::BloomConfig;
pub use model::{load_bloom_from_gguf, BloomArchitecture, BloomLayer, BloomModel};
