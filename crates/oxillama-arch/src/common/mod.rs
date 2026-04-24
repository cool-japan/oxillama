//! Common layer implementations shared across model architectures.
//!
//! These building blocks (RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, Linear) are used by
//! multiple model families and are implemented once here.

pub mod attention;
pub mod gelu;
pub mod layer_norm;
pub mod linear;
pub mod mla;
pub mod moe;
pub mod rms_norm;
pub mod rope;
pub mod sequence_state;
pub mod swiglu;

pub use attention::effective_attention_span;
pub use mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
pub use sequence_state::{
    AttentionSequenceState, Mamba2SequenceState, SequenceState, SsmLayerState,
};
