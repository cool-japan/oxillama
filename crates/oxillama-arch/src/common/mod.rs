//! Common layer implementations shared across model architectures.
//!
//! These building blocks (RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, Linear,
//! ALiBi) are used by multiple model families and are implemented once here.

pub mod alibi;
pub mod attention;
pub mod embedding;
pub mod gelu;
pub mod kv_view;
pub mod layer_norm;
pub mod linear;
pub mod loader;
pub mod mla;
pub mod moe;
pub mod mrope;
pub mod rms_norm;
pub mod rope;
pub mod sequence_state;
pub mod swiglu;

pub use attention::{
    effective_attention_span, is_sliding_window_layer, swa_attend_start, validate_context_bounds,
    validate_token_ids, SwaPattern,
};
pub use embedding::TokenEmbedding;
pub use kv_view::{fetch_keys, fetch_values};
pub use loader::{
    dequant_to_f32, dequant_to_f32_slice, load_bias, load_dequant_tensor, load_lm_head,
    load_quant_linear, load_quant_linear_opt, load_quant_linear_with_bias, load_rms_norm_weight,
};
pub use mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
pub use mrope::MRopeAxis;
pub use mrope::MRopeTable;
pub use rope::{RopeParams, RopeScalingType, RopeStyle, RopeTable};
pub use sequence_state::{
    AttentionSequenceState, Mamba2SequenceState, SequenceState, SsmLayerState,
};
