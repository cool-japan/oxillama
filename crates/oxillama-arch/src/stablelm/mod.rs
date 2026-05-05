//! StableLM model architecture implementation.
//!
//! StableLM is a transformer family from Stability AI with three distinguishing
//! features relative to LLaMA:
//!
//! 1. **LayerNorm with bias** instead of RMSNorm.
//! 2. **Partial RoPE** — rotary embeddings are applied only to the first 25%
//!    (`partial_rotary_factor = 0.25`) of each Q/K head vector.
//! 3. **Parallel attention + FFN** — both branches receive the same normed
//!    residual and their outputs are added together to form the residual update:
//!    `y = x + Attn(LN_attn(x)) + FFN(LN_ffn(x))`.
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` / `.bias` — Pre-attention LayerNorm
//! - `blk.{i}.ffn_norm.weight` / `.bias` — Pre-FFN LayerNorm
//! - `blk.{i}.attn_q.weight` — Query projection
//! - `blk.{i}.attn_k.weight` — Key projection
//! - `blk.{i}.attn_v.weight` — Value projection
//! - `blk.{i}.attn_output.weight` — Attention output projection
//! - `blk.{i}.ffn_gate.weight` — SwiGLU gate projection
//! - `blk.{i}.ffn_up.weight` — SwiGLU up projection
//! - `blk.{i}.ffn_down.weight` — SwiGLU down projection
//! - `output_norm.weight` / `.bias` — Final LayerNorm
//! - `output.weight` — LM head

pub mod config;
mod model;

pub use config::StablelmConfig;
#[cfg(test)]
pub use model::make_test_layer;
pub use model::{StablelmLayer, StablelmModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// StableLM architecture plugin.
pub struct StablelmArchitecture;

impl StablelmArchitecture {
    /// Create a new `StablelmArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StablelmArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for StablelmArchitecture {
    fn arch_id(&self) -> &str {
        "stablelm"
    }

    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        if config.num_attention_heads == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        if config.hidden_size == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "hidden_size".to_string(),
                expected: ">0".to_string(),
                got: "0".to_string(),
            });
        }
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use StablelmModel::new for full loading)".to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        let mut patterns = vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding matrix".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final LayerNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.bias".to_string(),
                description: "Final LayerNorm bias".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding projection".to_string(),
                required: true,
            },
        ];

        let layer_tensors = [
            (
                "blk.{i}.attn_norm.weight",
                "Pre-attention LayerNorm scale (gamma)",
                true,
            ),
            (
                "blk.{i}.attn_norm.bias",
                "Pre-attention LayerNorm bias (beta)",
                true,
            ),
            (
                "blk.{i}.ffn_norm.weight",
                "Pre-FFN LayerNorm scale (gamma)",
                true,
            ),
            (
                "blk.{i}.ffn_norm.bias",
                "Pre-FFN LayerNorm bias (beta)",
                true,
            ),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            (
                "blk.{i}.ffn_gate.weight",
                "FFN gate projection (SwiGLU)",
                true,
            ),
            ("blk.{i}.ffn_up.weight", "FFN up projection", true),
            ("blk.{i}.ffn_down.weight", "FFN down projection", true),
        ];

        for (pat, desc, req) in layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required: req,
            });
        }

        patterns
    }
}
