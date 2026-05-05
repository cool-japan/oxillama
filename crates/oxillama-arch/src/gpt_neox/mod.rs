//! GPT-NeoX model architecture implementation.
//!
//! GPT-NeoX (EleutherAI, 2022) is a large-scale autoregressive decoder model
//! that introduced the **parallel residual** computation pattern later adopted
//! by StableLM and others.
//!
//! ## Key architectural features
//!
//! 1. **Parallel residual**: attention and FFN share the same pre-norm input
//!    and both outputs are added together to form the single residual update:
//!    `y = x + Attention(LN1(x)) + FFN(LN2(x))`.
//!
//! 2. **Two learned-bias LayerNorms** per layer: `ln1` (pre-attention) and
//!    `ln2` (pre-FFN), each with an independent bias parameter.
//!
//! 3. **Partial RoPE** (same fraction convention as StableLM): only the first
//!    `partial_rotary_factor × head_dim` dimensions of each Q/K head are
//!    rotated; the rest pass through unmodified.
//!
//! 4. **GELU FFN** (not SwiGLU): the FFN is a single gated layer
//!    `GELU(W_up @ x)` followed by a down projection, without a separate gate.
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.ln1.weight` / `.bias` — Pre-attention LayerNorm
//! - `blk.{i}.ln2.weight` / `.bias` — Pre-FFN LayerNorm
//! - `blk.{i}.attn_q.weight` — Q projection
//! - `blk.{i}.attn_k.weight` — K projection
//! - `blk.{i}.attn_v.weight` — V projection
//! - `blk.{i}.attn_output.weight` — Attention output projection
//! - `blk.{i}.ffn_up.weight` — FFN up/gate projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` / `.bias` — Final LayerNorm
//! - `output.weight` — LM head / unembedding

mod model;

#[cfg(test)]
pub use model::make_test_layer;
pub use model::{GptNeoxLayer, GptNeoxModel, DEFAULT_PARTIAL_ROTARY_FACTOR};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// GPT-NeoX architecture plugin.
pub struct GptNeoxArchitecture;

impl GptNeoxArchitecture {
    /// Create a new `GptNeoxArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GptNeoxArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for GptNeoxArchitecture {
    fn arch_id(&self) -> &str {
        "gptneox"
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
            name: "token_embd.weight (use GptNeoxModel::new for full loading)".to_string(),
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
                "blk.{i}.ln1.weight",
                "Pre-attention LayerNorm scale (gamma)",
                true,
            ),
            (
                "blk.{i}.ln1.bias",
                "Pre-attention LayerNorm bias (beta)",
                true,
            ),
            (
                "blk.{i}.ln2.weight",
                "Pre-FFN LayerNorm scale (gamma)",
                true,
            ),
            ("blk.{i}.ln2.bias", "Pre-FFN LayerNorm bias (beta)", true),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            (
                "blk.{i}.ffn_up.weight",
                "FFN up/gate projection (GELU)",
                true,
            ),
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
