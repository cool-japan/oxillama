//! StableLM model architecture implementation.
//!
//! StableLM is a transformer family from Stability AI. Relative to LLaMA it
//! differs in four ways, all verified against
//! `~/work/refs/llama.cpp/src/models/stablelm.cpp` and
//! `~/work/refs/llama.cpp/src/llama-model.cpp`:
//!
//! 1. **LayerNorm with bias** instead of RMSNorm (`LLM_NORM`, not
//!    `LLM_NORM_RMS`), with `hparams.f_norm_eps` read from
//!    `{arch}.attention.layer_norm_epsilon`.
//! 2. **Partial RoPE** over `hparams.n_rot` = `{arch}.rope.dimension_count`
//!    dimensions, GPT-NeoX paired (`x[i]` with `x[i + n_rot/2]`).
//! 3. **Two residual topologies**, selected by the presence of `ffn_norm`:
//!    sequential (all StableLM except 2 12B) or parallel, in which the FFN
//!    consumes `attn_norm`'s output rather than a second norm.
//! 4. **Optional per-head QK LayerNorm** whose weight is `{head_dim, n_head}`
//!    — one distinct vector per head, unlike Qwen3's shared `[head_dim]`.
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — token embedding matrix
//! - `output_norm.weight` / `.bias` — final LayerNorm (both required)
//! - `output.weight` — LM head
//! - `blk.{i}.attn_norm.weight` / `.bias` — pre-attention LayerNorm (both required)
//! - `blk.{i}.attn_q/k/v.weight` — separate Q/K/V projections (never fused)
//! - `blk.{i}.attn_q/k/v.bias` — optional projection biases
//! - `blk.{i}.attn_output.weight` — attention output projection (no bias)
//! - `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` — optional QK LayerNorm
//! - `blk.{i}.ffn_norm.weight` / `.bias` — optional pre-FFN LayerNorm
//! - `blk.{i}.ffn_gate/up/down.weight` — SwiGLU FFN

pub mod config;
mod head_norm;
mod loader;
mod model;

pub use config::{StablelmConfig, DEFAULT_PARTIAL_ROTARY_FACTOR};
pub use head_norm::PerHeadLayerNorm;
pub use loader::load_stablelm_from_gguf;
pub use model::{StablelmLayer, StablelmLayerWeights, StablelmModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

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
        // `build` only receives the tensor *metadata* table, which carries no
        // payload to dequantize.  `build_from_gguf` below is the real entry
        // point and is what the registry and the engine should call.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_stablelm_from_gguf / build_from_gguf)".to_string(),
        })
    }

    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_stablelm_from_gguf(model, config)?))
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

        // `required` mirrors llama.cpp's `create_tensor` flags for
        // `LLM_ARCH_STABLELM`: `0` (required) vs `TENSOR_NOT_REQUIRED`.
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
                "Pre-FFN LayerNorm scale (gamma); absent ⇒ parallel residual",
                false,
            ),
            (
                "blk.{i}.ffn_norm.bias",
                "Pre-FFN LayerNorm bias (beta); optional alongside its weight",
                false,
            ),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_q.bias",
                "Query projection bias (Stable LM 2 1.6B)",
                false,
            ),
            (
                "blk.{i}.attn_k.bias",
                "Key projection bias (Stable LM 2 1.6B)",
                false,
            ),
            (
                "blk.{i}.attn_v.bias",
                "Value projection bias (Stable LM 2 1.6B)",
                false,
            ),
            (
                "blk.{i}.attn_q_norm.weight",
                "Per-head Q LayerNorm [head_dim, n_head] (StableLM 2 12B)",
                false,
            ),
            (
                "blk.{i}.attn_k_norm.weight",
                "Per-head K LayerNorm [head_dim, n_head_kv] (StableLM 2 12B)",
                false,
            ),
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
