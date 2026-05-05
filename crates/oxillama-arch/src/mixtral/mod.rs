//! Mixtral model architecture implementation.
//!
//! Mixtral is a Sparse Mixture-of-Experts (SMoE) variant of Mistral/LLaMA.
//! Every FFN block contains a pool of 8 SwiGLU experts; a learned router
//! activates the top-2 experts per token, keeping compute identical to a
//! 2-expert dense model while using 8x more parameters.
//!
//! ## Differences from Mistral
//!
//! - FFN replaced by sparse MoE: `blk.{i}.ffn_gate_inp.weight` (router) +
//!   packed expert weights (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)
//! - Default: 8 experts total, 2 activated per token (`top_k = 2`)
//! - `num_experts` / `num_experts_used` configurable from GGUF metadata
//!   (`mixtral.expert_count` / `mixtral.expert_used_count`)
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` — Pre-attention RMSNorm
//! - `blk.{i}.attn_q.weight` — Query projection
//! - `blk.{i}.attn_k.weight` — Key projection
//! - `blk.{i}.attn_v.weight` — Value projection
//! - `blk.{i}.attn_output.weight` — Output projection
//! - `blk.{i}.ffn_norm.weight` — Pre-FFN RMSNorm
//! - `blk.{i}.ffn_gate_inp.weight` — MoE router `[num_experts, hidden_size]`
//! - `blk.{i}.ffn_gate_exps.weight` — All expert gate projections (packed)
//! - `blk.{i}.ffn_up_exps.weight` — All expert up projections (packed)
//! - `blk.{i}.ffn_down_exps.weight` — All expert down projections (packed)
//! - `output_norm.weight` — Final RMSNorm
//! - `output.weight` — LM head / unembedding

mod model;

#[cfg(test)]
pub use model::make_test_layer;
pub use model::{MixtralLayer, MixtralModel, MixtralMoeConfig};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Mixtral architecture plugin.
pub struct MixtralArchitecture;

impl MixtralArchitecture {
    /// Create a new `MixtralArchitecture` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MixtralArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for MixtralArchitecture {
    fn arch_id(&self) -> &str {
        "mixtral"
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
        // Require the presence of MoE tensors (full load via from_gguf).
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use MixtralModel::new for full loading)".to_string(),
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
                description: "Final RMSNorm scale weights".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding projection".to_string(),
                required: true,
            },
        ];

        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm", true),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm", true),
            (
                "blk.{i}.ffn_gate_inp.weight",
                "MoE router weights [num_experts, hidden_size]",
                true,
            ),
            (
                "blk.{i}.ffn_gate_exps.weight",
                "All expert gate projections (packed) [num_experts, intermediate, hidden]",
                true,
            ),
            (
                "blk.{i}.ffn_up_exps.weight",
                "All expert up projections (packed) [num_experts, intermediate, hidden]",
                true,
            ),
            (
                "blk.{i}.ffn_down_exps.weight",
                "All expert down projections (packed) [num_experts, hidden, intermediate]",
                true,
            ),
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
