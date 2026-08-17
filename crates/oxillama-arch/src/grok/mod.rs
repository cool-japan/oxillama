//! Grok-1 model architecture.
//!
//! Grok-1 is a Mixture-of-Experts decoder-only transformer produced by xAI.
//! It uses:
//! - Standard grouped-query attention with a large RoPE base (θ = 1,000,000).
//! - 8 routed experts per FFN layer, with top-2 activation per token.
//!
//! Registered under the GGUF architecture identifier `"grok"`.
//!
//! ## Sub-modules
//! - [`config`]: `GrokConfig` parsed from GGUF metadata.
//! - [`model`]: `GrokModel` full transformer + `ForwardPass` impl.

pub mod config;
pub mod loader;
pub mod model;
pub mod moe;
#[cfg(test)]
pub(crate) mod testkit;

pub use config::GrokConfig;
pub use loader::load_grok_from_gguf;
pub use model::{build_grok_model, GrokDenseFfn, GrokLayer, GrokModel};
pub use moe::GrokMoe;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// Architecture plugin for Grok-1 models.
///
/// Registered under the identifier `"grok"` (matching the GGUF
/// `general.architecture` value used in Grok GGUF files).
pub struct GrokArchitecture;

impl GrokArchitecture {
    /// Create a new `GrokArchitecture` plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrokArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for GrokArchitecture {
    fn arch_id(&self) -> &str {
        "grok"
    }

    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(ArchError::MissingTensor {
            name: "GrokArchitecture::build() is not the loader entry point; \
                   call load_grok_from_gguf() instead"
                .to_string(),
        })
    }

    /// Route the registry straight at [`load_grok_from_gguf`].
    ///
    /// Without this override the default `NotSupported` made Grok unreachable
    /// from the engine.
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        _config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_grok_from_gguf(model)?))
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        vec![
            TensorNamePattern {
                pattern: "token_embd.weight".to_string(),
                description: "Token embedding table".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final RMSNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_norm.weight".to_string(),
                description: "Per-layer pre-attention RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_q.weight".to_string(),
                description: "Query projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_k.weight".to_string(),
                description: "Key projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_v.weight".to_string(),
                description: "Value projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_output.weight".to_string(),
                description: "Attention output projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_output_norm.weight".to_string(),
                description: "Post-attention RMSNorm, applied before the residual".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ffn_norm.weight".to_string(),
                description: "Per-layer pre-FFN RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.layer_output_norm.weight".to_string(),
                description: "Post-FFN RMSNorm, applied before the residual \
                              (or blk.*.post_ffw_norm.weight)"
                    .to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ffn_gate_inp.weight".to_string(),
                description: "MoE router projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ffn_gate_exps.weight".to_string(),
                description: "Stacked expert gate projections".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ffn_up_exps.weight".to_string(),
                description: "Stacked expert up projections".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.ffn_down_exps.weight".to_string(),
                description: "Stacked expert down projections".to_string(),
                required: true,
            },
        ]
    }
}
