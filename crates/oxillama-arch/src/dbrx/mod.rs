//! DBRX model architecture.
//!
//! DBRX is a fine-grained Mixture-of-Experts (MoE) decoder-only transformer
//! produced by Databricks. It uses:
//! - Standard multi-head grouped-query attention (no MLA).
//! - 16 routed experts per FFN layer, with top-4 activation per token.
//!
//! Registered under the GGUF architecture identifier `"dbrx"`.
//!
//! ## Sub-modules
//! - [`config`]: `DbrxConfig` parsed from GGUF metadata.
//! - [`model`]: `DbrxModel` full transformer + `ForwardPass` impl.
//! - [`loader`]: GGUF tensor loading.

pub mod config;
pub mod loader;
pub mod model;
#[cfg(test)]
pub(crate) mod testkit;

pub use config::DbrxConfig;
pub use loader::load_dbrx_from_gguf;
pub use model::{build_dbrx_model, DbrxLayer, DbrxModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// Architecture plugin for DBRX models.
///
/// Registered under the identifier `"dbrx"` (matching the GGUF
/// `general.architecture` value used in DBRX GGUF files).
pub struct DbrxArchitecture;

impl DbrxArchitecture {
    /// Create a new `DbrxArchitecture` plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DbrxArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for DbrxArchitecture {
    fn arch_id(&self) -> &str {
        "dbrx"
    }

    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(ArchError::MissingTensor {
            name: "DbrxArchitecture::build() is not the loader entry point; \
                   call load_dbrx_from_gguf() instead"
                .to_string(),
        })
    }

    /// Route the registry straight at [`load_dbrx_from_gguf`].
    ///
    /// Without this override the engine could not build DBRX at all: the
    /// default returns `NotSupported`, and [`Self::build`] only ever sees the
    /// tensor metadata table.
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        _config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_dbrx_from_gguf(model)?))
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
                description: "Final LayerNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_norm.weight".to_string(),
                description: "Per-layer pre-attention LayerNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_qkv.weight".to_string(),
                description: "Fused QKV projection [hidden + 2*kv_dim, hidden]".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_output.weight".to_string(),
                description: "Attention output projection".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "blk.*.attn_output_norm.weight".to_string(),
                description: "Per-layer pre-FFN LayerNorm (DBRX has no ffn_norm)".to_string(),
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
