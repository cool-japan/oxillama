//! DeepSeek-V2 model architecture.
//!
//! Implements the DeepSeek-V2 transformer using Multi-head Latent Attention (MLA)
//! and a Mixture-of-Experts (MoE) FFN in the majority of layers.
//!
//! ## Sub-modules
//! - [`moe`]: DeepSeek MoE FFN with shared + routed experts and top-k routing.
//! - [`model`]: Full transformer model, per-layer weights, and `ForwardPass` impl.
//!
//! ## Loading
//! The `ModelArchitecture::build()` impl returns an `Err(MissingTensor)` pointing
//! callers to `load_deepseek_from_gguf()` which is the recommended entry point.

pub mod loader;
pub mod mla;
pub mod model;
pub mod moe;
pub mod quant_moe;

pub use loader::{load_deepseek_from_gguf, DeepSeekRouting};
pub use mla::{ds_mla_forward, DeepSeekMlaWeights, QProjection};
pub use model::{
    build_deepseek_model, DeepSeekLayer, DeepSeekModel, DenseFfn, FfnKind, N_DENSE_LAYERS,
};
pub use moe::{moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode};
pub use quant_moe::{ExpertActivation, GatingFunc, RoutedMoeConfig, RoutedQuantMoe};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// Architecture plugin for DeepSeek-V2 models.
///
/// Registered under the identifier `"deepseek2"` (matching the GGUF
/// `general.architecture` value used in DeepSeek GGUF files).
pub struct DeepSeekArchitecture;

impl DeepSeekArchitecture {
    /// Create a new `DeepSeekArchitecture` plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DeepSeekArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for DeepSeekArchitecture {
    fn arch_id(&self) -> &str {
        "deepseek2"
    }

    fn build(
        &self,
        _config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Err(ArchError::MissingTensor {
            name: "DeepSeekArchitecture::build() is not the loader entry point; \
                   call load_deepseek_from_gguf() instead"
                .to_string(),
        })
    }

    /// Route the registry straight at [`load_deepseek_from_gguf`].
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        _config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_deepseek_from_gguf(model)?))
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        // Standard GGUF tensor name patterns for DeepSeek-V2.
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
                pattern: "blk.*".to_string(),
                description: "Per-layer weights (MLA, FFN)".to_string(),
                required: true,
            },
        ]
    }
}
