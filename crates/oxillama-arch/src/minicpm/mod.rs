//! MiniCPM model architecture plugin.
//!
//! MiniCPM (base) is tensor-for-tensor identical to LLaMA — llama.cpp puts
//! `LLM_ARCH_MINICPM` in the same `load_tensors()` case as `LLM_ARCH_LLAMA`
//! and builds its graph with `llm_build_granite`.  The only differences are
//! three scalar hyper-parameters read from GGUF metadata:
//!
//! * `minicpm.embedding_scale` — multiplies the embedded token vector once,
//!   before layer 0;
//! * `minicpm.residual_scale` — multiplies the attention output *and* the FFN
//!   output, each immediately before its residual add;
//! * `minicpm.logit_scale` — the LM head's logits are multiplied by
//!   `1.0 / logit_scale`.
//!
//! See [`config::MiniCpmConfig`] for the primary-source citations and the
//! backward-compatibility defaults used by GGUFs that predate those keys.
//!
//! MiniCPM3 (`general.architecture = "minicpm3"`) is a *different*
//! architecture (MLA attention, NeoX-style RoPE) and is not handled here.
//!
//! ## Tensor naming (GGUF)
//!
//! MiniCPM uses the same GGUF tensor names as LLaMA:
//!
//! | Tensor | Description |
//! |--------|-------------|
//! | `token_embd.weight` | Token embedding |
//! | `blk.{i}.attn_norm.weight` | Pre-attention RMSNorm |
//! | `blk.{i}.ffn_norm.weight` | Pre-FFN RMSNorm |
//! | `blk.{i}.attn_q.weight` | Query projection |
//! | `blk.{i}.attn_k.weight` | Key projection |
//! | `blk.{i}.attn_v.weight` | Value projection |
//! | `blk.{i}.attn_output.weight` | Attention output projection |
//! | `blk.{i}.ffn_gate.weight` | FFN gate projection (SwiGLU) |
//! | `blk.{i}.ffn_up.weight` | FFN up projection |
//! | `blk.{i}.ffn_down.weight` | FFN down projection |
//! | `output_norm.weight` | Final RMSNorm |
//! | `output.weight` | LM head |

pub mod config;
pub mod forward;
pub mod loader;
pub mod tensor_names;

pub use config::{
    default_logit_scale, default_residual_scale, MiniCpmConfig, DEFAULT_EMBEDDING_SCALE,
};
pub use forward::{MiniCpmForward, MiniCpmLayer};
pub use loader::load_minicpm_from_gguf;
pub use tensor_names::minicpm_tensor_name_patterns;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// MiniCPM architecture plugin for the [`ArchitectureRegistry`](crate::registry::ArchitectureRegistry).
///
/// Matches GGUF files whose `general.architecture` field equals `"minicpm"`.
pub struct MiniCpmArchitecture;

impl MiniCpmArchitecture {
    /// Create a new [`MiniCpmArchitecture`] plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MiniCpmArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for MiniCpmArchitecture {
    fn arch_id(&self) -> &str {
        "minicpm"
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

        let _cfg = MiniCpmConfig::from_model_config(config)?;

        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_minicpm_from_gguf for full loading)".to_string(),
        })
    }

    /// Build a runnable MiniCPM model from a fully-loaded GGUF file.
    ///
    /// [`Self::build`] only receives the tensor *metadata* table and therefore
    /// cannot read a single weight; this entry point receives the payload, so
    /// the registry can construct MiniCPM without the engine hard-coding a
    /// match on the architecture name.
    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_minicpm_from_gguf(model, config)?))
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        minicpm_tensor_name_patterns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue, TensorStore};

    fn make_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("minicpm".to_string()),
        );
        store.insert(
            "minicpm.embedding_length".to_string(),
            MetadataValue::Uint32(256),
        );
        store.insert("minicpm.block_count".to_string(), MetadataValue::Uint32(4));
        store.insert(
            "minicpm.attention.head_count".to_string(),
            MetadataValue::Uint32(4),
        );
        store.insert(
            "minicpm.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(4),
        );
        store.insert(
            "minicpm.feed_forward_length".to_string(),
            MetadataValue::Uint32(512),
        );
        store
    }

    fn make_config() -> ModelConfig {
        ModelConfig::from_metadata(&make_metadata()).expect("config")
    }

    #[test]
    fn test_arch_id() {
        assert_eq!(MiniCpmArchitecture::new().arch_id(), "minicpm");
    }

    #[test]
    fn test_tensor_names_non_empty() {
        let arch = MiniCpmArchitecture::new();
        assert!(!arch.tensor_names().is_empty());
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = MiniCpmArchitecture::new();
        assert!(arch
            .tensor_names()
            .iter()
            .any(|p| p.pattern.contains("token_embd")));
    }

    #[test]
    fn test_build_returns_missing_tensor() {
        let arch = MiniCpmArchitecture::new();
        let cfg = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&cfg, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "build() should return MissingTensor"
        );
    }

    #[test]
    fn test_build_zero_heads_returns_config_mismatch() {
        let arch = MiniCpmArchitecture::new();
        let mut cfg = make_config();
        cfg.num_attention_heads = 0;
        let tensors = TensorStore::new();
        assert!(matches!(
            arch.build(&cfg, &tensors),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }

    #[test]
    fn test_build_zero_hidden_returns_config_mismatch() {
        let arch = MiniCpmArchitecture::new();
        let mut cfg = make_config();
        cfg.hidden_size = 0;
        let tensors = TensorStore::new();
        assert!(matches!(
            arch.build(&cfg, &tensors),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }
}
