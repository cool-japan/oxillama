//! MiniCPM model architecture plugin.
//!
//! MiniCPM is a scaled-embedding variant of LLaMA: the token embeddings are
//! multiplied by `hidden_size / dim_model_base` before the first transformer
//! layer.  All other components (RMSNorm, RoPE, GQA, SwiGLU) are identical
//! to standard LLaMA.
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
pub mod tensor_names;

pub use config::MiniCpmConfig;
pub use forward::{MiniCpmForward, MiniCpmLayer};
pub use tensor_names::minicpm_tensor_name_patterns;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

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
            name: "token_embd.weight (use MiniCpmForward::new for full loading)".to_string(),
        })
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
