//! Falcon model architecture plugin.
//!
//! Supports both:
//! * **Falcon-1** (7B / 40B): parallel attention + FFN, ALiBi positional bias,
//!   fused QKV with MHA (1 K/V head), old GGUF tensor naming.
//! * **Falcon-2** (11B): Group-Query Attention, RoPE, sequential attention then
//!   FFN, updated GGUF tensor naming.
//!
//! ## Tensor naming (GGUF, llama.cpp convention)
//!
//! | Tensor | Description |
//! |--------|-------------|
//! | `token_embd.weight` | Token embedding |
//! | `blk.{i}.attn_norm.weight/bias` | Pre-attention LayerNorm |
//! | `blk.{i}.ffn_norm.weight/bias` | Pre-FFN LayerNorm (Falcon-2 only) |
//! | `blk.{i}.attn_qkv.weight` | Fused Q,K,V projection |
//! | `blk.{i}.attn_output.weight` | Attention output projection |
//! | `blk.{i}.ffn_up.weight` | FFN up-projection |
//! | `blk.{i}.ffn_down.weight` | FFN down-projection |
//! | `output_norm.weight/bias` | Final LayerNorm |
//! | `output.weight` | LM head |

pub mod config;
pub mod forward;
pub mod tensor_names;

pub use config::FalconConfig;
pub use forward::{FalconForward, FalconLayer};
pub use tensor_names::falcon_tensor_name_patterns;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Falcon architecture plugin for the [`ArchitectureRegistry`](crate::registry::ArchitectureRegistry).
///
/// Matches GGUF files whose `general.architecture` field equals `"falcon"`.
pub struct FalconArchitecture;

impl FalconArchitecture {
    /// Create a new [`FalconArchitecture`] plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FalconArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for FalconArchitecture {
    fn arch_id(&self) -> &str {
        "falcon"
    }

    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        // Validate generic config fields.
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

        // Validate that a Falcon-specific config can be derived from metadata.
        let _falcon_cfg = FalconConfig::from_model_config(config)?;

        // Full tensor loading requires a GgufModel (use load_falcon_from_gguf).
        // The build() path through TensorStore alone cannot access raw data;
        // this is a validation + config-check path only.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use FalconForward::from_gguf for full loading)".to_string(),
        })
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        falcon_tensor_name_patterns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue, TensorStore};

    fn make_falcon_metadata(arch: &str) -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        );
        store.insert(
            "falcon.embedding_length".to_string(),
            MetadataValue::Uint32(256),
        );
        store.insert("falcon.block_count".to_string(), MetadataValue::Uint32(4));
        store.insert(
            "falcon.attention.head_count".to_string(),
            MetadataValue::Uint32(4),
        );
        store.insert(
            "falcon.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(1),
        );
        store.insert(
            "falcon.feed_forward_length".to_string(),
            MetadataValue::Uint32(512),
        );
        store
    }

    fn make_config() -> ModelConfig {
        let store = make_falcon_metadata("falcon");
        ModelConfig::from_metadata(&store).expect("falcon config")
    }

    #[test]
    fn test_arch_id() {
        assert_eq!(FalconArchitecture::new().arch_id(), "falcon");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let names = FalconArchitecture::new().tensor_names();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let names = FalconArchitecture::new().tensor_names();
        assert!(
            names.iter().any(|p| p.pattern.contains("token_embd")),
            "must contain token_embd pattern"
        );
    }

    #[test]
    fn test_build_with_valid_config_returns_missing_tensor_err() {
        // build() always returns MissingTensor (full loading needs GgufModel).
        let arch = FalconArchitecture::new();
        let cfg = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&cfg, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "build() should return MissingTensor"
        );
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_mismatch() {
        let arch = FalconArchitecture::new();
        let mut cfg = make_config();
        cfg.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&cfg, &tensors);
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "build() with zero heads should return ConfigMismatch"
        );
    }

    #[test]
    fn test_build_with_zero_hidden_size_returns_config_mismatch() {
        let arch = FalconArchitecture::new();
        let mut cfg = make_config();
        cfg.hidden_size = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&cfg, &tensors);
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "build() with hidden_size=0 should return ConfigMismatch"
        );
    }

    #[test]
    fn test_falcon_config_from_valid_model_config() {
        let cfg = make_config();
        let fcfg = FalconConfig::from_model_config(&cfg).unwrap();
        assert!(fcfg.n_heads > 0);
        assert!(fcfg.n_kv_heads > 0);
        assert!(fcfg.n_layers > 0);
        assert!(fcfg.hidden_size > 0);
    }

    #[test]
    fn test_required_tensor_names_present() {
        let names = FalconArchitecture::new().tensor_names();
        let required: Vec<_> = names.iter().filter(|p| p.required).collect();
        assert!(
            !required.is_empty(),
            "must have at least one required tensor"
        );
        let req_pats: Vec<&str> = required.iter().map(|p| p.pattern.as_str()).collect();
        assert!(req_pats.contains(&"token_embd.weight"));
        assert!(req_pats.contains(&"output.weight"));
    }
}
