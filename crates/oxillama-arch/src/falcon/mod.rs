//! Falcon model architecture plugin.
//!
//! Covers every checkpoint llama.cpp maps onto `LLM_ARCH_FALCON`:
//! * **Falcon-7B**: MQA (1 K/V head), single per-layer LayerNorm.
//! * **Falcon-40B**: GQA plus a *second* per-layer LayerNorm (`attn_norm_2`)
//!   that feeds the attention branch.
//! * **Falcon-2-11B**: GQA, single per-layer LayerNorm.
//!
//! All three run the same graph — `src/models/falcon.cpp`'s
//! `llm_build_falcon` — which is unconditionally RoPE (neox pairing) and
//! unconditionally *parallel*: `x' = ffn(attn_norm(x)) + attn(...) + x`.
//! There is no ALiBi path and no `ffn_norm` anywhere in the architecture.
//!
//! ## Tensor naming (GGUF, llama.cpp convention)
//!
//! | Tensor | Required | Description |
//! |--------|----------|-------------|
//! | `token_embd.weight` | yes | Token embedding |
//! | `blk.{i}.attn_norm.weight/bias` | yes | Pre-attention LayerNorm; also the FFN's input norm |
//! | `blk.{i}.attn_norm_2.weight/bias` | no | Second pre-attention LayerNorm (Falcon-40B) |
//! | `blk.{i}.attn_qkv.weight` | yes | Fused Q,K,V projection `[n_embd, n_embd + 2*n_embd_gqa]` |
//! | `blk.{i}.attn_output.weight` | yes | Attention output projection |
//! | `blk.{i}.ffn_up.weight` | yes | FFN up-projection (GELU, no gate) |
//! | `blk.{i}.ffn_down.weight` | yes | FFN down-projection |
//! | `output_norm.weight/bias` | yes | Final LayerNorm |
//! | `output.weight` | no | LM head; tied to `token_embd.weight` when absent |
//!
//! Falcon has no linear bias tensors at all — only the LayerNorm shifts.

pub mod config;
pub mod forward;
pub mod loader;
pub mod tensor_names;

pub use config::FalconConfig;
pub use forward::{FalconForward, FalconLayer};
pub use loader::load_falcon_from_gguf;
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

        // Full tensor loading requires a `GgufModel`: `TensorStore` carries the
        // tensor *descriptors* but not the payload, so no weight can be read
        // from here.  Use [`Self::build_from_gguf`] (or
        // [`load_falcon_from_gguf`] directly); this path stays a validation +
        // config-check path, matching starcoder and qwen3.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_falcon_from_gguf for full loading)".to_string(),
        })
    }

    /// Route the registry straight at [`load_falcon_from_gguf`].
    ///
    /// Unlike [`Self::build`], this entry point receives the tensor payload, so
    /// it can produce a real model instead of the `MissingTensor` sentinel.
    fn build_from_gguf(
        &self,
        model: &oxillama_gguf::GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_falcon_from_gguf(model, config)?))
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
        assert!(req_pats.contains(&"output_norm.weight"));
        // `output.weight` is deliberately NOT required: llama.cpp declares it
        // `TENSOR_NOT_REQUIRED` and duplicates `token_embd.weight` into it on
        // tied checkpoints.
        assert!(!req_pats.contains(&"output.weight"));
    }
}
