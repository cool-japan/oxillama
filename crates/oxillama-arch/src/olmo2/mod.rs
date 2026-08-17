//! OLMo2 model architecture plugin.
//!
//! OLMo2 is post-norm: `attn_post_norm` and `ffn_post_norm` are applied to the
//! sublayer *output*, **before** it is added back to the residual.  It also
//! applies RMSNorm to the Q and K projections — over the whole projected
//! vector, before the reshape into heads — and has no biases anywhere.
//!
//! ## Tensor naming (GGUF)
//!
//! | Tensor | Shape | Description |
//! |--------|-------|-------------|
//! | `token_embd.weight` | `[n_embd, n_vocab]` | Token embedding |
//! | `blk.{i}.attn_q.weight` | `[n_embd, n_embd]` | Query projection |
//! | `blk.{i}.attn_k.weight` | `[n_embd, n_embd_gqa]` | Key projection |
//! | `blk.{i}.attn_v.weight` | `[n_embd, n_embd_gqa]` | Value projection |
//! | `blk.{i}.attn_output.weight` | `[n_embd, n_embd]` | Attention output projection |
//! | `blk.{i}.attn_q_norm.weight` | `[n_embd]` | Query RMSNorm |
//! | `blk.{i}.attn_k_norm.weight` | `[n_head_kv * n_embd_head]` | Key RMSNorm |
//! | `blk.{i}.post_attention_norm.weight` | `[n_embd]` | Post-attention RMSNorm |
//! | `blk.{i}.ffn_gate.weight` | `[n_embd, n_ff]` | FFN gate projection (SwiGLU) |
//! | `blk.{i}.ffn_up.weight` | `[n_embd, n_ff]` | FFN up projection |
//! | `blk.{i}.ffn_down.weight` | `[n_ff, n_embd]` | FFN down projection |
//! | `blk.{i}.post_ffw_norm.weight` | `[n_embd]` | Post-FFN RMSNorm |
//! | `output_norm.weight` | `[n_embd]` | Final RMSNorm |
//! | `output.weight` | `[n_embd, n_vocab]` | LM head |
//!
//! The two post-norm names come from the shared `LLM_TENSOR_NAMES` table in
//! `~/work/refs/llama.cpp/src/llama-arch.cpp` (lines 355 / 359) and are *not*
//! `attn_post_norm` / `ffn_post_norm`.

pub mod config;
pub mod forward;
pub mod loader;
pub mod tensor_names;

pub use config::Olmo2Config;
pub use forward::{Olmo2Forward, Olmo2Layer};
pub use loader::load_olmo2_from_gguf;
pub use tensor_names::olmo2_tensor_name_patterns;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::{GgufModel, TensorStore};

/// OLMo2 architecture plugin for the [`ArchitectureRegistry`](crate::registry::ArchitectureRegistry).
///
/// Matches GGUF files whose `general.architecture` field equals `"olmo2"`.
pub struct Olmo2Architecture;

impl Olmo2Architecture {
    /// Create a new [`Olmo2Architecture`] plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Olmo2Architecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for Olmo2Architecture {
    fn arch_id(&self) -> &str {
        "olmo2"
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

        let _cfg = Olmo2Config::from_model_config(config)?;

        // `build()` only receives the tensor *metadata* table, which carries no
        // payload to dequantize.  Full loading goes through
        // [`Self::build_from_gguf`] / [`load_olmo2_from_gguf`].
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_olmo2_from_gguf for full loading)".to_string(),
        })
    }

    fn build_from_gguf(
        &self,
        model: &GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(load_olmo2_from_gguf(model, config)?))
    }

    fn tensor_names(&self) -> Vec<TensorNamePattern> {
        olmo2_tensor_name_patterns()
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
            MetadataValue::String("olmo2".to_string()),
        );
        store.insert(
            "olmo2.embedding_length".to_string(),
            MetadataValue::Uint32(256),
        );
        store.insert("olmo2.block_count".to_string(), MetadataValue::Uint32(4));
        store.insert(
            "olmo2.attention.head_count".to_string(),
            MetadataValue::Uint32(4),
        );
        store.insert(
            "olmo2.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert(
            "olmo2.feed_forward_length".to_string(),
            MetadataValue::Uint32(512),
        );
        store
    }

    fn make_config() -> ModelConfig {
        ModelConfig::from_metadata(&make_metadata()).expect("config")
    }

    #[test]
    fn test_arch_id() {
        assert_eq!(Olmo2Architecture::new().arch_id(), "olmo2");
    }

    #[test]
    fn test_tensor_names_non_empty() {
        let arch = Olmo2Architecture::new();
        assert!(!arch.tensor_names().is_empty());
    }

    #[test]
    fn test_tensor_names_contains_post_norm_tensors() {
        let arch = Olmo2Architecture::new();
        let names = arch.tensor_names();
        assert!(names
            .iter()
            .any(|p| p.pattern == "blk.{i}.post_attention_norm.weight"));
        assert!(names
            .iter()
            .any(|p| p.pattern == "blk.{i}.post_ffw_norm.weight"));
    }

    #[test]
    fn test_tensor_names_contains_qk_norm() {
        let arch = Olmo2Architecture::new();
        let names = arch.tensor_names();
        assert!(names.iter().any(|p| p.pattern.contains("attn_q_norm")));
        assert!(names.iter().any(|p| p.pattern.contains("attn_k_norm")));
    }

    #[test]
    fn test_build_returns_missing_tensor() {
        let arch = Olmo2Architecture::new();
        let cfg = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&cfg, &tensors);
        assert!(matches!(result, Err(ArchError::MissingTensor { .. })));
    }

    #[test]
    fn test_build_zero_heads_returns_config_mismatch() {
        let arch = Olmo2Architecture::new();
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
        let arch = Olmo2Architecture::new();
        let mut cfg = make_config();
        cfg.hidden_size = 0;
        let tensors = TensorStore::new();
        assert!(matches!(
            arch.build(&cfg, &tensors),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }
}
