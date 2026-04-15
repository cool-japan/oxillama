//! LLaMA model architecture implementation.
//!
//! Supports LLaMA 2, LLaMA 3.x, and LLaMA 4.x model families.
//! The architecture follows the standard decoder-only transformer with:
//! - RMSNorm pre-normalization
//! - Grouped-Query Attention (GQA) with RoPE
//! - SwiGLU feed-forward network
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
//! - `blk.{i}.ffn_gate.weight` — FFN gate projection (SwiGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` — Final RMSNorm
//! - `output.weight` — LM head (unembedding)

mod model;

pub(crate) use model::{
    dequant_to_f32, load_dequant_tensor, load_quant_linear, load_rms_norm_weight, softmax_inplace,
};
pub use model::{load_llama_from_gguf, LlamaModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// LLaMA architecture plugin.
pub struct LlamaArchitecture;

impl LlamaArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for LlamaArchitecture {
    fn arch_id(&self) -> &str {
        "llama"
    }

    fn build(
        &self,
        config: &ModelConfig,
        _tensors: &TensorStore,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        // Validate config
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

        // Note: actual tensor loading happens via LlamaModel::from_gguf()
        // which takes the full GgufModel. The build() path through TensorStore
        // alone cannot access raw data. This is a validation + config check path.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use LlamaModel::from_gguf for full loading)".to_string(),
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
                description: "Final RMSNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding".to_string(),
                required: true,
            },
        ];

        // Per-layer tensors (pattern uses {i} as placeholder)
        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm"),
            ("blk.{i}.attn_q.weight", "Query projection"),
            ("blk.{i}.attn_k.weight", "Key projection"),
            ("blk.{i}.attn_v.weight", "Value projection"),
            ("blk.{i}.attn_output.weight", "Attention output projection"),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm"),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection"),
            ("blk.{i}.ffn_up.weight", "FFN up projection"),
            ("blk.{i}.ffn_down.weight", "FFN down projection"),
        ];

        for (pat, desc) in layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required: true,
            });
        }

        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue, TensorStore};

    fn make_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal llama config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = LlamaArchitecture::new();
        assert_eq!(arch.arch_id(), "llama");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = LlamaArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = LlamaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_patterns() {
        let arch = LlamaArchitecture::new();
        let names = arch.tensor_names();
        let required: Vec<&str> = names
            .iter()
            .filter(|p| p.required)
            .map(|p| p.pattern.as_str())
            .collect();
        assert!(
            !required.is_empty(),
            "should have at least one required pattern"
        );
        assert!(
            required.contains(&"token_embd.weight"),
            "token_embd.weight must be required"
        );
        assert!(
            required.contains(&"output.weight"),
            "output.weight must be required"
        );
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = LlamaArchitecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err(), "build with zero heads should fail");
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "error should be ConfigMismatch"
        );
    }

    #[test]
    fn test_build_with_zero_hidden_size_returns_config_error() {
        let arch = LlamaArchitecture::new();
        let mut config = make_config();
        config.hidden_size = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err(), "build with zero hidden_size should fail");
        assert!(
            matches!(result, Err(ArchError::ConfigMismatch { .. })),
            "error should be ConfigMismatch"
        );
    }

    #[test]
    fn test_build_with_valid_config_returns_missing_tensor_error() {
        let arch = LlamaArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        // build() always returns Err(MissingTensor) since full loading
        // goes through load_llama_from_gguf, not this path.
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
