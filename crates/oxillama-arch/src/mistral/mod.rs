//! Mistral model architecture implementation.
//!
//! Mistral is structurally very similar to LLaMA with one key addition:
//! **sliding window attention** — each layer only attends to the last W
//! positions instead of the full sequence, reducing KV cache memory from
//! O(n) to O(W) per layer.
//!
//! ## Differences from LLaMA
//!
//! - Sliding window attention (configurable window size, typically 4096)
//! - Grouped-Query Attention (GQA) is standard (fewer KV heads than Q heads)
//! - Same tensor naming convention as LLaMA in GGUF
//!
//! ## Tensor naming convention (GGUF)
//!
//! Same as LLaMA:
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
//! - `output.weight` — LM head

mod model;

pub use model::{load_mistral_from_gguf, MistralModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Mistral architecture plugin.
pub struct MistralArchitecture;

impl MistralArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MistralArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for MistralArchitecture {
    fn arch_id(&self) -> &str {
        "mistral"
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

        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use MistralModel::from_gguf for full loading)".to_string(),
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
            MetadataValue::String("mistral".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal mistral config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = MistralArchitecture::new();
        assert_eq!(arch.arch_id(), "mistral");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = MistralArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = MistralArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_patterns() {
        let arch = MistralArchitecture::new();
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
        let arch = MistralArchitecture::new();
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
        let arch = MistralArchitecture::new();
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
        let arch = MistralArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
