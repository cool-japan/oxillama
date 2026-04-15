//! Qwen3 model architecture implementation.
//!
//! Qwen3 uses the same decoder-only transformer structure as LLaMA with:
//! - RMSNorm pre-normalization
//! - Grouped-Query Attention (GQA) with RoPE (may include attention bias)
//! - SwiGLU feed-forward network
//!
//! Key differences from LLaMA:
//! - `attention_bias: true` by default (Q, K, V, and output projections have bias)
//! - Different tensor naming convention in GGUF
//! - Different default RoPE base frequency
//!
//! ## Tensor naming convention (GGUF)
//!
//! Same as LLaMA convention:
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` — Pre-attention RMSNorm
//! - `blk.{i}.attn_q.weight` / `.bias` — Query projection
//! - `blk.{i}.attn_k.weight` / `.bias` — Key projection
//! - `blk.{i}.attn_v.weight` / `.bias` — Value projection
//! - `blk.{i}.attn_output.weight` / `.bias` — Output projection
//! - `blk.{i}.ffn_norm.weight` — Pre-FFN RMSNorm
//! - `blk.{i}.ffn_gate.weight` — FFN gate projection (SwiGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` — Final RMSNorm
//! - `output.weight` — LM head

mod model;

pub use model::{load_qwen3_from_gguf, Qwen3Model};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Qwen3 architecture plugin.
pub struct Qwen3Architecture;

impl Qwen3Architecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Qwen3Architecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for Qwen3Architecture {
    fn arch_id(&self) -> &str {
        "qwen3"
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

        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use Qwen3Model::from_gguf for full loading)".to_string(),
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
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm", true),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_q.bias", "Query bias", false),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_k.bias", "Key bias", false),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            ("blk.{i}.attn_v.bias", "Value bias", false),
            ("blk.{i}.attn_output.weight", "Output projection", true),
            ("blk.{i}.attn_output.bias", "Output bias", false),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm", true),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection", true),
            ("blk.{i}.ffn_up.weight", "FFN up projection", true),
            ("blk.{i}.ffn_down.weight", "FFN down projection", true),
        ];

        for (pat, desc, required) in layer_tensors {
            patterns.push(TensorNamePattern {
                pattern: pat.to_string(),
                description: desc.to_string(),
                required,
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
            MetadataValue::String("qwen3".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal qwen3 config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = Qwen3Architecture::new();
        assert_eq!(arch.arch_id(), "qwen3");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = Qwen3Architecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = Qwen3Architecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_and_optional_patterns() {
        let arch = Qwen3Architecture::new();
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
        // Bias tensors are optional in Qwen3
        let optional: Vec<&str> = names
            .iter()
            .filter(|p| !p.required)
            .map(|p| p.pattern.as_str())
            .collect();
        assert!(
            optional.iter().any(|p| p.contains(".bias")),
            "bias patterns should be optional in Qwen3"
        );
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = Qwen3Architecture::new();
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
    fn test_build_with_valid_config_returns_missing_tensor_error() {
        let arch = Qwen3Architecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        // build() always returns Err(MissingTensor) since full loading
        // goes through load_qwen3_from_gguf, not this path.
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
