//! Phi model architecture implementation.
//!
//! Supports Phi-3 and Phi-4 model families from Microsoft.
//!
//! ## Key differences from LLaMA
//!
//! - **Merged QKV projection**: single `attn_qkv.weight` tensor instead of
//!   separate Q, K, V tensors (packed as [Q; K; V] along output dim)
//! - **Partial RoPE**: only applies RoPE to the first `partial_rotary_factor`
//!   fraction of each head's dimensions (rest are not rotated)
//! - **Long context**: Phi-3.5 supports up to 128K context with YaRN-style RoPE
//! - Same SwiGLU FFN activation as LLaMA
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` — Pre-attention RMSNorm
//! - `blk.{i}.attn_qkv.weight` — Merged Q/K/V projection
//! - `blk.{i}.attn_output.weight` — Output projection
//! - `blk.{i}.ffn_norm.weight` — Pre-FFN RMSNorm
//! - `blk.{i}.ffn_gate.weight` — FFN gate projection (SwiGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` — Final RMSNorm
//! - `output.weight` — LM head

mod model;

pub use model::{load_phi_from_gguf, PhiModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Phi architecture plugin.
pub struct PhiArchitecture;

impl PhiArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PhiArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for PhiArchitecture {
    fn arch_id(&self) -> &str {
        "phi3"
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
            name: "token_embd.weight (use PhiModel::from_gguf for full loading)".to_string(),
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
            ("blk.{i}.attn_qkv.weight", "Merged Q/K/V projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
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
            MetadataValue::String("phi3".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal phi3 config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = PhiArchitecture::new();
        assert_eq!(arch.arch_id(), "phi3");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = PhiArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = PhiArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_uses_merged_qkv() {
        let arch = PhiArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        // Phi uses merged QKV, not separate Q/K/V
        assert!(
            patterns.iter().any(|p| p.contains("attn_qkv")),
            "Phi should have a merged attn_qkv pattern"
        );
        assert!(
            !patterns.contains(&"blk.{i}.attn_q.weight"),
            "Phi should not have a separate attn_q.weight pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_patterns() {
        let arch = PhiArchitecture::new();
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
            required.contains(&"blk.{i}.attn_qkv.weight"),
            "merged attn_qkv.weight must be required"
        );
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = PhiArchitecture::new();
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
        let arch = PhiArchitecture::new();
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
        let arch = PhiArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
