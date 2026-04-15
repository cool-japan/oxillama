//! LLaVA (Large Language and Vision Assistant) architecture.
//!
//! Implements LLaVA-1.5 compatible multimodal inference combining a
//! CLIP vision encoder with a LLaMA language backbone.
//!
//! Model loading follows the GGUF tensor naming convention used by
//! llama.cpp's LLaVA implementation:
//! - Language model: same tensors as LLaMA (`token_embd`, `blk.*`, `output*`)
//! - Vision encoder: `v.patch_embd.weight`, `v.position_embd.weight`, `v.blk.*`
//! - MM projector: `mm.0.weight`, `mm.0.bias`, `mm.2.weight`, `mm.2.bias`

pub mod model;

pub use model::{load_llava_from_gguf, LlavaModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// LLaVA architecture plugin.
pub struct LlavaArchitecture;

impl LlavaArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlavaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for LlavaArchitecture {
    fn arch_id(&self) -> &str {
        "llava"
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

        // Actual tensor loading happens via load_llava_from_gguf() which takes
        // the full GgufModel. This path is config validation only.
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use load_llava_from_gguf for full loading)".to_string(),
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
            TensorNamePattern {
                pattern: "mm.0.weight".to_string(),
                description: "MM projector FC1 weight".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.0.bias".to_string(),
                description: "MM projector FC1 bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "mm.2.weight".to_string(),
                description: "MM projector FC2 weight".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "mm.2.bias".to_string(),
                description: "MM projector FC2 bias".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.patch_embd.weight".to_string(),
                description: "CLIP patch embedding weight".to_string(),
                required: false,
            },
            TensorNamePattern {
                pattern: "v.position_embd.weight".to_string(),
                description: "CLIP position embedding".to_string(),
                required: false,
            },
        ];

        // Per-layer LLaMA backbone tensors
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
            MetadataValue::String("llava".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal llava config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = LlavaArchitecture::new();
        assert_eq!(arch.arch_id(), "llava");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_contains_mm_projector() {
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        // LLaVA has a multimodal projector
        assert!(
            patterns.iter().any(|p| p.starts_with("mm.")),
            "LLaVA should have mm projector tensor patterns"
        );
        // mm.0.weight and mm.2.weight should be required
        let required_patterns: Vec<&str> = names
            .iter()
            .filter(|p| p.required)
            .map(|p| p.pattern.as_str())
            .collect();
        assert!(
            required_patterns.contains(&"mm.0.weight"),
            "mm.0.weight should be required"
        );
        assert!(
            required_patterns.contains(&"mm.2.weight"),
            "mm.2.weight should be required"
        );
    }

    #[test]
    fn test_tensor_names_contains_vision_patterns() {
        let arch = LlavaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        // LLaVA has CLIP vision encoder tensors (optional)
        assert!(
            patterns.iter().any(|p| p.starts_with("v.")),
            "LLaVA should have vision encoder tensor patterns"
        );
    }

    #[test]
    fn test_tensor_names_has_required_patterns() {
        let arch = LlavaArchitecture::new();
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
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = LlavaArchitecture::new();
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
        let arch = LlavaArchitecture::new();
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
        let arch = LlavaArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
