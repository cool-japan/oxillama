//! InternLM3 architecture.
//!
//! Dense decoder-only Transformer following the LLaMA topology with:
//! - RMSNorm pre-normalization (no bias)
//! - Grouped-query attention (GQA) from GGUF `internlm3.attention.head_count_kv`
//! - RoPE positional embeddings
//! - SwiGLU feed-forward network (ReLU² variant treated as SwiGLU for struct purposes)
//! - Tied input/output embeddings
//!
//! GGUF `general.architecture` = `"internlm3"` (also sometimes `"internlm2"`).

mod model;

pub use model::InternLm3Architecture;

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

impl ModelArchitecture for InternLm3Architecture {
    fn arch_id(&self) -> &str {
        "internlm3"
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
            name: "token_embd.weight (use InternLm3Model::from_gguf for full loading)".to_string(),
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
                description: "LM head / unembedding (may be tied to token_embd)".to_string(),
                required: false,
            },
        ];

        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm"),
            ("blk.{i}.attn_q.weight", "Query projection"),
            ("blk.{i}.attn_k.weight", "Key projection"),
            ("blk.{i}.attn_v.weight", "Value projection"),
            ("blk.{i}.attn_output.weight", "Attention output projection"),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm"),
            ("blk.{i}.ffn_gate.weight", "FFN gate projection (SwiGLU)"),
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
    use crate::registry::ArchitectureRegistry;
    use oxillama_gguf::{MetadataStore, MetadataValue, TensorStore};

    fn make_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("internlm3".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal internlm3 config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = InternLm3Architecture::new();
        assert_eq!(arch.arch_id(), "internlm3");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = InternLm3Architecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = InternLm3Architecture::new();
        let names = arch.tensor_names();
        assert!(
            names.iter().any(|p| p.pattern.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_block_patterns() {
        let arch = InternLm3Architecture::new();
        let names = arch.tensor_names();
        let required_patterns = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.{i}.attn_q.weight",
            "blk.{i}.ffn_gate.weight",
        ];
        for pat in required_patterns {
            assert!(
                names.iter().any(|p| p.pattern == pat),
                "missing required pattern: {pat}"
            );
        }
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = InternLm3Architecture::new();
        let mut config = make_config();
        config.num_attention_heads = 0;
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(result.is_err());
        assert!(matches!(result, Err(ArchError::ConfigMismatch { .. })));
    }

    #[test]
    fn internlm3_in_registry() {
        let registry = ArchitectureRegistry::with_builtins();
        assert!(
            registry.contains("internlm3"),
            "internlm3 must be present in the default registry"
        );
    }
}
