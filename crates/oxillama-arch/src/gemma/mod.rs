//! Gemma model architecture implementation.
//!
//! Supports Gemma 2 and Gemma 3 model families from Google DeepMind.
//!
//! ## Key differences from LLaMA
//!
//! - **Embedding scaling**: embeddings are multiplied by `sqrt(hidden_size)`
//! - **Post-normalization**: RMSNorm applied after attention and FFN (in addition to pre-norm)
//! - **GeGLU activation**: uses GELU-gated linear unit instead of SwiGLU
//! - **Interleaved attention**: alternating sliding window (local) and full causal (global)
//!   attention layers in Gemma 2 (configurable via `attention_window_size`)
//! - **Logit soft-capping**: attention logits and final logits are soft-capped
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` — Pre-attention RMSNorm
//! - `blk.{i}.attn_post_norm.weight` — Post-attention RMSNorm (Gemma 2+)
//! - `blk.{i}.attn_q.weight` — Query projection
//! - `blk.{i}.attn_k.weight` — Key projection
//! - `blk.{i}.attn_v.weight` — Value projection
//! - `blk.{i}.attn_output.weight` — Output projection
//! - `blk.{i}.ffn_norm.weight` — Pre-FFN RMSNorm
//! - `blk.{i}.ffn_post_norm.weight` — Post-FFN RMSNorm (Gemma 2+)
//! - `blk.{i}.ffn_gate.weight` — FFN gate projection (GeGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` — Final RMSNorm
//! - `output.weight` — LM head (may be tied to token_embd.weight)

mod model;

pub use model::{load_gemma_from_gguf, GemmaModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Gemma architecture plugin.
pub struct GemmaArchitecture;

impl GemmaArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GemmaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for GemmaArchitecture {
    fn arch_id(&self) -> &str {
        "gemma"
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
            name: "token_embd.weight (use GemmaModel::from_gguf for full loading)".to_string(),
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
                required: false, // may be tied
            },
        ];

        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention RMSNorm", true),
            (
                "blk.{i}.attn_post_norm.weight",
                "Post-attention RMSNorm",
                false,
            ),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN RMSNorm", true),
            ("blk.{i}.ffn_post_norm.weight", "Post-FFN RMSNorm", false),
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
            MetadataValue::String("gemma".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal gemma config should parse")
    }

    #[test]
    fn test_arch_id() {
        let arch = GemmaArchitecture::new();
        assert_eq!(arch.arch_id(), "gemma");
    }

    #[test]
    fn test_tensor_names_is_non_empty() {
        let arch = GemmaArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_tensor_names_contains_token_embd() {
        let arch = GemmaArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_tensor_names_has_required_and_optional_patterns() {
        let arch = GemmaArchitecture::new();
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
        // output.weight is optional in Gemma (may be tied to token_embd)
        let output_pattern = names
            .iter()
            .find(|p| p.pattern == "output.weight")
            .expect("output.weight should be listed");
        assert!(
            !output_pattern.required,
            "output.weight should be optional in Gemma (weight tying)"
        );
        // Post-norms should be optional
        let optional: Vec<&str> = names
            .iter()
            .filter(|p| !p.required)
            .map(|p| p.pattern.as_str())
            .collect();
        assert!(
            optional.iter().any(|p| p.contains("post_norm")),
            "post_norm patterns should be optional in Gemma"
        );
    }

    #[test]
    fn test_build_with_zero_heads_returns_config_error() {
        let arch = GemmaArchitecture::new();
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
        let arch = GemmaArchitecture::new();
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
        let arch = GemmaArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
