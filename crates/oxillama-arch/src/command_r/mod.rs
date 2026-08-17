//! Command-R model architecture implementation.
//!
//! Command-R (by Cohere) is a decoder-only transformer that is architecturally
//! almost identical to LLaMA, with two key additions:
//!
//! 1. **Optional Q/K normalization** (`blk.{i}.attn_q_norm.weight` and
//!    `blk.{i}.attn_k_norm.weight`) — present in Command-R+ but absent in the
//!    base Command-R models.
//!
//! 2. **Logit scaling** — the final logits are multiplied by a scalar read from
//!    `command-r.logit_scale` in the GGUF metadata (defaults to 1.0 when absent).
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `blk.{i}.attn_norm.weight` — Pre-block LayerNorm (no bias; feeds BOTH
//!   attention and FFN — Command-R has no `ffn_norm` tensor)
//! - `blk.{i}.attn_q.weight` — Query projection
//! - `blk.{i}.attn_k.weight` — Key projection
//! - `blk.{i}.attn_v.weight` — Value projection
//! - `blk.{i}.attn_output.weight` — Attention output projection
//! - `blk.{i}.attn_q_norm.weight` — Q normalization (optional, Command-R+)
//! - `blk.{i}.attn_k_norm.weight` — K normalization (optional, Command-R+)
//! - `blk.{i}.ffn_gate.weight` — FFN gate projection (SwiGLU)
//! - `blk.{i}.ffn_up.weight` — FFN up projection
//! - `blk.{i}.ffn_down.weight` — FFN down projection
//! - `output_norm.weight` — Final LayerNorm
//! - `output.weight` — LM head, always tied to `token_embd.weight` on real
//!   checkpoints (llama.cpp never creates a standalone `output.weight` for
//!   this architecture)

mod model;

pub use model::{load_command_r_from_gguf, CommandRLayer, CommandRModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// Command-R architecture plugin.
pub struct CommandRArchitecture;

impl CommandRArchitecture {
    /// Create a new `CommandRArchitecture` plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommandRArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for CommandRArchitecture {
    fn arch_id(&self) -> &str {
        "command-r"
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

        // Full loading requires a GgufModel (use load_command_r_from_gguf).
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use CommandRModel::from_gguf for full loading)".to_string(),
        })
    }

    fn build_from_gguf(
        &self,
        model: &oxillama_gguf::GgufModel,
        config: &ModelConfig,
    ) -> ArchResult<Box<dyn ForwardPass>> {
        Ok(Box::new(model::load_command_r_from_gguf(model, config)?))
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
                description: "Final LayerNorm".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                // llama.cpp never creates a standalone `output.weight` for
                // this architecture — it always duplicates `token_embd`.
                description: "LM head / unembedding (always tied to token_embd.weight)".to_string(),
                required: false,
            },
        ];

        let layer_tensors = [
            (
                "blk.{i}.attn_norm.weight",
                "Pre-block LayerNorm (feeds both attention and FFN)",
                true,
            ),
            ("blk.{i}.attn_q.weight", "Query projection", true),
            ("blk.{i}.attn_k.weight", "Key projection", true),
            ("blk.{i}.attn_v.weight", "Value projection", true),
            (
                "blk.{i}.attn_output.weight",
                "Attention output projection",
                true,
            ),
            (
                "blk.{i}.attn_q_norm.weight",
                "Per-head Q normalization (Command-R+ only)",
                false,
            ),
            (
                "blk.{i}.attn_k_norm.weight",
                "Per-head K normalization (Command-R+ only)",
                false,
            ),
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
            MetadataValue::String("command-r".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal command-r config should parse")
    }

    #[test]
    fn test_command_r_arch_id() {
        let arch = CommandRArchitecture::new();
        assert_eq!(arch.arch_id(), "command-r");
    }

    #[test]
    fn test_command_r_tensor_names_is_non_empty() {
        let arch = CommandRArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_command_r_tensor_names_contains_token_embd() {
        let arch = CommandRArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_command_r_tensor_names_contains_required() {
        let arch = CommandRArchitecture::new();
        let names = arch.tensor_names();

        // Check required tensors are listed.
        let required_patterns: Vec<&str> = names
            .iter()
            .filter(|p| p.required)
            .map(|p| p.pattern.as_str())
            .collect();

        assert!(
            required_patterns.contains(&"token_embd.weight"),
            "missing token_embd.weight"
        );

        // `output.weight` is listed but NOT required: llama.cpp always ties
        // it to `token_embd.weight` for this architecture and never creates
        // a standalone tensor (see `load_lm_head` in `model.rs`).
        let output = names
            .iter()
            .find(|p| p.pattern == "output.weight")
            .expect("output.weight pattern should be present");
        assert!(
            !output.required,
            "output.weight should be optional (tied-embedding fallback)"
        );

        // Optional Q/K norm tensors should be listed but not required.
        let q_norm = names
            .iter()
            .find(|p| p.pattern == "blk.{i}.attn_q_norm.weight")
            .expect("q_norm pattern should be present");
        assert!(!q_norm.required, "q_norm should be optional");

        let k_norm = names
            .iter()
            .find(|p| p.pattern == "blk.{i}.attn_k_norm.weight")
            .expect("k_norm pattern should be present");
        assert!(!k_norm.required, "k_norm should be optional");
    }

    #[test]
    fn test_command_r_build_with_zero_heads_returns_config_error() {
        let arch = CommandRArchitecture::new();
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
    fn test_command_r_build_with_zero_hidden_size_returns_config_error() {
        let arch = CommandRArchitecture::new();
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
    fn test_command_r_build_with_valid_config_returns_missing_tensor_error() {
        let arch = CommandRArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
