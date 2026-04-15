//! StarCoder model architecture implementation.
//!
//! StarCoder (BigCode) uses a GPT-BigCode architecture, which differs
//! significantly from LLaMA:
//!
//! - **LayerNorm** (not RMSNorm): mean subtraction + variance normalisation
//! - **GELU activation** (not SwiGLU): gate-free FFN
//! - **Absolute position embeddings** (not RoPE)
//! - **Multi-Query Attention (MQA)**: single K/V head shared by all Q heads
//! - **Fused QKV projection**: `[(num_heads+2)*head_dim, hidden_size]`
//! - **Bias on all projections and norms**
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight` — Token embedding matrix
//! - `position_embd.weight` — Learned absolute position embeddings
//! - `output_norm.weight`, `output_norm.bias` — Final LayerNorm
//! - `output.weight` — LM head
//! - Per layer:
//!   - `blk.{i}.attn_norm.weight`, `blk.{i}.attn_norm.bias`
//!   - `blk.{i}.attn_qkv.weight`, `blk.{i}.attn_qkv.bias`
//!   - `blk.{i}.attn_out.weight`, `blk.{i}.attn_out.bias`
//!   - `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_norm.bias`
//!   - `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_up.bias`
//!   - `blk.{i}.ffn_down.weight`, `blk.{i}.ffn_down.bias`

mod model;

pub use model::{load_starcoder_from_gguf, StarcoderLayer, StarcoderModel};

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, ModelArchitecture, TensorNamePattern};
use oxillama_gguf::TensorStore;

/// StarCoder architecture plugin.
pub struct StarcoderArchitecture;

impl StarcoderArchitecture {
    /// Create a new `StarcoderArchitecture` plugin.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StarcoderArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelArchitecture for StarcoderArchitecture {
    fn arch_id(&self) -> &str {
        "starcoder"
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

        // Full loading requires a GgufModel (use load_starcoder_from_gguf).
        Err(ArchError::MissingTensor {
            name: "token_embd.weight (use StarcoderModel::from_gguf for full loading)".to_string(),
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
                pattern: "position_embd.weight".to_string(),
                description: "Learned absolute position embeddings".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.weight".to_string(),
                description: "Final LayerNorm scale".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output_norm.bias".to_string(),
                description: "Final LayerNorm bias".to_string(),
                required: true,
            },
            TensorNamePattern {
                pattern: "output.weight".to_string(),
                description: "LM head / unembedding".to_string(),
                required: true,
            },
        ];

        let layer_tensors = [
            ("blk.{i}.attn_norm.weight", "Pre-attention LayerNorm scale"),
            ("blk.{i}.attn_norm.bias", "Pre-attention LayerNorm bias"),
            ("blk.{i}.attn_qkv.weight", "Fused QKV projection weight"),
            ("blk.{i}.attn_qkv.bias", "Fused QKV projection bias"),
            (
                "blk.{i}.attn_out.weight",
                "Attention output projection weight",
            ),
            ("blk.{i}.attn_out.bias", "Attention output projection bias"),
            ("blk.{i}.ffn_norm.weight", "Pre-FFN LayerNorm scale"),
            ("blk.{i}.ffn_norm.bias", "Pre-FFN LayerNorm bias"),
            ("blk.{i}.ffn_up.weight", "FFN up projection weight"),
            ("blk.{i}.ffn_up.bias", "FFN up projection bias"),
            ("blk.{i}.ffn_down.weight", "FFN down projection weight"),
            ("blk.{i}.ffn_down.bias", "FFN down projection bias"),
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
            MetadataValue::String("starcoder".to_string()),
        );
        ModelConfig::from_metadata(&store).expect("minimal starcoder config should parse")
    }

    #[test]
    fn test_starcoder_arch_id() {
        let arch = StarcoderArchitecture::new();
        assert_eq!(arch.arch_id(), "starcoder");
    }

    #[test]
    fn test_starcoder_tensor_names_is_non_empty() {
        let arch = StarcoderArchitecture::new();
        let names = arch.tensor_names();
        assert!(!names.is_empty(), "tensor_names should not be empty");
    }

    #[test]
    fn test_starcoder_tensor_names_contains_token_embd() {
        let arch = StarcoderArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        assert!(
            patterns.iter().any(|p| p.contains("token_embd")),
            "should contain a token embedding pattern"
        );
    }

    #[test]
    fn test_starcoder_tensor_names_has_position_embd() {
        let arch = StarcoderArchitecture::new();
        let names = arch.tensor_names();
        let patterns: Vec<&str> = names.iter().map(|p| p.pattern.as_str()).collect();
        // StarCoder uses absolute position embeddings, not RoPE
        assert!(
            patterns.iter().any(|p| p.contains("position_embd")),
            "StarCoder should have a position_embd pattern for absolute positional encodings"
        );
    }

    #[test]
    fn test_starcoder_tensor_names_required() {
        let arch = StarcoderArchitecture::new();
        let names = arch.tensor_names();

        let required_patterns: Vec<&str> = names
            .iter()
            .filter(|p| p.required)
            .map(|p| p.pattern.as_str())
            .collect();

        // Global tensors
        assert!(
            required_patterns.contains(&"token_embd.weight"),
            "missing token_embd.weight"
        );
        assert!(
            required_patterns.contains(&"position_embd.weight"),
            "missing position_embd.weight"
        );
        assert!(
            required_patterns.contains(&"output_norm.bias"),
            "missing output_norm.bias"
        );

        // Per-layer tensors with bias
        assert!(
            required_patterns.contains(&"blk.{i}.attn_qkv.weight"),
            "missing attn_qkv.weight"
        );
        assert!(
            required_patterns.contains(&"blk.{i}.attn_qkv.bias"),
            "missing attn_qkv.bias"
        );
        assert!(
            required_patterns.contains(&"blk.{i}.ffn_up.bias"),
            "missing ffn_up.bias"
        );
    }

    #[test]
    fn test_starcoder_build_with_zero_heads_returns_config_error() {
        let arch = StarcoderArchitecture::new();
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
    fn test_starcoder_build_with_zero_hidden_size_returns_config_error() {
        let arch = StarcoderArchitecture::new();
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
    fn test_starcoder_build_with_valid_config_returns_missing_tensor_error() {
        let arch = StarcoderArchitecture::new();
        let config = make_config();
        let tensors = TensorStore::new();
        let result = arch.build(&config, &tensors);
        assert!(
            matches!(result, Err(ArchError::MissingTensor { .. })),
            "valid config with empty tensor store should return MissingTensor"
        );
    }
}
