//! BLOOM-specific model configuration.
//!
//! BLOOM uses standard LayerNorm (not RMSNorm), fused QKV projections,
//! ALiBi positional biases (no RoPE), and GELU FFN activations.
//!
//! All GGUF keys are prefixed with `bloom.*`.

use crate::config::ModelConfig;

/// BLOOM architecture configuration.
///
/// Derived from a [`ModelConfig`] parsed from GGUF metadata, plus sensible
/// defaults for any missing BLOOM-specific keys.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Number of attention heads (MHA, so `num_kv_heads = num_attention_heads`).
    pub num_attention_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// LayerNorm epsilon for numerical stability.
    pub layer_norm_epsilon: f32,
}

impl From<&ModelConfig> for BloomConfig {
    fn from(cfg: &ModelConfig) -> Self {
        Self {
            num_hidden_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            num_attention_heads: cfg.num_attention_heads,
            vocab_size: cfg.vocab_size,
            // BLOOM uses layer_norm_epsilon from rms_norm_eps field (reused for LN eps)
            layer_norm_epsilon: cfg.rms_norm_eps,
        }
    }
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            num_hidden_layers: 30,
            hidden_size: 4096,
            num_attention_heads: 32,
            vocab_size: 250880,
            layer_norm_epsilon: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_config_from_model_config() {
        let mc = ModelConfig {
            architecture: "bloom".to_string(),
            hidden_size: 64,
            num_layers: 2,
            num_attention_heads: 8,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            ..ModelConfig::default()
        };
        let bc = BloomConfig::from(&mc);
        assert_eq!(bc.hidden_size, 64);
        assert_eq!(bc.num_hidden_layers, 2);
        assert_eq!(bc.num_attention_heads, 8);
        assert_eq!(bc.vocab_size, 32);
    }
}
