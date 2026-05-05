//! Phi-3.5-MoE architecture configuration.
//!
//! Phi-3.5-MoE extends the Phi-3 architecture with Sparse Mixture-of-Experts
//! FFN layers, using the same Phi-style fused QKV projections and partial RoPE.
//!
//! Key differences from Phi-3 (dense):
//! - `num_experts`: total number of expert FFN modules per layer.
//! - `num_experts_per_tok`: how many experts are selected per token (default 2).
//! - `partial_rotary_factor`: fraction of head_dim to apply RoPE (default 0.5 for Phi-3.5).

use crate::config::ModelConfig;

/// Phi-3.5-MoE specific configuration.
///
/// Derived from a [`ModelConfig`] plus GGUF metadata keys.
#[derive(Debug, Clone)]
pub struct PhiMoeConfig {
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA; equals `num_attention_heads` for MHA).
    pub num_key_value_heads: usize,
    /// Intermediate size per expert (FFN hidden dimension within each expert).
    pub intermediate_size: usize,
    /// Total number of expert FFNs per layer.
    pub num_experts: usize,
    /// Top-K experts activated per token (default 2 for Phi-3.5-MoE).
    pub num_experts_per_tok: usize,
    /// Fraction of `head_dim` to apply RoPE (default 0.5 for Phi-3.5-MoE).
    ///
    /// Only the first `round(partial_rotary_factor * head_dim)` dimensions of
    /// each Q/K head vector receive rotary positional embeddings. The remainder
    /// are passed through unchanged.
    pub partial_rotary_factor: f32,
}

impl From<&ModelConfig> for PhiMoeConfig {
    fn from(cfg: &ModelConfig) -> Self {
        Self {
            num_hidden_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_kv_heads,
            intermediate_size: cfg.intermediate_size,
            num_experts: cfg.num_experts.max(1),
            num_experts_per_tok: cfg.num_experts_used.max(1),
            partial_rotary_factor: 0.5, // Phi-3.5-MoE default
        }
    }
}

impl PhiMoeConfig {
    /// Return the number of rotated dimensions per head (always even for RoPE pairing).
    pub fn rotary_dims(&self, head_dim: usize) -> usize {
        let raw = (self.partial_rotary_factor * head_dim as f32) as usize;
        // Must be even for RoPE pair rotation.
        (raw & !1).min(head_dim)
    }
}

impl Default for PhiMoeConfig {
    fn default() -> Self {
        Self {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 6400,
            num_experts: 16,
            num_experts_per_tok: 2,
            partial_rotary_factor: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_moe_config_from_model_config() {
        let mc = ModelConfig {
            architecture: "phimoe".to_string(),
            hidden_size: 64,
            num_layers: 1,
            num_attention_heads: 8,
            num_kv_heads: 4,
            intermediate_size: 128,
            num_experts: 4,
            num_experts_used: 2,
            ..ModelConfig::default()
        };
        let pc = PhiMoeConfig::from(&mc);
        assert_eq!(pc.hidden_size, 64);
        assert_eq!(pc.num_attention_heads, 8);
        assert_eq!(pc.num_key_value_heads, 4);
        assert_eq!(pc.num_experts, 4);
        assert_eq!(pc.num_experts_per_tok, 2);
        assert!((pc.partial_rotary_factor - 0.5).abs() < 1e-6);
    }

    #[test]
    fn phi_moe_rotary_dims_half_factor() {
        let cfg = PhiMoeConfig::default();
        // 0.5 * head_dim(= 4096/32=128) = 64, already even
        let head_dim = 128usize;
        let rot = cfg.rotary_dims(head_dim);
        assert_eq!(rot, 64);
    }

    #[test]
    fn phi_moe_rotary_dims_is_always_even() {
        let cfg = PhiMoeConfig {
            partial_rotary_factor: 0.3,
            ..PhiMoeConfig::default()
        };
        let head_dim = 64;
        let rot = cfg.rotary_dims(head_dim);
        assert_eq!(rot % 2, 0, "rotary_dims must always be even, got {rot}");
    }
}
