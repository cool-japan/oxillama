//! Phi-3.5-MoE architecture configuration.
//!
//! Phi-3.5-MoE extends the Phi-3 architecture with Sparse Mixture-of-Experts
//! FFN layers, using the same Phi-style fused QKV projections and partial RoPE.
//!
//! Key differences from Phi-3 (dense):
//! - `num_experts`: total number of expert FFN modules per layer.
//! - `num_experts_per_tok`: how many experts are selected per token (default 2).
//!
//! The number of RoPE-rotated dimensions per head is read directly from GGUF
//! metadata (`{arch}.rope.dimension_count`, defaulting to the full
//! `head_dim`) by `load_phi_moe_from_gguf` / `PhiMoeModel::new`, not derived
//! from a `partial_rotary_factor` here — no GGUF converter ever writes that
//! key (see `super::model::load_phi_moe_from_gguf` for the verified reference
//! trail). A `partial_rotary_factor` field used to live on this struct but
//! was always overwritten with a hardcoded `0.5` and never actually consulted
//! by `PhiMoeModel::new` (which took its own `partial_rotary_factor`
//! parameter instead) — it has been removed as dead configuration.

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
        }
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
    }

    /// `partial_rotary_factor`/`rotary_dims()` were removed (G13): the field
    /// was always overwritten with a hardcoded `0.5` and never actually
    /// consulted by `PhiMoeModel::new`, which took its own
    /// `partial_rotary_factor` parameter instead — dead configuration.
    /// `load_phi_moe_from_gguf` now reads `{arch}.rope.dimension_count`
    /// directly. This test exists to document the removal for anyone
    /// tempted to re-add it.
    #[test]
    fn phi_moe_config_has_no_rotary_dims_field() {
        let cfg = PhiMoeConfig::default();
        // Compiles iff `PhiMoeConfig` has exactly these fields — a stray
        // `partial_rotary_factor` would make this a "missing field" error,
        // and an extra field would make it a "field not covered" error.
        let PhiMoeConfig {
            num_hidden_layers: _,
            hidden_size: _,
            num_attention_heads: _,
            num_key_value_heads: _,
            intermediate_size: _,
            num_experts: _,
            num_experts_per_tok: _,
        } = cfg;
    }
}
