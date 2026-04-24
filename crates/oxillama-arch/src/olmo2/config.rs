//! OLMo2-specific configuration parsed from GGUF metadata.
//!
//! OLMo2 uses post-norm style: layer norms are applied AFTER attention and FFN
//! outputs (not before, as in pre-norm LLaMA). It also uses per-head QK-norm.

use crate::config::ModelConfig;
use crate::error::ArchResult;

/// OLMo2-specific hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Olmo2Config {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden size / model dimension.
    pub hidden_size: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA).
    pub n_kv_heads: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_context_length: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency (default 500 000.0 for OLMo2).
    pub rope_freq_base: f32,
    /// Dimension of each attention head.
    pub head_dim: usize,
}

impl Olmo2Config {
    /// Parse an [`Olmo2Config`] from a generic [`ModelConfig`].
    pub fn from_model_config(config: &ModelConfig) -> ArchResult<Self> {
        let n_heads = config.num_attention_heads.max(1);
        let n_kv_heads = config.num_kv_heads.max(1);
        let n_layers = config.num_layers.max(1);
        let hidden_size = config.hidden_size.max(1);
        let vocab_size = config.vocab_size.max(1);
        let intermediate_size = config.intermediate_size.max(1);
        let max_context_length = config.max_context_length.max(1);
        let head_dim = hidden_size.checked_div(n_heads).unwrap_or(64).max(1);
        let norm_eps = config.rms_norm_eps.max(1e-9);

        // OLMo2 default RoPE freq base is 500 000.0; ModelConfig default is
        // 10 000.0. Accept whatever the metadata gives us.
        let rope_freq_base = if config.rope_freq_base > 0.0 {
            config.rope_freq_base
        } else {
            500_000.0
        };

        Ok(Self {
            n_layers,
            hidden_size,
            n_heads,
            n_kv_heads,
            intermediate_size,
            vocab_size,
            max_context_length,
            norm_eps,
            rope_freq_base,
            head_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue};

    fn make_olmo2_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("olmo2".to_string()),
        );
        store.insert(
            "olmo2.embedding_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        store.insert("olmo2.block_count".to_string(), MetadataValue::Uint32(16));
        store.insert(
            "olmo2.attention.head_count".to_string(),
            MetadataValue::Uint32(16),
        );
        store.insert(
            "olmo2.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        store.insert(
            "olmo2.feed_forward_length".to_string(),
            MetadataValue::Uint32(8192),
        );
        store.insert(
            "olmo2.rope.freq_base".to_string(),
            MetadataValue::Float32(500_000.0),
        );
        store
    }

    #[test]
    fn test_config_from_metadata() {
        let store = make_olmo2_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = Olmo2Config::from_model_config(&model_cfg).expect("olmo2 config");

        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_layers, 16);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.head_dim, 128);
        assert!(cfg.rope_freq_base > 0.0);
    }

    #[test]
    fn test_default_rope_freq_base_is_positive() {
        // Verify the default is applied when metadata is absent
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("olmo2".to_string()),
        );
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = Olmo2Config::from_model_config(&model_cfg).expect("olmo2 config");
        assert!(cfg.rope_freq_base > 0.0);
    }

    #[test]
    fn test_head_dim_computed() {
        let store = make_olmo2_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = Olmo2Config::from_model_config(&model_cfg).expect("olmo2 config");
        assert_eq!(cfg.head_dim, cfg.hidden_size / cfg.n_heads);
    }

    #[test]
    fn test_gqa_kv_heads_less_than_q_heads() {
        let store = make_olmo2_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = Olmo2Config::from_model_config(&model_cfg).expect("olmo2 config");
        assert!(cfg.n_kv_heads <= cfg.n_heads);
    }
}
