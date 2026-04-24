//! MiniCPM-specific configuration parsed from GGUF metadata.
//!
//! MiniCPM is a scaled-embedding variant of LLaMA where the embedding
//! vectors are multiplied by `sqrt(hidden_size)` (or more precisely by
//! `hidden_size / dim_model_base`) before the first layer.

use crate::config::ModelConfig;
use crate::error::ArchResult;

/// MiniCPM-specific hyperparameters extracted from GGUF metadata.
///
/// The generic [`ModelConfig`] fields hold most values; this struct adds the
/// MiniCPM-specific `dim_model_base` and derived `embedding_scale`.
#[derive(Debug, Clone)]
pub struct MiniCpmConfig {
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
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Base dimension used to compute the embedding scale.
    ///
    /// Corresponds to `minicpm.dim_model_base` in GGUF metadata.
    /// Defaults to `hidden_size` when absent (scale → 1.0).
    pub dim_model_base: usize,
    /// Input embedding scale factor: `hidden_size / dim_model_base`.
    ///
    /// Applied once to the token embeddings before the first transformer layer.
    pub embedding_scale: f32,
}

impl MiniCpmConfig {
    /// Parse a [`MiniCpmConfig`] from a generic [`ModelConfig`].
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

        let rope_freq_base = if config.rope_freq_base > 0.0 {
            config.rope_freq_base
        } else {
            10_000.0
        };

        // `dim_model_base` is stored in extra metadata that ModelConfig doesn't
        // carry directly — we default it to `hidden_size` so that when the
        // key is absent the scale is 1.0.  Callers that parse GGUF directly
        // can adjust this field after construction.
        let dim_model_base = hidden_size;
        let embedding_scale = 1.0; // hidden_size / dim_model_base when both == hidden_size

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
            dim_model_base,
            embedding_scale,
        })
    }

    /// Construct a config with an explicit `dim_model_base`.
    ///
    /// `embedding_scale` is recomputed as `hidden_size as f32 / dim_model_base as f32`.
    pub fn with_dim_model_base(mut self, dim_model_base: usize) -> Self {
        let base = dim_model_base.max(1);
        self.dim_model_base = base;
        self.embedding_scale = self.hidden_size as f32 / base as f32;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue};

    fn make_minicpm_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("minicpm".to_string()),
        );
        store.insert(
            "minicpm.embedding_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        store.insert("minicpm.block_count".to_string(), MetadataValue::Uint32(40));
        store.insert(
            "minicpm.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "minicpm.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "minicpm.feed_forward_length".to_string(),
            MetadataValue::Uint32(5504),
        );
        store.insert(
            "minicpm.rope.freq_base".to_string(),
            MetadataValue::Float32(10000.0),
        );
        store
    }

    #[test]
    fn test_config_from_metadata() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_model_config(&model_cfg).expect("minicpm config");

        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 32);
        assert_eq!(cfg.intermediate_size, 5504);
        assert!(cfg.rope_freq_base > 0.0);
        assert!(cfg.norm_eps > 0.0);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn test_embedding_scale_default_is_one() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_model_config(&model_cfg).expect("minicpm config");
        // Default: dim_model_base == hidden_size → scale == 1.0
        assert!((cfg.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_scale_with_dim_model_base() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_model_config(&model_cfg)
            .expect("minicpm config")
            .with_dim_model_base(256);

        // 2048 / 256 = 8.0
        assert!((cfg.embedding_scale - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_has_positive_head_dim() {
        let store = make_minicpm_metadata();
        let model_cfg = ModelConfig::from_metadata(&store).expect("model config");
        let cfg = MiniCpmConfig::from_model_config(&model_cfg).expect("minicpm config");
        assert!(cfg.head_dim > 0);
    }
}
