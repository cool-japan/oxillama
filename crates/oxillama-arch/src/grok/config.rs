//! Grok-1 architecture configuration.
//!
//! Parsed from GGUF metadata keys prefixed with `grok.*`.

use oxillama_gguf::MetadataStore;

/// Configuration for a Grok-1 model.
///
/// Grok-1 uses Mixture-of-Experts with 8 total experts and top-2 routing.
/// Attention is standard grouped-query attention with RoPE theta = 1_000_000.
#[derive(Debug, Clone)]
pub struct GrokConfig {
    /// Hidden (embedding) dimension.
    pub hidden_size: usize,
    /// Number of transformer layers (blocks).
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key-value heads.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Total number of routed MoE experts (typically 8).
    pub expert_count: usize,
    /// Number of experts activated per token (top-k; typically 2).
    pub expert_used_count: usize,
    /// FFN intermediate size for each expert.
    pub ffn_hidden_size: usize,
    /// RoPE base frequency (Grok-1 uses 1_000_000.0).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl GrokConfig {
    /// Parse a `GrokConfig` from GGUF metadata.
    ///
    /// Reads `grok.*` keys with sensible defaults for any missing entries.
    /// Grok-1 defaults: 8 experts, top-2, rope_theta = 1_000_000.
    pub fn from_metadata(metadata: &MetadataStore) -> Self {
        let hidden_size = metadata
            .get_u32("grok.embedding_length")
            .map(|v| v as usize)
            .unwrap_or(6144);

        let num_layers = metadata
            .get_u32("grok.block_count")
            .map(|v| v as usize)
            .unwrap_or(64);

        let num_heads = metadata
            .get_u32("grok.attention.head_count")
            .map(|v| v as usize)
            .unwrap_or(48);

        let num_kv_heads = metadata
            .get_u32("grok.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let head_dim = hidden_size.checked_div(num_heads).unwrap_or(128);

        let vocab_size = metadata
            .get_u32("grok.vocab_size")
            .or_else(|_| metadata.get_u32("tokenizer.ggml.tokens.length"))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let max_seq_len = metadata
            .get_u32("grok.context_length")
            .map(|v| v as usize)
            .unwrap_or(8192);

        let expert_count = metadata
            .get_u32("grok.expert_count")
            .map(|v| v as usize)
            .unwrap_or(8);

        let expert_used_count = metadata
            .get_u32("grok.expert_used_count")
            .map(|v| v as usize)
            .unwrap_or(2);

        let ffn_hidden_size = metadata
            .get_u32("grok.feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4 / expert_count.max(1));

        // Grok-1 uses a very large rope_theta (1e6) by default.
        let rope_theta = metadata
            .get_f32("grok.rope.freq_base")
            .unwrap_or(1_000_000.0);

        let rms_norm_eps = metadata
            .get_f32("grok.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        Self {
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            expert_count,
            expert_used_count,
            ffn_hidden_size,
            rope_theta,
            rms_norm_eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::MetadataValue;

    #[test]
    fn defaults_are_grok1() {
        let store = MetadataStore::new();
        let cfg = GrokConfig::from_metadata(&store);
        assert_eq!(cfg.expert_count, 8, "default Grok-1 expert count is 8");
        assert_eq!(cfg.expert_used_count, 2, "default Grok-1 top-k is 2");
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1.0,
            "default Grok-1 rope_theta is 1e6"
        );
    }

    #[test]
    fn parses_custom_fields() {
        let mut store = MetadataStore::new();
        store.insert(
            "grok.embedding_length".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert("grok.block_count".to_string(), MetadataValue::Uint32(2));
        store.insert(
            "grok.attention.head_count".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert("grok.expert_count".to_string(), MetadataValue::Uint32(8));
        store.insert(
            "grok.expert_used_count".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert("grok.vocab_size".to_string(), MetadataValue::Uint32(32));
        let cfg = GrokConfig::from_metadata(&store);
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.expert_count, 8);
        assert_eq!(cfg.expert_used_count, 2);
        assert_eq!(cfg.head_dim, 16); // 32 / 2
    }
}
