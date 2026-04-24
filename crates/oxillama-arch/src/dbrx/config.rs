//! DBRX architecture configuration.
//!
//! Parsed from GGUF metadata keys prefixed with `dbrx.*`.

use oxillama_gguf::MetadataStore;

/// Configuration for a DBRX model.
///
/// DBRX uses fine-grained Mixture-of-Experts with 16 total experts and
/// top-4 routing per token. Attention is standard multi-head (no MLA).
#[derive(Debug, Clone)]
pub struct DbrxConfig {
    /// Hidden (embedding) dimension.
    pub hidden_size: usize,
    /// Number of transformer layers (blocks).
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (equals `num_heads` for DBRX — standard MHA).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Total number of routed MoE experts (typically 16).
    pub expert_count: usize,
    /// Number of experts activated per token (top-k; typically 4).
    pub expert_used_count: usize,
    /// FFN intermediate size for each expert.
    pub ffn_hidden_size: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl DbrxConfig {
    /// Parse a `DbrxConfig` from GGUF metadata.
    ///
    /// Reads `dbrx.*` keys with sensible defaults for any missing entries.
    pub fn from_metadata(metadata: &MetadataStore) -> Self {
        let hidden_size = metadata
            .get_u32("dbrx.embedding_length")
            .map(|v| v as usize)
            .unwrap_or(6144);

        let num_layers = metadata
            .get_u32("dbrx.block_count")
            .map(|v| v as usize)
            .unwrap_or(40);

        let num_heads = metadata
            .get_u32("dbrx.attention.head_count")
            .map(|v| v as usize)
            .unwrap_or(48);

        let num_kv_heads = metadata
            .get_u32("dbrx.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let head_dim = hidden_size.checked_div(num_heads).unwrap_or(128);

        let vocab_size = metadata
            .get_u32("dbrx.vocab_size")
            .or_else(|_| metadata.get_u32("tokenizer.ggml.tokens.length"))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let max_seq_len = metadata
            .get_u32("dbrx.context_length")
            .map(|v| v as usize)
            .unwrap_or(32768);

        let expert_count = metadata
            .get_u32("dbrx.expert_count")
            .map(|v| v as usize)
            .unwrap_or(16);

        let expert_used_count = metadata
            .get_u32("dbrx.expert_used_count")
            .map(|v| v as usize)
            .unwrap_or(4);

        let ffn_hidden_size = metadata
            .get_u32("dbrx.feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4 / expert_count);

        let rope_theta = metadata.get_f32("dbrx.rope.freq_base").unwrap_or(10000.0);

        let rms_norm_eps = metadata
            .get_f32("dbrx.attention.layer_norm_rms_epsilon")
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
    fn defaults_are_reasonable() {
        let store = MetadataStore::new();
        let cfg = DbrxConfig::from_metadata(&store);
        assert_eq!(cfg.expert_count, 16, "default DBRX expert count is 16");
        assert_eq!(cfg.expert_used_count, 4, "default DBRX top-k is 4");
    }

    #[test]
    fn parses_custom_fields() {
        let mut store = MetadataStore::new();
        store.insert(
            "dbrx.embedding_length".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert("dbrx.block_count".to_string(), MetadataValue::Uint32(2));
        store.insert(
            "dbrx.attention.head_count".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert("dbrx.expert_count".to_string(), MetadataValue::Uint32(4));
        store.insert(
            "dbrx.expert_used_count".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert("dbrx.vocab_size".to_string(), MetadataValue::Uint32(32));
        let cfg = DbrxConfig::from_metadata(&store);
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.expert_count, 4);
        assert_eq!(cfg.expert_used_count, 2);
        assert_eq!(cfg.head_dim, 16); // 32 / 2
    }
}
