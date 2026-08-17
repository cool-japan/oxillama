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
    /// LayerNorm epsilon (`dbrx.attention.layer_norm_epsilon`).
    ///
    /// DBRX normalises with **LayerNorm**, not RMSNorm: `build_dbrx` uses
    /// `LLM_NORM` for all three norms and the hparams block reads
    /// `LLM_KV_ATTENTION_LAYERNORM_EPS`.  `convert_hf_to_gguf.py::DbrxModel`
    /// writes `add_layer_norm_eps(1e-5)`, so no DBRX GGUF carries
    /// `dbrx.attention.layer_norm_rms_epsilon` — the key this config used to
    /// read, which therefore always fell through to its default.
    pub layer_norm_eps: f32,
    /// Symmetric clamp applied to the fused QKV projection
    /// (`dbrx.attention.clamp_kqv`, written by `add_clamp_kqv(clip_qkv)`).
    ///
    /// `build_dbrx` runs `ggml_clamp(cur, -f_clamp_kqv, +f_clamp_kqv)` right
    /// after the `wqkv` matmul.  The real checkpoint ships `clip_qkv = 8.0`.
    /// A non-positive value disables the clamp.
    pub clamp_kqv: f32,
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

        // `add_layer_norm_eps` → `{arch}.attention.layer_norm_epsilon`.  The
        // RMS spelling is accepted as a fallback only so hand-written fixtures
        // keep loading; no converter emits it for DBRX.
        let layer_norm_eps = metadata
            .get_f32("dbrx.attention.layer_norm_epsilon")
            .or_else(|_| metadata.get_f32("dbrx.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        // `add_clamp_kqv(attn_config["clip_qkv"])` → 8.0 for the real DBRX.
        let clamp_kqv = metadata.get_f32("dbrx.attention.clamp_kqv").unwrap_or(0.0);

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
            layer_norm_eps,
            clamp_kqv,
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

    /// `convert_hf_to_gguf.py::DbrxModel` writes `add_clamp_kqv(clip_qkv)` and
    /// `add_layer_norm_eps(1e-5)`.  Neither key was read before; `clamp_kqv`
    /// had no field at all, so `ggml_clamp(±8.0)` was simply never applied.
    #[test]
    fn reads_clamp_kqv_and_layer_norm_eps() {
        let mut store = MetadataStore::new();
        store.insert(
            "dbrx.attention.clamp_kqv".to_string(),
            MetadataValue::Float32(8.0),
        );
        store.insert(
            "dbrx.attention.layer_norm_epsilon".to_string(),
            MetadataValue::Float32(1e-6),
        );
        let cfg = DbrxConfig::from_metadata(&store);
        assert!(
            (cfg.clamp_kqv - 8.0).abs() < 1e-9,
            "clip_qkv must come from dbrx.attention.clamp_kqv, got {}",
            cfg.clamp_kqv
        );
        assert!(
            (cfg.layer_norm_eps - 1e-6).abs() < 1e-12,
            "eps must come from dbrx.attention.layer_norm_epsilon, got {}",
            cfg.layer_norm_eps
        );
    }

    /// Absent `clamp_kqv` must disable the clamp rather than clamp to zero.
    #[test]
    fn clamp_defaults_to_disabled() {
        let cfg = DbrxConfig::from_metadata(&MetadataStore::new());
        assert_eq!(cfg.clamp_kqv, 0.0);
    }
}
