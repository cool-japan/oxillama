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
    /// Multiplier applied to the token embedding (`grok.embedding_scale`).
    ///
    /// llama.cpp hard-codes `78.38367176906169` for `LLM_ARCH_GROK` and lets a
    /// GGUF key override it.  That constant is `sqrt(6144)` — the square root of
    /// Grok-1's hidden size — which is why omitting it scaled every hidden state
    /// down by ~78x before the first layer.
    pub embedding_scale: f32,
    /// Multiplier applied to the final logits (`grok.logit_scale`).
    ///
    /// llama.cpp default `0.5773502691896257` = `1/sqrt(3)`.
    pub logit_scale: f32,
    /// Pre-softmax attention score multiplier (`grok.attention.output_scale`).
    ///
    /// llama.cpp default `0.08838834764831845` = `1/sqrt(128)`.  Note this is a
    /// **constant**, not `1/sqrt(head_dim)`: `build_grok` passes `kq_scale =
    /// 1.0f` to `build_attn` and the multiplier is folded into the tanh
    /// soft-cap at `llm_graph_context::build_attn_mha`.
    pub attn_output_scale: f32,
    /// Attention logit soft-cap (`grok.attn_logit_softcapping`, default 30.0).
    ///
    /// `kq = cap * tanh(kq * attn_output_scale / cap)`.  A non-positive value
    /// disables the cap (and then `attn_output_scale` is applied directly).
    pub attn_logit_softcapping: f32,
    /// Final logit soft-cap (`grok.final_logit_softcapping`).
    ///
    /// **0.0 for Grok-1** — llama.cpp comments "no final_logit_softcapping in
    /// grok-1" — but the key exists for later Grok releases.
    pub final_logit_softcapping: f32,
    /// Router logit soft-cap (`grok.router_logit_softcapping`, default 30.0).
    ///
    /// Parsed for completeness.  The reference **reads this key and never uses
    /// it**: `f_router_logit_softcapping` appears only in the hparams loader and
    /// the model saver, never in `build_moe_ffn`.  It is therefore recorded here
    /// and deliberately not applied.
    pub router_logit_softcapping: f32,
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

        // Defaults straight out of `case LLM_ARCH_GROK:` in
        // `llama_model::load_hparams`; each is overridable by a GGUF key.
        let embedding_scale = metadata
            .get_f32("grok.embedding_scale")
            .unwrap_or(78.383_67);
        let logit_scale = metadata.get_f32("grok.logit_scale").unwrap_or(0.577_350_26);
        let attn_output_scale = metadata
            .get_f32("grok.attention.output_scale")
            .unwrap_or(0.088_388_35);
        let attn_logit_softcapping = metadata
            .get_f32("grok.attn_logit_softcapping")
            .unwrap_or(30.0);
        let router_logit_softcapping = metadata
            .get_f32("grok.router_logit_softcapping")
            .unwrap_or(30.0);
        let final_logit_softcapping = metadata
            .get_f32("grok.final_logit_softcapping")
            .unwrap_or(0.0);

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
            embedding_scale,
            logit_scale,
            attn_output_scale,
            attn_logit_softcapping,
            final_logit_softcapping,
            router_logit_softcapping,
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

    /// Grok's distinctive scalars had no fields at all, so the embedding was
    /// ~78x too small and the logits ~1.73x too large.
    #[test]
    fn distinctive_scales_default_to_the_llama_cpp_constants() {
        let cfg = GrokConfig::from_metadata(&MetadataStore::new());
        assert!(
            (cfg.embedding_scale - 78.383_67).abs() < 1e-3,
            "embedding_scale = {}",
            cfg.embedding_scale
        );
        assert!(
            (cfg.logit_scale - 0.577_350_26).abs() < 1e-6,
            "logit_scale = {}",
            cfg.logit_scale
        );
        assert!(
            (cfg.attn_output_scale - 0.088_388_35).abs() < 1e-7,
            "attn_output_scale = {}",
            cfg.attn_output_scale
        );
        assert!((cfg.attn_logit_softcapping - 30.0).abs() < 1e-6);
        assert!((cfg.router_logit_softcapping - 30.0).abs() < 1e-6);
        assert_eq!(
            cfg.final_logit_softcapping, 0.0,
            "grok-1 has no final logit soft-cap"
        );
    }

    #[test]
    fn scales_can_be_overridden_by_gguf() {
        let mut store = MetadataStore::new();
        store.insert(
            "grok.embedding_scale".to_string(),
            MetadataValue::Float32(2.0),
        );
        store.insert("grok.logit_scale".to_string(), MetadataValue::Float32(3.0));
        store.insert(
            "grok.attention.output_scale".to_string(),
            MetadataValue::Float32(0.25),
        );
        let cfg = GrokConfig::from_metadata(&store);
        assert!((cfg.embedding_scale - 2.0).abs() < 1e-6);
        assert!((cfg.logit_scale - 3.0).abs() < 1e-6);
        assert!((cfg.attn_output_scale - 0.25).abs() < 1e-6);
    }
}
