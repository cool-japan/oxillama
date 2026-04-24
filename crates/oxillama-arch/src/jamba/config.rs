//! Configuration for the Jamba hybrid (LLaMA × Mamba-2) architecture.
//!
//! Jamba interleaves LLaMA-style attention blocks with Mamba-2-style SSM blocks
//! on a configurable period. Every `attn_layer_period`-th layer (starting at
//! `attn_layer_offset`) uses attention; all others use SSM.
//!
//! Reference: AI21 Labs "Jamba: A Hybrid Transformer-Mamba Language Model" (2024).

/// Configuration for a Jamba hybrid model.
#[derive(Debug, Clone)]
pub struct JambaConfig {
    /// Total number of transformer/SSM blocks.
    pub n_layers: usize,
    /// Period: every `attn_layer_period`-th layer is an attention block.
    /// Default: 8 (same as the reference Jamba-1.5 model).
    pub attn_layer_period: usize,
    /// Offset: first layer index that is an attention block.
    /// Default: 0 (index 0 is attention, indices 1..period-1 are SSM).
    pub attn_layer_offset: usize,
    /// Number of MoE experts per FFN layer (0 = dense FFN).
    pub expert_count: usize,
    /// MoE top-k routing (active experts per token). Default: 2.
    pub expert_top_k: usize,
    /// SSM state dimension `d_state` for SSM layers.
    pub d_state: usize,
    /// SSM inner dimension `d_inner` (= d_model * expand) for SSM layers.
    pub d_inner: usize,
    /// Conv kernel width `d_conv` for SSM conv layers. Default: 4.
    pub d_conv: usize,
    /// Hidden size shared across all layers.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA). Defaults to `num_attention_heads`.
    pub num_kv_heads: usize,
    /// Intermediate size for attention-block FFN.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub max_context_length: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl JambaConfig {
    /// Parse configuration from GGUF metadata.
    pub fn from_metadata(metadata: &oxillama_gguf::MetadataStore) -> Self {
        let hidden_size = metadata
            .get_u32("jamba.embedding_length")
            .or_else(|_| metadata.get_u32("jamba.d_model"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let n_layers = metadata
            .get_u32("jamba.block_count")
            .map(|v| v as usize)
            .unwrap_or(4);

        let attn_layer_period = metadata
            .get_u32("jamba.attention_layer_period")
            .map(|v| v as usize)
            .unwrap_or(2);

        let attn_layer_offset = metadata
            .get_u32("jamba.attention_layer_offset")
            .map(|v| v as usize)
            .unwrap_or(0);

        let expert_count = metadata
            .get_u32("jamba.expert_count")
            .map(|v| v as usize)
            .unwrap_or(0);

        let expert_top_k = metadata
            .get_u32("jamba.expert_used_count")
            .map(|v| v as usize)
            .unwrap_or(2);

        let d_state = metadata
            .get_u32("jamba.ssm_state_size")
            .map(|v| v as usize)
            .unwrap_or(8);

        let d_inner = metadata
            .get_u32("jamba.ssm_inner_size")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 2);

        let d_conv = metadata
            .get_u32("jamba.ssm_conv_kernel")
            .map(|v| v as usize)
            .unwrap_or(4);

        let num_attention_heads = metadata
            .get_u32("jamba.attention.head_count")
            .map(|v| v as usize)
            .unwrap_or(2);

        let num_kv_heads = metadata
            .get_u32("jamba.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let intermediate_size = metadata
            .get_u32("jamba.feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        let vocab_size = metadata
            .get_u32("jamba.vocab_size")
            .or_else(|_| metadata.get_u32("tokenizer.ggml.tokens.length"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let max_context_length = metadata
            .get_u32("jamba.context_length")
            .map(|v| v as usize)
            .unwrap_or(512);

        let rms_norm_eps = metadata
            .get_f32("jamba.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        Self {
            n_layers,
            attn_layer_period,
            attn_layer_offset,
            expert_count,
            expert_top_k,
            d_state,
            d_inner,
            d_conv,
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            max_context_length,
            rms_norm_eps,
        }
    }
}

/// What kind of block occupies layer position `i` in a Jamba model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// LLaMA-style attention + SwiGLU FFN block.
    Attention,
    /// Mamba-2 SSM block.
    Ssm,
}

impl JambaConfig {
    /// Determine the kind of block at layer index `layer_idx`.
    ///
    /// A layer is an attention block iff
    /// `(layer_idx + attn_layer_offset) % attn_layer_period == 0`.
    pub fn layer_kind(&self, layer_idx: usize) -> LayerKind {
        let period = self.attn_layer_period.max(1);
        let pos = (layer_idx + self.attn_layer_offset) % period;
        if pos == 0 {
            LayerKind::Attention
        } else {
            LayerKind::Ssm
        }
    }

    /// Return the layer-kind pattern for all layers.
    pub fn layer_pattern(&self) -> Vec<LayerKind> {
        (0..self.n_layers).map(|i| self.layer_kind(i)).collect()
    }

    /// Count of attention-type layers.
    pub fn attn_layer_count(&self) -> usize {
        self.layer_pattern()
            .iter()
            .filter(|&&k| k == LayerKind::Attention)
            .count()
    }

    /// Count of SSM-type layers.
    pub fn ssm_layer_count(&self) -> usize {
        self.layer_pattern()
            .iter()
            .filter(|&&k| k == LayerKind::Ssm)
            .count()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> JambaConfig {
        JambaConfig {
            n_layers: 16,
            attn_layer_period: 8,
            attn_layer_offset: 0,
            expert_count: 0,
            expert_top_k: 2,
            d_state: 8,
            d_inner: 32,
            d_conv: 4,
            hidden_size: 16,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 64,
            vocab_size: 32,
            max_context_length: 512,
            rms_norm_eps: 1e-5,
        }
    }

    /// 16-layer model, period=8, offset=0: layers 0 and 8 are attention.
    #[test]
    fn jamba_layer_pattern_default_n8() {
        let cfg = minimal_config();
        let pattern = cfg.layer_pattern();

        assert_eq!(
            pattern[0],
            LayerKind::Attention,
            "layer 0 must be attention"
        );
        assert_eq!(pattern[1], LayerKind::Ssm, "layer 1 must be SSM");
        assert_eq!(pattern[7], LayerKind::Ssm, "layer 7 must be SSM");
        assert_eq!(
            pattern[8],
            LayerKind::Attention,
            "layer 8 must be attention"
        );
        assert_eq!(pattern[9], LayerKind::Ssm, "layer 9 must be SSM");

        assert_eq!(cfg.attn_layer_count(), 2, "2 attention layers out of 16");
        assert_eq!(cfg.ssm_layer_count(), 14, "14 SSM layers out of 16");
    }

    /// Period = 1: every layer is attention.
    #[test]
    fn all_attention_when_period_one() {
        let mut cfg = minimal_config();
        cfg.attn_layer_period = 1;
        for (i, kind) in cfg.layer_pattern().iter().enumerate() {
            assert_eq!(
                *kind,
                LayerKind::Attention,
                "layer {i} must be attention with period=1"
            );
        }
    }

    /// Non-zero offset shifts the pattern.
    #[test]
    fn offset_shifts_pattern() {
        let mut cfg = minimal_config();
        cfg.attn_layer_period = 4;
        cfg.attn_layer_offset = 2;
        // (0+2)%4=2 → SSM, (2+2)%4=0 → Attention
        assert_eq!(cfg.layer_kind(0), LayerKind::Ssm);
        assert_eq!(cfg.layer_kind(2), LayerKind::Attention);
    }

    /// `layer_kind()` and `layer_pattern()` are consistent.
    #[test]
    fn layer_kind_consistent_with_pattern() {
        let cfg = minimal_config();
        let pattern = cfg.layer_pattern();
        for (i, &kind) in pattern.iter().enumerate() {
            assert_eq!(
                cfg.layer_kind(i),
                kind,
                "layer_kind({i}) must match pattern[{i}]"
            );
        }
    }
}
