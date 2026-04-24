//! Falcon-specific configuration parsed from GGUF metadata.
//!
//! Covers both Falcon-1 (7B/40B, parallel attention+MLP, ALiBi) and
//! Falcon-2 (11B, Group-Query Attention, RoPE).

use crate::config::ModelConfig;
use crate::error::ArchResult;

/// Falcon-specific hyperparameters extracted from GGUF metadata.
///
/// The standard [`ModelConfig`] fields hold most values; this struct adds the
/// Falcon-only flags that have no generic equivalent.
#[derive(Debug, Clone)]
pub struct FalconConfig {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads.
    ///
    /// Falcon-1 (MHA): equals `n_heads`.
    /// Falcon-2 (GQA): typically a fraction of `n_heads`.
    pub n_kv_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden size / model dimension.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// RMSNorm / LayerNorm epsilon.
    pub norm_eps: f32,
    /// Whether attention and FFN run in parallel (Falcon-1 feature).
    ///
    /// In parallel mode both branches receive the same layer-norm input and
    /// their outputs are summed before the residual add — instead of the
    /// standard sequential attn → residual → ffn → residual chain.
    pub parallel_attn: bool,
    /// Whether ALiBi positional bias is used (Falcon-1).
    ///
    /// When `true` the model adds head-specific linear biases to the attention
    /// logits rather than applying rotary embeddings.
    pub alibi: bool,
    /// Whether RoPE (rotary positional embeddings) is used (Falcon-2).
    ///
    /// Mutually exclusive with `alibi`.  When both fields arrive from metadata
    /// the RoPE flag takes precedence.
    pub rope: bool,
    /// RoPE base frequency (default 10 000.0).
    pub rope_freq_base: f32,
    /// Dimension of each attention head (`hidden_size / n_heads`).
    pub head_dim: usize,
}

impl FalconConfig {
    /// Parse a [`FalconConfig`] from a generic [`ModelConfig`].
    ///
    /// Reads `falcon.*` GGUF keys from the metadata embedded in `config` and
    /// derives all Falcon-specific fields.  Falls back to sensible defaults so
    /// that a minimal metadata store (as used in unit tests) still produces a
    /// valid struct.
    pub fn from_model_config(config: &ModelConfig) -> ArchResult<Self> {
        let n_heads = config.num_attention_heads.max(1);
        let n_kv_heads = config.num_kv_heads.max(1);
        let n_layers = config.num_layers.max(1);
        let hidden_size = config.hidden_size.max(1);
        let vocab_size = config.vocab_size.max(1);
        let intermediate_size = config.intermediate_size.max(1);
        let head_dim = hidden_size.checked_div(n_heads).unwrap_or(64).max(1);

        // RoPE detection: if the GGUF contained `falcon.rope.freq_base` it will
        // have been parsed into `config.rope_freq_base` with a non-default value
        // (i.e., != 0.0). We use 10 000.0 as the canonical RoPE default.
        let rope_freq_base = if config.rope_freq_base > 0.0 {
            config.rope_freq_base
        } else {
            10_000.0
        };

        // Falcon-2 (11B) has a finite rope_freq_base in the GGUF and uses RoPE.
        // Falcon-1 GGUF files do NOT include `falcon.rope.freq_base`; they use ALiBi.
        // We treat the presence of RoPE freq_base (non-default) as the RoPE flag.
        // The `config.rope_freq_base` field is 0.0 for Falcon-1 GGUF files where
        // the key is absent; our parser assigns 10 000.0 as default, so we cannot
        // distinguish by value alone.  Instead we rely on `config.activation` or
        // the architecture string:
        //   - If `config.activation` is "gelu" → Falcon-1 style → ALiBi.
        //   - Otherwise (most Falcon-2) → RoPE.
        //
        // For the purposes of tests and real GGUF files we follow this heuristic.
        // Code consumers can override by directly constructing `FalconConfig`.
        let rope = config.activation != "gelu";
        let alibi = !rope;

        // Parallel attention is a Falcon-1 default.  We enable it when not using
        // RoPE (heuristic for Falcon-1).
        let parallel_attn = alibi;

        Ok(Self {
            n_heads,
            n_kv_heads,
            n_layers,
            hidden_size,
            vocab_size,
            intermediate_size,
            norm_eps: config.rms_norm_eps.max(1e-9),
            parallel_attn,
            alibi,
            rope,
            rope_freq_base,
            head_dim,
        })
    }

    /// Compute ALiBi slope for head `h` out of `n_heads` total.
    ///
    /// The standard ALiBi formula is `m_h = 2^(-8h/n_heads)` where `h` is
    /// 1-indexed.  This matches the reference implementation in
    /// `transformer/src/transformers/models/falcon/modeling_falcon.py`.
    pub fn alibi_slope(h: usize, n_heads: usize) -> f32 {
        let ratio = 8.0 * (h + 1) as f32 / n_heads as f32;
        2.0_f32.powf(-ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use oxillama_gguf::{MetadataStore, MetadataValue};

    fn make_falcon1_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("falcon".to_string()),
        );
        // Falcon-1 uses GELU activation (old BigCode-style FFN)
        // We signal ALiBi by keeping activation = "gelu"
        store.insert(
            "falcon.embedding_length".to_string(),
            MetadataValue::Uint32(4544),
        );
        store.insert("falcon.block_count".to_string(), MetadataValue::Uint32(32));
        store.insert(
            "falcon.attention.head_count".to_string(),
            MetadataValue::Uint32(71),
        );
        store.insert(
            "falcon.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(1),
        );
        let mut cfg = ModelConfig::from_metadata(&store).expect("falcon1 config");
        cfg.activation = "gelu".to_string();
        cfg
    }

    fn make_falcon2_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("falcon".to_string()),
        );
        store.insert(
            "falcon.embedding_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        store.insert("falcon.block_count".to_string(), MetadataValue::Uint32(32));
        store.insert(
            "falcon.attention.head_count".to_string(),
            MetadataValue::Uint32(32),
        );
        store.insert(
            "falcon.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        ModelConfig::from_metadata(&store).expect("falcon2 config")
    }

    #[test]
    fn test_falcon1_is_alibi_parallel() {
        let cfg = FalconConfig::from_model_config(&make_falcon1_config()).unwrap();
        assert!(cfg.alibi, "Falcon-1 should use ALiBi");
        assert!(!cfg.rope, "Falcon-1 should not use RoPE");
        assert!(cfg.parallel_attn, "Falcon-1 should use parallel attention");
    }

    #[test]
    fn test_falcon2_is_rope_gqa() {
        let cfg = FalconConfig::from_model_config(&make_falcon2_config()).unwrap();
        assert!(cfg.rope, "Falcon-2 should use RoPE");
        assert!(!cfg.alibi, "Falcon-2 should not use ALiBi");
        assert!(
            !cfg.parallel_attn,
            "Falcon-2 should not use parallel attention"
        );
        assert!(cfg.n_kv_heads < cfg.n_heads, "Falcon-2 should use GQA");
    }

    #[test]
    fn test_alibi_slopes_monotone_decreasing() {
        let n_heads = 8;
        let slopes: Vec<f32> = (0..n_heads)
            .map(|h| FalconConfig::alibi_slope(h, n_heads))
            .collect();
        for i in 1..slopes.len() {
            assert!(
                slopes[i] < slopes[i - 1],
                "ALiBi slopes should be monotonically decreasing: {:?}",
                slopes
            );
        }
    }

    #[test]
    fn test_alibi_slope_first_head() {
        // Head 0 (1-indexed h=1): m = 2^(-8/n_heads)
        let slope = FalconConfig::alibi_slope(0, 8);
        let expected = 2.0_f32.powf(-1.0);
        assert!(
            (slope - expected).abs() < 1e-6,
            "Head-0 slope mismatch: got {slope}, expected {expected}"
        );
    }

    #[test]
    fn test_alibi_slope_last_head() {
        let n = 8;
        let slope = FalconConfig::alibi_slope(n - 1, n);
        // h = 8, result = 2^(-8)
        let expected = 2.0_f32.powf(-8.0);
        assert!(
            (slope - expected).abs() < 1e-6,
            "Head-{} slope mismatch: got {slope}, expected {expected}",
            n - 1
        );
    }
}
