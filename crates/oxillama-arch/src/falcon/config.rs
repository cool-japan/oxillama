//! Falcon-specific configuration parsed from GGUF metadata.
//!
//! Covers the whole family that llama.cpp maps onto `LLM_ARCH_FALCON`:
//! Falcon-7B, Falcon-40B (which additionally ships a second per-layer
//! LayerNorm, `attn_norm_2`) and Falcon-2-11B (GQA).
//!
//! # Reference
//!
//! `~/work/refs/llama.cpp/src/models/falcon.cpp` (`llm_build_falcon`) is the
//! single graph every one of those checkpoints runs.  It contains **no**
//! hyper-parameter branch whatsoever:
//!
//! * `ggml_rope_ext` is applied to Q and K unconditionally ("using mode = 2
//!   for neox mode"), and there is no ALiBi code anywhere in the function.
//! * `GGML_ASSERT(n_embd_head == hparams.n_rot)` — full-head-dim rotary, never
//!   partial.
//! * The residual combination is unconditionally parallel:
//!   `cur = ffn_out; cur = add(cur, ffn_inp /* attention out */); cur =
//!   add(cur, inpL /* residual */)`, with the FFN reading the **first** norm
//!   (`build_ffn(attn_norm, ...)`, commented `// !! use the attn norm, not the
//!   result`).
//!
//! [`FalconConfig::from_model_config`] therefore derives
//! `rope = true`, `alibi = false`, `parallel_attn = true` for every checkpoint,
//! with no heuristic in between.

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
    /// Falcon-7B (MQA): `1`.
    /// Falcon-40B / Falcon-2 (GQA): a fraction of `n_heads`.
    pub n_kv_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden size / model dimension.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f32,
    /// Whether attention and FFN run in parallel.
    ///
    /// **Always `true`** when derived from GGUF metadata: `llm_build_falcon`
    /// sums the attention output, the FFN output and the residual stream
    /// unconditionally, and never re-normalises the post-attention hidden
    /// state.  The field (and the sequential branch it selects in
    /// [`FalconForward`](crate::falcon::FalconForward)) is retained only for
    /// hand-constructed configurations; no Falcon checkpoint can reach it.
    ///
    /// Before this was fixed the flag was derived as `config.activation ==
    /// "gelu"`, which is `false` for every real Falcon GGUF (the crate's
    /// default activation for `falcon` is `"silu"`), so every checkpoint ran
    /// the *sequential* graph — the wrong topology — and then silently reused
    /// `attn_norm` as its pre-FFN norm because Falcon ships no `ffn_norm`.
    pub parallel_attn: bool,
    /// Whether ALiBi positional bias is used instead of rotary embeddings.
    ///
    /// **Always `false`** when derived from GGUF metadata: `llm_build_falcon`
    /// has no ALiBi path.  Kept as an explicit opt-in for hand-constructed
    /// configurations (and to keep the flag honest — when set, the attention
    /// kernel really does apply the bias).
    pub alibi: bool,
    /// Whether RoPE (rotary positional embeddings) is used.
    ///
    /// **Always `true`** when derived from GGUF metadata.
    pub rope: bool,
    /// RoPE base frequency (default 10 000.0).
    pub rope_freq_base: f32,
    /// Dimension of each attention head.
    ///
    /// Taken from [`ModelConfig::head_dim`], which prefers the GGUF
    /// `{arch}.attention.key_length` key and only falls back to
    /// `hidden_size / n_heads`.
    pub head_dim: usize,
}

impl FalconConfig {
    /// Parse a [`FalconConfig`] from a generic [`ModelConfig`].
    ///
    /// Reads `falcon.*` GGUF keys from the metadata embedded in `config` and
    /// derives all Falcon-specific fields.  Falls back to sensible defaults so
    /// that a minimal metadata store (as used in unit tests) still produces a
    /// valid struct.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature stays fallible so that future
    /// metadata validation does not become a breaking change.
    pub fn from_model_config(config: &ModelConfig) -> ArchResult<Self> {
        let n_heads = config.num_attention_heads.max(1);
        let n_kv_heads = config.num_kv_heads.max(1);
        let n_layers = config.num_layers.max(1);
        let hidden_size = config.hidden_size.max(1);
        let vocab_size = config.vocab_size.max(1);
        let intermediate_size = config.intermediate_size.max(1);

        // `ModelConfig::head_dim` already prefers `{arch}.attention.key_length`
        // and falls back to `hidden_size / n_heads`; re-deriving the division
        // here would throw the GGUF-declared width away.
        let head_dim = if config.head_dim > 0 {
            config.head_dim
        } else {
            hidden_size.checked_div(n_heads).unwrap_or(64).max(1)
        };

        let rope_freq_base = if config.rope_freq_base > 0.0 {
            config.rope_freq_base
        } else {
            10_000.0
        };

        // Constants, not heuristics — see the module docs and
        // `src/models/falcon.cpp`.  `llm_build_falcon` ropes Q/K
        // unconditionally, has no ALiBi code path, and combines
        // `ffn_out + attn_out + residual` with no branch.
        let rope = true;
        let alibi = false;
        let parallel_attn = true;

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

    /// Compute the ALiBi slope for head `h` out of `n_heads` total.
    ///
    /// The standard ALiBi formula is `m_h = 2^(-8h/n_heads)` where `h` is
    /// 1-indexed.
    ///
    /// **Not reached by any GGUF-derived Falcon configuration** — llama.cpp's
    /// `llm_build_falcon` applies RoPE, never ALiBi.  Retained as the kernel
    /// behind the opt-in [`FalconConfig::alibi`] flag.
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

    /// Falcon-7B-ish metadata with the activation forced to `"gelu"`.
    ///
    /// `"gelu"` is the exact string the pre-fix formula keyed off
    /// (`let rope = config.activation != "gelu";`), so this fixture is the one
    /// that used to come out as ALiBi-without-RoPE.
    fn make_gelu_falcon_config() -> ModelConfig {
        let mut store = MetadataStore::new();
        store.insert(
            "general.architecture".to_string(),
            MetadataValue::String("falcon".to_string()),
        );
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
        let mut cfg = ModelConfig::from_metadata(&store).expect("falcon config");
        cfg.activation = "gelu".to_string();
        cfg
    }

    /// Falcon-2-11B-ish metadata (GQA), default activation.
    fn make_gqa_falcon_config() -> ModelConfig {
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
        ModelConfig::from_metadata(&store).expect("falcon config")
    }

    /// F1 regression — the `"gelu"` activation must not switch RoPE off.
    ///
    /// Pre-fix values for this exact config: `rope = false`, `alibi = true`,
    /// `parallel_attn = true` (from `rope = activation != "gelu"`,
    /// `alibi = !rope`, `parallel_attn = alibi`).  `llm_build_falcon` has no
    /// ALiBi path at all, so two of those three were wrong.
    #[test]
    fn test_gelu_activation_still_uses_rope_never_alibi() {
        let cfg = FalconConfig::from_model_config(&make_gelu_falcon_config()).expect("config");
        assert!(cfg.rope, "Falcon always ropes (ggml_rope_ext, neox mode)");
        assert!(!cfg.alibi, "llm_build_falcon has no ALiBi code path");
        assert!(cfg.parallel_attn, "Falcon is unconditionally parallel");
    }

    /// F1 regression — GQA Falcon is parallel too.
    ///
    /// Pre-fix values for this config (activation defaults to `"silu"` for
    /// `falcon`): `rope = true`, `alibi = false`, `parallel_attn = false`.
    /// The last one made every real Falcon GGUF run the sequential graph.
    #[test]
    fn test_gqa_falcon_is_rope_and_parallel() {
        let cfg = FalconConfig::from_model_config(&make_gqa_falcon_config()).expect("config");
        assert!(cfg.rope, "Falcon-2 should use RoPE");
        assert!(!cfg.alibi, "Falcon-2 should not use ALiBi");
        assert!(
            cfg.parallel_attn,
            "attention and FFN are unconditionally parallel in llm_build_falcon"
        );
        assert!(cfg.n_kv_heads < cfg.n_heads, "Falcon-2 should use GQA");
    }

    /// The three topology flags do not depend on the activation string.
    #[test]
    fn test_flags_are_activation_independent() {
        for activation in ["gelu", "silu", "gelu_pytorch_tanh", ""] {
            let mut model_config = make_gqa_falcon_config();
            model_config.activation = activation.to_string();
            let cfg = FalconConfig::from_model_config(&model_config).expect("config");
            assert!(cfg.rope, "activation {activation:?} must not disable RoPE");
            assert!(
                !cfg.alibi,
                "activation {activation:?} must not enable ALiBi"
            );
            assert!(
                cfg.parallel_attn,
                "activation {activation:?} must not disable parallel attention"
            );
        }
    }

    #[test]
    fn test_head_dim_comes_from_model_config() {
        let cfg = FalconConfig::from_model_config(&make_gqa_falcon_config()).expect("config");
        assert_eq!(cfg.head_dim, 2048 / 32);
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.hidden_size);
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
                "ALiBi slopes should be monotonically decreasing: {slopes:?}"
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
