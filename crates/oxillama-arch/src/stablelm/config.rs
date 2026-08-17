//! StableLM-specific configuration.
//!
//! StableLM uses `LayerNorm` (mean-centered, with bias) instead of `RMSNorm`
//! and applies RoPE only to a leading prefix of each head dimension.
//!
//! # Reference
//!
//! `~/work/refs/llama.cpp/src/llama-model.cpp`:
//!
//! * `case LLM_ARCH_STABLELM:` in `load_hparams()` (~line 1022) reads
//!   `LLM_KV_ATTENTION_LAYERNORM_EPS` — i.e. `{arch}.attention.layer_norm_epsilon`
//!   — into `hparams.f_norm_eps`.  The `_rms_` spelling is never consulted for
//!   this architecture.
//! * The shared hparams path (~line 618) sets `hparams.n_rot = n_embd_head_k`
//!   and then overrides it from `LLM_KV_ROPE_DIMENSION_COUNT`
//!   (`{arch}.rope.dimension_count`) when the key is present.  That key — not a
//!   hard-coded `0.25` factor — is the authoritative rotary width.

use oxillama_gguf::MetadataStore;

use crate::config::ModelConfig;

/// Canonical StableLM partial rotary factor.
///
/// This is only the **last-resort** fallback used when a checkpoint ships no
/// `{arch}.rope.dimension_count` key; every real GGUF written by
/// `convert_hf_to_gguf.py` carries that key, and it wins whenever present.
pub const DEFAULT_PARTIAL_ROTARY_FACTOR: f32 = 0.25;

/// Configuration specific to StableLM architectures.
///
/// These values are either read directly from GGUF metadata or set to their
/// canonical StableLM defaults.
#[derive(Debug, Clone)]
pub struct StablelmConfig {
    /// Fraction of head_dim to which RoPE is applied (e.g. 0.25 → first 25%).
    ///
    /// Only consulted when [`Self::rope_dimension_count`] is `None`: the GGUF
    /// key is authoritative because `convert_hf_to_gguf.py` already folded the
    /// HuggingFace `partial_rotary_factor` into it.
    pub partial_rotary_factor: f32,
    /// `{arch}.rope.dimension_count` (llama.cpp's `hparams.n_rot`).
    ///
    /// `None` when the checkpoint omits the key, in which case
    /// [`Self::partial_rotary_factor`] is used instead.
    pub rope_dimension_count: Option<usize>,
    /// Total number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (≤ num_heads; equals num_heads for MHA).
    pub num_kv_heads: usize,
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// LayerNorm epsilon (`hparams.f_norm_eps`).
    pub layer_norm_eps: f32,
}

impl StablelmConfig {
    /// Read the StableLM-specific hyperparameters out of GGUF metadata.
    ///
    /// Both keys are optional:
    ///
    /// | GGUF key | Field | Fallback |
    /// |----------|-------|----------|
    /// | `{arch}.rope.dimension_count` | [`Self::rope_dimension_count`] | `None` → [`DEFAULT_PARTIAL_ROTARY_FACTOR`] |
    /// | `{arch}.attention.layer_norm_epsilon` | [`Self::layer_norm_eps`] | [`ModelConfig::rms_norm_eps`] |
    ///
    /// The metadata store is read directly rather than through
    /// [`ExtraHparams`](crate::config::ExtraHparams) so this stays independent
    /// of which optional hyperparameters that side-car happens to carry.
    pub fn from_metadata(metadata: &MetadataStore, config: &ModelConfig) -> Self {
        let arch = &config.architecture;

        let rope_dimension_count = metadata
            .get_u32(&format!("{arch}.rope.dimension_count"))
            .ok()
            .map(|v| v as usize)
            .filter(|&v| v > 0);

        // `ModelConfig::rms_norm_eps` already falls back to the plain
        // LayerNorm key, but read it explicitly so the intent is local and a
        // future change to that fallback cannot silently retarget StableLM.
        let layer_norm_eps = metadata
            .get_f32(&format!("{arch}.attention.layer_norm_epsilon"))
            .unwrap_or(config.rms_norm_eps);

        Self {
            partial_rotary_factor: DEFAULT_PARTIAL_ROTARY_FACTOR,
            rope_dimension_count,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            layer_norm_eps,
        }
    }

    /// Return the number of rotated dimensions per head (llama.cpp's `n_rot`).
    ///
    /// `{arch}.rope.dimension_count` wins when present; otherwise the value is
    /// `floor(partial_rotary_factor * head_dim)`.  Either way the result is
    /// clamped to `head_dim` and rounded **down** to an even number, because
    /// RoPE rotates pairs.
    pub fn rotary_dims(&self, head_dim: usize) -> usize {
        let raw = match self.rope_dimension_count {
            Some(n) if n > 0 => n,
            _ => (self.partial_rotary_factor * head_dim as f32) as usize,
        };
        raw.min(head_dim) & !1
    }
}

impl Default for StablelmConfig {
    fn default() -> Self {
        Self {
            partial_rotary_factor: DEFAULT_PARTIAL_ROTARY_FACTOR,
            rope_dimension_count: None,
            num_heads: 32,
            num_kv_heads: 32,
            hidden_size: 2048,
            intermediate_size: 5504,
            layer_norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_dims_default_25_percent() {
        let cfg = StablelmConfig::default();
        // head_dim = hidden / num_heads = 2048 / 32 = 64
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rot = cfg.rotary_dims(head_dim);
        // 0.25 * 64 = 16, must be even
        assert_eq!(rot, 16, "25% of head_dim=64 should give 16 rotary dims");
    }

    #[test]
    fn rotary_dims_clamps_to_head_dim() {
        let cfg = StablelmConfig {
            partial_rotary_factor: 2.0, // > 1.0
            ..StablelmConfig::default()
        };
        let head_dim = 32;
        assert_eq!(
            cfg.rotary_dims(head_dim),
            head_dim,
            "factor > 1 must clamp to head_dim"
        );
    }

    #[test]
    fn rotary_dims_is_always_even() {
        let cfg = StablelmConfig {
            partial_rotary_factor: 0.3,
            ..StablelmConfig::default()
        };
        // 0.3 * 64 = 19.2 → floor = 19 → must round down to 18 (even)
        let head_dim = 64;
        let rot = cfg.rotary_dims(head_dim);
        assert_eq!(rot % 2, 0, "rotary_dims must always be even, got {rot}");
    }

    /// **S3 regression.** `{arch}.rope.dimension_count` is authoritative: it
    /// must beat the 0.25 factor, in both directions.
    #[test]
    fn rope_dimension_count_overrides_partial_rotary_factor() {
        let head_dim = 16;

        let from_key = StablelmConfig {
            rope_dimension_count: Some(4),
            partial_rotary_factor: 0.75, // would give 12
            ..StablelmConfig::default()
        };
        assert_eq!(
            from_key.rotary_dims(head_dim),
            4,
            "the GGUF key must win over partial_rotary_factor"
        );

        let no_key = StablelmConfig {
            rope_dimension_count: None,
            partial_rotary_factor: DEFAULT_PARTIAL_ROTARY_FACTOR,
            ..StablelmConfig::default()
        };
        assert_eq!(
            no_key.rotary_dims(head_dim),
            4,
            "without the key, 0.25 * 16 = 4 is the fallback"
        );
    }

    /// A `rope.dimension_count` wider than the head is clamped, never trusted
    /// blindly — an over-long rotation would index past the head vector.
    #[test]
    fn rope_dimension_count_is_clamped_to_head_dim() {
        let cfg = StablelmConfig {
            rope_dimension_count: Some(999),
            ..StablelmConfig::default()
        };
        assert_eq!(cfg.rotary_dims(16), 16);
    }
}
