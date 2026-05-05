//! StableLM-specific configuration.
//!
//! StableLM uses `LayerNorm` (with bias) instead of `RMSNorm` and applies
//! RoPE only to a configurable leading fraction of each head dimension
//! (`partial_rotary_factor`, default 0.25).

/// Configuration specific to StableLM architectures.
///
/// These values are either read directly from GGUF metadata or set to their
/// canonical StableLM defaults.
#[derive(Debug, Clone)]
pub struct StablelmConfig {
    /// Fraction of head_dim to which RoPE is applied (e.g. 0.25 → first 25%).
    ///
    /// Only the first `round(partial_rotary_factor * head_dim)` dimensions of
    /// each Q/K head vector are rotated; the remaining dimensions are passed
    /// through unchanged.
    pub partial_rotary_factor: f32,
    /// Total number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (≤ num_heads; equals num_heads for MHA).
    pub num_kv_heads: usize,
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f32,
}

impl StablelmConfig {
    /// Return the number of rotated dimensions per head.
    ///
    /// Equal to `floor(partial_rotary_factor * head_dim)` clamped to
    /// `[0, head_dim]`. The returned value is guaranteed to be even (RoPE
    /// requires pairs).
    pub fn rotary_dims(&self, head_dim: usize) -> usize {
        let raw = (self.partial_rotary_factor * head_dim as f32) as usize;
        // Must be even for RoPE pair rotation.
        (raw & !1).min(head_dim)
    }
}

impl Default for StablelmConfig {
    fn default() -> Self {
        Self {
            partial_rotary_factor: 0.25,
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
}
