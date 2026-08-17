//! Attention span / sliding-window utilities shared across architectures.
//!
//! This module is the **single source of truth** for which layers use sliding
//! window attention (SWA) and how far back each layer may attend.  Every
//! architecture that implements SWA must route both its attention mask and its
//! KV-cache trimming through [`effective_attention_span`] /
//! [`swa_attend_start`] rather than open-coding a layer-parity test.
//!
//! # Reference
//!
//! llama.cpp encodes the per-layer pattern in `llama_hparams::set_swa_pattern`:
//!
//! ```text
//! // dense_first == false (the common case)
//! swa_layers[il] = (n_pattern == 0) || (il % n_pattern < n_pattern - 1);
//! // dense_first == true
//! swa_layers[il] = (n_pattern == 0) || (il % n_pattern != 0);
//! ```
//!
//! With `n_pattern = 2` (Gemma-2) that makes **even** layers sliding and odd
//! layers global — the exact opposite of what this module used to return.  With
//! `n_pattern = 6` (Gemma-3) only `il % 6 == 5` is global.
//!
//! Masking follows `llama_hparams::is_masked_swa` for
//! `LLAMA_SWA_TYPE_STANDARD`: a key at `p0` is masked for a query at `p1`
//! when `p1 - p0 >= n_swa`.

use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};

/// Per-layer sliding-window pattern.
///
/// Mirrors the two forms of llama.cpp's `llama_hparams::set_swa_pattern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwaPattern {
    /// Every layer uses the sliding window (Mistral, Phi-3 small, …).
    ///
    /// Equivalent to llama.cpp's `set_swa_pattern(1)`.
    All,
    /// One global layer at the **end** of every `n`-layer group:
    /// layer `il` is sliding when `il % n < n - 1`.
    ///
    /// `n = 2` → Gemma-2 (`0` sliding, `1` global, …).
    /// `n = 6` → Gemma-3 (`0..=4` sliding, `5` global, …).
    EveryNth(u32),
    /// One global layer at the **start** of every `n`-layer group:
    /// layer `il` is sliding when `il % n != 0`.
    ///
    /// This is `set_swa_pattern(n, dense_first = true)` (Llama-4, Exaone-4).
    EveryNthDenseFirst(u32),
}

impl SwaPattern {
    /// Whether this pattern mixes sliding and global layers.
    ///
    /// `All` and `EveryNth(1)` (llama.cpp's "SWA disabled" spelling) are
    /// uniform; everything else alternates.
    pub fn is_interleaved(self) -> bool {
        !matches!(self, Self::All | Self::EveryNth(0) | Self::EveryNth(1))
    }

    /// Whether layer `layer_idx` uses the sliding window under this pattern.
    pub fn is_sliding(self, layer_idx: usize) -> bool {
        match self {
            Self::All => true,
            Self::EveryNth(n) => {
                if n == 0 {
                    true
                } else {
                    (layer_idx % n as usize) < (n as usize - 1)
                }
            }
            Self::EveryNthDenseFirst(n) => {
                if n == 0 {
                    true
                } else {
                    !(layer_idx.is_multiple_of(n as usize))
                }
            }
        }
    }
}

/// Compute the effective KV attention span (in tokens) for a given layer.
///
/// Returns `u32::MAX` for a globally-attending layer and the window size for a
/// sliding-window layer.
///
/// - `config.swa_window == None` → every layer is global.
/// - Otherwise the pattern comes from [`ModelConfig::swa_pattern`], which
///   reproduces llama.cpp's per-architecture `set_swa_pattern` call.
///
/// # Correctness note
///
/// The previous implementation returned `u32::MAX` for **even** layers under
/// `swa_interleaved`, while Gemma-2 makes even layers the *sliding* ones.  It
/// also had no non-test caller, so no KV cache was ever trimmed.
pub fn effective_attention_span(config: &ModelConfig, layer_idx: usize) -> u32 {
    match (config.swa_window, config.swa_pattern()) {
        (Some(window), Some(pattern)) if pattern.is_sliding(layer_idx) => window,
        _ => u32::MAX,
    }
}

/// Whether layer `layer_idx` attends through a sliding window.
///
/// Convenience wrapper over [`effective_attention_span`].
pub fn is_sliding_window_layer(config: &ModelConfig, layer_idx: usize) -> bool {
    effective_attention_span(config, layer_idx) != u32::MAX
}

/// First KV-cache index a query at `query_pos` may attend to on `layer_idx`.
///
/// This is the value architectures must use to trim their key/value scan and
/// their score buffer: iterate `swa_attend_start(..)..=query_pos` instead of
/// `0..=query_pos`.  That turns the per-layer cost from `O(n)` into `O(W)` and
/// is what makes Mistral's documented sliding-window claim real.
///
/// Follows `llama_hparams::is_masked_swa` for `LLAMA_SWA_TYPE_STANDARD`: a key
/// at `p0` is masked when `query_pos - p0 >= window`, so the first *visible*
/// key is `query_pos - window + 1`.
pub fn swa_attend_start(config: &ModelConfig, layer_idx: usize, query_pos: usize) -> usize {
    let span = effective_attention_span(config, layer_idx);
    if span == u32::MAX {
        return 0;
    }
    let window = span as usize;
    if window == 0 {
        return query_pos;
    }
    query_pos.saturating_sub(window - 1)
}

/// Validate that a forward pass stays inside the model's precomputed context.
///
/// Call this at the **top of every `forward()`**, before any RoPE table lookup
/// or `buf_attn_scores[pos]` write.  The runtime's decode loop is guarded but
/// prefill is not, and the prompt length is attacker-controlled through the
/// HTTP server.
///
/// # Arguments
/// * `config` – the model configuration (supplies `max_context_length`).
/// * `start_position` – the sequence position the first new token will occupy.
/// * `n_tokens` – how many tokens this call will process.
///
/// # Errors
///
/// [`ArchError::ConfigMismatch`] when `start_position + n_tokens` exceeds
/// `config.max_context_length`, including on arithmetic overflow.
pub fn validate_context_bounds(
    config: &ModelConfig,
    start_position: usize,
    n_tokens: usize,
) -> ArchResult<()> {
    let max_ctx = config.max_context_length;
    let end = start_position
        .checked_add(n_tokens)
        .ok_or_else(|| ArchError::ConfigMismatch {
            param: "context_length".to_string(),
            expected: format!("<= {max_ctx}"),
            got: format!("{start_position} + {n_tokens} (overflow)"),
        })?;

    if end > max_ctx {
        return Err(ArchError::ConfigMismatch {
            param: "context_length".to_string(),
            expected: format!("<= {max_ctx}"),
            got: end.to_string(),
        });
    }
    Ok(())
}

/// Validate that every token id in `tokens` is inside the vocabulary.
///
/// Companion guard to [`validate_context_bounds`]: an out-of-range token id
/// indexes past the end of the embedding matrix.
///
/// # Errors
///
/// [`ArchError::ConfigMismatch`] naming the first offending id.
pub fn validate_token_ids(config: &ModelConfig, tokens: &[u32]) -> ArchResult<()> {
    let vocab = config.vocab_size;
    for &t in tokens {
        if t as usize >= vocab {
            return Err(ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< vocab_size ({vocab})"),
                got: t.to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(arch: &str, window: Option<u32>, interleaved: bool) -> ModelConfig {
        ModelConfig {
            architecture: arch.to_string(),
            swa_window: window,
            swa_interleaved: interleaved,
            max_context_length: 128,
            vocab_size: 100,
            ..ModelConfig::default()
        }
    }

    #[test]
    fn gemma2_even_layers_are_sliding() {
        let c = cfg("gemma2", Some(4096), true);
        assert_eq!(effective_attention_span(&c, 0), 4096, "layer 0 is sliding");
        assert_eq!(effective_attention_span(&c, 1), u32::MAX, "layer 1 global");
        assert_eq!(effective_attention_span(&c, 2), 4096);
        assert_eq!(effective_attention_span(&c, 3), u32::MAX);
    }

    #[test]
    fn gemma3_is_global_every_sixth_layer() {
        let c = cfg("gemma3", Some(1024), true);
        for il in 0..12 {
            let expect_global = il % 6 == 5;
            let span = effective_attention_span(&c, il);
            assert_eq!(
                span == u32::MAX,
                expect_global,
                "layer {il}: got span {span}"
            );
        }
    }

    #[test]
    fn gemma1_has_no_sliding_window_pattern() {
        // Gemma v1 has no SWA at all; even with a window key it must not
        // inherit Gemma-2's interleaving.
        let c = cfg("gemma", Some(4096), false);
        for il in 0..4 {
            assert_eq!(effective_attention_span(&c, il), 4096);
        }
    }

    #[test]
    fn mistral_uses_window_on_every_layer() {
        let c = cfg("mistral", Some(4096), false);
        for il in 0..4 {
            assert_eq!(effective_attention_span(&c, il), 4096);
        }
    }

    #[test]
    fn no_window_is_global_everywhere() {
        let c = cfg("llama", None, false);
        for il in 0..4 {
            assert_eq!(effective_attention_span(&c, il), u32::MAX);
        }
    }

    #[test]
    fn attend_start_trims_to_window() {
        let c = cfg("mistral", Some(8), false);
        assert_eq!(swa_attend_start(&c, 0, 3), 0);
        assert_eq!(swa_attend_start(&c, 0, 20), 13);
        let g = cfg("llama", None, false);
        assert_eq!(swa_attend_start(&g, 0, 20), 0);
    }

    #[test]
    fn context_bounds_reject_overlong_prompt() {
        let c = cfg("llama", None, false);
        assert!(validate_context_bounds(&c, 0, 128).is_ok());
        assert!(validate_context_bounds(&c, 0, 129).is_err());
        assert!(validate_context_bounds(&c, 120, 9).is_err());
        assert!(validate_context_bounds(&c, usize::MAX, 1).is_err());
    }

    #[test]
    fn token_ids_are_range_checked() {
        let c = cfg("llama", None, false);
        assert!(validate_token_ids(&c, &[0, 42, 99]).is_ok());
        assert!(validate_token_ids(&c, &[100]).is_err());
    }

    #[test]
    fn swa_pattern_matches_llama_cpp_set_swa_pattern() {
        // set_swa_pattern(2): il % 2 < 1
        assert!(SwaPattern::EveryNth(2).is_sliding(0));
        assert!(!SwaPattern::EveryNth(2).is_sliding(1));
        // set_swa_pattern(4, dense_first = true): il % 4 != 0
        assert!(!SwaPattern::EveryNthDenseFirst(4).is_sliding(0));
        assert!(SwaPattern::EveryNthDenseFirst(4).is_sliding(1));
        assert!(SwaPattern::EveryNthDenseFirst(4).is_sliding(3));
        assert!(!SwaPattern::EveryNthDenseFirst(4).is_sliding(4));
    }
}
