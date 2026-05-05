//! Attention span utilities shared across architectures.

use crate::config::ModelConfig;

/// Compute the effective KV attention span (in tokens) for a given layer.
///
/// - If `swa_window` is `None`, returns `u32::MAX` (global attention).
/// - If `swa_interleaved` is `true`, even-indexed layers are global
///   (Gemma-3 convention). Global layers return `u32::MAX`.
/// - Otherwise every layer uses `swa_window` (Mistral convention).
pub fn effective_attention_span(config: &ModelConfig, layer_idx: usize) -> u32 {
    match config.swa_window {
        None => u32::MAX,
        Some(w) => {
            if config.swa_interleaved && layer_idx.is_multiple_of(2) {
                u32::MAX // global layer in Gemma interleaved pattern
            } else {
                w
            }
        }
    }
}
