//! `ReferenceModel` — a [`ForwardPass`] implementation that operates in f32.
//!
//! This is a **stub** reference path for CI numeric diff testing.
//! It performs:
//! 1. Token embedding lookup from `token_embd.weight`.
//! 2. A single f32 matmul via `output.weight` to produce logits.
//!
//! It does **not** model transformer blocks — the purpose is to verify that
//! loading and dequantizing weights produces finite, reproducible numbers.
//! For full architecture correctness, use the quantized forward paths.

use super::loader::ReferenceWeights;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};

/// A `ForwardPass` implementation that operates entirely in f32.
///
/// Not for production use — for CI numeric diff vs reference implementations.
pub struct ReferenceModel {
    weights: ReferenceWeights,
    vocab_size: usize,
    max_context_length: usize,
    hidden_size: usize,
}

impl ReferenceModel {
    /// Construct a `ReferenceModel` from pre-dequantized weights.
    pub fn new(
        weights: ReferenceWeights,
        vocab_size: usize,
        max_context_length: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            weights,
            vocab_size,
            max_context_length,
            hidden_size,
        }
    }

    /// Reference f32 matrix-vector multiply: `y = A @ x`.
    ///
    /// `a` is a row-major matrix of shape `[out_dim × in_dim]`.
    /// `x` has length `in_dim`, result has length `out_dim`.
    fn reference_matmul_f32(a: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for (o, y_o) in y.iter_mut().enumerate() {
            let row_start = o * in_dim;
            *y_o = a[row_start..row_start + in_dim]
                .iter()
                .zip(x.iter())
                .map(|(a_val, x_val)| a_val * x_val)
                .sum();
        }
        y
    }
}

impl ForwardPass for ReferenceModel {
    /// Minimal reference forward: embed → lm_head projection → logits.
    fn forward(
        &mut self,
        tokens: &[u32],
        _kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.hidden_size;

        let embed_w = self
            .weights
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| ArchError::MissingTensor {
                name: "token_embd.weight".to_string(),
            })?;

        let token = tokens
            .last()
            .copied()
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: 0,
                message: "empty token sequence".to_string(),
            })? as usize;

        let embed_start = token * hidden;
        let embed_end = embed_start + hidden;
        if embed_end > embed_w.len() {
            return Err(ArchError::ForwardPassError {
                layer: 0,
                message: format!(
                    "token {token} embedding out of bounds (vocab_size={})",
                    embed_w.len() / hidden.max(1)
                ),
            });
        }
        let embed: Vec<f32> = embed_w[embed_start..embed_end].to_vec();

        let lm_head =
            self.weights
                .tensors
                .get("output.weight")
                .ok_or_else(|| ArchError::MissingTensor {
                    name: "output.weight".to_string(),
                })?;

        let logits = Self::reference_matmul_f32(lm_head, &embed, self.vocab_size, hidden);
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
