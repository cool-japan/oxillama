//! Per-head LayerNorm for StableLM's optional QK normalization.
//!
//! # Why this is not [`LayerNorm`](crate::common::layer_norm::LayerNorm)
//!
//! StableLM 2 12B ships `blk.{i}.attn_q_norm.weight` and
//! `blk.{i}.attn_k_norm.weight`.  Their GGUF shape is **2-D**:
//!
//! ```text
//! attn_q_norm : {n_embd_head_k, n_head}     // llama-model.cpp, LLM_ARCH_STABLELM
//! attn_k_norm : {n_embd_head_k, n_head_kv}
//! ```
//!
//! and `~/work/refs/llama.cpp/src/models/stablelm.cpp` (`llm_build_stablelm`)
//! applies them *after* the head reshape:
//!
//! ```text
//! Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
//! ...
//! Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM, il);
//! ```
//!
//! `build_norm(.., LLM_NORM, ..)` is `ggml_norm` — a **mean-centered**
//! LayerNorm that normalizes along dim 0 only, i.e. independently per
//! `(head, token)` over a `head_dim`-wide window — followed by `ggml_mul` with
//! the weight tensor.  Because the weight is `[head_dim, n_head]` and the
//! activation is `[head_dim, n_head, n_tokens]`, the multiply broadcasts over
//! *tokens only*: head `h` is scaled by `weight[h * head_dim .. (h+1) * head_dim]`,
//! **a different slice for every head**.
//!
//! This is emphatically *not* Qwen3's QK-norm, whose weight is a single
//! `[head_dim]` vector shared by every head.  A loader that reuses head 0's
//! slice for heads `1..n` silently reads the wrong scales — the model still
//! runs, and still produces plausible-looking logits, which is what makes the
//! mistake expensive.
//!
//! Note also the `NULL` third argument: this norm has **no bias**, even though
//! `attn_norm` / `ffn_norm` / `output_norm` all do.

use crate::error::{ArchError, ArchResult};

/// A biasless, mean-centered LayerNorm with a distinct weight slice per head.
///
/// `forward` consumes a flat `[num_heads * head_dim]` activation buffer — the
/// layout the Q/K projections already write — and normalizes each
/// `head_dim`-wide window on its own statistics before scaling it by that
/// head's own slice of [`Self::weight`].
#[derive(Debug, Clone)]
pub struct PerHeadLayerNorm {
    /// Row-major `[num_heads][head_dim]` scales.
    pub weight: Vec<f32>,
    /// Width of one head (`n_embd_head_k`).
    pub head_dim: usize,
    /// Number of heads this norm covers (`n_head` for Q, `n_head_kv` for K).
    pub num_heads: usize,
    /// Variance epsilon (`hparams.f_norm_eps`).
    pub eps: f32,
}

impl PerHeadLayerNorm {
    /// Build a per-head norm, validating the weight length.
    ///
    /// A `head_dim`-length weight — one shared vector rather than one per head
    /// — is accepted and **replicated** across all heads.  No GGUF written by
    /// llama.cpp's converters has that shape (its `create_tensor` call would
    /// reject it), but replicating is the only reading that stays numerically
    /// faithful if one ever appears, and it keeps the per-head contract of this
    /// type intact for the forward pass.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `weight.len()` is neither
    /// `num_heads * head_dim` nor `head_dim`, or when either dimension is zero.
    pub fn new(
        name: &str,
        weight: Vec<f32>,
        head_dim: usize,
        num_heads: usize,
        eps: f32,
    ) -> ArchResult<Self> {
        if head_dim == 0 || num_heads == 0 {
            return Err(ArchError::InvalidShape {
                name: name.to_string(),
                expected: vec![num_heads, head_dim],
                got: vec![weight.len()],
            });
        }

        let expected = num_heads.saturating_mul(head_dim);
        let weight = if weight.len() == expected {
            weight
        } else if weight.len() == head_dim {
            // Shared-across-heads spelling: replicate so the hot path stays a
            // single indexed slice per head.
            weight.repeat(num_heads)
        } else {
            return Err(ArchError::InvalidShape {
                name: name.to_string(),
                expected: vec![num_heads, head_dim],
                got: vec![weight.len()],
            });
        };

        Ok(Self {
            weight,
            head_dim,
            num_heads,
            eps,
        })
    }

    /// Normalize every head of `x` in place.
    ///
    /// `x` is a flat `[num_heads * head_dim]` buffer.  Heads beyond
    /// `x.len() / head_dim` (a buffer shorter than declared) are skipped rather
    /// than panicking — this runs on every decoded token.
    pub fn forward(&self, x: &mut [f32]) {
        let head_dim = self.head_dim;
        if head_dim == 0 {
            return;
        }
        let inv_n = 1.0 / head_dim as f32;

        for h in 0..self.num_heads {
            let start = h * head_dim;
            let Some(head) = x.get_mut(start..start + head_dim) else {
                return;
            };
            let Some(scale) = self.weight.get(start..start + head_dim) else {
                return;
            };

            let mean: f32 = head.iter().sum::<f32>() * inv_n;
            let var: f32 = head
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                * inv_n;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for (v, &w) in head.iter_mut().zip(scale.iter()) {
                *v = (*v - mean) * inv_std * w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;
    const NUM_HEADS: usize = 3;

    /// **S2 regression.** Each head must be scaled by its *own* weight slice.
    ///
    /// Head 0's slice is all-zeros, head 1's all-ones, head 2's all-twos.  A
    /// loader/forward pair that reuses head 0's slice for every head (the Qwen3
    /// shared-vector assumption) would zero all three heads.
    #[test]
    fn each_head_uses_its_own_weight_slice() {
        let mut weight = Vec::with_capacity(NUM_HEADS * HEAD_DIM);
        for h in 0..NUM_HEADS {
            weight.extend_from_slice(&[h as f32; HEAD_DIM]);
        }
        let norm = PerHeadLayerNorm::new("attn_q_norm", weight, HEAD_DIM, NUM_HEADS, 1e-5)
            .expect("per-head weight must be accepted");

        // Identical activations in every head, so any difference in the output
        // can only come from the weights.
        let mut x = [1.0f32, 2.0, 3.0, 4.0].repeat(NUM_HEADS);
        norm.forward(&mut x);

        for (d, v) in x.iter().take(HEAD_DIM).enumerate() {
            assert!(
                v.abs() < 1e-6,
                "head 0 (weight 0) dim {d} must be zeroed, got {v}"
            );
        }
        // head 2's weight is twice head 1's, so its output must be too.
        for d in 0..HEAD_DIM {
            let h1 = x[HEAD_DIM + d];
            let h2 = x[2 * HEAD_DIM + d];
            assert!(h1.abs() > 1e-3, "head 1 must be non-trivial, got {h1}");
            assert!(
                (h2 - 2.0 * h1).abs() < 1e-5,
                "head 2 must be exactly 2x head 1: {h2} vs {h1}"
            );
        }
    }

    /// Normalization statistics are per head, not over the whole buffer.
    #[test]
    fn statistics_are_computed_per_head() {
        let norm = PerHeadLayerNorm::new(
            "attn_k_norm",
            vec![1.0f32; NUM_HEADS * HEAD_DIM],
            HEAD_DIM,
            NUM_HEADS,
            1e-5,
        )
        .expect("weight accepted");

        // Wildly different per-head magnitudes: a single global statistic would
        // leave head 0 near zero and blow head 2 up.
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0, // head 0
            101.0, 102.0, 103.0, 104.0, // head 1
            -50.0, -49.0, -48.0, -47.0, // head 2
        ];
        norm.forward(&mut x);

        for h in 0..NUM_HEADS {
            let head = &x[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let mean: f32 = head.iter().sum::<f32>() / HEAD_DIM as f32;
            assert!(
                mean.abs() < 1e-3,
                "head {h} must have ~zero mean after per-head LayerNorm, got {mean}"
            );
        }
        // All three heads share the same *shape*, so after per-head
        // normalization they must be numerically identical.
        for d in 0..HEAD_DIM {
            assert!(
                (x[d] - x[HEAD_DIM + d]).abs() < 1e-3,
                "identically-shaped heads must normalize identically"
            );
        }
    }

    #[test]
    fn wrong_weight_length_is_rejected() {
        let err = PerHeadLayerNorm::new("attn_q_norm", vec![1.0f32; 7], HEAD_DIM, NUM_HEADS, 1e-5);
        assert!(
            matches!(err, Err(ArchError::InvalidShape { .. })),
            "a mis-sized QK-norm weight must be an error, not a silent truncation"
        );
    }

    #[test]
    fn shared_weight_is_replicated_across_heads() {
        let norm = PerHeadLayerNorm::new(
            "attn_q_norm",
            vec![2.0f32; HEAD_DIM],
            HEAD_DIM,
            NUM_HEADS,
            1e-5,
        )
        .expect("shared weight accepted");
        assert_eq!(norm.weight.len(), NUM_HEADS * HEAD_DIM);
        assert!(norm.weight.iter().all(|&w| (w - 2.0).abs() < 1e-9));
    }
}
