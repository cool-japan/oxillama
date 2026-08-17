//! DeepSeek Multi-head Latent Attention with a variant-aware Q path.
//!
//! [`crate::common::mla::MlaWeights`] hard-requires `w_q_a` + `q_a_norm` +
//! `w_q_b`, and `mla_forward` unconditionally runs all three.  That is the
//! **non-Lite** shape only.  `llm_build_deepseek2` branches on it explicitly:
//!
//! ```text
//! const bool is_lite = model.layers[il].wq;
//! if (!is_lite) {
//!     q = wq_a @ cur;  q = norm(q, attn_q_a_norm);  q = wq_b @ q;
//! } else {
//!     q = wq @ cur;
//! }
//! ```
//!
//! and the tensor table lists `LLM_TENSOR_ATTN_Q` alongside `ATTN_Q_A` /
//! `ATTN_Q_B`.  DeepSeek-V2-Lite and DeepSeek-Coder-V2-Lite ship
//! `q_lora_rank = null` and a single `blk.N.attn_q.weight`, so they could not be
//! loaded at all — the loader demanded `attn_q_a*`/`attn_q_b*` and errored out.
//!
//! Everything else (decoupled RoPE, the latent cache, the two-pass attention)
//! matches `crate::common::mla` and is reproduced here only because the Q branch
//! sits in the middle of the token loop.

use oxillama_quant::KernelDispatcher;

use crate::common::linear::QuantLinear;
use crate::common::mla::{MlaConfig, MlaLatentCache};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::error::{ArchError, ArchResult};

/// How the query is produced from the hidden state.
pub enum QProjection {
    /// Full DeepSeek-V2/V3: `w_q_b · rms_norm(w_q_a · x)`.
    ///
    /// Used whenever `deepseek2.attention.q_lora_rank` is present.
    LowRank {
        /// Q down-projection `[q_lora_rank, hidden]` (`blk.N.attn_q_a.weight`).
        w_q_a: QuantLinear,
        /// RMSNorm over the `q_lora_rank` latent (`blk.N.attn_q_a_norm.weight`).
        q_a_norm: RmsNorm,
        /// Q up-projection `[num_heads · qk_head_dim, q_lora_rank]`
        /// (`blk.N.attn_q_b.weight`).
        w_q_b: QuantLinear,
    },
    /// DeepSeek-V2-Lite / Coder-V2-Lite: one dense `blk.N.attn_q.weight`.
    Dense {
        /// `[num_heads · qk_head_dim, hidden]`.
        w_q: QuantLinear,
    },
}

impl QProjection {
    /// Whether this is the "lite" single-matrix form.
    pub fn is_lite(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }
}

/// Weights for one DeepSeek MLA layer.
pub struct DeepSeekMlaWeights {
    /// Query path (low-rank or dense).
    pub q: QProjection,
    /// KV combined down-projection `[kv_lora_rank + qk_rope, hidden]`
    /// (`blk.N.attn_kv_a_mqa.weight`).
    pub w_kv_a: QuantLinear,
    /// RMSNorm over the `kv_lora_rank` latent (`blk.N.attn_kv_a_norm.weight`).
    pub kv_a_norm: RmsNorm,
    /// KV up-projection `[num_heads · (qk_nope + v_head_dim), kv_lora_rank]`
    /// (`blk.N.attn_kv_b.weight`).
    pub w_kv_b: QuantLinear,
    /// Output projection `[hidden, num_heads · v_head_dim]`.
    pub w_o: QuantLinear,
    /// Precomputed RoPE table over the `qk_rope` slice.
    pub rope: RopeTable,
}

/// Run one MLA forward pass over `seq_len` tokens starting at `position`.
///
/// # Errors
///
/// Propagates shape mismatches, cache overflow and kernel failures.
pub fn ds_mla_forward(
    x: &[f32],
    weights: &DeepSeekMlaWeights,
    cfg: &MlaConfig,
    cache: &mut MlaLatentCache,
    position: usize,
) -> ArchResult<Vec<f32>> {
    let hidden_size = weights.w_o.out_features;
    if hidden_size == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "ds_mla_forward: hidden_size is 0".to_string(),
        });
    }
    let seq_len = x.len() / hidden_size;
    if x.len() != seq_len * hidden_size {
        return Err(ArchError::InvalidShape {
            name: "x".to_string(),
            expected: vec![seq_len, hidden_size],
            got: vec![x.len()],
        });
    }

    let dispatcher = KernelDispatcher::new();
    let kv_a_kernel = dispatcher.get_kernel(weights.w_kv_a.weight.tensor_type)?;
    let kv_b_kernel = dispatcher.get_kernel(weights.w_kv_b.weight.tensor_type)?;
    let w_o_kernel = dispatcher.get_kernel(weights.w_o.weight.tensor_type)?;
    let (q_first_kernel, q_second_kernel) = match &weights.q {
        QProjection::LowRank { w_q_a, w_q_b, .. } => (
            dispatcher.get_kernel(w_q_a.weight.tensor_type)?,
            Some(dispatcher.get_kernel(w_q_b.weight.tensor_type)?),
        ),
        QProjection::Dense { w_q } => (dispatcher.get_kernel(w_q.weight.tensor_type)?, None),
    };

    let mut q_rope_all = vec![0.0f32; seq_len * cfg.num_heads * cfg.qk_rope_head_dim];
    let mut q_nope_all = vec![0.0f32; seq_len * cfg.num_heads * cfg.qk_nope_head_dim];

    let mut q_latent = vec![0.0f32; cfg.q_lora_rank.max(1)];
    let mut q_full = vec![0.0f32; cfg.q_full_dim()];
    let mut kv_combined = vec![0.0f32; cfg.kv_combined_dim()];
    let mut kv_latent_normed = vec![0.0f32; cfg.kv_lora_rank];
    let mut k_rope_raw = vec![0.0f32; cfg.qk_rope_head_dim];
    let mut q_rope_head = vec![0.0f32; cfg.qk_rope_head_dim];

    for t in 0..seq_len {
        let x_t = &x[t * hidden_size..(t + 1) * hidden_size];

        match (&weights.q, q_second_kernel.as_ref()) {
            (
                QProjection::LowRank {
                    w_q_a,
                    q_a_norm,
                    w_q_b,
                },
                Some(second),
            ) => {
                w_q_a.forward(&*q_first_kernel, x_t, &mut q_latent[..cfg.q_lora_rank])?;
                q_a_norm.forward(&mut q_latent[..cfg.q_lora_rank]);
                w_q_b.forward(&**second, &q_latent[..cfg.q_lora_rank], &mut q_full)?;
            }
            (QProjection::Dense { w_q }, _) => {
                w_q.forward(&*q_first_kernel, x_t, &mut q_full)?;
            }
            (QProjection::LowRank { .. }, None) => {
                return Err(ArchError::InvalidConfig {
                    detail: "ds_mla_forward: low-rank Q path without a w_q_b kernel".to_string(),
                });
            }
        }

        let global_pos = position + t;
        for h in 0..cfg.num_heads {
            let head_off = h * cfg.qk_head_dim();
            let nope_slot = (t * cfg.num_heads + h) * cfg.qk_nope_head_dim;
            q_nope_all[nope_slot..nope_slot + cfg.qk_nope_head_dim]
                .copy_from_slice(&q_full[head_off..head_off + cfg.qk_nope_head_dim]);

            let rope_src = head_off + cfg.qk_nope_head_dim;
            q_rope_head.copy_from_slice(&q_full[rope_src..rope_src + cfg.qk_rope_head_dim]);
            weights.rope.apply(&mut q_rope_head, global_pos);
            let rope_slot = (t * cfg.num_heads + h) * cfg.qk_rope_head_dim;
            q_rope_all[rope_slot..rope_slot + cfg.qk_rope_head_dim].copy_from_slice(&q_rope_head);
        }

        weights
            .w_kv_a
            .forward(&*kv_a_kernel, x_t, &mut kv_combined)?;
        kv_latent_normed.copy_from_slice(&kv_combined[..cfg.kv_lora_rank]);
        k_rope_raw.copy_from_slice(&kv_combined[cfg.kv_lora_rank..]);
        weights.kv_a_norm.forward(&mut kv_latent_normed);
        weights.rope.apply(&mut k_rope_raw, global_pos);
        cache.append(&kv_latent_normed, &k_rope_raw)?;
    }

    let mut output = vec![0.0f32; seq_len * hidden_size];
    let mut attn_head_out = vec![0.0f32; cfg.attn_out_dim()];
    let mut kv_up = vec![0.0f32; cfg.kv_b_full_dim()];
    let mut scores = vec![0.0f32; cfg.num_heads * cache.seq_len.max(1)];

    let cache_start = cache.seq_len - seq_len;

    for t in 0..seq_len {
        let attend_len = cache_start + t + 1;
        attn_head_out.fill(0.0);

        for s in 0..attend_len {
            let lat_off = s * cfg.kv_lora_rank;
            let kv_lat_s = &cache.kv_latent[lat_off..lat_off + cfg.kv_lora_rank];
            weights
                .w_kv_b
                .forward(&*kv_b_kernel, kv_lat_s, &mut kv_up)?;

            let rope_off = s * cfg.qk_rope_head_dim;
            let k_rope_s = &cache.k_rope[rope_off..rope_off + cfg.qk_rope_head_dim];

            for h in 0..cfg.num_heads {
                let q_nope_h = &q_nope_all[(t * cfg.num_heads + h) * cfg.qk_nope_head_dim
                    ..(t * cfg.num_heads + h + 1) * cfg.qk_nope_head_dim];
                let q_rope_h = &q_rope_all[(t * cfg.num_heads + h) * cfg.qk_rope_head_dim
                    ..(t * cfg.num_heads + h + 1) * cfg.qk_rope_head_dim];
                let h_off = h * cfg.kv_b_per_head_dim();
                let k_nope_s = &kv_up[h_off..h_off + cfg.qk_nope_head_dim];

                let score: f32 = q_nope_h
                    .iter()
                    .zip(k_nope_s.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + q_rope_h
                        .iter()
                        .zip(k_rope_s.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                scores[h * attend_len + s] = score * cfg.softmax_scale;
            }
        }

        for h in 0..cfg.num_heads {
            softmax_inplace(&mut scores[h * attend_len..(h + 1) * attend_len]);
        }

        for s in 0..attend_len {
            let lat_off = s * cfg.kv_lora_rank;
            let kv_lat_s = &cache.kv_latent[lat_off..lat_off + cfg.kv_lora_rank];
            weights
                .w_kv_b
                .forward(&*kv_b_kernel, kv_lat_s, &mut kv_up)?;

            for h in 0..cfg.num_heads {
                let w = scores[h * attend_len + s];
                let h_off = h * cfg.kv_b_per_head_dim();
                let v_s = &kv_up
                    [h_off + cfg.qk_nope_head_dim..h_off + cfg.qk_nope_head_dim + cfg.v_head_dim];
                let v_head_out = &mut attn_head_out[h * cfg.v_head_dim..(h + 1) * cfg.v_head_dim];
                for (vo, &vs) in v_head_out.iter_mut().zip(v_s.iter()) {
                    *vo += w * vs;
                }
            }
        }

        let out_t = &mut output[t * hidden_size..(t + 1) * hidden_size];
        weights.w_o.forward(&*w_o_kernel, &attn_head_out, out_t)?;
    }

    Ok(output)
}

/// Numerically stable in-place softmax.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
