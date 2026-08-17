//! Batched (multi-token) prefill for Qwen3.
//!
//! # Why
//!
//! Prompt processing used to replay the decode path once per prompt token, so
//! a 900-token prompt streamed the whole quantized weight set through the core
//! 900 times.  That is pure waste: every token of a prompt is already known, so
//! `M` of them can share one pass over the weights.
//!
//! This module keeps `M` tokens in flight through a layer at a time.  Each
//! projection becomes a matmul with `M` columns via
//! [`QuantLinear::forward_q8_fused_batch`], whose NEON kernels decode a weight
//! block once and dot it against all `M` activation vectors.  The weight
//! stream — the bottleneck — is divided by `M`; the SDOT work, which is the
//! irreducible arithmetic, is not.
//!
//! Batching the matmuls turned out to be only half the job, and on a long
//! prompt not even the visible half.  Attention costs `O(position)` per token,
//! and the per-token path evaluates it on **one** thread, so past a few hundred
//! tokens it outweighs the entire layer stack.  Batching the matmuls alone
//! moved a 921-token prompt from 91.5 s to 93.2 s wall — i.e. not at all,
//! because attention had become the bottleneck — even though total CPU time
//! fell from 375 s to 244 s.  The tile therefore *also* spreads its
//! `M x num_heads` independent attention computations across the shared kernel
//! pool (see the comment at that loop).
//!
//! Measured on Apple M3 (4P+4E), Qwen3-4B `Q4_K_M`, 921-token prompt,
//! `measure_prefill.sh` median of 3 on an otherwise idle machine:
//!
//! ```text
//! per-token prefill                          9.34 prompt tok/s
//! + batched matmuls (tile 16)                ~9.3         attention-bound, no gain
//! + threaded per-(token, head) attention     28.8  = 3.1x
//! ```
//!
//! A 185-token prompt, where attention is a much smaller share, isolates the
//! matmul half on its own: 13.42 s → 8.24 s wall for the same work.
//!
//! Decode is untouched: `forward` only routes here for multi-token calls.
//!
//! # Bit-identity with the per-token path
//!
//! Switching prefill to batches must not move a single logit, or temperature
//! sampling turns rounding noise into different text.  Three properties make
//! that hold, and each is pinned by a test:
//!
//! 1. **Activations.** Every token is quantized to Q8_0 on its own — no shared
//!    scale, no cross-token state — so the activation bytes a weight row sees
//!    are the bytes the per-token path would have handed it
//!    (`batched_activation_encoding_is_per_vector`).
//! 2. **Matmul.** The batched kernels hoist only the *weight decode* out of the
//!    token loop; the `K` accumulation order per `(row, token)` is untouched
//!    (`q4_k_batched_matmul_is_bit_identical_to_sequential` and its Q6_K twin).
//! 3. **Everything else** — RMSNorm, QK-norm, RoPE, attention, softmax,
//!    SwiGLU, residual adds — stays per token, in the same order, over the same
//!    values.  Attention masks causally by simply stopping at the token's own
//!    position, exactly as the per-token path does with `seq_len = position + 1`.
//!
//! # KV cache
//!
//! [`KvCacheAccess`] is a per-token interface: `store_kv` writes at the cache's
//! current position and `advance()` moves it once per token, *after* every
//! layer has written.  A batch therefore cannot commit layer 0's `M` keys
//! before layer 1 needs to write its own.
//!
//! Rather than widen the trait — which every implementation, including the
//! paged cache, would have to follow — a tile stages its `M` keys and values
//! per layer in [`BatchScratch`], serves intra-tile attention from that staging
//! area, and replays the ordinary `store_kv`/`advance` sequence once the tile
//! is complete.  The cache observes exactly the call sequence a sequential
//! prefill would have made.

use crate::common::linear::QuantLinear;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::qwen3::model::{softmax_inplace, Qwen3Model};
use crate::traits::KvCacheAccess;
use oxillama_quant::{quantize_activations_q8_0_batch_into, QuantKernel, Q8_0_ACT_BLOCK_BYTES};

/// Tokens processed per pass over the weights.
///
/// The weight stream shrinks as `1/TILE` while the live scratch (staged KV for
/// every layer plus the per-token hidden states) grows linearly, so the useful
/// range is bounded at both ends.  Measured on Apple M3 (4P+4E), Qwen3-4B
/// `Q4_K_M`, 921-token prompt, `measure_prefill.sh`:
///
/// ```text
/// TILE   prompt tok/s   (median of 3)
///    1        9.34      the per-token path this replaces
///    8       26.9
///   16       28.8
///   32       28.7
/// ```
///
/// 16 is the knee.  Past it the matmuls stop being weight-bandwidth-bound and
/// become SDOT-throughput-bound, so 32 measures no better while doubling the
/// scratch and the latency granularity — and it would sit exactly on the
/// kernels' `MAX_FUSED_BATCH` ceiling with no headroom.
pub(crate) const PREFILL_TILE: usize = 16;

/// Smallest prompt that takes the batched path.
///
/// Single-token `forward()` calls — the decode loop — must never reach here:
/// batching cannot help a batch of one, and the tile scratch would be pure
/// overhead.
pub(crate) const MIN_BATCH_TOKENS: usize = 2;

/// Per-tile scratch for [`Qwen3Model::forward_prefill_batched`].
///
/// Allocated on the first batched prefill and reused for the life of the
/// model.  Every buffer is token-major (`[tile][dim]`) except [`Self::proj`],
/// which holds a matmul result in the kernels' native feature-major
/// `[out_features][tile]` layout until it is transposed.
#[derive(Default)]
pub(crate) struct BatchScratch {
    /// Tile width the buffers below are sized for; `0` means unallocated.
    tile: usize,
    /// `[tile][hidden]` residual stream.
    hidden: Vec<f32>,
    /// `[tile][hidden]` post-RMSNorm activations.
    norm: Vec<f32>,
    /// `[tile][num_heads * head_dim]` queries.
    q: Vec<f32>,
    /// `[tile][kv_dim]` keys for this tile.
    k: Vec<f32>,
    /// `[tile][kv_dim]` values for this tile.
    v: Vec<f32>,
    /// `[tile][num_heads * head_dim]` concatenated attention output.
    attn_out: Vec<f32>,
    /// `[tile][intermediate]` SwiGLU gate branch.
    gate: Vec<f32>,
    /// `[tile][intermediate]` SwiGLU up branch.
    up: Vec<f32>,
    /// `[tile][hidden]` transposed projection result (attention output, FFN down).
    tmp: Vec<f32>,
    /// Q8_0 image of `tile` activation vectors, back to back.
    acts_q8: Vec<u8>,
    /// `[max_out_features][tile]` feature-major matmul accumulator.
    proj: Vec<f32>,
    /// `[layers][tile][kv_dim]` keys awaiting commit to the KV cache.
    staged_k: Vec<f32>,
    /// `[layers][tile][kv_dim]` values awaiting commit to the KV cache.
    staged_v: Vec<f32>,
}

impl BatchScratch {
    /// Size every buffer for `tile` tokens of `config`, if not already sized.
    fn ensure(&mut self, config: &ModelConfig, tile: usize) {
        if self.tile == tile {
            return;
        }
        let hidden = config.hidden_size;
        let attn_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;
        let max_out = attn_dim.max(kv_dim).max(hidden).max(inter);
        // Widest activation vector any projection consumes, in whole Q8_0
        // blocks: a K-quant pairs each 256-weight block with eight of them.
        let max_blocks = hidden.max(attn_dim).max(inter).div_ceil(256) * 8;

        self.tile = tile;
        self.hidden = vec![0.0; tile * hidden];
        self.norm = vec![0.0; tile * hidden];
        self.q = vec![0.0; tile * attn_dim];
        self.k = vec![0.0; tile * kv_dim];
        self.v = vec![0.0; tile * kv_dim];
        self.attn_out = vec![0.0; tile * attn_dim];
        self.gate = vec![0.0; tile * inter];
        self.up = vec![0.0; tile * inter];
        self.tmp = vec![0.0; tile * hidden];
        self.acts_q8 = Vec::with_capacity(tile * max_blocks * Q8_0_ACT_BLOCK_BYTES);
        self.proj = vec![0.0; tile * max_out];
        self.staged_k = vec![0.0; config.num_layers * tile * kv_dim];
        self.staged_v = vec![0.0; config.num_layers * tile * kv_dim];
    }
}

/// Rewrite a feature-major `[n][m]` matmul result as token-major `[m][n]`.
///
/// Pure data movement — no arithmetic, so it cannot perturb any value.
fn transpose_to_token_major(src: &[f32], dst: &mut [f32], n: usize, m: usize) {
    for (row, chunk) in src[..n * m].chunks_exact(m).enumerate() {
        for (t, &value) in chunk.iter().enumerate() {
            dst[t * n + row] = value;
        }
    }
}

/// Q8_0 activation-block count for a projection, or a `NotSupported` error.
///
/// Unreachable in practice: [`Qwen3Model::batched_prefill_supported`] has
/// already established that every projection opts into the fused path.  It
/// exists so this module contains no panicking path.
fn fused_blocks(linear: &QuantLinear, kernel: &dyn QuantKernel, what: &str) -> ArchResult<usize> {
    linear
        .q8_fused_blocks(kernel)
        .ok_or_else(|| ArchError::NotSupported {
            detail: format!("{what}: kernel has no fused Q8_0 activation path"),
        })
}

/// Run one projection over a whole tile: quantized matmul into the
/// feature-major `proj` accumulator, then transpose into token-major `dst`.
fn project_tile(
    linear: &QuantLinear,
    kernel: &dyn QuantKernel,
    acts_q8: &[u8],
    proj: &mut [f32],
    dst: &mut [f32],
    m: usize,
) -> ArchResult<()> {
    let n = linear.out_features;
    linear.forward_q8_fused_batch(kernel, acts_q8, &mut proj[..n * m], m)?;
    transpose_to_token_major(proj, dst, n, m);
    Ok(())
}

impl Qwen3Model {
    /// Whether every matmul in this model can take the batched prefill route.
    ///
    /// Requires that each projection's kernel opts into the fused Q8_0
    /// activation path (`q8_fused_acts_blocks`) and that no layer carries a
    /// LoRA adapter, whose per-token low-rank correction the batched matmul
    /// does not apply.  When either fails the caller falls back to the
    /// per-token loop, which is always correct.
    ///
    /// Re-evaluated on every prefill rather than cached: the walk is one
    /// kernel lookup per projection (~250 for a 4B model) against a prompt
    /// that costs whole seconds, and a cache would go stale the moment a
    /// caller attached a LoRA adapter through the public `layers` field.
    pub(crate) fn batched_prefill_supported(&self) -> bool {
        let fused =
            |linear: &QuantLinear| match self.dispatcher.get_kernel(linear.weight.tensor_type) {
                Ok(kernel) => linear.lora.is_none() && linear.q8_fused_blocks(&*kernel).is_some(),
                Err(_) => false,
            };
        self.layers.iter().all(|layer| {
            fused(&layer.attn_q)
                && fused(&layer.attn_k)
                && fused(&layer.attn_v)
                && fused(&layer.attn_output)
                && fused(&layer.ffn_gate)
                && fused(&layer.ffn_up)
                && fused(&layer.ffn_down)
        })
    }

    /// Run `tokens` through every layer in tiles of [`PREFILL_TILE`], leaving
    /// the **last** token's pre-output-norm hidden state in `self.buf_hidden`.
    ///
    /// The caller applies `output_norm` and (for `forward`) the LM head, both
    /// of which only ever look at the final token — exactly the state the
    /// per-token loop leaves behind.
    pub(crate) fn forward_prefill_batched(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let start_pos = kv_cache.seq_len();
        let tile = PREFILL_TILE.min(tokens.len());
        self.batch.ensure(&self.config, tile);

        let mut done = 0usize;
        while done < tokens.len() {
            let m = tile.min(tokens.len() - done);
            self.prefill_tile(&tokens[done..done + m], start_pos + done, kv_cache)?;
            done += m;
        }

        // The residual stream of the final token is what the LM head consumes.
        let hidden = self.config.hidden_size;
        let last = (tokens.len() - 1) % tile;
        self.buf_hidden
            .copy_from_slice(&self.batch.hidden[last * hidden..(last + 1) * hidden]);
        Ok(())
    }

    /// One tile: `m` tokens whose first sits at absolute position `p0`.
    fn prefill_tile(
        &mut self,
        tokens: &[u32],
        p0: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let m = tokens.len();
        let hidden = self.config.hidden_size;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let tile = self.batch.tile;

        for (t, &token) in tokens.iter().enumerate() {
            // One dequantized embedding row per token — see `TokenEmbedding`.
            // `token_embd`/`dispatcher` and `batch` are disjoint fields, so
            // the destructuring borrow keeps this inside one statement.
            let Self {
                token_embd,
                dispatcher,
                batch,
                ..
            } = self;
            token_embd.row_into(
                dispatcher,
                token,
                &mut batch.hidden[t * hidden..(t + 1) * hidden],
            )?;
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let b = &mut self.batch;
            for t in 0..m {
                layer.attn_norm.forward_to(
                    &b.hidden[t * hidden..(t + 1) * hidden],
                    &mut b.norm[t * hidden..(t + 1) * hidden],
                );
            }

            self.tile_attention(layer_idx, p0, m, kv_cache)?;

            let layer = &self.layers[layer_idx];
            let b = &mut self.batch;
            for t in 0..m {
                layer.ffn_norm.forward_to(
                    &b.hidden[t * hidden..(t + 1) * hidden],
                    &mut b.norm[t * hidden..(t + 1) * hidden],
                );
            }

            self.tile_feed_forward(layer_idx, m)?;
        }

        // Commit the tile's keys and values in the same order the per-token
        // path uses: every layer of one token, then advance, then the next.
        for t in 0..m {
            for layer_idx in 0..self.layers.len() {
                let base = (layer_idx * tile + t) * kv_dim;
                kv_cache.store_kv(
                    layer_idx,
                    &self.batch.staged_k[base..base + kv_dim],
                    &self.batch.staged_v[base..base + kv_dim],
                )?;
            }
            kv_cache.advance();
        }

        Ok(())
    }

    /// Attention for one tile of one layer.
    fn tile_attention(
        &mut self,
        layer_idx: usize,
        p0: usize,
        m: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden = self.config.hidden_size;
        let attn_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let tile = self.batch.tile;

        let q_kernel = self.kernel_for(&self.layers[layer_idx].attn_q)?;
        let k_kernel = self.kernel_for(&self.layers[layer_idx].attn_k)?;
        let v_kernel = self.kernel_for(&self.layers[layer_idx].attn_v)?;
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;

        let layer = &self.layers[layer_idx];
        let rope = &self.rope;
        let b = &mut self.batch;

        // --- Q / K / V ----------------------------------------------------
        // All three read the post-attn-norm activations, so the tile is
        // quantized once and shared, mirroring the per-token path's single
        // `quantize_activations_q8_0_into` for the same three GEMVs.
        let n_blocks = fused_blocks(&layer.attn_q, &*q_kernel, "attn_q")?
            .max(fused_blocks(&layer.attn_k, &*k_kernel, "attn_k")?)
            .max(fused_blocks(&layer.attn_v, &*v_kernel, "attn_v")?);
        quantize_activations_q8_0_batch_into(&b.norm, m, hidden, hidden, n_blocks, &mut b.acts_q8);

        project_tile(
            &layer.attn_q,
            &*q_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.q,
            m,
        )?;
        project_tile(
            &layer.attn_k,
            &*k_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.k,
            m,
        )?;
        project_tile(
            &layer.attn_v,
            &*v_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.v,
            m,
        )?;

        // --- QK-norm and RoPE, per token ----------------------------------
        for t in 0..m {
            let position = p0 + t;
            for h in 0..num_heads {
                let base = t * attn_dim + h * head_dim;
                let q_head = &mut b.q[base..base + head_dim];
                if let Some(ref q_norm) = layer.attn_q_norm {
                    q_norm.forward(q_head);
                }
                rope.apply(q_head, position);
            }
            for h in 0..num_kv_heads {
                let base = t * kv_dim + h * head_dim;
                let k_head = &mut b.k[base..base + head_dim];
                if let Some(ref k_norm) = layer.attn_k_norm {
                    k_norm.forward(k_head);
                }
                rope.apply(k_head, position);
            }
        }

        // Stage this layer's K/V; the tile commits them all once every layer
        // has run (see `prefill_tile`).
        let stage = layer_idx * tile * kv_dim;
        b.staged_k[stage..stage + m * kv_dim].copy_from_slice(&b.k[..m * kv_dim]);
        b.staged_v[stage..stage + m * kv_dim].copy_from_slice(&b.v[..m * kv_dim]);

        // --- Scaled dot-product attention, per token ----------------------
        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        if cached_keys.len() < p0 * kv_dim || cached_values.len() < p0 * kv_dim {
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!(
                    "kv cache holds {} key floats, batched prefill needs {} for positions < {p0}",
                    cached_keys.len(),
                    p0 * kv_dim
                ),
            });
        }

        // One task per (token, head): the attention output of a head is a
        // contiguous `head_dim` slice of `attn_out`, and no two heads read or
        // write the same slot, so `for_each_chunk_init` can hand each pair to a
        // worker without changing a single accumulation order.
        //
        // Threading this matters *more* than batching the matmuls once the
        // prompt is long: attention is O(position) per token, so at position
        // ~900 it outweighs the whole layer stack.  Differencing a 185-token
        // and a 921-token prompt on Apple M3 puts ~0.16 ms per cached position
        // per token here — about three quarters of a 921-token prefill once
        // the matmuls are batched, and the reason batching them alone measured
        // as no gain at all.
        {
            let BatchScratch {
                q,
                staged_k,
                staged_v,
                attn_out,
                ..
            } = &mut *b;
            let q: &[f32] = q;
            let staged_k: &[f32] = staged_k;
            let staged_v: &[f32] = staged_v;

            oxillama_quant::parallel::for_each_chunk_init(
                &mut attn_out[..m * attn_dim],
                head_dim,
                // Roughly the MACs one (token, head) performs: two passes
                // (scores, then the value-weighted sum) over `seq_len` keys.
                (p0 + m) * 2,
                Vec::<f32>::new,
                |scores, idx, out_head| {
                    let t = idx / num_heads;
                    let h = idx % num_heads;
                    let kv_head = h / heads_per_kv;
                    let seq_len = p0 + t + 1;
                    let q_base = t * attn_dim + h * head_dim;
                    let q_head = &q[q_base..q_base + head_dim];

                    scores.clear();
                    scores.resize(seq_len, 0.0);

                    // Positions committed before this tile live in the cache …
                    for (pos, score) in scores[..p0].iter_mut().enumerate() {
                        let k_off = pos * kv_dim + kv_head * head_dim;
                        let k_vec = &cached_keys[k_off..k_off + head_dim];
                        let mut acc = 0.0f32;
                        for d in 0..head_dim {
                            acc += q_head[d] * k_vec[d];
                        }
                        *score = acc * scale;
                    }
                    // … and positions inside this tile in the staging area.
                    // The loop stops at `t`, which *is* the causal mask.
                    for j in 0..=t {
                        let k_off = stage + j * kv_dim + kv_head * head_dim;
                        let k_vec = &staged_k[k_off..k_off + head_dim];
                        let mut acc = 0.0f32;
                        for d in 0..head_dim {
                            acc += q_head[d] * k_vec[d];
                        }
                        scores[p0 + j] = acc * scale;
                    }

                    softmax_inplace(scores);

                    out_head.fill(0.0);
                    for (pos, &w) in scores[..p0].iter().enumerate() {
                        let v_off = pos * kv_dim + kv_head * head_dim;
                        let v_vec = &cached_values[v_off..v_off + head_dim];
                        for d in 0..head_dim {
                            out_head[d] += w * v_vec[d];
                        }
                    }
                    for j in 0..=t {
                        let v_off = stage + j * kv_dim + kv_head * head_dim;
                        let v_vec = &staged_v[v_off..v_off + head_dim];
                        let w = scores[p0 + j];
                        for d in 0..head_dim {
                            out_head[d] += w * v_vec[d];
                        }
                    }
                },
            );
        }

        // --- Output projection and residual add ---------------------------
        let n_blocks_o = fused_blocks(&layer.attn_output, &*o_kernel, "attn_output")?;
        quantize_activations_q8_0_batch_into(
            &b.attn_out,
            m,
            attn_dim,
            attn_dim,
            n_blocks_o,
            &mut b.acts_q8,
        );
        project_tile(
            &layer.attn_output,
            &*o_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.tmp,
            m,
        )?;

        for (h_val, &p) in b.hidden[..m * hidden]
            .iter_mut()
            .zip(b.tmp[..m * hidden].iter())
        {
            *h_val += p;
        }

        Ok(())
    }

    /// SwiGLU feed-forward for one tile of one layer.
    fn tile_feed_forward(&mut self, layer_idx: usize, m: usize) -> ArchResult<()> {
        let hidden = self.config.hidden_size;
        let inter = self.config.intermediate_size;

        let gate_kernel = self.kernel_for(&self.layers[layer_idx].ffn_gate)?;
        let up_kernel = self.kernel_for(&self.layers[layer_idx].ffn_up)?;
        let down_kernel = self.kernel_for(&self.layers[layer_idx].ffn_down)?;

        let layer = &self.layers[layer_idx];
        let b = &mut self.batch;

        let n_blocks = fused_blocks(&layer.ffn_gate, &*gate_kernel, "ffn_gate")?.max(fused_blocks(
            &layer.ffn_up,
            &*up_kernel,
            "ffn_up",
        )?);
        quantize_activations_q8_0_batch_into(&b.norm, m, hidden, hidden, n_blocks, &mut b.acts_q8);

        project_tile(
            &layer.ffn_gate,
            &*gate_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.gate,
            m,
        )?;
        project_tile(
            &layer.ffn_up,
            &*up_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.up,
            m,
        )?;

        for t in 0..m {
            let (gate, up) = (&mut b.gate, &b.up);
            swiglu_inplace(
                &mut gate[t * inter..(t + 1) * inter],
                &up[t * inter..(t + 1) * inter],
            );
        }

        let n_blocks_down = fused_blocks(&layer.ffn_down, &*down_kernel, "ffn_down")?;
        quantize_activations_q8_0_batch_into(
            &b.gate,
            m,
            inter,
            inter,
            n_blocks_down,
            &mut b.acts_q8,
        );
        project_tile(
            &layer.ffn_down,
            &*down_kernel,
            &b.acts_q8,
            &mut b.proj,
            &mut b.tmp,
            m,
        )?;

        for (h_val, &f) in b.hidden[..m * hidden]
            .iter_mut()
            .zip(b.tmp[..m * hidden].iter())
        {
            *h_val += f;
        }

        Ok(())
    }
}
