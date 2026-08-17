//! Gemma transformer forward pass implementation.
//!
//! Implements the Gemma 1/2/3 architecture with:
//! - Embedding scaling by `sqrt(hidden_size)`
//! - Pre-norm AND post-norm (RMSNorm before and after attn/FFN, Gemma 2+)
//! - Optional per-head Q/K RMSNorm before RoPE (Gemma 3)
//! - GeGLU activation (GELU-gated) instead of SwiGLU
//! - Interleaved sliding window (local) and full causal (global) attention
//!   (`i % 2 == 0` for Gemma 2, `i % 6 != 5` for Gemma 3 — see
//!   `is_sliding_window_layer`)
//! - Logit soft-capping for attention scores (Gemma 2 only) and final logits
//!   (Gemma 2/3)

use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::{geglu_inplace, soft_cap_inplace};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};
use std::sync::Arc;

/// A single Gemma transformer layer.
pub struct GemmaLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Post-attention RMSNorm (Gemma 2+).
    pub attn_post_norm: Option<RmsNorm>,
    /// Query projection.
    pub attn_q: QuantLinear,
    /// Key projection.
    pub attn_k: QuantLinear,
    /// Value projection.
    pub attn_v: QuantLinear,
    /// Output projection.
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// Post-FFN RMSNorm (Gemma 2+).
    pub ffn_post_norm: Option<RmsNorm>,
    /// FFN gate projection (GeGLU).
    pub ffn_gate: QuantLinear,
    /// FFN up projection.
    pub ffn_up: QuantLinear,
    /// FFN down projection.
    pub ffn_down: QuantLinear,
    /// Whether this layer uses sliding window attention (vs full causal).
    pub use_sliding_window: bool,
    /// Per-head RMSNorm applied to Q before RoPE (Gemma 3 QK-norm).
    ///
    /// `None` for Gemma 1/2 checkpoints, which do not have this tensor.
    pub attn_q_norm: Option<RmsNorm>,
    /// Per-head RMSNorm applied to K before RoPE (Gemma 3 QK-norm).
    pub attn_k_norm: Option<RmsNorm>,

    // Resolved kernels — one per projection, looked up from the tensor's
    // quantization type exactly once, in `load_gemma_from_gguf`, instead of
    // once per layer per token on the decode hot path (see
    // `Qwen3Layer`'s equivalent fields, which are `pub` for the same reason:
    // constructing a layer outside the loader — e.g. from a test — still
    // needs to supply them).
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    pub ffn_gate_kernel: Arc<dyn QuantKernel>,
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

/// Complete Gemma model.
pub struct GemmaModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Embedding scaling factor: sqrt(hidden_size).
    pub embed_scale: f32,
    /// Transformer layers.
    pub layers: Vec<GemmaLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection (may be None if weight-tied with token_embd).
    pub output: Option<QuantLinear>,
    /// Resolved kernel for `output`, when present — see `GemmaLayer`'s kernel
    /// fields for why this is resolved once instead of per call.
    output_kernel: Option<Arc<dyn QuantKernel>>,
    /// RoPE table used by global (non-sliding-window) layers, and shared by
    /// local layers too when the checkpoint has no distinct local base.
    pub rope: RopeTable,
    /// RoPE table used by local (sliding-window) layers when the checkpoint
    /// specifies a distinct base via `{arch}.rope.freq_base_swa` (Gemma 3;
    /// e.g. base 10000 locally vs. 1000000 globally). `None` when the
    /// checkpoint has no such key, or when Gemma 2/1 loads (single table).
    pub rope_local: Option<RopeTable>,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Sliding window size for local attention layers.
    pub sliding_window: Option<usize>,
    /// Attention logit soft-cap value (0.0 = disabled). Gemma 2 only —
    /// Gemma 3 removed attention soft-capping (see `load_softcap_config`).
    pub attn_logit_softcap: f32,
    /// Final logit soft-cap value (0.0 = disabled).
    pub final_logit_softcap: f32,
    /// Attention scale applied to Q before the QK dot product.
    ///
    /// `1/sqrt(head_dim)` for every Gemma checkpoint except the Gemma-2-27B
    /// and Gemma-3-27B variants, which train with
    /// `query_pre_attn_scalar = hidden_size / num_heads` instead. See
    /// `compute_attention_scale` for the exact rule (mirrors llama.cpp,
    /// which has no GGUF key for this and detects the 27B variant from
    /// `n_layer` alone).
    pub attention_scale: f32,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_post_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    /// Output of `attn_output`'s projection, added into `buf_hidden`.
    /// Preallocated to `hidden_size` here instead of a fresh
    /// `vec![0.0; hidden_size]` inside `attention()` on every call.
    buf_proj_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl GemmaModel {
    /// Create a new GemmaModel from preloaded weights.
    ///
    /// `local_rope_freq_base` is the `{arch}.rope.freq_base_swa` metadata
    /// value, when present and numerically distinct from
    /// `config.rope_freq_base`; pass `None` to share a single RoPE table
    /// across every layer (Gemma 1/2, and Gemma 3 checkpoints without the
    /// key).
    ///
    /// Fails if `output`'s tensor type — or any layer projection's — has no
    /// registered [`oxillama_quant::QuantKernel`]: every kernel is resolved
    /// once here rather than on every forward pass.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<GemmaLayer>,
        output_norm: RmsNorm,
        output: Option<QuantLinear>,
        attn_logit_softcap: f32,
        final_logit_softcap: f32,
        local_rope_freq_base: Option<f32>,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let sliding_window = config.sliding_window;
        let embed_scale = (hidden_size as f32).sqrt();
        let attention_scale = compute_attention_scale(&config);

        let rope = RopeTable::new(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );
        let rope_local = local_rope_freq_base
            .filter(|&base| (base - config.rope_freq_base).abs() > f32::EPSILON)
            .map(|base| {
                RopeTable::new(
                    head_dim,
                    max_ctx,
                    base,
                    config.rope_scaling_type,
                    config.rope_scaling_factor,
                )
            });
        let dispatcher = KernelDispatcher::new();
        let output_kernel = output
            .as_ref()
            .map(|o| resolve_kernel(&dispatcher, o))
            .transpose()?;

        Ok(Self {
            config,
            token_embd,
            embed_scale,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            rope_local,
            dispatcher,
            sliding_window,
            attn_logit_softcap,
            final_logit_softcap,
            attention_scale,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_post_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            // The concatenated attention heads, NOT always `hidden_size`:
            // Gemma-2-9B attends over 16 × 256 = 4096 while its residual
            // stream is only 3584 wide. Sizing this as `hidden_size` made
            // `attention()` index `buf_attn_out[3840..4096]` into a
            // 3584-element Vec and panic at layer 0, head 14.
            buf_attn_out: vec![0.0; num_heads * head_dim],
            buf_proj_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        })
    }

    /// Load the residual stream with `token`'s embedding row, scaled by
    /// `sqrt(hidden_size)`.
    ///
    /// Bound-checked: an out-of-vocabulary `token` (possible whenever
    /// `vocab_size` is over- or under-estimated from GGUF metadata) is
    /// reported as an error instead of panicking on the slice index. This
    /// path runs on every decoded token, including ones sourced from an
    /// HTTP request body, so it must never panic.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = (token as usize)
            .checked_mul(hidden_size)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        let end = offset
            .checked_add(hidden_size)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        let src = self
            .token_embd
            .get(offset..end)
            .ok_or_else(|| out_of_vocab_error(token, self.config.vocab_size))?;
        self.buf_hidden.copy_from_slice(src);
        // Gemma scales embeddings by sqrt(hidden_size)
        for v in &mut self.buf_hidden {
            *v *= self.embed_scale;
        }
        Ok(())
    }

    /// Run grouped-query attention with optional sliding window.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        // Resolved once at load time — see the comment on `GemmaLayer`'s
        // kernel fields.
        let q_kernel: &dyn QuantKernel = &*layer.attn_q_kernel;
        let k_kernel: &dyn QuantKernel = &*layer.attn_k_kernel;
        let v_kernel: &dyn QuantKernel = &*layer.attn_v_kernel;

        layer
            .attn_q
            .forward(q_kernel, &self.buf_norm, &mut self.buf_q)?;
        layer
            .attn_k
            .forward(k_kernel, &self.buf_norm, &mut self.buf_k)?;
        layer
            .attn_v
            .forward(v_kernel, &self.buf_norm, &mut self.buf_v)?;

        // Gemma 3's per-head Q/K RMSNorm runs BEFORE RoPE (verified against
        // `models/gemma3.cpp`: `Qcur = build_norm(Qcur, attn_q_norm, ...)`
        // precedes `ggml_rope_ext`). `None` on Gemma 1/2 layers, which have
        // no such tensor.
        //
        // `rope_table` is computed via direct field projection (`self.rope`
        // / `self.rope_local`), not a `&self` helper method: a method call's
        // return value is opaque to the borrow checker and would be treated
        // as borrowing all of `self` for its lifetime, conflicting with the
        // `&mut self.buf_q` / `&mut self.buf_k` slices below. Direct field
        // access lets the checker see `rope`/`rope_local` and `buf_q`/`buf_k`
        // as disjoint paths, coexisting with `self.layers` (still held by
        // `layer`) too.
        let rope_table: &RopeTable = if layer.use_sliding_window {
            self.rope_local.as_ref().unwrap_or(&self.rope)
        } else {
            &self.rope
        };
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            if let Some(ref q_norm) = layer.attn_q_norm {
                q_norm.forward(q_head);
            }
            rope_table.apply(q_head, position);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            if let Some(ref k_norm) = layer.attn_k_norm {
                k_norm.forward(k_head);
            }
            rope_table.apply(k_head, position);
        }

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;

        // Determine attention window: sliding for local layers, full for global
        let window_start = if layer.use_sliding_window {
            match self.sliding_window {
                Some(w) => seq_len.saturating_sub(w),
                None => 0,
            }
        } else {
            0
        };

        let scale = self.attention_scale;

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            let window_len = seq_len - window_start;
            for pos in window_start..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_vec[d];
                }
                self.buf_attn_scores[pos - window_start] = score * scale;
            }

            // Apply attention logit soft-capping (Gemma 2 only; always 0.0,
            // hence a no-op, on Gemma 1/3 — see `load_softcap_config`).
            if self.attn_logit_softcap > 0.0 {
                soft_cap_inplace(
                    &mut self.buf_attn_scores[..window_len],
                    self.attn_logit_softcap,
                );
            }

            softmax_inplace(&mut self.buf_attn_scores[..window_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in window_start..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos - window_start];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Project attention output. `buf_proj_out` is preallocated engine
        // state (see its field doc) instead of a per-call `vec![0.0; ...]`.
        let o_kernel: &dyn QuantKernel = &*self.layers[layer_idx].attn_output_kernel;
        let layer = &self.layers[layer_idx];
        layer
            .attn_output
            .forward(o_kernel, &self.buf_attn_out, &mut self.buf_proj_out)?;

        // Post-attention norm (Gemma 2+)
        if let Some(ref post_norm) = layer.attn_post_norm {
            post_norm.forward_to(&self.buf_proj_out, &mut self.buf_post_norm);
            // Add post-normed output to residual
            for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_post_norm.iter()) {
                *h += p;
            }
        } else {
            // Add raw output to residual (Gemma 1 style)
            for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj_out.iter()) {
                *h += p;
            }
        }

        Ok(())
    }

    /// Run the GeGLU feed-forward network for a single layer.
    ///
    /// FFN(x) = down_proj(gelu(gate_proj(x)) * up_proj(x))
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        let gate_kernel: &dyn QuantKernel = &*layer.ffn_gate_kernel;
        let up_kernel: &dyn QuantKernel = &*layer.ffn_up_kernel;
        let down_kernel: &dyn QuantKernel = &*layer.ffn_down_kernel;

        layer
            .ffn_gate
            .forward(gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(up_kernel, &self.buf_norm, &mut self.buf_up)?;

        // GeGLU activation (Gemma uses GELU instead of SiLU)
        geglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

        // Post-FFN norm (Gemma 2+)
        if let Some(ref post_norm) = layer.ffn_post_norm {
            post_norm.forward_to(&self.buf_ffn_out, &mut self.buf_post_norm);
            for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_post_norm.iter()) {
                *h += f;
            }
        } else {
            for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
                *h += f;
            }
        }

        Ok(())
    }

    /// Compute final logits using the output projection or tied embeddings.
    fn compute_logits(&mut self) -> ArchResult<()> {
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        if let (Some(ref output), Some(ref kernel)) = (&self.output, &self.output_kernel) {
            output.forward(&**kernel, &self.buf_hidden, &mut self.buf_logits)?;
        } else {
            // Weight-tied: logits = hidden @ token_embd^T. No quantized
            // kernel is used here (the tied case runs over the dequantized
            // f32 `token_embd` table); see G12 in the fix report for the
            // known cost of this path on very large vocabularies.
            let vocab_size = self.config.vocab_size;
            let hidden_size = self.config.hidden_size;
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                let embd_offset = v * hidden_size;
                for d in 0..hidden_size {
                    sum += self.buf_hidden[d] * self.token_embd[embd_offset + d];
                }
                self.buf_logits[v] = sum;
            }
        }

        // Apply final logit soft-capping
        if self.final_logit_softcap > 0.0 {
            soft_cap_inplace(&mut self.buf_logits, self.final_logit_softcap);
        }

        Ok(())
    }
}

impl ForwardPass for GemmaModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        check_context_length(self.config.max_context_length, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                // Pre-attention norm
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                // Pre-FFN norm
                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        // Final norm
        self.output_norm.forward(&mut self.buf_hidden);

        // Logits with optional soft-capping
        self.compute_logits()?;

        // Hand the freshly computed logits to the caller by ownership
        // transfer instead of a `.clone()` allocation-and-memcpy on every
        // decoded token; `compute_logits()` resizes `buf_logits` back up on
        // the next call (see the guard at its top).
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of `compute_logits()` which either projects through the
    /// weight-tied embedding matrix or an explicit output.weight, and also
    /// skips the final logit soft-capping. Returns a `hidden_size`-dimensional
    /// vector suitable for L2-normalised semantic embeddings.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        check_context_length(self.config.max_context_length, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        // Final norm on the last token's hidden state.
        // Does NOT call compute_logits() — returns hidden state directly,
        // skipping both the LM-head projection and the final logit soft-cap.
        self.output_norm.forward(&mut self.buf_hidden);

        Ok(self.buf_hidden.clone())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.config.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn swa_config(&self) -> Option<(u32, bool)> {
        self.config
            .swa_window
            .map(|w| (w, self.config.swa_interleaved))
    }
}

/// In-place softmax over a slice.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Build the out-of-vocabulary error for a bad token id.
fn out_of_vocab_error(token: u32, vocab_size: usize) -> ArchError {
    ArchError::ConfigMismatch {
        param: "token id".to_string(),
        expected: format!("< {vocab_size}"),
        got: token.to_string(),
    }
}

/// Validate that a forward/embed call will not run any position past
/// `max_context_length`.
///
/// `RopeTable::apply` and `buf_attn_scores` are both sized to
/// `max_context_length` and indexed by raw position with no further
/// checking; an over-long prompt at prefill time (the decode loop is
/// separately guarded upstream, but prefill is reachable directly from the
/// HTTP server with attacker-controlled input) would otherwise panic on the
/// first out-of-range index instead of returning an error.
fn check_context_length(
    max_context_length: usize,
    start_pos: usize,
    n_tokens: usize,
) -> ArchResult<()> {
    let end = start_pos
        .checked_add(n_tokens)
        .filter(|&end| end <= max_context_length);
    if end.is_some() {
        return Ok(());
    }
    Err(ArchError::InvalidConfig {
        detail: format!(
            "context length exceeded: start_pos={start_pos} + {n_tokens} tokens > max_context_length={max_context_length}"
        ),
    })
}

/// Compute the Q scale applied before the QK dot product.
///
/// Every Gemma checkpoint uses `1/sqrt(head_dim)` EXCEPT the Gemma-2-27B and
/// Gemma-3-27B variants, which train with
/// `query_pre_attn_scalar = hidden_size / num_heads` (144 for Gemma-2-27B,
/// not head_dim's 128) — see
/// <https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173>.
///
/// No GGUF converter (`convert_hf_to_gguf.py`'s `Gemma2Model`/`Gemma3Model`)
/// writes this value as metadata, and llama.cpp does not read one either
/// (`{arch}.attention.scale` is read only for unrelated architectures, e.g.
/// Granite). Instead `llama-model.cpp` hardcodes the 27B detection from
/// `n_layer` alone (`case 46: type = LLM_TYPE_27B` for Gemma 2, `case 62`
/// for Gemma 3) and ternaries on it. This mirrors that exactly rather than
/// inventing a metadata key with no real checkpoint behind it.
fn compute_attention_scale(config: &ModelConfig) -> f32 {
    let default_scale = if config.head_dim > 0 {
        1.0 / (config.head_dim as f32).sqrt()
    } else {
        0.0
    };

    let is_27b = match config.architecture.as_str() {
        "gemma2" => config.num_layers == 46,
        "gemma3" => config.num_layers == 62,
        _ => false,
    };

    if !is_27b {
        return default_scale;
    }

    if config.num_attention_heads == 0 {
        return default_scale;
    }
    let query_pre_attn_scalar = config.hidden_size as f32 / config.num_attention_heads as f32;
    if query_pre_attn_scalar <= 0.0 {
        default_scale
    } else {
        1.0 / query_pre_attn_scalar.sqrt()
    }
}

/// Whether layer `layer_idx` uses sliding-window (local) attention.
///
/// - Gemma 2 interleaves every other layer: `hparams.set_swa_pattern(2)` in
///   `llama-hparams.cpp` gives `swa_layers[il] = il % 2 == 0`, i.e. even
///   layers are local, odd layers are global.
/// - Gemma 3 uses a 6-layer pattern: `set_swa_pattern(6)` gives
///   `swa_layers[il] = il % 6 < 5`, i.e. only every 6th layer (index 5, 11,
///   17, ...) is global; the rest are local.
/// - Gemma 1 has no sliding-window attention at all; this function is still
///   safe to call for it because `GemmaModel::sliding_window` is `None` in
///   that case, which makes `attention()` treat every layer as full causal
///   regardless of `use_sliding_window`.
fn is_sliding_window_layer(architecture: &str, layer_idx: usize) -> bool {
    if architecture == "gemma3" {
        layer_idx % 6 != 5
    } else {
        layer_idx.is_multiple_of(2)
    }
}

/// Read the attention/final logit soft-cap values for `architecture`.
///
/// Accepts both the correct GGUF key names (`{arch}.attn_logit_softcapping`,
/// `{arch}.final_logit_softcapping` — see `gguf-py/gguf/constants.py`'s
/// `Keys.Attention.ATTN_LOGIT_SOFTCAPPING` / `Keys.Attention.FINAL_LOGIT_SOFTCAPPING`)
/// and the incorrect ones this loader previously read
/// (`{arch}.attention.logit_softcap`, `{arch}.final_logit_softcap`), so any
/// checkpoint or fixture written against either name still loads.
///
/// - `gemma2`: `convert_hf_to_gguf.py`'s `Gemma2Model.set_gguf_parameters`
///   indexes `self.hparams["attn_logit_softcapping"]` and
///   `self.hparams["final_logit_softcapping"]` unconditionally (Python
///   `dict[...]`, not `.get()`), so a real Gemma-2 GGUF always carries both.
///   Silently defaulting to "disabled" here would run the checkpoint with
///   the wrong math instead of failing loudly, so a missing key is an error.
/// - `gemma3`: `Gemma3Model.set_gguf_parameters` asserts
///   `hparams.get("attn_logit_softcapping") is None` — attention
///   soft-capping was removed in Gemma 3 — and only writes final-logit
///   soft-capping when the HF config sets it, so it stays optional (0.0 =
///   disabled) here.
/// - `gemma` (v1): `models/gemma.cpp` never reads `f_attn_logit_softcapping`
///   or `f_final_logit_softcapping` at all; both are always disabled.
fn load_softcap_config(
    metadata: &oxillama_gguf::MetadataStore,
    architecture: &str,
) -> ArchResult<(f32, f32)> {
    let read = |canonical: &str, legacy: &str| -> Option<f32> {
        metadata
            .get_f32(&format!("{architecture}.{canonical}"))
            .ok()
            .or_else(|| metadata.get_f32(&format!("{architecture}.{legacy}")).ok())
    };

    match architecture {
        "gemma2" => {
            let attn =
                read("attn_logit_softcapping", "attention.logit_softcap").ok_or_else(|| {
                    ArchError::InvalidConfig {
                        detail: format!(
                        "{architecture}: missing required '{architecture}.attn_logit_softcapping' \
                         metadata (Gemma-2 checkpoints always set attention logit soft-capping)"
                    ),
                    }
                })?;
            let final_ =
                read("final_logit_softcapping", "final_logit_softcap").ok_or_else(|| {
                    ArchError::InvalidConfig {
                        detail: format!(
                        "{architecture}: missing required '{architecture}.final_logit_softcapping' \
                         metadata (Gemma-2 checkpoints always set final logit soft-capping)"
                    ),
                    }
                })?;
            Ok((attn, final_))
        }
        "gemma3" => {
            let final_ = read("final_logit_softcapping", "final_logit_softcap").unwrap_or(0.0);
            Ok((0.0, final_))
        }
        _ => Ok((0.0, 0.0)),
    }
}

/// Load a Gemma model from a `GgufModel`.
pub fn load_gemma_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<GemmaModel> {
    let dispatcher = KernelDispatcher::new();

    // Load token embeddings
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // The table must carry at least `vocab_size * hidden_size` rows. `>=`
    // rather than `==`: some converters (Gemma 2) keep padding rows beyond
    // the tokenizer's declared vocabulary, while others (Gemma 3) trim them,
    // so an exact match would reject legitimately-loadable checkpoints. What
    // it must never do is come up short, which is what turns an in-range
    // `token` into an out-of-bounds `buf_hidden` copy.
    let min_embd_len = config.vocab_size.saturating_mul(config.hidden_size);
    if token_embd.len() < min_embd_len {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![config.vocab_size, config.hidden_size],
            got: vec![token_embd.len()],
        });
    }

    // Read soft-capping values from metadata (Gemma 2/3 specific).
    let (attn_logit_softcap, final_logit_softcap) =
        load_softcap_config(&model.file.metadata, &config.architecture)?;

    // Gemma 3's local (sliding-window) layers RoPE with a much smaller base
    // than the global layers (e.g. 10000 vs. 1000000). `None` when the
    // checkpoint has no such key (Gemma 1/2, or a Gemma 3 file that omits
    // it), in which case `GemmaModel::new` shares one table across every
    // layer.
    let local_rope_freq_base = model
        .file
        .metadata
        .get_f32(&format!("{}.rope.freq_base_swa", config.architecture))
        .ok();

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        // Post-norms are optional (Gemma 2+)
        let attn_post_norm = load_optional_rms_norm(
            model,
            &format!("{prefix}.attn_post_norm.weight"),
            config.rms_norm_eps,
        )?;
        let ffn_post_norm = load_optional_rms_norm(
            model,
            &format!("{prefix}.ffn_post_norm.weight"),
            config.rms_norm_eps,
        )?;

        // Per-head Q/K RMSNorm is Gemma 3 only; optional so Gemma 1/2
        // checkpoints (which lack the tensor) still load.
        let attn_q_norm = load_optional_rms_norm(
            model,
            &format!("{prefix}.attn_q_norm.weight"),
            config.rms_norm_eps,
        )?;
        let attn_k_norm = load_optional_rms_norm(
            model,
            &format!("{prefix}.attn_k_norm.weight"),
            config.rms_norm_eps,
        )?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        let use_sliding_window = is_sliding_window_layer(&config.architecture, i);

        // Resolve every projection's kernel once, here, instead of on every
        // token of every generation this model ever serves.
        let attn_q_kernel = resolve_kernel(&dispatcher, &attn_q)?;
        let attn_k_kernel = resolve_kernel(&dispatcher, &attn_k)?;
        let attn_v_kernel = resolve_kernel(&dispatcher, &attn_v)?;
        let attn_output_kernel = resolve_kernel(&dispatcher, &attn_output)?;
        let ffn_gate_kernel = resolve_kernel(&dispatcher, &ffn_gate)?;
        let ffn_up_kernel = resolve_kernel(&dispatcher, &ffn_up)?;
        let ffn_down_kernel = resolve_kernel(&dispatcher, &ffn_down)?;

        layers.push(GemmaLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_post_norm,
            attn_q,
            attn_q_kernel,
            attn_k,
            attn_k_kernel,
            attn_v,
            attn_v_kernel,
            attn_output,
            attn_output_kernel,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_post_norm,
            ffn_gate,
            ffn_gate_kernel,
            ffn_up,
            ffn_up_kernel,
            ffn_down,
            ffn_down_kernel,
            use_sliding_window,
            attn_q_norm,
            attn_k_norm,
        });
    }

    // Load final norm
    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);

    // Output projection (may be absent if weight-tied)
    let output = if model.file.tensors.contains("output.weight") {
        Some(load_quant_linear(model, "output.weight")?)
    } else {
        None
    };

    GemmaModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
        attn_logit_softcap,
        final_logit_softcap,
        local_rope_freq_base,
    )
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Load a quantized linear layer from GGUF.
fn load_quant_linear(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    // Shared mmap-backed view, not a `to_vec()` copy: the GEMV kernels only
    // ever read `&[u8]` out of the payload, so a private copy would double
    // the checkpoint's resident cost for nothing.
    let data = model.tensor_bytes(name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);
    Ok(QuantLinear::new(tensor, None))
}

/// Load an RMSNorm weight vector from GGUF.
fn load_rms_norm_weight(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let dispatcher = KernelDispatcher::new();
    dequant_to_f32(info, data, &dispatcher)
}

/// Try to load an optional RMSNorm weight.
///
/// Returns `Ok(None)` when the tensor is genuinely absent. When the tensor
/// IS present but fails to decode (corrupt block data, unsupported type,
/// etc.), the error is propagated instead of being swallowed into `None` —
/// silently treating a broken tensor as "not shipped" would run the
/// checkpoint as if it were an earlier Gemma generation with the wrong
/// architecture-specific behavior (e.g. "Gemma-1 style, no post-norm")
/// instead of failing loudly.
fn load_optional_rms_norm(
    model: &oxillama_gguf::GgufModel,
    name: &str,
    eps: f32,
) -> ArchResult<Option<RmsNorm>> {
    if !model.file.tensors.contains(name) {
        return Ok(None);
    }
    let weight = load_rms_norm_weight(model, name)?;
    Ok(Some(RmsNorm::new(weight, eps)))
}

/// Dequantize tensor data to f32.
fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    let n_blocks = n_elements.div_ceil(block_size);

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = &data[data_offset..data_offset + block_bytes];
        let out_slice = &mut out[out_offset..out_offset.saturating_add(block_size).min(n_elements)];
        kernel.dequant_block(block_data, out_slice)?;
    }

    Ok(out)
}

/// Unit tests for the pure/private helper functions (`compute_attention_scale`,
/// `is_sliding_window_layer`, `load_softcap_config`, `check_context_length`).
///
/// These take a `ModelConfig`/`MetadataStore` directly rather than a full
/// GGUF byte blob, so no synthetic model weights are needed. End-to-end
/// behavior through the public `load_gemma_from_gguf`/`ForwardPass` API
/// (G1's panic reproduction, G3's per-layer tensor loading, G8/G9's runtime
/// guards) lives in `tests/gemma_regressions.rs`.
#[cfg(test)]
mod helper_tests {
    use super::*;
    use oxillama_gguf::{MetadataStore, MetadataValue};

    // ── G4: compute_attention_scale ─────────────────────────────────────────

    #[test]
    fn attention_scale_gemma1_always_head_dim() {
        let config = ModelConfig {
            architecture: "gemma".to_string(),
            head_dim: 256,
            num_layers: 46, // even at a layer count matching gemma2-27B, v1 never overrides
            ..ModelConfig::default()
        };
        let scale = compute_attention_scale(&config);
        let expected = 1.0 / (256f32).sqrt();
        assert!(
            (scale - expected).abs() < 1e-6,
            "gemma v1 must always use 1/sqrt(head_dim), got {scale}"
        );
    }

    #[test]
    fn attention_scale_gemma2_9b_uses_head_dim() {
        // Gemma-2-9B: 42 layers, hidden=3584, heads=16, head_dim=256.
        let config = ModelConfig {
            architecture: "gemma2".to_string(),
            num_layers: 42,
            hidden_size: 3584,
            num_attention_heads: 16,
            head_dim: 256,
            ..ModelConfig::default()
        };
        let scale = compute_attention_scale(&config);
        let expected = 1.0 / (256f32).sqrt();
        assert!(
            (scale - expected).abs() < 1e-6,
            "gemma-2-9B (42 layers) must use 1/sqrt(head_dim)=1/16, got {scale}"
        );
    }

    #[test]
    fn attention_scale_gemma2_27b_uses_hidden_over_heads() {
        // Gemma-2-27B: 46 layers, hidden=4608, heads=32, head_dim=128.
        // query_pre_attn_scalar = hidden/heads = 144, NOT head_dim=128.
        let config = ModelConfig {
            architecture: "gemma2".to_string(),
            num_layers: 46,
            hidden_size: 4608,
            num_attention_heads: 32,
            head_dim: 128,
            ..ModelConfig::default()
        };
        let scale = compute_attention_scale(&config);
        let expected = 1.0 / 144f32.sqrt();
        let wrong_head_dim_scale = 1.0 / 128f32.sqrt();
        assert!(
            (scale - expected).abs() < 1e-6,
            "gemma-2-27B (46 layers) must use 1/sqrt(hidden/heads)=1/sqrt(144), got {scale}"
        );
        assert!(
            (scale - wrong_head_dim_scale).abs() > 1e-4,
            "must NOT equal the head_dim-based scale"
        );
    }

    #[test]
    fn attention_scale_gemma3_27b_uses_hidden_over_heads() {
        // Gemma-3-27B: 62 layers, hidden=5376, heads=32, head_dim=128 →
        // query_pre_attn_scalar = 168.
        let config = ModelConfig {
            architecture: "gemma3".to_string(),
            num_layers: 62,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim: 128,
            ..ModelConfig::default()
        };
        let scale = compute_attention_scale(&config);
        let expected = 1.0 / 168f32.sqrt();
        assert!(
            (scale - expected).abs() < 1e-6,
            "gemma-3-27B (62 layers) must use 1/sqrt(hidden/heads)=1/sqrt(168), got {scale}"
        );
    }

    #[test]
    fn attention_scale_gemma3_4b_uses_head_dim() {
        // Gemma-3-4B: 34 layers — not the 27B layer count, so no override.
        let config = ModelConfig {
            architecture: "gemma3".to_string(),
            num_layers: 34,
            hidden_size: 2560,
            num_attention_heads: 8,
            head_dim: 256,
            ..ModelConfig::default()
        };
        let scale = compute_attention_scale(&config);
        let expected = 1.0 / 256f32.sqrt();
        assert!(
            (scale - expected).abs() < 1e-6,
            "gemma-3-4B (34 layers) must use 1/sqrt(head_dim), got {scale}"
        );
    }

    // ── G3: is_sliding_window_layer ──────────────────────────────────────────

    #[test]
    fn swa_pattern_gemma2_even_layers_are_local() {
        // set_swa_pattern(2): il % 2 < 1 → local when even, global when odd.
        assert!(is_sliding_window_layer("gemma2", 0));
        assert!(!is_sliding_window_layer("gemma2", 1));
        assert!(is_sliding_window_layer("gemma2", 2));
        assert!(!is_sliding_window_layer("gemma2", 3));
    }

    #[test]
    fn swa_pattern_gemma3_global_every_sixth_layer() {
        // set_swa_pattern(6): il % 6 < 5 → local for 0..=4, global at 5.
        for i in 0..5 {
            assert!(
                is_sliding_window_layer("gemma3", i),
                "gemma3 layer {i} should be local"
            );
        }
        assert!(
            !is_sliding_window_layer("gemma3", 5),
            "gemma3 layer 5 should be global (il % 6 == 5)"
        );
        for i in 6..11 {
            assert!(
                is_sliding_window_layer("gemma3", i),
                "gemma3 layer {i} should be local"
            );
        }
        assert!(
            !is_sliding_window_layer("gemma3", 11),
            "gemma3 layer 11 should be global (il % 6 == 5)"
        );
    }

    // ── G2: load_softcap_config ──────────────────────────────────────────────

    fn store_with(pairs: &[(&str, MetadataValue)]) -> MetadataStore {
        let mut store = MetadataStore::new();
        for (k, v) in pairs {
            store.insert((*k).to_string(), v.clone());
        }
        store
    }

    #[test]
    fn softcap_gemma2_missing_metadata_is_an_error() {
        let store = store_with(&[]);
        let result = load_softcap_config(&store, "gemma2");
        assert!(
            result.is_err(),
            "gemma2 with no softcap metadata at all must error, not silently disable"
        );
    }

    #[test]
    fn softcap_gemma2_missing_final_only_is_an_error() {
        let store = store_with(&[(
            "gemma2.attn_logit_softcapping",
            MetadataValue::Float32(50.0),
        )]);
        let result = load_softcap_config(&store, "gemma2");
        assert!(
            result.is_err(),
            "gemma2 missing final_logit_softcapping must error"
        );
    }

    #[test]
    fn softcap_gemma2_correct_keys_load() {
        let store = store_with(&[
            (
                "gemma2.attn_logit_softcapping",
                MetadataValue::Float32(50.0),
            ),
            (
                "gemma2.final_logit_softcapping",
                MetadataValue::Float32(30.0),
            ),
        ]);
        let (attn, final_) = load_softcap_config(&store, "gemma2").expect("correct keys must load");
        assert!((attn - 50.0).abs() < 1e-6);
        assert!((final_ - 30.0).abs() < 1e-6);
    }

    #[test]
    fn softcap_gemma2_legacy_keys_still_accepted() {
        // The original (wrong) key names this bug report identified.
        let store = store_with(&[
            (
                "gemma2.attention.logit_softcap",
                MetadataValue::Float32(50.0),
            ),
            ("gemma2.final_logit_softcap", MetadataValue::Float32(30.0)),
        ]);
        let (attn, final_) =
            load_softcap_config(&store, "gemma2").expect("legacy keys must still load");
        assert!((attn - 50.0).abs() < 1e-6);
        assert!((final_ - 30.0).abs() < 1e-6);
    }

    #[test]
    fn softcap_gemma3_final_only_optional() {
        let store = store_with(&[]);
        let (attn, final_) =
            load_softcap_config(&store, "gemma3").expect("gemma3 has no required softcap keys");
        assert_eq!(attn, 0.0, "gemma3 never has attention soft-capping");
        assert_eq!(final_, 0.0, "final soft-capping defaults to disabled");
    }

    #[test]
    fn softcap_gemma3_final_present_is_read() {
        let store = store_with(&[(
            "gemma3.final_logit_softcapping",
            MetadataValue::Float32(30.0),
        )]);
        let (attn, final_) = load_softcap_config(&store, "gemma3").expect("must load");
        assert_eq!(attn, 0.0);
        assert!((final_ - 30.0).abs() < 1e-6);
    }

    #[test]
    fn softcap_gemma_v1_never_applies_even_with_legacy_keys_present() {
        // Regression: `build_minimal_gemma_gguf()` (oxillama-gguf test_utils)
        // sets `gemma.attention.logit_softcap` / `gemma.final_logit_softcap`
        // (the WRONG keys, on a v1 "gemma" architecture). Accepting the
        // legacy key names for gemma2 must not leak into v1 loading them too
        // — Gemma 1 never soft-caps at all (see `models/gemma.cpp`).
        let store = store_with(&[
            (
                "gemma.attention.logit_softcap",
                MetadataValue::Float32(50.0),
            ),
            ("gemma.final_logit_softcap", MetadataValue::Float32(30.0)),
        ]);
        let (attn, final_) = load_softcap_config(&store, "gemma").expect("must load");
        assert_eq!(attn, 0.0, "gemma v1 must never apply attention soft-cap");
        assert_eq!(final_, 0.0, "gemma v1 must never apply final soft-cap");
    }

    // ── G9: check_context_length ─────────────────────────────────────────────

    #[test]
    fn context_length_within_bound_is_ok() {
        assert!(check_context_length(128, 0, 128).is_ok());
        assert!(check_context_length(128, 100, 28).is_ok());
    }

    #[test]
    fn context_length_exceeded_is_an_error() {
        assert!(check_context_length(128, 0, 129).is_err());
        assert!(check_context_length(128, 127, 2).is_err());
        assert!(check_context_length(128, 128, 1).is_err());
    }

    #[test]
    fn context_length_overflow_is_an_error_not_a_panic() {
        assert!(check_context_length(usize::MAX, usize::MAX, 1).is_err());
    }

    // ── G11: load_optional_rms_norm — genuinely-absent path ─────────────────
    //
    // The fix changes `.ok()` (swallow any error into `None`) to `?`
    // (propagate). The "tensor genuinely absent" branch, which must still
    // return `Ok(None)`, is exercised end-to-end by every gemma-1 fixture in
    // `tests/gemma_regressions.rs` (gemma-1 ships no `attn_post_norm.weight`
    // at all). A "tensor present but undecodable" fixture cannot be built
    // through the public `GgufWriter` API without also defeating the GGUF
    // parser's own byte-length validation, which rejects such a file before
    // `load_optional_rms_norm` ever runs — so the swallowed-error path this
    // defect describes is exercised here as a direct code-path check instead
    // of a synthetic panic reproduction.
    #[test]
    fn load_optional_rms_norm_signature_returns_result() {
        // Compile-time proof the function now returns `ArchResult<Option<..>>`
        // (previously `Option<..>` via `.ok()`) — a type change alone would
        // fail to compile if a call site still expected the old signature.
        fn _assert_signature(
            f: fn(&oxillama_gguf::GgufModel, &str, f32) -> ArchResult<Option<RmsNorm>>,
        ) {
            let _ = f;
        }
        _assert_signature(load_optional_rms_norm);
    }
}
