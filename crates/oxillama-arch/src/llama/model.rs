//! LLaMA transformer forward pass implementation.
//!
//! This module contains the full LLaMA decoder-only transformer:
//! embedding → N×(RMSNorm → GQA → residual → RMSNorm → SwiGLU FFN → residual) → RMSNorm → LM head
//!
//! When `config.num_experts > 0`, the FFN layers use a sparse Mixture-of-Experts
//! (MoE) layout (Mixtral-style) instead of the standard dense SwiGLU FFN.  That
//! is not a hypothetical: llama.cpp converts `MixtralForCausalLM` to
//! `MODEL_ARCH.LLAMA`, so every Mixtral GGUF arrives *here*, never through
//! `crate::mixtral`.
//!
//! Weight loading lives in [`super::loader`]; the multi-token prefill in
//! [`super::batch`].

use std::sync::Arc;

use oxillama_quant::{
    quantize_activations_q8_0_into, KernelDispatcher, QuantKernel, Q8_0_ACT_BLOCK_BYTES,
};

use crate::common::embedding::TokenEmbedding;
use crate::common::linear::QuantLinear;
use crate::common::moe::{MoeScratch, QuantMoeFfn};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::attention::{axpy_f32, dot_f32, softmax_inplace};
use crate::llama::batch::{BatchScratch, MIN_BATCH_TOKENS};
use crate::llama::rope_norm::apply_rope_norm;
use crate::lora::LoadedLora;
use crate::traits::{
    remap_kernel_slot, BatchedKvView, ForwardPass, KvCacheAccess, QuantKernelRemap,
};

/// Weights for a dense SwiGLU FFN layer.
///
/// Stored on the heap (boxed) so that `FfnVariant` doesn't have a large-size
/// difference between its variants.
pub struct DenseFfn {
    /// Gate projection `[intermediate_size, hidden_size]`.
    pub gate: QuantLinear,
    /// Up projection `[intermediate_size, hidden_size]`.
    pub up: QuantLinear,
    /// Down projection `[hidden_size, intermediate_size]`.
    pub down: QuantLinear,
    /// Kernel for [`Self::gate`], resolved once at load time.
    pub gate_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::up`], resolved once at load time.
    pub up_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::down`], resolved once at load time.
    pub down_kernel: Arc<dyn QuantKernel>,
}

/// FFN layer variant: either a standard dense SwiGLU or a sparse MoE FFN.
pub enum FfnVariant {
    /// Standard dense SwiGLU FFN using quantized weights.
    Dense(Box<DenseFfn>),
    /// Sparse Mixture-of-Experts FFN, experts left quantized.
    Moe(Box<QuantMoeFfn>),
}

/// A single transformer layer (decoder block).
///
/// The `*_kernel` fields hold each projection's [`QuantKernel`], looked up from
/// its tensor type exactly once in [`super::loader::load_llama_from_gguf`].
/// The decode loop used to call `dispatcher.get_kernel()` — a walk down the full
/// tensor-type match ladder plus a `Box` allocation — seven times per layer per
/// token, i.e. 225 times per token on a 32-layer model including the LM head.
/// Storing the resolved kernel as an `Arc` on the layer that owns the weight
/// turns every one of those into a field load and a cheap deref.
pub struct LlamaLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection `[num_heads * head_dim, hidden_size]`.
    pub attn_q: QuantLinear,
    /// Key projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantLinear,
    /// Value projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantLinear,
    /// Output projection `[hidden_size, num_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// Kernel for [`Self::attn_q`].
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`].
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`].
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`].
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN variant: dense SwiGLU or sparse MoE.
    pub ffn: FfnVariant,
}

/// Complete LLaMA model with all weights and forward pass logic.
pub struct LlamaModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding table.
    ///
    /// Held in whatever form the checkpoint allows — quantized and looked up one
    /// row at a time when the GGUF's rows are block-aligned (the normal case).
    /// Llama-3-8B's `[128256, 4096]` table is 525 M elements; as `Vec<f32>` that
    /// was **2.10 GB** resident against ~295 MB of `Q4_K` on disk, for a matrix
    /// a forward pass reads one row of per token.  See [`TokenEmbedding`].
    pub token_embd: TokenEmbedding,
    /// Transformer layers.
    pub layers: Vec<LlamaLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Resolved kernel for [`Self::output`] — see [`LlamaLayer`]'s kernel fields.
    output_kernel: Arc<dyn QuantKernel>,
    /// RoPE precomputed frequency table.
    ///
    /// Applied with LLaMA's NORM convention — see `super::rope_norm`, **not**
    /// [`RopeTable::apply`], which is the NeoX split.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,

    // Scratch buffers (reused across forward calls to avoid allocation)
    pub(crate) buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated attention heads, `[num_heads * head_dim]`.
    ///
    /// **Not** `hidden_size`: the two coincide for Llama-2/3 but not in general
    /// (Qwen3-4B attends over 32 × 128 = 4096 with a 2560-wide residual stream),
    /// and `attn_output`'s input width is the former.  Sizing this from
    /// `hidden_size` while indexing it by `num_heads * head_dim` was a latent
    /// out-of-bounds panic.
    buf_attn_out: Vec<f32>,
    /// Output of `attn_output`'s projection, added into `buf_hidden`.
    ///
    /// Preallocated instead of `vec![0.0; hidden_size]`-ed inside `attention()`:
    /// that was a fresh heap allocation per layer per token — 32 × 16 KB per
    /// token on Llama-3-8B.
    buf_proj: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    /// Q8_0 image of whichever activation vector the next matmul consumes.
    ///
    /// One buffer suffices because the projections that share an input
    /// (`attn_q`/`attn_k`/`attn_v`, then `ffn_gate`/`ffn_up`) are issued back to
    /// back, so the vector is quantized once and consumed before the next one
    /// overwrites it.
    buf_acts_q8: Vec<u8>,
    /// Router/expert scratch for MoE layers; untouched by dense models.
    moe_scratch: MoeScratch,
    /// Scratch for the batched prefill path ([`super::batch`]).  Empty until the
    /// first multi-token `forward`/`embed`; decode never touches it.
    pub(crate) batch: BatchScratch,
}

impl LlamaModel {
    /// Create a new `LlamaModel` from preloaded weights.
    ///
    /// Takes the embedding table already dequantized.  Loaders that can keep it
    /// quantized — [`super::loader::load_llama_from_gguf`] does — should call
    /// [`Self::with_embedding`] and hand over a [`TokenEmbedding::Quantized`].
    ///
    /// Fails if `output`'s tensor type has no registered [`QuantKernel`]: the
    /// kernel is resolved once here rather than on every forward pass.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<LlamaLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        Self::with_embedding(
            config,
            TokenEmbedding::dense(token_embd, hidden_size),
            layers,
            output_norm,
            output,
        )
    }

    /// Create a new `LlamaModel` over an arbitrary [`TokenEmbedding`].
    ///
    /// Fails if `output`'s tensor type has no registered [`QuantKernel`].
    pub fn with_embedding(
        config: ModelConfig,
        token_embd: TokenEmbedding,
        layers: Vec<LlamaLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let attn_dim = num_heads * head_dim;
        let num_experts = config.num_experts;

        let rope = RopeTable::new(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );
        let dispatcher = KernelDispatcher::new();
        let output_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(output.weight.tensor_type)?.into();

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            dispatcher,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; attn_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; attn_dim],
            buf_proj: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            // Widest activation any projection consumes, rounded up to whole
            // K-quant blocks (256 weights → 8 Q8_0 blocks).
            buf_acts_q8: Vec::with_capacity(
                hidden_size
                    .max(attn_dim)
                    .max(intermediate_size)
                    .div_ceil(256)
                    * 8
                    * Q8_0_ACT_BLOCK_BYTES,
            ),
            moe_scratch: MoeScratch::new(hidden_size, intermediate_size, num_experts),
            batch: BatchScratch::default(),
        })
    }

    /// Run every token of `tokens` through all layers, leaving the last token's
    /// pre-output-norm hidden state in `self.buf_hidden`.
    ///
    /// Multi-token calls (prompt prefill) take the batched path in
    /// [`super::batch`] when every projection supports it; single-token calls
    /// (decode) and models the batched path cannot serve replay the per-token
    /// loop.  The two produce bit-identical hidden states.
    pub(crate) fn run_layers(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let start_pos = kv_cache.seq_len();
        // Guard the context window *before* any weight is touched.  Past
        // `max_context_length` the RoPE table has no entry for the position and
        // the KV cache has no slot for the key: the old code walked straight off
        // the end of both and aborted the process.
        let end = start_pos.saturating_add(tokens.len());
        if end > self.config.max_context_length {
            return Err(ArchError::ForwardPassError {
                layer: 0,
                message: format!(
                    "context overflow: {} cached + {} new tokens exceeds max_context_length {}",
                    start_pos,
                    tokens.len(),
                    self.config.max_context_length
                ),
            });
        }

        if tokens.len() >= MIN_BATCH_TOKENS && self.batched_prefill_supported() {
            return self.forward_prefill_batched(tokens, kv_cache);
        }

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

        Ok(())
    }

    /// Load the residual stream with `token`'s embedding row.
    ///
    /// Dequantizes exactly one row.  An out-of-vocabulary id is an **error**,
    /// not a panic: `vocab_size` is frequently over-estimated (`config.rs` falls
    /// back to the tokenizer token-array length and then to a hard-coded 32000),
    /// and a server must not abort because a client sent a stray token id.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let Self {
            token_embd,
            dispatcher,
            buf_hidden,
            ..
        } = self;
        token_embd.row_into(dispatcher, token, buf_hidden)
    }

    /// Run grouped-query attention for a single layer.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let attn_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        if num_kv_heads == 0 || head_dim == 0 || num_heads < num_kv_heads {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "attention geometry: head_count={num_heads}, head_count_kv={num_kv_heads}, \
                     head_dim={head_dim}"
                ),
            });
        }
        let heads_per_kv = num_heads / num_kv_heads;

        let layer = &self.layers[layer_idx];

        // Q/K/V all read the same post-attn-norm activation vector, so it is
        // quantized to Q8_0 exactly once here and shared by all three GEMVs —
        // the `f32 → i8` conversion then happens once per matmul *input* rather
        // than once per weight *row*, and the row loop multiplies i8 by i8 in
        // integer registers.  The block count is the max over the three so that
        // a mixed-precision checkpoint (Q4_K q/k with a Q6_K v, which is exactly
        // what `Q4_K_M` ships) still gets a long enough buffer.
        let q_kernel: &dyn QuantKernel = &*layer.attn_q_kernel;
        let k_kernel: &dyn QuantKernel = &*layer.attn_k_kernel;
        let v_kernel: &dyn QuantKernel = &*layer.attn_v_kernel;
        let q_fused = layer.attn_q.q8_fused_blocks(q_kernel);
        let k_fused = layer.attn_k.q8_fused_blocks(k_kernel);
        let v_fused = layer.attn_v.q8_fused_blocks(v_kernel);
        if let Some(n_blocks) = q_fused
            .iter()
            .chain(&k_fused)
            .chain(&v_fused)
            .copied()
            .max()
        {
            quantize_activations_q8_0_into(&self.buf_norm, n_blocks, &mut self.buf_acts_q8);
        }

        if q_fused.is_some() {
            layer.attn_q.forward_q8_fused(
                q_kernel,
                &self.buf_norm,
                &self.buf_acts_q8,
                &mut self.buf_q,
            )?;
        } else {
            layer
                .attn_q
                .forward(q_kernel, &self.buf_norm, &mut self.buf_q)?;
        }
        if k_fused.is_some() {
            layer.attn_k.forward_q8_fused(
                k_kernel,
                &self.buf_norm,
                &self.buf_acts_q8,
                &mut self.buf_k,
            )?;
        } else {
            layer
                .attn_k
                .forward(k_kernel, &self.buf_norm, &mut self.buf_k)?;
        }
        if v_fused.is_some() {
            layer.attn_v.forward_q8_fused(
                v_kernel,
                &self.buf_norm,
                &self.buf_acts_q8,
                &mut self.buf_v,
            )?;
        } else {
            layer
                .attn_v
                .forward(v_kernel, &self.buf_norm, &mut self.buf_v)?;
        }

        // RoPE, LLaMA's NORM convention (consecutive pairs) — see `rope_norm`.
        for h in 0..num_heads {
            apply_rope_norm(
                &self.rope,
                &mut self.buf_q[h * head_dim..(h + 1) * head_dim],
                position,
            );
        }
        for h in 0..num_kv_heads {
            apply_rope_norm(
                &self.rope,
                &mut self.buf_k[h * head_dim..(h + 1) * head_dim],
                position,
            );
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;
        let needed = seq_len * kv_dim;
        if cached_keys.len() < needed || cached_values.len() < needed {
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!(
                    "kv cache exposes {} key / {} value floats, attention at position \
                     {position} needs {needed}",
                    cached_keys.len(),
                    cached_values.len()
                ),
            });
        }
        let scale = 1.0 / (head_dim as f32).sqrt();

        // One task per head.  A head's output is a contiguous `head_dim` slice
        // of `buf_attn_out` that no other head reads or writes, so handing each
        // to a worker changes no accumulation order — and attention was the last
        // single-threaded stretch of the forward pass, with a cost that grows
        // linearly in the context length while the GEMVs around it already used
        // every core.  The per-head score vector is the `init` state, so it is
        // allocated once per worker rather than once per head.
        {
            let q: &[f32] = &self.buf_q;
            oxillama_quant::parallel::for_each_chunk_init(
                &mut self.buf_attn_out[..attn_dim],
                head_dim,
                // Roughly the MACs one head performs: two passes (scores, then
                // the value-weighted sum) over `seq_len` cached keys.
                seq_len * 2,
                Vec::<f32>::new,
                |scores, h, out_head| {
                    let kv_head = h / heads_per_kv;
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];

                    scores.clear();
                    scores.resize(seq_len, 0.0);
                    for (pos, score) in scores.iter_mut().enumerate() {
                        let off = pos * kv_dim + kv_head * head_dim;
                        *score = dot_f32(q_head, &cached_keys[off..off + head_dim]) * scale;
                    }

                    softmax_inplace(scores);

                    out_head.fill(0.0);
                    for (pos, &w) in scores.iter().enumerate() {
                        let off = pos * kv_dim + kv_head * head_dim;
                        axpy_f32(out_head, w, &cached_values[off..off + head_dim]);
                    }
                },
            );
        }

        // Project the concatenated heads back to the residual stream.
        let layer = &self.layers[layer_idx];
        let o_kernel: &dyn QuantKernel = &*layer.attn_output_kernel;
        let o_fused = layer.attn_output.q8_fused_blocks(o_kernel);
        if let Some(n_blocks) = o_fused {
            quantize_activations_q8_0_into(&self.buf_attn_out, n_blocks, &mut self.buf_acts_q8);
            layer.attn_output.forward_q8_fused(
                o_kernel,
                &self.buf_attn_out,
                &self.buf_acts_q8,
                &mut self.buf_proj,
            )?;
        } else {
            layer
                .attn_output
                .forward(o_kernel, &self.buf_attn_out, &mut self.buf_proj)?;
        }

        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj.iter()) {
            *h += p;
        }

        Ok(())
    }

    /// Run the feed-forward network for a single layer.
    ///
    /// Dense: `FFN(x) = down(silu(gate(x)) * up(x))`
    /// MoE:   weighted sum of top-K expert SwiGLU outputs
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        match &self.layers[layer_idx].ffn {
            FfnVariant::Dense(dense) => {
                let gate_kernel: &dyn QuantKernel = &*dense.gate_kernel;
                let up_kernel: &dyn QuantKernel = &*dense.up_kernel;
                let down_kernel: &dyn QuantKernel = &*dense.down_kernel;

                // `gate` and `up` share the post-ffn-norm activation.
                let gate_fused = dense.gate.q8_fused_blocks(gate_kernel);
                let up_fused = dense.up.q8_fused_blocks(up_kernel);
                if let Some(n_blocks) = gate_fused.iter().chain(&up_fused).copied().max() {
                    quantize_activations_q8_0_into(&self.buf_norm, n_blocks, &mut self.buf_acts_q8);
                }

                if gate_fused.is_some() {
                    dense.gate.forward_q8_fused(
                        gate_kernel,
                        &self.buf_norm,
                        &self.buf_acts_q8,
                        &mut self.buf_gate,
                    )?;
                } else {
                    dense
                        .gate
                        .forward(gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
                }
                if up_fused.is_some() {
                    dense.up.forward_q8_fused(
                        up_kernel,
                        &self.buf_norm,
                        &self.buf_acts_q8,
                        &mut self.buf_up,
                    )?;
                } else {
                    dense
                        .up
                        .forward(up_kernel, &self.buf_norm, &mut self.buf_up)?;
                }

                swiglu_inplace(&mut self.buf_gate, &self.buf_up);

                match dense.down.q8_fused_blocks(down_kernel) {
                    Some(n_blocks) => {
                        quantize_activations_q8_0_into(
                            &self.buf_gate,
                            n_blocks,
                            &mut self.buf_acts_q8,
                        );
                        dense.down.forward_q8_fused(
                            down_kernel,
                            &self.buf_gate,
                            &self.buf_acts_q8,
                            &mut self.buf_ffn_out,
                        )?;
                    }
                    None => {
                        dense
                            .down
                            .forward(down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?
                    }
                }
            }
            FfnVariant::Moe(moe) => {
                // No `self.buf_norm.clone()` here: the router reads `buf_norm`
                // and the experts write `buf_ffn_out`/`moe_scratch`, which are
                // disjoint fields, so nothing needs copying.
                moe.forward(&self.buf_norm, &mut self.buf_ffn_out, &mut self.moe_scratch)?;
            }
        }

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }

    /// Project the final hidden state through the LM head into `buf_logits`.
    fn project_logits(&mut self) -> ArchResult<()> {
        // `buf_logits` may have been handed to the caller by ownership on a
        // previous call (see the `mem::take` in `forward`) and therefore be
        // empty; restore its length before the kernel writes into it.
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        let output_kernel: &dyn QuantKernel = &*self.output_kernel;
        match self.output.q8_fused_blocks(output_kernel) {
            Some(n_blocks) => {
                quantize_activations_q8_0_into(&self.buf_hidden, n_blocks, &mut self.buf_acts_q8);
                self.output.forward_q8_fused(
                    output_kernel,
                    &self.buf_hidden,
                    &self.buf_acts_q8,
                    &mut self.buf_logits,
                )?;
            }
            None => self
                .output
                .forward(output_kernel, &self.buf_hidden, &mut self.buf_logits)?,
        }
        Ok(())
    }
}

impl ForwardPass for LlamaModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        self.output_norm.forward(&mut self.buf_hidden);
        self.project_logits()?;

        // Hand the freshly computed logits to the caller by ownership transfer
        // instead of `self.buf_logits.clone()`.  The clone was a 513 KB
        // (Llama-3 vocab_size == 128256) allocation-and-memcpy on every decoded
        // token; `mem::take` moves the `Vec`'s (ptr, len, cap) for free and
        // leaves `buf_logits` empty, which `project_logits` repairs next call.
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of the LM-head projection (output.weight) that maps
    /// hidden_size → vocab_size. Returns a `hidden_size`-dimensional vector
    /// suitable for L2-normalised semantic embeddings.
    ///
    /// Unlike `forward`, this **clones**: `buf_hidden` is the model's live
    /// residual stream, not a write-only output buffer, and it is `hidden_size`
    /// (16 KB) rather than `vocab_size` (513 KB) — and `embed` runs once per
    /// request, not once per generated token.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
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

    /// Attach LoRA adapters to this model's linear layers.
    ///
    /// Iterates over all transformer layers and the LM head, calling
    /// `set_lora()` on any `QuantLinear` whose GGUF tensor name appears in
    /// the loaded adapter.
    ///
    /// For MoE layers, LoRA patching on individual expert weights is not
    /// currently supported and is silently skipped. Only the attention
    /// projections are patched.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections are present in all variants.
            let attn_candidates: [(&str, &mut QuantLinear); 4] = [
                (&format!("blk.{i}.attn_q.weight"), &mut layer.attn_q),
                (&format!("blk.{i}.attn_k.weight"), &mut layer.attn_k),
                (&format!("blk.{i}.attn_v.weight"), &mut layer.attn_v),
                (
                    &format!("blk.{i}.attn_output.weight"),
                    &mut layer.attn_output,
                ),
            ];
            for (tensor_name, linear) in attn_candidates {
                if let Some(adapter) = lora.get(tensor_name) {
                    linear.set_lora(adapter);
                }
            }

            // FFN LoRA: only dense layers support it.
            if let FfnVariant::Dense(dense) = &mut layer.ffn {
                let ffn_candidates: [(&str, &mut QuantLinear); 3] = [
                    (&format!("blk.{i}.ffn_gate.weight"), &mut dense.gate),
                    (&format!("blk.{i}.ffn_up.weight"), &mut dense.up),
                    (&format!("blk.{i}.ffn_down.weight"), &mut dense.down),
                ];
                for (tensor_name, linear) in ffn_candidates {
                    if let Some(adapter) = lora.get(tensor_name) {
                        linear.set_lora(adapter);
                    }
                }
            }
            // MoE expert LoRA is not supported in this implementation.
        }
        Ok(())
    }

    fn unapply_all_loras(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn_q.clear_lora();
            layer.attn_k.clear_lora();
            layer.attn_v.clear_lora();
            layer.attn_output.clear_lora();
            if let FfnVariant::Dense(dense) = &mut layer.ffn {
                dense.gate.clear_lora();
                dense.up.clear_lora();
                dense.down.clear_lora();
            }
        }
    }

    /// Visit every kernel the per-token path dispatches through: each layer's
    /// four attention projections, its three dense-FFN projections, then the
    /// LM head as `(None, "output")`.
    ///
    /// Two bindings this model owns are deliberately **not** offered:
    ///
    /// * [`FfnVariant::Moe`] layers.  Their expert and router kernels live
    ///   inside [`QuantMoeFfn`], which does not expose them, so a Mixtral
    ///   checkpoint (which arrives here, not through `crate::mixtral`) yields
    ///   only its attention sites and its LM head.
    /// * The token-embedding row lookup, which re-dispatches from
    ///   [`Self::dispatcher`] per token rather than reading a stored `Arc`.
    ///
    /// The tiled prefill in `super::batch` likewise re-dispatches per tile,
    /// so a remapped kernel governs decode but not a multi-token `forward`
    /// that qualifies for the batched path — see the trait contract.
    fn remap_quant_kernels(&mut self, f: &mut QuantKernelRemap<'_>) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let idx = Some(i);
            remap_kernel_slot(
                f,
                idx,
                "attn_q",
                &layer.attn_q.weight,
                &mut layer.attn_q_kernel,
            );
            remap_kernel_slot(
                f,
                idx,
                "attn_k",
                &layer.attn_k.weight,
                &mut layer.attn_k_kernel,
            );
            remap_kernel_slot(
                f,
                idx,
                "attn_v",
                &layer.attn_v.weight,
                &mut layer.attn_v_kernel,
            );
            remap_kernel_slot(
                f,
                idx,
                "attn_output",
                &layer.attn_output.weight,
                &mut layer.attn_output_kernel,
            );

            if let FfnVariant::Dense(dense) = &mut layer.ffn {
                remap_kernel_slot(
                    f,
                    idx,
                    "ffn_gate",
                    &dense.gate.weight,
                    &mut dense.gate_kernel,
                );
                remap_kernel_slot(f, idx, "ffn_up", &dense.up.weight, &mut dense.up_kernel);
                remap_kernel_slot(
                    f,
                    idx,
                    "ffn_down",
                    &dense.down.weight,
                    &mut dense.down_kernel,
                );
            }
        }

        remap_kernel_slot(
            f,
            None,
            "output",
            &self.output.weight,
            &mut self.output_kernel,
        );
    }

    fn forward_batched(
        &mut self,
        q_batch: &[f32],
        kv_view: &dyn BatchedKvView,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> ArchResult<Vec<f32>> {
        // Proof-of-concept: iterate per-slot, run scaled-dot-product attention for each.
        let head_stride = num_heads * head_dim;
        if head_stride == 0 {
            return Err(ArchError::ForwardPassError {
                layer: 0,
                message: "num_heads and head_dim must be > 0".to_string(),
            });
        }

        let batch_size = q_batch.len() / head_stride;
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let slot_count = kv_view.slot_count();
        if slot_count != batch_size {
            return Err(ArchError::ForwardPassError {
                layer: 0,
                message: format!("batch_size {batch_size} != kv_view slot_count {slot_count}"),
            });
        }

        let mut out = vec![0.0f32; batch_size * head_stride];

        // Process each slot independently (single-slot scaled-dot-product attention).
        for slot_idx in 0..slot_count {
            let q_slot = &q_batch[slot_idx * head_stride..(slot_idx + 1) * head_stride];
            let (keys, values) = kv_view.kv_for_slot(slot_idx);
            let pos = kv_view.position(slot_idx);

            if pos == 0 || keys.is_empty() {
                // No KV cache — output zeros (will be projected by the caller).
                continue;
            }

            // kv_dim = num_heads * head_dim (full KV, not GQA for simplicity).
            let kv_head_dim = if keys.len() % (pos * num_heads) == 0 {
                keys.len() / (pos * num_heads)
            } else {
                head_dim // fallback
            };

            // Per-head scaled dot-product attention.
            for h in 0..num_heads {
                let q_head = &q_slot[h * head_dim..(h + 1) * head_dim];

                // Compute attention scores: q @ K^T
                let kv_h = h.min(
                    keys.len()
                        .checked_div(pos.saturating_mul(kv_head_dim))
                        .unwrap_or(0)
                        .saturating_sub(1),
                );
                let mut scores = vec![0.0f32; pos];
                for (p, score) in scores.iter_mut().enumerate() {
                    let k_base = p * num_heads * kv_head_dim + kv_h * kv_head_dim;
                    let k_end = (k_base + kv_head_dim).min(keys.len());
                    if k_base >= keys.len() {
                        break;
                    }
                    let k_pos = &keys[k_base..k_end];
                    *score = dot_f32(q_head, k_pos) * scale;
                }

                // Softmax over scores.
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
                let inv_sum = 1.0 / sum.max(1e-10);
                for s in &mut scores {
                    *s = (*s - max_s).exp() * inv_sum;
                }

                // Weighted sum of values.
                let out_start = slot_idx * head_stride + h * head_dim;
                let out_end = slot_idx * head_stride + (h + 1) * head_dim;
                let out_head = &mut out[out_start..out_end];
                for (p, &attn_weight) in scores.iter().enumerate() {
                    let v_start =
                        (p * num_heads * kv_head_dim + kv_h * kv_head_dim).min(values.len());
                    let v_end = (v_start + head_dim).min(values.len());
                    if v_start >= values.len() {
                        break;
                    }
                    axpy_f32(out_head, attn_weight, &values[v_start..v_end]);
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::moe::{QuantExpert, QuantMoeFfn};
    use crate::traits::remap_test_support::CountingKernel;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Deliberately small, and deliberately mutually distinct: `ATTN_DIM`,
    // `HIDDEN`, `FFN` and `VOCAB` all differ, so a site that reports the wrong
    // weight tensor is caught by its shape rather than passing unnoticed.
    const HIDDEN: usize = 8;
    const FFN: usize = 16;
    const VOCAB: usize = 12;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 6;
    const ATTN_DIM: usize = HEADS * HEAD_DIM;
    const LAYERS: usize = 2;
    const EXPERTS: usize = 2;

    /// An F32 `[out_features, in_features]` weight with deterministic,
    /// non-degenerate values.
    ///
    /// Zero weights would make every logit zero and the parity assertion
    /// vacuous, so `seed` decorrelates the projections.
    fn f32_linear(out_features: usize, in_features: usize, seed: usize) -> QuantLinear {
        let mut bytes = Vec::with_capacity(out_features * in_features * 4);
        for i in 0..out_features * in_features {
            let v = (((i * 37 + seed * 13) % 23) as f32 - 11.0) / 32.0;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        QuantLinear::new(
            QuantTensor::new(bytes, vec![out_features, in_features], GgufTensorType::F32),
            None,
        )
    }

    fn f32_kernel(dispatcher: &KernelDispatcher) -> Arc<dyn QuantKernel> {
        dispatcher
            .get_kernel(GgufTensorType::F32)
            .expect("the F32 kernel must be registered")
            .into()
    }

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            hidden_size: HIDDEN,
            intermediate_size: FFN,
            num_layers: LAYERS,
            num_attention_heads: HEADS,
            num_kv_heads: HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 16,
            ..ModelConfig::default()
        }
    }

    fn attention_block(dispatcher: &KernelDispatcher, layer: usize) -> LlamaLayer {
        LlamaLayer {
            attn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            attn_q: f32_linear(ATTN_DIM, HIDDEN, layer * 7 + 1),
            attn_k: f32_linear(ATTN_DIM, HIDDEN, layer * 7 + 2),
            attn_v: f32_linear(ATTN_DIM, HIDDEN, layer * 7 + 3),
            attn_output: f32_linear(HIDDEN, ATTN_DIM, layer * 7 + 4),
            attn_q_kernel: f32_kernel(dispatcher),
            attn_k_kernel: f32_kernel(dispatcher),
            attn_v_kernel: f32_kernel(dispatcher),
            attn_output_kernel: f32_kernel(dispatcher),
            ffn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            ffn: FfnVariant::Dense(Box::new(DenseFfn {
                gate: f32_linear(FFN, HIDDEN, layer * 7 + 5),
                up: f32_linear(FFN, HIDDEN, layer * 7 + 6),
                down: f32_linear(HIDDEN, FFN, layer * 7 + 7),
                gate_kernel: f32_kernel(dispatcher),
                up_kernel: f32_kernel(dispatcher),
                down_kernel: f32_kernel(dispatcher),
            })),
        }
    }

    /// A two-layer dense LLaMA over F32 weights.
    fn tiny_llama() -> LlamaModel {
        let dispatcher = KernelDispatcher::new();
        let layers = (0..LAYERS)
            .map(|l| attention_block(&dispatcher, l))
            .collect();
        let token_embd: Vec<f32> = (0..VOCAB * HIDDEN)
            .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
            .collect();
        LlamaModel::new(
            tiny_config(),
            token_embd,
            layers,
            RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            f32_linear(VOCAB, HIDDEN, 99),
        )
        .expect("the F32 LM head must resolve a kernel")
    }

    /// The same model with layer 0's dense FFN replaced by a 2-expert MoE.
    fn tiny_llama_with_moe_layer() -> LlamaModel {
        let dispatcher = KernelDispatcher::new();
        let mut layers: Vec<LlamaLayer> = (0..LAYERS)
            .map(|l| attention_block(&dispatcher, l))
            .collect();
        let experts = (0..EXPERTS)
            .map(|e| {
                QuantExpert::new(
                    f32_linear(FFN, HIDDEN, 40 + e),
                    f32_linear(FFN, HIDDEN, 50 + e),
                    f32_linear(HIDDEN, FFN, 60 + e),
                )
                .expect("expert projections must compose")
            })
            .collect();
        let moe = QuantMoeFfn::new(f32_linear(EXPERTS, HIDDEN, 70), experts, 1)
            .expect("router and experts must agree on hidden_size");
        layers[0].ffn = FfnVariant::Moe(Box::new(moe));

        let token_embd: Vec<f32> = (0..VOCAB * HIDDEN)
            .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
            .collect();
        LlamaModel::new(
            tiny_config(),
            token_embd,
            layers,
            RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            f32_linear(VOCAB, HIDDEN, 99),
        )
        .expect("the F32 LM head must resolve a kernel")
    }

    /// Per-layer KV cache backing the forward-pass tests.
    struct TestKv {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        seq_len: usize,
    }

    impl TestKv {
        fn new(layers: usize) -> Self {
            Self {
                keys: vec![Vec::new(); layers],
                values: vec![Vec::new(); layers],
                seq_len: 0,
            }
        }
    }

    impl KvCacheAccess for TestKv {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            self.keys[layer].extend_from_slice(key);
            self.values[layer].extend_from_slice(value);
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys[layer])
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.values[layer])
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
    }

    /// Collect `(layer, role, weight shape)` for every visited site.
    fn visit_sites(model: &mut LlamaModel) -> Vec<(Option<usize>, &'static str, Vec<usize>)> {
        let mut sites = Vec::new();
        model.remap_quant_kernels(&mut |site, kernel| {
            sites.push((site.layer, site.role, site.weight.shape.clone()));
            kernel
        });
        sites
    }

    /// Every projection the decode path reads is offered exactly once, in
    /// block order, with the weight tensor that projection owns.
    #[test]
    fn remap_visits_seven_sites_per_layer_plus_the_lm_head() {
        let mut model = tiny_llama();
        let sites = visit_sites(&mut model);

        let mut expected: Vec<(Option<usize>, &'static str, Vec<usize>)> = Vec::new();
        for l in 0..LAYERS {
            expected.push((Some(l), "attn_q", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_k", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_v", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_output", vec![HIDDEN, ATTN_DIM]));
            expected.push((Some(l), "ffn_gate", vec![FFN, HIDDEN]));
            expected.push((Some(l), "ffn_up", vec![FFN, HIDDEN]));
            expected.push((Some(l), "ffn_down", vec![HIDDEN, FFN]));
        }
        expected.push((None, "output", vec![VOCAB, HIDDEN]));

        assert_eq!(
            sites.len(),
            7 * LAYERS + 1,
            "a dense model exposes 7 sites per layer plus the LM head"
        );
        assert_eq!(
            sites, expected,
            "roles, layer indices and weight shapes must match the contract exactly"
        );
    }

    /// A MoE layer contributes only its attention sites: the expert and router
    /// kernels live inside `QuantMoeFfn` and are out of contract.
    #[test]
    fn remap_skips_moe_expert_kernels() {
        let mut model = tiny_llama_with_moe_layer();
        let sites = visit_sites(&mut model);

        assert_eq!(
            sites.len(),
            4 + 7 + 1,
            "the MoE layer contributes 4 attention sites, the dense layer 7, plus the LM head"
        );
        assert!(
            !sites
                .iter()
                .any(|(layer, role, _)| *layer == Some(0) && role.starts_with("ffn_")),
            "no FFN site may be reported for the MoE layer"
        );
        assert!(
            sites
                .iter()
                .any(|(layer, role, _)| layer.is_none() && *role == "output"),
            "the LM head must still be offered on a MoE model"
        );
    }

    /// Wrapping every kernel in a forwarding decorator changes no logit, and
    /// the decorator really is on the path that produced them.
    #[test]
    fn remapped_delegating_kernels_reproduce_the_baseline_logits() {
        // Single-token steps: a multi-token `forward` may take the tiled
        // prefill path, which re-dispatches kernels and would bypass the remap.
        let mut baseline = tiny_llama();
        let mut kv = TestKv::new(LAYERS);
        let first = baseline
            .forward(&[3], &mut kv)
            .expect("baseline decode step must succeed");
        let second = baseline
            .forward(&[7], &mut kv)
            .expect("baseline decode step must succeed");
        assert!(
            first.iter().any(|&v| v != 0.0),
            "the fixture must produce non-trivial logits, else parity is vacuous"
        );

        let mut remapped = tiny_llama();
        let calls = Arc::new(AtomicUsize::new(0));
        remapped.remap_quant_kernels(&mut |_site, kernel| {
            CountingKernel::wrap(kernel, Arc::clone(&calls))
        });

        let mut kv = TestKv::new(LAYERS);
        let first_remapped = remapped
            .forward(&[3], &mut kv)
            .expect("remapped decode step must succeed");
        let second_remapped = remapped
            .forward(&[7], &mut kv)
            .expect("remapped decode step must succeed");

        assert_eq!(
            first_remapped, first,
            "a forwarding decorator must not perturb the logits"
        );
        assert_eq!(second_remapped, second, "same on the second decode step");
        assert!(
            calls.load(Ordering::Relaxed) >= 2 * (7 * LAYERS + 1),
            "each of the {} sites must be driven once per decode step, got {} calls",
            7 * LAYERS + 1,
            calls.load(Ordering::Relaxed)
        );
    }
}
