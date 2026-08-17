//! Qwen3 transformer forward pass implementation.
//!
//! Structurally identical to LLaMA with optional attention bias.
//! Reuses common layers (RmsNorm, RoPE, SwiGLU, QuantLinear).

use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::qwen3::batch::{BatchScratch, MIN_BATCH_TOKENS};
use crate::qwen3::embedding::TokenEmbedding;
use crate::traits::{remap_kernel_slot, ForwardPass, KvCacheAccess, QuantKernelRemap};
use oxillama_quant::{
    quantize_activations_q8_0_into, KernelDispatcher, QuantKernel, QuantTensor,
    Q8_0_ACT_BLOCK_BYTES,
};
use std::sync::Arc;

/// A single Qwen3 transformer layer.
pub struct Qwen3Layer {
    pub attn_norm: RmsNorm,
    pub attn_q: QuantLinear,
    pub attn_k: QuantLinear,
    pub attn_v: QuantLinear,
    pub attn_output: QuantLinear,
    /// Per-head RMSNorm applied to Q before RoPE (Qwen3's QK-norm).
    ///
    /// `None` for checkpoints that omit `blk.{i}.attn_q_norm.weight`.
    pub attn_q_norm: Option<RmsNorm>,
    /// Per-head RMSNorm applied to K before RoPE (Qwen3's QK-norm).
    pub attn_k_norm: Option<RmsNorm>,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: QuantLinear,
    pub ffn_up: QuantLinear,
    pub ffn_down: QuantLinear,

    // Resolved kernels — one per projection, looked up from the tensor's
    // quantization type exactly once, in `load_qwen3_from_gguf`.
    //
    // The decode loop used to call `Qwen3Model::kernel_for` (a fresh
    // `dispatcher.get_kernel()` dispatch, allocating a new `Box<dyn
    // QuantKernel>`) once per projection per token: 7 projections x 36
    // layers = 252 allocations per decoded token, plus one more for the LM
    // head (see `Qwen3Model::output_kernel`).  Storing the resolved kernel
    // as an `Arc` on the layer that owns the weight turns every one of
    // those into a field load + a cheap `Arc` deref, with no allocation and
    // no dispatch logic re-run on the hot path.
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    pub ffn_gate_kernel: Arc<dyn QuantKernel>,
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

/// Complete Qwen3 model.
pub struct Qwen3Model {
    pub config: ModelConfig,
    /// Token embedding table.
    ///
    /// Held in whatever form the checkpoint allows — quantized and looked up
    /// one row at a time when the GGUF's rows are block-aligned (the normal
    /// case), which is what keeps Qwen3-4B's 151 936 × 2560 table off the heap
    /// as 1.556 GB of f32.  See [`TokenEmbedding`].
    pub token_embd: TokenEmbedding,
    pub layers: Vec<Qwen3Layer>,
    pub output_norm: RmsNorm,
    pub output: QuantLinear,
    /// Resolved kernel for `output` (the LM head), looked up once from its
    /// tensor type in [`Self::with_embedding`] instead of on every call to
    /// [`ForwardPass::forward`].  See the comment on [`Qwen3Layer`]'s kernel
    /// fields for why this matters on the decode path.
    output_kernel: Arc<dyn QuantKernel>,
    pub rope: RopeTable,
    pub dispatcher: KernelDispatcher,

    // Scratch buffers
    pub(crate) buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    /// Output of `attn_output`'s projection, added into `buf_hidden` as the
    /// attention block's residual.  Preallocated to `hidden_size` here
    /// instead of `vec![0.0; hidden_size]`-ed inside `attention()` on every
    /// call: that was a fresh heap allocation per layer per token (36x per
    /// token on Qwen3-4B).
    buf_proj_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
    /// Q8_0 image of whichever activation vector the next matmul consumes.
    ///
    /// Every projection whose kernel advertises a fused path
    /// ([`QuantLinear::q8_fused_blocks`]) reads its activations from here
    /// instead of re-quantizing them per row.  One buffer suffices because the
    /// projections that share an input (`attn_q`/`attn_k`/`attn_v`, then
    /// `ffn_gate`/`ffn_up`) are issued back to back, so the vector is
    /// quantized once and consumed before the next one overwrites it.
    buf_acts_q8: Vec<u8>,
    /// Scratch for the batched prefill path (`super::batch`).  Empty until the
    /// first multi-token `forward`/`embed`; decode never touches it.
    pub(crate) batch: BatchScratch,
}

impl Qwen3Model {
    /// Create a new Qwen3Model from preloaded weights.
    ///
    /// Takes the embedding table already dequantized.  Loaders that can keep
    /// it quantized — [`load_qwen3_from_gguf`] does — should call
    /// [`Self::with_embedding`] instead and hand over a
    /// [`TokenEmbedding::Quantized`].
    ///
    /// Fails if `output`'s tensor type has no registered [`QuantKernel`] —
    /// the kernel is resolved once here rather than on every forward pass.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<Qwen3Layer>,
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

    /// Create a new Qwen3Model over an arbitrary [`TokenEmbedding`].
    ///
    /// Fails if `output`'s tensor type has no registered [`QuantKernel`].
    pub fn with_embedding(
        config: ModelConfig,
        token_embd: TokenEmbedding,
        layers: Vec<Qwen3Layer>,
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
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            // The concatenated attention heads feed `attn_output`, whose input
            // width is `num_heads * head_dim`.  That is NOT always
            // `hidden_size`: Qwen3-4B attends over 32 × 128 = 4096 while its
            // residual stream is only 2560 wide.
            buf_attn_out: vec![0.0; num_heads * head_dim],
            buf_proj_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
            // Widest activation any projection consumes, rounded up to whole
            // Q4_K/Q6_K blocks (256 weights → 8 Q8_0 blocks).  Pre-sizing here
            // keeps the very first forward pass allocation-free too.
            buf_acts_q8: Vec::with_capacity(
                hidden_size
                    .max(num_heads * head_dim)
                    .max(intermediate_size)
                    .div_ceil(256)
                    * 8
                    * Q8_0_ACT_BLOCK_BYTES,
            ),
            batch: BatchScratch::default(),
        })
    }

    /// Dispatch a fresh kernel for `linear`'s tensor type.
    ///
    /// This allocates (`dispatcher.get_kernel` returns a `Box`) and is used
    /// only by the batched prefill path (`super::batch`), which calls it a
    /// handful of times per *tile* of up to [`super::batch::PREFILL_TILE`]
    /// tokens — amortized, not hot.  The single-token decode loop in this
    /// file does **not** call this: `Qwen3Layer`'s `*_kernel` fields and
    /// `Qwen3Model::output_kernel` hold the same lookup resolved once at
    /// load time instead.
    pub(crate) fn kernel_for(
        &self,
        linear: &QuantLinear,
    ) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Run every token of `tokens` through all layers, leaving the last
    /// token's pre-output-norm hidden state in `self.buf_hidden`.
    ///
    /// Multi-token calls (prompt prefill) go through the batched path in
    /// [`super::batch`], which processes [`super::batch::PREFILL_TILE`] tokens
    /// per pass over the weights; single-token calls (decode) and models the
    /// batched path cannot serve — a kernel with no fused Q8_0 route, or a
    /// LoRA-patched layer — replay the per-token loop.  The two produce
    /// bit-identical hidden states; see the module docs of [`super::batch`].
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        if tokens.len() >= MIN_BATCH_TOKENS && self.batched_prefill_supported() {
            return self.forward_prefill_batched(tokens, kv_cache);
        }

        let start_pos = kv_cache.seq_len();
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
    /// Dequantizes exactly one row (see [`TokenEmbedding`]); the destructuring
    /// borrow keeps `token_embd`/`dispatcher` and `buf_hidden` disjoint.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let Self {
            token_embd,
            dispatcher,
            buf_hidden,
            ..
        } = self;
        token_embd.row_into(dispatcher, token, buf_hidden)
    }

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

        // Resolved once at load time (`load_qwen3_from_gguf`) — see the
        // comment on `Qwen3Layer`'s kernel fields.  No dispatch, no
        // allocation on the decode path.
        let q_kernel: &dyn QuantKernel = &*layer.attn_q_kernel;
        let k_kernel: &dyn QuantKernel = &*layer.attn_k_kernel;
        let v_kernel: &dyn QuantKernel = &*layer.attn_v_kernel;

        // Q/K/V all read the same post-attn-norm activation vector, so it is
        // quantized to Q8_0 exactly once here and shared by all three GEMVs.
        // The block count is the max over the three so that a mixed-precision
        // checkpoint (e.g. Q4_K q/k with a Q6_K v) still gets a buffer long
        // enough for every kernel that opted in.
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

        // QK-norm then RoPE.  Qwen3 RMS-normalises every Q and K head
        // (over `head_dim`, with its own learned scale) before the rotation;
        // skipping it desynchronises the query/key magnitudes and the logits
        // degenerate into noise.
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            if let Some(ref q_norm) = layer.attn_q_norm {
                q_norm.forward(q_head);
            }
            self.rope.apply(q_head, position);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            if let Some(ref k_norm) = layer.attn_k_norm {
                k_norm.forward(k_head);
            }
            self.rope.apply(k_head, position);
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_vec[d];
                }
                self.buf_attn_scores[pos] = score * scale;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in 0..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        let layer = &self.layers[layer_idx];
        let o_kernel: &dyn QuantKernel = &*layer.attn_output_kernel;
        let o_fused = layer.attn_output.q8_fused_blocks(o_kernel);
        if let Some(n_blocks) = o_fused {
            quantize_activations_q8_0_into(&self.buf_attn_out, n_blocks, &mut self.buf_acts_q8);
        }
        // `buf_proj_out` is preallocated engine state (see its field doc) —
        // no more `vec![0.0f32; hidden_size]` per layer per token here.
        if o_fused.is_some() {
            layer.attn_output.forward_q8_fused(
                o_kernel,
                &self.buf_attn_out,
                &self.buf_acts_q8,
                &mut self.buf_proj_out,
            )?;
        } else {
            layer
                .attn_output
                .forward(o_kernel, &self.buf_attn_out, &mut self.buf_proj_out)?;
        }

        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj_out.iter()) {
            *h += p;
        }

        Ok(())
    }

    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        // Resolved once at load time — see the comment on `Qwen3Layer`'s
        // kernel fields.
        let gate_kernel: &dyn QuantKernel = &*layer.ffn_gate_kernel;
        let up_kernel: &dyn QuantKernel = &*layer.ffn_up_kernel;
        let down_kernel: &dyn QuantKernel = &*layer.ffn_down_kernel;

        // `ffn_gate` and `ffn_up` share the post-ffn-norm activation: quantize
        // it once for both.
        let gate_fused = layer.ffn_gate.q8_fused_blocks(gate_kernel);
        let up_fused = layer.ffn_up.q8_fused_blocks(up_kernel);
        if let Some(n_blocks) = gate_fused.iter().chain(&up_fused).copied().max() {
            quantize_activations_q8_0_into(&self.buf_norm, n_blocks, &mut self.buf_acts_q8);
        }

        if gate_fused.is_some() {
            layer.ffn_gate.forward_q8_fused(
                gate_kernel,
                &self.buf_norm,
                &self.buf_acts_q8,
                &mut self.buf_gate,
            )?;
        } else {
            layer
                .ffn_gate
                .forward(gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        }
        if up_fused.is_some() {
            layer.ffn_up.forward_q8_fused(
                up_kernel,
                &self.buf_norm,
                &self.buf_acts_q8,
                &mut self.buf_up,
            )?;
        } else {
            layer
                .ffn_up
                .forward(up_kernel, &self.buf_norm, &mut self.buf_up)?;
        }

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        match layer.ffn_down.q8_fused_blocks(down_kernel) {
            Some(n_blocks) => {
                quantize_activations_q8_0_into(&self.buf_gate, n_blocks, &mut self.buf_acts_q8);
                layer.ffn_down.forward_q8_fused(
                    down_kernel,
                    &self.buf_gate,
                    &self.buf_acts_q8,
                    &mut self.buf_ffn_out,
                )?;
            }
            None => layer
                .ffn_down
                .forward(down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?,
        }

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }
}

impl ForwardPass for Qwen3Model {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        self.output_norm.forward(&mut self.buf_hidden);

        // `buf_logits` may have been handed to the caller by ownership on a
        // previous call (see the `mem::take` below) and therefore be empty;
        // restore its length before the kernel writes into it.  A no-op on
        // every call except the one right after a `mem::take`.
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        // Resolved once at load time (`Qwen3Model::with_embedding`) instead
        // of dispatched here — see the comment on the `output_kernel` field.
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

        // Hand the freshly computed logits to the caller by ownership
        // transfer instead of `self.buf_logits.clone()`.  The clone was a
        // 608 KB (Qwen3-4B vocab_size == 151936) allocation-and-memcpy on
        // every decoded token; `mem::take` moves the `Vec`'s (ptr, len, cap)
        // for free and leaves `buf_logits` empty, which the resize above
        // repairs on the next call.
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of the LM-head projection (output.weight) that maps
    /// hidden_size → vocab_size. Returns a `hidden_size`-dimensional vector
    /// suitable for L2-normalised semantic embeddings.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        // Final norm on the last token's hidden state.
        // Does NOT project through the LM head — returns hidden state directly.
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

    /// Visit every kernel the per-token path dispatches through: each layer's
    /// four attention projections, its three FFN projections, then the LM head
    /// as `(None, "output")`.
    ///
    /// Qwen3 is dense throughout, so every stored binding is offered.  Two
    /// kernel *uses* still escape the remap because they re-dispatch from
    /// [`Self::dispatcher`] instead of reading a stored `Arc`: the
    /// token-embedding row lookup, and the tiled prefill in `super::batch`
    /// via `Self::kernel_for`.  A remapped kernel therefore governs decode
    /// but not a multi-token `forward` that qualifies for the batched path —
    /// see the trait contract.
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
            remap_kernel_slot(
                f,
                idx,
                "ffn_gate",
                &layer.ffn_gate.weight,
                &mut layer.ffn_gate_kernel,
            );
            remap_kernel_slot(
                f,
                idx,
                "ffn_up",
                &layer.ffn_up.weight,
                &mut layer.ffn_up_kernel,
            );
            remap_kernel_slot(
                f,
                idx,
                "ffn_down",
                &layer.ffn_down.weight,
                &mut layer.ffn_down_kernel,
            );
        }

        remap_kernel_slot(
            f,
            None,
            "output",
            &self.output.weight,
            &mut self.output_kernel,
        );
    }
}

pub(crate) fn softmax_inplace(x: &mut [f32]) {
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

/// Load a Qwen3 model from a `GgufModel`.
pub fn load_qwen3_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<Qwen3Model> {
    let dispatcher = KernelDispatcher::new();

    let token_embd = load_token_embedding(model, config, &dispatcher)?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_q.bias"),
        )?;
        let attn_k = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_k.weight"),
            &format!("{prefix}.attn_k.bias"),
        )?;
        let attn_v = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_v.weight"),
            &format!("{prefix}.attn_v.bias"),
        )?;
        let attn_output = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_output.weight"),
            &format!("{prefix}.attn_output.bias"),
        )?;

        let attn_q_norm = load_optional_head_norm(
            model,
            &format!("{prefix}.attn_q_norm.weight"),
            config.rms_norm_eps,
        )?;
        let attn_k_norm = load_optional_head_norm(
            model,
            &format!("{prefix}.attn_k_norm.weight"),
            config.rms_norm_eps,
        )?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        // Resolve every projection's kernel once, here, instead of on every
        // token of every generation this model ever serves — see the
        // comment on `Qwen3Layer`'s kernel fields.
        let attn_q_kernel = resolve_kernel(&dispatcher, &attn_q)?;
        let attn_k_kernel = resolve_kernel(&dispatcher, &attn_k)?;
        let attn_v_kernel = resolve_kernel(&dispatcher, &attn_v)?;
        let attn_output_kernel = resolve_kernel(&dispatcher, &attn_output)?;
        let ffn_gate_kernel = resolve_kernel(&dispatcher, &ffn_gate)?;
        let ffn_up_kernel = resolve_kernel(&dispatcher, &ffn_up)?;
        let ffn_down_kernel = resolve_kernel(&dispatcher, &ffn_down)?;

        layers.push(Qwen3Layer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q,
            attn_q_kernel,
            attn_k,
            attn_k_kernel,
            attn_v,
            attn_v_kernel,
            attn_output,
            attn_output_kernel,
            attn_q_norm,
            attn_k_norm,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_gate,
            ffn_gate_kernel,
            ffn_up,
            ffn_up_kernel,
            ffn_down,
            ffn_down_kernel,
        });
    }

    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);
    let output = load_lm_head(model)?;

    Qwen3Model::with_embedding(config.clone(), token_embd, layers, output_norm, output)
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
///
/// `dispatcher.get_kernel` allocates a `Box<dyn QuantKernel>`; `.into()`
/// converts it to an `Arc` (`impl From<Box<T>> for Arc<T>`, no extra copy)
/// so the same lookup can be stored on the layer and cloned for free instead
/// of being re-run — and re-allocated — on every forward pass.
fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Load `token_embd.weight`, keeping it quantized whenever possible.
///
/// Materialising this tensor as f32 is what made loading Qwen3-4B `Q4_K_M`
/// cost 1.556 GB on top of the file: `[151936, 2560]` f32 is bigger than the
/// entire 2.38 GB checkpoint's worth of Q4_K blocks.  A forward pass reads one
/// row per token, so the table is kept in its GGUF form (an mmap view — see
/// [`load_quant_linear`]) and rows are dequantized on lookup.
///
/// The bulk-dequantized [`TokenEmbedding::Dense`] remains the fallback for
/// checkpoints whose rows are not a whole number of quantization blocks, so
/// nothing becomes unloadable.
///
/// This does **not** touch the LM head: [`load_lm_head`] loads the same tensor
/// separately as a [`QuantLinear`] and the tied head keeps running through the
/// quantized fused GEMV path.  Both share one payload, so tying is still free.
fn load_token_embedding(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
    dispatcher: &KernelDispatcher,
) -> ArchResult<TokenEmbedding> {
    const NAME: &str = "token_embd.weight";

    let info = model
        .file
        .tensors
        .get(NAME)
        .map_err(|_| ArchError::MissingTensor {
            name: NAME.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    let n_elements = info.n_elements() as usize;
    let data = model.tensor_bytes(NAME)?;

    if let Some(embd) =
        TokenEmbedding::quantized(QuantTensor::from_shared(data.clone(), shape, tensor_type))
    {
        return Ok(embd);
    }

    // Fallback: rows are not block-aligned (or the shape is not 2-D), so the
    // per-row decode cannot address them.  Dequantize the whole table.
    let hidden = if config.hidden_size == 0 {
        n_elements
    } else {
        config.hidden_size
    };
    let dense = dequant_to_f32_slice(tensor_type, &data, n_elements, dispatcher)?;
    Ok(TokenEmbedding::dense(dense, hidden))
}

/// Load an optional per-head RMSNorm weight (Qwen3's QK-norm).
///
/// Returns `Ok(None)` when the tensor is absent, which keeps pre-QK-norm
/// checkpoints — and the synthetic fixtures — loadable.
fn load_optional_head_norm(
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

/// Load the LM head, falling back to the tied input embedding.
///
/// Qwen3 checkpoints at the 4B size class and below tie the input and output
/// embeddings: the GGUF ships `token_embd.weight` but no standalone
/// `output.weight`.  When `output.weight` is absent the embedding matrix is
/// reused as the LM head, mirroring llama.cpp's fallback.  Both tensors carry
/// the same dimensions, and the reuse goes through [`load_quant_linear`], so
/// the head stays quantized (no dequantization on the matmul path).
///
/// When neither tensor is present the error still names `output.weight`.
fn load_lm_head(model: &oxillama_gguf::GgufModel) -> ArchResult<QuantLinear> {
    if !model.file.tensors.contains("output.weight")
        && model.file.tensors.contains("token_embd.weight")
    {
        return load_quant_linear(model, "token_embd.weight");
    }
    load_quant_linear(model, "output.weight")
}

/// Load a quantized linear layer from GGUF.
///
/// The weight payload is taken as a [`SharedBytes`][oxillama_gguf::SharedBytes]
/// view, not a `to_vec()` copy: the GEMV kernels only ever read `&[u8]` out of
/// it, so a copy would double the resident cost of the checkpoint for nothing.
/// With `load_mmap` behind it, the 2.3 GB of Qwen3-4B `Q4_K_M` projection
/// weights stay clean file-backed pages the kernel can evict and refault
/// instead of anonymous RSS.  Alignment is unaffected — GGUF tensor offsets
/// are 32-byte multiples from a page-aligned base, i.e. strictly better than
/// the 1-byte guarantee a `Vec<u8>` gave.
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
    let data = model.tensor_bytes(name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);
    Ok(QuantLinear::new(tensor, None))
}

/// Load a quantized linear layer with optional bias from GGUF.
fn load_quant_linear_with_bias(
    model: &oxillama_gguf::GgufModel,
    weight_name: &str,
    bias_name: &str,
) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(weight_name)
        .map_err(|_| ArchError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    // Shared view, not a copy — see `load_quant_linear`.
    let data = model.tensor_bytes(weight_name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);

    // Load bias if present (bias tensors are always F32)
    let bias = if model.file.tensors.contains(bias_name) {
        let bias_data = model.tensor_data(bias_name)?;
        let bias_info = model.file.tensors.get(bias_name)?;
        let n = bias_info.n_elements() as usize;
        let mut bias_vec = vec![0.0f32; n];
        for (i, chunk) in bias_data.chunks_exact(4).enumerate().take(n) {
            bias_vec[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Some(bias_vec)
    } else {
        None
    };

    Ok(QuantLinear::new(tensor, bias))
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

/// Dequantize tensor data to f32.
fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    dequant_to_f32_slice(
        info.tensor_type,
        data,
        info.n_elements() as usize,
        dispatcher,
    )
}

/// Dequantize `n_elements` weights of `tensor_type` out of `data`.
fn dequant_to_f32_slice(
    tensor_type: oxillama_gguf::GgufTensorType,
    data: &[u8],
    n_elements: usize,
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::remap_test_support::CountingKernel;
    use oxillama_gguf::{GgufModel, GgufTensorType, GgufWriter, MetadataValue};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Tiny but structurally valid dimensions.  `VOCAB != HIDDEN` on purpose so
    // that a transposed or mis-sourced LM head cannot pass the shape checks,
    // and `HEADS * HEAD_DIM != HIDDEN` so the fixture reproduces the Qwen3-4B
    // geometry in miniature: 32 × 128-wide heads over a 2560-wide residual
    // stream, where `hidden_size / head_count` would give the wrong head width.
    const HIDDEN: usize = 32;
    const VOCAB: usize = 48;
    const FFN: usize = 64;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 24;
    /// Concatenated attention width — deliberately different from `HIDDEN`.
    const ATTN_DIM: usize = HEADS * HEAD_DIM;

    /// Q8_0 payload for a `[VOCAB, HIDDEN]` embedding matrix.
    ///
    /// `HIDDEN` equals the Q8_0 block size, so each row is exactly one block:
    /// an f16 scale followed by 32 int8 weights.
    fn q8_0_embedding_bytes() -> Vec<u8> {
        let mut out = Vec::with_capacity(VOCAB * (2 + HIDDEN));
        for row in 0..VOCAB {
            out.extend_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            for col in 0..HIDDEN {
                out.push((((row + col) % 15) as i8 - 7) as u8);
            }
        }
        out
    }

    /// Build a 1-layer Qwen3 GGUF whose `token_embd.weight` is Q8_0.
    ///
    /// With `with_output = false` the fixture mimics a tied checkpoint: no
    /// standalone `output.weight`.  With `with_output = true` an explicit F32
    /// LM head is written, so the tensor type distinguishes the two paths.
    /// `with_qk_norm` toggles the `attn_q_norm`/`attn_k_norm` pair that real
    /// Qwen3 checkpoints ship.
    fn build_tiny_qwen3_gguf(embd: &[u8], with_output: bool, with_qk_norm: bool) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        for (key, value) in [
            (
                "general.architecture",
                MetadataValue::String("qwen3".to_string()),
            ),
            (
                "qwen3.embedding_length",
                MetadataValue::Uint32(HIDDEN as u32),
            ),
            (
                "qwen3.feed_forward_length",
                MetadataValue::Uint32(FFN as u32),
            ),
            ("qwen3.block_count", MetadataValue::Uint32(1)),
            (
                "qwen3.attention.head_count",
                MetadataValue::Uint32(HEADS as u32),
            ),
            (
                "qwen3.attention.head_count_kv",
                MetadataValue::Uint32(HEADS as u32),
            ),
            (
                "qwen3.attention.key_length",
                MetadataValue::Uint32(HEAD_DIM as u32),
            ),
            (
                "qwen3.attention.value_length",
                MetadataValue::Uint32(HEAD_DIM as u32),
            ),
            ("qwen3.context_length", MetadataValue::Uint32(128)),
            ("qwen3.vocab_size", MetadataValue::Uint32(VOCAB as u32)),
            ("qwen3.rope.freq_base", MetadataValue::Float32(10000.0)),
        ] {
            writer.add_metadata(key, value);
        }

        // GGUF writes `ne` fastest-changing-first, so every weight is declared
        // as [in_features, out_features] — the reverse of the math shape.
        writer.add_tensor(
            "token_embd.weight",
            &[HIDDEN as u64, VOCAB as u64],
            GgufTensorType::Q8_0,
            embd,
        );

        let f32_tensors: &[(&str, [usize; 2])] = &[
            ("blk.0.attn_q.weight", [HIDDEN, ATTN_DIM]),
            ("blk.0.attn_k.weight", [HIDDEN, ATTN_DIM]),
            ("blk.0.attn_v.weight", [HIDDEN, ATTN_DIM]),
            ("blk.0.attn_output.weight", [ATTN_DIM, HIDDEN]),
            ("blk.0.ffn_gate.weight", [HIDDEN, FFN]),
            ("blk.0.ffn_up.weight", [HIDDEN, FFN]),
            ("blk.0.ffn_down.weight", [FFN, HIDDEN]),
        ];
        for (name, [in_features, out_features]) in f32_tensors {
            writer.add_tensor(
                name,
                &[*in_features as u64, *out_features as u64],
                GgufTensorType::F32,
                &vec![0u8; in_features * out_features * 4],
            );
        }

        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        ] {
            writer.add_tensor(
                name,
                &[HIDDEN as u64],
                GgufTensorType::F32,
                &[0u8; HIDDEN * 4],
            );
        }

        if with_qk_norm {
            // QK-norm scales are per-head, so they are HEAD_DIM wide — not
            // HIDDEN wide like the block norms above.
            for name in ["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
                writer.add_tensor(
                    name,
                    &[HEAD_DIM as u64],
                    GgufTensorType::F32,
                    &[0u8; HEAD_DIM * 4],
                );
            }
        }

        if with_output {
            writer.add_tensor(
                "output.weight",
                &[HIDDEN as u64, VOCAB as u64],
                GgufTensorType::F32,
                &vec![0u8; VOCAB * HIDDEN * 4],
            );
        }

        let mut bytes = Vec::new();
        writer
            .write_to(&mut bytes)
            .expect("synthetic Qwen3 GGUF must serialize");
        bytes
    }

    /// Load a fixture into a `Qwen3Model`.
    fn load_fixture(bytes: Vec<u8>) -> (GgufModel, Qwen3Model) {
        let gguf = GgufModel::from_bytes(bytes).expect("synthetic Qwen3 GGUF must parse");
        let config =
            ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
        let model =
            load_qwen3_from_gguf(&gguf, &config).expect("load_qwen3_from_gguf must succeed");
        (gguf, model)
    }

    /// Single-layer KV cache backing the forward-pass test.
    struct TestKv {
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_len: usize,
    }

    impl KvCacheAccess for TestKv {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(&mut self, _layer: usize, k: &[f32], v: &[f32]) -> ArchResult<()> {
            self.keys.extend_from_slice(k);
            self.values.extend_from_slice(v);
            Ok(())
        }
        fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys)
        }
        fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.values)
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
    }

    /// A checkpoint without `output.weight` loads by tying the LM head to
    /// `token_embd.weight`, keeping the quantized bytes untouched.
    #[test]
    fn qwen3_tied_embedding_lm_head_fallback() {
        let embd = q8_0_embedding_bytes();
        let (gguf, model) = load_fixture(build_tiny_qwen3_gguf(&embd, false, true));

        assert!(
            !gguf.file.tensors.contains("output.weight"),
            "fixture must model a tied checkpoint (no standalone output.weight)"
        );
        assert_eq!(
            model.output.weight.shape,
            vec![VOCAB, HIDDEN],
            "tied LM head must keep token_embd's dimensions"
        );
        assert_eq!(
            model.output.weight.tensor_type,
            GgufTensorType::Q8_0,
            "tied LM head must stay quantized (no dequantization)"
        );
        assert_eq!(
            model.output.weight.data, embd,
            "tied LM head must reuse the token_embd bytes verbatim"
        );
    }

    /// The tied LM head is usable: a forward pass yields `vocab_size` logits.
    #[test]
    fn qwen3_tied_embedding_forward_produces_vocab_logits() {
        let embd = q8_0_embedding_bytes();
        let (_gguf, mut model) = load_fixture(build_tiny_qwen3_gguf(&embd, false, true));

        let mut kv = TestKv {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
        };
        let logits = model
            .forward(&[1u32, 2], &mut kv)
            .expect("forward through the tied LM head must succeed");

        assert_eq!(logits.len(), VOCAB, "logits length must equal vocab_size");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit {i} must be finite, got {v}");
        }
    }

    /// The head width comes from `attention.key_length`, not from a division.
    ///
    /// Qwen3 decouples the two: `hidden_size / head_count` is 16 here but the
    /// checkpoint attends over 24-wide heads.  Deriving the width by division
    /// under-sizes every Q/K/V scratch buffer and the projections reject them.
    #[test]
    fn qwen3_head_dim_follows_key_length_not_hidden_over_heads() {
        let embd = q8_0_embedding_bytes();
        let (_gguf, model) = load_fixture(build_tiny_qwen3_gguf(&embd, false, true));

        assert_ne!(
            HEAD_DIM,
            HIDDEN / HEADS,
            "the fixture must decouple head_dim from hidden_size / head_count"
        );
        assert_eq!(
            model.config.head_dim, HEAD_DIM,
            "head_dim must be read from qwen3.attention.key_length"
        );
        assert_eq!(
            model.layers[0].attn_q.out_features, ATTN_DIM,
            "attn_q must project hidden_size → head_count × head_dim"
        );
        assert_eq!(
            model.layers[0].attn_q.in_features, HIDDEN,
            "attn_q must consume the residual stream, not the attention width"
        );
        assert_eq!(
            model.layers[0].attn_output.in_features, ATTN_DIM,
            "attn_output must consume the concatenated heads"
        );
    }

    /// `attn_q_norm`/`attn_k_norm` are picked up when the checkpoint has them.
    #[test]
    fn qwen3_qk_norm_is_loaded_when_present() {
        let embd = q8_0_embedding_bytes();
        let (_gguf, model) = load_fixture(build_tiny_qwen3_gguf(&embd, false, true));

        let layer = &model.layers[0];
        let q_norm = layer
            .attn_q_norm
            .as_ref()
            .expect("attn_q_norm must be loaded when the tensor is present");
        let k_norm = layer
            .attn_k_norm
            .as_ref()
            .expect("attn_k_norm must be loaded when the tensor is present");
        assert_eq!(
            q_norm.weight.len(),
            HEAD_DIM,
            "QK-norm scales are per-head, so they are head_dim wide"
        );
        assert_eq!(k_norm.weight.len(), HEAD_DIM, "same for the key norm");
    }

    /// A checkpoint without QK-norm tensors still loads, with the norms unset.
    #[test]
    fn qwen3_qk_norm_absent_leaves_the_layer_unnormalised() {
        let embd = q8_0_embedding_bytes();
        let (gguf, model) = load_fixture(build_tiny_qwen3_gguf(&embd, false, false));

        assert!(
            !gguf.file.tensors.contains("blk.0.attn_q_norm.weight"),
            "fixture must omit the QK-norm tensors"
        );
        assert!(
            model.layers[0].attn_q_norm.is_none(),
            "a missing attn_q_norm must not be invented"
        );
        assert!(
            model.layers[0].attn_k_norm.is_none(),
            "a missing attn_k_norm must not be invented"
        );
    }

    /// An explicit `output.weight` still wins over the tied fallback.
    #[test]
    fn qwen3_explicit_output_weight_takes_precedence() {
        let embd = q8_0_embedding_bytes();
        let (_gguf, model) = load_fixture(build_tiny_qwen3_gguf(&embd, true, true));

        assert_eq!(
            model.output.weight.tensor_type,
            GgufTensorType::F32,
            "explicit F32 output.weight must be used, not the Q8_0 token_embd"
        );
        assert_eq!(
            model.output.weight.data.len(),
            VOCAB * HIDDEN * 4,
            "explicit LM head must carry the output.weight payload"
        );
    }

    /// With neither tensor present the error still names `output.weight`.
    #[test]
    fn qwen3_lm_head_missing_both_reports_output_weight() {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataValue::String("qwen3".to_string()),
        );
        writer.add_tensor(
            "output_norm.weight",
            &[HIDDEN as u64],
            GgufTensorType::F32,
            &[0u8; HIDDEN * 4],
        );
        let mut bytes = Vec::new();
        writer
            .write_to(&mut bytes)
            .expect("headless GGUF must serialize");
        let gguf = GgufModel::from_bytes(bytes).expect("headless GGUF must parse");

        match load_lm_head(&gguf) {
            Err(ArchError::MissingTensor { name }) => assert_eq!(
                name, "output.weight",
                "error must name output.weight, not the tied fallback"
            ),
            Err(other) => panic!("expected MissingTensor, got: {other}"),
            Ok(_) => panic!("loading an LM head from a GGUF with neither tensor must fail"),
        }
    }

    // ── Kernel remapping ────────────────────────────────────────────────────
    //
    // These build the model directly instead of through `build_tiny_qwen3_gguf`
    // because that fixture's weights are all zero: with zeroed norms and
    // projections every logit is zero and a "logits are unchanged" assertion
    // would hold no matter what the remap did.

    /// Number of transformer blocks in the remap fixture.
    const REMAP_LAYERS: usize = 2;
    /// Experts are irrelevant here; the FFN width for the remap fixture.
    const REMAP_FFN: usize = 16;

    /// An F32 `[out_features, in_features]` weight with deterministic,
    /// non-degenerate values.
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

    /// A two-layer Qwen3 over F32 weights, with QK-norm attached.
    fn tiny_qwen3() -> Qwen3Model {
        let dispatcher = KernelDispatcher::new();
        let config = ModelConfig {
            architecture: "qwen3".to_string(),
            hidden_size: HIDDEN,
            intermediate_size: REMAP_FFN,
            num_layers: REMAP_LAYERS,
            num_attention_heads: HEADS,
            num_kv_heads: HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 16,
            ..ModelConfig::default()
        };

        let layers = (0..REMAP_LAYERS)
            .map(|l| Qwen3Layer {
                attn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
                attn_q: f32_linear(ATTN_DIM, HIDDEN, l * 7 + 1),
                attn_k: f32_linear(ATTN_DIM, HIDDEN, l * 7 + 2),
                attn_v: f32_linear(ATTN_DIM, HIDDEN, l * 7 + 3),
                attn_output: f32_linear(HIDDEN, ATTN_DIM, l * 7 + 4),
                attn_q_norm: Some(RmsNorm::new(vec![1.0; HEAD_DIM], 1e-5)),
                attn_k_norm: Some(RmsNorm::new(vec![1.0; HEAD_DIM], 1e-5)),
                ffn_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
                ffn_gate: f32_linear(REMAP_FFN, HIDDEN, l * 7 + 5),
                ffn_up: f32_linear(REMAP_FFN, HIDDEN, l * 7 + 6),
                ffn_down: f32_linear(HIDDEN, REMAP_FFN, l * 7 + 7),
                attn_q_kernel: f32_kernel(&dispatcher),
                attn_k_kernel: f32_kernel(&dispatcher),
                attn_v_kernel: f32_kernel(&dispatcher),
                attn_output_kernel: f32_kernel(&dispatcher),
                ffn_gate_kernel: f32_kernel(&dispatcher),
                ffn_up_kernel: f32_kernel(&dispatcher),
                ffn_down_kernel: f32_kernel(&dispatcher),
            })
            .collect();

        let token_embd: Vec<f32> = (0..VOCAB * HIDDEN)
            .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
            .collect();
        Qwen3Model::new(
            config,
            token_embd,
            layers,
            RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            f32_linear(VOCAB, HIDDEN, 99),
        )
        .expect("the F32 LM head must resolve a kernel")
    }

    /// Per-layer KV cache for the multi-layer remap fixture.
    struct LayeredKv {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        seq_len: usize,
    }

    impl LayeredKv {
        fn new(layers: usize) -> Self {
            Self {
                keys: vec![Vec::new(); layers],
                values: vec![Vec::new(); layers],
                seq_len: 0,
            }
        }
    }

    impl KvCacheAccess for LayeredKv {
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

    /// Every projection the decode path reads is offered exactly once, in
    /// block order, with the weight tensor that projection owns.
    #[test]
    fn remap_visits_seven_sites_per_layer_plus_the_lm_head() {
        let mut model = tiny_qwen3();
        let mut sites: Vec<(Option<usize>, &'static str, Vec<usize>)> = Vec::new();
        model.remap_quant_kernels(&mut |site, kernel| {
            sites.push((site.layer, site.role, site.weight.shape.clone()));
            kernel
        });

        let mut expected: Vec<(Option<usize>, &'static str, Vec<usize>)> = Vec::new();
        for l in 0..REMAP_LAYERS {
            expected.push((Some(l), "attn_q", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_k", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_v", vec![ATTN_DIM, HIDDEN]));
            expected.push((Some(l), "attn_output", vec![HIDDEN, ATTN_DIM]));
            expected.push((Some(l), "ffn_gate", vec![REMAP_FFN, HIDDEN]));
            expected.push((Some(l), "ffn_up", vec![REMAP_FFN, HIDDEN]));
            expected.push((Some(l), "ffn_down", vec![HIDDEN, REMAP_FFN]));
        }
        expected.push((None, "output", vec![VOCAB, HIDDEN]));

        assert_eq!(
            sites.len(),
            7 * REMAP_LAYERS + 1,
            "Qwen3 is dense: 7 sites per layer plus the LM head"
        );
        assert_eq!(
            sites, expected,
            "roles, layer indices and weight shapes must match the contract exactly"
        );
    }

    /// Wrapping every kernel in a forwarding decorator changes no logit, and
    /// the decorator really is on the path that produced them.
    #[test]
    fn remapped_delegating_kernels_reproduce_the_baseline_logits() {
        // Single-token steps: a multi-token `forward` may take the tiled
        // prefill path, which re-dispatches kernels and would bypass the remap.
        let mut baseline = tiny_qwen3();
        let mut kv = LayeredKv::new(REMAP_LAYERS);
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

        let mut remapped = tiny_qwen3();
        let calls = Arc::new(AtomicUsize::new(0));
        remapped.remap_quant_kernels(&mut |_site, kernel| {
            CountingKernel::wrap(kernel, Arc::clone(&calls))
        });

        let mut kv = LayeredKv::new(REMAP_LAYERS);
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
            calls.load(Ordering::Relaxed) >= 2 * (7 * REMAP_LAYERS + 1),
            "each of the {} sites must be driven once per decode step, got {} calls",
            7 * REMAP_LAYERS + 1,
            calls.load(Ordering::Relaxed)
        );
    }
}
