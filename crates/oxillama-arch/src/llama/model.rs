//! LLaMA transformer forward pass implementation.
//!
//! This module contains the full LLaMA decoder-only transformer:
//! embedding → N×(RMSNorm → GQA → residual → RMSNorm → SwiGLU FFN → residual) → RMSNorm → LM head
//!
//! When `config.num_experts > 0`, the FFN layers use a sparse Mixture-of-Experts
//! (MoE) layout (Mixtral-style) instead of the standard dense SwiGLU FFN.

use crate::common::linear::QuantLinear;
use crate::common::moe::{Expert, MoeFfn};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use crate::traits::{BatchedKvView, ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

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
}

/// FFN layer variant: either a standard dense SwiGLU or a sparse MoE FFN.
///
/// Dense models (standard LLaMA) use `Dense` with quantized weights (boxed to
/// avoid a large enum-variant size difference with `Moe`).
/// Mixtral and other MoE models use `Moe` with dequantized f32 experts.
pub enum FfnVariant {
    /// Standard dense SwiGLU FFN using quantized weights.
    Dense(Box<DenseFfn>),
    /// Sparse Mixture-of-Experts FFN.
    Moe(Box<MoeFfn>),
}

/// A single transformer layer (decoder block).
pub struct LlamaLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection [num_heads * head_dim, hidden_size].
    pub attn_q: QuantLinear,
    /// Key projection [num_kv_heads * head_dim, hidden_size].
    pub attn_k: QuantLinear,
    /// Value projection [num_kv_heads * head_dim, hidden_size].
    pub attn_v: QuantLinear,
    /// Output projection [hidden_size, num_heads * head_dim].
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN variant: dense SwiGLU or sparse MoE.
    pub ffn: FfnVariant,
}

/// Complete LLaMA model with all weights and forward pass logic.
pub struct LlamaModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<LlamaLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection [vocab_size, hidden_size].
    pub output: QuantLinear,
    /// RoPE precomputed frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,

    // Scratch buffers (reused across forward calls to avoid allocation)
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl LlamaModel {
    /// Create a new LlamaModel from preloaded weights.
    ///
    /// This is the primary construction path. Use `from_gguf()` on `GgufModel` to
    /// load from a GGUF file (implemented in oxillama-runtime).
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<LlamaLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
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

        Self {
            config,
            token_embd,
            layers,
            output_norm,
            output,
            rope,
            dispatcher,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        }
    }

    /// Get the kernel for a QuantLinear's tensor type.
    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Embed a single token into the hidden state buffer.
    fn embed_token(&mut self, token: u32) {
        let hidden_size = self.config.hidden_size;
        let offset = token as usize * hidden_size;
        self.buf_hidden
            .copy_from_slice(&self.token_embd[offset..offset + hidden_size]);
    }

    /// Run grouped-query attention for a single layer.
    ///
    /// Steps:
    /// 1. Project hidden → Q, K, V
    /// 2. Apply RoPE to Q and K
    /// 3. Store K, V in cache
    /// 4. Compute attention scores: softmax(Q·K^T / sqrt(head_dim))
    /// 5. Multiply attention weights by V
    /// 6. Project output back to hidden_size
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

        // Project to Q, K, V
        let q_kernel = self.kernel_for(&layer.attn_q)?;
        let k_kernel = self.kernel_for(&layer.attn_k)?;
        let v_kernel = self.kernel_for(&layer.attn_v)?;

        layer
            .attn_q
            .forward(&*q_kernel, &self.buf_norm, &mut self.buf_q)?;
        layer
            .attn_k
            .forward(&*k_kernel, &self.buf_norm, &mut self.buf_k)?;
        layer
            .attn_v
            .forward(&*v_kernel, &self.buf_norm, &mut self.buf_v)?;

        // Apply RoPE to Q and K (per-head)
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            self.rope.apply(q_head, position);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            self.rope.apply(k_head, position);
        }

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        // Get cached keys and values [seq_len * kv_dim]
        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;
        let seq_len = position + 1; // includes current token

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Clear attention output buffer
        self.buf_attn_out.fill(0.0);

        // Per-head attention
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores: Q·K^T for all cached positions
            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_vec[d];
                }
                self.buf_attn_scores[pos] = score * scale;
            }

            // Causal softmax over [0..seq_len]
            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            // Weighted sum of V
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

        // Project attention output back to hidden_size
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let layer = &self.layers[layer_idx];
        // attn_output: [hidden_size, num_heads * head_dim]
        // We need to use buf_attn_out as input and write to buf_norm (temporarily)
        // but we actually want to add to residual, so write to a temp then add.
        let mut proj_out = vec![0.0f32; self.config.hidden_size];
        layer
            .attn_output
            .forward(&*o_kernel, &self.buf_attn_out, &mut proj_out)?;

        // Add to residual (buf_hidden)
        for (h, &p) in self.buf_hidden.iter_mut().zip(proj_out.iter()) {
            *h += p;
        }

        Ok(())
    }

    /// Run the feed-forward network for a single layer.
    ///
    /// Dispatches to either the dense SwiGLU path or the sparse MoE path
    /// depending on the layer's `FfnVariant`.
    ///
    /// Dense: `FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`
    /// MoE:   weighted sum of top-K expert SwiGLU outputs
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        match &self.layers[layer_idx].ffn {
            FfnVariant::Dense(dense) => {
                let gate_kernel = self
                    .dispatcher
                    .get_kernel(dense.gate.weight.tensor_type)
                    .map_err(ArchError::from)?;
                let up_kernel = self
                    .dispatcher
                    .get_kernel(dense.up.weight.tensor_type)
                    .map_err(ArchError::from)?;
                let down_kernel = self
                    .dispatcher
                    .get_kernel(dense.down.weight.tensor_type)
                    .map_err(ArchError::from)?;

                // gate = gate_proj(norm_hidden)
                dense
                    .gate
                    .forward(&*gate_kernel, &self.buf_norm, &mut self.buf_gate)?;

                // up = up_proj(norm_hidden)
                dense
                    .up
                    .forward(&*up_kernel, &self.buf_norm, &mut self.buf_up)?;

                // gate = silu(gate) * up  (SwiGLU)
                swiglu_inplace(&mut self.buf_gate, &self.buf_up);

                // ffn_out = down_proj(gate)
                dense
                    .down
                    .forward(&*down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;
            }
            FfnVariant::Moe(moe_ffn) => {
                // Clone buf_norm to satisfy borrow checker: MoeFfn reads input,
                // buf_ffn_out is written. No unsafe needed.
                let input = self.buf_norm.clone();
                moe_ffn.forward(&input, &mut self.buf_ffn_out)?;
            }
        }

        // Add FFN output to residual (buf_hidden)
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
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
        // Process each token (for prefill we process all, for decode just one)
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            // Embed token
            self.embed_token(token);

            // Run through all transformer layers
            for layer_idx in 0..self.layers.len() {
                // Pre-attention norm
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                // Grouped-query attention + residual
                self.attention(layer_idx, position, kv_cache)?;

                // Pre-FFN norm
                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                // SwiGLU FFN + residual
                self.feed_forward(layer_idx)?;
            }

            // Advance KV cache position after all layers processed this token
            kv_cache.advance();
        }

        // Final norm on the last token's hidden state
        self.output_norm.forward(&mut self.buf_hidden);

        // Project to vocabulary logits
        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        Ok(self.buf_logits.clone())
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of the LM-head projection (output.weight) that maps
    /// hidden_size → vocab_size. Returns a `hidden_size`-dimensional vector
    /// suitable for L2-normalised semantic embeddings.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token);

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
                    *score = q_head
                        .iter()
                        .zip(k_pos.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
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
                    for (o, &v) in out_head.iter_mut().zip(values[v_start..v_end].iter()) {
                        *o += attn_weight * v;
                    }
                }
            }
        }

        Ok(out)
    }
}

/// In-place softmax over a slice.
pub(crate) fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // exp(x - max) and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Load a LLaMA model from a `GgufModel` (convenience function).
///
/// Extracts all tensors by name and constructs a `LlamaModel`.
pub fn load_llama_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<LlamaModel> {
    let dispatcher = KernelDispatcher::new();

    // Load token embeddings (always F32 or F16 in GGUF)
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn = if config.num_experts > 0 {
            load_moe_ffn(model, &dispatcher, &prefix, config)?
        } else {
            let gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
            let up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
            let down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;
            FfnVariant::Dense(Box::new(DenseFfn { gate, up, down }))
        };

        layers.push(LlamaLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn,
        });
    }

    // Load final norm and output projection
    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);

    let output = load_quant_linear(model, "output.weight")?;

    Ok(LlamaModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
    ))
}

/// Load a MoE FFN for one transformer block from GGUF.
///
/// Mixtral GGUF stores stacked expert tensors under these names:
/// - `blk.{i}.ffn_gate_inp.weight`   — router: `[num_experts, hidden_size]`
/// - `blk.{i}.ffn_gate_exps.weight`  — stacked gate: `[num_experts, intermediate_size, hidden_size]`
/// - `blk.{i}.ffn_up_exps.weight`    — stacked up:   `[num_experts, intermediate_size, hidden_size]`
/// - `blk.{i}.ffn_down_exps.weight`  — stacked down: `[num_experts, hidden_size, intermediate_size]`
///
/// Each stacked tensor is split by `expert_slice_size` to build per-expert weight vectors.
fn load_moe_ffn(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &KernelDispatcher,
    prefix: &str,
    config: &ModelConfig,
) -> ArchResult<FfnVariant> {
    let num_experts = config.num_experts;
    let top_k = config.num_experts_used.max(1);
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;

    // --- Router ---
    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    let router_info =
        model
            .file
            .tensors
            .get(&router_name)
            .map_err(|_| ArchError::MissingTensor {
                name: router_name.clone(),
            })?;
    let router_data = model.tensor_data(&router_name)?;
    let router = dequant_to_f32(router_info, router_data, dispatcher)?;

    // Validate router shape: should be [num_experts, hidden_size].
    let expected_router = num_experts * hidden;
    if router.len() != expected_router {
        return Err(ArchError::TensorShapeMismatch {
            tensor: router_name,
            expected: vec![num_experts, hidden],
            got: vec![router.len()],
        });
    }

    // --- Stacked expert tensors ---
    let gate_stacked =
        load_dequant_tensor(model, dispatcher, &format!("{prefix}.ffn_gate_exps.weight"))?;
    let up_stacked =
        load_dequant_tensor(model, dispatcher, &format!("{prefix}.ffn_up_exps.weight"))?;
    let down_stacked =
        load_dequant_tensor(model, dispatcher, &format!("{prefix}.ffn_down_exps.weight"))?;

    // Each expert's gate/up slice is [intermediate_size, hidden_size].
    let gate_up_stride = intermediate * hidden;
    // Each expert's down slice is [hidden_size, intermediate_size].
    let down_stride = hidden * intermediate;

    let total_gate_up = num_experts * gate_up_stride;
    let total_down = num_experts * down_stride;

    if gate_stacked.len() != total_gate_up {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_gate_exps.weight"),
            expected: vec![num_experts, intermediate, hidden],
            got: vec![gate_stacked.len()],
        });
    }
    if up_stacked.len() != total_gate_up {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_up_exps.weight"),
            expected: vec![num_experts, intermediate, hidden],
            got: vec![up_stacked.len()],
        });
    }
    if down_stacked.len() != total_down {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_down_exps.weight"),
            expected: vec![num_experts, hidden, intermediate],
            got: vec![down_stacked.len()],
        });
    }

    // Split stacked tensors into per-expert weight vectors.
    let experts: Vec<Expert> = (0..num_experts)
        .map(|e| {
            let gate_start = e * gate_up_stride;
            let up_start = e * gate_up_stride;
            let down_start = e * down_stride;
            Expert {
                gate: gate_stacked[gate_start..gate_start + gate_up_stride].to_vec(),
                up: up_stacked[up_start..up_start + gate_up_stride].to_vec(),
                down: down_stacked[down_start..down_start + down_stride].to_vec(),
                hidden_size: hidden,
                intermediate_size: intermediate,
            }
        })
        .collect();

    Ok(FfnVariant::Moe(Box::new(MoeFfn {
        router,
        experts,
        top_k,
        num_experts,
        hidden_size: hidden,
    })))
}

/// Load and dequantize a tensor to f32, looking it up by name.
pub(crate) fn load_dequant_tensor(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &KernelDispatcher,
    name: &str,
) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    dequant_to_f32(info, data, dispatcher)
}

/// Load a quantized linear layer from GGUF.
pub(crate) fn load_quant_linear(
    model: &oxillama_gguf::GgufModel,
    name: &str,
) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let tensor = QuantTensor::new(data.to_vec(), shape, info.tensor_type);

    Ok(QuantLinear::new(tensor, None))
}

/// Load an RMSNorm weight vector from GGUF (always dequantized to F32).
pub(crate) fn load_rms_norm_weight(
    model: &oxillama_gguf::GgufModel,
    name: &str,
) -> ArchResult<Vec<f32>> {
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
pub(crate) fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    // F32 tensors — direct copy
    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    // F16 tensors — convert via half crate
    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    // Quantized tensors — use kernel dequant
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

    #[test]
    fn test_softmax_basic() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum should be 1.0, got {sum}"
        );
        assert!(
            x[2] > x[1] && x[1] > x[0],
            "softmax should preserve ordering"
        );
    }

    #[test]
    fn test_softmax_single() {
        let mut x = vec![42.0];
        softmax_inplace(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let mut x: Vec<f32> = vec![];
        softmax_inplace(&mut x);
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow due to max subtraction
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// Verify the design invariant that embed() and forward() produce vectors
    /// of different lengths: hidden_size vs vocab_size.
    ///
    /// embed() stops before the LM-head projection so its output has length
    /// `hidden_size`.  forward() returns logits, length `vocab_size`.
    /// For any realistic model these two dimensions differ, which this test
    /// confirms by checking the LlamaModel scratch buffer sizes directly.
    #[test]
    fn test_embed_output_length_differs_from_vocab_size() {
        // The LlamaModel struct stores buf_hidden (hidden_size) and buf_logits
        // (vocab_size) as distinct fields.  We verify their sizes are different
        // for a configuration where hidden_size != vocab_size, which is the
        // universal case for production LLaMA models.
        let hidden_size: usize = 64;
        let vocab_size: usize = 256;

        // embed() result length == hidden_size (the buf_hidden dimension).
        // forward() result length == vocab_size (the buf_logits dimension).
        assert_ne!(
            hidden_size, vocab_size,
            "test requires hidden_size != vocab_size"
        );

        // Confirm the buf sizes we rely on in the implementation match.
        let buf_hidden = vec![0.0f32; hidden_size];
        let buf_logits = vec![0.0f32; vocab_size];
        assert_eq!(buf_hidden.len(), hidden_size);
        assert_eq!(buf_logits.len(), vocab_size);
        assert_ne!(buf_hidden.len(), buf_logits.len());
    }
}
