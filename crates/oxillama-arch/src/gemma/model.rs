//! Gemma transformer forward pass implementation.
//!
//! Implements the Gemma 2/3 architecture with:
//! - Embedding scaling by `sqrt(hidden_size)`
//! - Pre-norm AND post-norm (RMSNorm before and after attn/FFN)
//! - GeGLU activation (GELU-gated) instead of SwiGLU
//! - Interleaved sliding window (local) and full causal (global) attention
//! - Logit soft-capping for attention scores and final logits

use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::{geglu_inplace, soft_cap_inplace};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

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
    /// RoPE precomputed frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Sliding window size for local attention layers.
    pub sliding_window: Option<usize>,
    /// Attention logit soft-cap value (0.0 = disabled).
    pub attn_logit_softcap: f32,
    /// Final logit soft-cap value (0.0 = disabled).
    pub final_logit_softcap: f32,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_post_norm: Vec<f32>,
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

impl GemmaModel {
    /// Create a new GemmaModel from preloaded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<GemmaLayer>,
        output_norm: RmsNorm,
        output: Option<QuantLinear>,
        attn_logit_softcap: f32,
        final_logit_softcap: f32,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let sliding_window = config.sliding_window;
        let embed_scale = (hidden_size as f32).sqrt();

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
            embed_scale,
            layers,
            output_norm,
            output,
            rope,
            dispatcher,
            sliding_window,
            attn_logit_softcap,
            final_logit_softcap,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_post_norm: vec![0.0; hidden_size],
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

    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    fn embed_token(&mut self, token: u32) {
        let hidden_size = self.config.hidden_size;
        let offset = token as usize * hidden_size;
        self.buf_hidden
            .copy_from_slice(&self.token_embd[offset..offset + hidden_size]);
        // Gemma scales embeddings by sqrt(hidden_size)
        for v in &mut self.buf_hidden {
            *v *= self.embed_scale;
        }
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

        // Apply RoPE
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

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;
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

        let scale = 1.0 / (head_dim as f32).sqrt();

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

            // Apply attention logit soft-capping
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

        // Project attention output
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let layer = &self.layers[layer_idx];
        let mut proj_out = vec![0.0f32; self.config.hidden_size];
        layer
            .attn_output
            .forward(&*o_kernel, &self.buf_attn_out, &mut proj_out)?;

        // Post-attention norm (Gemma 2+)
        if let Some(ref post_norm) = layer.attn_post_norm {
            post_norm.forward_to(&proj_out, &mut self.buf_post_norm);
            // Add post-normed output to residual
            for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_post_norm.iter()) {
                *h += p;
            }
        } else {
            // Add raw output to residual (Gemma 1 style)
            for (h, &p) in self.buf_hidden.iter_mut().zip(proj_out.iter()) {
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

        let gate_kernel = self.kernel_for(&layer.ffn_gate)?;
        let up_kernel = self.kernel_for(&layer.ffn_up)?;
        let down_kernel = self.kernel_for(&layer.ffn_down)?;

        layer
            .ffn_gate
            .forward(&*gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(&*up_kernel, &self.buf_norm, &mut self.buf_up)?;

        // GeGLU activation (Gemma uses GELU instead of SiLU)
        geglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(&*down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

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
        if let Some(ref output) = self.output {
            let kernel = self.kernel_for(output)?;
            output.forward(&*kernel, &self.buf_hidden, &mut self.buf_logits)?;
        } else {
            // Weight-tied: logits = hidden @ token_embd^T
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

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            self.embed_token(token);

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

        Ok(self.buf_logits.clone())
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

    // Read soft-capping values from metadata (Gemma 2 specific)
    let attn_logit_softcap = model
        .file
        .metadata
        .get_f32(&format!("{}.attention.logit_softcap", config.architecture))
        .unwrap_or(0.0);

    let final_logit_softcap = model
        .file
        .metadata
        .get_f32(&format!("{}.final_logit_softcap", config.architecture))
        .unwrap_or(0.0);

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
        );
        let ffn_post_norm = load_optional_rms_norm(
            model,
            &format!("{prefix}.ffn_post_norm.weight"),
            config.rms_norm_eps,
        );

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        // Gemma 2: even layers use sliding window, odd use full causal
        let use_sliding_window = i % 2 == 0;

        layers.push(GemmaLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_post_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_post_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            use_sliding_window,
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

    Ok(GemmaModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
        attn_logit_softcap,
        final_logit_softcap,
    ))
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
    let data = model.tensor_data(name)?;
    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let tensor = QuantTensor::new(data.to_vec(), shape, info.tensor_type);
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

/// Try to load an optional RMSNorm weight. Returns None if tensor is not present.
fn load_optional_rms_norm(
    model: &oxillama_gguf::GgufModel,
    name: &str,
    eps: f32,
) -> Option<RmsNorm> {
    if !model.file.tensors.contains(name) {
        return None;
    }
    load_rms_norm_weight(model, name)
        .ok()
        .map(|w| RmsNorm::new(w, eps))
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
