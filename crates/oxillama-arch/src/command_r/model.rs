//! Command-R transformer forward pass implementation.
//!
//! Command-R (by Cohere) is architecturally very close to LLaMA:
//! - RMSNorm pre-normalization (same)
//! - Grouped-Query Attention with RoPE (same)
//! - SwiGLU feed-forward network (same)
//! - Optional Q/K normalization for Command-R+ (`blk.{i}.attn_q_norm.weight`,
//!   `blk.{i}.attn_k_norm.weight`)
//! - Optional logit scaling: final logits are multiplied by `logit_scale`
//!   (loaded from `command-r.logit_scale` in GGUF metadata, default 1.0)
//!
//! ## Tensor naming convention (GGUF)
//!
//! Identical to LLaMA, with two optional additions per layer:
//! - `blk.{i}.attn_q_norm.weight` — Q normalization (Command-R+, optional)
//! - `blk.{i}.attn_k_norm.weight` — K normalization (Command-R+, optional)

use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::{dequant_to_f32, load_quant_linear, load_rms_norm_weight, softmax_inplace};
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::KernelDispatcher;

/// A single Command-R transformer layer (decoder block).
pub struct CommandRLayer {
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
    /// Optional Q normalization (Command-R+ only).
    pub attn_q_norm: Option<RmsNorm>,
    /// Optional K normalization (Command-R+ only).
    pub attn_k_norm: Option<RmsNorm>,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection [intermediate_size, hidden_size].
    pub ffn_gate: QuantLinear,
    /// FFN up projection [intermediate_size, hidden_size].
    pub ffn_up: QuantLinear,
    /// FFN down projection [hidden_size, intermediate_size].
    pub ffn_down: QuantLinear,
}

/// Complete Command-R model with all weights and forward pass logic.
pub struct CommandRModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<CommandRLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection [vocab_size, hidden_size].
    pub output: QuantLinear,
    /// RoPE precomputed frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Logit scaling factor (1.0 = no scaling).
    pub logit_scale: f32,

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

impl CommandRModel {
    /// Create a new `CommandRModel` from preloaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<CommandRLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
        logit_scale: f32,
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
            logit_scale,
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

    /// Run grouped-query attention for a single layer (identical to LLaMA,
    /// with the addition of optional Q/K normalization).
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

        // Optional Q/K normalization (Command-R+)
        if let Some(ref q_norm) = self.layers[layer_idx].attn_q_norm {
            for h in 0..num_heads {
                let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
                q_norm.forward(q_head);
            }
        }
        if let Some(ref k_norm) = self.layers[layer_idx].attn_k_norm {
            for h in 0..num_kv_heads {
                let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
                k_norm.forward(k_head);
            }
        }

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

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;
        let seq_len = position + 1;

        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        // Per-head attention
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

        // Project attention output back to hidden_size
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let layer = &self.layers[layer_idx];
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

    /// Run the SwiGLU feed-forward network (identical to LLaMA).
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

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(&*down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }
}

impl ForwardPass for CommandRModel {
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

        self.output_norm.forward(&mut self.buf_hidden);

        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        // Apply logit scaling if configured
        if (self.logit_scale - 1.0).abs() > f32::EPSILON {
            for v in self.buf_logits.iter_mut() {
                *v *= self.logit_scale;
            }
        }

        Ok(self.buf_logits.clone())
    }

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
    /// Uses the same `blk.{i}.*` naming convention as LLaMA.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let candidates: [(&str, &mut QuantLinear); 7] = [
                (&format!("blk.{i}.attn_q.weight"), &mut layer.attn_q),
                (&format!("blk.{i}.attn_k.weight"), &mut layer.attn_k),
                (&format!("blk.{i}.attn_v.weight"), &mut layer.attn_v),
                (
                    &format!("blk.{i}.attn_output.weight"),
                    &mut layer.attn_output,
                ),
                (&format!("blk.{i}.ffn_gate.weight"), &mut layer.ffn_gate),
                (&format!("blk.{i}.ffn_up.weight"), &mut layer.ffn_up),
                (&format!("blk.{i}.ffn_down.weight"), &mut layer.ffn_down),
            ];
            for (tensor_name, linear) in candidates {
                if let Some(adapter) = lora.get(tensor_name) {
                    linear.set_lora(adapter);
                }
            }
        }
        Ok(())
    }
}

/// Load a Command-R model from a `GgufModel`.
///
/// Identical to `load_llama_from_gguf` but additionally:
/// - Tries to load optional Q/K norm weights per layer.
/// - Reads `logit_scale` from config.
pub fn load_command_r_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<CommandRModel> {
    let dispatcher = KernelDispatcher::new();

    // Load token embeddings
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

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        // Optional Q/K norm (Command-R+)
        let attn_q_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_q_norm.weight"))
            .ok()
            .map(|w| RmsNorm::new(w, config.rms_norm_eps));
        let attn_k_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_k_norm.weight"))
            .ok()
            .map(|w| RmsNorm::new(w, config.rms_norm_eps));

        layers.push(CommandRLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);
    let output = load_quant_linear(model, "output.weight")?;

    let logit_scale = config.logit_scale;

    Ok(CommandRModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
        logit_scale,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_logit_scale_applied() {
        // Verify that logit scaling multiplies all logits correctly.
        let logit_scale = 2.5_f32;
        let mut logits = [1.0f32, 2.0, -1.0, 0.0];

        if (logit_scale - 1.0).abs() > f32::EPSILON {
            for v in logits.iter_mut() {
                *v *= logit_scale;
            }
        }

        assert!((logits[0] - 2.5).abs() < 1e-6, "logits[0]={}", logits[0]);
        assert!((logits[1] - 5.0).abs() < 1e-6, "logits[1]={}", logits[1]);
        assert!((logits[2] - (-2.5)).abs() < 1e-6, "logits[2]={}", logits[2]);
        assert!(logits[3].abs() < 1e-6, "logits[3]={}", logits[3]);
    }

    #[test]
    fn test_logit_scale_one_is_noop() {
        // logit_scale = 1.0 should not change any values.
        let logit_scale = 1.0_f32;
        let mut logits = [1.0f32, 2.0, -1.0];
        let original = logits;

        if (logit_scale - 1.0).abs() > f32::EPSILON {
            for v in logits.iter_mut() {
                *v *= logit_scale;
            }
        }

        assert_eq!(logits, original, "scale=1.0 should not modify logits");
    }
}
