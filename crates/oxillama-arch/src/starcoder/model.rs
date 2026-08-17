//! StarCoder (GPT-BigCode) transformer forward pass implementation.
//!
//! StarCoder uses a GPT-BigCode architecture which differs from LLaMA in
//! several important ways:
//!
//! 1. **LayerNorm** (not RMSNorm): `y = (x - mean) / sqrt(var + eps) * w + b`
//! 2. **GELU activation** (not SwiGLU): gate-free FFN with up → GELU → down
//! 3. **Learned absolute position embeddings** (not RoPE)
//! 4. **Multi-Query Attention (MQA)**: all Q heads share a single K/V head
//! 5. **Fused QKV projection**: single weight of shape
//!    `[(num_heads + 2) * head_dim, hidden_size]`
//! 6. **Bias everywhere**: attention projections, FFN projections, and norms
//!    all have bias terms

use crate::common::gelu::gelu_inplace;
use crate::common::layer_norm::LayerNorm;
use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::common::loader::dequant_to_f32;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

/// A single StarCoder transformer layer (GPT-BigCode decoder block).
pub struct StarcoderLayer {
    /// Pre-attention LayerNorm (with bias).
    pub attn_norm: LayerNorm,
    /// Fused QKV projection `[(num_heads+2)*head_dim, hidden_size]`.
    pub attn_qkv: QuantLinear,
    /// Fused QKV bias `[(num_heads+2)*head_dim]`.
    pub attn_qkv_bias: Vec<f32>,
    /// Attention output projection `[hidden_size, num_heads*head_dim]`.
    pub attn_out: QuantLinear,
    /// Output projection bias `[hidden_size]`.
    pub attn_out_bias: Vec<f32>,
    /// Pre-FFN LayerNorm (with bias).
    pub ffn_norm: LayerNorm,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// FFN up bias `[intermediate_size]`.
    pub ffn_up_bias: Vec<f32>,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,
    /// FFN down bias `[hidden_size]`.
    pub ffn_down_bias: Vec<f32>,
}

/// Complete StarCoder model with all weights and forward pass logic.
pub struct StarcoderModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Absolute position embeddings `[max_position * hidden_size]`.
    pub position_embd: Vec<f32>,
    /// Maximum number of positions loaded from the position embedding tensor.
    pub max_position: usize,
    /// Transformer layers.
    pub layers: Vec<StarcoderLayer>,
    /// Final LayerNorm before LM head.
    pub output_norm: LayerNorm,
    /// LM head `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,

    // Scratch buffers (reused across forward calls)
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    /// Fused QKV output: `[(num_heads+2)*head_dim]`
    buf_qkv: Vec<f32>,
    /// Q portion: `[num_heads * head_dim]`
    buf_q: Vec<f32>,
    /// K portion: `[head_dim]` (MQA — single K head)
    buf_k: Vec<f32>,
    /// V portion: `[head_dim]` (MQA — single V head)
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_ffn: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl StarcoderModel {
    /// Create a new `StarcoderModel` from preloaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        position_embd: Vec<f32>,
        max_position: usize,
        layers: Vec<StarcoderLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        // MQA: always 1 K/V head, but respect config in case GGUF disagrees.
        // In practice `num_kv_heads` will be 1 for StarCoder.
        let head_dim = config.head_dim;
        let qkv_total = (num_heads + 2) * head_dim; // fused QKV
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let dispatcher = KernelDispatcher::new();

        Self {
            config,
            token_embd,
            position_embd,
            max_position,
            layers,
            output_norm,
            output,
            dispatcher,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_qkv: vec![0.0; qkv_total],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; head_dim],
            buf_v: vec![0.0; head_dim],
            buf_attn_out: vec![0.0; hidden_size],
            buf_ffn: vec![0.0; intermediate_size],
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

    /// Embed a single token at a given position.
    ///
    /// `buf_hidden = token_embd[token] + position_embd[position]`
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] when `position >= max_position` — StarCoder
    /// has no positional-encoding fallback past its trained context (unlike
    /// RoPE-based architectures, which can extrapolate), so silently clamping
    /// to the last learned embedding used to produce wrong-but-finite output
    /// for every token past the training context instead of a clear error.
    fn embed_token(&mut self, token: u32, position: usize) -> ArchResult<()> {
        if position >= self.max_position {
            return Err(ArchError::ConfigMismatch {
                param: "position".to_string(),
                expected: format!("< max_position ({})", self.max_position),
                got: position.to_string(),
            });
        }
        let hidden_size = self.config.hidden_size;
        let tok_offset = token as usize * hidden_size;
        let pos_offset = position * hidden_size;

        let tok_row = self
            .token_embd
            .get(tok_offset..tok_offset + hidden_size)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {}", self.config.vocab_size),
                got: token.to_string(),
            })?;
        let pos_row = &self.position_embd[pos_offset..pos_offset + hidden_size];

        for i in 0..hidden_size {
            self.buf_hidden[i] = tok_row[i] + pos_row[i];
        }
        Ok(())
    }

    /// Run Multi-Query Attention for a single layer.
    ///
    /// MQA: all Q heads share one K head and one V head.
    /// The fused QKV weight produces `[(num_heads + 2) * head_dim]` outputs:
    /// - Q: rows `[0 .. num_heads * head_dim]`
    /// - K: rows `[num_heads * head_dim .. (num_heads+1) * head_dim]`
    /// - V: rows `[(num_heads+1) * head_dim .. (num_heads+2) * head_dim]`
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let qkv_total = (num_heads + 2) * head_dim;

        // Fused QKV projection + bias
        let qkv_kernel = self.kernel_for(&self.layers[layer_idx].attn_qkv)?;
        let layer = &self.layers[layer_idx];
        layer
            .attn_qkv
            .forward(&*qkv_kernel, &self.buf_norm, &mut self.buf_qkv)?;

        // Add fused QKV bias
        let qkv_bias = &layer.attn_qkv_bias;
        for (v, &b) in self.buf_qkv.iter_mut().zip(qkv_bias.iter()) {
            *v += b;
        }

        // Split Q / K / V
        let q_len = num_heads * head_dim;
        self.buf_q.copy_from_slice(&self.buf_qkv[..q_len]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[q_len..q_len + head_dim]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[q_len + head_dim..qkv_total]);

        // StarCoder uses absolute position embeddings — no RoPE applied here.

        // Store K, V (single head) in cache.
        // kv_dim = head_dim (MQA: 1 K/V head)
        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;

        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        // Per-head attention (all Q heads attend to the single K/V head)
        for h in 0..num_heads {
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                // MQA: K/V cache stores 1 head, so kv_dim = head_dim
                let k_offset = pos * head_dim;
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
                let v_offset = pos * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Output projection + bias
        let out_kernel = self.kernel_for(&self.layers[layer_idx].attn_out)?;
        let layer = &self.layers[layer_idx];
        let mut proj_out = vec![0.0f32; self.config.hidden_size];
        layer
            .attn_out
            .forward(&*out_kernel, &self.buf_attn_out, &mut proj_out)?;

        // Add output bias
        for (p, &b) in proj_out.iter_mut().zip(layer.attn_out_bias.iter()) {
            *p += b;
        }

        // Residual add
        for (h, &p) in self.buf_hidden.iter_mut().zip(proj_out.iter()) {
            *h += p;
        }

        Ok(())
    }

    /// Run the GELU feed-forward network.
    ///
    /// FFN(x) = ffn_down(gelu(ffn_up(x) + ffn_up_bias) + ffn_down_bias)
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        let up_kernel = self.kernel_for(&layer.ffn_up)?;
        let down_kernel = self.kernel_for(&layer.ffn_down)?;

        // up = ffn_up(norm) + bias
        layer
            .ffn_up
            .forward(&*up_kernel, &self.buf_norm, &mut self.buf_ffn)?;
        for (v, &b) in self.buf_ffn.iter_mut().zip(layer.ffn_up_bias.iter()) {
            *v += b;
        }

        // Apply GELU in-place (gate-free)
        gelu_inplace(&mut self.buf_ffn);

        // down = ffn_down(up) + bias
        layer
            .ffn_down
            .forward(&*down_kernel, &self.buf_ffn, &mut self.buf_ffn_out)?;
        for (v, &b) in self.buf_ffn_out.iter_mut().zip(layer.ffn_down_bias.iter()) {
            *v += b;
        }

        // Residual add
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }
}

impl ForwardPass for StarcoderModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        crate::common::validate_context_bounds(&self.config, start_pos, tokens.len())?;
        crate::common::validate_token_ids(&self.config, tokens)?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;

            // Embed token + absolute position
            self.embed_token(token, position)?;

            for layer_idx in 0..self.layers.len() {
                // Pre-attention LayerNorm
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                // MQA attention + residual
                self.attention(layer_idx, position, kv_cache)?;

                // Pre-FFN LayerNorm
                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                // GELU FFN + residual
                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }

        // Final norm
        self.output_norm.forward(&mut self.buf_hidden);

        // LM head
        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        Ok(self.buf_logits.clone())
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let start_pos = kv_cache.seq_len();
        crate::common::validate_context_bounds(&self.config, start_pos, tokens.len())?;
        crate::common::validate_token_ids(&self.config, tokens)?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token(token, position)?;

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
    /// Delegates to [`Self::apply_lora_scaled`] with `scale = 1.0`.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        self.apply_lora_scaled(lora, 1.0)
    }

    /// Attach LoRA adapters with an extra scale multiplier.
    ///
    /// StarCoder's four projections per layer — the fused `attn_qkv`, the
    /// attention output projection, and the two FFN projections — are looked
    /// up by their base GGUF tensor names and pushed (accumulating) rather
    /// than replaced, so a second call — or an entry from
    /// [`LoraStack`](crate::lora::LoraStack) — composes instead of clobbering.
    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let candidates: [(&str, &mut QuantLinear); 4] = [
                (&format!("blk.{i}.attn_qkv.weight"), &mut layer.attn_qkv),
                (&format!("blk.{i}.attn_output.weight"), &mut layer.attn_out),
                (&format!("blk.{i}.ffn_up.weight"), &mut layer.ffn_up),
                (&format!("blk.{i}.ffn_down.weight"), &mut layer.ffn_down),
            ];
            for (tensor_name, linear) in candidates {
                if let Some(adapter) = lora.get(tensor_name) {
                    linear.push_lora(adapter, scale);
                }
            }
        }
        Ok(())
    }

    fn unapply_all_loras(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn_qkv.clear_lora();
            layer.attn_out.clear_lora();
            layer.ffn_up.clear_lora();
            layer.ffn_down.clear_lora();
        }
    }
}

/// In-place numerically-stable softmax.
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

/// Load a StarCoder model from a `GgufModel`.
pub fn load_starcoder_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<StarcoderModel> {
    let dispatcher = KernelDispatcher::new();

    // Load token embeddings (F32 or F16)
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // Load position embeddings
    let pos_data = model.tensor_data("position_embd.weight")?;
    let pos_info = model.file.tensors.get("position_embd.weight")?;
    let position_embd = dequant_to_f32(pos_info, pos_data, &dispatcher)?;

    // Determine max_position from the position embedding tensor shape.
    // GGUF order is fastest-changing-first, so the tensor is stored as
    // [hidden_size, max_position]; the position count is the second dimension.
    let max_position = if pos_info.dimensions.len() >= 2 {
        pos_info.dimensions[1] as usize
    } else {
        config.max_context_length
    };

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        // Pre-attention LayerNorm (weight + bias)
        let attn_norm_w =
            load_f32_tensor(model, &format!("{prefix}.attn_norm.weight"), &dispatcher)?;
        let attn_norm_b = load_f32_tensor(model, &format!("{prefix}.attn_norm.bias"), &dispatcher)?;
        let attn_norm = LayerNorm::new(attn_norm_w, Some(attn_norm_b), config.rms_norm_eps);

        // Fused QKV weight + bias
        let attn_qkv = load_starcoder_quant_linear(model, &format!("{prefix}.attn_qkv.weight"))?;
        let attn_qkv_bias =
            load_f32_tensor(model, &format!("{prefix}.attn_qkv.bias"), &dispatcher)?;

        // Attention output projection + bias
        //
        // GGUF names this `attn_output.weight`/`.bias` for every architecture
        // (gguf-py's `TENSOR_NAMES[MODEL_TENSOR.ATTN_OUT] =
        // "blk.{bid}.attn_output"`); a prior version of this loader read
        // `attn_out.weight`, which never matches a real checkpoint.
        let attn_out = load_starcoder_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;
        let attn_out_bias =
            load_f32_tensor(model, &format!("{prefix}.attn_output.bias"), &dispatcher)?;

        // Pre-FFN LayerNorm (weight + bias)
        let ffn_norm_w = load_f32_tensor(model, &format!("{prefix}.ffn_norm.weight"), &dispatcher)?;
        let ffn_norm_b = load_f32_tensor(model, &format!("{prefix}.ffn_norm.bias"), &dispatcher)?;
        let ffn_norm = LayerNorm::new(ffn_norm_w, Some(ffn_norm_b), config.rms_norm_eps);

        // FFN up + bias
        let ffn_up = load_starcoder_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_up_bias = load_f32_tensor(model, &format!("{prefix}.ffn_up.bias"), &dispatcher)?;

        // FFN down + bias
        let ffn_down = load_starcoder_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;
        let ffn_down_bias =
            load_f32_tensor(model, &format!("{prefix}.ffn_down.bias"), &dispatcher)?;

        layers.push(StarcoderLayer {
            attn_norm,
            attn_qkv,
            attn_qkv_bias,
            attn_out,
            attn_out_bias,
            ffn_norm,
            ffn_up,
            ffn_up_bias,
            ffn_down,
            ffn_down_bias,
        });
    }

    // Final LayerNorm + LM head
    let output_norm_w = load_f32_tensor(model, "output_norm.weight", &dispatcher)?;
    let output_norm_b = load_f32_tensor(model, "output_norm.bias", &dispatcher)?;
    let output_norm = LayerNorm::new(output_norm_w, Some(output_norm_b), config.rms_norm_eps);

    // GPT-BigCode defaults `tie_word_embeddings=true`, so most StarCoder
    // checkpoints ship no standalone `output.weight` at all; llama.cpp
    // duplicates `token_embd.weight` into `output` in that case
    // (`llama-model.cpp`, `LLM_ARCH_STARCODER`: `create_tensor(..., OUTPUT,
    // ..., TENSOR_NOT_REQUIRED)` then falls back to `TOKEN_EMBD` when null).
    let output = crate::common::loader::load_lm_head(model, "output.weight", "token_embd.weight")?;

    Ok(StarcoderModel::new(
        config.clone(),
        token_embd,
        position_embd,
        max_position,
        layers,
        output_norm,
        output,
    ))
}

/// Load a F32 (or F16) tensor from GGUF and dequantize it to `Vec<f32>`.
///
/// Bias tensors in StarCoder GGUFs are always stored as F32.
fn load_f32_tensor(
    model: &oxillama_gguf::GgufModel,
    name: &str,
    dispatcher: &KernelDispatcher,
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

/// Load a quantized linear layer from GGUF (without bias; bias is stored separately).
fn load_starcoder_quant_linear(
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
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    // Shared mmap-backed view, not a `to_vec()` copy: the GEMV kernels only
    // ever read `&[u8]` out of the payload, so a private copy would double
    // the checkpoint's resident cost for nothing.
    let data = model.tensor_bytes(name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);

    Ok(QuantLinear::new(tensor, None))
}

#[cfg(test)]
mod tests {
    /// Verify that the MQA split logic correctly partitions QKV outputs.
    #[test]
    fn test_mqa_qkv_split() {
        let num_heads: usize = 4;
        let head_dim: usize = 8;
        let qkv_total = (num_heads + 2) * head_dim; // 48

        let qkv: Vec<f32> = (0..qkv_total as i32).map(|x| x as f32).collect();

        let q_len = num_heads * head_dim; // 32
        let q = &qkv[..q_len]; // indices 0..32
        let k = &qkv[q_len..q_len + head_dim]; // indices 32..40
        let v = &qkv[q_len + head_dim..qkv_total]; // indices 40..48

        assert_eq!(q.len(), num_heads * head_dim);
        assert_eq!(k.len(), head_dim);
        assert_eq!(v.len(), head_dim);

        // Q starts at 0
        assert!((q[0] - 0.0).abs() < 1e-6);
        // K starts at q_len
        assert!((k[0] - q_len as f32).abs() < 1e-6, "k[0]={}", k[0]);
        // V starts at q_len + head_dim
        assert!(
            (v[0] - (q_len + head_dim) as f32).abs() < 1e-6,
            "v[0]={}",
            v[0]
        );
    }

    /// Verify absolute position embedding: hidden = token_embd + pos_embd.
    #[test]
    fn test_position_embedding_add() {
        let hidden_size: usize = 4;
        let token_embd = [1.0f32, 2.0, 3.0, 4.0];
        let pos_embd = [0.1f32, 0.2, 0.3, 0.4];

        let mut buf_hidden = [0.0f32; 4];
        for i in 0..hidden_size {
            buf_hidden[i] = token_embd[i] + pos_embd[i];
        }

        assert!(
            (buf_hidden[0] - 1.1).abs() < 1e-5,
            "buf[0]={}",
            buf_hidden[0]
        );
        assert!(
            (buf_hidden[3] - 4.4).abs() < 1e-5,
            "buf[3]={}",
            buf_hidden[3]
        );
    }
}
