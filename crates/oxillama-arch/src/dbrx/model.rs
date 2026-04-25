//! DBRX transformer model implementation.
//!
//! DBRX is a fine-grained MoE decoder with:
//! - Standard multi-head attention (GQA; no MLA).
//! - 16 routed experts, top-4 activation per token.
//! - RMSNorm + SwiGLU FFN per expert.
//!
//! Forward per layer: `RMSNorm → MHA → residual → RMSNorm → MoE → residual`.

use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::dbrx::config::DbrxConfig;
use crate::deepseek::moe::{moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode};
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// One DBRX transformer layer.
pub struct DbrxLayer {
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
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// MoE FFN weights.
    pub moe_weights: MoeWeights,
    /// MoE configuration.
    pub moe_config: MoeConfig,
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete DBRX model.
pub struct DbrxModel {
    /// Common model config (embedding dim, num layers, vocab size, etc.).
    pub config: ModelConfig,
    /// DBRX-specific config (MoE layout, etc.).
    pub dbrx_config: DbrxConfig,
    /// Token embedding table `[vocab_size, hidden_size]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<DbrxLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Precomputed RoPE frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Current token position (incremented on each forward call).
    current_pos: usize,
    // Scratch buffers (reused across calls).
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl DbrxModel {
    /// Create a new `DbrxModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        dbrx_config: DbrxConfig,
        token_embd: Vec<f32>,
        layers: Vec<DbrxLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
        let rope = RopeTable::new_standard(
            config.head_dim,
            config.max_context_length,
            config.rope_freq_base,
        );
        let dispatcher = KernelDispatcher::new();
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let max_ctx = config.max_context_length;
        Self {
            dispatcher,
            rope,
            current_pos: 0,
            buf_q: vec![0.0f32; q_dim],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_scores: vec![0.0f32; max_ctx],
            config,
            dbrx_config,
            token_embd,
            layers,
            output_norm,
            output,
        }
    }

    /// Reset sequence position.
    pub fn reset_position(&mut self) {
        self.current_pos = 0;
    }

    /// Run grouped-query attention for a single token at position `pos`.
    ///
    /// Returns the attention output of length `hidden_size`.
    fn attention_single_token(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let num_heads = self.config.num_attention_heads;
        let num_kv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let hidden = self.config.hidden_size;
        let heads_per_kv = num_heads.checked_div(num_kv).unwrap_or(1);

        let q_kernel = self
            .dispatcher
            .get_kernel(self.layers[layer_idx].attn_q.weight.tensor_type)
            .map_err(ArchError::from)?;
        let k_kernel = self
            .dispatcher
            .get_kernel(self.layers[layer_idx].attn_k.weight.tensor_type)
            .map_err(ArchError::from)?;
        let v_kernel = self
            .dispatcher
            .get_kernel(self.layers[layer_idx].attn_v.weight.tensor_type)
            .map_err(ArchError::from)?;

        self.layers[layer_idx]
            .attn_q
            .forward(&*q_kernel, x, &mut self.buf_q)?;
        self.layers[layer_idx]
            .attn_k
            .forward(&*k_kernel, x, &mut self.buf_k)?;
        self.layers[layer_idx]
            .attn_v
            .forward(&*v_kernel, x, &mut self.buf_v)?;

        // Apply RoPE to Q and K.
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * hd..(h + 1) * hd];
            self.rope.apply(q_head, pos);
        }
        for h in 0..num_kv {
            let k_head = &mut self.buf_k[h * hd..(h + 1) * hd];
            self.rope.apply(k_head, pos);
        }

        // Store KV.
        kv_cache.store_kv(layer_idx, &self.buf_k.clone(), &self.buf_v.clone())?;

        let cached_keys = kv_cache.get_keys(layer_idx)?;
        let cached_values = kv_cache.get_values(layer_idx)?;
        let seq_len = kv_cache.seq_len();

        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; hidden];
        let kv_stride = num_kv * hd;

        let n_tokens = (seq_len + 1).min(self.config.max_context_length);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * hd..(h + 1) * hd];

            // Compute attention scores.
            for t in 0..n_tokens {
                let k_off = t * kv_stride + kv_head * hd;
                if k_off + hd <= cached_keys.len() {
                    let k_head_t = &cached_keys[k_off..k_off + hd];
                    let score: f32 = q_head
                        .iter()
                        .zip(k_head_t.iter())
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        * scale;
                    self.buf_attn_scores[t] = score;
                } else {
                    self.buf_attn_scores[t] = f32::NEG_INFINITY;
                }
            }

            // Softmax over scores.
            let scores = &mut self.buf_attn_scores[..n_tokens];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            if exp_sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= exp_sum;
                }
            }

            // Accumulate V weighted by attention scores.
            for (t, &weight) in scores[..n_tokens].iter().enumerate() {
                let v_off = t * kv_stride + kv_head * hd;
                if v_off + hd <= cached_values.len() {
                    let v_head_t = &cached_values[v_off..v_off + hd];
                    let out_start = h * hd;
                    for (i, &v) in v_head_t.iter().enumerate() {
                        attn_out[out_start + i] += weight * v;
                    }
                }
            }
        }

        // Output projection.
        let out_kernel = self
            .dispatcher
            .get_kernel(self.layers[layer_idx].attn_output.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut projected = vec![0.0f32; hidden];
        self.layers[layer_idx]
            .attn_output
            .forward(&*out_kernel, &attn_out, &mut projected)?;

        kv_cache.advance();
        Ok(projected)
    }
}

impl ForwardPass for DbrxModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward: empty token sequence".to_string(),
            });
        }

        // ── Token embedding lookup ──────────────────────────────────────────────
        let mut hidden_states = vec![0.0f32; seq_len * hidden];
        for (t, &tok_id) in tokens.iter().enumerate() {
            let tok = tok_id as usize;
            if tok >= self.config.vocab_size {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "token id {tok} out of range (vocab_size={})",
                        self.config.vocab_size
                    ),
                });
            }
            let off = tok * hidden;
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[off..off + hidden]);
        }

        // ── Transformer layers ──────────────────────────────────────────────────
        let n_layers = self.layers.len();
        for layer_idx in 0..n_layers {
            for t in 0..seq_len {
                let pos = self.current_pos + t;

                // ─ Pre-attention norm ──────────────────────────────────────────
                let mut normed: Vec<f32> = hidden_states[t * hidden..(t + 1) * hidden].to_vec();
                self.layers[layer_idx].attn_norm.forward(&mut normed);

                // ─ MHA ────────────────────────────────────────────────────────
                let attn_out = self.attention_single_token(layer_idx, &normed, pos, kv_cache)?;

                // ─ Residual ────────────────────────────────────────────────────
                for (h, a) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(attn_out.iter())
                {
                    *h += a;
                }

                // ─ Pre-FFN norm ────────────────────────────────────────────────
                let mut ffn_normed: Vec<f32> = hidden_states[t * hidden..(t + 1) * hidden].to_vec();
                self.layers[layer_idx].ffn_norm.forward(&mut ffn_normed);

                // ─ MoE FFN ────────────────────────────────────────────────────
                let ffn_out = {
                    let layer = &self.layers[layer_idx];
                    moe_forward(&ffn_normed, &layer.moe_weights, &layer.moe_config).map_err(
                        |e| ArchError::ForwardPassError {
                            layer: layer_idx,
                            message: format!("MoE: {e}"),
                        },
                    )?
                };

                // ─ Residual after FFN ──────────────────────────────────────────
                for (h, f) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(ffn_out.iter())
                {
                    *h += f;
                }
            }
        }

        // ── Final norm + LM head (last token only) ──────────────────────────────
        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        let lm_kernel = self
            .dispatcher
            .get_kernel(self.output.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut logits = vec![0.0f32; vocab];
        self.output.forward(&*lm_kernel, last, &mut logits)?;

        self.current_pos += seq_len;

        Ok(logits)
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Runs all transformer layers (token embedding → N×(attn norm → MHA → residual
    /// → FFN norm → MoE → residual)) and the final `output_norm`, then returns the
    /// normalised last-token hidden state *without* projecting through the LM head.
    ///
    /// The returned vector has length `hidden_size`, not `vocab_size`.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "embed: empty token sequence".to_string(),
            });
        }

        // ── Token embedding lookup ──────────────────────────────────────────────
        let mut hidden_states = vec![0.0f32; seq_len * hidden];
        for (t, &tok_id) in tokens.iter().enumerate() {
            let tok = tok_id as usize;
            if tok >= self.config.vocab_size {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "token id {tok} out of range (vocab_size={})",
                        self.config.vocab_size
                    ),
                });
            }
            let off = tok * hidden;
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[off..off + hidden]);
        }

        // ── Transformer layers ──────────────────────────────────────────────────
        let n_layers = self.layers.len();
        for layer_idx in 0..n_layers {
            for t in 0..seq_len {
                let pos = self.current_pos + t;

                // ─ Pre-attention norm ──────────────────────────────────────────
                let mut normed: Vec<f32> = hidden_states[t * hidden..(t + 1) * hidden].to_vec();
                self.layers[layer_idx].attn_norm.forward(&mut normed);

                // ─ MHA ────────────────────────────────────────────────────────
                let attn_out = self.attention_single_token(layer_idx, &normed, pos, kv_cache)?;

                // ─ Residual ────────────────────────────────────────────────────
                for (h, a) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(attn_out.iter())
                {
                    *h += a;
                }

                // ─ Pre-FFN norm ────────────────────────────────────────────────
                let mut ffn_normed: Vec<f32> = hidden_states[t * hidden..(t + 1) * hidden].to_vec();
                self.layers[layer_idx].ffn_norm.forward(&mut ffn_normed);

                // ─ MoE FFN ────────────────────────────────────────────────────
                let ffn_out = {
                    let layer = &self.layers[layer_idx];
                    moe_forward(&ffn_normed, &layer.moe_weights, &layer.moe_config).map_err(
                        |e| ArchError::ForwardPassError {
                            layer: layer_idx,
                            message: format!("MoE: {e}"),
                        },
                    )?
                };

                // ─ Residual after FFN ──────────────────────────────────────────
                for (h, f) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(ffn_out.iter())
                {
                    *h += f;
                }
            }
        }

        // ── Final norm on last token (stop before LM head) ──────────────────────
        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        self.current_pos += seq_len;

        Ok(last.to_vec())
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
}

// ─── Builder helpers (for tests and custom loaders) ───────────────────────────

/// Build a `DbrxLayer` with zero-weight experts (for testing).
pub fn make_dbrx_layer(
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_experts: usize,
    top_k: usize,
    expert_inter: usize,
) -> DbrxLayer {
    use oxillama_gguf::GgufTensorType;

    let make_ql = |rows: usize, cols: usize| -> QuantLinear {
        let data = vec![0u8; rows * cols * 4];
        let weight = QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32);
        QuantLinear::new(weight, None)
    };

    let make_expert = |h: usize, inter: usize| -> DeepSeekExpert {
        DeepSeekExpert {
            gate: vec![0.0f32; inter * h],
            up: vec![0.0f32; inter * h],
            down: vec![0.0f32; h * inter],
            hidden_size: h,
            intermediate_size: inter,
        }
    };

    let router = vec![0.0f32; n_experts * hidden];
    let moe_weights = MoeWeights {
        router,
        routed_experts: (0..n_experts)
            .map(|_| make_expert(hidden, expert_inter))
            .collect(),
        shared_experts: vec![],
        expert_bias: None,
    };
    let moe_config = MoeConfig {
        hidden_size: hidden,
        expert_intermediate_size: expert_inter,
        n_shared_experts: 0,
        n_routed_experts: n_experts,
        top_k,
        routed_scaling_factor: 1.0,
        scoring_mode: ScoringMode::Softmax,
        shared_expert_intermediate_size: expert_inter,
    };

    DbrxLayer {
        attn_norm: RmsNorm::new(vec![1.0f32; hidden], 1e-5),
        attn_q: make_ql(num_heads * head_dim, hidden),
        attn_k: make_ql(num_kv_heads * head_dim, hidden),
        attn_v: make_ql(num_kv_heads * head_dim, hidden),
        attn_output: make_ql(hidden, num_heads * head_dim),
        ffn_norm: RmsNorm::new(vec![1.0f32; hidden], 1e-5),
        moe_weights,
        moe_config,
    }
}

/// Construct a `DbrxModel` from raw weights.
pub fn build_dbrx_model(
    config: ModelConfig,
    dbrx_config: DbrxConfig,
    token_embd: Vec<f32>,
    layers: Vec<DbrxLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> DbrxModel {
    DbrxModel::new(config, dbrx_config, token_embd, layers, output_norm, output)
}

// ─── GGUF loader helpers (local copies; do NOT extract to common) ─────────────

/// Dequantize raw tensor bytes to a `Vec<f32>`.
fn dequant_to_f32_local(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &oxillama_quant::KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    use oxillama_gguf::GgufTensorType;

    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    if tensor_type == GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == GgufTensorType::F16 {
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

/// Load and dequantize a named tensor to `Vec<f32>`.
fn load_dequant_tensor_local(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &oxillama_quant::KernelDispatcher,
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
    dequant_to_f32_local(info, data, dispatcher)
}

/// Load a named tensor as a `QuantLinear` (keeps data quantized).
fn load_quant_linear_local(
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

/// Load and dequantize a 1-D RMSNorm weight tensor.
fn load_rms_norm_weight_local(
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
    let dispatcher = oxillama_quant::KernelDispatcher::new();
    dequant_to_f32_local(info, data, &dispatcher)
}

// ─── GGUF loader ─────────────────────────────────────────────────────────────

/// Load a DBRX model from a parsed GGUF file.
///
/// DBRX GGUF tensor naming convention:
/// - `token_embd.weight` — embedding table `[vocab_size, hidden_size]`
/// - `blk.{i}.attn_norm.weight` — pre-attention RMSNorm
/// - `blk.{i}.attn_q.weight` — query projection
/// - `blk.{i}.attn_k.weight` — key projection
/// - `blk.{i}.attn_v.weight` — value projection
/// - `blk.{i}.attn_output.weight` — output projection
/// - `blk.{i}.ffn_norm.weight` — pre-FFN RMSNorm
/// - `blk.{i}.ffn_gate_inp.weight` — MoE router `[n_experts, hidden_size]`
/// - `blk.{i}.ffn_gate_exps.weight` — stacked gate `[n_experts, ffn_hidden, hidden]`
/// - `blk.{i}.ffn_up_exps.weight` — stacked up `[n_experts, ffn_hidden, hidden]`
/// - `blk.{i}.ffn_down_exps.weight` — stacked down `[n_experts, hidden, ffn_hidden]`
/// - `output_norm.weight` — final RMSNorm
/// - `output.weight` — LM head `[vocab_size, hidden_size]` (falls back to tied embd)
pub fn load_dbrx_from_gguf(model: &oxillama_gguf::GgufModel) -> ArchResult<DbrxModel> {
    let metadata = &model.file.metadata;
    let dispatcher = oxillama_quant::KernelDispatcher::new();

    // ── Parse configuration ────────────────────────────────────────────────────
    let model_config = crate::config::ModelConfig::from_metadata(metadata)?;
    let dbrx_config = crate::dbrx::config::DbrxConfig::from_metadata(metadata);

    let hidden = dbrx_config.hidden_size;
    let n_layers = dbrx_config.num_layers;
    let n_experts = dbrx_config.expert_count;
    let top_k = dbrx_config.expert_used_count;
    let ffn_hidden = dbrx_config.ffn_hidden_size;
    let rms_eps = dbrx_config.rms_norm_eps;

    // ── Token embedding ────────────────────────────────────────────────────────
    let embd_info =
        model
            .file
            .tensors
            .get("token_embd.weight")
            .map_err(|_| ArchError::MissingTensor {
                name: "token_embd.weight".to_string(),
            })?;
    let embd_data = model.tensor_data("token_embd.weight")?;
    let token_embd = dequant_to_f32_local(embd_info, embd_data, &dispatcher)?;

    // ── Transformer layers ─────────────────────────────────────────────────────
    let mut layers = Vec::with_capacity(n_layers);

    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        // Attention norms and projections
        let attn_norm_weights =
            load_rms_norm_weight_local(model, &format!("{prefix}.attn_norm.weight"))?;
        let attn_norm = RmsNorm::new(attn_norm_weights, rms_eps);

        let attn_q = load_quant_linear_local(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear_local(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear_local(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear_local(model, &format!("{prefix}.attn_output.weight"))?;

        // FFN norm
        let ffn_norm_weights =
            load_rms_norm_weight_local(model, &format!("{prefix}.ffn_norm.weight"))?;
        let ffn_norm = RmsNorm::new(ffn_norm_weights, rms_eps);

        // MoE router: [n_experts, hidden_size]
        let router_name = format!("{prefix}.ffn_gate_inp.weight");
        let router = load_dequant_tensor_local(model, &dispatcher, &router_name)?;

        let expected_router = n_experts * hidden;
        if router.len() != expected_router {
            return Err(ArchError::TensorShapeMismatch {
                tensor: router_name,
                expected: vec![n_experts, hidden],
                got: vec![router.len()],
            });
        }

        // Stacked expert tensors:
        // ffn_gate_exps: [n_experts, ffn_hidden, hidden]
        // ffn_up_exps:   [n_experts, ffn_hidden, hidden]
        // ffn_down_exps: [n_experts, hidden, ffn_hidden]
        let gate_name = format!("{prefix}.ffn_gate_exps.weight");
        let up_name = format!("{prefix}.ffn_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");

        let gate_stacked = load_dequant_tensor_local(model, &dispatcher, &gate_name)?;
        let up_stacked = load_dequant_tensor_local(model, &dispatcher, &up_name)?;
        let down_stacked = load_dequant_tensor_local(model, &dispatcher, &down_name)?;

        let gate_up_stride = ffn_hidden * hidden;
        let down_stride = hidden * ffn_hidden;

        let expected_gate_up = n_experts * gate_up_stride;
        let expected_down = n_experts * down_stride;

        if gate_stacked.len() != expected_gate_up {
            return Err(ArchError::TensorShapeMismatch {
                tensor: gate_name,
                expected: vec![n_experts, ffn_hidden, hidden],
                got: vec![gate_stacked.len()],
            });
        }
        if up_stacked.len() != expected_gate_up {
            return Err(ArchError::TensorShapeMismatch {
                tensor: up_name,
                expected: vec![n_experts, ffn_hidden, hidden],
                got: vec![up_stacked.len()],
            });
        }
        if down_stacked.len() != expected_down {
            return Err(ArchError::TensorShapeMismatch {
                tensor: down_name,
                expected: vec![n_experts, hidden, ffn_hidden],
                got: vec![down_stacked.len()],
            });
        }

        // Slice stacked tensors into per-expert weights
        let routed_experts: Vec<crate::deepseek::moe::DeepSeekExpert> = (0..n_experts)
            .map(|e| {
                let gate_start = e * gate_up_stride;
                let up_start = e * gate_up_stride;
                let down_start = e * down_stride;
                crate::deepseek::moe::DeepSeekExpert {
                    gate: gate_stacked[gate_start..gate_start + gate_up_stride].to_vec(),
                    up: up_stacked[up_start..up_start + gate_up_stride].to_vec(),
                    down: down_stacked[down_start..down_start + down_stride].to_vec(),
                    hidden_size: hidden,
                    intermediate_size: ffn_hidden,
                }
            })
            .collect();

        let moe_weights = MoeWeights {
            router,
            routed_experts,
            shared_experts: vec![],
            expert_bias: None,
        };

        let moe_config = MoeConfig {
            hidden_size: hidden,
            expert_intermediate_size: ffn_hidden,
            n_shared_experts: 0,
            n_routed_experts: n_experts,
            top_k,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: ffn_hidden,
        };

        layers.push(DbrxLayer {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm,
            moe_weights,
            moe_config,
        });
    }

    // ── Final norm and LM head ─────────────────────────────────────────────────
    let output_norm_weights = load_rms_norm_weight_local(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weights, rms_eps);

    // Try explicit output.weight; fall back to tied token_embd.weight
    let output = load_quant_linear_local(model, "output.weight")
        .or_else(|_| load_quant_linear_local(model, "token_embd.weight"))?;

    Ok(DbrxModel::new(
        model_config,
        dbrx_config,
        token_embd,
        layers,
        output_norm,
        output,
    ))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::KvCacheAccess;
    use oxillama_gguf::GgufTensorType;

    struct NullKv;
    impl KvCacheAccess for NullKv {
        fn seq_len(&self) -> usize {
            0
        }
        fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn advance(&mut self) {}
    }

    fn make_f32_ql(rows: usize, cols: usize) -> QuantLinear {
        let data = vec![0u8; rows * cols * 4];
        let weight = QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32);
        QuantLinear::new(weight, None)
    }

    fn build_tiny_model() -> DbrxModel {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const N_LAYERS: usize = 1;
        const N_EXPERTS: usize = 4;
        const TOP_K: usize = 2;
        const EXPERT_INTER: usize = 8;

        let dbrx_cfg = DbrxConfig {
            hidden_size: H,
            num_layers: N_LAYERS,
            num_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_seq_len: 64,
            expert_count: N_EXPERTS,
            expert_used_count: TOP_K,
            ffn_hidden_size: EXPERT_INTER,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let model_cfg = ModelConfig {
            architecture: "dbrx".to_string(),
            model_name: "test-dbrx".to_string(),
            hidden_size: H,
            intermediate_size: EXPERT_INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|_| {
                make_dbrx_layer(
                    H,
                    N_HEADS,
                    N_HEADS,
                    HEAD_DIM,
                    N_EXPERTS,
                    TOP_K,
                    EXPERT_INTER,
                )
            })
            .collect();

        let token_embd = vec![0.0f32; VOCAB * H];
        let output_norm = RmsNorm::new(vec![1.0f32; H], 1e-5);
        let output = make_f32_ql(VOCAB, H);

        build_dbrx_model(model_cfg, dbrx_cfg, token_embd, layers, output_norm, output)
    }

    #[test]
    fn forward_shape_correct() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[1u32], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), 32, "logits must have vocab_size=32 elements");
    }

    #[test]
    fn forward_all_finite() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let logits = model
            .forward(&[0u32], &mut kv)
            .expect("forward must succeed");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
    }

    #[test]
    fn empty_tokens_returns_error() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let result = model.forward(&[], &mut kv);
        assert!(result.is_err(), "empty token sequence must return an error");
    }

    #[test]
    fn embed_returns_hidden_size() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let embedding = model.embed(&[1u32], &mut kv).expect("embed must succeed");
        assert_eq!(
            embedding.len(),
            16,
            "embed output must have hidden_size=16 elements, got {}",
            embedding.len()
        );
    }

    #[test]
    fn embed_all_finite() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let embedding = model.embed(&[0u32], &mut kv).expect("embed must succeed");
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "all embedding values must be finite"
        );
    }

    #[test]
    fn embed_empty_tokens_returns_error() {
        let mut model = build_tiny_model();
        let mut kv = NullKv;
        let result = model.embed(&[], &mut kv);
        assert!(
            result.is_err(),
            "embed with empty token sequence must return an error"
        );
    }

    // ─── GGUF round-trip loader tests ─────────────────────────────────────────

    #[test]
    fn dbrx_loader_round_trip() {
        // Build a minimal valid DBRX GGUF binary (2 layers, 4 experts, hidden=32,
        // vocab=32, context=128) and verify that load_dbrx_from_gguf() succeeds
        // and that the resulting model has the expected structural properties.
        let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
        let gguf_model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("GGUF parse must succeed");

        let model = load_dbrx_from_gguf(&gguf_model).expect("load_dbrx_from_gguf must succeed");

        // The fixture encodes: vocab_size=32, hidden_size=32, context_length=128.
        assert_eq!(model.config.vocab_size, 32, "vocab_size from GGUF fixture");
        assert_eq!(
            model.config.hidden_size, 32,
            "hidden_size from GGUF fixture"
        );
        assert_eq!(
            model.config.max_context_length, 128,
            "max_context_length from GGUF fixture"
        );
        // 2-layer fixture.
        assert_eq!(model.layers.len(), 2, "num_layers from GGUF fixture");
        // 4 experts per layer.
        assert_eq!(
            model.dbrx_config.expert_count, 4,
            "expert_count from GGUF fixture"
        );
        // top-2 from 4.
        assert_eq!(
            model.dbrx_config.expert_used_count, 2,
            "expert_used_count from GGUF fixture"
        );
    }

    #[test]
    fn dbrx_loader_forward_no_nan() {
        // Load from the synthetic GGUF and run one forward pass.
        // The weights are all-zero F32, so logits will be exactly 0.0,
        // which satisfies the "no NaN" requirement.
        let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
        let gguf_model =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("GGUF parse must succeed");

        let mut model = load_dbrx_from_gguf(&gguf_model).expect("load_dbrx_from_gguf must succeed");

        let mut kv = NullKv;
        let logits = model.forward(&[0u32], &mut kv).expect("forward after load");

        // vocab_size = 32 tokens.
        assert_eq!(logits.len(), 32, "logit count must equal vocab_size=32");

        assert!(
            logits.iter().all(|v| !v.is_nan()),
            "forward pass after GGUF load must produce no NaN logits"
        );
    }
}
