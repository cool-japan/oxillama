//! Grok-1 transformer model implementation.
//!
//! Grok-1 is a MoE decoder-only transformer with:
//! - Standard grouped-query attention with RoPE (theta = 1e6).
//! - 8 routed experts, top-2 activation per token.
//! - RMSNorm + SwiGLU FFN per expert.
//!
//! Forward per layer: `RMSNorm → MHA → residual → RMSNorm → MoE → residual`.
//!
//! This implementation mirrors DBRX closely — only config defaults differ.

use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::deepseek::moe::{moe_forward, DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode};
use crate::error::{ArchError, ArchResult};
use crate::grok::config::GrokConfig;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// One Grok-1 transformer layer.
pub struct GrokLayer {
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

/// Complete Grok-1 model.
pub struct GrokModel {
    /// Common model config.
    pub config: ModelConfig,
    /// Grok-specific config (MoE layout, rope_theta, etc.).
    pub grok_config: GrokConfig,
    /// Token embedding table `[vocab_size, hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<GrokLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Precomputed RoPE frequency table.
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Current token position.
    current_pos: usize,
    // Scratch buffers.
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl GrokModel {
    /// Create a new `GrokModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        grok_config: GrokConfig,
        token_embd: Vec<f32>,
        layers: Vec<GrokLayer>,
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
            grok_config,
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

            // Softmax.
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

            // Accumulate V.
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

impl ForwardPass for GrokModel {
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

// ─── Builder helpers ──────────────────────────────────────────────────────────

/// Build a `GrokLayer` with zero-weight experts (for testing).
pub fn make_grok_layer(
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_experts: usize,
    top_k: usize,
    expert_inter: usize,
) -> GrokLayer {
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

    GrokLayer {
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

/// Construct a `GrokModel` from raw weights.
pub fn build_grok_model(
    config: ModelConfig,
    grok_config: GrokConfig,
    token_embd: Vec<f32>,
    layers: Vec<GrokLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> GrokModel {
    GrokModel::new(config, grok_config, token_embd, layers, output_norm, output)
}

// ─── Local GGUF helper functions (duplicated from llama; do not import from there) ─

/// Dequantize tensor data to f32.
fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;
    let tensor_type = info.tensor_type;

    // F32 tensors — direct copy.
    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    // F16 tensors — convert via half crate.
    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    // Quantized tensors — use kernel dequant.
    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    let n_blocks = n_elements.div_ceil(block_size);

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = &data[data_offset..data_offset + block_bytes];
        let out_slice =
            &mut out[out_offset..out_offset.saturating_add(block_size).min(n_elements)];
        kernel.dequant_block(block_data, out_slice)?;
    }

    Ok(out)
}

/// Load and dequantize a tensor to f32, looking it up by name.
fn load_dequant_tensor(
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

/// Load an RMSNorm weight vector from GGUF (always dequantized to F32).
fn load_rms_norm_weight(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let dispatcher = KernelDispatcher::new();
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    dequant_to_f32(info, data, &dispatcher)
}

/// Load the stacked MoE expert tensors for one Grok-1 layer and return `MoeWeights`.
///
/// Grok-1 GGUF layout per layer:
/// - `{prefix}.ffn_gate_inp.weight`  (fallback: `{prefix}.output_router.weight`)
///   Router matrix `[n_experts, hidden_size]`
/// - `{prefix}.ffn_gate_exps.weight` Stacked gate `[n_experts, ffn_hidden, hidden]`
/// - `{prefix}.ffn_up_exps.weight`   Stacked up   `[n_experts, ffn_hidden, hidden]`
/// - `{prefix}.ffn_down_exps.weight` Stacked down `[n_experts, hidden, ffn_hidden]`
fn load_grok_moe(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &KernelDispatcher,
    prefix: &str,
    n_experts: usize,
    hidden: usize,
    intermediate: usize,
) -> ArchResult<MoeWeights> {
    // --- Router: try primary name, fall back to alternate Grok variant ---
    let primary_router = format!("{prefix}.ffn_gate_inp.weight");
    let fallback_router = format!("{prefix}.output_router.weight");

    let router_name = if model.file.tensors.get(&primary_router).is_ok() {
        primary_router
    } else {
        fallback_router
    };

    let router = load_dequant_tensor(model, dispatcher, &router_name)?;

    // Validate: router must hold n_experts * hidden floats.
    let expected_router = n_experts * hidden;
    if router.len() != expected_router {
        return Err(ArchError::TensorShapeMismatch {
            tensor: router_name,
            expected: vec![n_experts, hidden],
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

    let gate_up_stride = intermediate * hidden;
    let down_stride = hidden * intermediate;

    if gate_stacked.len() != n_experts * gate_up_stride {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_gate_exps.weight"),
            expected: vec![n_experts, intermediate, hidden],
            got: vec![gate_stacked.len()],
        });
    }
    if up_stacked.len() != n_experts * gate_up_stride {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_up_exps.weight"),
            expected: vec![n_experts, intermediate, hidden],
            got: vec![up_stacked.len()],
        });
    }
    if down_stacked.len() != n_experts * down_stride {
        return Err(ArchError::TensorShapeMismatch {
            tensor: format!("{prefix}.ffn_down_exps.weight"),
            expected: vec![n_experts, hidden, intermediate],
            got: vec![down_stacked.len()],
        });
    }

    // Split stacked tensors into per-expert weight vectors.
    let routed_experts: Vec<DeepSeekExpert> = (0..n_experts)
        .map(|e| {
            let gate_start = e * gate_up_stride;
            let up_start = e * gate_up_stride;
            let down_start = e * down_stride;
            DeepSeekExpert {
                gate: gate_stacked[gate_start..gate_start + gate_up_stride].to_vec(),
                up: up_stacked[up_start..up_start + gate_up_stride].to_vec(),
                down: down_stacked[down_start..down_start + down_stride].to_vec(),
                hidden_size: hidden,
                intermediate_size: intermediate,
            }
        })
        .collect();

    Ok(MoeWeights {
        router,
        routed_experts,
        shared_experts: vec![],
        expert_bias: None,
    })
}

/// Load a Grok-1 model from a parsed GGUF file.
///
/// Parses `grok.*` metadata to build `GrokConfig` and `ModelConfig`, then loads
/// all tensors layer-by-layer.  Expert tensors are stored in GGUF as stacked
/// slabs (`ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`);
/// this function splits them into per-expert `DeepSeekExpert` structs and wraps
/// them in the standard `MoeWeights` / `MoeConfig` used by `moe_forward`.
pub fn load_grok_from_gguf(model: &oxillama_gguf::GgufModel) -> ArchResult<GrokModel> {
    let metadata = &model.file.metadata;

    // --- Parse architecture configs ---
    let grok_config = crate::grok::config::GrokConfig::from_metadata(metadata);

    let hidden = grok_config.hidden_size;
    let n_layers = grok_config.num_layers;
    let n_heads = grok_config.num_heads;
    let n_kv_heads = grok_config.num_kv_heads;
    let head_dim = grok_config.head_dim;
    let vocab = grok_config.vocab_size;
    let max_ctx = grok_config.max_seq_len;
    let n_experts = grok_config.expert_count;
    let top_k = grok_config.expert_used_count.max(1);
    let intermediate = grok_config.ffn_hidden_size;
    let rope_theta = grok_config.rope_theta;
    let norm_eps = grok_config.rms_norm_eps;

    // Build a ModelConfig from the parsed grok config values.
    let model_config = crate::config::ModelConfig {
        architecture: "grok".to_string(),
        model_name: metadata
            .get_string("general.name")
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "grok".to_string()),
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: n_kv_heads,
        head_dim,
        vocab_size: vocab,
        max_context_length: max_ctx,
        rms_norm_eps: norm_eps,
        rope_freq_base: rope_theta,
        num_experts: n_experts,
        num_experts_used: top_k,
        ..crate::config::ModelConfig::default()
    };

    let dispatcher = KernelDispatcher::new();

    // --- Token embedding ---
    let token_embd = load_dequant_tensor(model, &dispatcher, "token_embd.weight")?;
    if token_embd.len() != vocab * hidden {
        return Err(ArchError::TensorShapeMismatch {
            tensor: "token_embd.weight".to_string(),
            expected: vec![vocab, hidden],
            got: vec![token_embd.len()],
        });
    }

    // --- Transformer layers ---
    let mut layers: Vec<GrokLayer> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        // Pre-attention RMSNorm.
        let attn_norm_w = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let attn_norm = RmsNorm::new(attn_norm_w, norm_eps);

        // Attention projections (quantized, kept as-is).
        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        // Pre-FFN RMSNorm.
        let ffn_norm_w = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;
        let ffn_norm = RmsNorm::new(ffn_norm_w, norm_eps);

        // MoE FFN.
        let moe_weights =
            load_grok_moe(model, &dispatcher, &prefix, n_experts, hidden, intermediate)?;
        let moe_config = MoeConfig {
            hidden_size: hidden,
            expert_intermediate_size: intermediate,
            n_shared_experts: 0,
            n_routed_experts: n_experts,
            top_k,
            routed_scaling_factor: 1.0,
            scoring_mode: ScoringMode::Softmax,
            shared_expert_intermediate_size: intermediate,
        };

        layers.push(GrokLayer {
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

    // --- Output norm and LM head ---
    let output_norm_w = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_w, norm_eps);
    let output = load_quant_linear(model, "output.weight")?;

    Ok(GrokModel::new(
        model_config,
        grok_config,
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

    fn build_tiny_model() -> GrokModel {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const N_LAYERS: usize = 1;
        const N_EXPERTS: usize = 8;
        const TOP_K: usize = 2;
        const EXPERT_INTER: usize = 8;

        let grok_cfg = GrokConfig {
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
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
        };

        let model_cfg = ModelConfig {
            architecture: "grok".to_string(),
            model_name: "test-grok".to_string(),
            hidden_size: H,
            intermediate_size: EXPERT_INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|_| {
                make_grok_layer(
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

        build_grok_model(model_cfg, grok_cfg, token_embd, layers, output_norm, output)
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
        let embedding = model
            .embed(&[1u32], &mut kv)
            .expect("embed must succeed");
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
        let embedding = model
            .embed(&[0u32], &mut kv)
            .expect("embed must succeed");
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
        assert!(result.is_err(), "embed with empty token sequence must return an error");
    }

    #[test]
    fn rope_theta_is_1e6() {
        let mut store = oxillama_gguf::MetadataStore::new();
        // No key → should default to 1e6
        let cfg = crate::grok::config::GrokConfig::from_metadata(&store);
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1.0,
            "Grok-1 default rope_theta must be 1e6"
        );

        // Explicit override should be respected.
        store.insert(
            "grok.rope.freq_base".to_string(),
            oxillama_gguf::MetadataValue::Float32(500_000.0),
        );
        let cfg2 = crate::grok::config::GrokConfig::from_metadata(&store);
        assert!(
            (cfg2.rope_theta - 500_000.0).abs() < 1.0,
            "explicit rope_theta override must be respected"
        );
    }

    // ─── GGUF loader tests ─────────────────────────────────────────────────────

    /// Load a model from the minimal Grok GGUF fixture and verify the call succeeds.
    #[test]
    fn grok_loader_round_trip() {
        let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
        let gguf =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("GGUF parse must succeed");
        let model = load_grok_from_gguf(&gguf).expect("load_grok_from_gguf must succeed");

        // Config sanity: hidden=32, vocab=32, 2 layers, 2 experts.
        assert_eq!(model.config.hidden_size, 32, "hidden_size must be 32");
        assert_eq!(model.config.vocab_size, 32, "vocab_size must be 32");
        assert_eq!(model.layers.len(), 2, "must have 2 layers");
        assert_eq!(
            model.config.num_experts, 2,
            "must have 2 MoE experts"
        );
    }

    /// Build a model directly with `num_kv_heads=1` (MQA) and verify forward pass.
    #[test]
    fn grok_loader_handles_mqa() {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 4;
        const N_KV_HEADS: usize = 1; // MQA
        const HEAD_DIM: usize = 4;
        const N_LAYERS: usize = 1;
        const N_EXPERTS: usize = 2;
        const TOP_K: usize = 2;
        const EXPERT_INTER: usize = 8;

        let grok_cfg = GrokConfig {
            hidden_size: H,
            num_layers: N_LAYERS,
            num_heads: N_HEADS,
            num_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_seq_len: 64,
            expert_count: N_EXPERTS,
            expert_used_count: TOP_K,
            ffn_hidden_size: EXPERT_INTER,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
        };

        let model_cfg = crate::config::ModelConfig {
            architecture: "grok".to_string(),
            model_name: "mqa-test".to_string(),
            hidden_size: H,
            intermediate_size: EXPERT_INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 64,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            num_experts: N_EXPERTS,
            num_experts_used: TOP_K,
            ..crate::config::ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|_| make_grok_layer(H, N_HEADS, N_KV_HEADS, HEAD_DIM, N_EXPERTS, TOP_K, EXPERT_INTER))
            .collect();
        let token_embd = vec![0.0f32; VOCAB * H];
        let output_norm = RmsNorm::new(vec![1.0f32; H], 1e-5);
        let output = make_f32_ql(VOCAB, H);

        let mut model = build_grok_model(model_cfg, grok_cfg, token_embd, layers, output_norm, output);
        let mut kv = NullKv;

        let logits = model.forward(&[0u32], &mut kv).expect("MQA forward must succeed");
        assert_eq!(logits.len(), VOCAB, "logits must have vocab_size elements");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all MQA logits must be finite"
        );
    }

    /// Load from GGUF, run forward with token [0], assert all logits are finite.
    #[test]
    fn grok_loader_forward_no_nan() {
        let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
        let gguf =
            oxillama_gguf::GgufModel::from_bytes(bytes).expect("GGUF parse must succeed");
        let mut model = load_grok_from_gguf(&gguf).expect("load_grok_from_gguf must succeed");
        let mut kv = NullKv;

        let logits = model.forward(&[0u32], &mut kv).expect("forward must succeed");
        assert_eq!(
            logits.len(),
            model.config.vocab_size,
            "logit count must equal vocab_size"
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite (no NaN / Inf)"
        );
    }
}
