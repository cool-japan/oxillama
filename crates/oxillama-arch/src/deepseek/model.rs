//! DeepSeek-V2 transformer model implementation.
//!
//! Provides the `DeepSeekModel` struct (embedding → N×DeepSeekLayer → output norm
//! → LM head) and `load_deepseek_from_gguf()` for loading from a GGUF file.
//!
//! Each `DeepSeekLayer` contains:
//! - Pre-attention RMSNorm
//! - Multi-head Latent Attention (`mla_forward`)
//! - An arch-internal `MlaLatentCache` (NOT a `KvCacheAccess` extension)
//! - Pre-FFN RMSNorm
//! - A `FfnKind` — either dense SwiGLU (first N_DENSE layers) or DeepSeek MoE
//!
//! The `ForwardPass` impl dispatches through the layers sequentially.

use crate::common::linear::QuantLinear;
use crate::common::mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
use crate::common::rms_norm::RmsNorm;
use crate::common::swiglu::swiglu_inplace;
use crate::config::{DeepSeekConfig, ModelConfig};
use crate::deepseek::moe::{moe_forward, MoeConfig, MoeWeights};
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

/// How many leading layers use a dense SwiGLU FFN (the rest use MoE).
///
/// DeepSeek-V2 documentation specifies 1 dense layer at the start. This
/// constant is exported for use by loaders and callers building layer vectors.
pub const N_DENSE_LAYERS: usize = 1;

// ─── Layer FFN variant ────────────────────────────────────────────────────────

/// Dense SwiGLU FFN weights for the non-MoE layers.
pub struct DenseFfn {
    /// Gate projection `[intermediate_size, hidden_size]`.
    pub gate: QuantLinear,
    /// Up projection `[intermediate_size, hidden_size]`.
    pub up: QuantLinear,
    /// Down projection `[hidden_size, intermediate_size]`.
    pub down: QuantLinear,
}

/// FFN variant for a DeepSeek layer.
pub enum FfnKind {
    /// Standard dense SwiGLU FFN (leading layers).
    Dense(Box<DenseFfn>),
    /// DeepSeek sparse MoE FFN.
    Moe {
        /// MoE layer weights.
        weights: Box<MoeWeights>,
        /// Configuration (shared with the layer).
        config: MoeConfig,
    },
}

// ─── Single transformer layer ─────────────────────────────────────────────────

/// One DeepSeek-V2 transformer block.
pub struct DeepSeekLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// MLA weights and RoPE table.
    pub mla_weights: MlaWeights,
    /// MLA configuration.
    pub mla_config: MlaConfig,
    /// Arch-internal latent KV cache for this layer.
    pub mla_cache: MlaLatentCache,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN variant.
    pub ffn: FfnKind,
}

// ─── Full model ───────────────────────────────────────────────────────────────

/// Complete DeepSeek-V2 model.
pub struct DeepSeekModel {
    /// Common model config (embedding dim, num layers, vocab size, etc.).
    pub config: ModelConfig,
    /// DeepSeek-specific config (MLA ranks, MoE layout, etc.).
    pub ds_config: DeepSeekConfig,
    /// Token embedding table `[vocab_size, hidden_size]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<DeepSeekLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Sequence position counter (incremented on each forward call).
    current_pos: usize,
}

impl DeepSeekModel {
    /// Construct a `DeepSeekModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        ds_config: DeepSeekConfig,
        token_embd: Vec<f32>,
        layers: Vec<DeepSeekLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
        Self {
            dispatcher: KernelDispatcher::new(),
            current_pos: 0,
            config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        }
    }

    /// Reset the sequence position (use between independent sequences).
    pub fn reset_position(&mut self) {
        self.current_pos = 0;
        for layer in &mut self.layers {
            layer.mla_cache.clear();
        }
    }
}

impl ForwardPass for DeepSeekModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        _kv_cache: &mut dyn KvCacheAccess,
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
        // Build [seq_len × hidden_size] input tensor from token ids.
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
            let embd_off = tok * hidden;
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[embd_off..embd_off + hidden]);
        }

        let position = self.current_pos;

        // ── Transformer layers ──────────────────────────────────────────────────
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // ─ Pre-attention norm ──────────────────────────────────────────────
            // Normalise each token's hidden state individually.
            let mut normed = vec![0.0f32; seq_len * hidden];
            for t in 0..seq_len {
                let src = &hidden_states[t * hidden..(t + 1) * hidden];
                let dst = &mut normed[t * hidden..(t + 1) * hidden];
                dst.copy_from_slice(src);
                layer
                    .attn_norm
                    .forward(&mut normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ MLA forward ────────────────────────────────────────────────────
            let attn_out = mla_forward(
                &normed,
                &layer.mla_weights,
                &layer.mla_config,
                &mut layer.mla_cache,
                position,
            )
            .map_err(|e| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("MLA: {e}"),
            })?;

            // ─ Residual connection ─────────────────────────────────────────────
            for (h, a) in hidden_states.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // ─ Pre-FFN norm ────────────────────────────────────────────────────
            let mut ffn_normed = hidden_states.clone();
            for t in 0..seq_len {
                layer
                    .ffn_norm
                    .forward(&mut ffn_normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ FFN (dense or MoE) ──────────────────────────────────────────────
            let mut ffn_out = vec![0.0f32; seq_len * hidden];
            match &layer.ffn {
                FfnKind::Dense(dense) => {
                    let kernel_gate = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.gate.weight,
                        layer_idx,
                        "gate",
                    )?;
                    let kernel_up = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.up.weight,
                        layer_idx,
                        "up",
                    )?;
                    let kernel_down = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.down.weight,
                        layer_idx,
                        "down",
                    )?;

                    let intermediate = dense.gate.out_features;
                    let mut buf_gate = vec![0.0f32; intermediate];
                    let mut buf_up = vec![0.0f32; intermediate];
                    let mut buf_ffn = vec![0.0f32; hidden];

                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        dense
                            .gate
                            .forward(&*kernel_gate, x_t, &mut buf_gate)
                            .map_err(ArchError::from)?;
                        dense
                            .up
                            .forward(&*kernel_up, x_t, &mut buf_up)
                            .map_err(ArchError::from)?;
                        swiglu_inplace(&mut buf_gate, &buf_up);
                        dense
                            .down
                            .forward(&*kernel_down, &buf_gate, &mut buf_ffn)
                            .map_err(ArchError::from)?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&buf_ffn);
                    }
                }
                FfnKind::Moe { weights, config } => {
                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        let tok_out = moe_forward(x_t, weights, config).map_err(|e| {
                            ArchError::ForwardPassError {
                                layer: layer_idx,
                                message: format!("MoE: {e}"),
                            }
                        })?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&tok_out);
                    }
                }
            }

            // ─ Residual after FFN ──────────────────────────────────────────────
            for (h, f) in hidden_states.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
        }

        // ── Final norm + LM head (last token only) ──────────────────────────────
        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        let kernel = self
            .dispatcher
            .get_kernel(self.output.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut logits = vec![0.0f32; vocab];
        self.output
            .forward(&*kernel, last, &mut logits)
            .map_err(ArchError::from)?;

        self.current_pos += seq_len;

        Ok(logits)
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` through to and including `output_norm.forward(last)`.
    /// Does NOT call the LM-head projection (`self.output.forward`). Returns a
    /// `hidden_size`-dimensional vector suitable for semantic embedding use.
    fn embed(&mut self, tokens: &[u32], _kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
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
            let embd_off = tok * hidden;
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[embd_off..embd_off + hidden]);
        }

        let position = self.current_pos;

        // ── Transformer layers ──────────────────────────────────────────────────
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // ─ Pre-attention norm ──────────────────────────────────────────────
            let mut normed = vec![0.0f32; seq_len * hidden];
            for t in 0..seq_len {
                let src = &hidden_states[t * hidden..(t + 1) * hidden];
                let dst = &mut normed[t * hidden..(t + 1) * hidden];
                dst.copy_from_slice(src);
                layer
                    .attn_norm
                    .forward(&mut normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ MLA forward ────────────────────────────────────────────────────
            let attn_out = mla_forward(
                &normed,
                &layer.mla_weights,
                &layer.mla_config,
                &mut layer.mla_cache,
                position,
            )
            .map_err(|e| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("MLA: {e}"),
            })?;

            // ─ Residual connection ─────────────────────────────────────────────
            for (h, a) in hidden_states.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // ─ Pre-FFN norm ────────────────────────────────────────────────────
            let mut ffn_normed = hidden_states.clone();
            for t in 0..seq_len {
                layer
                    .ffn_norm
                    .forward(&mut ffn_normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ FFN (dense or MoE) ──────────────────────────────────────────────
            let mut ffn_out = vec![0.0f32; seq_len * hidden];
            match &layer.ffn {
                FfnKind::Dense(dense) => {
                    let kernel_gate = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.gate.weight,
                        layer_idx,
                        "gate",
                    )?;
                    let kernel_up = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.up.weight,
                        layer_idx,
                        "up",
                    )?;
                    let kernel_down = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.down.weight,
                        layer_idx,
                        "down",
                    )?;

                    let intermediate = dense.gate.out_features;
                    let mut buf_gate = vec![0.0f32; intermediate];
                    let mut buf_up = vec![0.0f32; intermediate];
                    let mut buf_ffn = vec![0.0f32; hidden];

                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        dense
                            .gate
                            .forward(&*kernel_gate, x_t, &mut buf_gate)
                            .map_err(ArchError::from)?;
                        dense
                            .up
                            .forward(&*kernel_up, x_t, &mut buf_up)
                            .map_err(ArchError::from)?;
                        swiglu_inplace(&mut buf_gate, &buf_up);
                        dense
                            .down
                            .forward(&*kernel_down, &buf_gate, &mut buf_ffn)
                            .map_err(ArchError::from)?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&buf_ffn);
                    }
                }
                FfnKind::Moe { weights, config } => {
                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        let tok_out = moe_forward(x_t, weights, config).map_err(|e| {
                            ArchError::ForwardPassError {
                                layer: layer_idx,
                                message: format!("MoE: {e}"),
                            }
                        })?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&tok_out);
                    }
                }
            }

            // ─ Residual after FFN ──────────────────────────────────────────────
            for (h, f) in hidden_states.iter_mut().zip(ffn_out.iter()) {
                *h += f;
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

// ─── Helper ───────────────────────────────────────────────────────────────────

fn layer_dispatcher_kernel(
    dispatcher: &KernelDispatcher,
    weight: &QuantTensor,
    layer_idx: usize,
    name: &str,
) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
    dispatcher
        .get_kernel(weight.tensor_type)
        .map_err(|e| ArchError::ForwardPassError {
            layer: layer_idx,
            message: format!("kernel dispatch for {name}: {e}"),
        })
}

// ─── GGUF loader helpers ──────────────────────────────────────────────────────

/// Dequantize tensor data to f32.
///
/// Handles F32 (direct copy), F16 (half-precision conversion), and all
/// quantized block formats via the kernel dispatcher.
fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
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

/// Load a tensor and dequantize it to f32, looking it up by name.
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
    let data = model
        .tensor_data(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    dequant_to_f32(info, data, dispatcher)
}

/// Load a quantized linear layer from GGUF by tensor name.
fn load_quant_linear(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model
        .tensor_data(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let tensor = QuantTensor::new(data.to_vec(), shape, info.tensor_type);
    Ok(QuantLinear::new(tensor, None))
}

/// Load an RMSNorm weight vector from GGUF (always dequantized to F32).
fn load_rms_norm_weight(model: &oxillama_gguf::GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let dispatcher = KernelDispatcher::new();
    load_dequant_tensor(model, &dispatcher, name)
}

/// Load MLA weights for one layer from GGUF.
///
/// Tensor name format: `blk.{layer}.attn_{qka_proj|qa_norm|qkb_proj|kva_proj|kva_norm|kvb_proj|output}.weight`
///
/// The `_proj` suffix convention is used first (matching GGUF v3 synthetic fixtures).
/// If a tensor is absent under the `_proj` name, we fall back to the short form
/// (e.g., `blk.0.attn_q_a.weight`) for compatibility with external converters.
fn load_mla_weights(
    model: &oxillama_gguf::GgufModel,
    mla_cfg: &crate::common::mla::MlaConfig,
    layer_idx: usize,
    max_seq: usize,
) -> ArchResult<MlaWeights> {
    let p = format!("blk.{layer_idx}");

    /// Try `primary` first; if missing, fall back to `fallback`.
    fn try_quant(
        model: &oxillama_gguf::GgufModel,
        primary: &str,
        fallback: &str,
    ) -> ArchResult<QuantLinear> {
        if model.file.tensors.contains(primary) {
            load_quant_linear(model, primary)
        } else {
            load_quant_linear(model, fallback)
        }
    }

    fn try_norm(
        model: &oxillama_gguf::GgufModel,
        primary: &str,
        fallback: &str,
        eps: f32,
    ) -> ArchResult<RmsNorm> {
        let name = if model.file.tensors.contains(primary) {
            primary
        } else {
            fallback
        };
        let w = load_rms_norm_weight(model, name)?;
        Ok(RmsNorm::new(w, eps))
    }

    let rms_eps = 1e-5f32;

    let w_q_a = try_quant(
        model,
        &format!("{p}.attn_q_a_proj.weight"),
        &format!("{p}.attn_q_a.weight"),
    )?;
    let q_a_norm = try_norm(
        model,
        &format!("{p}.attn_q_a_norm.weight"),
        &format!("{p}.attn_q_norm.weight"),
        rms_eps,
    )?;
    let w_q_b = try_quant(
        model,
        &format!("{p}.attn_q_b_proj.weight"),
        &format!("{p}.attn_q_b.weight"),
    )?;
    let w_kv_a = try_quant(
        model,
        &format!("{p}.attn_kv_a_proj.weight"),
        &format!("{p}.attn_kv_a.weight"),
    )?;
    let kv_a_norm = try_norm(
        model,
        &format!("{p}.attn_kv_a_norm.weight"),
        &format!("{p}.attn_kv_norm.weight"),
        rms_eps,
    )?;
    let w_kv_b = try_quant(
        model,
        &format!("{p}.attn_kv_b_proj.weight"),
        &format!("{p}.attn_kv_b.weight"),
    )?;
    let w_o = load_quant_linear(model, &format!("{p}.attn_output.weight"))?;

    let rope = crate::common::rope::RopeTable::new_standard(
        mla_cfg.qk_rope_head_dim,
        max_seq,
        mla_cfg.rope_theta,
    );

    Ok(MlaWeights {
        w_q_a,
        q_a_norm,
        w_q_b,
        w_kv_a,
        kv_a_norm,
        w_kv_b,
        w_o,
        rope,
    })
}

/// Load a single DeepSeek expert (gate/up/down projections) from GGUF.
fn load_expert(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &KernelDispatcher,
    name_prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
) -> ArchResult<crate::deepseek::moe::DeepSeekExpert> {
    let gate = load_dequant_tensor(model, dispatcher, &format!("{name_prefix}.ffn_gate.weight"))?;
    let up = load_dequant_tensor(model, dispatcher, &format!("{name_prefix}.ffn_up.weight"))?;
    let down = load_dequant_tensor(model, dispatcher, &format!("{name_prefix}.ffn_down.weight"))?;

    Ok(crate::deepseek::moe::DeepSeekExpert {
        gate,
        up,
        down,
        hidden_size,
        intermediate_size,
    })
}

/// Load dense FFN weights for one layer from GGUF.
fn load_dense_ffn(model: &oxillama_gguf::GgufModel, layer_idx: usize) -> ArchResult<DenseFfn> {
    let p = format!("blk.{layer_idx}");
    Ok(DenseFfn {
        gate: load_quant_linear(model, &format!("{p}.ffn_gate.weight"))?,
        up: load_quant_linear(model, &format!("{p}.ffn_up.weight"))?,
        down: load_quant_linear(model, &format!("{p}.ffn_down.weight"))?,
    })
}

/// Load MoE FFN weights for one layer from GGUF.
///
/// Builds:
/// - Router: `blk.{i}.ffn_gate_inp.weight`
/// - Routed experts: `blk.{i}.ffn_exp.{e}.ffn_{gate,up,down}.weight`
/// - Shared experts: `blk.{i}.ffn_shared_exp.{e}.ffn_{gate,up,down}.weight`
/// - Optional expert bias: `blk.{i}.exp_probs_b.weight` (SigmoidWithBias mode)
fn load_moe_ffn(
    model: &oxillama_gguf::GgufModel,
    dispatcher: &KernelDispatcher,
    ds_cfg: &DeepSeekConfig,
    layer_idx: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> ArchResult<(MoeWeights, MoeConfig)> {
    let p = format!("blk.{layer_idx}");

    // Router: [n_routed_experts, hidden_size]
    let router = load_dequant_tensor(model, dispatcher, &format!("{p}.ffn_gate_inp.weight"))?;

    // Routed experts
    let routed_experts: Vec<crate::deepseek::moe::DeepSeekExpert> = (0..ds_cfg.n_routed_experts)
        .map(|e| {
            load_expert(
                model,
                dispatcher,
                &format!("{p}.ffn_exp.{e}"),
                hidden_size,
                intermediate_size,
            )
        })
        .collect::<ArchResult<Vec<_>>>()?;

    // Shared experts
    let shared_experts: Vec<crate::deepseek::moe::DeepSeekExpert> = (0..ds_cfg.n_shared_experts)
        .map(|e| {
            load_expert(
                model,
                dispatcher,
                &format!("{p}.ffn_shared_exp.{e}"),
                hidden_size,
                ds_cfg.shared_expert_intermediate_size,
            )
        })
        .collect::<ArchResult<Vec<_>>>()?;

    // Optional per-expert bias (indicates SigmoidWithBias mode)
    let bias_name = format!("{p}.exp_probs_b.weight");
    let (expert_bias, scoring_mode) = if model.file.tensors.contains(&bias_name) {
        let bias = load_dequant_tensor(model, dispatcher, &bias_name)?;
        (
            Some(bias),
            crate::deepseek::moe::ScoringMode::SigmoidWithBias,
        )
    } else {
        (None, crate::deepseek::moe::ScoringMode::Softmax)
    };

    let moe_weights = MoeWeights {
        router,
        routed_experts,
        shared_experts,
        expert_bias,
    };

    let moe_config = MoeConfig {
        hidden_size,
        expert_intermediate_size: intermediate_size,
        n_shared_experts: ds_cfg.n_shared_experts,
        n_routed_experts: ds_cfg.n_routed_experts,
        top_k: ds_cfg.top_k_routed,
        routed_scaling_factor: ds_cfg.routed_scaling_factor,
        scoring_mode,
        shared_expert_intermediate_size: ds_cfg.shared_expert_intermediate_size,
    };

    Ok((moe_weights, moe_config))
}

// ─── GGUF loader ──────────────────────────────────────────────────────────────

/// Load a DeepSeek-V2/V3 model from a parsed GGUF file.
///
/// Reads all model hyperparameters from GGUF metadata, then loads each layer's
/// weights (MLA + FFN) and the output projection. All tensors are either kept
/// in their quantized form (`QuantLinear`) or dequantized to f32 (expert
/// weights, RMSNorm vectors, token embeddings).
///
/// # DeepSeek-V3 detection
/// If any layer contains `blk.{i}.exp_probs_b.weight`, the MoE router for that
/// layer switches to `ScoringMode::SigmoidWithBias` (V3 style). Otherwise the
/// standard `Softmax` routing (V2) is used.
///
/// # Dense vs. MoE layers
/// Layers `0..ds_config.first_k_dense_replace` use a dense SwiGLU FFN.
/// Layers `ds_config.first_k_dense_replace..num_layers` use sparse MoE FFN.
///
/// # Errors
/// Returns `ArchError::MissingTensor` if any required tensor is absent, or
/// other `ArchError` variants on shape mismatches or metadata errors.
pub fn load_deepseek_from_gguf(model: &oxillama_gguf::GgufModel) -> ArchResult<DeepSeekModel> {
    // ── Parse model configuration ─────────────────────────────────────────────
    let config = ModelConfig::from_metadata(&model.file.metadata)?;
    let ds_config = DeepSeekConfig::from_metadata(&model.file.metadata, config.hidden_size);

    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let num_layers = config.num_layers;
    let max_seq = config.max_context_length;

    let dispatcher = KernelDispatcher::new();

    // ── MLA configuration derived from metadata ────────────────────────────────
    let qk_head_dim = ds_config.qk_nope_head_dim + ds_config.qk_rope_head_dim;
    let softmax_scale = 1.0 / (qk_head_dim as f32).sqrt();
    let mla_cfg = crate::common::mla::MlaConfig {
        num_heads: config.num_attention_heads,
        q_lora_rank: ds_config.q_lora_rank,
        kv_lora_rank: ds_config.kv_lora_rank,
        qk_nope_head_dim: ds_config.qk_nope_head_dim,
        qk_rope_head_dim: ds_config.qk_rope_head_dim,
        v_head_dim: ds_config.v_head_dim,
        rope_theta: config.rope_freq_base,
        softmax_scale,
    };

    // ── Token embedding ───────────────────────────────────────────────────────
    let token_embd = load_dequant_tensor(model, &dispatcher, "token_embd.weight")?;

    // ── Transformer layers ────────────────────────────────────────────────────
    let first_k_dense = ds_config.first_k_dense_replace;

    let layers: Vec<DeepSeekLayer> = (0..num_layers)
        .map(|layer_idx| {
            let p = format!("blk.{layer_idx}");

            let attn_norm_w = load_rms_norm_weight(model, &format!("{p}.attn_norm.weight"))?;
            let attn_norm = RmsNorm::new(attn_norm_w, config.rms_norm_eps);

            let ffn_norm_w = load_rms_norm_weight(model, &format!("{p}.ffn_norm.weight"))?;
            let ffn_norm = RmsNorm::new(ffn_norm_w, config.rms_norm_eps);

            let mla_weights = load_mla_weights(model, &mla_cfg, layer_idx, max_seq)?;
            let mla_cache = crate::common::mla::MlaLatentCache::new(max_seq, &mla_cfg);

            let ffn = if layer_idx < first_k_dense {
                let dense = load_dense_ffn(model, layer_idx)?;
                FfnKind::Dense(Box::new(dense))
            } else {
                let (moe_weights, moe_config) = load_moe_ffn(
                    model,
                    &dispatcher,
                    &ds_config,
                    layer_idx,
                    hidden_size,
                    intermediate_size,
                )?;
                FfnKind::Moe {
                    weights: Box::new(moe_weights),
                    config: moe_config,
                }
            };

            Ok(DeepSeekLayer {
                attn_norm,
                mla_weights,
                mla_config: mla_cfg.clone(),
                mla_cache,
                ffn_norm,
                ffn,
            })
        })
        .collect::<ArchResult<Vec<_>>>()?;

    // ── Output norm + LM head ─────────────────────────────────────────────────
    let output_norm_w = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_w, config.rms_norm_eps);

    let output = load_quant_linear(model, "output.weight")?;

    Ok(DeepSeekModel::new(
        config,
        ds_config,
        token_embd,
        layers,
        output_norm,
        output,
    ))
}

/// Construct a `DeepSeekModel` from raw weights (intended for tests and
/// custom loaders that have already materialised the tensors).
///
/// All shape checks are deferred to the constituent layer/weight constructors.
pub fn build_deepseek_model(
    config: ModelConfig,
    ds_config: DeepSeekConfig,
    token_embd: Vec<f32>,
    layers: Vec<DeepSeekLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> DeepSeekModel {
    DeepSeekModel::new(config, ds_config, token_embd, layers, output_norm, output)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::mla::{MlaConfig, MlaLatentCache, MlaWeights};
    use crate::common::rms_norm::RmsNorm;
    use crate::common::rope::RopeTable;
    use crate::deepseek::moe::{DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode};
    use crate::error::ArchResult;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    /// Minimal LCG (same as in mla.rs tests, no dependency on test infrastructure).
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
            let bits = mantissa | 0x3f80_0000u32;
            (f32::from_bits(bits) - 1.5) * 0.02
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    fn rand_f32_tensor(lcg: &mut Lcg, rows: usize, cols: usize) -> QuantTensor {
        let n = rows * cols;
        let mut vals = vec![0.0f32; n];
        lcg.fill(&mut vals);
        let mut data = Vec::with_capacity(n * 4);
        for &v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
    }

    fn build_test_model(lcg: &mut Lcg) -> DeepSeekModel {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const INTERMEDIATE: usize = 32;
        const N_LAYERS: usize = 2;
        const MAX_SEQ: usize = 64;

        let mla_cfg = MlaConfig {
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            rope_theta: 10000.0,
            softmax_scale: 1.0 / (8.0f32).sqrt(),
        };

        let build_mla_weights = |lcg: &mut Lcg| -> MlaWeights {
            let q_full = mla_cfg.q_full_dim();
            let kv_comb = mla_cfg.kv_combined_dim();
            let kv_b_full = mla_cfg.kv_b_full_dim();
            let attn_out = mla_cfg.attn_out_dim();
            MlaWeights {
                w_q_a: QuantLinear::new(rand_f32_tensor(lcg, mla_cfg.q_lora_rank, H), None),
                q_a_norm: RmsNorm::new(
                    {
                        let mut w = vec![0.0f32; mla_cfg.q_lora_rank];
                        lcg.fill(&mut w);
                        w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                        w
                    },
                    1e-5,
                ),
                w_q_b: QuantLinear::new(rand_f32_tensor(lcg, q_full, mla_cfg.q_lora_rank), None),
                w_kv_a: QuantLinear::new(rand_f32_tensor(lcg, kv_comb, H), None),
                kv_a_norm: RmsNorm::new(
                    {
                        let mut w = vec![0.0f32; mla_cfg.kv_lora_rank];
                        lcg.fill(&mut w);
                        w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                        w
                    },
                    1e-5,
                ),
                w_kv_b: QuantLinear::new(
                    rand_f32_tensor(lcg, kv_b_full, mla_cfg.kv_lora_rank),
                    None,
                ),
                w_o: QuantLinear::new(rand_f32_tensor(lcg, H, attn_out), None),
                rope: RopeTable::new_standard(
                    mla_cfg.qk_rope_head_dim,
                    MAX_SEQ,
                    mla_cfg.rope_theta,
                ),
            }
        };

        let build_dense_ffn = |lcg: &mut Lcg| -> DenseFfn {
            DenseFfn {
                gate: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, H), None),
                up: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, H), None),
                down: QuantLinear::new(rand_f32_tensor(lcg, H, INTERMEDIATE), None),
            }
        };

        let build_moe_ffn = |lcg: &mut Lcg| -> (MoeWeights, MoeConfig) {
            const N_EXPERTS: usize = 4;
            const TOP_K: usize = 2;
            let moe_cfg = MoeConfig {
                hidden_size: H,
                expert_intermediate_size: 16,
                n_shared_experts: 1,
                n_routed_experts: N_EXPERTS,
                top_k: TOP_K,
                routed_scaling_factor: 1.0,
                scoring_mode: ScoringMode::Softmax,
                shared_expert_intermediate_size: 16,
            };
            let make_expert = |lcg: &mut Lcg, inter: usize| -> DeepSeekExpert {
                let mut gate = vec![0.0f32; inter * H];
                let mut up = vec![0.0f32; inter * H];
                let mut down = vec![0.0f32; H * inter];
                lcg.fill(&mut gate);
                lcg.fill(&mut up);
                lcg.fill(&mut down);
                DeepSeekExpert {
                    gate,
                    up,
                    down,
                    hidden_size: H,
                    intermediate_size: inter,
                }
            };
            let mut router = vec![0.0f32; N_EXPERTS * H];
            lcg.fill(&mut router);
            let moe_weights = MoeWeights {
                router,
                routed_experts: (0..N_EXPERTS).map(|_| make_expert(lcg, 16)).collect(),
                shared_experts: vec![make_expert(lcg, 16)],
                expert_bias: None,
            };
            (moe_weights, moe_cfg)
        };

        let mut token_embd = vec![0.0f32; VOCAB * H];
        lcg.fill(&mut token_embd);

        let mut norm_w = vec![0.0f32; H];
        lcg.fill(&mut norm_w);
        norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);

        let ds_config = DeepSeekConfig {
            q_lora_rank: mla_cfg.q_lora_rank,
            kv_lora_rank: mla_cfg.kv_lora_rank,
            qk_nope_head_dim: mla_cfg.qk_nope_head_dim,
            qk_rope_head_dim: mla_cfg.qk_rope_head_dim,
            v_head_dim: mla_cfg.v_head_dim,
            n_shared_experts: 1,
            n_routed_experts: 4,
            top_k_routed: 2,
            shared_expert_intermediate_size: 16,
            routed_scaling_factor: 1.0,
            first_k_dense_replace: N_DENSE_LAYERS,
        };

        let model_config = ModelConfig {
            architecture: "deepseek2".to_string(),
            model_name: "test-deepseek".to_string(),
            hidden_size: H,
            intermediate_size: INTERMEDIATE,
            num_layers: N_LAYERS,
            num_attention_heads: mla_cfg.num_heads,
            num_kv_heads: mla_cfg.num_heads,
            head_dim: mla_cfg.qk_head_dim(),
            vocab_size: VOCAB,
            max_context_length: MAX_SEQ,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|idx| {
                let mut attn_norm_w = vec![0.0f32; H];
                lcg.fill(&mut attn_norm_w);
                attn_norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                let mut ffn_norm_w = vec![0.0f32; H];
                lcg.fill(&mut ffn_norm_w);
                ffn_norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);

                let ffn = if idx < N_DENSE_LAYERS {
                    FfnKind::Dense(Box::new(build_dense_ffn(lcg)))
                } else {
                    let (moe_weights, moe_cfg) = build_moe_ffn(lcg);
                    FfnKind::Moe {
                        weights: Box::new(moe_weights),
                        config: moe_cfg,
                    }
                };

                DeepSeekLayer {
                    attn_norm: RmsNorm::new(attn_norm_w, 1e-5),
                    mla_weights: build_mla_weights(lcg),
                    mla_config: mla_cfg.clone(),
                    mla_cache: MlaLatentCache::new(MAX_SEQ, &mla_cfg),
                    ffn_norm: RmsNorm::new(ffn_norm_w, 1e-5),
                    ffn,
                }
            })
            .collect();

        let output_norm = RmsNorm::new(norm_w, 1e-5);
        let output = QuantLinear::new(rand_f32_tensor(lcg, VOCAB, H), None);

        DeepSeekModel::new(
            model_config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        )
    }

    /// Forward pass produces logits of the correct shape (vocab_size).
    #[test]
    fn forward_shape() {
        let mut lcg = Lcg::new(1234);
        let mut model = build_test_model(&mut lcg);

        // Stub KV cache (MLA doesn't use it; the trait methods return Results)
        struct NullKv;
        impl KvCacheAccess for NullKv {
            fn seq_len(&self) -> usize {
                0
            }
            fn store_kv(
                &mut self,
                _layer: usize,
                _keys: &[f32],
                _values: &[f32],
            ) -> ArchResult<()> {
                Ok(())
            }
            fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn advance(&mut self) {}
        }

        let mut kv = NullKv;
        let out = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed");
        assert_eq!(
            out.len(),
            32,
            "logits must have vocab_size=32 elements, got {}",
            out.len()
        );
    }

    /// Forward pass is deterministic (same tokens → same logits after reset).
    #[test]
    fn forward_determinism() {
        let mut lcg = Lcg::new(9999);
        let mut model = build_test_model(&mut lcg);

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

        let mut kv = NullKv;
        let out1 = model.forward(&[1u32, 0], &mut kv).expect("first forward");

        model.reset_position();
        let mut kv2 = NullKv;
        let out2 = model.forward(&[1u32, 0], &mut kv2).expect("second forward");

        assert_eq!(out1.len(), out2.len());
        for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            assert_eq!(
                a_bits, b_bits,
                "logits[{i}] must be bit-for-bit identical: {a} vs {b}"
            );
        }
    }

    /// embed() returns a vector of length hidden_size (not vocab_size).
    #[test]
    fn embed_returns_hidden_size() {
        let mut lcg = Lcg::new(5678);
        let mut model = build_test_model(&mut lcg);

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

        let mut kv = NullKv;
        let embedding = model.embed(&[1u32], &mut kv).expect("embed must succeed");
        assert_eq!(
            embedding.len(),
            16,
            "embed output must have hidden_size=16 elements, got {}",
            embedding.len()
        );
    }

    /// embed() output is all finite.
    #[test]
    fn embed_all_finite() {
        let mut lcg = Lcg::new(1111);
        let mut model = build_test_model(&mut lcg);

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

        let mut kv = NullKv;
        let embedding = model.embed(&[0u32], &mut kv).expect("embed must succeed");
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "all embedding values must be finite"
        );
    }

    /// embed() errors on empty token slice.
    #[test]
    fn embed_empty_tokens_returns_error() {
        let mut lcg = Lcg::new(2222);
        let mut model = build_test_model(&mut lcg);

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

        let mut kv = NullKv;
        let result = model.embed(&[], &mut kv);
        assert!(
            result.is_err(),
            "embed with empty token sequence must return an error"
        );
    }

    // ── GGUF loader tests ─────────────────────────────────────────────────────

    /// Load the minimal V2 DeepSeek GGUF (all-dense layout) without error.
    ///
    /// The V2 fixture sets `block_count=1` with `leading_dense_block_count`
    /// defaulting to 1, so the single layer uses a dense SwiGLU FFN.
    #[test]
    fn deepseek_loader_round_trip_dense() {
        let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes)
            .expect("synthetic DeepSeek V2 GGUF must parse");

        let model = load_deepseek_from_gguf(&gguf)
            .expect("load_deepseek_from_gguf must succeed for V2 dense fixture");

        assert_eq!(model.config.num_layers, 1, "1-layer model");
        assert_eq!(model.layers.len(), 1, "one DeepSeekLayer loaded");
        assert!(
            matches!(model.layers[0].ffn, FfnKind::Dense(_)),
            "layer 0 must be Dense (first_k_dense_replace defaults to 1)"
        );
        assert_eq!(model.config.vocab_size, 32);
        assert_eq!(model.config.hidden_size, 32);
    }

    /// Load the minimal V3 DeepSeek GGUF (all-MoE layout) without error.
    ///
    /// The V3 fixture sets `leading_dense_block_count=0`, so the single layer
    /// uses the sparse MoE FFN with `SigmoidWithBias` routing (exp_probs_b present).
    #[test]
    fn deepseek_loader_round_trip_moe() {
        let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_v3_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes)
            .expect("synthetic DeepSeek V3 GGUF must parse");

        let model = load_deepseek_from_gguf(&gguf)
            .expect("load_deepseek_from_gguf must succeed for V3 MoE fixture");

        assert_eq!(model.config.num_layers, 1, "1-layer model");
        assert_eq!(model.layers.len(), 1, "one DeepSeekLayer loaded");

        match &model.layers[0].ffn {
            FfnKind::Moe { weights, config } => {
                assert_eq!(
                    config.n_routed_experts, 2,
                    "n_routed_experts must match fixture"
                );
                assert_eq!(
                    config.n_shared_experts, 1,
                    "n_shared_experts must match fixture"
                );
                assert!(
                    weights.expert_bias.is_some(),
                    "V3 layer must have expert_bias (SigmoidWithBias mode)"
                );
                assert_eq!(
                    config.scoring_mode,
                    crate::deepseek::moe::ScoringMode::SigmoidWithBias,
                    "V3 layer must use SigmoidWithBias scoring"
                );
            }
            FfnKind::Dense(_) => panic!("layer 0 must be MoE for V3 fixture"),
        }
    }

    /// Forward pass on a GGUF-loaded model produces finite logits.
    ///
    /// Uses the V2 (dense) fixture for simplicity; the V3 MoE path is covered
    /// separately in `deepseek_loader_round_trip_moe`.
    #[test]
    fn deepseek_loader_forward_no_nan() {
        struct NullKvL;
        impl KvCacheAccess for NullKvL {
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

        let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes)
            .expect("synthetic DeepSeek V2 GGUF must parse");

        let mut model =
            load_deepseek_from_gguf(&gguf).expect("load_deepseek_from_gguf must succeed");

        let mut kv = NullKvL;
        let logits = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed on GGUF-loaded model");

        assert_eq!(
            logits.len(),
            model.config.vocab_size,
            "logits length must equal vocab_size"
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite (no NaN or Inf)"
        );
    }
}
