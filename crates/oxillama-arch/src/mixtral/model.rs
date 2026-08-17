//! Mixtral transformer forward pass implementation.
//!
//! Mixtral is a Sparse Mixture-of-Experts (SMoE) variant of Mistral: LLaMA
//! topology throughout, with every dense FFN replaced by a pool of expert SwiGLU
//! FFNs of which a learned router activates the top 2 per token.
//!
//! ```text
//! embedding → N×(RMSNorm → GQA+RoPE → residual → RMSNorm → MoE-FFN → residual) → RMSNorm → LM head
//! ```
//!
//! # RoPE convention
//!
//! **NORM**, i.e. consecutive pairs — the same as LLaMA, *not* NeoX.
//! `convert_hf_to_gguf.py` registers `MixtralForCausalLM` on its `LlamaModel`
//! converter (`model_arch = gguf.MODEL_ARCH.LLAMA`, `undo_permute = True`), and
//! `llama_model_rope_type` maps `LLM_ARCH_LLAMA` to `LLAMA_ROPE_TYPE_NORM`
//! (`src/llama-model.cpp`).  See `crate::llama`'s `rope_norm` module.
//!
//! # How a real Mixtral checkpoint is routed
//!
//! It is **not** routed here.  Because the converter writes
//! `general.architecture = "llama"`, a stock `Mixtral-8x7B-*.gguf` is loaded by
//! [`crate::llama::load_llama_from_gguf`], which builds
//! [`FfnVariant::Moe`][crate::llama::FfnVariant::Moe] whenever
//! `expert_count > 0`.  That is precisely why the placeholder this file used to
//! contain — `Q = K = V = norm(hidden)`, no attention weights at all, no
//! `RopeTable`, no loader — went unnoticed: nothing ever reached it, yet
//! `MixtralModel` was `pub`, implemented `ForwardPass`, returned finite logits
//! and reported no error.
//!
//! This module now implements the block for real and ships
//! [`load_mixtral_from_gguf`] for GGUFs that *do* declare
//! `general.architecture = "mixtral"` (third-party converters and re-quantizers
//! emit those), so the registry entry is no longer a lie.
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `blk.{i}.attn_{norm,q,k,v,output}.weight` — as LLaMA
//! - `blk.{i}.ffn_gate_inp.weight` — router `[num_experts, hidden_size]`
//! - `blk.{i}.ffn_gate_exps.weight` — all expert gate projections (stacked)
//! - `blk.{i}.ffn_up_exps.weight`   — all expert up projections (stacked)
//! - `blk.{i}.ffn_down_exps.weight` — all expert down projections (stacked)

use std::sync::Arc;

use oxillama_quant::{
    quantize_activations_q8_0_into, KernelDispatcher, QuantKernel, Q8_0_ACT_BLOCK_BYTES,
};

use crate::common::linear::QuantLinear;
use crate::common::moe::{MoeScratch, QuantMoeFfn};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::{
    apply_rope_norm, axpy_f32, dot_f32, load_lm_head, load_quant_linear, load_quant_moe,
    load_rms_norm_weight, load_token_embedding, resolve_kernel, softmax_inplace, TokenEmbedding,
};
use crate::traits::{ForwardPass, KvCacheAccess};

/// Configuration specific to Mixtral's MoE FFN blocks.
#[derive(Debug, Clone)]
pub struct MixtralMoeConfig {
    /// Total number of experts in the pool (default 8 for Mixtral-8x7B).
    pub num_experts: usize,
    /// Number of experts activated per token (default 2 for Mixtral).
    pub num_experts_used: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Intermediate size for each individual expert FFN.
    pub intermediate_size: usize,
}

impl MixtralMoeConfig {
    /// Extract Mixtral MoE config from `ModelConfig`.
    ///
    /// Falls back to 8 total experts / 2 used if not set in metadata.
    pub fn from_model_config(config: &ModelConfig) -> Self {
        let num_experts = if config.num_experts > 0 {
            config.num_experts
        } else {
            8
        };
        let num_experts_used = if config.num_experts_used > 0 {
            config.num_experts_used
        } else {
            2
        };
        Self {
            num_experts,
            num_experts_used,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        }
    }
}

/// A single Mixtral transformer layer: GQA attention plus a sparse MoE FFN.
pub struct MixtralLayer {
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
    /// Kernel for [`Self::attn_q`], resolved once at load time.
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`], resolved once at load time.
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`], resolved once at load time.
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`], resolved once at load time.
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// Sparse MoE FFN (top-2-of-8 by default), experts left quantized.
    pub moe_ffn: QuantMoeFfn,
}

/// Complete Mixtral model.
pub struct MixtralModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// MoE configuration derived from the model config.
    pub moe_config: MixtralMoeConfig,
    /// Sliding window size (`None` = full causal).
    pub sliding_window: Option<usize>,
    /// Token embedding table, dequantized one row per lookup.
    pub token_embd: TokenEmbedding,
    /// Transformer layers.
    pub layers: Vec<MixtralLayer>,
    /// Final RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Resolved kernel for [`Self::output`].
    output_kernel: Arc<dyn QuantKernel>,
    /// RoPE table, applied with the NORM convention (see the module docs).
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated attention heads, `[num_heads * head_dim]` — not `hidden_size`.
    buf_attn_out: Vec<f32>,
    buf_proj: Vec<f32>,
    buf_moe_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_acts_q8: Vec<u8>,
    moe_scratch: MoeScratch,
}

impl MixtralModel {
    /// Create a new `MixtralModel` from pre-loaded weights.
    ///
    /// Fails if `output`'s tensor type has no registered [`QuantKernel`].
    pub fn new(
        config: ModelConfig,
        token_embd: TokenEmbedding,
        layers: Vec<MixtralLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let intermediate_size = config.intermediate_size;
        let sliding_window = config.sliding_window;
        let moe_config = MixtralMoeConfig::from_model_config(&config);
        let attn_dim = num_heads * head_dim;
        let num_experts = moe_config.num_experts;

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
            moe_config,
            sliding_window,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            dispatcher,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; attn_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; attn_dim],
            buf_proj: vec![0.0f32; hidden_size],
            buf_moe_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_acts_q8: Vec::with_capacity(
                hidden_size
                    .max(attn_dim)
                    .max(intermediate_size)
                    .div_ceil(256)
                    * 8
                    * Q8_0_ACT_BLOCK_BYTES,
            ),
            moe_scratch: MoeScratch::new(hidden_size, intermediate_size, num_experts),
        })
    }

    /// Load the residual stream with `token`'s embedding row.
    ///
    /// An out-of-vocabulary id is an error, not a panic.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let Self {
            token_embd,
            dispatcher,
            buf_hidden,
            ..
        } = self;
        token_embd.row_into(dispatcher, token, buf_hidden)
    }

    /// Grouped-query attention with RoPE and an optional sliding window.
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
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = position + 1;

        let layer = &self.layers[layer_idx];
        let q_kernel: &dyn QuantKernel = &*layer.attn_q_kernel;
        let k_kernel: &dyn QuantKernel = &*layer.attn_k_kernel;
        let v_kernel: &dyn QuantKernel = &*layer.attn_v_kernel;

        // Q/K/V share the post-attn-norm activation: quantize it once and let
        // all three GEMVs read the same Q8_0 image.
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

        // RoPE, NORM convention — Mixtral ships as arch "llama".
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

        // Sliding-window attention: only the last `w` positions are visible.
        let window_start = match self.sliding_window {
            Some(w) if w > 0 => seq_len.saturating_sub(w),
            _ => 0,
        };
        let window_len = seq_len - window_start;

        // One task per head; a head's output is a contiguous `head_dim` slice
        // that no other head touches, so threading it changes no accumulation
        // order.  The per-head score vector is the worker-local `init` state.
        {
            let q: &[f32] = &self.buf_q;
            oxillama_quant::parallel::for_each_chunk_init(
                &mut self.buf_attn_out[..attn_dim],
                head_dim,
                window_len * 2,
                Vec::<f32>::new,
                |scores, h, out_head| {
                    let kv_head = h / heads_per_kv;
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];

                    scores.clear();
                    scores.resize(window_len, 0.0);
                    for (i, score) in scores.iter_mut().enumerate() {
                        let off = (window_start + i) * kv_dim + kv_head * head_dim;
                        *score = dot_f32(q_head, &cached_keys[off..off + head_dim]) * scale;
                    }

                    softmax_inplace(scores);

                    out_head.fill(0.0);
                    for (i, &w) in scores.iter().enumerate() {
                        let off = (window_start + i) * kv_dim + kv_head * head_dim;
                        axpy_f32(out_head, w, &cached_values[off..off + head_dim]);
                    }
                },
            );
        }

        let layer = &self.layers[layer_idx];
        let o_kernel: &dyn QuantKernel = &*layer.attn_output_kernel;
        match layer.attn_output.q8_fused_blocks(o_kernel) {
            Some(n_blocks) => {
                quantize_activations_q8_0_into(&self.buf_attn_out, n_blocks, &mut self.buf_acts_q8);
                layer.attn_output.forward_q8_fused(
                    o_kernel,
                    &self.buf_attn_out,
                    &self.buf_acts_q8,
                    &mut self.buf_proj,
                )?;
            }
            None => layer
                .attn_output
                .forward(o_kernel, &self.buf_attn_out, &mut self.buf_proj)?,
        }

        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj.iter()) {
            *h += p;
        }
        Ok(())
    }

    /// Run the sparse MoE FFN for one layer and add it to the residual.
    fn moe_feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        self.layers[layer_idx].moe_ffn.forward(
            &self.buf_norm,
            &mut self.buf_moe_out,
            &mut self.moe_scratch,
        )?;
        for (h, &m) in self.buf_hidden.iter_mut().zip(self.buf_moe_out.iter()) {
            *h += m;
        }
        Ok(())
    }

    /// Run every token through all layers, leaving the last token's
    /// pre-output-norm hidden state in `buf_hidden`.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        let start_pos = kv_cache.seq_len();
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

                self.moe_feed_forward(layer_idx)?;
            }
            // Exactly once per token, after every layer has written its K/V.
            kv_cache.advance();
        }
        Ok(())
    }
}

impl ForwardPass for MixtralModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);

        // `buf_logits` was handed to the caller by `mem::take` last call and is
        // empty; restore its length before the LM head writes into it.
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

        Ok(std::mem::take(&mut self.buf_logits))
    }

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

    fn swa_config(&self) -> Option<(u32, bool)> {
        self.config.swa_window.map(|w| (w, false))
    }
}

/// Load a Mixtral model from a `GgufModel`.
///
/// Reads the same tensor set [`crate::llama::load_llama_from_gguf`] does for an
/// MoE checkpoint — per-layer attention projections plus the stacked `ffn_*_exps`
/// expert tensors — through the same helpers, so a GGUF that declares
/// `general.architecture = "mixtral"` and one that declares `"llama"` with
/// `expert_count > 0` produce identical weights.
pub fn load_mixtral_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<MixtralModel> {
    let dispatcher = KernelDispatcher::new();
    let moe_config = MixtralMoeConfig::from_model_config(config);

    let token_embd = load_token_embedding(model, config, &dispatcher)?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let moe_ffn = load_quant_moe(
            model,
            &prefix,
            moe_config.num_experts,
            moe_config.num_experts_used.max(1),
        )?;

        layers.push(MixtralLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q_kernel: resolve_kernel(&dispatcher, &attn_q)?,
            attn_k_kernel: resolve_kernel(&dispatcher, &attn_k)?,
            attn_v_kernel: resolve_kernel(&dispatcher, &attn_v)?,
            attn_output_kernel: resolve_kernel(&dispatcher, &attn_output)?,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            moe_ffn,
        });
    }

    let output_norm = RmsNorm::new(
        load_rms_norm_weight(model, "output_norm.weight")?,
        config.rms_norm_eps,
    );
    let output = load_lm_head(model)?;

    MixtralModel::new(config.clone(), token_embd, layers, output_norm, output)
}
