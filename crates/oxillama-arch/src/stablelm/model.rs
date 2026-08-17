//! StableLM transformer forward pass implementation.
//!
//! Every structural decision here is taken from
//! `~/work/refs/llama.cpp/src/models/stablelm.cpp` (`llm_build_stablelm`) and
//! `~/work/refs/llama.cpp/src/llama-model.cpp` (`case LLM_ARCH_STABLELM:` in
//! `load_tensors()`).  StableLM departs from LLaMA/Mistral in four ways:
//!
//! 1. **LayerNorm, not RMSNorm.** `build_norm(..., LLM_NORM, ...)` — mean
//!    centred, with a learned bias — is used for `attn_norm`, `ffn_norm` and
//!    `output_norm`, all of which ship a `.bias` tensor.  The epsilon comes
//!    from `{arch}.attention.layer_norm_epsilon`.
//!
//! 2. **Partial RoPE.** The rotation covers `hparams.n_rot` dimensions, taken
//!    from `{arch}.rope.dimension_count`, which is **not** `head_dim` for this
//!    family.  The pairing is GPT-NeoX (`LLAMA_ROPE_TYPE_NEOX`), i.e.
//!    `(x[i], x[i + n_rot/2])` — note `n_rot/2`, not `head_dim/2`.
//!
//! 3. **Two residual topologies.**  With `ffn_norm` present (the common case,
//!    every StableLM except StableLM 2 12B) the block is the ordinary
//!    sequential one.  Without it the block is *parallel*, and the FFN reads
//!    `inpSA` — the output of `attn_norm` — not a second norm of the residual:
//!
//!    ```text
//!    cur     = LN(inpL,  attn_norm)      // inpSA
//!    attn    = Attention(cur)
//!    ffn_inp = attn + inpL               // residual add, both topologies
//!    ffn_in  = ffn_norm ? LN(ffn_inp, ffn_norm) : inpSA
//!    out     = FFN(ffn_in) + ffn_inp
//!    ```
//!
//! 4. **Optional per-head QK LayerNorm.**  See
//!    [`PerHeadLayerNorm`](super::head_norm::PerHeadLayerNorm): the weight is
//!    `{head_dim, n_head}`, one distinct vector per head.
//!
//! Q/K/V are three separate projections (never fused into `attn_qkv`) and each
//! carries an **optional** bias (`bq`/`bk`/`bv`, present in Stable LM 2 1.6B).
//! `attn_output` has no bias tensor at all.  The FFN is SwiGLU
//! (`LLM_FFN_SILU, LLM_FFN_PAR` over `ffn_gate`/`ffn_up`/`ffn_down`).
//!
//! ## Tensor names (GGUF)
//!
//! | Name | Required |
//! |------|----------|
//! | `token_embd.weight` | yes |
//! | `output_norm.weight` / `.bias` | yes (both) |
//! | `output.weight` | yes |
//! | `blk.{i}.attn_norm.weight` / `.bias` | yes (both) |
//! | `blk.{i}.attn_q.weight`, `attn_k.weight`, `attn_v.weight` | yes |
//! | `blk.{i}.attn_q.bias`, `attn_k.bias`, `attn_v.bias` | no |
//! | `blk.{i}.attn_output.weight` | yes |
//! | `blk.{i}.attn_q_norm.weight`, `attn_k_norm.weight` | no |
//! | `blk.{i}.ffn_norm.weight`, `ffn_norm.bias` | no (independently) |
//! | `blk.{i}.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` | yes |

use std::sync::Arc;

use oxillama_quant::{quantize_activations_q8_0_into, KernelDispatcher, QuantKernel};

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};

use super::config::StablelmConfig;
use super::head_norm::PerHeadLayerNorm;

/// The weights of one StableLM block, before kernel resolution.
///
/// Passed to [`StablelmLayer::new`], which resolves each projection's
/// quantization kernel **once** and stores it on the layer.  Re-dispatching
/// per token — as the pre-loader implementation effectively did by carrying
/// dense `Vec<f32>` weights and a hand-rolled GEMV — costs an allocation per
/// projection per token.
pub struct StablelmLayerWeights {
    /// Pre-attention LayerNorm (`attn_norm.weight` + `attn_norm.bias`).
    pub attn_norm: LayerNorm,
    /// Q projection; `bias` carries the optional `attn_q.bias`.
    pub attn_q: QuantLinear,
    /// K projection; `bias` carries the optional `attn_k.bias`.
    pub attn_k: QuantLinear,
    /// V projection; `bias` carries the optional `attn_v.bias`.
    pub attn_v: QuantLinear,
    /// Attention output projection (never has a bias in this architecture).
    pub attn_output: QuantLinear,
    /// Optional per-head Q LayerNorm (`attn_q_norm.weight`, StableLM 2 12B).
    pub attn_q_norm: Option<PerHeadLayerNorm>,
    /// Optional per-head K LayerNorm (`attn_k_norm.weight`, StableLM 2 12B).
    pub attn_k_norm: Option<PerHeadLayerNorm>,
    /// Optional pre-FFN LayerNorm.  `None` selects the parallel-residual
    /// topology, in which the FFN reads `attn_norm`'s output instead.
    pub ffn_norm: Option<LayerNorm>,
    /// SwiGLU gate projection.
    pub ffn_gate: QuantLinear,
    /// SwiGLU up projection.
    pub ffn_up: QuantLinear,
    /// SwiGLU down projection.
    pub ffn_down: QuantLinear,
}

/// A single StableLM transformer layer.
pub struct StablelmLayer {
    /// Pre-attention LayerNorm (with bias).
    pub attn_norm: LayerNorm,
    /// Q projection `[num_heads * head_dim, hidden_size]` with optional bias.
    pub attn_q: QuantLinear,
    /// K projection `[num_kv_heads * head_dim, hidden_size]` with optional bias.
    pub attn_k: QuantLinear,
    /// V projection `[num_kv_heads * head_dim, hidden_size]` with optional bias.
    pub attn_v: QuantLinear,
    /// Attention output projection `[hidden_size, num_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// Optional per-head Q LayerNorm.
    pub attn_q_norm: Option<PerHeadLayerNorm>,
    /// Optional per-head K LayerNorm.
    pub attn_k_norm: Option<PerHeadLayerNorm>,
    /// Optional pre-FFN LayerNorm (`None` ⇒ parallel residual).
    pub ffn_norm: Option<LayerNorm>,
    /// SwiGLU gate projection `[intermediate_size, hidden_size]`.
    pub ffn_gate: QuantLinear,
    /// SwiGLU up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// SwiGLU down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,

    /// Kernel for [`Self::attn_q`], resolved once at load time.
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`], resolved once at load time.
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`], resolved once at load time.
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`], resolved once at load time.
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_gate`], resolved once at load time.
    pub ffn_gate_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_up`], resolved once at load time.
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_down`], resolved once at load time.
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

impl StablelmLayer {
    /// Resolve every projection's kernel and assemble the layer.
    ///
    /// # Errors
    ///
    /// [`ArchError::Quant`] when a weight's tensor type has no registered
    /// dequantization kernel.
    pub fn new(dispatcher: &KernelDispatcher, w: StablelmLayerWeights) -> ArchResult<Self> {
        let attn_q_kernel = resolve_kernel(dispatcher, &w.attn_q)?;
        let attn_k_kernel = resolve_kernel(dispatcher, &w.attn_k)?;
        let attn_v_kernel = resolve_kernel(dispatcher, &w.attn_v)?;
        let attn_output_kernel = resolve_kernel(dispatcher, &w.attn_output)?;
        let ffn_gate_kernel = resolve_kernel(dispatcher, &w.ffn_gate)?;
        let ffn_up_kernel = resolve_kernel(dispatcher, &w.ffn_up)?;
        let ffn_down_kernel = resolve_kernel(dispatcher, &w.ffn_down)?;

        Ok(Self {
            attn_norm: w.attn_norm,
            attn_q: w.attn_q,
            attn_k: w.attn_k,
            attn_v: w.attn_v,
            attn_output: w.attn_output,
            attn_q_norm: w.attn_q_norm,
            attn_k_norm: w.attn_k_norm,
            ffn_norm: w.ffn_norm,
            ffn_gate: w.ffn_gate,
            ffn_up: w.ffn_up,
            ffn_down: w.ffn_down,
            attn_q_kernel,
            attn_k_kernel,
            attn_v_kernel,
            attn_output_kernel,
            ffn_gate_kernel,
            ffn_up_kernel,
            ffn_down_kernel,
        })
    }

    /// Whether this block uses the parallel-residual topology.
    ///
    /// True exactly when the checkpoint shipped no `ffn_norm.weight`, which is
    /// llama.cpp's own discriminator (`if (model.layers[il].ffn_norm) { … }
    /// else { /* parallel residual */ cur = inpSA; }`).
    pub fn is_parallel_residual(&self) -> bool {
        self.ffn_norm.is_none()
    }
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Run one projection, preferring the fused Q8_0 activation path when the
/// kernel offers one and the caller has already quantized the activations.
fn run_linear(
    linear: &QuantLinear,
    kernel: &dyn QuantKernel,
    input: &[f32],
    acts_q8: &[u8],
    output: &mut [f32],
) -> ArchResult<()> {
    if !acts_q8.is_empty() && linear.q8_fused_blocks(kernel).is_some() {
        linear.forward_q8_fused(kernel, input, acts_q8, output)?;
    } else {
        linear.forward(kernel, input, output)?;
    }
    Ok(())
}

/// Complete StableLM model.
pub struct StablelmModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// StableLM-specific configuration.
    pub stablelm_config: StablelmConfig,
    /// Token embeddings `[vocab_size, hidden_size]` (f32, row-major).
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<StablelmLayer>,
    /// Final LayerNorm (with bias).
    pub output_norm: LayerNorm,
    /// LM head (`output.weight`), kept in its GGUF quantization.
    pub output: QuantLinear,
    /// Kernel for [`Self::output`], resolved once at load time.
    pub output_kernel: Arc<dyn QuantKernel>,
    /// Precomputed RoPE table.
    ///
    /// Built over `n_rot` — the **rotary** dimension count — so
    /// `rope.half_dim == n_rot / 2` and [`RopeTable::apply`] rotates exactly
    /// `(x[i], x[i + n_rot/2])` for `i < n_rot/2`, leaving the tail of each
    /// head untouched.  Building it over `head_dim` (as this file used to)
    /// produced both the wrong frequencies and the wrong pair offset.
    pub rope: RopeTable,

    /// Cached `stablelm_config.rotary_dims(head_dim)`.
    rotary_dims: usize,

    // ── Scratch buffers, all allocated once at load time ──────────────────
    buf_hidden: Vec<f32>,
    buf_attn_norm: Vec<f32>,
    buf_ffn_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
    buf_acts_q8: Vec<u8>,
}

impl StablelmModel {
    /// Assemble a `StablelmModel` from pre-loaded weights.
    ///
    /// # Errors
    ///
    /// * [`ArchError::ConfigMismatch`] when `num_attention_heads`,
    ///   `num_kv_heads` or `hidden_size` is zero, when `num_kv_heads` does not
    ///   divide `num_attention_heads`, or when
    ///   `num_attention_heads * head_dim != hidden_size`.  llama.cpp creates
    ///   StableLM's `wq`/`wo` as `{n_embd, n_embd}` and reshapes Q to
    ///   `[n_embd_head, n_head, n_tokens]`, so the product is an identity for
    ///   this architecture — the previous code instead clamped the GEMV's row
    ///   stride with `.min(hidden_size)`, which silently made every row of the
    ///   output projection read from the wrong offset.
    /// * [`ArchError::InvalidShape`] when `token_embd` is too short to hold
    ///   `vocab_size × hidden_size` weights.
    /// * [`ArchError::Quant`] when the LM head's tensor type has no kernel.
    pub fn new(
        config: ModelConfig,
        stablelm_config: StablelmConfig,
        token_embd: Vec<f32>,
        layers: Vec<StablelmLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length.max(1);

        if hidden_size == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "hidden_size".to_string(),
                expected: "> 0".to_string(),
                got: "0".to_string(),
            });
        }
        if num_heads == 0 {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads".to_string(),
                expected: "> 0".to_string(),
                got: "0".to_string(),
            });
        }
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(ArchError::ConfigMismatch {
                param: "num_kv_heads".to_string(),
                expected: format!("a non-zero divisor of num_attention_heads ({num_heads})"),
                got: num_kv_heads.to_string(),
            });
        }
        // ── S5 ────────────────────────────────────────────────────────────
        if num_heads.saturating_mul(head_dim) != hidden_size {
            return Err(ArchError::ConfigMismatch {
                param: "num_attention_heads * head_dim".to_string(),
                expected: format!("hidden_size ({hidden_size})"),
                got: format!("{num_heads} * {head_dim} = {}", num_heads * head_dim),
            });
        }

        // `>=`, not `==`: some converters keep padding rows past the declared
        // vocabulary.  Falling *short* is what turns an in-range token id into
        // an out-of-bounds read.
        let min_embd_len = vocab_size.saturating_mul(hidden_size);
        if token_embd.len() < min_embd_len {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![vocab_size, hidden_size],
                got: vec![token_embd.len()],
            });
        }

        let rotary_dims = stablelm_config.rotary_dims(head_dim);
        // `new_with_style` rather than `new_standard_with_style`: it keeps the
        // checkpoint's RoPE scaling instead of forcing `Standard`/1.0, and
        // `config.rope_style()` keeps `rope_style_for_arch` ("stablelm" ⇒
        // NeoX, matching `llama_model_rope_type`'s `LLAMA_ROPE_TYPE_NEOX`
        // group) as the single source of truth.
        let rope = RopeTable::new_with_style(
            rotary_dims,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
            config.rope_style(),
        );

        let dispatcher = KernelDispatcher::new();
        let output_kernel = resolve_kernel(&dispatcher, &output)?;

        Ok(Self {
            config,
            stablelm_config,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            rotary_dims,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_attn_norm: vec![0.0f32; hidden_size],
            buf_ffn_norm: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; num_heads * head_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_gate: vec![0.0f32; intermediate_size],
            buf_up: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_attn_scores: vec![0.0f32; max_ctx],
            buf_acts_q8: Vec::new(),
        })
    }

    /// Number of rotated dimensions per head (llama.cpp's `n_rot`).
    pub fn rotary_dims(&self) -> usize {
        self.rotary_dims
    }

    /// Apply StableLM's partial RoPE to a single head vector in place.
    ///
    /// `head` is a full `head_dim`-wide vector; only its first
    /// [`Self::rotary_dims`] elements are touched, and they are rotated in
    /// `(i, i + rotary_dims/2)` pairs — the GPT-NeoX convention llama.cpp
    /// selects for StableLM.  This is the *same* call the decode path makes,
    /// so the two can no longer disagree.
    pub fn apply_partial_rope(&self, head: &mut [f32], position: usize) {
        self.rope.apply(head, position);
    }

    /// Load the residual stream with `token`'s embedding row.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let Self {
            config,
            token_embd,
            buf_hidden,
            ..
        } = self;
        let hidden = config.hidden_size;
        let offset = (token as usize)
            .checked_mul(hidden)
            .ok_or_else(|| out_of_vocab(token, config.vocab_size))?;
        let row = token_embd
            .get(offset..offset + hidden)
            .ok_or_else(|| out_of_vocab(token, config.vocab_size))?;
        buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Scaled dot-product attention with causal masking and GQA.
    ///
    /// Reads the post-`attn_norm` activation from `buf_attn_norm` and writes
    /// the output projection into `buf_attn_proj`.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // `new()` rejects a non-dividing `num_kv_heads`, so this is exact.
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = position + 1;

        // ── Q / K / V projections (+ optional bq/bk/bv) ───────────────────
        {
            let Self {
                layers,
                buf_attn_norm,
                buf_q,
                buf_k,
                buf_v,
                buf_acts_q8,
                ..
            } = self;
            let layer = layers
                .get(layer_idx)
                .ok_or_else(|| layer_index_error(layer_idx))?;

            let q_kernel: &dyn QuantKernel = &*layer.attn_q_kernel;
            let k_kernel: &dyn QuantKernel = &*layer.attn_k_kernel;
            let v_kernel: &dyn QuantKernel = &*layer.attn_v_kernel;

            // Q/K/V all read the same activation vector, so it is quantized to
            // Q8_0 exactly once and shared by all three GEMVs.
            let q_fused = layer.attn_q.q8_fused_blocks(q_kernel);
            let k_fused = layer.attn_k.q8_fused_blocks(k_kernel);
            let v_fused = layer.attn_v.q8_fused_blocks(v_kernel);
            buf_acts_q8.clear();
            if let Some(n_blocks) = q_fused
                .iter()
                .chain(&k_fused)
                .chain(&v_fused)
                .copied()
                .max()
            {
                quantize_activations_q8_0_into(buf_attn_norm, n_blocks, buf_acts_q8);
            }

            // `QuantLinear::forward` adds the optional bias after the GEMV,
            // which is exactly `Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq)`
            // in `llm_build_stablelm`.  A checkpoint without biases carries
            // `None` and the add is skipped.
            run_linear(&layer.attn_q, q_kernel, buf_attn_norm, buf_acts_q8, buf_q)?;
            run_linear(&layer.attn_k, k_kernel, buf_attn_norm, buf_acts_q8, buf_k)?;
            run_linear(&layer.attn_v, v_kernel, buf_attn_norm, buf_acts_q8, buf_v)?;
        }

        // ── Optional per-head QK LayerNorm, then partial RoPE ─────────────
        {
            let Self {
                layers,
                buf_q,
                buf_k,
                rope,
                ..
            } = self;
            let layer = layers
                .get(layer_idx)
                .ok_or_else(|| layer_index_error(layer_idx))?;

            if let Some(q_norm) = layer.attn_q_norm.as_ref() {
                q_norm.forward(buf_q);
            }
            if let Some(k_norm) = layer.attn_k_norm.as_ref() {
                k_norm.forward(buf_k);
            }

            for h in 0..num_heads {
                rope.apply(&mut buf_q[h * head_dim..(h + 1) * head_dim], position);
            }
            for h in 0..num_kv_heads {
                rope.apply(&mut buf_k[h * head_dim..(h + 1) * head_dim], position);
            }
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        // Borrowed, never copied: the previous implementation called
        // `.to_vec()` on the whole cache for both K and V on every layer of
        // every token, which is `2 · n_layer · seq_len · kv_dim` floats copied
        // per decoded token.
        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = cached_keys
                    .get(k_offset..k_offset + head_dim)
                    .ok_or_else(|| kv_range_error(layer_idx, "keys"))?;
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
                let v_vec = cached_values
                    .get(v_offset..v_offset + head_dim)
                    .ok_or_else(|| kv_range_error(layer_idx, "values"))?;
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // ── Output projection (no bias in this architecture) ──────────────
        let Self {
            layers,
            buf_attn_out,
            buf_attn_proj,
            buf_acts_q8,
            ..
        } = self;
        let layer = layers
            .get(layer_idx)
            .ok_or_else(|| layer_index_error(layer_idx))?;
        let o_kernel: &dyn QuantKernel = &*layer.attn_output_kernel;
        buf_acts_q8.clear();
        if let Some(n_blocks) = layer.attn_output.q8_fused_blocks(o_kernel) {
            quantize_activations_q8_0_into(buf_attn_out, n_blocks, buf_acts_q8);
        }
        run_linear(
            &layer.attn_output,
            o_kernel,
            buf_attn_out,
            buf_acts_q8,
            buf_attn_proj,
        )?;

        Ok(())
    }

    /// SwiGLU feed-forward, writing its output into `buf_ffn_out`.
    ///
    /// `parallel` selects the FFN's *input* buffer, mirroring
    /// `llm_build_stablelm`: with `ffn_norm` present the input is the
    /// normalized `ffn_inp` (`buf_ffn_norm`); without it the input is `inpSA`,
    /// the output of `attn_norm` (`buf_attn_norm`).
    fn feed_forward(&mut self, layer_idx: usize, parallel: bool) -> ArchResult<()> {
        let Self {
            layers,
            buf_attn_norm,
            buf_ffn_norm,
            buf_gate,
            buf_up,
            buf_ffn_out,
            buf_acts_q8,
            ..
        } = self;
        let layer = layers
            .get(layer_idx)
            .ok_or_else(|| layer_index_error(layer_idx))?;

        let input: &[f32] = if parallel {
            buf_attn_norm
        } else {
            buf_ffn_norm
        };

        let gate_kernel: &dyn QuantKernel = &*layer.ffn_gate_kernel;
        let up_kernel: &dyn QuantKernel = &*layer.ffn_up_kernel;
        let down_kernel: &dyn QuantKernel = &*layer.ffn_down_kernel;

        let gate_fused = layer.ffn_gate.q8_fused_blocks(gate_kernel);
        let up_fused = layer.ffn_up.q8_fused_blocks(up_kernel);
        buf_acts_q8.clear();
        if let Some(n_blocks) = gate_fused.iter().chain(&up_fused).copied().max() {
            quantize_activations_q8_0_into(input, n_blocks, buf_acts_q8);
        }

        run_linear(&layer.ffn_gate, gate_kernel, input, buf_acts_q8, buf_gate)?;
        run_linear(&layer.ffn_up, up_kernel, input, buf_acts_q8, buf_up)?;

        swiglu_inplace(buf_gate, buf_up);

        buf_acts_q8.clear();
        if let Some(n_blocks) = layer.ffn_down.q8_fused_blocks(down_kernel) {
            quantize_activations_q8_0_into(buf_gate, n_blocks, buf_acts_q8);
        }
        run_linear(
            &layer.ffn_down,
            down_kernel,
            buf_gate,
            buf_acts_q8,
            buf_ffn_out,
        )?;

        Ok(())
    }

    /// Run one transformer block over the residual stream in `buf_hidden`.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        // `cur = build_norm(inpL, attn_norm, attn_norm_b, LLM_NORM)`;
        // `inpSA = cur` — kept intact in `buf_attn_norm` for the whole block
        // because the parallel-residual FFN reads it.
        {
            let Self {
                layers,
                buf_hidden,
                buf_attn_norm,
                ..
            } = self;
            let layer = layers
                .get(layer_idx)
                .ok_or_else(|| layer_index_error(layer_idx))?;
            layer.attn_norm.forward_to(buf_hidden, buf_attn_norm);
        }

        self.attention(layer_idx, position, kv_cache)?;

        // `ffn_inp = ggml_add(ctx0, cur, inpL)` — happens in BOTH topologies.
        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_attn_proj.iter()) {
            *h += p;
        }

        // `if (ffn_norm) { cur = build_norm(ffn_inp, ffn_norm, ffn_norm_b) }
        //  else          { cur = inpSA; }`
        let parallel = {
            let Self {
                layers,
                buf_hidden,
                buf_ffn_norm,
                ..
            } = self;
            let layer = layers
                .get(layer_idx)
                .ok_or_else(|| layer_index_error(layer_idx))?;
            match layer.ffn_norm.as_ref() {
                Some(norm) => {
                    norm.forward_to(buf_hidden, buf_ffn_norm);
                    false
                }
                None => true,
            }
        };

        self.feed_forward(layer_idx, parallel)?;

        // `cur = ggml_add(ctx0, cur, ffn_inp)`
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }

    /// Run every token through all layers, leaving the last token's
    /// pre-output-norm hidden state in `buf_hidden`.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] for an out-of-vocabulary token id or a
    /// prompt that would run past `max_context_length`.  Both used to be
    /// panics reachable from the HTTP server with attacker-controlled input:
    /// the embedding lookup indexed `token_embd` unchecked, and
    /// `buf_attn_scores` (sized `max_context_length`) was indexed by the raw
    /// position with no prefill guard at all.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        validate_token_ids(&self.config, tokens)?;

        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }

            kv_cache.advance();
        }

        Ok(())
    }

    /// Apply `output_norm` and project through the LM head into `buf_logits`.
    fn project_logits(&mut self) -> ArchResult<()> {
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        let Self {
            output,
            output_kernel,
            buf_hidden,
            buf_logits,
            buf_acts_q8,
            ..
        } = self;
        let kernel: &dyn QuantKernel = &**output_kernel;
        buf_acts_q8.clear();
        if let Some(n_blocks) = output.q8_fused_blocks(kernel) {
            quantize_activations_q8_0_into(buf_hidden, n_blocks, buf_acts_q8);
        }
        run_linear(output, kernel, buf_hidden, buf_acts_q8, buf_logits)
    }
}

impl ForwardPass for StablelmModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);
        self.project_logits()?;
        // Ownership transfer instead of a `vocab_size`-wide clone per token;
        // `project_logits` restores the length on the next call.
        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Post-`output_norm` hidden state, without the LM-head projection.
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
}

/// Out-of-vocabulary token id error.
fn out_of_vocab(token: u32, vocab_size: usize) -> ArchError {
    ArchError::ConfigMismatch {
        param: "token_id".to_string(),
        expected: format!("< vocab_size ({vocab_size})"),
        got: token.to_string(),
    }
}

/// A layer index past the end of `layers` (defensive; `run_layers` iterates
/// `0..layers.len()`).
fn layer_index_error(layer: usize) -> ArchError {
    ArchError::ForwardPassError {
        layer,
        message: "layer index out of range".to_string(),
    }
}

/// KV-cache slice out of range (a cache shorter than the sequence claims).
fn kv_range_error(layer: usize, which: &str) -> ArchError {
    ArchError::ForwardPassError {
        layer,
        message: format!("KV cache {which} shorter than the current sequence length"),
    }
}

/// Numerically stable in-place softmax.
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
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, -1.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
        assert!(
            x.windows(2).take(3).all(|w| w[0] != w[1]),
            "distinct logits must map to distinct probabilities"
        );
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let mut x = vec![1000.0f32, 1000.0, 1000.0];
        softmax_inplace(&mut x);
        for v in &x {
            assert!(
                (*v - 1.0 / 3.0).abs() < 1e-6,
                "identical large logits must give a uniform distribution, got {v}"
            );
        }
    }

    #[test]
    fn softmax_handles_empty_slice() {
        let mut x: Vec<f32> = Vec::new();
        softmax_inplace(&mut x);
    }
}
