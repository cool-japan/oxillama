//! GPT-NeoX (EleutherAI / Pythia) transformer forward pass implementation.
//!
//! Ported from `~/work/refs/llama.cpp/src/models/gptneox.cpp`
//! (`llm_build_gptneox`) and `src/llama-model.cpp`
//! (`case LLM_ARCH_GPTNEOX:` in both `load_hparams()` and `load_tensors()`).
//!
//! ## Block structure
//!
//! `hparams.use_par_res` (GGUF `{arch}.use_parallel_residual`) selects between
//! **two completely different** per-layer computations:
//!
//! ```text
//! // use_par_res == true  (GPT-NeoX / Pythia default)
//! attn_out = attn(attn_norm(x))          // attn_norm reads the residual stream
//! ffn_out  = ffn (ffn_norm (x))          // ffn_norm ALSO reads the residual stream
//! x        = ffn_out + x + attn_out      // one combined residual update
//!
//! // use_par_res == false (sequential pre-norm block)
//! ffn_inp  = attn(attn_norm(x)) + x
//! x        = ffn(ffn_norm(ffn_inp)) + ffn_inp
//! ```
//!
//! The parallel branch's second norm reads the **original** `inpL`, not the
//! attention output — that is the whole point of "parallel residual" and is
//! why the two branches cannot be folded into one.
//!
//! ## Other architectural facts (all confirmed against the reference)
//!
//! * **Fused QKV.** One `attn_qkv` tensor per layer whose output splits into
//!   three contiguous blocks `Q | K | V` (`ggml_view_3d` offsets `0*n_embd`,
//!   `1*n_embd`, `1*(n_embd + n_embd_gqa)`).  The per-head interleave of the
//!   HuggingFace checkpoint is undone by `convert_hf_to_gguf.py`, not here.
//! * **Bias everywhere.** `bqkv`, `bo`, `ffn_up_b`, `ffn_down_b`,
//!   `attn_norm_b`, `ffn_norm_b`, `output_norm_b` are all created with
//!   `flags = 0` — required, never optional.
//! * **Plain LayerNorm** (`LLM_NORM`), not RMSNorm.
//! * **Gate-free GELU FFN** (`LLM_FFN_GELU, LLM_FFN_SEQ`).
//! * **Partial NeoX RoPE.** ggml rotates only the leading `n_rot` elements of
//!   each head with `theta_scale = freq_base^(-2/n_rot)`, pairing
//!   `(x[i], x[i + n_rot/2])`, and copies `x[n_rot..head_dim]` through
//!   untouched (`ggml/src/ggml-cpu/ops.cpp`,
//!   `rotate_pairs(n_dims, n_dims/2, …)` plus the "fill the remain channels"
//!   loop).  `n_rot` is `{arch}.rope.dimension_count`, **not** `head_dim`.
//! * **Attention scale** `1/sqrt(head_dim)`.

use std::sync::Arc;

use oxillama_quant::{KernelDispatcher, QuantKernel};

use crate::common::gelu::gelu_inplace;
use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};

/// Default for `{arch}.use_parallel_residual` when the checkpoint omits it.
///
/// `convert_hf_to_gguf.py`'s `GPTNeoXModel.set_gguf_parameters` writes
/// `hparams.get("use_parallel_residual", True)`, so a converted checkpoint
/// always carries the key explicitly; this default only covers hand-built or
/// truncated metadata.
pub const DEFAULT_USE_PARALLEL_RESIDUAL: bool = true;

/// The plain-LayerNorm weights/bias and quantized projections of one block.
///
/// Split out of [`GptNeoxLayer`] so the constructor takes a single argument
/// instead of eleven, and so the resolved [`QuantKernel`]s can be derived from
/// the weights rather than passed in alongside them.
pub struct GptNeoxLayerWeights {
    /// Pre-attention LayerNorm (`attn_norm.weight` + `attn_norm.bias`).
    pub attn_norm: LayerNorm,
    /// Fused QKV projection `[n_embd + 2 * n_embd_gqa, n_embd]`.
    pub attn_qkv: QuantLinear,
    /// Fused QKV bias `[n_embd + 2 * n_embd_gqa]` (required).
    pub attn_qkv_bias: Vec<f32>,
    /// Attention output projection `[n_embd, n_embd]`.
    pub attn_output: QuantLinear,
    /// Attention output bias `[n_embd]` (required).
    pub attn_out_bias: Vec<f32>,
    /// Pre-FFN LayerNorm (`ffn_norm.weight` + `ffn_norm.bias`).
    pub ffn_norm: LayerNorm,
    /// FFN up projection `[n_ff, n_embd]`.
    pub ffn_up: QuantLinear,
    /// FFN up bias `[n_ff]` (required).
    pub ffn_up_bias: Vec<f32>,
    /// FFN down projection `[n_embd, n_ff]`.
    pub ffn_down: QuantLinear,
    /// FFN down bias `[n_embd]` (required).
    pub ffn_down_bias: Vec<f32>,
}

/// A single GPT-NeoX transformer block.
///
/// Every linear projection carries a bias — GPT-NeoX/Pythia has no bias-free
/// projection anywhere (`llama-model.cpp`, `LLM_ARCH_GPTNEOX`: `bqkv`, `bo`,
/// `ffn_up_b` and `ffn_down_b` are all `flags = 0`).
pub struct GptNeoxLayer {
    /// Pre-attention LayerNorm (with bias).
    pub attn_norm: LayerNorm,
    /// Fused QKV projection.
    pub attn_qkv: QuantLinear,
    /// Fused QKV bias.
    pub attn_qkv_bias: Vec<f32>,
    /// Attention output projection.
    pub attn_output: QuantLinear,
    /// Attention output bias.
    pub attn_out_bias: Vec<f32>,
    /// Pre-FFN LayerNorm (with bias).
    pub ffn_norm: LayerNorm,
    /// FFN up projection.
    pub ffn_up: QuantLinear,
    /// FFN up bias.
    pub ffn_up_bias: Vec<f32>,
    /// FFN down projection.
    pub ffn_down: QuantLinear,
    /// FFN down bias.
    pub ffn_down_bias: Vec<f32>,

    // ── Kernels resolved once, at load time ──────────────────────────────────
    //
    // The decode loop used to call `dispatcher.get_kernel()` — a fresh
    // `Box<dyn QuantKernel>` allocation plus the full dispatch cascade — for
    // every projection of every layer of every token.  Resolving them here
    // (from the tensor type the weight already carries) turns each of those
    // into a field load and an `Arc` deref.  Same fix as `Qwen3Layer`'s
    // `*_kernel` fields, which is where the measured 6.34 → 12.98 tok/s came
    // from.
    /// Kernel for [`Self::attn_qkv`].
    pub attn_qkv_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`].
    pub attn_output_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_up`].
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_down`].
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

impl GptNeoxLayer {
    /// Build a layer, resolving one [`QuantKernel`] per projection up front.
    ///
    /// # Errors
    ///
    /// [`ArchError::Quant`] when any projection's tensor type has no
    /// registered kernel.
    pub fn new(dispatcher: &KernelDispatcher, w: GptNeoxLayerWeights) -> ArchResult<Self> {
        let attn_qkv_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(w.attn_qkv.weight.tensor_type)?.into();
        let attn_output_kernel: Arc<dyn QuantKernel> = dispatcher
            .get_kernel(w.attn_output.weight.tensor_type)?
            .into();
        let ffn_up_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(w.ffn_up.weight.tensor_type)?.into();
        let ffn_down_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(w.ffn_down.weight.tensor_type)?.into();

        Ok(Self {
            attn_norm: w.attn_norm,
            attn_qkv: w.attn_qkv,
            attn_qkv_bias: w.attn_qkv_bias,
            attn_output: w.attn_output,
            attn_out_bias: w.attn_out_bias,
            ffn_norm: w.ffn_norm,
            ffn_up: w.ffn_up,
            ffn_up_bias: w.ffn_up_bias,
            ffn_down: w.ffn_down,
            ffn_down_bias: w.ffn_down_bias,
            attn_qkv_kernel,
            attn_output_kernel,
            ffn_up_kernel,
            ffn_down_kernel,
        })
    }
}

/// Complete GPT-NeoX model.
pub struct GptNeoxModel {
    /// Base model configuration.
    pub config: ModelConfig,
    /// Rotary dimension count `n_rot` (`{arch}.rope.dimension_count`).
    ///
    /// **Not** `head_dim`: GPT-NeoX rotates `rotary_pct × head_dim` leading
    /// elements of every head (Pythia ships `rotary_pct = 0.25`).  The RoPE
    /// frequency ladder is derived from *this* value, not from `head_dim` —
    /// see [`Self::new`].
    pub rotary_dims: usize,
    /// Whether attention and FFN share the same residual input
    /// (`{arch}.use_parallel_residual`).
    pub use_parallel_residual: bool,
    /// Token embeddings, dequantized, `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<GptNeoxLayer>,
    /// Final LayerNorm (with bias).
    pub output_norm: LayerNorm,
    /// LM head `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Kernel for [`Self::output`], resolved once in [`Self::new`].
    output_kernel: Arc<dyn QuantKernel>,
    /// Precomputed RoPE table over `rotary_dims` (so `half_dim == n_rot / 2`).
    pub rope: RopeTable,
    /// Kernel dispatcher retained for out-of-band operations (LoRA, tooling).
    pub dispatcher: KernelDispatcher,

    // ── Scratch buffers, allocated once ──────────────────────────────────────
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_ffn: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

/// Validate the attention geometry a GPT-NeoX block requires.
///
/// llama.cpp hard-codes `wo` as `{n_embd, n_embd}` and the fused QKV as
/// `{n_embd, n_embd + 2*n_embd_gqa}`, i.e. the concatenated query heads *are*
/// the residual stream: `n_head * n_embd_head == n_embd`.  A checkpoint whose
/// metadata disagrees must be rejected, not silently clamped.
///
/// # Errors
///
/// [`ArchError::ConfigMismatch`] naming the offending parameter.
pub fn validate_gpt_neox_shapes(config: &ModelConfig) -> ArchResult<()> {
    let mismatch = |param: &str, expected: String, got: String| ArchError::ConfigMismatch {
        param: format!("gptneox.{param}"),
        expected,
        got,
    };

    let heads = config.num_attention_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let hidden = config.hidden_size;

    if heads == 0 {
        return Err(mismatch("attention.head_count", "> 0".into(), "0".into()));
    }
    if kv_heads == 0 {
        return Err(mismatch(
            "attention.head_count_kv",
            "> 0".into(),
            "0".into(),
        ));
    }
    if head_dim == 0 {
        return Err(mismatch("attention.key_length", "> 0".into(), "0".into()));
    }
    if hidden == 0 {
        return Err(mismatch("embedding_length", "> 0".into(), "0".into()));
    }
    if config.vocab_size == 0 {
        return Err(mismatch("vocab_size", "> 0".into(), "0".into()));
    }
    if config.intermediate_size == 0 {
        return Err(mismatch("feed_forward_length", "> 0".into(), "0".into()));
    }
    if config.max_context_length == 0 {
        return Err(mismatch("context_length", "> 0".into(), "0".into()));
    }
    if kv_heads > heads || !heads.is_multiple_of(kv_heads) {
        return Err(mismatch(
            "attention.head_count_kv",
            format!("a divisor of head_count ({heads})"),
            kv_heads.to_string(),
        ));
    }

    // Defect N5: the old code computed the attention-output row stride as
    // `(num_heads * head_dim).min(hidden_size)`, which reads every weight row
    // from the wrong offset the moment the two disagree.  A clamp cannot
    // repair a disagreement — the checkpoint and the metadata simply do not
    // describe the same model.
    let attn_dim = heads.checked_mul(head_dim).ok_or_else(|| {
        mismatch(
            "attention.head_count * key_length",
            "no overflow".into(),
            format!("{heads} * {head_dim}"),
        )
    })?;
    if attn_dim != hidden {
        return Err(mismatch(
            "attention.head_count * key_length",
            format!("== embedding_length ({hidden})"),
            attn_dim.to_string(),
        ));
    }

    Ok(())
}

/// Validate `n_rot` against ggml's two `GGML_ASSERT`s in `ggml_rope_ext`:
/// `n_dims <= ne0` and `n_dims % 2 == 0`.
///
/// # Errors
///
/// [`ArchError::ConfigMismatch`] for an odd or over-long rotary count.
pub fn validate_rotary_dims(rotary_dims: usize, head_dim: usize) -> ArchResult<()> {
    if !rotary_dims.is_multiple_of(2) {
        return Err(ArchError::ConfigMismatch {
            param: "gptneox.rope.dimension_count".to_string(),
            expected: "an even number (ggml: n_dims % 2 == 0)".to_string(),
            got: rotary_dims.to_string(),
        });
    }
    if rotary_dims > head_dim {
        return Err(ArchError::ConfigMismatch {
            param: "gptneox.rope.dimension_count".to_string(),
            expected: format!("<= head_dim ({head_dim})"),
            got: rotary_dims.to_string(),
        });
    }
    Ok(())
}

impl GptNeoxModel {
    /// Assemble a model from already-loaded weights.
    ///
    /// # Arguments
    /// * `rotary_dims` — `n_rot`, i.e. `{arch}.rope.dimension_count`.  Pass
    ///   `head_dim` for a full-rotary checkpoint; llama.cpp uses exactly that
    ///   as its fallback (`llama-model.cpp`: `hparams.n_rot =
    ///   hparams.n_embd_head_k;` immediately before the optional
    ///   `get_key(LLM_KV_ROPE_DIMENSION_COUNT, …, /*required=*/false)`).
    /// * `use_parallel_residual` — `{arch}.use_parallel_residual`.
    ///
    /// # Errors
    ///
    /// * [`ArchError::ConfigMismatch`] for an inconsistent attention geometry
    ///   or an illegal `rotary_dims` (see [`validate_gpt_neox_shapes`] and
    ///   [`validate_rotary_dims`]).
    /// * [`ArchError::InvalidShape`] when `token_embd` is not
    ///   `vocab_size × hidden_size` long.
    /// * [`ArchError::Quant`] when the LM head's tensor type has no kernel.
    pub fn new(
        config: ModelConfig,
        rotary_dims: usize,
        use_parallel_residual: bool,
        token_embd: Vec<f32>,
        layers: Vec<GptNeoxLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        validate_gpt_neox_shapes(&config)?;
        validate_rotary_dims(rotary_dims, config.head_dim)?;

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;

        let expected_embd =
            vocab_size
                .checked_mul(hidden_size)
                .ok_or_else(|| ArchError::InvalidShape {
                    name: "token_embd.weight".to_string(),
                    expected: vec![vocab_size, hidden_size],
                    got: vec![token_embd.len()],
                })?;
        if token_embd.len() != expected_embd {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![expected_embd],
                got: vec![token_embd.len()],
            });
        }

        let attn_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_total = attn_dim + 2 * kv_dim;

        // Defect N4: the table is built over `rotary_dims`, so its frequency
        // ladder is `base^(-2i/n_rot)` — matching ggml's
        // `theta_scale = powf(freq_base, -2.0f/n_dims)` with `n_dims = n_rot`.
        // Building it over `head_dim` and then using only the leading
        // `n_rot/2` entries (what this file used to do) produces a different
        // ladder whenever `n_rot != head_dim`, which is *always* for Pythia.
        let rope = RopeTable::new_with_style(
            rotary_dims,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
            config.rope_style(),
        );

        let dispatcher = KernelDispatcher::new();
        let output_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(output.weight.tensor_type)?.into();

        Ok(Self {
            config,
            rotary_dims,
            use_parallel_residual,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            dispatcher,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_qkv: vec![0.0f32; qkv_total],
            buf_q: vec![0.0f32; attn_dim],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_out: vec![0.0f32; attn_dim],
            buf_attn_proj: vec![0.0f32; hidden_size],
            buf_ffn: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_attn_scores: vec![0.0f32; max_ctx],
        })
    }

    /// Copy `token`'s embedding row into the residual stream.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] for an out-of-vocabulary id.  The old
    /// implementation sliced `token_embd[offset..offset + h]` directly, which
    /// aborted the process on an OOV id reachable from the HTTP server.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden = self.config.hidden_size;
        let offset = (token as usize).saturating_mul(hidden);
        let row = self
            .token_embd
            .get(offset..offset + hidden)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< vocab_size ({})", self.config.vocab_size),
                got: token.to_string(),
            })?;
        self.buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Fused-QKV multi-head attention with partial NeoX RoPE.
    ///
    /// Reads `self.buf_norm` (the post-`attn_norm` activation) and leaves the
    /// projected, biased attention output in `self.buf_attn_proj`.  It does
    /// **not** touch the residual stream: which residual the result is added
    /// to is the caller's decision, and that decision is exactly what
    /// `use_parallel_residual` selects.
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
        let qkv_total = attn_dim + 2 * kv_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_rot = self.rotary_dims;
        let seq_len = position + 1;

        // ── Fused QKV projection + required bias ─────────────────────────────
        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_qkv
                .forward(&*layer.attn_qkv_kernel, &self.buf_norm, &mut self.buf_qkv)?;
            // Defect N2: the block had no bias fields at all, so `bqkv` — a
            // required tensor in every GPT-NeoX checkpoint — was dropped.
            for (v, &b) in self.buf_qkv.iter_mut().zip(layer.attn_qkv_bias.iter()) {
                *v += b;
            }
        }

        // ── Split the fused output into contiguous Q | K | V blocks ──────────
        self.buf_q.copy_from_slice(&self.buf_qkv[..attn_dim]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[attn_dim..attn_dim + kv_dim]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[attn_dim + kv_dim..qkv_total]);

        // ── Partial RoPE over the leading `n_rot` elements of each head ──────
        for h in 0..num_heads {
            let head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            self.rope.try_apply(&mut head[..n_rot], position)?;
        }
        for h in 0..num_kv_heads {
            let head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            self.rope.try_apply(&mut head[..n_rot], position)?;
        }

        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        // Defect N6: these used to be `.to_vec()` copies of the *entire* KV
        // cache — every layer, every token.  A borrow costs nothing and the
        // trait already hands out `&[f32]`.
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
                    .ok_or_else(|| ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!(
                            "KV cache holds {} key floats, need {} for position {pos}",
                            cached_keys.len(),
                            k_offset + head_dim
                        ),
                    })?;
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
                let Some(v_vec) = cached_values.get(v_offset..v_offset + head_dim) else {
                    return Err(ArchError::ForwardPassError {
                        layer: layer_idx,
                        message: format!(
                            "KV cache holds {} value floats, need {} for position {pos}",
                            cached_values.len(),
                            v_offset + head_dim
                        ),
                    });
                };
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // ── Output projection + required bias ────────────────────────────────
        let layer = &self.layers[layer_idx];
        layer.attn_output.forward(
            &*layer.attn_output_kernel,
            &self.buf_attn_out,
            &mut self.buf_attn_proj,
        )?;
        for (p, &b) in self
            .buf_attn_proj
            .iter_mut()
            .zip(layer.attn_out_bias.iter())
        {
            *p += b;
        }

        Ok(())
    }

    /// Gate-free GELU FFN: `down(gelu(up(x) + up_b)) + down_b`.
    ///
    /// Reads `self.buf_norm`, writes `self.buf_ffn_out`.  Matches
    /// `build_ffn(..., LLM_FFN_GELU, LLM_FFN_SEQ, il)` in the reference.
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        layer
            .ffn_up
            .forward(&*layer.ffn_up_kernel, &self.buf_norm, &mut self.buf_ffn)?;
        for (v, &b) in self.buf_ffn.iter_mut().zip(layer.ffn_up_bias.iter()) {
            *v += b;
        }

        gelu_inplace(&mut self.buf_ffn);

        layer.ffn_down.forward(
            &*layer.ffn_down_kernel,
            &self.buf_ffn,
            &mut self.buf_ffn_out,
        )?;
        for (v, &b) in self.buf_ffn_out.iter_mut().zip(layer.ffn_down_bias.iter()) {
            *v += b;
        }

        Ok(())
    }

    /// Run one block, honouring `use_parallel_residual`.
    ///
    /// Defect N3: only the parallel branch existed, hard-coded.  Both formulas
    /// are transcribed from `llm_build_gptneox`'s `if (hparams.use_par_res)`.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        self.layers[layer_idx]
            .attn_norm
            .forward_to(&self.buf_hidden, &mut self.buf_norm);

        self.attention(layer_idx, position, kv_cache)?;

        if self.use_parallel_residual {
            // `attention()` deliberately left `buf_hidden` alone, so the FFN's
            // norm still sees the ORIGINAL residual stream — that is what
            // "parallel" means:  x = ffn(ffn_norm(x)) + x + attn(attn_norm(x)).
            self.layers[layer_idx]
                .ffn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
            self.feed_forward(layer_idx)?;

            for ((h, &a), &f) in self
                .buf_hidden
                .iter_mut()
                .zip(self.buf_attn_proj.iter())
                .zip(self.buf_ffn_out.iter())
            {
                *h += a + f;
            }
        } else {
            // Sequential:  ffn_inp = attn(attn_norm(x)) + x
            //              x       = ffn(ffn_norm(ffn_inp)) + ffn_inp
            for (h, &a) in self.buf_hidden.iter_mut().zip(self.buf_attn_proj.iter()) {
                *h += a;
            }

            self.layers[layer_idx]
                .ffn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
            self.feed_forward(layer_idx)?;

            for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
                *h += f;
            }
        }

        Ok(())
    }

    /// Push every token through all layers, leaving the last token's
    /// pre-`output_norm` hidden state in `self.buf_hidden`.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        let start_pos = kv_cache.seq_len();

        // Both guards run before ANY RoPE table index or `buf_attn_scores[pos]`
        // write: prefill length and token ids are attacker-controlled through
        // the HTTP server, and neither used to be checked here.
        crate::common::validate_context_bounds(&self.config, start_pos, tokens.len())?;
        crate::common::validate_token_ids(&self.config, tokens)?;

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
}

impl ForwardPass for GptNeoxModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        self.output_norm.forward(&mut self.buf_hidden);

        // `buf_logits` is handed to the caller by ownership below, so restore
        // its length first (a no-op on every call but the one after a take).
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        self.output
            .forward(&*self.output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

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

    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        self.apply_lora_scaled(lora, 1.0)
    }

    /// Attach LoRA adapters to the four projections of every block.
    ///
    /// The fused `attn_qkv` is a single adapter target — that is how a LoRA
    /// trained against a GGUF GPT-NeoX names it, because the checkpoint has no
    /// separate `attn_q`/`attn_k`/`attn_v` tensors to target.
    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let targets: [(String, &mut QuantLinear); 4] = [
                (format!("blk.{i}.attn_qkv.weight"), &mut layer.attn_qkv),
                (
                    format!("blk.{i}.attn_output.weight"),
                    &mut layer.attn_output,
                ),
                (format!("blk.{i}.ffn_up.weight"), &mut layer.ffn_up),
                (format!("blk.{i}.ffn_down.weight"), &mut layer.ffn_down),
            ];
            for (tensor_name, linear) in targets {
                if let Some(adapter) = lora.get(&tensor_name) {
                    linear.push_lora(adapter, scale);
                }
            }
        }
        Ok(())
    }

    fn unapply_all_loras(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn_qkv.clear_lora();
            layer.attn_output.clear_lora();
            layer.ffn_up.clear_lora();
            layer.ffn_down.clear_lora();
        }
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

// ─── Test helpers ─────────────────────────────────────────────────────────────

/// Build an F32 [`QuantLinear`] of shape `[out_features, in_features]`.
#[cfg(test)]
pub fn f32_linear(values: &[f32], out_features: usize, in_features: usize) -> QuantLinear {
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    QuantLinear::new(
        QuantTensor::new(data, vec![out_features, in_features], GgufTensorType::F32),
        None,
    )
}

/// Deterministic pseudo-random values in `base ± amp`.
///
/// Test weights must **vary** along the output dimension.  A matrix whose rows
/// are all identical produces an output that is constant across the hidden
/// dimension, and the mean-subtraction inside the final LayerNorm then deletes
/// it — so every "changing X changes the logits" assertion would pass
/// vacuously (or, worse, fail for a reason unrelated to the code under test).
#[cfg(test)]
pub fn test_pattern(n: usize, seed: u32, base: f32, amp: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 9) as f32 / 8_388_608.0) - 1.0;
            base + amp * unit
        })
        .collect()
}

/// Build a minimal [`GptNeoxLayer`] with small varying weights, for tests.
#[cfg(test)]
pub fn make_test_layer(config: &ModelConfig) -> ArchResult<GptNeoxLayer> {
    let hidden = config.hidden_size;
    let ffn = config.intermediate_size;
    let attn_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let qkv_total = attn_dim + 2 * kv_dim;

    let dispatcher = KernelDispatcher::new();
    GptNeoxLayer::new(
        &dispatcher,
        GptNeoxLayerWeights {
            attn_norm: LayerNorm::new(vec![1.0f32; hidden], Some(vec![0.0f32; hidden]), 1e-5),
            attn_qkv: f32_linear(
                &test_pattern(qkv_total * hidden, 101, 0.0, 0.2),
                qkv_total,
                hidden,
            ),
            attn_qkv_bias: vec![0.0f32; qkv_total],
            attn_output: f32_linear(
                &test_pattern(hidden * attn_dim, 103, 0.0, 0.2),
                hidden,
                attn_dim,
            ),
            attn_out_bias: vec![0.0f32; hidden],
            ffn_norm: LayerNorm::new(vec![1.0f32; hidden], Some(vec![0.0f32; hidden]), 1e-5),
            ffn_up: f32_linear(&test_pattern(ffn * hidden, 107, 0.0, 0.2), ffn, hidden),
            ffn_up_bias: vec![0.0f32; ffn],
            ffn_down: f32_linear(&test_pattern(hidden * ffn, 109, 0.0, 0.2), hidden, ffn),
            ffn_down_bias: vec![0.0f32; hidden],
        },
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::test_pattern as pattern;
    use super::*;
    use crate::gpt_neox::GptNeoxArchitecture;
    use crate::registry::ArchitectureRegistry;
    use crate::traits::ModelArchitecture;

    const HIDDEN: usize = 16;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const ROT: usize = 2;

    fn minimal_config() -> ModelConfig {
        ModelConfig {
            architecture: "gptneox".to_string(),
            hidden_size: HIDDEN,
            intermediate_size: 32,
            num_layers: 1,
            num_attention_heads: HEADS,
            num_kv_heads: HEADS,
            head_dim: HEAD_DIM,
            vocab_size: 4,
            max_context_length: 8,
            ..ModelConfig::default()
        }
    }

    /// Build a one-layer test model.
    ///
    /// The token embeddings and the LM head carry a *varying* pattern rather
    /// than a single repeated constant.  With constant rows the final
    /// LayerNorm — which subtracts the mean — annihilates every signal and all
    /// logits collapse to ~1e-9, which would make every "output changed"
    /// assertion below vacuous.
    fn make_model(config: &ModelConfig, rotary_dims: usize, parallel: bool) -> GptNeoxModel {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let layer = make_test_layer(config).expect("layer");
        GptNeoxModel::new(
            config.clone(),
            rotary_dims,
            parallel,
            pattern(vocab * hidden, 3, 0.0, 0.5),
            vec![layer],
            LayerNorm::new(vec![1.0f32; hidden], Some(vec![0.0f32; hidden]), 1e-5),
            f32_linear(&pattern(vocab * hidden, 5, 0.0, 0.5), vocab, hidden),
        )
        .expect("model")
    }

    /// Minimal KV cache for forward-pass tests.
    struct SimpleKvCache {
        kv_dim: usize,
        max_seq: usize,
        n_layers: usize,
        position: usize,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    }

    impl SimpleKvCache {
        fn new(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
            Self {
                kv_dim,
                max_seq,
                n_layers,
                position: 0,
                keys: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
                values: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
            }
        }
    }

    impl KvCacheAccess for SimpleKvCache {
        fn seq_len(&self) -> usize {
            self.position
        }

        fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            if layer >= self.n_layers {
                return Err(ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                });
            }
            let offset = self.position * self.kv_dim;
            let ck = key.len().min(self.kv_dim);
            let cv = value.len().min(self.kv_dim);
            self.keys[layer][offset..offset + ck].copy_from_slice(&key[..ck]);
            self.values[layer][offset..offset + cv].copy_from_slice(&value[..cv]);
            Ok(())
        }

        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            let end = (self.position + 1) * self.kv_dim;
            self.keys
                .get(layer)
                .and_then(|k| k.get(..end))
                .ok_or_else(|| ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                })
        }

        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            let end = (self.position + 1) * self.kv_dim;
            self.values
                .get(layer)
                .and_then(|v| v.get(..end))
                .ok_or_else(|| ArchError::InvalidConfig {
                    detail: format!("layer {layer} out of range"),
                })
        }

        fn advance(&mut self) {
            self.position = (self.position + 1).min(self.max_seq - 1);
        }
    }

    fn kv_for(config: &ModelConfig) -> SimpleKvCache {
        SimpleKvCache::new(
            config.num_layers,
            config.num_kv_heads * config.head_dim,
            config.max_context_length,
        )
    }

    // ── Registry ──────────────────────────────────────────────────────────────

    #[test]
    fn gptneox_registry_lookup() {
        let registry = ArchitectureRegistry::with_builtins();
        let arch = registry.get("gptneox");
        assert!(
            arch.is_ok(),
            "registry.get('gptneox') must succeed; got: {:?}",
            arch.err()
        );
        assert_eq!(arch.expect("gptneox arch").arch_id(), "gptneox");
    }

    /// Defect N1 regression at the plugin surface: the declared names must be
    /// the ones a real GGUF carries.  The pre-fix list (`ln1`, `ln2`,
    /// `attn_q`/`attn_k`/`attn_v`) matched no checkpoint ever produced.
    #[test]
    fn gptneox_tensor_names_are_real_gguf_names() {
        let names: Vec<String> = GptNeoxArchitecture::new()
            .tensor_names()
            .into_iter()
            .map(|p| p.pattern)
            .collect();

        for required in [
            "token_embd.weight",
            "output_norm.weight",
            "output_norm.bias",
            "output.weight",
            "blk.{i}.attn_norm.weight",
            "blk.{i}.attn_qkv.weight",
            "blk.{i}.attn_qkv.bias",
            "blk.{i}.ffn_norm.weight",
        ] {
            assert!(
                names.iter().any(|n| n == required),
                "tensor_names() must contain '{required}'; got {names:?}"
            );
        }
        for name in &names {
            for phantom in ["ln1", "ln2", "attn_q.", "attn_k.", "attn_v."] {
                assert!(
                    !name.contains(phantom),
                    "'{name}' contains phantom fragment '{phantom}'"
                );
            }
        }
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    #[test]
    fn gptneox_forward_produces_vocab_logits() {
        let config = minimal_config();
        let mut model = make_model(&config, ROT, true);
        let mut kv = kv_for(&config);

        let logits = model.forward(&[0u32], &mut kv).expect("forward");
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
    }

    #[test]
    fn gptneox_forward_is_deterministic() {
        let config = minimal_config();
        let mut m1 = make_model(&config, ROT, true);
        let mut m2 = make_model(&config, ROT, true);
        let mut kv1 = kv_for(&config);
        let mut kv2 = kv_for(&config);

        let a = m1.forward(&[2u32], &mut kv1).expect("forward1");
        let b = m2.forward(&[2u32], &mut kv2).expect("forward2");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-9, "{x} != {y}");
        }
    }

    #[test]
    fn gptneox_embed_returns_hidden_state() {
        let config = minimal_config();
        let mut model = make_model(&config, ROT, true);
        let mut kv = kv_for(&config);

        let h = model.embed(&[1u32], &mut kv).expect("embed");
        assert_eq!(h.len(), config.hidden_size);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    /// Two `forward()` calls in a row must not re-allocate a stale `buf_logits`
    /// into the wrong length (the `mem::take` path).
    #[test]
    fn gptneox_forward_twice_keeps_logit_length() {
        let config = minimal_config();
        let mut model = make_model(&config, ROT, true);
        let mut kv = kv_for(&config);

        let first = model.forward(&[0u32], &mut kv).expect("first");
        let second = model.forward(&[1u32], &mut kv).expect("second");
        assert_eq!(first.len(), config.vocab_size);
        assert_eq!(second.len(), config.vocab_size);
    }

    // ── Defect N4: the RoPE table is built over n_rot, not head_dim ───────────

    #[test]
    fn gptneox_rope_table_uses_rotary_dims_not_head_dim() {
        let config = minimal_config();
        let model = make_model(&config, ROT, true);
        assert_eq!(
            model.rope.half_dim,
            ROT / 2,
            "RoPE table must have n_rot/2 rotation pairs, not head_dim/2"
        );
        assert_ne!(
            model.rope.half_dim,
            HEAD_DIM / 2,
            "a head_dim-sized table is exactly defect N4"
        );
    }

    /// The frequency ladder must be `base^(-2i/n_rot)`.  With `n_rot == 4`
    /// pair 1's angle at position 1 is `10000^(-2/4)` — a *different* number
    /// from the `10000^(-2/head_dim)` the old code produced.
    #[test]
    fn gptneox_rope_frequencies_follow_n_rot() {
        let mut config = minimal_config();
        config.head_dim = 8;
        config.hidden_size = 16;
        config.num_attention_heads = 2;
        config.num_kv_heads = 2;
        let model = make_model(&config, 4, true);

        assert_eq!(model.rope.half_dim, 2);
        let want = 1.0f32 / 10000.0f32.powf(2.0 / 4.0);
        // cos at (position 1, pair 1) encodes the raw frequency.
        let got = model.rope.cos[model.rope.half_dim + 1].acos();
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    /// Elements past `n_rot` are never rotated (ggml's "fill the remain
    /// channels" loop copies them verbatim).
    #[test]
    fn gptneox_partial_rope_leaves_tail_untouched() {
        let config = minimal_config();
        let model = make_model(&config, ROT, true);

        let mut head: Vec<f32> = (1..=HEAD_DIM).map(|i| i as f32).collect();
        let original = head.clone();
        model
            .rope
            .try_apply(&mut head[..ROT], 1)
            .expect("rope apply");

        assert!(
            (0..ROT).any(|i| (head[i] - original[i]).abs() > 1e-6),
            "the rotated prefix must change at position 1"
        );
        for i in ROT..HEAD_DIM {
            assert!(
                (head[i] - original[i]).abs() < 1e-9,
                "dim {i} outside the rotary range must be untouched"
            );
        }
    }

    // ── Defect N3: both residual formulas exist and differ ────────────────────

    #[test]
    fn gptneox_parallel_and_sequential_residual_differ() {
        let config = minimal_config();
        // Non-constant weights: with an all-equal FFN the two formulas would
        // coincide, which would make this assertion vacuous.
        let mut parallel = make_model(&config, ROT, true);
        let mut sequential = make_model(&config, ROT, false);
        randomize(&mut parallel);
        randomize(&mut sequential);

        let mut kv1 = kv_for(&config);
        let mut kv2 = kv_for(&config);
        let a = parallel.forward(&[1u32], &mut kv1).expect("parallel");
        let b = sequential.forward(&[1u32], &mut kv2).expect("sequential");

        let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(
            differs,
            "parallel and sequential residual must not produce identical logits: {a:?} vs {b:?}"
        );
    }

    /// Replace the constant test weights with a deterministic varying pattern.
    fn randomize(model: &mut GptNeoxModel) {
        let hidden = model.config.hidden_size;
        let ffn = model.config.intermediate_size;
        let attn_dim = model.config.num_attention_heads * model.config.head_dim;
        let kv_dim = model.config.num_kv_heads * model.config.head_dim;
        let qkv_total = attn_dim + 2 * kv_dim;
        let dispatcher = KernelDispatcher::new();

        for (idx, layer) in model.layers.iter_mut().enumerate() {
            let seed = idx as u32 + 1;
            *layer = GptNeoxLayer::new(
                &dispatcher,
                GptNeoxLayerWeights {
                    attn_norm: LayerNorm::new(
                        pattern(hidden, seed, 1.0, 0.1),
                        Some(pattern(hidden, seed + 7, 0.0, 0.05)),
                        1e-5,
                    ),
                    attn_qkv: f32_linear(
                        &pattern(qkv_total * hidden, seed + 11, 0.0, 0.2),
                        qkv_total,
                        hidden,
                    ),
                    attn_qkv_bias: pattern(qkv_total, seed + 13, 0.0, 0.3),
                    attn_output: f32_linear(
                        &pattern(hidden * attn_dim, seed + 17, 0.0, 0.2),
                        hidden,
                        attn_dim,
                    ),
                    attn_out_bias: pattern(hidden, seed + 19, 0.0, 0.3),
                    ffn_norm: LayerNorm::new(
                        pattern(hidden, seed + 23, 1.0, 0.1),
                        Some(pattern(hidden, seed + 29, 0.0, 0.05)),
                        1e-5,
                    ),
                    ffn_up: f32_linear(&pattern(ffn * hidden, seed + 31, 0.0, 0.2), ffn, hidden),
                    ffn_up_bias: pattern(ffn, seed + 37, 0.0, 0.3),
                    ffn_down: f32_linear(&pattern(hidden * ffn, seed + 41, 0.0, 0.2), hidden, ffn),
                    ffn_down_bias: pattern(hidden, seed + 43, 0.0, 0.3),
                },
            )
            .expect("randomized layer");
        }
    }

    // ── Defect N2: biases actually reach the arithmetic ───────────────────────

    #[test]
    fn gptneox_qkv_bias_changes_the_output() {
        let config = minimal_config();
        let mut zero_bias = make_model(&config, ROT, true);
        let mut with_bias = make_model(&config, ROT, true);
        for layer in with_bias.layers.iter_mut() {
            for (i, b) in layer.attn_qkv_bias.iter_mut().enumerate() {
                *b = 0.05 * (i as f32 + 1.0);
            }
        }

        let mut kv1 = kv_for(&config);
        let mut kv2 = kv_for(&config);
        let a = zero_bias.forward(&[0u32], &mut kv1).expect("zero bias");
        let b = with_bias.forward(&[0u32], &mut kv2).expect("with bias");

        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "attn_qkv.bias must influence the logits"
        );
    }

    #[test]
    fn gptneox_ffn_biases_change_the_output() {
        let config = minimal_config();
        let mut zero_bias = make_model(&config, ROT, true);
        let mut with_bias = make_model(&config, ROT, true);
        // Per-dimension varying values: a bias that is constant across the
        // hidden dimension is removed again by the mean-subtraction inside the
        // final LayerNorm, so a `fill()` would prove nothing.
        for layer in with_bias.layers.iter_mut() {
            let ffn = layer.ffn_up_bias.len();
            let hidden = layer.ffn_down_bias.len();
            layer
                .ffn_up_bias
                .copy_from_slice(&pattern(ffn, 61, 0.0, 0.5));
            layer
                .ffn_down_bias
                .copy_from_slice(&pattern(hidden, 67, 0.0, 0.5));
            let attn_out = layer.attn_out_bias.len();
            layer
                .attn_out_bias
                .copy_from_slice(&pattern(attn_out, 71, 0.0, 0.5));
        }

        let mut kv1 = kv_for(&config);
        let mut kv2 = kv_for(&config);
        let a = zero_bias.forward(&[0u32], &mut kv1).expect("zero bias");
        let b = with_bias.forward(&[0u32], &mut kv2).expect("with bias");

        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "ffn/attn output biases must influence the logits"
        );
    }

    // ── Defect N5 + shared guards ─────────────────────────────────────────────

    #[test]
    fn gptneox_rejects_head_geometry_mismatch() {
        let mut config = minimal_config();
        config.head_dim = HEAD_DIM + 1; // 2 * 9 = 18 != hidden_size 16
        let err = validate_gpt_neox_shapes(&config);
        match err {
            Err(ArchError::ConfigMismatch { param, .. }) => {
                assert!(param.contains("head_count"), "unexpected param {param}");
            }
            other => panic!("expected ConfigMismatch, got {other:?}"),
        }
    }

    #[test]
    fn gptneox_rejects_odd_or_oversized_rotary_dims() {
        assert!(validate_rotary_dims(3, 8).is_err(), "odd n_rot");
        assert!(validate_rotary_dims(10, 8).is_err(), "n_rot > head_dim");
        assert!(validate_rotary_dims(8, 8).is_ok(), "full rotary is legal");
        assert!(validate_rotary_dims(0, 8).is_ok(), "no rotation is legal");
    }

    #[test]
    fn gptneox_rejects_wrong_sized_token_embedding() {
        let config = minimal_config();
        let layer = make_test_layer(&config).expect("layer");
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let err = GptNeoxModel::new(
            config.clone(),
            ROT,
            true,
            vec![0.0f32; vocab * hidden - 1],
            vec![layer],
            LayerNorm::new(vec![1.0f32; hidden], Some(vec![0.0f32; hidden]), 1e-5),
            f32_linear(&vec![0.0f32; vocab * hidden], vocab, hidden),
        )
        .err();
        assert!(
            matches!(err, Some(ArchError::InvalidShape { .. })),
            "expected InvalidShape, got {err:?}"
        );
    }

    #[test]
    fn gptneox_out_of_vocabulary_token_errors() {
        let config = minimal_config();
        let mut model = make_model(&config, ROT, true);
        let mut kv = kv_for(&config);
        let err = model.forward(&[config.vocab_size as u32], &mut kv);
        assert!(
            matches!(err, Err(ArchError::ConfigMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn gptneox_overlong_prompt_errors() {
        let config = minimal_config();
        let mut model = make_model(&config, ROT, true);
        let mut kv = kv_for(&config);
        let tokens = vec![0u32; config.max_context_length + 1];
        let err = model.forward(&tokens, &mut kv);
        assert!(
            matches!(err, Err(ArchError::ConfigMismatch { .. })),
            "{err:?}"
        );
    }
}
