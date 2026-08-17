//! Falcon transformer forward pass.
//!
//! # Reference
//!
//! `~/work/refs/llama.cpp/src/models/falcon.cpp` (`llm_build_falcon`) is the
//! graph every `LLM_ARCH_FALCON` checkpoint runs — Falcon-7B, Falcon-40B and
//! Falcon-2-11B alike.  Per layer:
//!
//! ```text
//!   attn_norm = LN_1(x)
//!   qkv_in    = attn_norm_2 ? LN_2(x) : attn_norm        // LN_2 = Falcon-40B only
//!   attn_out  = W_o · Attention(RoPE(Q), RoPE(K), V)      // fused QKV, GQA, 1/sqrt(head_dim)
//!   ffn_out   = W_down · GELU(W_up · attn_norm)           // !! the *norm*, not the attn result
//!   x'        = ffn_out + attn_out + x
//! ```
//!
//! Three properties of that graph are unconditional — there is no
//! hyper-parameter branch anywhere in the function:
//!
//! * **Parallel**: the attention output, the FFN output and the residual
//!   stream are summed together.  The FFN input is `attn_norm`'s output, which
//!   the C++ flags explicitly (`// !! use the attn norm, not the result`).
//! * **RoPE**: `ggml_rope_ext` in "neox mode" on Q and K, with
//!   `GGML_ASSERT(n_embd_head == hparams.n_rot)` (full-head-dim rotary).  There
//!   is no ALiBi code path.
//! * **No `ffn_norm`**: Falcon has no second FFN normalisation to load.
//!
//! [`FalconConfig::from_model_config`] therefore always yields
//! `parallel_attn = true`, `rope = true`, `alibi = false`.  The sequential
//! branch and the ALiBi bias below are retained for hand-constructed
//! configurations only; no GGUF can select them.

use std::sync::Arc;

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::gelu::gelu_inplace;
use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::rope::RopeTable;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::falcon::config::FalconConfig;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel};

// ── Layer helpers ────────────────────────────────────────────────────────────

/// A single Falcon transformer layer.
///
/// Fields are `pub` so that
/// [`load_falcon_from_gguf`](crate::falcon::load_falcon_from_gguf) — and tests
/// — can build one with a struct literal.
pub struct FalconLayer {
    /// Pre-attention LayerNorm.
    ///
    /// Its output always feeds the FFN, and feeds the QKV projection too
    /// unless [`Self::attn_norm_2`] is present.
    pub attn_norm: LayerNorm,
    /// Falcon-40B's second pre-attention LayerNorm (`blk.{i}.attn_norm_2.*`).
    ///
    /// `TENSOR_NOT_REQUIRED` in llama.cpp.  When present, the *attention*
    /// branch reads this norm's output instead of [`Self::attn_norm`]'s; the
    /// FFN keeps reading [`Self::attn_norm`] either way.
    pub attn_norm_2: Option<LayerNorm>,
    /// Pre-FFN LayerNorm for the sequential branch.
    ///
    /// Real Falcon checkpoints never carry one (llama.cpp's
    /// `LLM_ARCH_FALCON` loader creates no `ffn_norm` tensor), so this is
    /// `None` for everything the GGUF loader produces.  It exists so that a
    /// hand-constructed sequential configuration has somewhere to put its
    /// norm — and so that its absence can be reported instead of silently
    /// papered over with `attn_norm`.
    pub ffn_norm: Option<LayerNorm>,
    /// Fused QKV projection `[(n_heads + 2*n_kv_heads) * head_dim, hidden_size]`.
    pub attn_qkv: QuantLinear,
    /// Attention output projection `[hidden_size, n_heads * head_dim]`.
    pub attn_out: QuantLinear,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,
}

/// Quantization kernels for one layer, resolved once at load time.
///
/// `KernelDispatcher::get_kernel` allocates a `Box<dyn QuantKernel>` on every
/// call.  Doing it inside the decode loop cost four allocations per layer per
/// token (60 layers × 4 = 240 for Falcon-40B).  Resolving into `Arc`s at load
/// time turns each of those into a field load.
struct FalconLayerKernels {
    attn_qkv: Arc<dyn QuantKernel>,
    attn_out: Arc<dyn QuantKernel>,
    ffn_up: Arc<dyn QuantKernel>,
    ffn_down: Arc<dyn QuantKernel>,
}

impl FalconLayerKernels {
    fn resolve(layer: &FalconLayer, dispatcher: &KernelDispatcher) -> ArchResult<Self> {
        Ok(Self {
            attn_qkv: dispatcher
                .get_kernel(layer.attn_qkv.weight.tensor_type)?
                .into(),
            attn_out: dispatcher
                .get_kernel(layer.attn_out.weight.tensor_type)?
                .into(),
            ffn_up: dispatcher
                .get_kernel(layer.ffn_up.weight.tensor_type)?
                .into(),
            ffn_down: dispatcher
                .get_kernel(layer.ffn_down.weight.tensor_type)?
                .into(),
        })
    }
}

/// Which pre-computed normalisation buffer a sub-block reads from.
///
/// Selecting by tag rather than by `&[f32]` keeps the borrow checker happy:
/// the source buffer and the destination buffer are then two distinct fields
/// of `self`, so neither needs to be cloned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormSource {
    /// `buf_norm` — the output of `attn_norm`.
    AttnNorm,
    /// `buf_norm2` — the output of `attn_norm_2` (Falcon-40B).
    AttnNorm2,
    /// `buf_ffn_norm` — the output of `ffn_norm` (sequential branch only).
    FfnNorm,
}

// ── Full model ───────────────────────────────────────────────────────────────

/// Loaded Falcon model capable of running forward passes.
pub struct FalconForward {
    /// Generic model configuration (context length, vocabulary, RoPE style).
    pub config: ModelConfig,
    /// Falcon-specific hyperparameters.
    pub cfg: FalconConfig,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<FalconLayer>,
    /// Per-layer kernels, resolved once at construction time.
    layer_kernels: Vec<FalconLayerKernels>,
    /// Final LayerNorm.
    pub output_norm: LayerNorm,
    /// LM head (tied to `token_embd.weight` on checkpoints without `output.weight`).
    pub output: QuantLinear,
    /// Kernel for [`Self::output`], resolved once at construction time.
    output_kernel: Arc<dyn QuantKernel>,
    /// Quantisation kernel dispatcher (kept for out-of-band lookups).
    pub dispatcher: KernelDispatcher,
    /// Precomputed RoPE table (present unless `cfg.rope` is false).
    rope_table: Option<RopeTable>,

    // ── Scratch buffers (reused across tokens) ───────────────────────────
    /// Residual stream `[hidden_size]`.
    buf_hidden: Vec<f32>,
    /// `attn_norm` output `[hidden_size]` — always the FFN's input.
    buf_norm: Vec<f32>,
    /// `attn_norm_2` output `[hidden_size]` — the attention branch's input on
    /// Falcon-40B.
    buf_norm2: Vec<f32>,
    /// `ffn_norm` output `[hidden_size]` (sequential branch only).
    buf_ffn_norm: Vec<f32>,
    /// Fused QKV projection output `[(n_heads + 2*n_kv_heads) * head_dim]`.
    buf_qkv: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated per-head attention outputs `[n_heads * head_dim]`.
    buf_attn_out: Vec<f32>,
    /// `attn_out` projection result `[hidden_size]`.
    buf_attn_proj: Vec<f32>,
    buf_ffn_mid: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl FalconForward {
    /// Construct a [`FalconForward`] from its components.
    ///
    /// # Errors
    ///
    /// [`ArchError::Quant`] when a weight's quantization type has no kernel.
    pub fn new(
        config: ModelConfig,
        cfg: FalconConfig,
        token_embd: Vec<f32>,
        layers: Vec<FalconLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let qkv_total = (n_heads + 2 * n_kv) * head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let max_context_length = config.max_context_length.max(1);

        let dispatcher = KernelDispatcher::new();
        let layer_kernels = layers
            .iter()
            .map(|layer| FalconLayerKernels::resolve(layer, &dispatcher))
            .collect::<ArchResult<Vec<_>>>()?;
        let output_kernel: Arc<dyn QuantKernel> =
            dispatcher.get_kernel(output.weight.tensor_type)?.into();

        // `GGML_ASSERT(n_embd_head == hparams.n_rot)` — Falcon always rotates
        // the full head, and the pairing convention is NeoX
        // (`rope_style_for_arch("falcon")`), never hard-coded here.
        let rope_table = if cfg.rope {
            Some(RopeTable::new_with_style(
                head_dim,
                max_context_length,
                cfg.rope_freq_base,
                config.rope_scaling_type,
                config.rope_scaling_factor,
                config.rope_style(),
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            cfg,
            token_embd,
            layers,
            layer_kernels,
            output_norm,
            output,
            output_kernel,
            dispatcher,
            rope_table,
            buf_hidden: vec![0.0; hidden],
            buf_norm: vec![0.0; hidden],
            buf_norm2: vec![0.0; hidden],
            buf_ffn_norm: vec![0.0; hidden],
            buf_qkv: vec![0.0; qkv_total],
            buf_q: vec![0.0; n_heads * head_dim],
            buf_k: vec![0.0; n_kv * head_dim],
            buf_v: vec![0.0; n_kv * head_dim],
            buf_attn_out: vec![0.0; n_heads * head_dim],
            buf_attn_proj: vec![0.0; hidden],
            buf_ffn_mid: vec![0.0; intermediate],
            buf_ffn_out: vec![0.0; hidden],
            buf_logits: vec![0.0; vocab],
            buf_attn_scores: vec![0.0; max_context_length],
        })
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Copy token `token`'s embedding row into `buf_hidden`.
    ///
    /// This path must never panic: `token` reaches it from the HTTP server.
    fn embed_token_into_hidden(&mut self, token: u32) -> ArchResult<()> {
        let hs = self.cfg.hidden_size;
        let oob = || ArchError::ConfigMismatch {
            param: "token_id".to_string(),
            expected: format!("< vocab_size ({})", self.cfg.vocab_size),
            got: token.to_string(),
        };
        let offset = (token as usize).checked_mul(hs).ok_or_else(oob)?;
        let end = offset.checked_add(hs).ok_or_else(oob)?;
        let row = self.token_embd.get(offset..end).ok_or_else(oob)?;
        self.buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Run fused QKV projection and split into Q/K/V buffers.
    ///
    /// Reference (`llm_build_falcon`): the three views into the fused output
    /// start at `0`, `n_embd` and `n_embd + n_embd_gqa` floats respectively —
    /// i.e. Q occupies `n_heads * head_dim` values, then K and V occupy
    /// `n_kv_heads * head_dim` each.
    ///
    /// Q → `buf_q`, K → `buf_k`, V → `buf_v`.
    fn project_qkv(&mut self, layer_idx: usize, source: NormSource) -> ArchResult<()> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: "layer index out of range".to_string(),
            })?;
        let kernels =
            self.layer_kernels
                .get(layer_idx)
                .ok_or_else(|| ArchError::ForwardPassError {
                    layer: layer_idx,
                    message: "layer kernels out of range".to_string(),
                })?;

        let input: &[f32] = match source {
            NormSource::AttnNorm => &self.buf_norm,
            NormSource::AttnNorm2 => &self.buf_norm2,
            NormSource::FfnNorm => &self.buf_ffn_norm,
        };

        layer
            .attn_qkv
            .forward(kernels.attn_qkv.as_ref(), input, &mut self.buf_qkv)
            .map_err(ArchError::from)?;

        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let q_len = n_heads * head_dim;
        let k_len = n_kv * head_dim;

        if self.buf_qkv.len() < q_len + 2 * k_len {
            return Err(ArchError::InvalidShape {
                name: format!("blk.{layer_idx}.attn_qkv.weight"),
                expected: vec![q_len + 2 * k_len],
                got: vec![self.buf_qkv.len()],
            });
        }

        self.buf_q.copy_from_slice(&self.buf_qkv[..q_len]);
        self.buf_k
            .copy_from_slice(&self.buf_qkv[q_len..q_len + k_len]);
        self.buf_v
            .copy_from_slice(&self.buf_qkv[q_len + k_len..q_len + 2 * k_len]);

        Ok(())
    }

    /// Apply ALiBi bias to attention scores for one head.
    ///
    /// Adds `-slope * (position - key_position)` to the already-scaled
    /// dot-product scores.  Unreachable for GGUF-derived configurations — see
    /// the module docs.
    fn apply_alibi(scores: &mut [f32], current_pos: usize, slope: f32) {
        for (k_pos, score) in scores.iter_mut().enumerate() {
            let distance = current_pos as f32 - k_pos as f32;
            *score -= slope * distance;
        }
    }

    /// Scaled dot-product attention for one head with optional ALiBi.
    ///
    /// `q`            — query for this head `[head_dim]`
    /// `cached_keys`  — all cached key vectors `[seq_len * n_kv_heads * head_dim]`
    /// `cached_vals`  — all cached value vectors, same layout
    /// `kv_head`      — which K/V head index to use (GQA mapping)
    /// `output`       — output slice for this head `[head_dim]`
    /// `scores`       — scratch, at least `seq_len` long
    /// `alibi_slope`  — pre-computed ALiBi slope (`0.0` → ALiBi disabled)
    #[allow(clippy::too_many_arguments)]
    fn sdpa(
        q: &[f32],
        cached_keys: &[f32],
        cached_vals: &[f32],
        kv_head: usize,
        output: &mut [f32],
        scores: &mut [f32],
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize,
        current_pos: usize,
        alibi_slope: f32,
    ) -> ArchResult<()> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Each cached position stores every K/V head back to back.
        let kv_stride = n_kv_heads * head_dim;
        let kv_off = kv_head * head_dim;

        let scores = scores
            .get_mut(..seq_len)
            .ok_or_else(|| ArchError::InvalidShape {
                name: "falcon.attn_scores".to_string(),
                expected: vec![seq_len],
                got: vec![0],
            })?;

        for (k_pos, score) in scores.iter_mut().enumerate() {
            let k_base = k_pos * kv_stride + kv_off;
            let k_vec = cached_keys.get(k_base..k_base + head_dim).ok_or_else(|| {
                ArchError::InvalidShape {
                    name: "falcon.kv_cache.keys".to_string(),
                    expected: vec![k_base + head_dim],
                    got: vec![cached_keys.len()],
                }
            })?;
            let dot: f32 = q.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum();
            *score = dot * scale;
        }

        if alibi_slope != 0.0 {
            Self::apply_alibi(scores, current_pos, alibi_slope);
        }

        // Causal masking is implicit: the cache only ever holds positions
        // `0..=current_pos`, so every entry scanned is in the past or present.

        softmax_inplace(scores);

        output.iter_mut().for_each(|x| *x = 0.0);
        for (k_pos, &w) in scores.iter().enumerate() {
            let v_base = k_pos * kv_stride + kv_off;
            let v_vec = cached_vals.get(v_base..v_base + head_dim).ok_or_else(|| {
                ArchError::InvalidShape {
                    name: "falcon.kv_cache.values".to_string(),
                    expected: vec![v_base + head_dim],
                    got: vec![cached_vals.len()],
                }
            })?;
            for (o, &v) in output.iter_mut().zip(v_vec.iter()) {
                *o += w * v;
            }
        }

        Ok(())
    }

    /// Multi-head attention for a single token at `position`.
    ///
    /// Assumes `buf_q`, `buf_k`, `buf_v` are already filled.  Writes the
    /// output-projected result into `buf_attn_proj`.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let alibi = self.cfg.alibi;

        // GQA group size: how many Q heads share one K/V head.  A crafted GGUF
        // can declare `n_kv_heads > n_heads`, which used to divide by zero.
        if n_kv == 0 || !n_heads.is_multiple_of(n_kv) {
            return Err(ArchError::ConfigMismatch {
                param: "falcon.attention.head_count_kv".to_string(),
                expected: format!("a divisor of head_count ({n_heads})"),
                got: n_kv.to_string(),
            });
        }
        let groups = n_heads / n_kv;

        // RoPE — applied to Q and K unconditionally by `llm_build_falcon`.
        if let Some(table) = self.rope_table.as_ref() {
            for h in 0..n_heads {
                let off = h * head_dim;
                table.try_apply(&mut self.buf_q[off..off + head_dim], position)?;
            }
            for h in 0..n_kv {
                let off = h * head_dim;
                table.try_apply(&mut self.buf_k[off..off + head_dim], position)?;
            }
        }

        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let seq_len = position + 1;
        if seq_len > self.buf_attn_scores.len() {
            return Err(ArchError::ConfigMismatch {
                param: "context_length".to_string(),
                expected: format!("<= {}", self.buf_attn_scores.len()),
                got: seq_len.to_string(),
            });
        }

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_vals = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_vals: &[f32] = &cached_vals;

        for h in 0..n_heads {
            let kv_head = h / groups;
            let off = h * head_dim;

            let alibi_slope = if alibi {
                FalconConfig::alibi_slope(h, n_heads)
            } else {
                0.0
            };

            // `buf_q`, `buf_attn_out` and `buf_attn_scores` are three distinct
            // fields, so the head output is written in place — no per-head
            // `vec![0.0; head_dim]` and no copy-back.
            Self::sdpa(
                &self.buf_q[off..off + head_dim],
                cached_keys,
                cached_vals,
                kv_head,
                &mut self.buf_attn_out[off..off + head_dim],
                &mut self.buf_attn_scores,
                head_dim,
                n_kv,
                seq_len,
                position,
                alibi_slope,
            )?;
        }

        // Output projection into its own buffer: `buf_attn_out` is
        // `n_heads * head_dim` wide and `buf_attn_proj` is `hidden_size` wide,
        // so they must not be swapped or aliased.
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: "layer index out of range".to_string(),
            })?;
        let kernels =
            self.layer_kernels
                .get(layer_idx)
                .ok_or_else(|| ArchError::ForwardPassError {
                    layer: layer_idx,
                    message: "layer kernels out of range".to_string(),
                })?;
        layer
            .attn_out
            .forward(
                kernels.attn_out.as_ref(),
                &self.buf_attn_out,
                &mut self.buf_attn_proj,
            )
            .map_err(ArchError::from)?;

        Ok(())
    }

    /// Feed-forward network for a single layer: `down(GELU(up(x)))`.
    ///
    /// Falcon's FFN is a plain up → GELU → down chain (`LLM_FFN_GELU`,
    /// `LLM_FFN_SEQ` with a null gate in `llm_build_falcon`), not SwiGLU.
    ///
    /// Input is read from the buffer `source` names; output lands in
    /// `buf_ffn_out`.
    fn ffn(&mut self, layer_idx: usize, source: NormSource) -> ArchResult<()> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: "layer index out of range".to_string(),
            })?;
        let kernels =
            self.layer_kernels
                .get(layer_idx)
                .ok_or_else(|| ArchError::ForwardPassError {
                    layer: layer_idx,
                    message: "layer kernels out of range".to_string(),
                })?;

        let input: &[f32] = match source {
            NormSource::AttnNorm => &self.buf_norm,
            NormSource::AttnNorm2 => &self.buf_norm2,
            NormSource::FfnNorm => &self.buf_ffn_norm,
        };

        layer
            .ffn_up
            .forward(kernels.ffn_up.as_ref(), input, &mut self.buf_ffn_mid)
            .map_err(ArchError::from)?;

        gelu_inplace(&mut self.buf_ffn_mid);

        // `buf_ffn_mid` and `buf_ffn_out` are distinct fields: the down
        // projection reads one and writes the other with no intermediate copy.
        let layer = &self.layers[layer_idx];
        let kernels = &self.layer_kernels[layer_idx];
        layer
            .ffn_down
            .forward(
                kernels.ffn_down.as_ref(),
                &self.buf_ffn_mid,
                &mut self.buf_ffn_out,
            )
            .map_err(ArchError::from)?;

        Ok(())
    }

    /// Run a single Falcon decoder layer.
    ///
    /// Implements `x' = ffn(attn_norm(x)) + attn(attn_norm_2(x) ?: attn_norm(x)) + x`.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        if layer_idx >= self.layers.len() {
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: "layer index out of range".to_string(),
            });
        }

        // 1. `attn_norm(x)` → `buf_norm`.  Borrowing the norm out of
        //    `self.layers` and writing into `self.buf_norm` touches two
        //    disjoint fields, so nothing is cloned: the previous code cloned
        //    the whole `LayerNorm` (two `hidden_size` vectors — ~64 KB per
        //    layer per token on Falcon-40B, ~4 MB per token over 60 layers).
        self.layers[layer_idx]
            .attn_norm
            .forward_to(&self.buf_hidden, &mut self.buf_norm);

        // 2. Falcon-40B only: `attn_norm_2(x)` → `buf_norm2`.  Note that the
        //    input is the *residual stream*, not `buf_norm` — both norms read
        //    `inpL` in `llm_build_falcon`.
        let qkv_source = if let Some(norm_2) = self.layers[layer_idx].attn_norm_2.as_ref() {
            norm_2.forward_to(&self.buf_hidden, &mut self.buf_norm2);
            NormSource::AttnNorm2
        } else {
            NormSource::AttnNorm
        };

        // 3. Fused QKV projection + attention → `buf_attn_proj`.
        self.project_qkv(layer_idx, qkv_source)?;
        self.attention(layer_idx, position, kv_cache)?;

        if self.cfg.parallel_attn {
            // ── Parallel (every real Falcon checkpoint) ──────────────────
            // The FFN reads `attn_norm`'s output — never `attn_norm_2`'s, and
            // never the attention result.
            self.ffn(layer_idx, NormSource::AttnNorm)?;

            for ((hidden, &attn), &ffn) in self
                .buf_hidden
                .iter_mut()
                .zip(self.buf_attn_proj.iter())
                .zip(self.buf_ffn_out.iter())
            {
                *hidden += attn + ffn;
            }
        } else {
            // ── Sequential (hand-constructed configurations only) ────────
            for (hidden, &attn) in self.buf_hidden.iter_mut().zip(self.buf_attn_proj.iter()) {
                *hidden += attn;
            }

            // A sequential graph needs a *second* normalisation before the
            // FFN.  Reusing `attn_norm` here — as this code used to do — is
            // not a fallback, it computes a different model: `attn_norm` was
            // fitted on the pre-attention residual stream.  Report the missing
            // tensor instead.
            let ffn_norm = self.layers[layer_idx].ffn_norm.as_ref().ok_or_else(|| {
                ArchError::MissingTensor {
                    name: format!(
                        "blk.{layer_idx}.ffn_norm.weight (required by the sequential branch; \
                         Falcon checkpoints are parallel and ship no ffn_norm)"
                    ),
                }
            })?;
            ffn_norm.forward_to(&self.buf_hidden, &mut self.buf_ffn_norm);

            self.ffn(layer_idx, NormSource::FfnNorm)?;

            for (hidden, &ffn) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
                *hidden += ffn;
            }
        }

        Ok(())
    }

    /// Run every transformer layer plus the final norm.
    ///
    /// On return, `buf_norm` holds the last token's post-`output_norm` hidden
    /// state — the input to both the LM head and [`ForwardPass::embed`].
    ///
    /// Both input guards live here so that neither `forward()` nor `embed()`
    /// can skip one.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        let seq_start = kv_cache.seq_len();

        // Prefill is reachable from the HTTP server with an attacker-chosen
        // prompt length, and the RoPE table / score buffer are both sized to
        // `max_context_length`.  Check before any indexing happens.
        validate_context_bounds(&self.config, seq_start, tokens.len())?;
        validate_token_ids(&self.config, tokens)?;

        let n_layers = self.cfg.n_layers;
        for (step, &token) in tokens.iter().enumerate() {
            let position = seq_start + step;

            self.embed_token_into_hidden(token)?;

            for layer_idx in 0..n_layers {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }

            kv_cache.advance();
        }

        // Final LayerNorm into `buf_norm` (a distinct field from
        // `buf_hidden`, so the input needs no clone).
        self.output_norm
            .forward_to(&self.buf_hidden, &mut self.buf_norm);

        Ok(())
    }
}

// ── ForwardPass impl ─────────────────────────────────────────────────────────

impl ForwardPass for FalconForward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        // `buf_logits` was handed to the caller by ownership on the previous
        // call (the `mem::take` below) and is therefore empty; restore its
        // length before the kernel writes into it.
        if self.buf_logits.len() != self.cfg.vocab_size {
            self.buf_logits.resize(self.cfg.vocab_size, 0.0);
        }

        self.output
            .forward(
                self.output_kernel.as_ref(),
                &self.buf_norm,
                &mut self.buf_logits,
            )
            .map_err(ArchError::from)?;

        // Ownership transfer instead of a `vocab_size`-wide clone per token.
        Ok(std::mem::take(&mut self.buf_logits))
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        Ok(self.buf_norm.clone())
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.config.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

/// Numerically stable in-place softmax.
///
/// Local to this module, matching the per-architecture convention in this
/// crate (and keeping the `falcon` feature independent of `llama`, which is
/// where this used to be imported from).
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

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::layer_norm::LayerNorm;
    use crate::error::ArchResult;
    use crate::traits::KvCacheAccess;

    // ── Minimal KV cache for tests ────────────────────────────────────────────
    struct DummyKvCache {
        seq_len: usize,
        keys: Vec<Vec<f32>>,
        vals: Vec<Vec<f32>>,
    }

    impl DummyKvCache {
        fn new(n_layers: usize) -> Self {
            Self {
                seq_len: 0,
                keys: vec![Vec::new(); n_layers],
                vals: vec![Vec::new(); n_layers],
            }
        }
    }

    impl KvCacheAccess for DummyKvCache {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            self.keys[layer].extend_from_slice(key);
            self.vals[layer].extend_from_slice(value);
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys[layer])
        }
        fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.vals[layer])
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
    }

    fn identity_norm(hidden: usize) -> LayerNorm {
        LayerNorm::new(vec![1.0; hidden], Some(vec![0.0; hidden]), 1e-5)
    }

    fn zero_linear(out_features: usize, in_features: usize) -> QuantLinear {
        use oxillama_gguf::GgufTensorType;
        use oxillama_quant::QuantTensor;

        QuantLinear::new(
            QuantTensor {
                tensor_type: GgufTensorType::F32,
                shape: vec![out_features, in_features],
                data: vec![0u8; out_features * in_features * 4].into(),
            },
            None,
        )
    }

    fn make_dummy_layer(cfg: &FalconConfig, with_norm_2: bool) -> FalconLayer {
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let qkv_total = (n_heads + 2 * n_kv) * head_dim;

        FalconLayer {
            attn_norm: identity_norm(hidden),
            attn_norm_2: with_norm_2.then(|| identity_norm(hidden)),
            ffn_norm: None,
            attn_qkv: zero_linear(qkv_total, hidden),
            attn_out: zero_linear(hidden, n_heads * head_dim),
            ffn_up: zero_linear(intermediate, hidden),
            ffn_down: zero_linear(hidden, intermediate),
        }
    }

    fn test_config(cfg: &FalconConfig) -> ModelConfig {
        ModelConfig {
            architecture: "falcon".to_string(),
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_layers: cfg.n_layers,
            num_attention_heads: cfg.n_heads,
            num_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
            max_context_length: 128,
            rms_norm_eps: cfg.norm_eps,
            rope_freq_base: cfg.rope_freq_base,
            ..ModelConfig::default()
        }
    }

    fn base_cfg() -> FalconConfig {
        FalconConfig {
            n_heads: 4,
            n_kv_heads: 2,
            n_layers: 2,
            hidden_size: 32,
            vocab_size: 64,
            intermediate_size: 64,
            norm_eps: 1e-5,
            parallel_attn: true,
            alibi: false,
            rope: true,
            rope_freq_base: 10_000.0,
            head_dim: 8,
        }
    }

    fn build(cfg: FalconConfig, layers: Vec<FalconLayer>) -> FalconForward {
        let model_config = test_config(&cfg);
        let token_embd = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let output = zero_linear(cfg.vocab_size, cfg.hidden_size);
        let output_norm = identity_norm(cfg.hidden_size);
        FalconForward::new(model_config, cfg, token_embd, layers, output_norm, output)
            .expect("FalconForward::new")
    }

    fn make_parallel_forward() -> FalconForward {
        let cfg = base_cfg();
        let layers = (0..cfg.n_layers)
            .map(|_| make_dummy_layer(&cfg, false))
            .collect();
        build(cfg, layers)
    }

    /// Falcon-40B shape: every layer carries a second pre-attention norm.
    fn make_dual_norm_forward() -> FalconForward {
        let cfg = base_cfg();
        let layers = (0..cfg.n_layers)
            .map(|_| make_dummy_layer(&cfg, true))
            .collect();
        build(cfg, layers)
    }

    #[test]
    fn test_forward_returns_vocab_size_logits() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), 64, "logits must match vocab_size");
    }

    /// F2: a layer with `attn_norm_2` runs the Falcon-40B attention input path.
    #[test]
    fn test_dual_norm_forward_runs() {
        let mut model = make_dual_norm_forward();
        assert!(model.layers[0].attn_norm_2.is_some());
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[1u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_forward_multi_token() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model
            .forward(&[0u32, 1u32, 2u32], &mut cache)
            .expect("forward");
        assert_eq!(logits.len(), 64);
    }

    /// Also guards the `mem::take` + resize dance: the second call must still
    /// return a full-width logits vector, not an empty one.
    #[test]
    fn test_kv_cache_advances_correctly() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        assert_eq!(cache.seq_len(), 0);

        let first = model.forward(&[0u32], &mut cache).expect("forward");
        assert_eq!(first.len(), 64, "first call must return vocab_size logits");
        assert_eq!(cache.seq_len(), 1);

        let second = model.forward(&[1u32], &mut cache).expect("forward");
        assert_eq!(
            second.len(),
            64,
            "second call must still return vocab_size logits after mem::take"
        );
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_vocab_size_and_hidden_size_accessors() {
        let model = make_parallel_forward();
        assert_eq!(model.vocab_size(), 64);
        assert_eq!(model.hidden_size(), 32);
        assert_eq!(model.max_context_length(), 128);
    }

    #[test]
    fn test_invalid_token_returns_error() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let result = model.forward(&[64u32], &mut cache);
        assert!(result.is_err(), "Out-of-vocab token should return an error");
    }

    #[test]
    fn test_embed_returns_hidden_size_vector() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let embedding = model.embed(&[0u32], &mut cache).expect("embed");
        assert_eq!(embedding.len(), 32);
        assert!(embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embed_rejects_out_of_vocab_token() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        assert!(model.embed(&[999u32], &mut cache).is_err());
    }

    /// F3: the sequential branch reports the norm it needs instead of silently
    /// substituting `attn_norm`, which would compute a different model.
    ///
    /// Only reachable from a hand-constructed `FalconConfig`: no GGUF-derived
    /// config sets `parallel_attn = false` (see `FalconConfig::from_model_config`).
    #[test]
    fn test_sequential_branch_without_ffn_norm_errors() {
        let mut cfg = base_cfg();
        cfg.parallel_attn = false;
        let layers = (0..cfg.n_layers)
            .map(|_| make_dummy_layer(&cfg, false))
            .collect();
        let mut model = build(cfg, layers);
        let mut cache = DummyKvCache::new(model.cfg.n_layers);

        match model.forward(&[0u32], &mut cache) {
            Err(ArchError::MissingTensor { name }) => {
                assert!(
                    name.contains("ffn_norm"),
                    "error must name the missing norm, got {name:?}"
                );
            }
            other => panic!("expected MissingTensor for the absent ffn_norm, got {other:?}"),
        }
    }

    /// The same sequential configuration succeeds once the norm is supplied.
    #[test]
    fn test_sequential_branch_with_ffn_norm_runs() {
        let mut cfg = base_cfg();
        cfg.parallel_attn = false;
        let hidden = cfg.hidden_size;
        let layers = (0..cfg.n_layers)
            .map(|_| {
                let mut layer = make_dummy_layer(&cfg, false);
                layer.ffn_norm = Some(identity_norm(hidden));
                layer
            })
            .collect();
        let mut model = build(cfg, layers);
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), 64);
    }

    /// The parallel path never needs a fallback norm: `attn_norm` feeds the
    /// FFN by design, so a `None` `ffn_norm` is not a defect there.
    #[test]
    fn test_parallel_branch_needs_no_ffn_norm() {
        let model = make_parallel_forward();
        assert!(
            model.layers.iter().all(|l| l.ffn_norm.is_none()),
            "Falcon layers carry no ffn_norm"
        );
        let mut model = model;
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        assert!(model.forward(&[0u32], &mut cache).is_ok());
    }

    /// Prefill longer than the precomputed context must be rejected, not
    /// indexed out of bounds.
    #[test]
    fn test_context_overflow_is_rejected() {
        let mut model = make_parallel_forward();
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let too_many = vec![0u32; model.max_context_length() + 1];
        assert!(model.forward(&too_many, &mut cache).is_err());
    }

    /// A `n_kv_heads` that does not divide `n_heads` used to divide by zero.
    #[test]
    fn test_bad_gqa_ratio_is_rejected() {
        let mut cfg = base_cfg();
        cfg.n_kv_heads = 3; // 4 % 3 != 0
        let layers = (0..cfg.n_layers)
            .map(|_| make_dummy_layer(&cfg, false))
            .collect();
        let mut model = build(cfg, layers);
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        assert!(matches!(
            model.forward(&[0u32], &mut cache),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }

    /// The ALiBi kernel still works when a hand-built config opts in, even
    /// though no Falcon GGUF can select it.
    #[test]
    fn test_optional_alibi_path_runs() {
        let mut cfg = base_cfg();
        cfg.alibi = true;
        cfg.rope = false;
        let layers = (0..cfg.n_layers)
            .map(|_| make_dummy_layer(&cfg, false))
            .collect();
        let mut model = build(cfg, layers);
        let mut cache = DummyKvCache::new(model.cfg.n_layers);
        let logits = model.forward(&[0u32, 1u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }
}
