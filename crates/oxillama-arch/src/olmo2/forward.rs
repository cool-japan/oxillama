//! OLMo2 transformer forward pass.
//!
//! Verified against `~/work/refs/llama.cpp`:
//!
//! * `src/models/olmo2.cpp` (`llm_build_olmo2`) — the graph reproduced below.
//! * `src/llama-model.cpp`, `case LLM_ARCH_OLMO2:` in `load_tensors()` — the
//!   tensor set and, crucially, the *widths* of `attn_q_norm` (`{n_embd}`) and
//!   `attn_k_norm` (`{n_head_kv * n_embd_head}` = `n_embd_gqa`).
//!
//! ## Forward per layer
//!
//! ```text
//!   inpSA = x                                   // no pre-attention norm
//!   q = x @ Wq ; k = x @ Wk ; v = x @ Wv         // no biases in OLMo2
//!   q = rms_norm(q, q_norm)   ← ONE norm over the WHOLE n_embd     vector
//!   k = rms_norm(k, k_norm)   ← ONE norm over the WHOLE n_embd_gqa vector
//!   (only now is the vector reshaped into heads)
//!   q, k = rope(q, pos), rope(k, pos)            // full head_dim, NeoX pairing
//!   attn_out = sdpa(q, k, v, kv_cache) @ Wo
//!   attn_out = rms_norm(attn_out, attn_post_norm)   ← post-norm BEFORE residual
//!   ffn_inp = inpSA + attn_out
//!
//!   ffn_out = swiglu(ffn_inp, Wgate, Wup, Wdown) // no pre-FFN norm
//!   ffn_out = rms_norm(ffn_out, ffn_post_norm)      ← post-norm BEFORE residual
//!   x = ffn_inp + ffn_out
//! ```
//!
//! The two `build_norm(..., *_post_norm, ...)` calls in `olmo2.cpp` happen
//! *before* their `ggml_add` residual, which is what makes OLMo2 a post-norm
//! architecture; this ordering was audited against the reference and is
//! deliberately preserved.

use std::sync::Arc;

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::olmo2::config::Olmo2Config;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantKernel};

// ── Layer storage ─────────────────────────────────────────────────────────────

/// A single OLMo2 transformer layer.
///
/// Every quantization kernel is resolved **once**, at load time, and every
/// `RmsNorm` is constructed **once**, at load time.  The previous
/// implementation rebuilt four `RmsNorm` objects — each cloning its full weight
/// vector — on every layer of every token.
pub struct Olmo2Layer {
    /// Query RMSNorm, weight width `n_heads * head_dim` (= `n_embd`).
    ///
    /// Applied to the **whole** projected Q vector, not per head.
    pub attn_q_norm: RmsNorm,
    /// Key RMSNorm, weight width `n_kv_heads * head_dim` (= `n_embd_gqa`).
    ///
    /// Under GQA this is **narrower** than [`Self::attn_q_norm`]; llama.cpp
    /// creates it with `{n_head_kv * n_embd_head}`.
    pub attn_k_norm: RmsNorm,
    /// Post-attention RMSNorm (`blk.{i}.post_attention_norm.weight`).
    pub attn_post_norm: RmsNorm,
    /// Post-FFN RMSNorm (`blk.{i}.post_ffw_norm.weight`).
    pub ffn_post_norm: RmsNorm,
    /// Query projection `[n_heads * head_dim, hidden_size]`.
    pub attn_q: QuantLinear,
    /// Key projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantLinear,
    /// Value projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantLinear,
    /// Attention output projection `[hidden_size, n_heads * head_dim]`.
    pub attn_out: QuantLinear,
    /// FFN gate projection (SwiGLU) `[intermediate_size, hidden_size]`.
    pub ffn_gate: QuantLinear,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,

    // ── Kernels resolved once at load time ─────────────────────────────────
    /// Kernel for [`Self::attn_q`].
    pub attn_q_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`].
    pub attn_k_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`].
    pub attn_v_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_out`].
    pub attn_out_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_gate`].
    pub ffn_gate_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_up`].
    pub ffn_up_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_down`].
    pub ffn_down_kernel: Arc<dyn QuantKernel>,
}

// ── Full model ────────────────────────────────────────────────────────────────

/// Loaded OLMo2 model capable of running forward passes.
pub struct Olmo2Forward {
    /// OLMo2-specific hyperparameters.
    pub cfg: Olmo2Config,
    /// Generic model configuration, used by the shared prefill guards.
    ///
    /// `max_context_length` here is the **effective** context: the same number
    /// that sizes `Self::buf_attn_scores` and the RoPE table, so the guard
    /// can never disagree with the buffers it protects.
    pub config: ModelConfig,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<Olmo2Layer>,
    /// Final output RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head `[vocab_size, hidden_size]`, kept in its GGUF quantization.
    pub output: QuantLinear,
    /// Kernel for [`Self::output`], resolved once at load time.
    pub output_kernel: Arc<dyn QuantKernel>,
    /// Quantisation kernel dispatcher.
    pub dispatcher: KernelDispatcher,
    /// Precomputed RoPE table (full head-dim, NeoX pairing).
    rope_table: RopeTable,

    // ── Scratch buffers ────────────────────────────────────────────────────
    buf_hidden: Vec<f32>,
    buf_residual: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_concat: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

/// Resolve the kernel for a quantized linear layer once, at load time.
pub(crate) fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

fn expect_len(name: &str, actual: usize, expected: usize) -> ArchResult<()> {
    if actual != expected {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![expected],
            got: vec![actual],
        });
    }
    Ok(())
}

impl Olmo2Forward {
    /// Construct an [`Olmo2Forward`] from pre-loaded components.
    ///
    /// `max_context_length` is authoritative: it sizes the RoPE table and the
    /// attention-score buffer, and it becomes the bound enforced by
    /// [`validate_context_bounds`].
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when any norm weight or the embedding table
    /// disagrees with `cfg`.  These are the widths a malformed checkpoint would
    /// otherwise turn into an out-of-bounds index inside `RmsNorm::forward`.
    pub fn new(
        cfg: Olmo2Config,
        token_embd: Vec<f32>,
        layers: Vec<Olmo2Layer>,
        output_norm: RmsNorm,
        output: QuantLinear,
        max_context_length: usize,
    ) -> ArchResult<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv * head_dim;
        let max_ctx = max_context_length.max(1);

        // The embedding table is indexed as `token * hidden_size`; a table that
        // does not match the declared vocabulary would let an in-range token id
        // read another token's row (or run off the end).
        expect_len("token_embd.weight", token_embd.len(), vocab * hidden)?;
        expect_len("output_norm.weight", output_norm.weight.len(), hidden)?;

        if layers.len() != cfg.n_layers {
            return Err(ArchError::InvalidShape {
                name: "olmo2.layers".to_string(),
                expected: vec![cfg.n_layers],
                got: vec![layers.len()],
            });
        }

        for (i, layer) in layers.iter().enumerate() {
            expect_len(
                &format!("blk.{i}.attn_q_norm.weight"),
                layer.attn_q_norm.weight.len(),
                q_width,
            )?;
            // GQA: `attn_k_norm` is `n_head_kv * n_embd_head` wide, NOT `n_embd`.
            expect_len(
                &format!("blk.{i}.attn_k_norm.weight"),
                layer.attn_k_norm.weight.len(),
                kv_width,
            )?;
            expect_len(
                &format!("blk.{i}.post_attention_norm.weight"),
                layer.attn_post_norm.weight.len(),
                hidden,
            )?;
            expect_len(
                &format!("blk.{i}.post_ffw_norm.weight"),
                layer.ffn_post_norm.weight.len(),
                hidden,
            )?;
        }

        let dispatcher = KernelDispatcher::new();
        let output_kernel = resolve_kernel(&dispatcher, &output)?;

        let config = ModelConfig {
            architecture: "olmo2".to_string(),
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_layers: cfg.n_layers,
            num_attention_heads: n_heads,
            num_kv_heads: n_kv,
            head_dim,
            vocab_size: vocab,
            max_context_length: max_ctx,
            rms_norm_eps: cfg.norm_eps,
            rope_freq_base: cfg.rope_freq_base,
            ..ModelConfig::default()
        };

        // `GGML_ASSERT(n_embd_head == hparams.n_rot)` in `llm_build_olmo2`:
        // OLMo2 rotates the *entire* head dimension, so there is no partial
        // rotary count to thread through here.  `rope_style_for_arch("olmo2")`
        // is NeoX (note the `olmo` vs `olmo2` trap: plain `olmo` is Norm).
        let rope_table = RopeTable::new_standard_with_style(
            head_dim,
            max_ctx,
            cfg.rope_freq_base,
            config.rope_style(),
        );

        Ok(Self {
            cfg,
            config,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            dispatcher,
            rope_table,
            buf_hidden: vec![0.0; hidden],
            buf_residual: vec![0.0; hidden],
            buf_q: vec![0.0; q_width],
            buf_k: vec![0.0; kv_width],
            buf_v: vec![0.0; kv_width],
            buf_attn_concat: vec![0.0; q_width],
            buf_attn_out: vec![0.0; hidden],
            buf_gate: vec![0.0; intermediate],
            buf_up: vec![0.0; intermediate],
            buf_ffn_out: vec![0.0; hidden],
            buf_logits: vec![0.0; vocab],
            buf_attn_scores: vec![0.0; max_ctx],
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Copy `token`'s embedding row into `buf_hidden`.
    fn embed_token_into_hidden(&mut self, token: u32) -> ArchResult<()> {
        let hs = self.cfg.hidden_size;
        let off = (token as usize)
            .checked_mul(hs)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< {}", self.cfg.vocab_size),
                got: token.to_string(),
            })?;
        let row = self
            .token_embd
            .get(off..off + hs)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token_id".to_string(),
                expected: format!("< {}", self.cfg.vocab_size),
                got: token.to_string(),
            })?;
        self.buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Scaled dot-product attention for one query head.
    ///
    /// `cached_keys` / `cached_vals` are laid out `[seq_len, kv_dim]` with
    /// `kv_dim = n_kv_heads * head_dim` — the layout every `KvCacheAccess`
    /// implementation in this workspace writes (`kv_cache/mod.rs`:
    /// `offset = seq_len * kv_dim`).  Striding by `head_dim` instead, as this
    /// function used to, reads position `p`'s data from row `p + kv_head`
    /// whenever the model is GQA.
    #[allow(clippy::too_many_arguments)]
    fn sdpa(
        q: &[f32],
        cached_keys: &[f32],
        cached_vals: &[f32],
        kv_head: usize,
        kv_dim: usize,
        head_dim: usize,
        output: &mut [f32],
        scores: &mut [f32],
    ) -> ArchResult<()> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_off = kv_head * head_dim;
        let short = |need: usize, got: usize| ArchError::InvalidShape {
            name: "olmo2.kv_cache".to_string(),
            expected: vec![need],
            got: vec![got],
        };

        for (k_pos, score) in scores.iter_mut().enumerate() {
            let base = k_pos * kv_dim + kv_off;
            let k_vec = cached_keys
                .get(base..base + head_dim)
                .ok_or_else(|| short(base + head_dim, cached_keys.len()))?;
            *score = q
                .iter()
                .zip(k_vec.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * scale;
        }

        softmax_inplace(scores);

        output.fill(0.0);
        for (k_pos, &w) in scores.iter().enumerate() {
            let base = k_pos * kv_dim + kv_off;
            let v_vec = cached_vals
                .get(base..base + head_dim)
                .ok_or_else(|| short(base + head_dim, cached_vals.len()))?;
            for (o, &v) in output.iter_mut().zip(v_vec.iter()) {
                *o += w * v;
            }
        }
        Ok(())
    }

    /// Attention for one layer at one position, leaving the **post-normed**
    /// attention output in `buf_attn_out` (the residual add is the caller's).
    fn run_attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        let kv_dim = n_kv * head_dim;
        let seq_len = kv_cache.seq_len() + 1;

        if seq_len > self.buf_attn_scores.len() {
            return Err(ArchError::ConfigMismatch {
                param: "context_length".to_string(),
                expected: format!("<= {}", self.buf_attn_scores.len()),
                got: seq_len.to_string(),
            });
        }

        // Q/K/V projections from the raw residual stream — OLMo2 has no
        // pre-attention norm and no attention biases.
        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_q
                .forward(&*layer.attn_q_kernel, &self.buf_hidden, &mut self.buf_q)?;
            layer
                .attn_k
                .forward(&*layer.attn_k_kernel, &self.buf_hidden, &mut self.buf_k)?;
            layer
                .attn_v
                .forward(&*layer.attn_v_kernel, &self.buf_hidden, &mut self.buf_v)?;
        }

        // QK-norm: ONE RMSNorm over the whole projected vector, matching
        // `llm_build_olmo2` where `build_norm(Qcur, attn_q_norm, ...)` runs
        // BEFORE `ggml_reshape_3d(..., n_embd_head, n_head, n_tokens)`.
        // Doing it per head would take the RMS over `head_dim` elements
        // instead of `n_embd`, and — because `RmsNorm::forward` indexes its
        // weight relative to the slice it is handed — would reuse head 0's
        // slice of the weight for every head.
        self.layers[layer_idx].attn_q_norm.forward(&mut self.buf_q);
        self.layers[layer_idx].attn_k_norm.forward(&mut self.buf_k);

        // RoPE over the full head dimension, applied after the norm+reshape.
        for h in 0..n_heads {
            let off = h * head_dim;
            self.rope_table
                .try_apply(&mut self.buf_q[off..off + head_dim], position)?;
        }
        for h in 0..n_kv {
            let off = h * head_dim;
            self.rope_table
                .try_apply(&mut self.buf_k[off..off + head_dim], position)?;
        }

        kv_cache.store_kv(layer_idx, &self.buf_k, &self.buf_v)?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_vals = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_vals: &[f32] = &cached_vals;

        let gqa_ratio = n_heads.checked_div(n_kv).unwrap_or(1).max(1);
        for h in 0..n_heads {
            let kv_head = h / gqa_ratio;
            let off = h * head_dim;
            Self::sdpa(
                &self.buf_q[off..off + head_dim],
                cached_keys,
                cached_vals,
                kv_head,
                kv_dim,
                head_dim,
                &mut self.buf_attn_concat[off..off + head_dim],
                &mut self.buf_attn_scores[..seq_len],
            )?;
        }

        {
            let layer = &self.layers[layer_idx];
            layer.attn_out.forward(
                &*layer.attn_out_kernel,
                &self.buf_attn_concat,
                &mut self.buf_attn_out,
            )?;
        }

        // Post-attention RMSNorm — applied BEFORE the residual add
        // (`olmo2.cpp`: `build_norm(cur, attn_post_norm, ...)` then
        // `ggml_add(ctx0, cur, inpSA)`).
        self.layers[layer_idx]
            .attn_post_norm
            .forward(&mut self.buf_attn_out);

        Ok(())
    }

    /// SwiGLU FFN for one layer, leaving the **post-normed** FFN output in
    /// `buf_ffn_out`.
    ///
    /// Reads `buf_hidden` directly: OLMo2 has **no** pre-FFN norm, so the FFN
    /// input is exactly `ffn_inp = inpSA + attn_post_norm(attn_out)`.
    fn run_ffn(&mut self, layer_idx: usize) -> ArchResult<()> {
        {
            let layer = &self.layers[layer_idx];
            layer.ffn_gate.forward(
                &*layer.ffn_gate_kernel,
                &self.buf_hidden,
                &mut self.buf_gate,
            )?;
            layer
                .ffn_up
                .forward(&*layer.ffn_up_kernel, &self.buf_hidden, &mut self.buf_up)?;
        }

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        {
            let layer = &self.layers[layer_idx];
            layer.ffn_down.forward(
                &*layer.ffn_down_kernel,
                &self.buf_gate,
                &mut self.buf_ffn_out,
            )?;
        }

        // Post-FFN RMSNorm — again BEFORE the residual add.
        self.layers[layer_idx]
            .ffn_post_norm
            .forward(&mut self.buf_ffn_out);

        Ok(())
    }

    /// Run every layer for **every** token in `tokens`, leaving the final
    /// output-normed hidden state of the last token in `buf_hidden`.
    ///
    /// One KV-cache advance per token: a prompt of `n` tokens leaves the cache
    /// `n` positions longer.  The previous implementation kept only
    /// `tokens[len - 1]` and advanced once, so every prompt token but the last
    /// was silently discarded and the KV cache never contained the prefix.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        if tokens.is_empty() {
            return Err(ArchError::ConfigMismatch {
                param: "tokens".to_string(),
                expected: "non-empty".to_string(),
                got: "empty".to_string(),
            });
        }

        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, tokens.len())?;
        validate_token_ids(&self.config, tokens)?;

        let n_layers = self.layers.len();
        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token_into_hidden(token)?;

            for layer_idx in 0..n_layers {
                // inpSA
                self.buf_residual.copy_from_slice(&self.buf_hidden);

                self.run_attention(layer_idx, position, kv_cache)?;

                // ffn_inp = inpSA + attn_post_norm(attn_out)
                for ((h, &r), &a) in self
                    .buf_hidden
                    .iter_mut()
                    .zip(self.buf_residual.iter())
                    .zip(self.buf_attn_out.iter())
                {
                    *h = r + a;
                }

                // Save ffn_inp: it is BOTH the FFN input and the FFN residual.
                self.buf_residual.copy_from_slice(&self.buf_hidden);

                self.run_ffn(layer_idx)?;

                for ((h, &r), &f) in self
                    .buf_hidden
                    .iter_mut()
                    .zip(self.buf_residual.iter())
                    .zip(self.buf_ffn_out.iter())
                {
                    *h = r + f;
                }
            }

            kv_cache.advance();
        }

        self.output_norm.forward(&mut self.buf_hidden);
        Ok(())
    }
}

impl ForwardPass for Olmo2Forward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        // `buf_logits` is handed to the caller by ownership below, so restore
        // its length before the kernel writes into it.
        if self.buf_logits.len() != self.cfg.vocab_size {
            self.buf_logits.resize(self.cfg.vocab_size, 0.0);
        }
        self.output
            .forward(&*self.output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        Ok(std::mem::take(&mut self.buf_logits))
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        Ok(self.buf_hidden.clone())
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

/// In-place numerically-stable softmax.
///
/// Kept local rather than imported from `crate::llama`, which is behind the
/// `llama` feature that `olmo2` does not enable.
pub(crate) fn softmax_inplace(x: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rms_norm::RmsNorm;
    use crate::olmo2::config::Olmo2Config;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    const HIDDEN: usize = 8;
    const N_HEADS: usize = 2;
    const N_KV: usize = 1;
    const HEAD_DIM: usize = HIDDEN / N_HEADS;
    const KV_DIM: usize = N_KV * HEAD_DIM;

    struct VecKvCache {
        seq_len: usize,
        kv_dim: usize,
        keys: Vec<f32>,
        vals: Vec<f32>,
    }

    impl VecKvCache {
        fn new(kv_dim: usize) -> Self {
            Self {
                seq_len: 0,
                kv_dim,
                keys: Vec::new(),
                vals: Vec::new(),
            }
        }
    }

    impl crate::traits::KvCacheAccess for VecKvCache {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(
            &mut self,
            _layer: usize,
            key: &[f32],
            value: &[f32],
        ) -> crate::error::ArchResult<()> {
            let end = (self.seq_len + 1) * self.kv_dim;
            self.keys.resize(end, 0.0);
            self.vals.resize(end, 0.0);
            let off = self.seq_len * self.kv_dim;
            self.keys[off..end].copy_from_slice(&key[..self.kv_dim]);
            self.vals[off..end].copy_from_slice(&value[..self.kv_dim]);
            Ok(())
        }
        fn get_keys(&self, _layer: usize) -> crate::error::ArchResult<&[f32]> {
            Ok(&self.keys)
        }
        fn get_values(&self, _layer: usize) -> crate::error::ArchResult<&[f32]> {
            Ok(&self.vals)
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
        fn kv_dim(&self) -> usize {
            self.kv_dim
        }
    }

    fn zero_linear(rows: usize, cols: usize) -> QuantLinear {
        QuantLinear::new(
            QuantTensor::new(
                vec![0u8; rows * cols * 4],
                vec![rows, cols],
                GgufTensorType::F32,
            ),
            None,
        )
    }

    fn ones_norm(size: usize) -> RmsNorm {
        RmsNorm::new(vec![1.0f32; size], 1e-5)
    }

    fn make_layer(intermediate: usize) -> Olmo2Layer {
        let dispatcher = KernelDispatcher::new();
        let attn_q = zero_linear(N_HEADS * HEAD_DIM, HIDDEN);
        let attn_k = zero_linear(KV_DIM, HIDDEN);
        let attn_v = zero_linear(KV_DIM, HIDDEN);
        let attn_out = zero_linear(HIDDEN, N_HEADS * HEAD_DIM);
        let ffn_gate = zero_linear(intermediate, HIDDEN);
        let ffn_up = zero_linear(intermediate, HIDDEN);
        let ffn_down = zero_linear(HIDDEN, intermediate);

        Olmo2Layer {
            // Q norm is the FULL Q width, K norm is the (narrower) KV width.
            attn_q_norm: ones_norm(N_HEADS * HEAD_DIM),
            attn_k_norm: ones_norm(KV_DIM),
            attn_post_norm: ones_norm(HIDDEN),
            ffn_post_norm: ones_norm(HIDDEN),
            attn_q_kernel: resolve_kernel(&dispatcher, &attn_q).expect("f32 kernel"),
            attn_k_kernel: resolve_kernel(&dispatcher, &attn_k).expect("f32 kernel"),
            attn_v_kernel: resolve_kernel(&dispatcher, &attn_v).expect("f32 kernel"),
            attn_out_kernel: resolve_kernel(&dispatcher, &attn_out).expect("f32 kernel"),
            ffn_gate_kernel: resolve_kernel(&dispatcher, &ffn_gate).expect("f32 kernel"),
            ffn_up_kernel: resolve_kernel(&dispatcher, &ffn_up).expect("f32 kernel"),
            ffn_down_kernel: resolve_kernel(&dispatcher, &ffn_down).expect("f32 kernel"),
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_down,
        }
    }

    fn make_forward(intermediate: usize, vocab: usize, max_ctx: usize) -> Olmo2Forward {
        let cfg = Olmo2Config {
            n_layers: 1,
            hidden_size: HIDDEN,
            n_heads: N_HEADS,
            n_kv_heads: N_KV,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_context_length: max_ctx,
            norm_eps: 1e-5,
            rope_freq_base: 500_000.0,
            head_dim: HEAD_DIM,
        };

        Olmo2Forward::new(
            cfg,
            vec![0.0f32; vocab * HIDDEN],
            vec![make_layer(intermediate)],
            ones_norm(HIDDEN),
            zero_linear(vocab, HIDDEN),
            max_ctx,
        )
        .expect("construction must succeed")
    }

    #[test]
    fn test_forward_returns_correct_vocab_size() {
        let vocab = 32;
        let mut fwd = make_forward(16, vocab, 16);
        let mut cache = VecKvCache::new(KV_DIM);
        let logits = fwd.forward(&[0u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), vocab);
    }

    #[test]
    fn test_vocab_size_accessor() {
        let fwd = make_forward(16, 100, 16);
        assert_eq!(fwd.vocab_size(), 100);
    }

    #[test]
    fn test_hidden_size_accessor() {
        let fwd = make_forward(16, 100, 16);
        assert_eq!(fwd.hidden_size(), HIDDEN);
    }

    #[test]
    fn test_max_context_length_accessor() {
        let fwd = make_forward(16, 100, 16);
        assert_eq!(fwd.max_context_length(), 16);
    }

    /// O1: every prompt token advances the KV cache, not just the last one.
    #[test]
    fn test_multi_token_prompt_advances_cache_per_token() {
        let mut fwd = make_forward(16, 32, 16);
        let mut cache = VecKvCache::new(KV_DIM);
        fwd.forward(&[1u32, 2, 3], &mut cache).expect("forward");
        assert_eq!(cache.seq_len(), 3, "three tokens must advance the cache 3×");
    }

    /// O2: a `attn_k_norm` sized to `n_embd` instead of `n_embd_gqa` is
    /// rejected at construction rather than read out of bounds at inference.
    #[test]
    fn test_wrong_k_norm_width_is_rejected() {
        let cfg = Olmo2Config {
            n_layers: 1,
            hidden_size: HIDDEN,
            n_heads: N_HEADS,
            n_kv_heads: N_KV,
            intermediate_size: 16,
            vocab_size: 8,
            max_context_length: 16,
            norm_eps: 1e-5,
            rope_freq_base: 500_000.0,
            head_dim: HEAD_DIM,
        };
        let mut layer = make_layer(16);
        layer.attn_k_norm = ones_norm(HIDDEN); // n_embd, not n_embd_gqa
        let result = Olmo2Forward::new(
            cfg,
            vec![0.0f32; 8 * HIDDEN],
            vec![layer],
            ones_norm(HIDDEN),
            zero_linear(8, HIDDEN),
            16,
        );
        assert!(
            matches!(result, Err(ArchError::InvalidShape { .. })),
            "a mis-sized attn_k_norm must be an InvalidShape error"
        );
    }

    /// Part 3: an out-of-vocabulary token id errors instead of panicking.
    #[test]
    fn test_oov_token_is_an_error() {
        let mut fwd = make_forward(16, 8, 16);
        let mut cache = VecKvCache::new(KV_DIM);
        assert!(fwd.forward(&[99u32], &mut cache).is_err());
    }

    /// Part 3: an over-long prefill errors instead of indexing past
    /// `buf_attn_scores`.
    #[test]
    fn test_context_overflow_is_an_error() {
        let mut fwd = make_forward(16, 8, 4);
        let mut cache = VecKvCache::new(KV_DIM);
        let tokens: Vec<u32> = (0..5).map(|i| i as u32 % 8).collect();
        assert!(fwd.forward(&tokens, &mut cache).is_err());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }
}
