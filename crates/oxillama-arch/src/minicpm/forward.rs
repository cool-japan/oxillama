//! MiniCPM transformer forward pass.
//!
//! MiniCPM (base) is LLaMA plus three scalar scales — see
//! [`MiniCpmConfig`] for where each one
//! comes from in the GGUF and in llama.cpp.  Everything else (RMSNorm,
//! `Norm`-style RoPE, GQA, SwiGLU) is standard LLaMA.
//!
//! ## Forward per token
//!
//! ```text
//!   x = token_embd[token] * embedding_scale          // once, before layer 0
//!   for each layer:
//!     residual = x
//!     h = rms_norm(x, attn_norm)
//!     q, k, v  = h @ Wq, h @ Wk, h @ Wv
//!     q, k     = rope(q, pos), rope(k, pos)
//!     attn_out = sdpa(q, k, v, kv_cache) @ Wo
//!     x = residual + attn_out * residual_scale       // scale #1
//!     residual = x
//!     h = rms_norm(x, ffn_norm)
//!     ffn_out = swiglu(h, Wgate, Wup, Wdown)
//!     x = residual + ffn_out * residual_scale        // scale #2
//!   x = rms_norm(x, output_norm)
//!   logits = (x @ Woutput) * (1.0 / logit_scale)     // scale #3
//! ```
//!
//! The residual placement mirrors `llm_build_granite::build_layer_ffn`
//! (`~/work/refs/llama.cpp/src/models/granite.cpp`), which
//! `LLM_ARCH_MINICPM` dispatches to; the logit scale mirrors the
//! `ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale)` at the end of the
//! same file's constructor.
//!
//! Attention itself is unscaled beyond the usual `1 / sqrt(head_dim)`:
//! `f_attention_scale` is a Granite-MoE field that MiniCPM never sets, so
//! `granite.cpp`'s `hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(...)`
//! always takes the ordinary branch.

use std::sync::Arc;

use oxillama_quant::{KernelDispatcher, QuantKernel};

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::{rope_style_for_arch, ModelConfig};
use crate::error::{ArchError, ArchResult};
use crate::minicpm::config::{MiniCpmConfig, MINICPM_ARCH};
use crate::traits::{ForwardPass, KvCacheAccess};

// ── Layer storage ─────────────────────────────────────────────────────────────

/// A single MiniCPM (LLaMA-style) transformer layer.
///
/// Every quantization kernel is resolved **once**, at load time, and stored
/// alongside the weight it decodes.  Dispatching per token — which this layer
/// used to do — walks the full tensor-type ladder and heap-allocates a
/// `Box<dyn QuantKernel>` seven times per layer per token.
pub struct MiniCpmLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// Query projection `[n_heads * head_dim, hidden_size]`.
    pub attn_q: QuantLinear,
    /// Key projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantLinear,
    /// Value projection `[n_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantLinear,
    /// Attention output projection `[hidden_size, n_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// FFN gate projection (SwiGLU) `[intermediate_size, hidden_size]`.
    pub ffn_gate: QuantLinear,
    /// FFN up projection (SwiGLU) `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,

    attn_q_kernel: Arc<dyn QuantKernel>,
    attn_k_kernel: Arc<dyn QuantKernel>,
    attn_v_kernel: Arc<dyn QuantKernel>,
    attn_output_kernel: Arc<dyn QuantKernel>,
    ffn_gate_kernel: Arc<dyn QuantKernel>,
    ffn_up_kernel: Arc<dyn QuantKernel>,
    ffn_down_kernel: Arc<dyn QuantKernel>,
}

impl MiniCpmLayer {
    /// Assemble a layer, resolving every weight's kernel up front.
    ///
    /// # Errors
    ///
    /// [`ArchError::Quant`] when the dispatcher has no kernel for one of the
    /// weight tensor types.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attn_norm: RmsNorm,
        ffn_norm: RmsNorm,
        attn_q: QuantLinear,
        attn_k: QuantLinear,
        attn_v: QuantLinear,
        attn_output: QuantLinear,
        ffn_gate: QuantLinear,
        ffn_up: QuantLinear,
        ffn_down: QuantLinear,
        dispatcher: &KernelDispatcher,
    ) -> ArchResult<Self> {
        Ok(Self {
            attn_q_kernel: resolve_kernel(dispatcher, &attn_q)?,
            attn_k_kernel: resolve_kernel(dispatcher, &attn_k)?,
            attn_v_kernel: resolve_kernel(dispatcher, &attn_v)?,
            attn_output_kernel: resolve_kernel(dispatcher, &attn_output)?,
            ffn_gate_kernel: resolve_kernel(dispatcher, &ffn_gate)?,
            ffn_up_kernel: resolve_kernel(dispatcher, &ffn_up)?,
            ffn_down_kernel: resolve_kernel(dispatcher, &ffn_down)?,
            attn_norm,
            ffn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_gate,
            ffn_up,
            ffn_down,
        })
    }
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
pub(crate) fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Numerically stable in-place softmax.
///
/// Defined locally rather than imported from `crate::llama`: the `minicpm`
/// Cargo feature does not imply `llama`, so `--no-default-features --features
/// minicpm` used to fail to compile on that import alone.
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

// ── Full model ────────────────────────────────────────────────────────────────

/// Loaded MiniCPM model capable of running forward passes.
pub struct MiniCpmForward {
    /// MiniCPM-specific hyperparameters.
    pub cfg: MiniCpmConfig,
    /// Dequantised token embedding table `[vocab_size * hidden_size]`.
    pub token_embd: Vec<f32>,
    /// All transformer layers.
    pub layers: Vec<MiniCpmLayer>,
    /// Final output RMSNorm.
    pub output_norm: RmsNorm,
    /// LM head, kept quantized (tied to `token_embd.weight` when the
    /// checkpoint ships no standalone `output.weight`).
    pub output: QuantLinear,

    /// Generic configuration, retained for the shared context/vocabulary
    /// guards in [`crate::common::attention`].
    ///
    /// Its `max_context_length` is normalised to the same clamped value the
    /// score buffer is sized with, so the guard and the buffer can never
    /// disagree.
    model_config: ModelConfig,
    output_kernel: Arc<dyn QuantKernel>,
    rope_table: RopeTable,

    // ── Scratch buffers (reused across tokens) ────────────────────────────
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_normed: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated per-head attention outputs `[n_heads * head_dim]`.
    buf_attn_heads: Vec<f32>,
    /// Attention output after the `Wo` projection `[hidden_size]`.
    buf_attn_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl MiniCpmForward {
    /// Construct a [`MiniCpmForward`] from pre-loaded components.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidShape`] when `token_embd` is not exactly
    ///   `vocab_size * hidden_size` elements — an undersized table would make
    ///   an in-range token id read past the end of the matrix.
    /// * [`ArchError::Quant`] when the LM head's tensor type has no kernel.
    pub fn new(
        cfg: MiniCpmConfig,
        model_config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<MiniCpmLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let max_ctx = cfg.max_context_length;

        let expected_embd = vocab
            .checked_mul(hidden)
            .ok_or_else(|| ArchError::InvalidConfig {
                detail: format!("minicpm: vocab_size {vocab} * hidden_size {hidden} overflows"),
            })?;
        if token_embd.len() != expected_embd {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![vocab, hidden],
                got: vec![token_embd.len()],
            });
        }
        if layers.len() != cfg.n_layers {
            return Err(ArchError::InvalidShape {
                name: "minicpm.block_count".to_string(),
                expected: vec![cfg.n_layers],
                got: vec![layers.len()],
            });
        }

        let dispatcher = KernelDispatcher::new();
        let output_kernel = resolve_kernel(&dispatcher, &output)?;

        // MiniCPM is `RopeStyle::Norm` in `crate::config::rope_style_for_arch`
        // (llama.cpp's `LLAMA_ROPE_TYPE_NORM` list includes LLM_ARCH_MINICPM):
        // its Q/K rows were permuted at conversion time, so consecutive pairs
        // `(x[2i], x[2i+1])` rotate together.  Never hard-code NeoX here.
        //
        // The selector keys on `ModelConfig::architecture`, and an unknown or
        // empty id falls through to the `Neox` default — which would silently
        // rotate the wrong element pairs for a hand-built configuration that
        // never went through `ModelConfig::from_metadata`.  Pin the lookup to
        // MiniCPM's own id in that case; MiniCPM is unconditionally `Norm`.
        let rope_style = if model_config.architecture.is_empty() {
            rope_style_for_arch(MINICPM_ARCH)
        } else {
            model_config.rope_style()
        };
        let rope_table =
            RopeTable::new_standard_with_style(head_dim, max_ctx, cfg.rope_freq_base, rope_style);

        // Keep the guard's bound and the score buffer's length identical:
        // `MiniCpmConfig` clamps `max_context_length` with `.max(1)` while
        // `ModelConfig` does not, and `buf_attn_scores` is indexed by raw
        // position.
        let mut model_config = model_config;
        model_config.max_context_length = max_ctx;
        model_config.vocab_size = vocab;

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            model_config,
            output_kernel,
            rope_table,
            buf_hidden: vec![0.0; hidden],
            buf_norm: vec![0.0; hidden],
            buf_normed: vec![0.0; hidden],
            buf_q: vec![0.0; n_heads * head_dim],
            buf_k: vec![0.0; n_kv * head_dim],
            buf_v: vec![0.0; n_kv * head_dim],
            buf_attn_heads: vec![0.0; n_heads * head_dim],
            buf_attn_out: vec![0.0; hidden],
            buf_gate: vec![0.0; intermediate],
            buf_up: vec![0.0; intermediate],
            buf_ffn_out: vec![0.0; hidden],
            buf_logits: vec![0.0; vocab],
            buf_attn_scores: vec![0.0; max_ctx],
        })
    }

    /// The generic configuration this model was loaded with.
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Write `token`'s embedding row into `buf_hidden`, scaled by
    /// `embedding_scale`.
    ///
    /// llama.cpp applies this once per token immediately after the lookup,
    /// before the first layer (`src/llama-graph.cpp::build_inp_embd`:
    /// `if (hparams.f_embedding_scale != 0.0f) cur = ggml_scale(cur,
    /// hparams.f_embedding_scale);`).  The `!= 0.0` test is llama.cpp's own
    /// "unset" sentinel, not a multiply-by-zero.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hs = self.cfg.hidden_size;
        let oob = || ArchError::ConfigMismatch {
            param: "token_id".to_string(),
            expected: format!("< vocab_size ({})", self.cfg.vocab_size),
            got: token.to_string(),
        };
        let off = (token as usize).checked_mul(hs).ok_or_else(oob)?;
        let end = off.checked_add(hs).ok_or_else(oob)?;
        let row = self.token_embd.get(off..end).ok_or_else(oob)?;
        self.buf_hidden.copy_from_slice(row);

        let scale = self.cfg.embedding_scale;
        if scale != 0.0 && (scale - 1.0).abs() > 1e-6 {
            for v in self.buf_hidden.iter_mut() {
                *v *= scale;
            }
        }
        Ok(())
    }

    /// Attention for one token, leaving the projected result in
    /// `buf_attn_out`.
    fn run_attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let n_heads = self.cfg.n_heads;
        let n_kv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.head_dim;
        // The KV cache stores one full `n_kv_heads * head_dim` row per token;
        // striding by `head_dim` alone reads head `kv_head` of the wrong token
        // for every GQA model (invisible only when `n_kv_heads == 1`).
        let kv_dim = n_kv * head_dim;
        let heads_per_kv = n_heads.checked_div(n_kv).unwrap_or(1).max(1);

        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_q
                .forward(&*layer.attn_q_kernel, &self.buf_norm, &mut self.buf_q)
                .map_err(ArchError::from)?;
            layer
                .attn_k
                .forward(&*layer.attn_k_kernel, &self.buf_norm, &mut self.buf_k)
                .map_err(ArchError::from)?;
            layer
                .attn_v
                .forward(&*layer.attn_v_kernel, &self.buf_norm, &mut self.buf_v)
                .map_err(ArchError::from)?;
        }

        for h in 0..n_heads {
            let off = h * head_dim;
            self.rope_table
                .apply(&mut self.buf_q[off..off + head_dim], position);
        }
        for h in 0..n_kv {
            let off = h * head_dim;
            self.rope_table
                .apply(&mut self.buf_k[off..off + head_dim], position);
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;
        let needed = seq_len
            .checked_mul(kv_dim)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("kv extent {seq_len} * {kv_dim} overflows"),
            })?;
        if cached_keys.len() < needed || cached_values.len() < needed {
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!(
                    "kv cache holds {} keys / {} values but position {position} needs {needed}",
                    cached_keys.len(),
                    cached_values.len()
                ),
            });
        }
        if seq_len > self.buf_attn_scores.len() {
            return Err(ArchError::ConfigMismatch {
                param: "context_length".to_string(),
                expected: format!("<= {}", self.buf_attn_scores.len()),
                got: seq_len.to_string(),
            });
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..n_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_off = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_off..k_off + head_dim];
                let mut score = 0.0f32;
                for (a, b) in q_head.iter().zip(k_vec.iter()) {
                    score += a * b;
                }
                self.buf_attn_scores[pos] = score * scale;
            }

            softmax_inplace(&mut self.buf_attn_scores[..seq_len]);

            let out_head = &mut self.buf_attn_heads[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0.0);
            for pos in 0..seq_len {
                let v_off = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_off..v_off + head_dim];
                let w = self.buf_attn_scores[pos];
                for (o, &v) in out_head.iter_mut().zip(v_vec.iter()) {
                    *o += w * v;
                }
            }
        }

        let layer = &self.layers[layer_idx];
        layer
            .attn_output
            .forward(
                &*layer.attn_output_kernel,
                &self.buf_attn_heads,
                &mut self.buf_attn_out,
            )
            .map_err(ArchError::from)?;

        Ok(())
    }

    /// SwiGLU FFN over `buf_norm`, leaving the result in `buf_ffn_out`.
    fn run_ffn(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        layer
            .ffn_gate
            .forward(&*layer.ffn_gate_kernel, &self.buf_norm, &mut self.buf_gate)
            .map_err(ArchError::from)?;
        layer
            .ffn_up
            .forward(&*layer.ffn_up_kernel, &self.buf_norm, &mut self.buf_up)
            .map_err(ArchError::from)?;
        swiglu_inplace(&mut self.buf_gate, &self.buf_up);
        layer
            .ffn_down
            .forward(
                &*layer.ffn_down_kernel,
                &self.buf_gate,
                &mut self.buf_ffn_out,
            )
            .map_err(ArchError::from)?;
        Ok(())
    }

    /// One transformer block, including both residual-scale multiplies.
    fn layer_forward(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        // `granite.cpp` guards both multiplies with `if (hparams
        // .f_residual_scale)` — a zero value means "unset", not "zero out the
        // residual branch".
        let residual_scale = self.cfg.residual_scale;
        let scale_residual = residual_scale != 0.0 && (residual_scale - 1.0).abs() > 1e-6;

        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
        }

        self.run_attention(layer_idx, position, kv_cache)?;

        if scale_residual {
            for a in self.buf_attn_out.iter_mut() {
                *a *= residual_scale;
            }
        }
        for (h, &a) in self.buf_hidden.iter_mut().zip(self.buf_attn_out.iter()) {
            *h += a;
        }

        {
            let layer = &self.layers[layer_idx];
            layer
                .ffn_norm
                .forward_to(&self.buf_hidden, &mut self.buf_norm);
        }

        self.run_ffn(layer_idx)?;

        if scale_residual {
            for f in self.buf_ffn_out.iter_mut() {
                *f *= residual_scale;
            }
        }
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }

    /// Run every token in `tokens` through every layer.
    ///
    /// **Every** token is processed, at its own position, and the KV cache is
    /// advanced once per token.  The previous implementation kept only
    /// `tokens[tokens.len() - 1]` "for simplicity", which threw away the whole
    /// prompt: a 200-token prefill wrote one KV entry and conditioned the
    /// model on a single token.  This is the loop shape used by
    /// `crate::bloom` and `crate::falcon`.
    ///
    /// After the call `buf_normed` holds the final RMSNorm of the **last**
    /// token's hidden state.  When `collect` is `Some`, the normed state of
    /// *every* token is appended to it in order.
    fn run_layers(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
        mut collect: Option<&mut Vec<f32>>,
    ) -> ArchResult<()> {
        if tokens.is_empty() {
            return Err(ArchError::ConfigMismatch {
                param: "tokens".to_string(),
                expected: "non-empty".to_string(),
                got: "empty".to_string(),
            });
        }

        // Both guards run before a single buffer is written: `buf_attn_scores`
        // is indexed by the raw position and the embedding table by the raw
        // token id, and prompt length + token ids are attacker-controlled
        // through the HTTP server.
        validate_token_ids(&self.model_config, tokens)?;
        let start_position = kv_cache.seq_len();
        validate_context_bounds(&self.model_config, start_position, tokens.len())?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_position + i;
            self.embed_token(token)?;

            for layer_idx in 0..self.cfg.n_layers {
                self.layer_forward(layer_idx, position, kv_cache)?;
            }
            kv_cache.advance();

            self.output_norm
                .forward_to(&self.buf_hidden, &mut self.buf_normed);
            if let Some(out) = collect.as_deref_mut() {
                out.extend_from_slice(&self.buf_normed);
            }
        }

        Ok(())
    }
}

impl ForwardPass for MiniCpmForward {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache, None)?;

        // `buf_logits` is handed out by `mem::take` below, so it may arrive
        // here empty.
        self.buf_logits.clear();
        self.buf_logits.resize(self.cfg.vocab_size, 0.0);
        self.output
            .forward(&*self.output_kernel, &self.buf_normed, &mut self.buf_logits)
            .map_err(ArchError::from)?;

        // `granite.cpp`: `cur = ggml_scale(ctx0, cur, 1.0f /
        // hparams.f_logit_scale);` — the logits are DIVIDED by `logit_scale`.
        // A zero `logit_scale` is rejected at config-parse time.
        let inv_logit_scale = 1.0 / self.cfg.logit_scale;
        if (inv_logit_scale - 1.0).abs() > 1e-6 {
            for l in self.buf_logits.iter_mut() {
                *l *= inv_logit_scale;
            }
        }

        Ok(std::mem::take(&mut self.buf_logits))
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache, None)?;
        // No logit scale here: `granite.cpp` assigns `res->t_embd` before the
        // LM head and before `ggml_scale`.
        Ok(self.buf_normed.clone())
    }

    fn embed_all(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let mut out = Vec::with_capacity(tokens.len() * self.cfg.hidden_size);
        self.run_layers(tokens, kv_cache, Some(&mut out))?;
        Ok(out)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn max_context_length(&self) -> usize {
        self.cfg.max_context_length
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rms_norm::RmsNorm;
    use crate::minicpm::config::MiniCpmConfig;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    const HIDDEN: usize = 4;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = HIDDEN / HEADS;
    const MAX_CTX: usize = 16;

    struct TestKvCache {
        kv_dim: usize,
        seq_len: usize,
        keys: std::collections::HashMap<usize, Vec<f32>>,
        vals: std::collections::HashMap<usize, Vec<f32>>,
    }

    impl TestKvCache {
        fn new(kv_dim: usize) -> Self {
            Self {
                kv_dim,
                seq_len: 0,
                keys: Default::default(),
                vals: Default::default(),
            }
        }
    }

    impl crate::traits::KvCacheAccess for TestKvCache {
        fn seq_len(&self) -> usize {
            self.seq_len
        }
        fn store_kv(
            &mut self,
            layer: usize,
            key: &[f32],
            value: &[f32],
        ) -> crate::error::ArchResult<()> {
            let offset = self.seq_len * self.kv_dim;
            let k = self.keys.entry(layer).or_default();
            k.resize(offset + self.kv_dim, 0.0);
            k[offset..offset + self.kv_dim].copy_from_slice(&key[..self.kv_dim]);
            let v = self.vals.entry(layer).or_default();
            v.resize(offset + self.kv_dim, 0.0);
            v[offset..offset + self.kv_dim].copy_from_slice(&value[..self.kv_dim]);
            Ok(())
        }
        fn get_keys(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
            self.keys.get(&layer).map(|v| v.as_slice()).ok_or_else(|| {
                crate::error::ArchError::MissingTensor {
                    name: format!("keys layer {layer}"),
                }
            })
        }
        fn get_values(&self, layer: usize) -> crate::error::ArchResult<&[f32]> {
            self.vals.get(&layer).map(|v| v.as_slice()).ok_or_else(|| {
                crate::error::ArchError::MissingTensor {
                    name: format!("values layer {layer}"),
                }
            })
        }
        fn advance(&mut self) {
            self.seq_len += 1;
        }
        fn kv_dim(&self) -> usize {
            self.kv_dim
        }
    }

    fn f32_linear(rows: usize, cols: usize, fill: f32) -> QuantLinear {
        let mut data = Vec::with_capacity(rows * cols * 4);
        for _ in 0..rows * cols {
            data.extend_from_slice(&fill.to_le_bytes());
        }
        QuantLinear::new(
            QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32),
            None,
        )
    }

    fn make_rms_norm(size: usize) -> RmsNorm {
        RmsNorm::new(vec![1.0f32; size], 1e-5)
    }

    fn make_config(vocab: usize, intermediate: usize) -> MiniCpmConfig {
        MiniCpmConfig {
            n_layers: 1,
            hidden_size: HIDDEN,
            n_heads: HEADS,
            n_kv_heads: KV_HEADS,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_context_length: MAX_CTX,
            norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            head_dim: HEAD_DIM,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            logit_scale: 1.0,
        }
    }

    fn make_model_config(vocab: usize, intermediate: usize) -> ModelConfig {
        ModelConfig {
            architecture: "minicpm".to_string(),
            hidden_size: HIDDEN,
            intermediate_size: intermediate,
            num_layers: 1,
            num_attention_heads: HEADS,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: vocab,
            max_context_length: MAX_CTX,
            ..ModelConfig::default()
        }
    }

    fn make_forward(intermediate: usize, vocab: usize) -> MiniCpmForward {
        let dispatcher = KernelDispatcher::new();
        let layer = MiniCpmLayer::new(
            make_rms_norm(HIDDEN),
            make_rms_norm(HIDDEN),
            f32_linear(HEADS * HEAD_DIM, HIDDEN, 0.05),
            f32_linear(KV_HEADS * HEAD_DIM, HIDDEN, 0.05),
            f32_linear(KV_HEADS * HEAD_DIM, HIDDEN, 0.05),
            f32_linear(HIDDEN, HEADS * HEAD_DIM, 0.05),
            f32_linear(intermediate, HIDDEN, 0.05),
            f32_linear(intermediate, HIDDEN, 0.05),
            f32_linear(HIDDEN, intermediate, 0.05),
            &dispatcher,
        )
        .expect("layer");

        let token_embd: Vec<f32> = (0..vocab * HIDDEN)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();

        MiniCpmForward::new(
            make_config(vocab, intermediate),
            make_model_config(vocab, intermediate),
            token_embd,
            vec![layer],
            make_rms_norm(HIDDEN),
            f32_linear(vocab, HIDDEN, 0.05),
        )
        .expect("forward")
    }

    #[test]
    fn test_forward_returns_correct_vocab_size() {
        let vocab = 32;
        let mut fwd = make_forward(8, vocab);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        let logits = fwd.forward(&[0u32], &mut cache).expect("forward");
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// M1: every prompt token advances the cache, not just the last one.
    #[test]
    fn test_multi_token_prefill_advances_cache_per_token() {
        let mut fwd = make_forward(8, 32);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        fwd.forward(&[1u32, 2, 3], &mut cache).expect("forward");
        assert_eq!(cache.seq_len(), 3, "three tokens must write three KV rows");
    }

    #[test]
    fn test_embed_all_returns_every_token() {
        let mut fwd = make_forward(8, 32);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        let all = fwd.embed_all(&[1u32, 2, 3], &mut cache).expect("embed_all");
        assert_eq!(all.len(), 3 * HIDDEN);
    }

    #[test]
    fn test_out_of_vocabulary_token_errors() {
        let vocab = 16;
        let mut fwd = make_forward(8, vocab);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        assert!(matches!(
            fwd.forward(&[vocab as u32], &mut cache),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }

    #[test]
    fn test_context_overflow_errors() {
        let mut fwd = make_forward(8, 32);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        let tokens = vec![1u32; MAX_CTX + 1];
        assert!(matches!(
            fwd.forward(&tokens, &mut cache),
            Err(ArchError::ConfigMismatch { .. })
        ));
    }

    #[test]
    fn test_empty_tokens_error() {
        let mut fwd = make_forward(8, 32);
        let mut cache = TestKvCache::new(KV_HEADS * HEAD_DIM);
        assert!(fwd.forward(&[], &mut cache).is_err());
    }

    #[test]
    fn test_token_embd_length_is_validated() {
        let dispatcher = KernelDispatcher::new();
        let layer = MiniCpmLayer::new(
            make_rms_norm(HIDDEN),
            make_rms_norm(HIDDEN),
            f32_linear(HEADS * HEAD_DIM, HIDDEN, 0.0),
            f32_linear(KV_HEADS * HEAD_DIM, HIDDEN, 0.0),
            f32_linear(KV_HEADS * HEAD_DIM, HIDDEN, 0.0),
            f32_linear(HIDDEN, HEADS * HEAD_DIM, 0.0),
            f32_linear(8, HIDDEN, 0.0),
            f32_linear(8, HIDDEN, 0.0),
            f32_linear(HIDDEN, 8, 0.0),
            &dispatcher,
        )
        .expect("layer");

        let err = MiniCpmForward::new(
            make_config(16, 8),
            make_model_config(16, 8),
            vec![0.0f32; 16 * HIDDEN - 1], // one element short
            vec![layer],
            make_rms_norm(HIDDEN),
            f32_linear(16, HIDDEN, 0.0),
        );
        assert!(matches!(err, Err(ArchError::InvalidShape { .. })));
    }

    #[test]
    fn test_vocab_size_accessor() {
        let fwd = make_forward(8, 100);
        assert_eq!(fwd.vocab_size(), 100);
    }

    #[test]
    fn test_hidden_size_accessor() {
        let fwd = make_forward(8, 100);
        assert_eq!(fwd.hidden_size(), HIDDEN);
    }

    #[test]
    fn test_max_context_length_accessor() {
        let fwd = make_forward(8, 100);
        assert_eq!(fwd.max_context_length(), MAX_CTX);
    }

    /// M2: the three scales change the computed logits.
    #[test]
    fn test_scales_change_logits() {
        let vocab = 16;
        let mut plain = make_forward(8, vocab);
        let mut cache_a = TestKvCache::new(KV_HEADS * HEAD_DIM);
        let base = plain.forward(&[1u32, 2], &mut cache_a).expect("forward");

        let mut scaled = make_forward(8, vocab);
        scaled.cfg.embedding_scale = 2.0;
        scaled.cfg.residual_scale = 0.5;
        scaled.cfg.logit_scale = 4.0;
        let mut cache_b = TestKvCache::new(KV_HEADS * HEAD_DIM);
        let other = scaled.forward(&[1u32, 2], &mut cache_b).expect("forward");

        assert_eq!(base.len(), other.len());
        assert!(
            base.iter()
                .zip(other.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "embedding/residual/logit scales must alter the logits"
        );
    }
}
