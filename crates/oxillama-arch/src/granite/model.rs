//! Granite-3.x transformer forward pass.
//!
//! ```text
//! embedding × embedding_scale
//!   → N × ( RMSNorm → GQA+RoPE → residual += residual_scale · attn
//!           → RMSNorm → SwiGLU FFN → residual += residual_scale · ffn )
//!   → RMSNorm → LM head → logits ÷ logit_scale
//! ```
//!
//! # Relationship to LLaMA
//!
//! The tensor set is LLaMA's, verbatim: `src/llama-model.cpp` creates Granite's
//! weights in the same `case LLM_ARCH_LLAMA: case LLM_ARCH_GRANITE:` arm, and
//! `src/llama-arch.cpp` gives `LLM_ARCH_GRANITE` the same twelve
//! `LLM_TENSOR_*` entries as `LLM_ARCH_LLAMA`.  What differs is
//! [`GraniteScales`] — four scalar multipliers that this crate did not read at
//! all before, and whose absence is invisible: the model still runs and still
//! returns finite logits, just the wrong ones.
//!
//! # RoPE convention
//!
//! **NORM** — consecutive pairs `(x[2i], x[2i+1])`, like LLaMA, *not* NeoX.
//! `llama_model_rope_type` (`src/llama-model.cpp`) lists `LLM_ARCH_GRANITE`
//! alongside `LLM_ARCH_LLAMA` under `LLAMA_ROPE_TYPE_NORM`.  The table is built
//! with [`RopeStyle::Norm`] so [`RopeTable::apply`] rotates the right pairs;
//! no separate `apply_rope_norm` helper (and therefore no dependency on the
//! `llama` feature) is needed.
//!
//! Granite additionally uses `rope_finetuned` as an on/off switch for RoPE
//! (`const bool use_rope = hparams.rope_finetuned;` in
//! `src/models/granite.cpp`), defaulting to **true** — see
//! [`GraniteScales::rope_finetuned`].

use std::sync::Arc;

use oxillama_quant::QuantKernel;

use crate::common::attention::{validate_context_bounds, validate_token_ids};
use crate::common::linear::QuantLinear;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::granite::scales::GraniteScales;
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};

/// A single Granite transformer layer — identical in shape to LLaMA's.
pub struct GraniteLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection `[num_heads * head_dim, hidden_size]`.
    pub attn_q: QuantLinear,
    /// Key projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantLinear,
    /// Value projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantLinear,
    /// Attention output projection `[hidden_size, num_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection `[intermediate_size, hidden_size]`.
    pub ffn_gate: QuantLinear,
    /// FFN up projection `[intermediate_size, hidden_size]`.
    pub ffn_up: QuantLinear,
    /// FFN down projection `[hidden_size, intermediate_size]`.
    pub ffn_down: QuantLinear,

    /// Kernel for [`Self::attn_q`], resolved once at load time.
    pub(crate) attn_q_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`], resolved once at load time.
    pub(crate) attn_k_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`], resolved once at load time.
    pub(crate) attn_v_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`], resolved once at load time.
    pub(crate) attn_output_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_gate`], resolved once at load time.
    pub(crate) ffn_gate_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_up`], resolved once at load time.
    pub(crate) ffn_up_kernel: Arc<dyn QuantKernel>,
    /// Kernel for [`Self::ffn_down`], resolved once at load time.
    pub(crate) ffn_down_kernel: Arc<dyn QuantKernel>,
}

/// A fully loaded Granite-3.x model.
pub struct GraniteModel {
    /// Model configuration (geometry, RoPE base, vocabulary, …).
    pub config: ModelConfig,
    /// The four Granite multipliers plus the `rope_finetuned` switch.
    pub scales: GraniteScales,
    /// Token embedding table, row-major `[vocab_size][hidden_size]`.
    ///
    /// Validated at load: `len() == vocab_size * hidden_size`.  See
    /// the `embed_token` doc comment for why the length matters more than
    /// `config.vocab_size` does.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<GraniteLayer>,
    /// Final RMSNorm before the LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) `[vocab_size, hidden_size]`; may be the tied
    /// `token_embd.weight`.
    pub output: QuantLinear,
    /// Kernel for [`Self::output`], resolved once at load time.
    output_kernel: Arc<dyn QuantKernel>,
    /// RoPE table, built with the NORM pairing convention.
    pub rope: RopeTable,

    // ── Scratch buffers, allocated once ──────────────────────────────────
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated attention heads, `[num_heads * head_dim]` — which is not
    /// necessarily `hidden_size`.
    buf_attn_out: Vec<f32>,
    buf_proj: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_scores: Vec<f32>,
}

impl GraniteModel {
    /// Assemble a model from already-loaded weights.
    ///
    /// # Errors
    ///
    /// * [`ArchError::InvalidConfig`] when the attention geometry is degenerate
    ///   (zero heads / head_dim, or `num_heads` not a multiple of
    ///   `num_kv_heads`) — checked once here rather than on every token.
    /// * [`ArchError::InvalidShape`] when `token_embd` is not exactly
    ///   `vocab_size * hidden_size` elements.
    /// * [`ArchError::Quant`] when the LM head's tensor type has no registered
    ///   [`QuantKernel`].
    pub fn new(
        config: ModelConfig,
        scales: GraniteScales,
        token_embd: Vec<f32>,
        layers: Vec<GraniteLayer>,
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

        if hidden_size == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "granite geometry: hidden_size={hidden_size}, head_count={num_heads}, \
                     head_count_kv={num_kv_heads}, head_dim={head_dim} — all must be > 0"
                ),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(ArchError::InvalidConfig {
                detail: format!(
                    "granite GQA: head_count ({num_heads}) must be a multiple of \
                     head_count_kv ({num_kv_heads})"
                ),
            });
        }

        // The embedding table is indexed by token id, and `config.vocab_size`
        // is only a *declaration*: `ModelConfig::from_metadata` falls back to
        // the tokenizer token array and then to a hard-coded 32000 when the
        // GGUF omits `{arch}.vocab_size`.  Checking the lookup against the
        // declared vocabulary alone therefore still walks off the end of a
        // table that has fewer rows than declared.
        let expected =
            vocab_size
                .checked_mul(hidden_size)
                .ok_or_else(|| ArchError::InvalidConfig {
                    detail: format!(
                        "granite token_embd: vocab_size ({vocab_size}) × hidden_size \
                     ({hidden_size}) overflows"
                    ),
                })?;
        if token_embd.len() != expected {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![vocab_size, hidden_size],
                got: vec![token_embd.len()],
            });
        }

        // NORM pairing — Granite is `LLAMA_ROPE_TYPE_NORM` (see the module doc).
        let rope = RopeTable::new_with_style(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
            crate::common::rope::RopeStyle::Norm,
        );

        let output_kernel: Arc<dyn QuantKernel> = oxillama_quant::global_dispatcher()
            .get_kernel(output.weight.tensor_type)
            .map_err(ArchError::from)?;

        Ok(Self {
            config,
            scales,
            token_embd,
            layers,
            output_norm,
            output,
            output_kernel,
            rope,
            buf_hidden: vec![0.0f32; hidden_size],
            buf_norm: vec![0.0f32; hidden_size],
            buf_q: vec![0.0f32; num_heads * head_dim],
            buf_k: vec![0.0f32; num_kv_heads * head_dim],
            buf_v: vec![0.0f32; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0f32; num_heads * head_dim],
            buf_proj: vec![0.0f32; hidden_size],
            buf_gate: vec![0.0f32; intermediate_size],
            buf_up: vec![0.0f32; intermediate_size],
            buf_ffn_out: vec![0.0f32; hidden_size],
            buf_logits: vec![0.0f32; vocab_size],
            buf_scores: vec![0.0f32; max_ctx],
        })
    }

    /// Load the residual stream with `token`'s embedding row, scaled by
    /// `granite.embedding_scale`.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] for an id that indexes past the *actual*
    /// table, never a panic — this runs on every decoded token, including
    /// attacker-controlled HTTP input.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden_size = self.config.hidden_size;
        let rows = self.token_embd.len() / hidden_size.max(1);
        let offset =
            (token as usize)
                .checked_mul(hidden_size)
                .ok_or_else(|| ArchError::ConfigMismatch {
                    param: "token id".to_string(),
                    expected: format!("< {rows}"),
                    got: token.to_string(),
                })?;
        let row = self
            .token_embd
            .get(offset..offset.saturating_add(hidden_size))
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {rows}"),
                got: token.to_string(),
            })?;
        self.buf_hidden.copy_from_slice(row);
        // `build_inp_embd`: for Granite the embedding is scaled before the
        // first block sees it.
        self.scales.scale_embedding(&mut self.buf_hidden);
        Ok(())
    }

    /// Grouped-query attention with NORM RoPE and Granite's softmax scale.
    ///
    /// Reads the post-`attn_norm` activation from `buf_norm` and accumulates
    /// `residual_scale · Wo·attn` into `buf_hidden` — llama.cpp scales the
    /// *branch*, not the stream (`src/models/granite.cpp`, `build_layer_ffn`).
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
        // Geometry was validated in `new()`, so this division is exact.
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = self.scales.attention_scale(head_dim);
        let seq_len = position + 1;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("layer index out of range (have {})", self.layers.len()),
            })?;

        layer
            .attn_q
            .forward(&*layer.attn_q_kernel, &self.buf_norm, &mut self.buf_q)?;
        layer
            .attn_k
            .forward(&*layer.attn_k_kernel, &self.buf_norm, &mut self.buf_k)?;
        layer
            .attn_v
            .forward(&*layer.attn_v_kernel, &self.buf_norm, &mut self.buf_v)?;

        // `use_rope = hparams.rope_finetuned` — Granite can ship without RoPE.
        if self.scales.rope_finetuned {
            for h in 0..num_heads {
                self.rope
                    .try_apply(&mut self.buf_q[h * head_dim..(h + 1) * head_dim], position)?;
            }
            for h in 0..num_kv_heads {
                self.rope
                    .try_apply(&mut self.buf_k[h * head_dim..(h + 1) * head_dim], position)?;
            }
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let needed = seq_len.saturating_mul(kv_dim);
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
        if self.buf_scores.len() < seq_len {
            // `validate_context_bounds` already rejected an over-long prompt;
            // this is the belt-and-braces guard for a caller that resized the
            // context after construction.
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!(
                    "score buffer holds {} positions, need {seq_len}",
                    self.buf_scores.len()
                ),
            });
        }

        // Granite has no sliding window: every layer attends globally
        // (`LLM_ARCH_GRANITE` makes no `set_swa_pattern` call).
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for (pos, score) in self.buf_scores[..seq_len].iter_mut().enumerate() {
                let off = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[off..off + head_dim];
                let mut acc = 0.0f32;
                for (a, b) in q_head.iter().zip(k_vec.iter()) {
                    acc += a * b;
                }
                *score = acc * scale;
            }

            softmax_inplace(&mut self.buf_scores[..seq_len]);

            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0.0);
            for (pos, &w) in self.buf_scores[..seq_len].iter().enumerate() {
                let off = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[off..off + head_dim];
                for (o, &v) in out_head.iter_mut().zip(v_vec.iter()) {
                    *o += w * v;
                }
            }
        }

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: "layer disappeared between projections".to_string(),
            })?;
        layer.attn_output.forward(
            &*layer.attn_output_kernel,
            &self.buf_attn_out[..attn_dim],
            &mut self.buf_proj,
        )?;

        // `if (hparams.f_residual_scale) cur = ggml_scale(cur, f_residual_scale);`
        // then `ffn_inp = ggml_add(cur, inpSA)`.
        let residual = self.scales.residual_factor();
        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj.iter()) {
            *h += residual * p;
        }
        Ok(())
    }

    /// SwiGLU feed-forward, accumulating `residual_scale · FFN(x)` into the
    /// residual stream — the second of Granite's two scaled residual adds.
    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("layer index out of range (have {})", self.layers.len()),
            })?;

        layer
            .ffn_gate
            .forward(&*layer.ffn_gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(&*layer.ffn_up_kernel, &self.buf_norm, &mut self.buf_up)?;

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer.ffn_down.forward(
            &*layer.ffn_down_kernel,
            &self.buf_gate,
            &mut self.buf_ffn_out,
        )?;

        let residual = self.scales.residual_factor();
        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += residual * f;
        }
        Ok(())
    }

    /// Run every token through all layers, leaving the last token's
    /// pre-`output_norm` hidden state in `buf_hidden`.
    ///
    /// [`KvCacheAccess::advance`] is called **exactly once per token**, after
    /// every layer has written its K/V — the contract documented at
    /// `oxillama-runtime/src/kv_cache/mod.rs`.  Advancing per layer would make
    /// each layer see a different sequence length.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, tokens.len())?;
        validate_token_ids(&self.config, tokens)?;

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

                self.feed_forward(layer_idx)?;
            }

            kv_cache.advance();
        }
        Ok(())
    }
}

impl ForwardPass for GraniteModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);

        // `buf_logits` was handed to the caller by `mem::take` on the previous
        // call and is empty; restore its length before the LM head writes.
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }
        self.output
            .forward(&*self.output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        // `cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);`
        // A DIVISION — see `crate::granite::scales`.
        self.scales.scale_logits(&mut self.buf_logits);

        Ok(std::mem::take(&mut self.buf_logits))
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;
        self.output_norm.forward(&mut self.buf_hidden);
        // Stops before the LM head, so `logit_scale` does not apply here —
        // llama.cpp likewise records `res->t_embd` *before* the `ggml_scale`.
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

    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let candidates: [(String, &mut QuantLinear); 7] = [
                (format!("blk.{i}.attn_q.weight"), &mut layer.attn_q),
                (format!("blk.{i}.attn_k.weight"), &mut layer.attn_k),
                (format!("blk.{i}.attn_v.weight"), &mut layer.attn_v),
                (
                    format!("blk.{i}.attn_output.weight"),
                    &mut layer.attn_output,
                ),
                (format!("blk.{i}.ffn_gate.weight"), &mut layer.ffn_gate),
                (format!("blk.{i}.ffn_up.weight"), &mut layer.ffn_up),
                (format!("blk.{i}.ffn_down.weight"), &mut layer.ffn_down),
            ];
            for (tensor_name, linear) in candidates {
                if let Some(adapter) = lora.get(&tensor_name) {
                    linear.push_lora(adapter, scale);
                }
            }
        }
        Ok(())
    }

    fn unapply_all_loras(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn_q.clear_lora();
            layer.attn_k.clear_lora();
            layer.attn_v.clear_lora();
            layer.attn_output.clear_lora();
            layer.ffn_gate.clear_lora();
            layer.ffn_up.clear_lora();
            layer.ffn_down.clear_lora();
        }
    }
}

/// In-place softmax over a slice.
///
/// Numerically stabilised by subtracting the maximum; an empty slice and an
/// all-`-inf` slice both leave the buffer alone rather than producing NaNs.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return;
    }
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
    fn softmax_is_a_distribution() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn softmax_handles_empty_and_degenerate_input() {
        let mut empty: Vec<f32> = Vec::new();
        softmax_inplace(&mut empty);
        assert!(empty.is_empty());

        let mut neg_inf = vec![f32::NEG_INFINITY; 3];
        softmax_inplace(&mut neg_inf);
        assert!(
            neg_inf.iter().all(|v| v.is_infinite()),
            "an all -inf row must not become NaN"
        );
    }
}
