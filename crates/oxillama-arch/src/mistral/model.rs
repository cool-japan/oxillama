//! Mistral transformer forward pass implementation.
//!
//! Structurally identical to LLaMA (sequential `RMSNorm -> attn -> residual
//! -> RMSNorm -> SwiGLU FFN -> residual` blocks — verified against
//! `llm_build_llama` in llama.cpp's `src/models/llama.cpp`, since Mistral
//! checkpoints convert to GGUF arch `"llama"`; see
//! `convert_hf_to_gguf.py`'s `@ModelBase.register("MistralForCausalLM", ...)`
//! → `model_arch = gguf.MODEL_ARCH.LLAMA`) with one key addition: sliding
//! window attention. Each attention layer only attends to the last
//! `window_size` positions, reducing memory usage from O(seq_len) to
//! O(window_size).
//!
//! Architecture: embedding → N×(RMSNorm → SWA-GQA → residual → RMSNorm → SwiGLU FFN → residual) → RMSNorm → LM head
//!
//! RoPE uses llama.cpp's NORM convention (interleaved pairs `(x[2i],
//! x[2i+1])`), the same one LLaMA uses — NOT the NeoX half-split. See
//! `apply_rope_norm`.

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    dequant_to_f32, load_lm_head, load_quant_linear, load_rms_norm_weight,
};
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::common::{swa_attend_start, validate_context_bounds, validate_token_ids};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::apply_rope_norm;
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::QuantKernel;
use std::sync::Arc;

/// A single Mistral transformer layer (same structure as LLaMA).
pub struct MistralLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// Query projection [num_heads * head_dim, hidden_size].
    pub attn_q: QuantLinear,
    /// Key projection [num_kv_heads * head_dim, hidden_size].
    pub attn_k: QuantLinear,
    /// Value projection [num_kv_heads * head_dim, hidden_size].
    pub attn_v: QuantLinear,
    /// Output projection [hidden_size, num_heads * head_dim].
    pub attn_output: QuantLinear,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN gate projection [intermediate_size, hidden_size].
    pub ffn_gate: QuantLinear,
    /// FFN up projection [intermediate_size, hidden_size].
    pub ffn_up: QuantLinear,
    /// FFN down projection [hidden_size, intermediate_size].
    pub ffn_down: QuantLinear,
}

/// Complete Mistral model with sliding window attention.
pub struct MistralModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Sliding window size (None = full causal, same as LLaMA).
    pub sliding_window: Option<usize>,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<MistralLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head (unembedding) projection [vocab_size, hidden_size]. Falls
    /// back to the tied `token_embd` when the checkpoint ships no standalone
    /// `output.weight` (see [`load_mistral_from_gguf`]).
    pub output: QuantLinear,
    /// RoPE precomputed frequency table (NORM/interleaved convention — see
    /// `apply_rope_norm`).
    pub rope: RopeTable,

    // Scratch buffers
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    /// Concatenated per-head attention output, `[num_heads * head_dim]` —
    /// NOT always `hidden_size` (see `qwen3::model::Qwen3Model::buf_attn_out`'s
    /// doc comment for a worked example of the two diverging).
    buf_attn_out: Vec<f32>,
    /// Attention output projection, `[hidden_size]` — pre-allocated once
    /// instead of `vec![0.0f32; hidden_size]` per layer per token.
    buf_proj_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl MistralModel {
    /// Create a new MistralModel from preloaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<MistralLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;
        let sliding_window = config.sliding_window;

        let rope = RopeTable::new(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );

        Self {
            config,
            sliding_window,
            token_embd,
            layers,
            output_norm,
            output,
            rope,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; num_heads * head_dim],
            buf_proj_out: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        }
    }

    /// Resolve the kernel for a `QuantLinear`'s tensor type through the
    /// process-global cache (`Arc<dyn QuantKernel>`, cheap `Arc::clone` after
    /// the first lookup per type) instead of `KernelDispatcher::get_kernel`
    /// (a fresh `Box<dyn QuantKernel>` allocation on every call — this used
    /// to run 7× per layer per token).
    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Arc<dyn QuantKernel>> {
        oxillama_quant::global_dispatcher()
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Write `token`'s embedding row into `buf_hidden`.
    ///
    /// # Errors
    ///
    /// [`ArchError::ConfigMismatch`] when `token >= vocab_size` (or the
    /// embedding table is shorter than the declared vocabulary — an
    /// over-estimated `vocab_size` is a real failure mode: `config.rs` falls
    /// back to the tokenizer token-array length, then a hard-coded 32000,
    /// when the GGUF omits `{arch}.vocab_size`). This path must never panic —
    /// it runs on every token, including attacker-controlled HTTP input.
    fn embed_token(&mut self, token: u32) -> ArchResult<()> {
        let hidden_size = self.config.hidden_size;
        let offset = token as usize * hidden_size;
        let row = self
            .token_embd
            .get(offset..offset + hidden_size)
            .ok_or_else(|| ArchError::ConfigMismatch {
                param: "token id".to_string(),
                expected: format!("< {}", self.config.vocab_size),
                got: token.to_string(),
            })?;
        self.buf_hidden.copy_from_slice(row);
        Ok(())
    }

    /// Run sliding window grouped-query attention for a single layer.
    ///
    /// When `sliding_window` is set, attention is restricted to the last W
    /// positions: `swa_attend_start` (the same helper every sliding-window
    /// architecture in this crate routes through) gives the first visible
    /// key index, so this computes Q·K^T for positions in
    /// `[swa_attend_start(..) ..= pos]` instead of `[0 ..= pos]`.
    fn attention(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        // Project to Q, K, V
        let q_kernel = self.kernel_for(&layer.attn_q)?;
        let k_kernel = self.kernel_for(&layer.attn_k)?;
        let v_kernel = self.kernel_for(&layer.attn_v)?;

        layer
            .attn_q
            .forward(&*q_kernel, &self.buf_norm, &mut self.buf_q)?;
        layer
            .attn_k
            .forward(&*k_kernel, &self.buf_norm, &mut self.buf_k)?;
        layer
            .attn_v
            .forward(&*v_kernel, &self.buf_norm, &mut self.buf_v)?;

        // Apply RoPE — NORM/interleaved convention (see module doc comment).
        for h in 0..num_heads {
            let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
            apply_rope_norm(&self.rope, q_head, position);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
            apply_rope_norm(&self.rope, k_head, position);
        }

        // Store K, V in cache
        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let seq_len = position + 1;

        // Sliding window: only attend to the last `window_size` positions,
        // via the same `swa_attend_start` every other sliding-window
        // architecture in this crate uses (`effective_attention_span`'s
        // sibling), not a hand-rolled duplicate of the window arithmetic.
        let window_start = swa_attend_start(&self.config, layer_idx, position);

        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores only within the sliding window
            let window_len = seq_len - window_start;
            for pos in window_start..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_vec[d];
                }
                self.buf_attn_scores[pos - window_start] = score * scale;
            }

            // Softmax over the window
            softmax_inplace(&mut self.buf_attn_scores[..window_len]);

            // Weighted sum of V within window
            let out_head = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
            for pos in window_start..seq_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos - window_start];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Project attention output back to hidden_size, into the
        // pre-allocated `buf_proj_out` (no per-layer-per-token `vec!`).
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let layer = &self.layers[layer_idx];
        layer
            .attn_output
            .forward(&*o_kernel, &self.buf_attn_out, &mut self.buf_proj_out)?;

        // Add to residual
        for (h, &p) in self.buf_hidden.iter_mut().zip(self.buf_proj_out.iter()) {
            *h += p;
        }

        Ok(())
    }

    fn feed_forward(&mut self, layer_idx: usize) -> ArchResult<()> {
        let layer = &self.layers[layer_idx];

        let gate_kernel = self.kernel_for(&layer.ffn_gate)?;
        let up_kernel = self.kernel_for(&layer.ffn_up)?;
        let down_kernel = self.kernel_for(&layer.ffn_down)?;

        layer
            .ffn_gate
            .forward(&*gate_kernel, &self.buf_norm, &mut self.buf_gate)?;
        layer
            .ffn_up
            .forward(&*up_kernel, &self.buf_norm, &mut self.buf_up)?;

        swiglu_inplace(&mut self.buf_gate, &self.buf_up);

        layer
            .ffn_down
            .forward(&*down_kernel, &self.buf_gate, &mut self.buf_ffn_out)?;

        for (h, &f) in self.buf_hidden.iter_mut().zip(self.buf_ffn_out.iter()) {
            *h += f;
        }

        Ok(())
    }

    /// Run every token of `tokens` through all layers, leaving the last
    /// token's pre-output-norm hidden state in `self.buf_hidden`.
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

impl ForwardPass for MistralModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        self.output_norm.forward(&mut self.buf_hidden);

        // `buf_logits` may have been handed to the caller by ownership
        // (`std::mem::take`, below) on a previous call, leaving it empty —
        // restore its capacity before writing into it.
        if self.buf_logits.len() != self.config.vocab_size {
            self.buf_logits.resize(self.config.vocab_size, 0.0);
        }

        let output_kernel = self.kernel_for(&self.output)?;
        self.output
            .forward(&*output_kernel, &self.buf_hidden, &mut self.buf_logits)?;

        Ok(std::mem::take(&mut self.buf_logits))
    }

    /// Extract the post-output-norm hidden state for embedding.
    ///
    /// Identical to `forward()` up to and including `output_norm.forward()`.
    /// Stops SHORT of the LM-head projection (output.weight) that maps
    /// hidden_size → vocab_size. Sliding-window attention is preserved as in
    /// the normal forward pass. Returns a `hidden_size`-dimensional vector
    /// suitable for L2-normalised semantic embeddings.
    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        self.run_layers(tokens, kv_cache)?;

        // Final norm on the last token's hidden state.
        // Does NOT project through the LM head — returns hidden state directly.
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

    /// Attach LoRA adapters to this model's linear layers.
    ///
    /// Delegates to [`Self::apply_lora_scaled`] with `scale = 1.0`.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        self.apply_lora_scaled(lora, 1.0)
    }

    /// Attach LoRA adapters with an extra scale multiplier.
    ///
    /// Uses the same `blk.{i}.*` naming convention as LLaMA/Command-R, and
    /// [`QuantLinear::push_lora`] (accumulating) rather than `set_lora`
    /// (replacing), so a second call — or an entry from
    /// [`LoraStack`](crate::lora::LoraStack) — composes instead of
    /// clobbering.
    fn apply_lora_scaled(&mut self, lora: &LoadedLora, scale: f32) -> ArchResult<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let candidates: [(&str, &mut QuantLinear); 7] = [
                (&format!("blk.{i}.attn_q.weight"), &mut layer.attn_q),
                (&format!("blk.{i}.attn_k.weight"), &mut layer.attn_k),
                (&format!("blk.{i}.attn_v.weight"), &mut layer.attn_v),
                (
                    &format!("blk.{i}.attn_output.weight"),
                    &mut layer.attn_output,
                ),
                (&format!("blk.{i}.ffn_gate.weight"), &mut layer.ffn_gate),
                (&format!("blk.{i}.ffn_up.weight"), &mut layer.ffn_up),
                (&format!("blk.{i}.ffn_down.weight"), &mut layer.ffn_down),
            ];
            for (tensor_name, linear) in candidates {
                if let Some(adapter) = lora.get(tensor_name) {
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

/// Load a Mistral model from a `GgufModel`.
pub fn load_mistral_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<MistralModel> {
    let dispatcher = oxillama_quant::KernelDispatcher::new();

    // Load token embeddings. `dequant_to_f32` (common::loader) validates the
    // payload against the declared element count instead of indexing
    // `data[data_offset..data_offset + block_bytes]` unchecked — a truncated
    // GGUF used to panic here.
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        layers.push(MistralLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    // Load final norm and output projection
    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);

    // Falls back to the tied `token_embd.weight` when the checkpoint ships no
    // standalone `output.weight` (common for smaller Mistral fine-tunes).
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    Ok(MistralModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MI5: Mistral checkpoints convert to GGUF arch `"llama"` and are
    /// `LLAMA_ROPE_TYPE_NORM` (interleaved `(x[2i], x[2i+1])` pairs) — the
    /// same convention llama.cpp uses for LLaMA/Command-R — NOT the NeoX
    /// half-split every other RoPE-using architecture in this crate defaults
    /// to ([`RopeStyle::Neox`](crate::common::rope::RopeStyle)). This rotates
    /// a one-hot head vector both ways and confirms only NORM
    /// (`apply_rope_norm`, what `attention()` actually calls) leaves the
    /// interleaved partner non-zero.
    #[test]
    fn mi5_rope_uses_norm_not_neox_convention() {
        let table = RopeTable::new(
            16,
            8,
            10000.0,
            crate::common::rope::RopeScalingType::Standard,
            1.0,
        );

        let mut norm_x = vec![0.0f32; 16];
        norm_x[0] = 1.0;
        apply_rope_norm(&table, &mut norm_x, 3);
        assert!(
            norm_x[1].abs() > 1e-6,
            "NORM convention must rotate x[0] into consecutive partner x[1], got {norm_x:?}"
        );
        assert!(
            norm_x[8].abs() < 1e-12,
            "NORM convention must NOT touch the NeoX partner x[8], got {norm_x:?}"
        );

        let mut neox_x = vec![0.0f32; 16];
        neox_x[0] = 1.0;
        table.apply(&mut neox_x, 3);
        assert!(
            neox_x[8].abs() > 1e-6,
            "sanity: NeoX convention DOES rotate x[0] into x[half], got {neox_x:?}"
        );
    }

    // ─── MI5 call-site discriminator ───────────────────────────────────────
    //
    // `mi5_rope_uses_norm_not_neox_convention` (above) only proves
    // `apply_rope_norm` itself rotates interleaved pairs — it never calls
    // `attention()`, so it cannot catch a wrong per-head slice offset or a
    // wrong `position` argument at the real call site. This test drives the
    // actual model (`attn_q.weight` projects the normed input to a
    // one-hot-per-head vector) and inspects `buf_q` after a real
    // `attention()` call at a NONZERO position, closing that gap.

    struct RopeCallSiteKv {
        kv_dim: usize,
        position: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
    }

    impl RopeCallSiteKv {
        fn new(kv_dim: usize, max_seq: usize) -> Self {
            Self {
                kv_dim,
                position: 0,
                keys: vec![0.0f32; max_seq * kv_dim],
                values: vec![0.0f32; max_seq * kv_dim],
            }
        }
    }

    impl KvCacheAccess for RopeCallSiteKv {
        fn seq_len(&self) -> usize {
            self.position
        }
        fn store_kv(&mut self, _layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
            let offset = self.position * self.kv_dim;
            self.keys[offset..offset + self.kv_dim].copy_from_slice(key);
            self.values[offset..offset + self.kv_dim].copy_from_slice(value);
            Ok(())
        }
        fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.keys[..(self.position + 1) * self.kv_dim])
        }
        fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
            Ok(&self.values[..(self.position + 1) * self.kv_dim])
        }
        fn advance(&mut self) {
            self.position += 1;
        }
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect()
    }

    /// Build a 2-head, `head_dim = 4` model whose `attn_q.weight` is zero
    /// everywhere except output row 0 (head 0, index 0) and output row 4
    /// (head 1, index 0), each picking input index 0 with weight 1.0. Since
    /// `attn_norm` is `RmsNorm` with unit weights, `buf_norm[0]` is generically
    /// non-zero for a non-zero embedding row, so pre-RoPE `buf_q` is
    /// `[c, 0, 0, 0, c, 0, 0, 0]` for some non-zero `c` — an exact
    /// one-hot-per-head vector.
    fn build_rope_call_site_test_model() -> MistralModel {
        const H: usize = 8;
        const HEADS: usize = 2; // head_dim = H / HEADS = 4
        const FFN: usize = 8;
        const VOCAB: usize = 3;

        let mut w = oxillama_gguf::GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            oxillama_gguf::MetadataValue::String("mistral".to_string()),
        );
        w.add_metadata(
            "mistral.embedding_length",
            oxillama_gguf::MetadataValue::Uint32(H as u32),
        );
        w.add_metadata(
            "mistral.feed_forward_length",
            oxillama_gguf::MetadataValue::Uint32(FFN as u32),
        );
        w.add_metadata(
            "mistral.block_count",
            oxillama_gguf::MetadataValue::Uint32(1),
        );
        w.add_metadata(
            "mistral.attention.head_count",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "mistral.attention.head_count_kv",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "mistral.context_length",
            oxillama_gguf::MetadataValue::Uint32(64),
        );
        w.add_metadata(
            "mistral.vocab_size",
            oxillama_gguf::MetadataValue::Uint32(VOCAB as u32),
        );
        w.add_metadata(
            "mistral.rope.freq_base",
            oxillama_gguf::MetadataValue::Float32(10000.0),
        );

        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let ones = |n: usize| f32_bytes(&vec![1.0f32; n]);
        let zeros = |n: usize| f32_bytes(&vec![0.0f32; n]);

        // attn_q: zero except out_idx {0, 4} <- in_idx 0, weight 1.0.
        let mut q_weight = vec![0.0f32; H * H];
        q_weight[0] = 1.0; // out_idx 0 (head 0, index 0) <- in_idx 0
        q_weight[4 * H] = 1.0; // out_idx 4 (head 1, index 0) <- in_idx 0

        w.add_tensor(
            "token_embd.weight",
            &[H as u64, VOCAB as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(VOCAB * H)),
        );
        w.add_tensor(
            "blk.0.attn_norm.weight",
            &[H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &ones(H),
        );
        w.add_tensor(
            "blk.0.attn_q.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&q_weight),
        );
        w.add_tensor(
            "blk.0.attn_k.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H * H),
        );
        w.add_tensor(
            "blk.0.attn_v.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H * H),
        );
        w.add_tensor(
            "blk.0.attn_output.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H * H),
        );
        w.add_tensor(
            "blk.0.ffn_norm.weight",
            &[H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &ones(H),
        );
        w.add_tensor(
            "blk.0.ffn_gate.weight",
            &[H as u64, FFN as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H * FFN),
        );
        w.add_tensor(
            "blk.0.ffn_up.weight",
            &[H as u64, FFN as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H * FFN),
        );
        w.add_tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(FFN * H),
        );
        w.add_tensor(
            "output_norm.weight",
            &[H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &ones(H),
        );
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(VOCAB * H),
        );

        let mut bytes = Vec::new();
        w.write_to(&mut bytes)
            .expect("synthetic mistral GGUF must serialize");
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("must parse");
        let config =
            ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
        load_mistral_from_gguf(&gguf, &config).expect("load")
    }

    /// MI5 (call site): at a non-zero position, `attention()` must rotate
    /// EACH head's index-0 value into that SAME head's index-1 (the
    /// interleaved NORM partner), never into index `head_dim/2` (the NeoX
    /// partner), and must do so identically for every head (proving the
    /// per-head slice offset `h * head_dim..(h + 1) * head_dim` — not a
    /// fixed or off-by-one slice — is what gets passed to `apply_rope_norm`).
    #[test]
    fn mi5_attention_rotates_every_head_with_norm_convention_at_the_real_call_site() {
        let mut model = build_rope_call_site_test_model();
        let mut kv = RopeCallSiteKv::new(4 * 2, 8);

        // Position 0: RoPE angle is 0 for every frequency, so this call must
        // leave buf_q un-rotated — confirms the one-hot-per-head setup itself
        // (not yet informative about NORM vs NeoX).
        model.forward(&[0u32], &mut kv).expect("forward @ pos 0");
        let c = model.buf_q[0];
        assert!(
            c.abs() > 1e-6,
            "buf_q[0] must be non-zero (attn_norm(embedding)[0] projected through \
             the one-hot attn_q weight), got {c}"
        );
        assert!(
            (model.buf_q[4] - c).abs() < 1e-6,
            "head 1 index 0 must equal head 0 index 0 pre-RoPE (same one-hot \
             weight row), got buf_q[4]={} vs buf_q[0]={c}",
            model.buf_q[4]
        );

        // Position 1: RoPE angle is non-zero, so both heads' index 0 must
        // have rotated into THAT SAME head's index 1.
        model.forward(&[1u32], &mut kv).expect("forward @ pos 1");
        assert!(
            model.buf_q[1].abs() > 1e-6,
            "head 0: index 0 must rotate into index 1 (NORM/interleaved \
             partner) at a non-zero position, got buf_q={:?}",
            model.buf_q
        );
        assert!(
            model.buf_q[5].abs() > 1e-6,
            "head 1: index 0 (buf_q[4]) must rotate into index 1 of THAT HEAD \
             (buf_q[5]) — this fails if attention() uses a wrong per-head \
             slice offset, got buf_q={:?}",
            model.buf_q
        );
        for &untouched in &[
            model.buf_q[2],
            model.buf_q[3],
            model.buf_q[6],
            model.buf_q[7],
        ] {
            assert!(
                untouched.abs() < 1e-6,
                "the OTHER pair in each head (indices 2,3) must stay zero — a \
                 non-zero value there would mean NeoX half-split rotation \
                 leaked in at the call site, got buf_q={:?}",
                model.buf_q
            );
        }
    }
}
