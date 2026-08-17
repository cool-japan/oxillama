//! Command-R transformer forward pass implementation.
//!
//! Command-R (by Cohere) shares LLaMA's GQA + SwiGLU building blocks but its
//! decoder block is structurally **different**, not merely a renamed LLaMA
//! layer:
//!
//! - **One LayerNorm per layer** (not RMSNorm, and not two norms). Verified
//!   against `llm_build_command_r` in llama.cpp's `src/models/command-r.cpp`:
//!   `cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM,
//!   il);` — `LLM_NORM` is `ggml_norm` (mean/variance LayerNorm), not
//!   `LLM_NORM_RMS`. `LLM_ARCH_COMMAND_R`'s tensor table
//!   (`src/llama-arch.cpp`) never lists `FFN_NORM` — there is no `ffn_norm`
//!   tensor in a real Command-R checkpoint.
//! - **Parallel attention + FFN**, both fed from the *same* normed input,
//!   summed with the pre-norm residual: `cur_ffn = build_ffn(ffn_inp, ...)`
//!   then `cur = ffn_out + inpL + attn_out` (three-way add, not two
//!   sequential residual adds).
//! - Optional Q/K normalization for Command-R+ (`blk.{i}.attn_q_norm.weight`,
//!   `blk.{i}.attn_k_norm.weight`) — also `LLM_NORM`, not RMS.
//! - Optional logit scaling: final logits are multiplied by `logit_scale`
//!   (loaded from `command-r.logit_scale` in GGUF metadata, default 1.0).
//! - RoPE uses llama.cpp's NORM convention (interleaved pairs `(x[2i],
//!   x[2i+1])`), like LLaMA/Mistral — see `apply_rope_norm`.
//!
//! ## Tensor naming convention (GGUF)
//!
//! - `token_embd.weight`, `blk.{i}.attn_norm.weight` (LayerNorm, no bias),
//!   `blk.{i}.attn_q/k/v.weight`, `blk.{i}.attn_output.weight`,
//!   `blk.{i}.ffn_gate/up/down.weight`, `output_norm.weight`.
//! - `output.weight` is tied to `token_embd.weight` (llama.cpp always
//!   duplicates `TOKEN_EMBD` into `OUTPUT` for this architecture — it never
//!   creates a standalone `output.weight` tensor at all).
//! - Optional: `blk.{i}.attn_q_norm.weight` / `blk.{i}.attn_k_norm.weight`
//!   (Command-R+ only, shape `[head_dim, num_heads]` / `[head_dim,
//!   num_kv_heads]` — a distinct weight row **per head**).

use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::loader::{
    dequant_to_f32, load_lm_head, load_quant_linear, load_rms_norm_weight,
};
use crate::common::rope::RopeTable;
use crate::common::swiglu::swiglu_inplace;
use crate::common::{validate_context_bounds, validate_token_ids};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::apply_rope_norm;
use crate::lora::LoadedLora;
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::KernelDispatcher;

/// A single Command-R transformer layer (decoder block).
pub struct CommandRLayer {
    /// Pre-block LayerNorm — feeds BOTH attention and FFN (one norm, not two).
    pub attn_norm: LayerNorm,
    /// Query projection [num_heads * head_dim, hidden_size].
    pub attn_q: QuantLinear,
    /// Key projection [num_kv_heads * head_dim, hidden_size].
    pub attn_k: QuantLinear,
    /// Value projection [num_kv_heads * head_dim, hidden_size].
    pub attn_v: QuantLinear,
    /// Output projection [hidden_size, num_heads * head_dim].
    pub attn_output: QuantLinear,
    /// Optional per-head Q normalization (Command-R+ only, `n_layer >= 64`).
    ///
    /// Flat `[head_dim * num_heads]` — a distinct weight row per head, not one
    /// row shared by every head. Use `layer_norm_head_slice` to apply it,
    /// never [`LayerNorm::forward`] directly (that would broadcast head 0's
    /// weights onto every head).
    pub attn_q_norm: Option<LayerNorm>,
    /// Optional per-head K normalization (Command-R+ only). Same layout note
    /// as [`Self::attn_q_norm`], sized `[head_dim * num_kv_heads]`.
    pub attn_k_norm: Option<LayerNorm>,
    /// FFN gate projection [intermediate_size, hidden_size].
    pub ffn_gate: QuantLinear,
    /// FFN up projection [intermediate_size, hidden_size].
    pub ffn_up: QuantLinear,
    /// FFN down projection [hidden_size, intermediate_size].
    pub ffn_down: QuantLinear,
}

/// Complete Command-R model with all weights and forward pass logic.
pub struct CommandRModel {
    /// Model configuration.
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_size] stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<CommandRLayer>,
    /// Final LayerNorm before LM head.
    pub output_norm: LayerNorm,
    /// LM head (unembedding) projection [vocab_size, hidden_size]. Tied to
    /// `token_embd` on every real checkpoint (see [`load_command_r_from_gguf`]).
    pub output: QuantLinear,
    /// RoPE precomputed frequency table (NORM/interleaved convention — see
    /// `apply_rope_norm`).
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Logit scaling factor (1.0 = no scaling).
    pub logit_scale: f32,

    // Scratch buffers (reused across forward calls to avoid allocation)
    buf_hidden: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    /// Attention output projection, `[hidden_size]` — written by
    /// `attention()` and combined into `buf_hidden` by the caller, instead of
    /// `attention()` mutating the residual itself (Command-R's block needs
    /// the PRE-attention, PRE-ffn hidden state to add both branches to).
    buf_attn_proj: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
    buf_attn_scores: Vec<f32>,
}

impl CommandRModel {
    /// Create a new `CommandRModel` from preloaded weights.
    pub fn new(
        config: ModelConfig,
        token_embd: Vec<f32>,
        layers: Vec<CommandRLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
        logit_scale: f32,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_ctx = config.max_context_length;

        let rope = RopeTable::new(
            head_dim,
            max_ctx,
            config.rope_freq_base,
            config.rope_scaling_type,
            config.rope_scaling_factor,
        );
        let dispatcher = KernelDispatcher::new();

        Self {
            config,
            token_embd,
            layers,
            output_norm,
            output,
            rope,
            dispatcher,
            logit_scale,
            buf_hidden: vec![0.0; hidden_size],
            buf_norm: vec![0.0; hidden_size],
            buf_q: vec![0.0; num_heads * head_dim],
            buf_k: vec![0.0; num_kv_heads * head_dim],
            buf_v: vec![0.0; num_kv_heads * head_dim],
            buf_attn_out: vec![0.0; num_heads * head_dim],
            buf_attn_proj: vec![0.0; hidden_size],
            buf_gate: vec![0.0; intermediate_size],
            buf_up: vec![0.0; intermediate_size],
            buf_ffn_out: vec![0.0; hidden_size],
            buf_logits: vec![0.0; vocab_size],
            buf_attn_scores: vec![0.0; max_ctx],
        }
    }

    /// Get the kernel for a QuantLinear's tensor type.
    fn kernel_for(&self, linear: &QuantLinear) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
        self.dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map_err(ArchError::from)
    }

    /// Embed a single token into the hidden state buffer.
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

    /// Run grouped-query attention for a single layer, reading the
    /// already-normed input from `self.buf_norm` and writing the (unadded)
    /// output projection to `self.buf_attn_proj`.
    ///
    /// Unlike LLaMA's sequential block, Command-R's attention does **not**
    /// add itself into the residual — the caller combines
    /// `hidden + attn_proj + ffn_out` after both branches have run from the
    /// same normed input.
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

        // Optional per-head Q/K normalization (Command-R+ only).
        if let Some(ref q_norm) = self.layers[layer_idx].attn_q_norm {
            for h in 0..num_heads {
                let q_head = &mut self.buf_q[h * head_dim..(h + 1) * head_dim];
                layer_norm_head_slice(&q_norm.weight, h, head_dim, q_head, q_norm.eps);
            }
        }
        if let Some(ref k_norm) = self.layers[layer_idx].attn_k_norm {
            for h in 0..num_kv_heads {
                let k_head = &mut self.buf_k[h * head_dim..(h + 1) * head_dim];
                layer_norm_head_slice(&k_norm.weight, h, head_dim, k_head, k_norm.eps);
            }
        }

        // Apply RoPE to Q and K (per-head), NORM/interleaved convention.
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

        let scale = 1.0 / (head_dim as f32).sqrt();

        self.buf_attn_out.fill(0.0);

        // Per-head attention
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * head_dim..(h + 1) * head_dim];

            for pos in 0..seq_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = &cached_keys[k_offset..k_offset + head_dim];
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
                let v_vec = &cached_values[v_offset..v_offset + head_dim];
                let w = self.buf_attn_scores[pos];
                for d in 0..head_dim {
                    out_head[d] += w * v_vec[d];
                }
            }
        }

        // Project attention output back to hidden_size. Written to
        // `buf_attn_proj`, NOT added to the residual here (see doc comment).
        let o_kernel = self.kernel_for(&self.layers[layer_idx].attn_output)?;
        let layer = &self.layers[layer_idx];
        layer
            .attn_output
            .forward(&*o_kernel, &self.buf_attn_out, &mut self.buf_attn_proj)?;

        Ok(())
    }

    /// Run the SwiGLU feed-forward network, reading the already-normed input
    /// from `self.buf_norm` (the SAME buffer `attention()` read) and writing
    /// the result to `self.buf_ffn_out`. Does not touch the residual.
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

        Ok(())
    }

    /// Run every token of `tokens` through all layers, leaving the final
    /// token's pre-output-norm hidden state in `self.buf_hidden`.
    fn run_layers(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<()> {
        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, tokens.len())?;
        validate_token_ids(&self.config, tokens)?;

        for (i, &token) in tokens.iter().enumerate() {
            let position = start_pos + i;
            self.embed_token(token)?;

            for layer_idx in 0..self.layers.len() {
                // ONE norm feeds both attention and FFN (Command-R's
                // parallel block — see the module doc comment).
                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&self.buf_hidden, &mut self.buf_norm);

                self.attention(layer_idx, position, kv_cache)?;
                self.feed_forward(layer_idx)?;

                // hidden = hidden + attn_proj + ffn_out (three-way add,
                // matching `cur = ggml_add(cur, inpL); cur = ggml_add(cur,
                // attn_out);` in llama.cpp where `cur` is the FFN output and
                // `inpL` is the pre-norm residual).
                for i in 0..self.buf_hidden.len() {
                    self.buf_hidden[i] += self.buf_attn_proj[i] + self.buf_ffn_out[i];
                }
            }

            kv_cache.advance();
        }

        Ok(())
    }
}

impl ForwardPass for CommandRModel {
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

        // Apply logit scaling if configured
        if (self.logit_scale - 1.0).abs() > f32::EPSILON {
            for v in self.buf_logits.iter_mut() {
                *v *= self.logit_scale;
            }
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

    /// Attach LoRA adapters to this model's linear layers.
    ///
    /// Delegates to [`Self::apply_lora_scaled`] with `scale = 1.0`.
    fn apply_lora(&mut self, lora: &LoadedLora) -> ArchResult<()> {
        self.apply_lora_scaled(lora, 1.0)
    }

    /// Attach LoRA adapters with an extra scale multiplier.
    ///
    /// Uses the same `blk.{i}.*` naming convention as LLaMA, and
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

/// Apply LayerNorm (no bias) to one head's slice of `x`, using the weight row
/// belonging to head `head_idx` within a flat `[head_dim * num_heads]` buffer.
///
/// Command-R+ (104B, `n_layer >= 64`) ships `attn_q_norm`/`attn_k_norm` as ONE
/// tensor of GGUF shape `{n_embd_head_k, n_head}` in `llama-model.cpp` — a
/// distinct weight row per head. Reusing one `head_dim`-length `LayerNorm`
/// object for every head (as if the tensor held a single shared row) would
/// silently apply head 0's weights to every other head.
fn layer_norm_head_slice(
    weight: &[f32],
    head_idx: usize,
    head_dim: usize,
    x: &mut [f32],
    eps: f32,
) {
    let start = head_idx * head_dim;
    let Some(w) = weight.get(start..start + head_dim) else {
        return;
    };
    let n = x.len().min(head_dim);
    if n == 0 {
        return;
    }
    let mean: f32 = x[..n].iter().sum::<f32>() / n as f32;
    let var: f32 = x[..n]
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for (xi, &wi) in x[..n].iter_mut().zip(w.iter()) {
        *xi = (*xi - mean) * inv_std * wi;
    }
}

/// In-place numerically-stable softmax.
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

/// Load a Command-R model from a `GgufModel`.
///
/// Differs from `load_llama_from_gguf` in the ways the module doc comment
/// describes: one LayerNorm per layer (no `ffn_norm`), a tied LM head, the
/// NORM RoPE convention, and optional per-head Q/K norms for Command-R+.
pub fn load_command_r_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<CommandRModel> {
    let dispatcher = KernelDispatcher::new();

    // Load token embeddings
    let embd_data = model.tensor_data("token_embd.weight")?;
    let embd_info = model.file.tensors.get("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        // Cohere's `LLM_NORM` has no bias tensor (`build_norm(..., NULL,
        // LLM_NORM, il)`); `LLM_ARCH_COMMAND_R`'s tensor table never lists
        // `ffn_norm` at all.
        let attn_norm_w = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let attn_norm = LayerNorm::new(attn_norm_w, None, config.rms_norm_eps);

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        // Optional per-head Q/K norm (Command-R+ only).
        let attn_q_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_q_norm.weight"))
            .ok()
            .map(|w| LayerNorm::new(w, None, config.rms_norm_eps));
        let attn_k_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_k_norm.weight"))
            .ok()
            .map(|w| LayerNorm::new(w, None, config.rms_norm_eps));

        layers.push(CommandRLayer {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = LayerNorm::new(output_norm_weight, None, config.rms_norm_eps);

    // Command-R never ships a standalone `output.weight`: llama.cpp always
    // duplicates `token_embd.weight` into `output` for this architecture
    // (`create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), ...,
    // TENSOR_DUPLICATED)` — there is no `TENSOR_NOT_REQUIRED` branch that
    // reads a distinct tensor at all). `load_lm_head` still checks for an
    // explicit `output.weight` first so a nonstandard checkpoint that DOES
    // ship one is still honoured.
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    let logit_scale = config.logit_scale;

    Ok(CommandRModel::new(
        config.clone(),
        token_embd,
        layers,
        output_norm,
        output,
        logit_scale,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CR3: Command-R checkpoints are `LLAMA_ROPE_TYPE_NORM` (interleaved
    /// `(x[2i], x[2i+1])` pairs) — the same convention llama.cpp uses for
    /// LLaMA/Mistral — NOT the NeoX half-split every other RoPE-using
    /// architecture in this crate defaults to
    /// ([`RopeStyle::Neox`](crate::common::rope::RopeStyle)). This rotates a
    /// one-hot head vector both ways and confirms only NORM
    /// (`apply_rope_norm`, what `attention()` actually calls) leaves the
    /// interleaved partner non-zero.
    #[test]
    fn cr3_rope_uses_norm_not_neox_convention() {
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

    #[test]
    fn test_logit_scale_applied() {
        // Verify that logit scaling multiplies all logits correctly.
        let logit_scale = 2.5_f32;
        let mut logits = [1.0f32, 2.0, -1.0, 0.0];

        if (logit_scale - 1.0).abs() > f32::EPSILON {
            for v in logits.iter_mut() {
                *v *= logit_scale;
            }
        }

        assert!((logits[0] - 2.5).abs() < 1e-6, "logits[0]={}", logits[0]);
        assert!((logits[1] - 5.0).abs() < 1e-6, "logits[1]={}", logits[1]);
        assert!((logits[2] - (-2.5)).abs() < 1e-6, "logits[2]={}", logits[2]);
        assert!(logits[3].abs() < 1e-6, "logits[3]={}", logits[3]);
    }

    #[test]
    fn test_logit_scale_one_is_noop() {
        // logit_scale = 1.0 should not change any values.
        let logit_scale = 1.0_f32;
        let mut logits = [1.0f32, 2.0, -1.0];
        let original = logits;

        if (logit_scale - 1.0).abs() > f32::EPSILON {
            for v in logits.iter_mut() {
                *v *= logit_scale;
            }
        }

        assert_eq!(logits, original, "scale=1.0 should not modify logits");
    }

    /// The per-head Q/K norm must use head `h`'s own weight row, not head 0's
    /// row broadcast to every head (see `layer_norm_head_slice`'s doc
    /// comment).
    #[test]
    fn per_head_norm_uses_its_own_weight_row_not_head_zeros() {
        let head_dim = 4;
        // Two heads, distinguishable weight rows: head 0 all-1.0, head 1 all-2.0.
        let weight = vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let eps = 1e-5;

        let mut head0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut head1 = head0.clone();

        layer_norm_head_slice(&weight, 0, head_dim, &mut head0, eps);
        layer_norm_head_slice(&weight, 1, head_dim, &mut head1, eps);

        // Same input normalized against different weight rows must diverge
        // by exactly the weight ratio (2x), since mean/var only depend on x.
        for i in 0..head_dim {
            assert!(
                (head1[i] - 2.0 * head0[i]).abs() < 1e-4,
                "head1[{i}]={} should be 2x head0[{i}]={} (weight row 1 is 2x row 0)",
                head1[i],
                head0[i]
            );
        }
    }

    /// An out-of-range head index (more heads than the flat weight covers)
    /// must not panic — it leaves `x` untouched.
    #[test]
    fn per_head_norm_out_of_range_head_does_not_panic() {
        let weight = vec![1.0f32, 1.0, 1.0, 1.0]; // only 1 head's worth
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let before = x.clone();
        layer_norm_head_slice(&weight, 5, 4, &mut x, 1e-5);
        assert_eq!(x, before, "out-of-range head must leave x untouched");
    }

    // ─── CR1 topology discriminator ────────────────────────────────────────
    //
    // The tests in `tests/command_r_regressions.rs` (an external crate) can
    // only observe finiteness/shape, because `buf_norm` is a private field —
    // a reverted (sequential, LLaMA-style) block would still produce finite
    // output of the right shape. This test lives in-crate specifically to
    // read `buf_norm` directly and prove the PARALLEL topology: attention and
    // FFN must both read `attn_norm(x)` where `x` is the pre-block residual,
    // never `attn_norm(x + attn_proj)`. A single-layer model's `buf_norm`
    // after `embed()` returns is exactly the last (only) layer's normed
    // input, since `embed()` applies `output_norm` in place to `buf_hidden`
    // and never touches `buf_norm` again — so this is a direct window into
    // what the FFN branch actually saw.

    /// Minimal in-memory single-layer KV cache for the topology test.
    struct TopologyTestKv {
        kv_dim: usize,
        position: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
    }

    impl TopologyTestKv {
        fn new(kv_dim: usize, max_seq: usize) -> Self {
            Self {
                kv_dim,
                position: 0,
                keys: vec![0.0f32; max_seq * kv_dim],
                values: vec![0.0f32; max_seq * kv_dim],
            }
        }
    }

    impl KvCacheAccess for TopologyTestKv {
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

    /// `n` deterministic non-zero values, so attention actually moves the
    /// output away from zero (a degenerate all-zero `attn_proj` would let a
    /// buggy sequential implementation coincidentally pass too).
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect()
    }

    fn build_topology_test_model() -> CommandRModel {
        const H: usize = 8;
        const HEADS: usize = 2; // head_dim = H / HEADS = 4
        const FFN: usize = 8;
        const VOCAB: usize = 3;

        let mut w = oxillama_gguf::GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            oxillama_gguf::MetadataValue::String("command-r".to_string()),
        );
        w.add_metadata(
            "command-r.embedding_length",
            oxillama_gguf::MetadataValue::Uint32(H as u32),
        );
        w.add_metadata(
            "command-r.feed_forward_length",
            oxillama_gguf::MetadataValue::Uint32(FFN as u32),
        );
        w.add_metadata(
            "command-r.block_count",
            oxillama_gguf::MetadataValue::Uint32(1),
        );
        w.add_metadata(
            "command-r.attention.head_count",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "command-r.attention.head_count_kv",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "command-r.context_length",
            oxillama_gguf::MetadataValue::Uint32(64),
        );
        w.add_metadata(
            "command-r.vocab_size",
            oxillama_gguf::MetadataValue::Uint32(VOCAB as u32),
        );
        w.add_metadata(
            "command-r.rope.freq_base",
            oxillama_gguf::MetadataValue::Float32(10000.0),
        );
        w.add_metadata(
            "command-r.logit_scale",
            oxillama_gguf::MetadataValue::Float32(1.0),
        );

        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let ones = |n: usize| f32_bytes(&vec![1.0f32; n]);
        let zeros = |n: usize| f32_bytes(&vec![0.0f32; n]);

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
            &f32_bytes(&ramp(H * H)),
        );
        w.add_tensor(
            "blk.0.attn_k.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(H * H)),
        );
        w.add_tensor(
            "blk.0.attn_v.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(H * H)),
        );
        w.add_tensor(
            "blk.0.attn_output.weight",
            &[H as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(H * H)),
        );
        w.add_tensor(
            "blk.0.ffn_gate.weight",
            &[H as u64, FFN as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(H * FFN)),
        );
        w.add_tensor(
            "blk.0.ffn_up.weight",
            &[H as u64, FFN as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(H * FFN)),
        );
        w.add_tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(FFN * H)),
        );
        w.add_tensor(
            "output_norm.weight",
            &[H as u64],
            oxillama_gguf::GgufTensorType::F32,
            &zeros(H),
        );
        w.add_tensor(
            "output.weight",
            &[H as u64, VOCAB as u64],
            oxillama_gguf::GgufTensorType::F32,
            &f32_bytes(&ramp(VOCAB * H)),
        );

        let mut bytes = Vec::new();
        w.write_to(&mut bytes)
            .expect("synthetic command-r GGUF must serialize");
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("must parse");
        let config =
            ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
        load_command_r_from_gguf(&gguf, &config).expect("load")
    }

    /// CR1 discriminator: after running a single-layer model one step, the
    /// normed buffer the FFN branch actually consumed (`buf_norm`) must equal
    /// `attn_norm(raw_embedding)` — the pre-attention residual — and NOT
    /// `attn_norm(raw_embedding + attn_proj)`, which is what a reverted
    /// sequential (LLaMA-style) block would leave behind. `attn_proj` is
    /// asserted non-zero first so this can't pass by degenerate coincidence.
    #[test]
    fn cr1_ffn_and_attention_read_the_same_pre_attention_norm() {
        let mut model = build_topology_test_model();
        let token = 1u32;

        // Independently compute what the CORRECT (parallel) topology must
        // leave in `buf_norm`: attn_norm applied to the untouched embedding
        // row, using the model's own loaded attn_norm weights.
        let hidden_size = model.config.hidden_size;
        let offset = token as usize * hidden_size;
        let raw_embedding = model.token_embd[offset..offset + hidden_size].to_vec();
        let mut expected_norm = vec![0.0f32; hidden_size];
        model.layers[0]
            .attn_norm
            .forward_to(&raw_embedding, &mut expected_norm);

        let mut kv = TopologyTestKv::new(4 * 2, 8);
        model.embed(&[token], &mut kv).expect("embed");

        assert!(
            model.buf_attn_proj.iter().any(|&v| v.abs() > 1e-6),
            "attn_proj must be non-zero for this test to discriminate topology, got {:?}",
            model.buf_attn_proj
        );

        for (i, (&got, &want)) in model.buf_norm.iter().zip(expected_norm.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "buf_norm[{i}]={got} must equal attn_norm(raw_embedding)[{i}]={want} — if \
                 this fails, the FFN branch is reading attn_norm(x + attn_proj) instead of \
                 attn_norm(x), i.e. the block has regressed to a sequential (LLaMA-style) \
                 topology instead of Command-R's parallel one"
            );
        }
    }

    // ─── CR3 call-site discriminator ───────────────────────────────────────
    //
    // `cr3_rope_uses_norm_not_neox_convention` (above) only proves
    // `apply_rope_norm` itself rotates interleaved pairs — it never calls
    // `attention()`, so it cannot catch a wrong per-head slice offset or a
    // wrong `position` argument at the real call site. This test drives the
    // actual model (`attn_q.weight` projects the normed input to a
    // one-hot-per-head vector) and inspects `buf_q` after a real
    // `attention()` call at a NONZERO position, closing that gap — mirrors
    // `mistral::model::tests::mi5_attention_rotates_every_head_with_norm_convention_at_the_real_call_site`.

    /// Build a 2-head, `head_dim = 4` Command-R model whose `attn_q.weight`
    /// is zero everywhere except output row 0 (head 0, index 0) and output
    /// row 4 (head 1, index 0), each picking input index 0 with weight 1.0.
    /// No `attn_q_norm`/`attn_k_norm` tensors, so the per-head Q/K norm stays
    /// `None` and doesn't disturb the one-hot vector before RoPE runs.
    fn build_rope_call_site_test_model() -> CommandRModel {
        const H: usize = 8;
        const HEADS: usize = 2; // head_dim = H / HEADS = 4
        const FFN: usize = 8;
        const VOCAB: usize = 3;

        let mut w = oxillama_gguf::GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            oxillama_gguf::MetadataValue::String("command-r".to_string()),
        );
        w.add_metadata(
            "command-r.embedding_length",
            oxillama_gguf::MetadataValue::Uint32(H as u32),
        );
        w.add_metadata(
            "command-r.feed_forward_length",
            oxillama_gguf::MetadataValue::Uint32(FFN as u32),
        );
        w.add_metadata(
            "command-r.block_count",
            oxillama_gguf::MetadataValue::Uint32(1),
        );
        w.add_metadata(
            "command-r.attention.head_count",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "command-r.attention.head_count_kv",
            oxillama_gguf::MetadataValue::Uint32(HEADS as u32),
        );
        w.add_metadata(
            "command-r.context_length",
            oxillama_gguf::MetadataValue::Uint32(64),
        );
        w.add_metadata(
            "command-r.vocab_size",
            oxillama_gguf::MetadataValue::Uint32(VOCAB as u32),
        );
        w.add_metadata(
            "command-r.rope.freq_base",
            oxillama_gguf::MetadataValue::Float32(10000.0),
        );
        w.add_metadata(
            "command-r.logit_scale",
            oxillama_gguf::MetadataValue::Float32(1.0),
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
            .expect("synthetic command-r GGUF must serialize");
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("must parse");
        let config =
            ModelConfig::from_metadata(&gguf.file.metadata).expect("fixture metadata must parse");
        load_command_r_from_gguf(&gguf, &config).expect("load")
    }

    /// CR3 (call site): at a non-zero position, `attention()` must rotate
    /// EACH head's index-0 value into that SAME head's index-1 (the
    /// interleaved NORM partner), never into index `head_dim/2` (the NeoX
    /// partner), and must do so identically for every head (proving the
    /// per-head slice offset `h * head_dim..(h + 1) * head_dim` — not a
    /// fixed or off-by-one slice — is what gets passed to `apply_rope_norm`).
    #[test]
    fn cr3_attention_rotates_every_head_with_norm_convention_at_the_real_call_site() {
        let mut model = build_rope_call_site_test_model();
        let mut kv = TopologyTestKv::new(4 * 2, 8);

        // Position 0: RoPE angle is 0 for every frequency, so this call must
        // leave buf_q un-rotated — confirms the one-hot-per-head setup itself
        // (not yet informative about NORM vs NeoX).
        model.embed(&[0u32], &mut kv).expect("embed @ pos 0");
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
        model.embed(&[1u32], &mut kv).expect("embed @ pos 1");
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
