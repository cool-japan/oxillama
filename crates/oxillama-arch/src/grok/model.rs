//! Grok-1 transformer block and forward pass.
//!
//! ```text
//! embedding × embedding_scale
//!   → N×( RMSNorm(attn_norm) → GQA+RoPE(NeoX) with a tanh-softcapped score
//!         → RMSNorm(attn_out_norm) → +residual
//!         → RMSNorm(ffn_norm) → MoE(GELU) [+ optional parallel dense FFN × √2/2]
//!         → RMSNorm(ffn_post_norm) → +residual )
//!   → RMSNorm(output_norm) → LM head × logit_scale [→ tanh soft-cap]
//! ```
//!
//! Every element above was verified against `llm_build_grok`
//! (`src/models/grok.cpp`), the `LLM_ARCH_GROK` hparams block and tensor-creation
//! block in `src/llama-model.cpp`, the `LLM_ARCH_GROK` tensor table in
//! `src/llama-arch.cpp`, and `llm_graph_context::build_attn_mha` in
//! `src/llama-graph.cpp`.
//!
//! # What was missing
//!
//! * **`embedding_scale = 78.38367176906169`** (`= sqrt(6144)`), applied to the
//!   token embedding.  Absent, every hidden state entered layer 0 ~78× too small.
//! * **`logit_scale = 0.5773502691896257`** (`= 1/sqrt(3)`) on the final logits.
//! * **`attn_out_norm` and `ffn_post_norm`.**  `GrokLayer` carried only
//!   `attn_norm`/`ffn_norm`; `build_grok` normalises the attention output
//!   *before* the residual add and the FFN output *before* its residual add.
//!   Both tensors are required by llama.cpp's loader
//!   (`ffn_post_norm` comes from `LLM_TENSOR_LAYER_OUT_NORM` =
//!   `blk.%d.layer_output_norm`, falling back to `LLM_TENSOR_FFN_POST_NORM` =
//!   `blk.%d.post_ffw_norm`).
//! * **GELU experts.**  `build_grok` passes `LLM_FFN_GELU`; the experts ran the
//!   hard-coded SiLU of `DeepSeekExpert::forward`.  See [`crate::grok::moe`].
//! * **The attention soft-cap.**  `build_attn_mha` does, for `LLM_ARCH_GROK`
//!   only, `kq = cap · tanh(kq · attn_out_scale / cap)` with `cap = 30`, and
//!   then `soft_max_ext(kq, mask, kq_scale = 1.0f)`.  The `1/sqrt(128)` factor
//!   in the old code *is* `f_attn_out_scale` and is kept — but it is a
//!   **constant** in llama.cpp, not `1/sqrt(head_dim)`, and the `tanh` around it
//!   was missing entirely.
//! * **`kv_cache.advance()` once per token** rather than once per layer.
//!
//! # Deliberately not implemented
//!
//! `grok.router_logit_softcapping` is parsed into
//! [`GrokConfig::router_logit_softcapping`] but **not applied**: in the
//! reference, `f_router_logit_softcapping` appears only in the hparams loader
//! and `llama-model-saver.cpp` — `build_moe_ffn` never reads it.  Applying it
//! would diverge from llama.cpp.

use oxillama_quant::{KernelDispatcher, QuantKernel};

use crate::common::attention::validate_context_bounds;
use crate::common::gelu::gelu;
use crate::common::linear::QuantLinear;
use crate::common::moe::MoeScratch;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::{RopeStyle, RopeTable};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::grok::config::GrokConfig;
use crate::grok::moe::GrokMoe;
use crate::traits::{ForwardPass, KvCacheAccess};

/// Grok's soft-capped pre-softmax attention score.
///
/// `llm_graph_context::build_attn_mha`, `LLM_ARCH_GROK` branch:
///
/// ```text
/// kq = ggml_tanh(ggml_scale(kq, f_attn_out_scale / f_attn_logit_softcapping));
/// kq = ggml_scale(kq, f_attn_logit_softcapping);
/// ...
/// kq = ggml_soft_max_ext(kq, kq_mask, /* kq_scale = */ 1.0f, ...);
/// ```
///
/// i.e. `cap · tanh(raw · scale / cap)`, and the softmax then applies a scale of
/// exactly 1.  `scale` is a **constant** `1/sqrt(128)` in the reference, not
/// `1/sqrt(head_dim)`.  A non-positive `cap` disables the tanh and leaves the
/// bare `raw · scale`.
#[inline]
fn softcap_score(raw: f32, scale: f32, cap: f32) -> f32 {
    if cap > 0.0 {
        cap * (raw * scale / cap).tanh()
    } else {
        raw * scale
    }
}

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// The optional dense FFN Grok runs **in parallel** with the MoE branch.
///
/// `build_grok`:
///
/// ```text
/// if (model.layers[il].ffn_up) {
///     ffn_out = build_ffn(cur, ffn_up, ffn_gate, ffn_down, LLM_FFN_GELU, LLM_FFN_PAR);
///     cur = ggml_scale(ggml_add(ffn_out, moe_out), sqrt(2) / 2);
/// } else {
///     cur = moe_out;
/// }
/// ```
pub struct GrokDenseFfn {
    /// Gate projection `[intermediate, hidden]`.
    pub gate: QuantLinear,
    /// Up projection `[intermediate, hidden]`.
    pub up: QuantLinear,
    /// Down projection `[hidden, intermediate]`.
    pub down: QuantLinear,
    /// Kernel for [`Self::gate`].
    pub gate_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::up`].
    pub up_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::down`].
    pub down_kernel: Box<dyn QuantKernel>,
}

/// One Grok-1 transformer layer.
pub struct GrokLayer {
    /// Pre-attention RMSNorm (`blk.N.attn_norm.weight`).
    pub attn_norm: RmsNorm,
    /// Query projection `[num_heads * head_dim, hidden_size]`.
    pub attn_q: QuantLinear,
    /// Key projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_k: QuantLinear,
    /// Value projection `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_v: QuantLinear,
    /// Output projection `[hidden_size, num_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// Post-attention RMSNorm applied **before** the residual add
    /// (`blk.N.attn_output_norm.weight`).
    pub attn_out_norm: RmsNorm,
    /// Pre-FFN RMSNorm (`blk.N.ffn_norm.weight`).
    pub ffn_norm: RmsNorm,
    /// Post-FFN RMSNorm applied **before** the residual add
    /// (`blk.N.layer_output_norm.weight`, or `blk.N.post_ffw_norm.weight`).
    pub ffn_post_norm: RmsNorm,
    /// Sparse MoE FFN with GELU experts.
    pub moe: GrokMoe,
    /// Optional dense FFN run in parallel with the MoE branch.
    pub dense_ffn: Option<GrokDenseFfn>,
    /// Kernel for [`Self::attn_q`].
    pub attn_q_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::attn_k`].
    pub attn_k_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::attn_v`].
    pub attn_v_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`].
    pub attn_output_kernel: Box<dyn QuantKernel>,
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete Grok-1 model.
pub struct GrokModel {
    /// Common model config.
    pub config: ModelConfig,
    /// Grok-specific config (MoE layout, rope_theta, and the four scalars).
    pub grok_config: GrokConfig,
    /// Token embedding table `[vocab_size, hidden_size]`.
    pub token_embd: Vec<f32>,
    /// Number of embedding rows actually present in `token_embd`.
    embd_rows: usize,
    /// Transformer layers.
    pub layers: Vec<GrokLayer>,
    /// Final RMSNorm before the LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Precomputed RoPE frequency table (NeoX pairing).
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_proj: Vec<f32>,
    buf_attn_scores: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_moe_out: Vec<f32>,
    buf_dense: Vec<f32>,
    buf_dense_act: Vec<f32>,
    moe_scratch: MoeScratch,
}

impl GrokModel {
    /// Create a new `GrokModel` from pre-loaded weights.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `token_embd` holds fewer than
    /// `vocab_size × hidden_size` values.
    pub fn new(
        config: ModelConfig,
        grok_config: GrokConfig,
        token_embd: Vec<f32>,
        layers: Vec<GrokLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        // GROK is NEOX in llama.cpp's `llama_model_rope_type` table.
        let rope = RopeTable::new_standard_with_style(
            config.head_dim,
            config.max_context_length,
            config.rope_freq_base,
            RopeStyle::Neox,
        );
        let hidden = config.hidden_size;
        if hidden == 0 || config.vocab_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "Grok requires hidden_size > 0 and vocab_size > 0".to_string(),
            });
        }
        let embd_rows = token_embd.len() / hidden;
        if embd_rows < config.vocab_size {
            return Err(ArchError::InvalidShape {
                name: "token_embd.weight".to_string(),
                expected: vec![config.vocab_size, hidden],
                got: vec![embd_rows, hidden],
            });
        }

        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = grok_config.ffn_hidden_size;
        let moe_scratch = layers
            .first()
            .map(|l| l.moe.make_scratch())
            .unwrap_or_default();

        Ok(Self {
            dispatcher: KernelDispatcher::new(),
            rope,
            buf_q: vec![0.0f32; q_dim],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_out: vec![0.0f32; q_dim],
            buf_proj: vec![0.0f32; hidden],
            buf_attn_scores: vec![0.0f32; config.max_context_length],
            buf_norm: vec![0.0f32; hidden],
            buf_moe_out: vec![0.0f32; hidden],
            buf_dense: vec![0.0f32; hidden],
            buf_dense_act: vec![0.0f32; inter],
            moe_scratch,
            embd_rows,
            config,
            grok_config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// Attention for one token at `pos`, leaving the projection in `buf_proj`.
    ///
    /// **Does not** call [`KvCacheAccess::advance`].
    fn attention(
        &mut self,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_kv = num_heads.checked_div(num_kv).unwrap_or(1);
        let kv_dim = num_kv * hd;
        let q_dim = num_heads * hd;

        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_q
                .forward(&*layer.attn_q_kernel, &self.buf_norm, &mut self.buf_q)?;
            layer
                .attn_k
                .forward(&*layer.attn_k_kernel, &self.buf_norm, &mut self.buf_k)?;
            layer
                .attn_v
                .forward(&*layer.attn_v_kernel, &self.buf_norm, &mut self.buf_v)?;
        }

        for h in 0..num_heads {
            self.rope.apply(&mut self.buf_q[h * hd..(h + 1) * hd], pos);
        }
        for h in 0..num_kv {
            self.rope.apply(&mut self.buf_k[h * hd..(h + 1) * hd], pos);
        }

        kv_cache.store_kv(layer_idx, &self.buf_k[..kv_dim], &self.buf_v[..kv_dim])?;

        let cached_keys = crate::common::fetch_keys(&*kv_cache, layer_idx)?;
        let cached_values = crate::common::fetch_values(&*kv_cache, layer_idx)?;
        let cached_keys: &[f32] = &cached_keys;
        let cached_values: &[f32] = &cached_values;
        let n_tokens = pos + 1;
        let needed = n_tokens * kv_dim;
        if cached_keys.len() < needed || cached_values.len() < needed {
            return Err(ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!(
                    "kv cache exposes {} key / {} value floats, attention at position {pos} \
                     needs {needed}",
                    cached_keys.len(),
                    cached_values.len()
                ),
            });
        }

        self.buf_attn_out[..q_dim].fill(0.0);
        let scale = self.grok_config.attn_output_scale;
        let cap = self.grok_config.attn_logit_softcapping;

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &self.buf_q[h * hd..(h + 1) * hd];
            let scores = &mut self.buf_attn_scores[..n_tokens];

            for (t, score) in scores.iter_mut().enumerate() {
                let off = t * kv_dim + kv_head * hd;
                let raw: f32 = q_head
                    .iter()
                    .zip(cached_keys[off..off + hd].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                *score = softcap_score(raw, scale, cap);
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            if exp_sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= exp_sum;
                }
            }

            let out_head = &mut self.buf_attn_out[h * hd..(h + 1) * hd];
            for (t, &w) in scores.iter().enumerate() {
                let off = t * kv_dim + kv_head * hd;
                for (o, &val) in out_head.iter_mut().zip(cached_values[off..off + hd].iter()) {
                    *o += w * val;
                }
            }
        }

        let layer = &self.layers[layer_idx];
        layer.attn_output.forward(
            &*layer.attn_output_kernel,
            &self.buf_attn_out[..q_dim],
            &mut self.buf_proj,
        )?;
        Ok(())
    }

    /// The optional parallel dense FFN, GELU-gated like the experts.
    fn dense_ffn(&mut self, layer_idx: usize) -> ArchResult<bool> {
        let Some(ffn) = self.layers[layer_idx].dense_ffn.as_ref() else {
            return Ok(false);
        };
        let inter = ffn.gate.out_features;
        if self.buf_dense_act.len() < inter {
            self.buf_dense_act.resize(inter, 0.0);
        }
        let mut gate_act = std::mem::take(&mut self.buf_dense_act);
        let run = (|| -> ArchResult<()> {
            let ffn = self.layers[layer_idx].dense_ffn.as_ref().ok_or_else(|| {
                ArchError::ForwardPassError {
                    layer: layer_idx,
                    message: "dense FFN vanished between checks".to_string(),
                }
            })?;
            ffn.gate
                .forward(&*ffn.gate_kernel, &self.buf_norm, &mut gate_act[..inter])?;
            for g in gate_act[..inter].iter_mut() {
                *g = gelu(*g);
            }
            let mut up = vec![0.0f32; inter];
            ffn.up.forward(&*ffn.up_kernel, &self.buf_norm, &mut up)?;
            for (g, u) in gate_act[..inter].iter_mut().zip(up.iter()) {
                *g *= u;
            }
            ffn.down
                .forward(&*ffn.down_kernel, &gate_act[..inter], &mut self.buf_dense)?;
            Ok(())
        })();
        self.buf_dense_act = gate_act;
        run?;
        Ok(true)
    }

    /// Run every token through all layers.
    fn run_layers(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward: empty token sequence".to_string(),
            });
        }
        let start_pos = kv_cache.seq_len();
        validate_context_bounds(&self.config, start_pos, seq_len)?;

        // `build_inp_embd`: `cur = ggml_scale(cur, hparams.f_embedding_scale)`.
        let embedding_scale = self.grok_config.embedding_scale;
        let mut hidden_states = vec![0.0f32; seq_len * hidden];
        for (t, &tok_id) in tokens.iter().enumerate() {
            let tok = tok_id as usize;
            if tok >= self.embd_rows.min(self.config.vocab_size) {
                return Err(ArchError::ConfigMismatch {
                    param: "token_id".to_string(),
                    expected: format!(
                        "< vocab_size ({}) and < token_embd rows ({})",
                        self.config.vocab_size, self.embd_rows
                    ),
                    got: tok.to_string(),
                });
            }
            let off = tok * hidden;
            for (dst, &src) in hidden_states[t * hidden..(t + 1) * hidden]
                .iter_mut()
                .zip(self.token_embd[off..off + hidden].iter())
            {
                *dst = src * embedding_scale;
            }
        }

        let sqrt2_over_2 = std::f32::consts::SQRT_2 / 2.0;
        let n_layers = self.layers.len();
        for t in 0..seq_len {
            let pos = start_pos + t;
            for layer_idx in 0..n_layers {
                let row = t * hidden..(t + 1) * hidden;

                self.layers[layer_idx]
                    .attn_norm
                    .forward_to(&hidden_states[row.clone()], &mut self.buf_norm);

                self.attention(layer_idx, pos, kv_cache)?;

                // `cur = build_norm(cur, attn_out_norm)` BEFORE the residual.
                self.layers[layer_idx]
                    .attn_out_norm
                    .forward(&mut self.buf_proj);
                for (h, p) in hidden_states[row.clone()]
                    .iter_mut()
                    .zip(self.buf_proj.iter())
                {
                    *h += p;
                }

                self.layers[layer_idx]
                    .ffn_norm
                    .forward_to(&hidden_states[row.clone()], &mut self.buf_norm);

                self.layers[layer_idx].moe.forward(
                    &self.buf_norm,
                    &mut self.buf_moe_out,
                    &mut self.moe_scratch,
                )?;

                if self.dense_ffn(layer_idx)? {
                    for (m, d) in self.buf_moe_out.iter_mut().zip(self.buf_dense.iter()) {
                        *m = (*m + *d) * sqrt2_over_2;
                    }
                }

                // `cur = build_norm(cur, ffn_post_norm)` BEFORE the residual.
                self.layers[layer_idx]
                    .ffn_post_norm
                    .forward(&mut self.buf_moe_out);
                for (h, m) in hidden_states[row].iter_mut().zip(self.buf_moe_out.iter()) {
                    *h += m;
                }
            }
            // Exactly once per token, after every layer has written its K/V.
            kv_cache.advance();
        }
        Ok(hidden_states)
    }
}

impl ForwardPass for GrokModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = tokens.len();
        let mut hidden_states = self.run_layers(tokens, kv_cache)?;

        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        let lm_kernel = self
            .dispatcher
            .get_kernel(self.output.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut logits = vec![0.0f32; vocab];
        self.output.forward(&*lm_kernel, last, &mut logits)?;

        // `cur = ggml_scale(cur, hparams.f_logit_scale)`.
        let logit_scale = self.grok_config.logit_scale;
        if logit_scale != 1.0 {
            for l in logits.iter_mut() {
                *l *= logit_scale;
            }
        }
        // Optional final soft-cap (0.0 for grok-1).
        let cap = self.grok_config.final_logit_softcapping;
        if cap > 0.0 {
            for l in logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
        Ok(logits)
    }

    fn embed(&mut self, tokens: &[u32], kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = tokens.len();
        let mut hidden_states = self.run_layers(tokens, kv_cache)?;
        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);
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

    /// Grok keeps no internal per-sequence state: positions come from
    /// [`KvCacheAccess::seq_len`].  The model used to carry a monotonic
    /// `current_pos` that was never reset.
    fn reset_sequence(&mut self) {
        self.buf_attn_scores.fill(0.0);
        self.buf_q.fill(0.0);
        self.buf_k.fill(0.0);
        self.buf_v.fill(0.0);
        self.buf_attn_out.fill(0.0);
        self.buf_proj.fill(0.0);
        self.buf_norm.fill(0.0);
        self.buf_moe_out.fill(0.0);
        self.buf_dense.fill(0.0);
    }
}

/// Construct a `GrokModel` from raw weights.
///
/// # Errors
///
/// See [`GrokModel::new`].
pub fn build_grok_model(
    config: ModelConfig,
    grok_config: GrokConfig,
    token_embd: Vec<f32>,
    layers: Vec<GrokLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> ArchResult<GrokModel> {
    GrokModel::new(config, grok_config, token_embd, layers, output_norm, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grok::testkit::{TestKvCache, TinyWeights};

    const H: usize = 16;
    const VOCAB: usize = 32;
    const N_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const N_LAYERS: usize = 2;
    const N_EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const INTER: usize = 8;

    fn tiny_model() -> GrokModel {
        let mut w = TinyWeights::new(11);
        let grok_config = GrokConfig::from_metadata(&oxillama_gguf::MetadataStore::new());
        let grok_config = GrokConfig {
            hidden_size: H,
            num_layers: N_LAYERS,
            num_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_seq_len: 64,
            expert_count: N_EXPERTS,
            expert_used_count: TOP_K,
            ffn_hidden_size: INTER,
            ..grok_config
        };
        let config = ModelConfig {
            architecture: "grok".to_string(),
            model_name: "test-grok".to_string(),
            hidden_size: H,
            intermediate_size: INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 64,
            rope_freq_base: 1_000_000.0,
            ..ModelConfig::default()
        };
        let kv_dim = N_HEADS * HEAD_DIM;
        let layers = (0..N_LAYERS)
            .map(|_| {
                let attn_q = w.linear(N_HEADS * HEAD_DIM, H);
                let attn_k = w.linear(kv_dim, H);
                let attn_v = w.linear(kv_dim, H);
                let attn_output = w.linear(H, N_HEADS * HEAD_DIM);
                GrokLayer {
                    attn_norm: RmsNorm::new(vec![1.0; H], 1e-5),
                    attn_q_kernel: w.kernel(&attn_q),
                    attn_k_kernel: w.kernel(&attn_k),
                    attn_v_kernel: w.kernel(&attn_v),
                    attn_output_kernel: w.kernel(&attn_output),
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_out_norm: RmsNorm::new(vec![1.0; H], 1e-5),
                    ffn_norm: RmsNorm::new(vec![1.0; H], 1e-5),
                    ffn_post_norm: RmsNorm::new(vec![1.0; H], 1e-5),
                    moe: w.grok_moe(H, INTER, N_EXPERTS, TOP_K),
                    dense_ffn: None,
                }
            })
            .collect();
        GrokModel::new(
            config,
            grok_config,
            w.embedding(VOCAB, H),
            layers,
            RmsNorm::new(vec![1.0; H], 1e-5),
            w.linear(VOCAB, H),
        )
        .expect("tiny Grok model builds")
    }

    fn cache(model: &GrokModel) -> TestKvCache {
        TestKvCache::new(
            model.layers.len(),
            model.config.num_kv_heads * model.config.head_dim,
            model.config.max_context_length,
        )
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        let logits = model.forward(&[1u32, 2], &mut kv).expect("forward");
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embed_returns_hidden_size() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        assert_eq!(model.embed(&[1u32], &mut kv).expect("embed").len(), H);
    }

    /// `advance()` is per-token, not per-layer.
    #[test]
    fn advance_is_called_once_per_token() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
        assert_eq!(kv.seq_len(), 3, "3 tokens must advance the cache 3 times");
        for l in 0..model.layers.len() {
            assert_eq!(kv.writes(l), 3, "layer {l} writes one row per token");
            for pos in 3..6 {
                assert!(
                    kv.key_row(l, pos).iter().all(|v| *v == 0.0),
                    "layer {l} wrote past position 2 (row {pos})"
                );
            }
        }
    }

    /// The embedding multiplier is real: doubling it must change the logits.
    #[test]
    fn embedding_scale_reaches_the_output() {
        let mut a = tiny_model();
        let mut b = tiny_model();
        b.grok_config.embedding_scale *= 2.0;
        let mut kv_a = cache(&a);
        let mut kv_b = cache(&b);
        let la = a.forward(&[1u32], &mut kv_a).expect("a");
        let lb = b.forward(&[1u32], &mut kv_b).expect("b");
        assert!(
            la.iter().zip(lb.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
            "doubling embedding_scale must change the logits"
        );
    }

    /// The logit multiplier scales the output linearly.
    #[test]
    fn logit_scale_multiplies_the_logits() {
        let mut a = tiny_model();
        let mut b = tiny_model();
        b.grok_config.logit_scale = a.grok_config.logit_scale * 2.0;
        let mut kv_a = cache(&a);
        let mut kv_b = cache(&b);
        let la = a.forward(&[1u32], &mut kv_a).expect("a");
        let lb = b.forward(&[1u32], &mut kv_b).expect("b");
        for (i, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
            assert!(
                (2.0 * x - y).abs() < 1e-3,
                "logit {i}: doubling logit_scale must double the logit ({x} → {y})"
            );
        }
    }

    /// The tanh soft-cap bounds every pre-softmax score by `cap`.
    #[test]
    fn attention_score_is_soft_capped() {
        let model = tiny_model();
        let cap = model.grok_config.attn_logit_softcapping;
        let scale = model.grok_config.attn_output_scale;
        assert!(cap > 0.0);
        for raw in [-1e6f32, -1000.0, 0.0, 1000.0, 1e6] {
            let s = softcap_score(raw, scale, cap);
            assert!(
                s.abs() <= cap + 1e-3,
                "score {s} for raw {raw} exceeds the {cap} soft-cap"
            );
        }
        // Small scores pass through as `raw * attn_output_scale`.
        assert!((softcap_score(1.0, scale, cap) - scale).abs() < 1e-4);
        // With the cap disabled the bare multiplier survives.
        assert!((softcap_score(4.0, scale, 0.0) - 4.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn context_overflow_is_reported() {
        let mut model = tiny_model();
        let max = model.config.max_context_length;
        let mut kv = cache(&model);
        let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
        assert!(model.forward(&tokens, &mut kv).is_err());
    }

    #[test]
    fn out_of_vocab_token_is_reported() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        assert!(model.forward(&[999u32], &mut kv).is_err());
    }
}
