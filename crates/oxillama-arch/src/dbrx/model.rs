//! DBRX transformer block and forward pass.
//!
//! DBRX (Databricks) is a fine-grained MoE decoder: 16 routed experts, top-4 per
//! token, 94 % of the parameters living in the expert pool.
//!
//! ```text
//! embedding
//!   → N×( LayerNorm → fused QKV (clamped) → GQA+RoPE(NeoX) → residual
//!         → LayerNorm(attn_output_norm) → MoE → residual )
//!   → LayerNorm → LM head
//! ```
//!
//! # Differences from what this module used to implement
//!
//! Every item below was verified against `llm_build_dbrx` in
//! `src/models/dbrx.cpp`, the `LLM_ARCH_DBRX` tensor table in
//! `src/llama-arch.cpp`, the hparams block in `src/llama-model.cpp`, and
//! `convert_hf_to_gguf.py::DbrxModel`:
//!
//! * **LayerNorm, not RMSNorm.**  `build_dbrx` uses `LLM_NORM` for
//!   `attn_norm`, `attn_out_norm` and `output_norm`, and the hparams block
//!   reads `LLM_KV_ATTENTION_LAYERNORM_EPS` (`dbrx.attention.layer_norm_epsilon`,
//!   written by `add_layer_norm_eps(1e-5)`) — **not**
//!   `…layer_norm_rms_epsilon`, which no DBRX GGUF contains.  HF's
//!   `DbrxNormAttentionNorm` is `nn.LayerNorm(bias=False)`, so the checkpoint
//!   ships a weight and no bias; a weight-only tensor is consistent with either
//!   norm type, which is why the previous RMSNorm choice went unnoticed.
//! * **Fused `blk.%d.attn_qkv.weight`.**  The arch table lists
//!   `LLM_TENSOR_ATTN_QKV` and *no* `ATTN_Q`/`ATTN_K`/`ATTN_V`;
//!   `tensor_mapping.py` maps it from `…norm_attn_norm.attn.Wqkv`.  The layout is
//!   `[n_embd + 2*n_embd_gqa, n_embd]` with Q first, then K, then V.
//! * **`ggml_clamp(±f_clamp_kqv)`** on the fused projection, from
//!   `dbrx.attention.clamp_kqv` (`clip_qkv = 8.0` in the real checkpoint).
//! * **`blk.%d.attn_output_norm.weight` replaces `ffn_norm`.**  DBRX has no
//!   `ffn_norm` at all; `attn_out_norm` sits in the pre-FFN slot, after the
//!   attention residual add.
//! * **Quantized experts.**  The expert pool stays in its GGUF form as shared
//!   mmap views.  Dequantizing DBRX's experts to `f32` at load — what this
//!   module used to do, twice: once for the stacked tensor and once more per
//!   expert via `.to_vec()` — needs more than 500 GB.
//! * **`kv_cache.advance()` exactly once per token.**  It used to be the last
//!   statement of `attention_single_token`, i.e. once per *layer*, so each token
//!   consumed `n_layers` cache slots and layer *l* read rows written by other
//!   layers.
//!
//! Verified and deliberately **not** changed: DBRX's top-k renormalisation
//! (`moe_normalize_expert_weights: 1.0`, llama.cpp passes `norm_w = true`) and
//! the `1/sqrt(head_dim)` attention scale (`build_dbrx` passes
//! `1.0f/sqrtf(float(n_embd_head))`).

use oxillama_quant::{KernelDispatcher, QuantKernel};

use crate::common::attention::validate_context_bounds;
use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::moe::{MoeScratch, QuantMoeFfn};
use crate::common::rope::{RopeStyle, RopeTable};
use crate::config::ModelConfig;
use crate::dbrx::config::DbrxConfig;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// One DBRX transformer layer.
pub struct DbrxLayer {
    /// Pre-attention LayerNorm (`blk.N.attn_norm.weight`).
    pub attn_norm: LayerNorm,
    /// Fused QKV projection `[hidden + 2*kv_dim, hidden]`
    /// (`blk.N.attn_qkv.weight`).
    pub attn_qkv: QuantLinear,
    /// Output projection `[hidden, num_heads * head_dim]`.
    pub attn_output: QuantLinear,
    /// Pre-FFN LayerNorm (`blk.N.attn_output_norm.weight`).
    ///
    /// DBRX has no `ffn_norm`; this tensor occupies that slot.
    pub attn_out_norm: LayerNorm,
    /// Sparse MoE FFN over quantized experts.
    ///
    /// DBRX's routing is exactly [`QuantMoeFfn`]'s: softmax over the router
    /// logits, top-`k`, re-normalise (`moe_normalize_expert_weights: 1.0`, and
    /// `build_dbrx` passes `norm_w = true`), SwiGLU experts.
    pub moe: QuantMoeFfn,
    /// Kernel for [`Self::attn_qkv`], resolved once at load time.
    pub attn_qkv_kernel: Box<dyn QuantKernel>,
    /// Kernel for [`Self::attn_output`], resolved once at load time.
    pub attn_output_kernel: Box<dyn QuantKernel>,
}

// ─── Full model ────────────────────────────────────────────────────────────────

/// Complete DBRX model.
pub struct DbrxModel {
    /// Common model config (embedding dim, num layers, vocab size, etc.).
    pub config: ModelConfig,
    /// DBRX-specific config (MoE layout, `clamp_kqv`, LayerNorm eps).
    pub dbrx_config: DbrxConfig,
    /// Token embedding table `[vocab_size, hidden_size]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Number of embedding rows actually present in `token_embd`.
    ///
    /// A GGUF may declare a `vocab_size` larger than the `token_embd` row count
    /// (padded tokenizers do this).  Bounding the lookup by `config.vocab_size`
    /// alone still indexes past the end of the buffer.
    embd_rows: usize,
    /// Transformer layers.
    pub layers: Vec<DbrxLayer>,
    /// Final LayerNorm before the LM head.
    pub output_norm: LayerNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Precomputed RoPE frequency table (NeoX pairing).
    pub rope: RopeTable,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    // Scratch buffers (reused across calls).
    buf_qkv: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_proj: Vec<f32>,
    buf_attn_scores: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_moe_out: Vec<f32>,
    moe_scratch: MoeScratch,
}

impl DbrxModel {
    /// Create a new `DbrxModel` from pre-loaded weights.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `token_embd` does not hold
    /// `vocab_size × hidden_size` values (checked here so the per-token lookup
    /// never has to).
    pub fn new(
        config: ModelConfig,
        dbrx_config: DbrxConfig,
        token_embd: Vec<f32>,
        layers: Vec<DbrxLayer>,
        output_norm: LayerNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        // DBRX is NEOX in llama.cpp's `llama_model_rope_type` table.
        let rope = RopeTable::new_standard_with_style(
            config.head_dim,
            config.max_context_length,
            config.rope_freq_base,
            RopeStyle::Neox,
        );
        let dispatcher = KernelDispatcher::new();
        let hidden = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let max_ctx = config.max_context_length;

        if hidden == 0 || config.vocab_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "DBRX requires hidden_size > 0 and vocab_size > 0".to_string(),
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

        let moe_scratch = layers
            .first()
            .map(|l| l.moe.make_scratch())
            .unwrap_or_default();

        Ok(Self {
            dispatcher,
            rope,
            buf_qkv: vec![0.0f32; q_dim + 2 * kv_dim],
            buf_attn_out: vec![0.0f32; q_dim],
            buf_proj: vec![0.0f32; hidden],
            buf_attn_scores: vec![0.0f32; max_ctx],
            buf_norm: vec![0.0f32; hidden],
            buf_moe_out: vec![0.0f32; hidden],
            moe_scratch,
            embd_rows,
            config,
            dbrx_config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// Run grouped-query attention for a single token at position `pos`,
    /// accumulating the result into `hidden_state`.
    ///
    /// **Does not** call [`KvCacheAccess::advance`] — that happens once per
    /// token in [`Self::run_layers`], after every layer has written its K/V.
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
        let q_dim = num_heads * hd;
        let kv_dim = num_kv * hd;

        {
            let layer = &self.layers[layer_idx];
            layer
                .attn_qkv
                .forward(&*layer.attn_qkv_kernel, &self.buf_norm, &mut self.buf_qkv)?;
        }

        // `ggml_clamp(cur, -f_clamp_kqv, +f_clamp_kqv)` on the fused projection.
        let clamp = self.dbrx_config.clamp_kqv;
        if clamp > 0.0 {
            for v in self.buf_qkv.iter_mut() {
                *v = v.clamp(-clamp, clamp);
            }
        }

        // RoPE on Q and K (NeoX pairing).
        for h in 0..num_heads {
            self.rope
                .apply(&mut self.buf_qkv[h * hd..(h + 1) * hd], pos);
        }
        for h in 0..num_kv {
            let off = q_dim + h * hd;
            self.rope.apply(&mut self.buf_qkv[off..off + hd], pos);
        }

        let (q, kv) = self.buf_qkv.split_at(q_dim);
        let (k, v) = kv.split_at(kv_dim);
        kv_cache.store_kv(layer_idx, &k[..kv_dim], &v[..kv_dim])?;

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

        let scale = 1.0 / (hd as f32).sqrt();
        self.buf_attn_out[..q_dim].fill(0.0);

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &q[h * hd..(h + 1) * hd];
            let scores = &mut self.buf_attn_scores[..n_tokens];

            for (t, score) in scores.iter_mut().enumerate() {
                let off = t * kv_dim + kv_head * hd;
                *score = q_head
                    .iter()
                    .zip(cached_keys[off..off + hd].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
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

    /// Run every token through all layers, leaving each token's hidden state in
    /// `hidden_states`.
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
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[off..off + hidden]);
        }

        let n_layers = self.layers.len();
        for t in 0..seq_len {
            let pos = start_pos + t;
            for layer_idx in 0..n_layers {
                self.layers[layer_idx].attn_norm.forward_to(
                    &hidden_states[t * hidden..(t + 1) * hidden],
                    &mut self.buf_norm,
                );

                self.attention(layer_idx, pos, kv_cache)?;

                for (h, p) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(self.buf_proj.iter())
                {
                    *h += p;
                }

                // DBRX's pre-FFN norm is `attn_output_norm`, not `ffn_norm`.
                self.layers[layer_idx].attn_out_norm.forward_to(
                    &hidden_states[t * hidden..(t + 1) * hidden],
                    &mut self.buf_norm,
                );

                self.layers[layer_idx].moe.forward(
                    &self.buf_norm,
                    &mut self.buf_moe_out,
                    &mut self.moe_scratch,
                )?;

                for (h, m) in hidden_states[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(self.buf_moe_out.iter())
                {
                    *h += m;
                }
            }
            // Exactly once per token, after every layer has written its K/V.
            // See the contract on `KvCacheAccess::advance`.
            kv_cache.advance();
        }

        Ok(hidden_states)
    }
}

impl ForwardPass for DbrxModel {
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
        Ok(logits)
    }

    /// Post-output-norm hidden state of the last token, without the LM head.
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

    /// DBRX keeps no internal per-sequence state: positions come from
    /// [`KvCacheAccess::seq_len`], which the runtime resets with the slot.
    ///
    /// The model used to carry a monotonic `self.current_pos` that was never
    /// returned to 0, so the second request in a slot started at the first
    /// one's final position and every RoPE angle was wrong.
    fn reset_sequence(&mut self) {
        self.buf_attn_scores.fill(0.0);
        self.buf_qkv.fill(0.0);
        self.buf_attn_out.fill(0.0);
        self.buf_proj.fill(0.0);
        self.buf_norm.fill(0.0);
        self.buf_moe_out.fill(0.0);
    }
}

// ─── Builder helpers (for tests and custom loaders) ───────────────────────────

/// Construct a `DbrxModel` from raw weights.
///
/// # Errors
///
/// See [`DbrxModel::new`].
pub fn build_dbrx_model(
    config: ModelConfig,
    dbrx_config: DbrxConfig,
    token_embd: Vec<f32>,
    layers: Vec<DbrxLayer>,
    output_norm: LayerNorm,
    output: QuantLinear,
) -> ArchResult<DbrxModel> {
    DbrxModel::new(config, dbrx_config, token_embd, layers, output_norm, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbrx::testkit::{TestKvCache, TinyWeights};

    fn tiny_model() -> DbrxModel {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const N_LAYERS: usize = 2;
        const N_EXPERTS: usize = 4;
        const TOP_K: usize = 2;
        const INTER: usize = 8;

        let mut w = TinyWeights::new(7);
        let dbrx_config = DbrxConfig {
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
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            clamp_kqv: 8.0,
        };
        let config = ModelConfig {
            architecture: "dbrx".to_string(),
            model_name: "test-dbrx".to_string(),
            hidden_size: H,
            intermediate_size: INTER,
            num_layers: N_LAYERS,
            num_attention_heads: N_HEADS,
            num_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };
        let kv_dim = N_HEADS * HEAD_DIM;
        let layers = (0..N_LAYERS)
            .map(|_| {
                let attn_qkv = w.linear(H + 2 * kv_dim, H);
                let attn_output = w.linear(H, H);
                let moe = w.quant_moe(H, INTER, N_EXPERTS, TOP_K);
                DbrxLayer {
                    attn_norm: LayerNorm::new(vec![1.0; H], None, 1e-5),
                    attn_qkv_kernel: w.kernel(&attn_qkv),
                    attn_output_kernel: w.kernel(&attn_output),
                    attn_qkv,
                    attn_output,
                    attn_out_norm: LayerNorm::new(vec![1.0; H], None, 1e-5),
                    moe,
                }
            })
            .collect();

        DbrxModel::new(
            config,
            dbrx_config,
            w.embedding(VOCAB, H),
            layers,
            LayerNorm::new(vec![1.0; H], None, 1e-5),
            w.linear(VOCAB, H),
        )
        .expect("tiny DBRX model builds")
    }

    fn cache(model: &DbrxModel) -> TestKvCache {
        TestKvCache::new(
            model.layers.len(),
            model.config.num_kv_heads * model.config.head_dim,
            model.config.max_context_length,
        )
    }

    #[test]
    fn forward_shape_correct() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        let logits = model
            .forward(&[1u32], &mut kv)
            .expect("forward must succeed");
        assert_eq!(logits.len(), 32);
    }

    #[test]
    fn forward_all_finite() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        let logits = model
            .forward(&[0u32], &mut kv)
            .expect("forward must succeed");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_tokens_returns_error() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        assert!(model.forward(&[], &mut kv).is_err());
    }

    #[test]
    fn embed_returns_hidden_size() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        let e = model.embed(&[1u32], &mut kv).expect("embed must succeed");
        assert_eq!(e.len(), 16);
    }

    /// `advance()` is a per-**token** operation.  When it was the last statement
    /// of `attention_single_token` it ran once per *layer*, so a 2-layer model
    /// burned two cache slots per token and layer 1 read rows layer 0 wrote.
    #[test]
    fn advance_is_called_once_per_token_not_once_per_layer() {
        let mut model = tiny_model();
        let n_layers = model.layers.len();
        assert!(n_layers >= 2, "the regression needs >= 2 layers to show");
        let mut kv = cache(&model);
        model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
        assert_eq!(
            kv.seq_len(),
            3,
            "3 tokens through {n_layers} layers must advance the cache exactly 3 times"
        );
        // Each layer must own exactly one row per token.
        for l in 0..n_layers {
            assert_eq!(
                kv.writes(l),
                3,
                "layer {l} must have written 3 rows, one per token"
            );
            // …and those rows must be 0, 1, 2 — not 0, 2, 4 (layer 0) and
            // 1, 3, 5 (layer 1), which is what a per-layer advance() produces.
            for pos in 3..6 {
                assert!(
                    kv.key_row(l, pos).iter().all(|v| *v == 0.0),
                    "layer {l} wrote past position 2 (row {pos} is non-zero): \
                     the cache advanced more than once per token"
                );
            }
        }
    }

    /// An over-long prompt must be reported, not written past the end of
    /// `buf_attn_scores`.
    #[test]
    fn context_overflow_is_reported() {
        let mut model = tiny_model();
        let max = model.config.max_context_length;
        let mut kv = cache(&model);
        let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
        assert!(
            model.forward(&tokens, &mut kv).is_err(),
            "a prompt longer than max_context_length must be rejected"
        );
    }

    /// A token id past the end of the embedding matrix must be an error, not a
    /// panic.
    #[test]
    fn out_of_vocab_token_is_reported() {
        let mut model = tiny_model();
        let mut kv = cache(&model);
        assert!(model.forward(&[999u32], &mut kv).is_err());
    }
}
