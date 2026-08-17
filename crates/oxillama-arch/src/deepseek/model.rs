//! DeepSeek-V2 / V2-Lite / V3 transformer block and forward pass.
//!
//! ```text
//! embedding
//!   → N×( RMSNorm → MLA (decoupled RoPE over the qk_rope slice) → residual
//!         → RMSNorm → dense FFN (first k layers) | routed MoE + shared expert
//!         → residual )
//!   → RMSNorm → LM head
//! ```
//!
//! # Fixed here
//!
//! * **The combination weight is the UNBIASED probability.**  The old
//!   `moe_forward` folded `exp_probs_b` into `scores` in place and then used
//!   that same biased value as the weight.  `build_moe_ffn` keeps them separate:
//!   `selection_probs = probs + exp_probs_b` selects, `weights = probs[selected]`
//!   combines — llama.cpp's own comment is "leave probs unbiased as it's later
//!   used to get expert weights".  Since `exp_probs_b` is a load-balancing term
//!   that can be negative, the old code could produce negative weights and the
//!   `weight_sum > 0.0` guard then zeroed the entire routed branch.
//! * **`top_k == 0` is an error, not a panic.**  `moe_forward` guarded only
//!   `top_k > n_routed_experts` and then called
//!   `select_nth_unstable_by(top_k - 1, …)`, underflowing on `top_k == 0` —
//!   a value that comes straight from `deepseek2.expert_used_count`.
//! * **Group-limited routing, `expert_weights_norm`, `expert_gating_func` and
//!   `expert_weights_scale`** — see [`crate::deepseek::loader::DeepSeekRouting`].
//! * **The Lite Q path** — see [`crate::deepseek::mla`].
//! * **`reset_sequence()` clears `MlaLatentCache`.**  It used to grow until
//!   `append` errored, and before that leaked the previous request's tokens into
//!   `attend_len`, so a second request attended to the first one's keys.
//! * **Quantized experts.**  `Expert { gate: Vec<f32>, … }` dequantized the
//!   whole pool at load.
//!
//! # Verified and NOT changed
//!
//! `llama_model_rope_type` puts `LLM_ARCH_DEEPSEEK2` in the `LLAMA_ROPE_TYPE_NORM`
//! group, so the loader builds the table with [`RopeStyle::Norm`].  That is safe
//! here precisely because RoPE only ever touches the `qk_rope` slice of `q_full`
//! and the `qk_rope` tail of `kv_a_mqa` — the MLA tensors never go through the
//! dense families' `permute()` path, so the NORM/NEOX distinction is a pure
//! pairing choice on those slices and matches the reference.
//!
//! [`RopeStyle::Norm`]: crate::common::rope::RopeStyle::Norm

use oxillama_quant::KernelDispatcher;

use crate::common::attention::validate_context_bounds;
use crate::common::linear::QuantLinear;
use crate::common::mla::{MlaConfig, MlaLatentCache};
use crate::common::moe::MoeScratch;
use crate::common::rms_norm::RmsNorm;
use crate::config::{DeepSeekConfig, ModelConfig};
use crate::deepseek::mla::{ds_mla_forward, DeepSeekMlaWeights};
use crate::deepseek::quant_moe::RoutedQuantMoe;
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};

/// Number of leading dense (non-MoE) layers in DeepSeek-V2.
///
/// Only a default; the real value is `deepseek2.leading_dense_block_count`.
pub const N_DENSE_LAYERS: usize = 1;

// ─── Layer FFN variant ────────────────────────────────────────────────────────

/// Dense SwiGLU FFN weights for the leading non-MoE layers.
pub struct DenseFfn {
    /// Gate projection `[intermediate_size, hidden_size]`.
    pub gate: QuantLinear,
    /// Up projection `[intermediate_size, hidden_size]`.
    pub up: QuantLinear,
    /// Down projection `[hidden_size, intermediate_size]`.
    pub down: QuantLinear,
}

/// FFN variant for a DeepSeek layer.
pub enum FfnKind {
    /// Standard dense SwiGLU FFN (`il < n_layer_dense_lead`).
    Dense(Box<DenseFfn>),
    /// Routed MoE plus the always-active shared expert.
    Moe(Box<RoutedQuantMoe>),
}

// ─── Single transformer layer ─────────────────────────────────────────────────

/// One DeepSeek transformer block.
pub struct DeepSeekLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// MLA weights and RoPE table.
    pub mla_weights: DeepSeekMlaWeights,
    /// MLA configuration.
    pub mla_config: MlaConfig,
    /// Arch-internal latent KV cache for this layer.
    pub mla_cache: MlaLatentCache,
    /// Pre-FFN RMSNorm.
    pub ffn_norm: RmsNorm,
    /// FFN variant.
    pub ffn: FfnKind,
}

// ─── Full model ───────────────────────────────────────────────────────────────

/// Complete DeepSeek model.
pub struct DeepSeekModel {
    /// Common model config (embedding dim, num layers, vocab size, etc.).
    pub config: ModelConfig,
    /// DeepSeek-specific config (MLA ranks, MoE layout, etc.).
    pub ds_config: DeepSeekConfig,
    /// Token embedding table `[vocab_size, hidden_size]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Number of embedding rows actually present in `token_embd`.
    embd_rows: usize,
    /// Transformer layers.
    pub layers: Vec<DeepSeekLayer>,
    /// Final RMSNorm before the LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    moe_scratch: MoeScratch,
}

impl DeepSeekModel {
    /// Construct a `DeepSeekModel` from pre-loaded weights.
    ///
    /// # Errors
    ///
    /// [`ArchError::InvalidShape`] when `token_embd` holds fewer than
    /// `vocab_size × hidden_size` values, or
    /// [`ArchError::InvalidConfig`] for a degenerate geometry.
    pub fn new(
        config: ModelConfig,
        ds_config: DeepSeekConfig,
        token_embd: Vec<f32>,
        layers: Vec<DeepSeekLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> ArchResult<Self> {
        let hidden = config.hidden_size;
        if hidden == 0 || config.vocab_size == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "DeepSeek requires hidden_size > 0 and vocab_size > 0".to_string(),
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
            .iter()
            .find_map(|l| match &l.ffn {
                FfnKind::Moe(m) => Some(m.make_scratch()),
                FfnKind::Dense(_) => None,
            })
            .unwrap_or_default();

        Ok(Self {
            dispatcher: KernelDispatcher::new(),
            moe_scratch,
            embd_rows,
            config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// Reset the per-sequence state (latent caches).
    ///
    /// Equivalent to [`ForwardPass::reset_sequence`]; kept as an inherent method
    /// because it predates the trait hook.
    pub fn reset_position(&mut self) {
        for layer in &mut self.layers {
            layer.mla_cache.clear();
        }
    }

    /// Run every token through all layers, returning `[seq_len × hidden]`.
    fn run_layers(&mut self, tokens: &[u32]) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward: empty token sequence".to_string(),
            });
        }

        // MLA keeps its own latent cache, so the position of the first new token
        // is that cache's current length rather than the external KV cache's.
        let start_pos = self.layers.first().map_or(0, |l| l.mla_cache.seq_len);
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

        let mut normed = vec![0.0f32; seq_len * hidden];
        let mut ffn_out = vec![0.0f32; hidden];
        let mut scratch = std::mem::take(&mut self.moe_scratch);
        let result = (|| -> ArchResult<()> {
            for layer_idx in 0..self.layers.len() {
                {
                    let layer = &self.layers[layer_idx];
                    for t in 0..seq_len {
                        layer.attn_norm.forward_to(
                            &hidden_states[t * hidden..(t + 1) * hidden],
                            &mut normed[t * hidden..(t + 1) * hidden],
                        );
                    }
                }

                let attn = {
                    let layer = &mut self.layers[layer_idx];
                    // Use *this* layer's cache length, not the hoisted
                    // `start_pos`: `ds_mla_forward` computes
                    // `cache_start = cache.seq_len - seq_len` after appending,
                    // so a layer whose cache ever fell out of lockstep would
                    // underflow rather than merely produce a wrong position.
                    let layer_pos = layer.mla_cache.seq_len;
                    ds_mla_forward(
                        &normed,
                        &layer.mla_weights,
                        &layer.mla_config,
                        &mut layer.mla_cache,
                        layer_pos,
                    )?
                };
                for (h, a) in hidden_states.iter_mut().zip(attn.iter()) {
                    *h += a;
                }

                for t in 0..seq_len {
                    let layer = &self.layers[layer_idx];
                    layer.ffn_norm.forward_to(
                        &hidden_states[t * hidden..(t + 1) * hidden],
                        &mut normed[t * hidden..(t + 1) * hidden],
                    );
                    let x = &normed[t * hidden..(t + 1) * hidden];
                    match &layer.ffn {
                        FfnKind::Dense(dense) => dense_ffn_forward(
                            dense,
                            &self.dispatcher,
                            x,
                            &mut ffn_out,
                            &mut scratch,
                        )?,
                        FfnKind::Moe(moe) => moe.forward(x, &mut ffn_out, &mut scratch)?,
                    }
                    for (h, f) in hidden_states[t * hidden..(t + 1) * hidden]
                        .iter_mut()
                        .zip(ffn_out.iter())
                    {
                        *h += f;
                    }
                }
            }
            Ok(())
        })();
        self.moe_scratch = scratch;
        result?;

        Ok(hidden_states)
    }
}

/// `down(silu(gate(x)) * up(x))` for the leading dense layers.
fn dense_ffn_forward(
    ffn: &DenseFfn,
    dispatcher: &KernelDispatcher,
    input: &[f32],
    output: &mut [f32],
    scratch: &mut MoeScratch,
) -> ArchResult<()> {
    let n = ffn.gate.out_features;
    scratch.ensure_activations(n);
    let gate_kernel = dispatcher.get_kernel(ffn.gate.weight.tensor_type)?;
    let up_kernel = dispatcher.get_kernel(ffn.up.weight.tensor_type)?;
    let down_kernel = dispatcher.get_kernel(ffn.down.weight.tensor_type)?;

    ffn.gate
        .forward(&*gate_kernel, input, &mut scratch.gate[..n])?;
    ffn.up.forward(&*up_kernel, input, &mut scratch.up[..n])?;
    for (u, g) in scratch.up[..n].iter_mut().zip(scratch.gate[..n].iter()) {
        *u *= *g / (1.0 + (-*g).exp());
    }
    ffn.down.forward(&*down_kernel, &scratch.up[..n], output)?;
    Ok(())
}

impl ForwardPass for DeepSeekModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        _kv_cache: &mut dyn KvCacheAccess,
    ) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = tokens.len();
        let mut hidden_states = self.run_layers(tokens)?;

        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        let lm_kernel = self.dispatcher.get_kernel(self.output.weight.tensor_type)?;
        let mut logits = vec![0.0f32; vocab];
        self.output.forward(&*lm_kernel, last, &mut logits)?;
        Ok(logits)
    }

    fn embed(&mut self, tokens: &[u32], _kv_cache: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = tokens.len();
        let mut hidden_states = self.run_layers(tokens)?;
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

    /// Clear every layer's [`MlaLatentCache`].
    ///
    /// Without this the cache grew until `append` returned `InvalidConfig`, and
    /// long before that a new request's `attend_len` included the previous
    /// request's tokens.  `tests/deepseek.rs` worked around it by rebuilding the
    /// whole model between sequences.
    fn reset_sequence(&mut self) {
        for layer in &mut self.layers {
            layer.mla_cache.clear();
        }
    }
}

/// Construct a `DeepSeekModel` from raw weights.
///
/// # Errors
///
/// See [`DeepSeekModel::new`].
pub fn build_deepseek_model(
    config: ModelConfig,
    ds_config: DeepSeekConfig,
    token_embd: Vec<f32>,
    layers: Vec<DeepSeekLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> ArchResult<DeepSeekModel> {
    DeepSeekModel::new(config, ds_config, token_embd, layers, output_norm, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek::loader::load_deepseek_from_gguf;

    struct NoKv;
    impl KvCacheAccess for NoKv {
        fn seq_len(&self) -> usize {
            0
        }
        fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
            Ok(())
        }
        fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
            Ok(&[])
        }
        fn advance(&mut self) {}
    }

    fn load_v2() -> DeepSeekModel {
        let bytes = oxillama_gguf::test_utils::build_minimal_deepseek_gguf();
        let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture parses");
        load_deepseek_from_gguf(&gguf).expect("load_deepseek_from_gguf")
    }

    #[test]
    fn forward_shape() {
        let mut model = load_v2();
        let mut kv = NoKv;
        let logits = model.forward(&[1u32, 2], &mut kv).expect("forward");
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embed_returns_hidden_size() {
        let mut model = load_v2();
        let mut kv = NoKv;
        let e = model.embed(&[1u32], &mut kv).expect("embed");
        assert_eq!(e.len(), model.config.hidden_size);
    }

    #[test]
    fn empty_tokens_returns_error() {
        let mut model = load_v2();
        let mut kv = NoKv;
        assert!(model.forward(&[], &mut kv).is_err());
    }

    /// `reset_sequence()` must clear the latent cache so a second request does
    /// not attend to the first request's keys.  Before it existed,
    /// `tests/deepseek.rs` rebuilt the entire model between sequences.
    #[test]
    fn reset_sequence_clears_the_latent_cache() {
        let mut model = load_v2();
        let mut kv = NoKv;
        let first = model.forward(&[1u32, 2], &mut kv).expect("first");
        assert!(
            model.layers[0].mla_cache.seq_len > 0,
            "the cache must hold the first request's tokens"
        );

        model.reset_sequence();
        for (i, layer) in model.layers.iter().enumerate() {
            assert_eq!(
                layer.mla_cache.seq_len, 0,
                "layer {i} latent cache must be empty after reset_sequence()"
            );
        }

        let second = model.forward(&[1u32, 2], &mut kv).expect("second");
        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logit {i} differs between two identical requests ({a} vs {b}): \
                 the second one attended to the first one's keys"
            );
        }
    }

    /// Without a reset the cache keeps growing; a run long enough to exceed
    /// `max_context_length` must be reported, not silently truncated.
    #[test]
    fn context_overflow_is_reported() {
        let mut model = load_v2();
        let mut kv = NoKv;
        let max = model.config.max_context_length;
        let tokens: Vec<u32> = (0..=max as u32).map(|t| t % 4).collect();
        assert!(model.forward(&tokens, &mut kv).is_err());
    }

    #[test]
    fn out_of_vocab_token_is_reported() {
        let mut model = load_v2();
        let mut kv = NoKv;
        assert!(model.forward(&[9999u32], &mut kv).is_err());
    }
}
