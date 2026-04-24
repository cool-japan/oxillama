//! DeepSeek-V2 transformer model implementation.
//!
//! Provides the `DeepSeekModel` struct (embedding → N×DeepSeekLayer → output norm
//! → LM head) and `load_deepseek_from_gguf()` for loading from a GGUF file.
//!
//! Each `DeepSeekLayer` contains:
//! - Pre-attention RMSNorm
//! - Multi-head Latent Attention (`mla_forward`)
//! - An arch-internal `MlaLatentCache` (NOT a `KvCacheAccess` extension)
//! - Pre-FFN RMSNorm
//! - A `FfnKind` — either dense SwiGLU (first N_DENSE layers) or DeepSeek MoE
//!
//! The `ForwardPass` impl dispatches through the layers sequentially.

use crate::common::linear::QuantLinear;
use crate::common::mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
use crate::common::rms_norm::RmsNorm;
use crate::common::swiglu::swiglu_inplace;
use crate::config::{DeepSeekConfig, ModelConfig};
use crate::deepseek::moe::{moe_forward, MoeConfig, MoeWeights};
use crate::error::{ArchError, ArchResult};
use crate::traits::{ForwardPass, KvCacheAccess};
use oxillama_quant::{KernelDispatcher, QuantTensor};

/// How many leading layers use a dense SwiGLU FFN (the rest use MoE).
///
/// DeepSeek-V2 documentation specifies 1 dense layer at the start. This
/// constant is exported for use by loaders and callers building layer vectors.
pub const N_DENSE_LAYERS: usize = 1;

// ─── Layer FFN variant ────────────────────────────────────────────────────────

/// Dense SwiGLU FFN weights for the non-MoE layers.
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
    /// Standard dense SwiGLU FFN (leading layers).
    Dense(Box<DenseFfn>),
    /// DeepSeek sparse MoE FFN.
    Moe {
        /// MoE layer weights.
        weights: Box<MoeWeights>,
        /// Configuration (shared with the layer).
        config: MoeConfig,
    },
}

// ─── Single transformer layer ─────────────────────────────────────────────────

/// One DeepSeek-V2 transformer block.
pub struct DeepSeekLayer {
    /// Pre-attention RMSNorm.
    pub attn_norm: RmsNorm,
    /// MLA weights and RoPE table.
    pub mla_weights: MlaWeights,
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

/// Complete DeepSeek-V2 model.
pub struct DeepSeekModel {
    /// Common model config (embedding dim, num layers, vocab size, etc.).
    pub config: ModelConfig,
    /// DeepSeek-specific config (MLA ranks, MoE layout, etc.).
    pub ds_config: DeepSeekConfig,
    /// Token embedding table `[vocab_size, hidden_size]` stored as f32.
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<DeepSeekLayer>,
    /// Final RMSNorm before LM head.
    pub output_norm: RmsNorm,
    /// LM head projection `[vocab_size, hidden_size]`.
    pub output: QuantLinear,
    /// Kernel dispatcher for quantized ops.
    pub dispatcher: KernelDispatcher,
    /// Sequence position counter (incremented on each forward call).
    current_pos: usize,
}

impl DeepSeekModel {
    /// Construct a `DeepSeekModel` from pre-loaded weights.
    pub fn new(
        config: ModelConfig,
        ds_config: DeepSeekConfig,
        token_embd: Vec<f32>,
        layers: Vec<DeepSeekLayer>,
        output_norm: RmsNorm,
        output: QuantLinear,
    ) -> Self {
        Self {
            dispatcher: KernelDispatcher::new(),
            current_pos: 0,
            config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        }
    }

    /// Reset the sequence position (use between independent sequences).
    pub fn reset_position(&mut self) {
        self.current_pos = 0;
        for layer in &mut self.layers {
            layer.mla_cache.clear();
        }
    }
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

        if seq_len == 0 {
            return Err(ArchError::InvalidConfig {
                detail: "forward: empty token sequence".to_string(),
            });
        }

        // ── Token embedding lookup ──────────────────────────────────────────────
        // Build [seq_len × hidden_size] input tensor from token ids.
        let mut hidden_states = vec![0.0f32; seq_len * hidden];
        for (t, &tok_id) in tokens.iter().enumerate() {
            let tok = tok_id as usize;
            if tok >= self.config.vocab_size {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "token id {tok} out of range (vocab_size={})",
                        self.config.vocab_size
                    ),
                });
            }
            let embd_off = tok * hidden;
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&self.token_embd[embd_off..embd_off + hidden]);
        }

        let position = self.current_pos;

        // ── Transformer layers ──────────────────────────────────────────────────
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // ─ Pre-attention norm ──────────────────────────────────────────────
            // Normalise each token's hidden state individually.
            let mut normed = vec![0.0f32; seq_len * hidden];
            for t in 0..seq_len {
                let src = &hidden_states[t * hidden..(t + 1) * hidden];
                let dst = &mut normed[t * hidden..(t + 1) * hidden];
                dst.copy_from_slice(src);
                layer
                    .attn_norm
                    .forward(&mut normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ MLA forward ────────────────────────────────────────────────────
            let attn_out = mla_forward(
                &normed,
                &layer.mla_weights,
                &layer.mla_config,
                &mut layer.mla_cache,
                position,
            )
            .map_err(|e| ArchError::ForwardPassError {
                layer: layer_idx,
                message: format!("MLA: {e}"),
            })?;

            // ─ Residual connection ─────────────────────────────────────────────
            for (h, a) in hidden_states.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // ─ Pre-FFN norm ────────────────────────────────────────────────────
            let mut ffn_normed = hidden_states.clone();
            for t in 0..seq_len {
                layer
                    .ffn_norm
                    .forward(&mut ffn_normed[t * hidden..(t + 1) * hidden]);
            }

            // ─ FFN (dense or MoE) ──────────────────────────────────────────────
            let mut ffn_out = vec![0.0f32; seq_len * hidden];
            match &layer.ffn {
                FfnKind::Dense(dense) => {
                    let kernel_gate = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.gate.weight,
                        layer_idx,
                        "gate",
                    )?;
                    let kernel_up = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.up.weight,
                        layer_idx,
                        "up",
                    )?;
                    let kernel_down = layer_dispatcher_kernel(
                        &self.dispatcher,
                        &dense.down.weight,
                        layer_idx,
                        "down",
                    )?;

                    let intermediate = dense.gate.out_features;
                    let mut buf_gate = vec![0.0f32; intermediate];
                    let mut buf_up = vec![0.0f32; intermediate];
                    let mut buf_ffn = vec![0.0f32; hidden];

                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        dense
                            .gate
                            .forward(&*kernel_gate, x_t, &mut buf_gate)
                            .map_err(ArchError::from)?;
                        dense
                            .up
                            .forward(&*kernel_up, x_t, &mut buf_up)
                            .map_err(ArchError::from)?;
                        swiglu_inplace(&mut buf_gate, &buf_up);
                        dense
                            .down
                            .forward(&*kernel_down, &buf_gate, &mut buf_ffn)
                            .map_err(ArchError::from)?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&buf_ffn);
                    }
                }
                FfnKind::Moe { weights, config } => {
                    for t in 0..seq_len {
                        let x_t = &ffn_normed[t * hidden..(t + 1) * hidden];
                        let tok_out = moe_forward(x_t, weights, config).map_err(|e| {
                            ArchError::ForwardPassError {
                                layer: layer_idx,
                                message: format!("MoE: {e}"),
                            }
                        })?;
                        ffn_out[t * hidden..(t + 1) * hidden].copy_from_slice(&tok_out);
                    }
                }
            }

            // ─ Residual after FFN ──────────────────────────────────────────────
            for (h, f) in hidden_states.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
        }

        // ── Final norm + LM head (last token only) ──────────────────────────────
        let last = &mut hidden_states[(seq_len - 1) * hidden..seq_len * hidden];
        self.output_norm.forward(last);

        let kernel = self
            .dispatcher
            .get_kernel(self.output.weight.tensor_type)
            .map_err(ArchError::from)?;
        let mut logits = vec![0.0f32; vocab];
        self.output
            .forward(&*kernel, last, &mut logits)
            .map_err(ArchError::from)?;

        self.current_pos += seq_len;

        Ok(logits)
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
}

// ─── Helper ───────────────────────────────────────────────────────────────────

fn layer_dispatcher_kernel(
    dispatcher: &KernelDispatcher,
    weight: &QuantTensor,
    layer_idx: usize,
    name: &str,
) -> ArchResult<Box<dyn oxillama_quant::QuantKernel>> {
    dispatcher
        .get_kernel(weight.tensor_type)
        .map_err(|e| ArchError::ForwardPassError {
            layer: layer_idx,
            message: format!("kernel dispatch for {name}: {e}"),
        })
}

// ─── GGUF loader (stub) ───────────────────────────────────────────────────────

/// Load a DeepSeek-V2 model from a parsed GGUF `TensorStore`.
///
/// # Errors
/// Returns `ArchError::MissingTensor` if any required tensor is absent.
///
/// # Current status
/// This function provides a structurally complete skeleton: it parses the
/// model config and verifies the tensor store is non-empty. Full tensor loading
/// (dequant, shape validation, per-layer weight construction) is deferred to a
/// follow-up implementation once the GGUF API for bulk tensor iteration is
/// stabilised.
pub fn load_deepseek_from_gguf(_model: &oxillama_gguf::GgufModel) -> ArchResult<DeepSeekModel> {
    Err(ArchError::MissingTensor {
        name: "load_deepseek_from_gguf: full loader not yet implemented; \
               use build_deepseek_model() directly for testing"
            .to_string(),
    })
}

/// Construct a `DeepSeekModel` from raw weights (intended for tests and
/// custom loaders that have already materialised the tensors).
///
/// All shape checks are deferred to the constituent layer/weight constructors.
pub fn build_deepseek_model(
    config: ModelConfig,
    ds_config: DeepSeekConfig,
    token_embd: Vec<f32>,
    layers: Vec<DeepSeekLayer>,
    output_norm: RmsNorm,
    output: QuantLinear,
) -> DeepSeekModel {
    DeepSeekModel::new(config, ds_config, token_embd, layers, output_norm, output)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::mla::{MlaConfig, MlaLatentCache, MlaWeights};
    use crate::common::rms_norm::RmsNorm;
    use crate::common::rope::RopeTable;
    use crate::deepseek::moe::{DeepSeekExpert, MoeConfig, MoeWeights, ScoringMode};
    use crate::error::ArchResult;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    /// Minimal LCG (same as in mla.rs tests, no dependency on test infrastructure).
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = (self.state >> 33) as u32 & 0x007f_ffff;
            let bits = mantissa | 0x3f80_0000u32;
            (f32::from_bits(bits) - 1.5) * 0.02
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    fn rand_f32_tensor(lcg: &mut Lcg, rows: usize, cols: usize) -> QuantTensor {
        let n = rows * cols;
        let mut vals = vec![0.0f32; n];
        lcg.fill(&mut vals);
        let mut data = Vec::with_capacity(n * 4);
        for &v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        QuantTensor::new(data, vec![rows, cols], GgufTensorType::F32)
    }

    fn build_test_model(lcg: &mut Lcg) -> DeepSeekModel {
        const H: usize = 16;
        const VOCAB: usize = 32;
        const INTERMEDIATE: usize = 32;
        const N_LAYERS: usize = 2;
        const MAX_SEQ: usize = 64;

        let mla_cfg = MlaConfig {
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            rope_theta: 10000.0,
            softmax_scale: 1.0 / (8.0f32).sqrt(),
        };

        let build_mla_weights = |lcg: &mut Lcg| -> MlaWeights {
            let q_full = mla_cfg.q_full_dim();
            let kv_comb = mla_cfg.kv_combined_dim();
            let kv_b_full = mla_cfg.kv_b_full_dim();
            let attn_out = mla_cfg.attn_out_dim();
            MlaWeights {
                w_q_a: QuantLinear::new(rand_f32_tensor(lcg, mla_cfg.q_lora_rank, H), None),
                q_a_norm: RmsNorm::new(
                    {
                        let mut w = vec![0.0f32; mla_cfg.q_lora_rank];
                        lcg.fill(&mut w);
                        w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                        w
                    },
                    1e-5,
                ),
                w_q_b: QuantLinear::new(rand_f32_tensor(lcg, q_full, mla_cfg.q_lora_rank), None),
                w_kv_a: QuantLinear::new(rand_f32_tensor(lcg, kv_comb, H), None),
                kv_a_norm: RmsNorm::new(
                    {
                        let mut w = vec![0.0f32; mla_cfg.kv_lora_rank];
                        lcg.fill(&mut w);
                        w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                        w
                    },
                    1e-5,
                ),
                w_kv_b: QuantLinear::new(
                    rand_f32_tensor(lcg, kv_b_full, mla_cfg.kv_lora_rank),
                    None,
                ),
                w_o: QuantLinear::new(rand_f32_tensor(lcg, H, attn_out), None),
                rope: RopeTable::new_standard(
                    mla_cfg.qk_rope_head_dim,
                    MAX_SEQ,
                    mla_cfg.rope_theta,
                ),
            }
        };

        let build_dense_ffn = |lcg: &mut Lcg| -> DenseFfn {
            DenseFfn {
                gate: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, H), None),
                up: QuantLinear::new(rand_f32_tensor(lcg, INTERMEDIATE, H), None),
                down: QuantLinear::new(rand_f32_tensor(lcg, H, INTERMEDIATE), None),
            }
        };

        let build_moe_ffn = |lcg: &mut Lcg| -> (MoeWeights, MoeConfig) {
            const N_EXPERTS: usize = 4;
            const TOP_K: usize = 2;
            let moe_cfg = MoeConfig {
                hidden_size: H,
                expert_intermediate_size: 16,
                n_shared_experts: 1,
                n_routed_experts: N_EXPERTS,
                top_k: TOP_K,
                routed_scaling_factor: 1.0,
                scoring_mode: ScoringMode::Softmax,
                shared_expert_intermediate_size: 16,
            };
            let make_expert = |lcg: &mut Lcg, inter: usize| -> DeepSeekExpert {
                let mut gate = vec![0.0f32; inter * H];
                let mut up = vec![0.0f32; inter * H];
                let mut down = vec![0.0f32; H * inter];
                lcg.fill(&mut gate);
                lcg.fill(&mut up);
                lcg.fill(&mut down);
                DeepSeekExpert {
                    gate,
                    up,
                    down,
                    hidden_size: H,
                    intermediate_size: inter,
                }
            };
            let mut router = vec![0.0f32; N_EXPERTS * H];
            lcg.fill(&mut router);
            let moe_weights = MoeWeights {
                router,
                routed_experts: (0..N_EXPERTS).map(|_| make_expert(lcg, 16)).collect(),
                shared_experts: vec![make_expert(lcg, 16)],
                expert_bias: None,
            };
            (moe_weights, moe_cfg)
        };

        let mut token_embd = vec![0.0f32; VOCAB * H];
        lcg.fill(&mut token_embd);

        let mut norm_w = vec![0.0f32; H];
        lcg.fill(&mut norm_w);
        norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);

        let ds_config = DeepSeekConfig {
            q_lora_rank: mla_cfg.q_lora_rank,
            kv_lora_rank: mla_cfg.kv_lora_rank,
            qk_nope_head_dim: mla_cfg.qk_nope_head_dim,
            qk_rope_head_dim: mla_cfg.qk_rope_head_dim,
            v_head_dim: mla_cfg.v_head_dim,
            n_shared_experts: 1,
            n_routed_experts: 4,
            top_k_routed: 2,
            shared_expert_intermediate_size: 16,
            routed_scaling_factor: 1.0,
        };

        let model_config = ModelConfig {
            architecture: "deepseek2".to_string(),
            model_name: "test-deepseek".to_string(),
            hidden_size: H,
            intermediate_size: INTERMEDIATE,
            num_layers: N_LAYERS,
            num_attention_heads: mla_cfg.num_heads,
            num_kv_heads: mla_cfg.num_heads,
            head_dim: mla_cfg.qk_head_dim(),
            vocab_size: VOCAB,
            max_context_length: MAX_SEQ,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            ..ModelConfig::default()
        };

        let layers = (0..N_LAYERS)
            .map(|idx| {
                let mut attn_norm_w = vec![0.0f32; H];
                lcg.fill(&mut attn_norm_w);
                attn_norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);
                let mut ffn_norm_w = vec![0.0f32; H];
                lcg.fill(&mut ffn_norm_w);
                ffn_norm_w.iter_mut().for_each(|v| *v = v.abs() + 0.1);

                let ffn = if idx < N_DENSE_LAYERS {
                    FfnKind::Dense(Box::new(build_dense_ffn(lcg)))
                } else {
                    let (moe_weights, moe_cfg) = build_moe_ffn(lcg);
                    FfnKind::Moe {
                        weights: Box::new(moe_weights),
                        config: moe_cfg,
                    }
                };

                DeepSeekLayer {
                    attn_norm: RmsNorm::new(attn_norm_w, 1e-5),
                    mla_weights: build_mla_weights(lcg),
                    mla_config: mla_cfg.clone(),
                    mla_cache: MlaLatentCache::new(MAX_SEQ, &mla_cfg),
                    ffn_norm: RmsNorm::new(ffn_norm_w, 1e-5),
                    ffn,
                }
            })
            .collect();

        let output_norm = RmsNorm::new(norm_w, 1e-5);
        let output = QuantLinear::new(rand_f32_tensor(lcg, VOCAB, H), None);

        DeepSeekModel::new(
            model_config,
            ds_config,
            token_embd,
            layers,
            output_norm,
            output,
        )
    }

    /// Forward pass produces logits of the correct shape (vocab_size).
    #[test]
    fn forward_shape() {
        let mut lcg = Lcg::new(1234);
        let mut model = build_test_model(&mut lcg);

        // Stub KV cache (MLA doesn't use it; the trait methods return Results)
        struct NullKv;
        impl KvCacheAccess for NullKv {
            fn seq_len(&self) -> usize {
                0
            }
            fn store_kv(
                &mut self,
                _layer: usize,
                _keys: &[f32],
                _values: &[f32],
            ) -> ArchResult<()> {
                Ok(())
            }
            fn get_keys(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn get_values(&self, _layer: usize) -> ArchResult<&[f32]> {
                Ok(&[])
            }
            fn advance(&mut self) {}
        }

        let mut kv = NullKv;
        let out = model
            .forward(&[1u32, 2, 3], &mut kv)
            .expect("forward must succeed");
        assert_eq!(
            out.len(),
            32,
            "logits must have vocab_size=32 elements, got {}",
            out.len()
        );
    }

    /// Forward pass is deterministic (same tokens → same logits after reset).
    #[test]
    fn forward_determinism() {
        let mut lcg = Lcg::new(9999);
        let mut model = build_test_model(&mut lcg);

        struct NullKv;
        impl KvCacheAccess for NullKv {
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

        let mut kv = NullKv;
        let out1 = model.forward(&[1u32, 0], &mut kv).expect("first forward");

        model.reset_position();
        let mut kv2 = NullKv;
        let out2 = model.forward(&[1u32, 0], &mut kv2).expect("second forward");

        assert_eq!(out1.len(), out2.len());
        for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            assert_eq!(
                a_bits, b_bits,
                "logits[{i}] must be bit-for-bit identical: {a} vs {b}"
            );
        }
    }
}
