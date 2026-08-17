//! GGUF loader for DeepSeek-V2 / V2-Lite / V3.
//!
//! # Tensor names
//!
//! The previous loader looked experts up as
//! `blk.{i}.ffn_exp.{e}.ffn_gate.weight` and
//! `blk.{i}.ffn_shared_exp.{e}.ffn_gate.weight`.  **Those names exist only in
//! this repository's own fixture.**  `llama_model::load_tensors`
//! (`case LLM_ARCH_DEEPSEEK2:`) creates
//!
//! ```text
//! blk.%d.ffn_gate_inp.weight     [n_expert, n_embd]
//! blk.%d.exp_probs_b.bias        [n_expert]                    (V3, optional)
//! blk.%d.ffn_gate_exps.weight    [n_expert, n_ff_exp, n_embd]  stacked
//! blk.%d.ffn_up_exps.weight      [n_expert, n_ff_exp, n_embd]  stacked
//! blk.%d.ffn_down_exps.weight    [n_expert, n_embd, n_ff_exp]  stacked
//! blk.%d.ffn_gate_shexp.weight   [n_ff_exp * n_expert_shared, n_embd]   ONE tensor
//! blk.%d.ffn_up_shexp.weight     [n_ff_exp * n_expert_shared, n_embd]
//! blk.%d.ffn_down_shexp.weight   [n_embd, n_ff_exp * n_expert_shared]
//! ```
//!
//! Note the two shapes that were also wrong: `exp_probs_b` is a **`.bias`**
//! suffix (`tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i)`), and the shared expert
//! is a **single** FFN of width `n_ff_exp * n_expert_shared`, not
//! `n_expert_shared` separate experts.
//!
//! Attention:
//!
//! ```text
//! blk.%d.attn_q.weight       (Lite: q_lora_rank absent)
//! blk.%d.attn_q_a.weight  + blk.%d.attn_q_a_norm.weight + blk.%d.attn_q_b.weight
//! blk.%d.attn_kv_a_mqa.weight  + blk.%d.attn_kv_a_norm.weight
//! blk.%d.attn_kv_b.weight
//! blk.%d.attn_output.weight
//! ```
//!
//! `attn_kv_a_mqa` is the name llama.cpp actually writes; the previous loader
//! tried `attn_kv_a_proj` then `attn_kv_a` and never `attn_kv_a_mqa`.
//!
//! # `qk_nope_head_dim` (D2)
//!
//! `crate::config::DeepSeekConfig` maps `qk_nope_head_dim ←
//! "deepseek2.attention.key_length"` and `qk_rope_head_dim ←
//! "deepseek2.attention.rope_head_dim"`.  Verified against
//! `convert_hf_to_gguf.py::DeepseekV2Model`:
//!
//! ```python
//! self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
//! self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
//! ```
//!
//! and `rg rope_head_dim gguf-py/` finds **nothing** — there is no
//! `deepseek2.attention.rope_head_dim` key anywhere.  On a real checkpoint
//! `qk_nope_head_dim` therefore became 192 (the *sum*) and `qk_rope_head_dim`
//! silently defaulted to 64, making the `w_q_b` head split,
//! `kv_b_per_head_dim()` and `softmax_scale` all wrong.  [`resolve_head_dims`]
//! applies the correct derivation locally; the same fix is needed in
//! `config.rs`.

use oxillama_gguf::{GgufModel, MetadataStore};
use oxillama_quant::KernelDispatcher;

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    dequant_to_f32, load_dequant_tensor, load_lm_head, load_quant_linear, load_stacked_experts,
};
use crate::common::mla::{MlaConfig, MlaLatentCache};
use crate::common::moe::QuantExpert;
use crate::common::rms_norm::RmsNorm;
use crate::common::rope::{RopeParams, RopeStyle, RopeTable};
use crate::config::{DeepSeekConfig, ModelConfig};
use crate::deepseek::mla::{DeepSeekMlaWeights, QProjection};
use crate::deepseek::model::{DeepSeekLayer, DeepSeekModel, DenseFfn, FfnKind};
use crate::deepseek::quant_moe::{ExpertActivation, GatingFunc, RoutedMoeConfig, RoutedQuantMoe};
use crate::error::{ArchError, ArchResult};

/// The two MLA head dimensions, derived the way `convert_hf_to_gguf.py` writes
/// them.
///
/// * `key_length      = qk_nope_head_dim + qk_rope_head_dim` (192 for V2/V3)
/// * `rope.dimension_count = qk_rope_head_dim`               (64 for V2/V3)
///
/// so `qk_nope_head_dim = key_length − rope_dimension_count`.
///
/// # Errors
///
/// [`ArchError::InvalidConfig`] when `rope.dimension_count >= key_length`,
/// which would make the nope slice empty or negative.
pub fn resolve_head_dims(
    metadata: &MetadataStore,
    fallback: &DeepSeekConfig,
) -> ArchResult<(usize, usize)> {
    let key_length = metadata
        .get_u32("deepseek2.attention.key_length")
        .map(|v| v as usize)
        .ok();
    let rope_dims = metadata
        .get_u32("deepseek2.rope.dimension_count")
        .map(|v| v as usize)
        .ok();

    match (key_length, rope_dims) {
        (Some(k), Some(r)) => {
            if r >= k {
                return Err(ArchError::InvalidConfig {
                    detail: format!(
                        "deepseek2.rope.dimension_count ({r}) must be < \
                         deepseek2.attention.key_length ({k}); \
                         qk_nope_head_dim = key_length - rope_dimension_count"
                    ),
                });
            }
            Ok((k - r, r))
        }
        // No `rope.dimension_count`: fall back to whatever config.rs produced.
        // The in-repo fixture writes the (non-existent) `attention.rope_head_dim`
        // key, which is the only reason this path is exercised.
        _ => Ok((fallback.qk_nope_head_dim, fallback.qk_rope_head_dim)),
    }
}

/// The routing knobs `config.rs` does not parse.
///
/// llama.cpp reads all five for `LLM_ARCH_DEEPSEEK2`.  None of them existed in
/// this crate, so DeepSeek-V3 routed flat over all 256 experts, always
/// re-normalised (V2 ships `norm_topk_prob = false`), and dropped the 2.5×
/// `expert_weights_scale`.
#[derive(Debug, Clone)]
pub struct DeepSeekRouting {
    /// `deepseek2.expert_group_count` (V3: 8).
    pub n_group: usize,
    /// `deepseek2.expert_group_used_count` (V3: 4).
    pub topk_group: usize,
    /// `deepseek2.expert_weights_norm` (V2: false, V3: true).
    pub weights_norm: bool,
    /// `deepseek2.expert_gating_func` (1 = softmax, 2 = sigmoid).
    pub gating: GatingFunc,
    /// `deepseek2.expert_weights_scale` (V3: 2.5).
    pub weights_scale: f32,
    /// `deepseek2.expert_feed_forward_length` — the **expert** FFN width, which
    /// is not `{arch}.feed_forward_length`.
    pub n_ff_exp: usize,
}

impl DeepSeekRouting {
    /// Read the five routing keys plus `n_ff_exp` from GGUF metadata.
    ///
    /// # Errors
    ///
    /// [`ArchError::NotSupported`] for an unknown `expert_gating_func`.
    pub fn from_metadata(metadata: &MetadataStore, fallback_ff: usize) -> ArchResult<Self> {
        let n_group = metadata
            .get_u32("deepseek2.expert_group_count")
            .map(|v| v as usize)
            .unwrap_or(1);
        let topk_group = metadata
            .get_u32("deepseek2.expert_group_used_count")
            .map(|v| v as usize)
            .unwrap_or(1);
        // llama.cpp reads this as a bool; accept either encoding because
        // `MetadataStore` has no `get_bool`.
        let weights_norm = match metadata.get("deepseek2.expert_weights_norm") {
            Some(oxillama_gguf::MetadataValue::Bool(b)) => *b,
            Some(oxillama_gguf::MetadataValue::Uint32(v)) => *v != 0,
            Some(oxillama_gguf::MetadataValue::Uint8(v)) => *v != 0,
            _ => false,
        };
        let gating = GatingFunc::from_gguf_value(
            metadata
                .get_u32("deepseek2.expert_gating_func")
                .unwrap_or(0),
        )?;
        let weights_scale = metadata
            .get_f32("deepseek2.expert_weights_scale")
            .unwrap_or(1.0);
        let n_ff_exp = metadata
            .get_u32("deepseek2.expert_feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(fallback_ff);
        Ok(Self {
            n_group,
            topk_group,
            weights_norm,
            gating,
            weights_scale,
            n_ff_exp,
        })
    }
}

/// Build the MLA config for one model, including the YaRN `mscale` correction.
///
/// `llm_build_deepseek2`:
///
/// ```text
/// mscale   = attn_factor_org * (1 + 0.1 * rope_yarn_log_mul * log(1 / freq_scale));
/// kq_scale = mscale * mscale / sqrt(n_embd_head_k);
/// ```
///
/// i.e. the softmax scale is `mscale² / sqrt(qk_nope + qk_rope)` over the
/// **full** 192-wide head, not `1/sqrt(...)`.  V2/V3 GGUFs ship
/// `rope.scaling.type = yarn` with `factor = 40`, so ignoring `mscale` scaled
/// every attention logit wrong.  [`RopeTable::mscale`] supplies the generic YaRN
/// magnitude correction that this crate's RoPE table already folds into
/// `cos`/`sin`; DeepSeek needs the *same* value squared on the softmax scale.
fn build_mla_config(
    config: &ModelConfig,
    ds: &DeepSeekConfig,
    qk_nope: usize,
    qk_rope: usize,
    rope: &RopeTable,
) -> MlaConfig {
    let head_dim = qk_nope + qk_rope;
    let mscale = rope.mscale();
    let softmax_scale = if head_dim > 0 {
        mscale * mscale / (head_dim as f32).sqrt()
    } else {
        1.0
    };
    MlaConfig {
        num_heads: config.num_attention_heads,
        q_lora_rank: ds.q_lora_rank,
        kv_lora_rank: ds.kv_lora_rank,
        qk_nope_head_dim: qk_nope,
        qk_rope_head_dim: qk_rope,
        v_head_dim: ds.v_head_dim,
        rope_theta: config.rope_freq_base,
        softmax_scale,
    }
}

/// Load a DeepSeek-V2 / V2-Lite / V3 model from a parsed GGUF file.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] naming the first absent tensor.
/// * [`ArchError::InvalidConfig`] on a contradictory head-dimension or routing
///   configuration, or when the checkpoint carries only the fused
///   `ffn_gate_up_exps` / the split `attn_k_b` + `attn_v_b` forms this loader
///   does not implement.
pub fn load_deepseek_from_gguf(model: &GgufModel) -> ArchResult<DeepSeekModel> {
    let metadata = &model.file.metadata;
    let dispatcher = KernelDispatcher::new();

    let config = ModelConfig::from_metadata(metadata)?;
    let ds_config = DeepSeekConfig::from_metadata(metadata, config.hidden_size);
    let (qk_nope, qk_rope) = resolve_head_dims(metadata, &ds_config)?;
    let routing = DeepSeekRouting::from_metadata(metadata, config.intermediate_size)?;

    let num_layers = config.num_layers;
    let max_seq = config.max_context_length;

    // DeepSeek is `LLAMA_ROPE_TYPE_NORM` in llama.cpp's rope-type table, and
    // `RopeParams::from_metadata` picks up the `rope.scaling.type = yarn`,
    // `factor = 40` and the beta/ext parameters that V2/V3 GGUFs ship.
    let mut rope_params = RopeParams::from_metadata(metadata, &config.architecture)?;
    rope_params.style = RopeStyle::Norm;
    let rope = RopeTable::new_with_params(qk_rope, max_seq, config.rope_freq_base, &rope_params)?;
    let mla_config = build_mla_config(&config, &ds_config, qk_nope, qk_rope, &rope);

    let embd_info =
        model
            .file
            .tensors
            .get("token_embd.weight")
            .map_err(|_| ArchError::MissingTensor {
                name: "token_embd.weight".to_string(),
            })?;
    let embd_data = model.tensor_data("token_embd.weight")?;
    let token_embd = dequant_to_f32(embd_info, embd_data, &dispatcher)?;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let p = format!("blk.{i}");
        let eps = config.rms_norm_eps;

        let attn_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{p}.attn_norm.weight"))?,
            eps,
        );

        // Q path: `const bool is_lite = model.layers[il].wq;`
        let q_name = format!("{p}.attn_q.weight");
        let q = if model.file.tensors.contains(&q_name) {
            QProjection::Dense {
                w_q: load_quant_linear(model, &q_name)?,
            }
        } else {
            QProjection::LowRank {
                w_q_a: load_quant_linear(model, &format!("{p}.attn_q_a.weight"))?,
                q_a_norm: RmsNorm::new(
                    load_dequant_tensor(model, &format!("{p}.attn_q_a_norm.weight"))?,
                    eps,
                ),
                w_q_b: load_quant_linear(model, &format!("{p}.attn_q_b.weight"))?,
            }
        };

        // `LLM_TENSOR_ATTN_KV_A_MQA` → `blk.%d.attn_kv_a_mqa`.
        let w_kv_a = load_quant_linear(model, &format!("{p}.attn_kv_a_mqa.weight"))?;
        let kv_a_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{p}.attn_kv_a_norm.weight"))?,
            eps,
        );
        let kv_b_name = format!("{p}.attn_kv_b.weight");
        if !model.file.tensors.contains(&kv_b_name) {
            let split = model.file.tensors.contains(&format!("{p}.attn_k_b.weight"));
            return Err(ArchError::NotSupported {
                detail: if split {
                    format!(
                        "layer {i} ships the split MLA tensors attn_k_b/attn_v_b \
                         (llama.cpp's `is_mla()` absorption path); this loader implements \
                         the unsplit attn_kv_b form only"
                    )
                } else {
                    format!("layer {i} has neither attn_kv_b nor attn_k_b/attn_v_b")
                },
            });
        }
        let w_kv_b = load_quant_linear(model, &kv_b_name)?;
        let w_o = load_quant_linear(model, &format!("{p}.attn_output.weight"))?;

        let mla_weights = DeepSeekMlaWeights {
            q,
            w_kv_a,
            kv_a_norm,
            w_kv_b,
            w_o,
            rope: RopeTable::new_with_params(
                qk_rope,
                max_seq,
                config.rope_freq_base,
                &rope_params,
            )?,
        };

        let ffn_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{p}.ffn_norm.weight"))?,
            eps,
        );

        let ffn = if i < ds_config.first_k_dense_replace {
            FfnKind::Dense(Box::new(DenseFfn {
                gate: load_quant_linear(model, &format!("{p}.ffn_gate.weight"))?,
                up: load_quant_linear(model, &format!("{p}.ffn_up.weight"))?,
                down: load_quant_linear(model, &format!("{p}.ffn_down.weight"))?,
            }))
        } else {
            FfnKind::Moe(Box::new(load_moe(model, &p, &ds_config, &routing)?))
        };

        layers.push(DeepSeekLayer {
            attn_norm,
            mla_weights,
            mla_config: mla_config.clone(),
            mla_cache: MlaLatentCache::new(max_seq, &mla_config),
            ffn_norm,
            ffn,
        });
    }

    let output_norm = RmsNorm::new(
        load_dequant_tensor(model, "output_norm.weight")?,
        config.rms_norm_eps,
    );
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    DeepSeekModel::new(config, ds_config, token_embd, layers, output_norm, output)
}

/// Build one MoE layer from the stacked GGUF expert tensors.
fn load_moe(
    model: &GgufModel,
    prefix: &str,
    ds: &DeepSeekConfig,
    routing: &DeepSeekRouting,
) -> ArchResult<RoutedQuantMoe> {
    let n_exp = ds.n_routed_experts;
    if n_exp == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "deepseek2: n_expert must be > 0".to_string(),
        });
    }
    if model
        .file
        .tensors
        .contains(&format!("{prefix}.ffn_gate_up_exps.weight"))
        && !model
            .file
            .tensors
            .contains(&format!("{prefix}.ffn_gate_exps.weight"))
    {
        return Err(ArchError::NotSupported {
            detail: format!(
                "{prefix} ships the fused `ffn_gate_up_exps` tensor; this loader implements \
                 the split `ffn_gate_exps` / `ffn_up_exps` form only"
            ),
        });
    }

    let router = load_quant_linear(model, &format!("{prefix}.ffn_gate_inp.weight"))?;
    let gate = load_stacked_experts(model, &format!("{prefix}.ffn_gate_exps.weight"), n_exp)?;
    let up = load_stacked_experts(model, &format!("{prefix}.ffn_up_exps.weight"), n_exp)?;
    let down = load_stacked_experts(model, &format!("{prefix}.ffn_down_exps.weight"), n_exp)?;
    let mut experts = Vec::with_capacity(n_exp);
    for ((g, u), d) in gate.into_iter().zip(up).zip(down) {
        experts.push(QuantExpert::new(
            QuantLinear::new(g, None),
            QuantLinear::new(u, None),
            QuantLinear::new(d, None),
        )?);
    }

    // ONE shared expert of width `n_ff_exp * n_expert_shared`, not
    // `n_expert_shared` separate experts.
    let shared = if ds.n_shared_experts > 0 {
        Some(QuantExpert::new(
            load_quant_linear(model, &format!("{prefix}.ffn_gate_shexp.weight"))?,
            load_quant_linear(model, &format!("{prefix}.ffn_up_shexp.weight"))?,
            load_quant_linear(model, &format!("{prefix}.ffn_down_shexp.weight"))?,
        )?)
    } else {
        None
    };

    // `tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i)` → `blk.%d.exp_probs_b.bias`.
    // The `.weight` spelling is accepted as a fallback for the in-repo fixture.
    let bias_bias = format!("{prefix}.exp_probs_b.bias");
    let bias_weight = format!("{prefix}.exp_probs_b.weight");
    let exp_probs_b = if model.file.tensors.contains(&bias_bias) {
        Some(load_dequant_tensor(model, &bias_bias)?)
    } else if model.file.tensors.contains(&bias_weight) {
        Some(load_dequant_tensor(model, &bias_weight)?)
    } else {
        None
    };

    RoutedQuantMoe::new(
        router,
        experts,
        shared,
        exp_probs_b,
        RoutedMoeConfig {
            top_k: ds.top_k_routed,
            activation: ExpertActivation::Silu,
            gating: routing.gating,
            norm_weights: routing.weights_norm,
            weight_scale: routing.weights_scale,
            n_group: routing.n_group,
            topk_group: routing.topk_group,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::MetadataValue;

    /// D2: `qk_nope_head_dim = key_length − rope_dimension_count`.
    ///
    /// A real DeepSeek-V2 GGUF carries `key_length = 192` (= 128 + 64) and
    /// `rope.dimension_count = 64`.  `config.rs` maps `qk_nope_head_dim` straight
    /// from `key_length`, producing 192, and reads a
    /// `deepseek2.attention.rope_head_dim` key that `gguf-py` never writes.
    #[test]
    fn head_dims_are_derived_from_key_length_minus_rope_dims() {
        let mut store = MetadataStore::new();
        store.insert(
            "deepseek2.attention.key_length".to_string(),
            MetadataValue::Uint32(192),
        );
        store.insert(
            "deepseek2.rope.dimension_count".to_string(),
            MetadataValue::Uint32(64),
        );
        let fallback = DeepSeekConfig::from_metadata(&store, 5120);
        assert_eq!(
            fallback.qk_nope_head_dim, 192,
            "precondition: config.rs reads key_length directly"
        );

        let (nope, rope) = resolve_head_dims(&store, &fallback).expect("derivation");
        assert_eq!(nope, 128, "qk_nope must be 192 - 64");
        assert_eq!(rope, 64);
    }

    #[test]
    fn contradictory_head_dims_are_rejected() {
        let mut store = MetadataStore::new();
        store.insert(
            "deepseek2.attention.key_length".to_string(),
            MetadataValue::Uint32(64),
        );
        store.insert(
            "deepseek2.rope.dimension_count".to_string(),
            MetadataValue::Uint32(64),
        );
        let fallback = DeepSeekConfig::from_metadata(&store, 5120);
        assert!(resolve_head_dims(&store, &fallback).is_err());
    }

    /// The five routing keys llama.cpp reads and this crate never did.
    #[test]
    fn routing_keys_are_parsed() {
        let mut store = MetadataStore::new();
        store.insert(
            "deepseek2.expert_group_count".to_string(),
            MetadataValue::Uint32(8),
        );
        store.insert(
            "deepseek2.expert_group_used_count".to_string(),
            MetadataValue::Uint32(4),
        );
        store.insert(
            "deepseek2.expert_weights_norm".to_string(),
            MetadataValue::Bool(true),
        );
        store.insert(
            "deepseek2.expert_gating_func".to_string(),
            MetadataValue::Uint32(2),
        );
        store.insert(
            "deepseek2.expert_weights_scale".to_string(),
            MetadataValue::Float32(2.5),
        );
        store.insert(
            "deepseek2.expert_feed_forward_length".to_string(),
            MetadataValue::Uint32(2048),
        );
        let r = DeepSeekRouting::from_metadata(&store, 64).expect("routing parses");
        assert_eq!(r.n_group, 8);
        assert_eq!(r.topk_group, 4);
        assert!(r.weights_norm);
        assert_eq!(r.gating, GatingFunc::Sigmoid);
        assert!((r.weights_scale - 2.5).abs() < 1e-6);
        assert_eq!(r.n_ff_exp, 2048);
    }

    /// DeepSeek-V2 ships `norm_topk_prob = false`; the default must not
    /// re-normalise.
    #[test]
    fn weights_norm_defaults_to_false() {
        let r = DeepSeekRouting::from_metadata(&MetadataStore::new(), 64).expect("routing");
        assert!(
            !r.weights_norm,
            "expert_weights_norm defaults to false (DeepSeek-V2 norm_topk_prob = false)"
        );
        assert_eq!(r.n_group, 1, "no group routing without expert_group_count");
        assert_eq!(r.gating, GatingFunc::Softmax);
    }
}
