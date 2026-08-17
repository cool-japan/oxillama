//! GGUF loader for DBRX.
//!
//! Tensor set, verified against the `LLM_ARCH_DBRX` entry in
//! `src/llama-arch.cpp` and the `case LLM_ARCH_DBRX:` tensor-creation block in
//! `src/llama-model.cpp`:
//!
//! ```text
//! token_embd.weight                 [vocab, hidden]
//! blk.{i}.attn_norm.weight          [hidden]                     LayerNorm
//! blk.{i}.attn_qkv.weight           [hidden + 2*kv_dim, hidden]  FUSED
//! blk.{i}.attn_output.weight        [hidden, hidden]
//! blk.{i}.attn_output_norm.weight   [hidden]                     LayerNorm, pre-FFN
//! blk.{i}.ffn_gate_inp.weight       [n_expert, hidden]           router
//! blk.{i}.ffn_gate_exps.weight      [n_expert, n_ff, hidden]     stacked
//! blk.{i}.ffn_up_exps.weight        [n_expert, n_ff, hidden]     stacked
//! blk.{i}.ffn_down_exps.weight      [n_expert, hidden, n_ff]     stacked
//! output_norm.weight                [hidden]                     LayerNorm
//! output.weight                     [vocab, hidden]
//! ```
//!
//! There is **no** `blk.{i}.ffn_norm.weight` and **no** separate
//! `attn_q`/`attn_k`/`attn_v`: the previous loader required both and would have
//! rejected every real DBRX checkpoint.
//!
//! All expert weights stay quantized as shared mmap views
//! ([`load_stacked_experts`]).  The previous loader dequantized each stacked
//! tensor to `f32` and then `.to_vec()`-copied every expert slice out of it —
//! twice the peak, and >500 GB for the real 132 B model whose experts hold 94 %
//! of the parameters.

use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::loader::{
    dequant_to_f32, load_dequant_tensor, load_lm_head, load_quant_linear, load_stacked_experts,
};
use crate::common::moe::{QuantExpert, QuantMoeFfn};
use crate::config::ModelConfig;
use crate::dbrx::config::DbrxConfig;
use crate::dbrx::model::{DbrxLayer, DbrxModel};
use crate::error::{ArchError, ArchResult};

/// Load a DBRX model from a parsed GGUF file.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] naming the first absent tensor.
/// * [`ArchError::InvalidShape`] when a tensor's declared shape disagrees with
///   the config, or when the payload is shorter than the declared shape.
/// * [`ArchError::InvalidConfig`] when `expert_used_count == 0`.
pub fn load_dbrx_from_gguf(model: &GgufModel) -> ArchResult<DbrxModel> {
    let metadata = &model.file.metadata;
    let dispatcher = KernelDispatcher::new();

    let model_config = ModelConfig::from_metadata(metadata)?;
    let dbrx_config = DbrxConfig::from_metadata(metadata);

    let hidden = dbrx_config.hidden_size;
    let n_layers = dbrx_config.num_layers;
    let n_experts = dbrx_config.expert_count;
    let top_k = dbrx_config.expert_used_count;
    let eps = dbrx_config.layer_norm_eps;

    if n_experts == 0 {
        // llama.cpp: `throw std::runtime_error("DBRX model cannot have zero experts")`.
        return Err(ArchError::InvalidConfig {
            detail: "DBRX model cannot have zero experts".to_string(),
        });
    }

    // ── Token embedding ────────────────────────────────────────────────────────
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

    // ── Transformer layers ─────────────────────────────────────────────────────
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = LayerNorm::new(
            load_dequant_tensor(model, &format!("{prefix}.attn_norm.weight"))?,
            None,
            eps,
        );
        // `LLM_TENSOR_ATTN_OUT_NORM` → `blk.%d.attn_output_norm`.  It sits in
        // the pre-FFN slot; DBRX has no `ffn_norm`.
        let attn_out_norm = LayerNorm::new(
            load_dequant_tensor(model, &format!("{prefix}.attn_output_norm.weight"))?,
            None,
            eps,
        );

        let attn_qkv = load_quant_linear(model, &format!("{prefix}.attn_qkv.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let kv_dim = model_config.num_kv_heads * model_config.head_dim;
        let q_dim = model_config.num_attention_heads * model_config.head_dim;
        let expected_qkv = q_dim + 2 * kv_dim;
        if attn_qkv.out_features != expected_qkv || attn_qkv.in_features != hidden {
            return Err(ArchError::InvalidShape {
                name: format!("{prefix}.attn_qkv.weight"),
                expected: vec![expected_qkv, hidden],
                got: vec![attn_qkv.out_features, attn_qkv.in_features],
            });
        }

        let moe = load_dbrx_moe(model, &prefix, n_experts, top_k)?;

        let dispatcher_ref = &dispatcher;
        let attn_qkv_kernel = dispatcher_ref.get_kernel(attn_qkv.weight.tensor_type)?;
        let attn_output_kernel = dispatcher_ref.get_kernel(attn_output.weight.tensor_type)?;

        layers.push(DbrxLayer {
            attn_norm,
            attn_qkv,
            attn_output,
            attn_out_norm,
            moe,
            attn_qkv_kernel,
            attn_output_kernel,
        });
    }

    let output_norm = LayerNorm::new(load_dequant_tensor(model, "output_norm.weight")?, None, eps);
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    DbrxModel::new(
        model_config,
        dbrx_config,
        token_embd,
        layers,
        output_norm,
        output,
    )
}

/// Build one block's [`QuantMoeFfn`] from the stacked GGUF expert tensors.
fn load_dbrx_moe(
    model: &GgufModel,
    prefix: &str,
    n_experts: usize,
    top_k: usize,
) -> ArchResult<QuantMoeFfn> {
    let router = load_quant_linear(model, &format!("{prefix}.ffn_gate_inp.weight"))?;
    let gate = load_stacked_experts(model, &format!("{prefix}.ffn_gate_exps.weight"), n_experts)?;
    let up = load_stacked_experts(model, &format!("{prefix}.ffn_up_exps.weight"), n_experts)?;
    let down = load_stacked_experts(model, &format!("{prefix}.ffn_down_exps.weight"), n_experts)?;

    let mut experts = Vec::with_capacity(n_experts);
    for ((g, u), d) in gate.into_iter().zip(up).zip(down) {
        experts.push(QuantExpert::new(
            QuantLinear::new(g, None),
            QuantLinear::new(u, None),
            QuantLinear::new(d, None),
        )?);
    }
    QuantMoeFfn::new(router, experts, top_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbrx::testkit::TestKvCache;
    use crate::traits::ForwardPass;

    fn fixture() -> GgufModel {
        let bytes = oxillama_gguf::test_utils::build_minimal_dbrx_gguf();
        GgufModel::from_bytes(bytes).expect("GGUF parse must succeed")
    }

    #[test]
    fn dbrx_loader_round_trip() {
        let gguf = fixture();
        let model = load_dbrx_from_gguf(&gguf).expect("load_dbrx_from_gguf must succeed");
        assert_eq!(model.config.vocab_size, 32);
        assert_eq!(model.config.hidden_size, 32);
        assert_eq!(model.config.max_context_length, 128);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.dbrx_config.expert_count, 4);
        assert_eq!(model.dbrx_config.expert_used_count, 2);
    }

    /// The fixture carries `dbrx.attention.clamp_kqv = 8.0`; the loader must
    /// surface it so `ggml_clamp` has something to clamp with.
    #[test]
    fn loader_picks_up_clamp_kqv() {
        let gguf = fixture();
        let model = load_dbrx_from_gguf(&gguf).expect("load");
        assert!(
            (model.dbrx_config.clamp_kqv - 8.0).abs() < 1e-6,
            "clamp_kqv = {}",
            model.dbrx_config.clamp_kqv
        );
    }

    /// The fixture has a fused `attn_qkv` and an `attn_output_norm`, and no
    /// `attn_q`/`attn_k`/`attn_v`/`ffn_norm` — exactly what llama.cpp emits.
    #[test]
    fn fixture_uses_the_llama_cpp_tensor_set() {
        let gguf = fixture();
        assert!(gguf.file.tensors.contains("blk.0.attn_qkv.weight"));
        assert!(gguf.file.tensors.contains("blk.0.attn_output_norm.weight"));
        assert!(
            !gguf.file.tensors.contains("blk.0.attn_q.weight"),
            "DBRX has no separate Q projection"
        );
        assert!(
            !gguf.file.tensors.contains("blk.0.ffn_norm.weight"),
            "DBRX has no ffn_norm"
        );
    }

    #[test]
    fn dbrx_loader_forward_no_nan() {
        let gguf = fixture();
        let mut model = load_dbrx_from_gguf(&gguf).expect("load");
        let mut kv = TestKvCache::new(
            model.layers.len(),
            model.config.num_kv_heads * model.config.head_dim,
            model.config.max_context_length,
        );
        let logits = model.forward(&[0u32], &mut kv).expect("forward after load");
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| !v.is_nan()));
    }
}
