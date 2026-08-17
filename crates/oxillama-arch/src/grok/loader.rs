//! GGUF loader for Grok-1.
//!
//! Tensor set, from the `case LLM_ARCH_GROK:` block of
//! `llama_model::load_tensors`:
//!
//! ```text
//! token_embd.weight                    [vocab, hidden]
//! blk.{i}.attn_norm.weight             [hidden]
//! blk.{i}.attn_{q,k,v,output}.weight
//! blk.{i}.attn_output_norm.weight      [hidden]   post-attention, pre-residual
//! blk.{i}.ffn_norm.weight              [hidden]
//! blk.{i}.ffn_gate_inp.weight          [n_expert, hidden]
//! blk.{i}.ffn_{gate,up,down}_exps.weight          stacked expert pool
//! blk.{i}.ffn_{gate,up,down}.weight    OPTIONAL parallel dense FFN
//! blk.{i}.layer_output_norm.weight     [hidden]   post-FFN, pre-residual
//!   (falls back to blk.{i}.post_ffw_norm.weight)
//! output_norm.weight                   [hidden]
//! output.weight                        [vocab, hidden]   (tied fallback allowed)
//! ```
//!
//! `attn_output_norm` and `layer_output_norm` had no fields in `GrokLayer` and
//! were never loaded, so two of Grok's four per-layer normalisations were simply
//! absent.  Expert weights now stay quantized as shared mmap views; the previous
//! loader dequantized the stacked tensors and then `.to_vec()`-copied each
//! expert out of them.

use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    dequant_to_f32, load_dequant_tensor, load_lm_head, load_quant_linear, load_quant_linear_opt,
    load_stacked_experts,
};
use crate::common::moe::QuantExpert;
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::grok::config::GrokConfig;
use crate::grok::model::{GrokDenseFfn, GrokLayer, GrokModel};
use crate::grok::moe::GrokMoe;

/// Load a Grok-1 model from a parsed GGUF file.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] naming the first absent required tensor.
/// * [`ArchError::InvalidConfig`] when the checkpoint declares zero experts.
/// * [`ArchError::InvalidShape`] on any shape disagreement.
pub fn load_grok_from_gguf(model: &GgufModel) -> ArchResult<GrokModel> {
    let metadata = &model.file.metadata;
    let dispatcher = KernelDispatcher::new();

    let model_config = ModelConfig::from_metadata(metadata)?;
    let grok_config = GrokConfig::from_metadata(metadata);

    let n_layers = grok_config.num_layers;
    let n_experts = grok_config.expert_count;
    let top_k = grok_config.expert_used_count;
    let eps = grok_config.rms_norm_eps;

    if n_experts == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "Grok model cannot have zero experts".to_string(),
        });
    }

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

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{prefix}.attn_norm.weight"))?,
            eps,
        );
        let attn_out_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{prefix}.attn_output_norm.weight"))?,
            eps,
        );
        let ffn_norm = RmsNorm::new(
            load_dequant_tensor(model, &format!("{prefix}.ffn_norm.weight"))?,
            eps,
        );
        // `LLM_TENSOR_LAYER_OUT_NORM` first, `LLM_TENSOR_FFN_POST_NORM` second —
        // the same order llama.cpp tries.
        let post_norm_weights =
            match load_dequant_tensor(model, &format!("{prefix}.layer_output_norm.weight")) {
                Ok(w) => w,
                Err(_) => load_dequant_tensor(model, &format!("{prefix}.post_ffw_norm.weight"))?,
            };
        let ffn_post_norm = RmsNorm::new(post_norm_weights, eps);

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let router = load_quant_linear(model, &format!("{prefix}.ffn_gate_inp.weight"))?;
        let gate =
            load_stacked_experts(model, &format!("{prefix}.ffn_gate_exps.weight"), n_experts)?;
        let up = load_stacked_experts(model, &format!("{prefix}.ffn_up_exps.weight"), n_experts)?;
        let down =
            load_stacked_experts(model, &format!("{prefix}.ffn_down_exps.weight"), n_experts)?;
        let mut experts = Vec::with_capacity(n_experts);
        for ((g, u), d) in gate.into_iter().zip(up).zip(down) {
            experts.push(QuantExpert::new(
                QuantLinear::new(g, None),
                QuantLinear::new(u, None),
                QuantLinear::new(d, None),
            )?);
        }
        let moe = GrokMoe::new(router, experts, top_k)?;

        // Optional dense FFN run in parallel with the MoE branch.  llama.cpp
        // keys the branch on `ffn_up` alone; all three must be present for the
        // computation to be well-defined, so a partial set is an error rather
        // than a silent skip.
        let dense_up = load_quant_linear_opt(model, &format!("{prefix}.ffn_up.weight"))?;
        let dense_ffn = match dense_up {
            None => None,
            Some(up) => {
                let gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
                let down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;
                Some(GrokDenseFfn {
                    gate_kernel: dispatcher.get_kernel(gate.weight.tensor_type)?,
                    up_kernel: dispatcher.get_kernel(up.weight.tensor_type)?,
                    down_kernel: dispatcher.get_kernel(down.weight.tensor_type)?,
                    gate,
                    up,
                    down,
                })
            }
        };

        let attn_q_kernel = dispatcher.get_kernel(attn_q.weight.tensor_type)?;
        let attn_k_kernel = dispatcher.get_kernel(attn_k.weight.tensor_type)?;
        let attn_v_kernel = dispatcher.get_kernel(attn_v.weight.tensor_type)?;
        let attn_output_kernel = dispatcher.get_kernel(attn_output.weight.tensor_type)?;

        layers.push(GrokLayer {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_out_norm,
            ffn_norm,
            ffn_post_norm,
            moe,
            dense_ffn,
            attn_q_kernel,
            attn_k_kernel,
            attn_v_kernel,
            attn_output_kernel,
        });
    }

    let output_norm = RmsNorm::new(load_dequant_tensor(model, "output_norm.weight")?, eps);
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    GrokModel::new(
        model_config,
        grok_config,
        token_embd,
        layers,
        output_norm,
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grok::testkit::TestKvCache;
    use crate::traits::ForwardPass;

    fn fixture() -> GgufModel {
        let bytes = oxillama_gguf::test_utils::build_minimal_grok_gguf();
        GgufModel::from_bytes(bytes).expect("GGUF parse must succeed")
    }

    #[test]
    fn grok_loader_round_trip() {
        let gguf = fixture();
        let model = load_grok_from_gguf(&gguf).expect("load_grok_from_gguf must succeed");
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.grok_config.expert_count, 2);
    }

    /// `attn_output_norm` and `layer_output_norm` are required by llama.cpp and
    /// were previously neither in the struct nor in the fixture.
    #[test]
    fn fixture_carries_both_extra_norms() {
        let gguf = fixture();
        assert!(gguf.file.tensors.contains("blk.0.attn_output_norm.weight"));
        assert!(gguf.file.tensors.contains("blk.0.layer_output_norm.weight"));
    }

    #[test]
    fn grok_loader_forward_no_nan() {
        let gguf = fixture();
        let mut model = load_grok_from_gguf(&gguf).expect("load");
        let mut kv = TestKvCache::new(
            model.layers.len(),
            model.config.num_kv_heads * model.config.head_dim,
            model.config.max_context_length,
        );
        let logits = model.forward(&[0u32], &mut kv).expect("forward");
        assert!(logits.iter().all(|v| !v.is_nan()));
    }
}
