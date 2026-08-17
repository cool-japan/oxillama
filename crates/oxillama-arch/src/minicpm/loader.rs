//! GGUF loading for MiniCPM (base).
//!
//! # Tensor set
//!
//! MiniCPM shares LLaMA's tensor-creation block verbatim — llama.cpp's
//! `load_tensors()` (`src/llama-model.cpp`, ~line 2995) lists
//! `LLM_ARCH_MINICPM` alongside `LLM_ARCH_LLAMA` and `LLM_ARCH_GRANITE` in the
//! *same* `case` — so there are **no MiniCPM-specific tensors at all**:
//!
//! | GGUF name | Required | Reference |
//! |-----------|----------|-----------|
//! | `token_embd.weight` | yes | `create_tensor(TOKEN_EMBD, {n_embd, n_vocab}, 0)` |
//! | `output_norm.weight` | yes | `create_tensor(OUTPUT_NORM, {n_embd}, 0)` |
//! | `output.weight` | no — tied fallback to `token_embd.weight` | `TENSOR_NOT_REQUIRED` then `TENSOR_DUPLICATED` |
//! | `blk.{i}.attn_norm.weight` | yes | `{n_embd}` |
//! | `blk.{i}.attn_q.weight` | yes | `{n_embd, n_embd_head_k * n_head}` |
//! | `blk.{i}.attn_k.weight` | yes | `{n_embd, n_embd_k_gqa}` |
//! | `blk.{i}.attn_v.weight` | yes | `{n_embd, n_embd_v_gqa}` |
//! | `blk.{i}.attn_output.weight` | yes | `{n_embd_head_k * n_head, n_embd}` |
//! | `blk.{i}.attn_{q,k,v,output}.bias` | no | `TENSOR_NOT_REQUIRED` |
//! | `blk.{i}.ffn_norm.weight` | yes | `{n_embd}` |
//! | `blk.{i}.ffn_gate.weight` | yes | `{n_embd, n_ff}` |
//! | `blk.{i}.ffn_up.weight` | yes | `{n_embd, n_ff}` |
//! | `blk.{i}.ffn_down.weight` | yes | `{n_ff, n_embd}` |
//! | `blk.{i}.ffn_{gate,up,down}.bias` | no | `TENSOR_NOT_REQUIRED` |
//!
//! Real MiniCPM checkpoints ship none of the optional biases, but they are
//! honoured when present because llama.cpp honours them.
//!
//! # Hyper-parameters
//!
//! The three MiniCPM scale factors are read straight out of the GGUF metadata
//! by [`MiniCpmConfig::from_metadata`] — see that type's documentation for the
//! `convert_hf_to_gguf.py` / `granite.cpp` citations.

use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    load_dequant_tensor, load_lm_head, load_quant_linear_with_bias, load_rms_norm_weight,
};
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::minicpm::config::MiniCpmConfig;
use crate::minicpm::forward::{MiniCpmForward, MiniCpmLayer};

/// Load a MiniCPM (base) model from a parsed GGUF file.
///
/// MiniCPM3 is a different architecture (`general.architecture = "minicpm3"`,
/// MLA attention with Q/KV LoRA ranks) and is **not** handled here.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] when a required tensor is absent.
/// * [`ArchError::InvalidShape`] when a tensor's shape contradicts the
///   configuration (which would otherwise make a GEMV read or write out of
///   bounds).
/// * [`ArchError::InvalidConfig`] when a MiniCPM scale factor is non-finite or
///   `logit_scale` is zero.
pub fn load_minicpm_from_gguf(
    model: &GgufModel,
    config: &ModelConfig,
) -> ArchResult<MiniCpmForward> {
    let cfg = MiniCpmConfig::from_metadata(config, &model.file.metadata)?;
    let dispatcher = KernelDispatcher::new();

    let hidden = cfg.hidden_size;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let ffn = cfg.intermediate_size;

    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = RmsNorm::new(
            load_norm(model, &format!("{prefix}.attn_norm.weight"), hidden)?,
            cfg.norm_eps,
        );
        let ffn_norm = RmsNorm::new(
            load_norm(model, &format!("{prefix}.ffn_norm.weight"), hidden)?,
            cfg.norm_eps,
        );

        let attn_q = load_projection(model, &prefix, "attn_q", q_dim, hidden)?;
        let attn_k = load_projection(model, &prefix, "attn_k", kv_dim, hidden)?;
        let attn_v = load_projection(model, &prefix, "attn_v", kv_dim, hidden)?;
        let attn_output = load_projection(model, &prefix, "attn_output", hidden, q_dim)?;

        let ffn_gate = load_projection(model, &prefix, "ffn_gate", ffn, hidden)?;
        let ffn_up = load_projection(model, &prefix, "ffn_up", ffn, hidden)?;
        let ffn_down = load_projection(model, &prefix, "ffn_down", hidden, ffn)?;

        layers.push(MiniCpmLayer::new(
            attn_norm,
            ffn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_gate,
            ffn_up,
            ffn_down,
            &dispatcher,
        )?);
    }

    let output_norm = RmsNorm::new(
        load_norm(model, "output_norm.weight", hidden)?,
        cfg.norm_eps,
    );

    // `output.weight` is `TENSOR_NOT_REQUIRED` for MiniCPM and duplicated from
    // `token_embd.weight` when absent (`src/llama-model.cpp`, the shared
    // LLAMA/MINICPM/GRANITE case).
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;
    check_shape("output.weight", &output, cfg.vocab_size, hidden)?;

    MiniCpmForward::new(cfg, config.clone(), token_embd, layers, output_norm, output)
}

/// Load a norm vector and check its width.
fn load_norm(model: &GgufModel, name: &str, expected: usize) -> ArchResult<Vec<f32>> {
    let weight = load_rms_norm_weight(model, name)?;
    if weight.len() != expected {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![expected],
            got: vec![weight.len()],
        });
    }
    Ok(weight)
}

/// Load `{prefix}.{stem}.weight` with its optional `.bias`, then check the
/// shape against the configuration.
fn load_projection(
    model: &GgufModel,
    prefix: &str,
    stem: &str,
    out_features: usize,
    in_features: usize,
) -> ArchResult<QuantLinear> {
    let weight_name = format!("{prefix}.{stem}.weight");
    let bias_name = format!("{prefix}.{stem}.bias");
    let linear = load_quant_linear_with_bias(model, &weight_name, &bias_name)?;
    check_shape(&weight_name, &linear, out_features, in_features)?;
    Ok(linear)
}

/// Reject a projection whose GGUF shape contradicts the configuration.
///
/// Without this a mismatched checkpoint reaches the GEMV kernels, which index
/// `out_features × in_features` out of buffers sized from the configuration.
fn check_shape(
    name: &str,
    linear: &QuantLinear,
    out_features: usize,
    in_features: usize,
) -> ArchResult<()> {
    if linear.out_features != out_features || linear.in_features != in_features {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![out_features, in_features],
            got: vec![linear.out_features, linear.in_features],
        });
    }
    if let Some(bias) = &linear.bias {
        if bias.len() != out_features {
            return Err(ArchError::InvalidShape {
                name: format!("{name} (bias)"),
                expected: vec![out_features],
                got: vec![bias.len()],
            });
        }
    }
    Ok(())
}
