//! GGUF loader for the StableLM architecture.
//!
//! # Primary source
//!
//! Every tensor name, requirement flag and shape below is transcribed from
//! `~/work/refs/llama.cpp/src/llama-model.cpp`, `case LLM_ARCH_STABLELM:` in
//! `load_tensors()`:
//!
//! ```text
//! tok_embd        = create_tensor(TOKEN_EMBD  "weight", {n_embd, n_vocab}, 0);
//! output_norm_b   = create_tensor(OUTPUT_NORM "bias",   {n_embd}, 0);   // required
//! output_norm     = create_tensor(OUTPUT_NORM "weight", {n_embd}, 0);   // required
//! output          = create_tensor(OUTPUT      "weight", {n_embd, n_vocab}, 0);
//! // per layer
//! attn_norm       = create_tensor(ATTN_NORM "weight", i, {n_embd}, 0);  // required
//! attn_norm_b     = create_tensor(ATTN_NORM "bias",   i, {n_embd}, 0);  // required
//! wq = {n_embd, n_embd}; wk = {n_embd, n_embd_gqa}; wv = {n_embd, n_embd_gqa};
//! wo = {n_embd, n_embd};
//! bq / bk / bv                    -> TENSOR_NOT_REQUIRED  (Stable LM 2 1.6B)
//! attn_q_norm {n_embd_head_k, n_head}    -> TENSOR_NOT_REQUIRED (StableLM 2 12B)
//! attn_k_norm {n_embd_head_k, n_head_kv} -> TENSOR_NOT_REQUIRED (StableLM 2 12B)
//! ffn_norm / ffn_norm_b           -> TENSOR_NOT_REQUIRED  (absent in 12B)
//! ffn_gate {n_embd, n_ff}; ffn_down {n_ff, n_embd}; ffn_up {n_embd, n_ff};
//! ```
//!
//! GGUF writes `ne` fastest-changing-first, so `{n_embd, n_embd_gqa}` is a
//! `[n_embd_gqa, n_embd]` math-order matrix — which is what
//! [`gguf_linear_shape`](crate::common::linear::gguf_linear_shape), and
//! therefore [`QuantLinear`], already produces.
//!
//! # LM head
//!
//! llama.cpp marks StableLM's `output.weight` **required** (flags = 0) with no
//! tied-embedding fallback.  This loader nevertheless goes through
//! [`load_lm_head`], which falls back to `token_embd.weight` when
//! `output.weight` is absent.  That is a strict superset: every checkpoint
//! llama.cpp accepts loads identically, and a hypothetical tied StableLM
//! export loads instead of failing.  The error message still names
//! `output.weight` first.

use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::loader::{
    load_bias, load_dequant_tensor, load_lm_head, load_quant_linear, load_quant_linear_with_bias,
};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};

use super::config::StablelmConfig;
use super::head_norm::PerHeadLayerNorm;
use super::model::{StablelmLayer, StablelmLayerWeights, StablelmModel};

/// Load a StableLM model from a parsed GGUF file.
///
/// Handles all four checkpoint shapes llama.cpp supports:
///
/// | Variant | `bq/bk/bv` | `attn_q/k_norm` | `ffn_norm` | Residual |
/// |---------|-----------|------------------|------------|----------|
/// | StableLM 1 (3B/7B) | – | – | ✓ | sequential |
/// | Stable LM 2 1.6B   | ✓ | – | ✓ | sequential |
/// | StableLM 2 12B     | – | ✓ | – | **parallel** |
/// | Stable Code / Zephyr | – | – | ✓ | sequential |
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] for any tensor llama.cpp marks required.
/// * [`ArchError::InvalidShape`] when a norm vector, the embedding table, or a
///   projection does not match the shape the configuration implies.
/// * [`ArchError::ConfigMismatch`] for the geometry checks in
///   [`StablelmModel::new`] (`num_heads * head_dim == hidden_size`, GQA
///   divisibility).
pub fn load_stablelm_from_gguf(
    model: &GgufModel,
    config: &ModelConfig,
) -> ArchResult<StablelmModel> {
    let dispatcher = KernelDispatcher::new();
    let stablelm_config = StablelmConfig::from_metadata(&model.file.metadata, config);
    let eps = stablelm_config.layer_norm_eps;

    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let intermediate_size = config.intermediate_size;
    let q_dim = num_heads.saturating_mul(head_dim);
    let kv_dim = num_kv_heads.saturating_mul(head_dim);

    // `token_embd.weight` is dequantized in bulk, matching every other
    // LayerNorm-family architecture in this crate (phi, starcoder, bloom).
    // The length check below is what makes `embed_token`'s row slice safe.
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        // ── Norms ─────────────────────────────────────────────────────────
        // `attn_norm` weight *and* bias are both flags = 0 in the reference.
        let attn_norm = load_required_layer_norm(
            model,
            &format!("{prefix}.attn_norm.weight"),
            &format!("{prefix}.attn_norm.bias"),
            hidden_size,
            eps,
        )?;

        // `ffn_norm` weight and bias are independently TENSOR_NOT_REQUIRED.
        // Its **weight** decides the residual topology; a weight without a
        // bias yields a biasless LayerNorm rather than an error.
        let ffn_norm = load_optional_layer_norm(
            model,
            &format!("{prefix}.ffn_norm.weight"),
            &format!("{prefix}.ffn_norm.bias"),
            hidden_size,
            eps,
        )?;

        // ── Attention projections ─────────────────────────────────────────
        let attn_q = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_q.bias"),
        )?;
        let attn_k = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_k.weight"),
            &format!("{prefix}.attn_k.bias"),
        )?;
        let attn_v = load_quant_linear_with_bias(
            model,
            &format!("{prefix}.attn_v.weight"),
            &format!("{prefix}.attn_v.bias"),
        )?;
        // `wo` has no bias tensor in the reference at all.
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        check_linear(
            &format!("{prefix}.attn_q.weight"),
            &attn_q,
            q_dim,
            hidden_size,
        )?;
        check_linear(
            &format!("{prefix}.attn_k.weight"),
            &attn_k,
            kv_dim,
            hidden_size,
        )?;
        check_linear(
            &format!("{prefix}.attn_v.weight"),
            &attn_v,
            kv_dim,
            hidden_size,
        )?;
        check_linear(
            &format!("{prefix}.attn_output.weight"),
            &attn_output,
            hidden_size,
            q_dim,
        )?;

        // ── Optional per-head QK LayerNorm ────────────────────────────────
        // The Q norm covers `n_head` heads and the K norm `n_head_kv` — the
        // two counts differ under GQA, and the tensors are genuinely
        // per-head (`{n_embd_head_k, n_head}`), not one shared vector.
        let attn_q_norm = load_optional_head_norm(
            model,
            &format!("{prefix}.attn_q_norm.weight"),
            head_dim,
            num_heads,
            eps,
        )?;
        let attn_k_norm = load_optional_head_norm(
            model,
            &format!("{prefix}.attn_k_norm.weight"),
            head_dim,
            num_kv_heads,
            eps,
        )?;

        // ── SwiGLU FFN ────────────────────────────────────────────────────
        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        check_linear(
            &format!("{prefix}.ffn_gate.weight"),
            &ffn_gate,
            intermediate_size,
            hidden_size,
        )?;
        check_linear(
            &format!("{prefix}.ffn_up.weight"),
            &ffn_up,
            intermediate_size,
            hidden_size,
        )?;
        check_linear(
            &format!("{prefix}.ffn_down.weight"),
            &ffn_down,
            hidden_size,
            intermediate_size,
        )?;

        layers.push(StablelmLayer::new(
            &dispatcher,
            StablelmLayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            },
        )?);
    }

    let output_norm = load_required_layer_norm(
        model,
        "output_norm.weight",
        "output_norm.bias",
        hidden_size,
        eps,
    )?;
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    StablelmModel::new(
        config.clone(),
        stablelm_config,
        token_embd,
        layers,
        output_norm,
        output,
    )
}

/// Load a LayerNorm whose weight *and* bias llama.cpp both mark required.
fn load_required_layer_norm(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
    hidden_size: usize,
    eps: f32,
) -> ArchResult<LayerNorm> {
    let weight = load_dequant_tensor(model, weight_name)?;
    check_vector(weight_name, weight.len(), hidden_size)?;

    let bias = load_dequant_tensor(model, bias_name)?;
    check_vector(bias_name, bias.len(), hidden_size)?;

    Ok(LayerNorm::new(weight, Some(bias), eps))
}

/// Load a LayerNorm that may be absent entirely, with an independently
/// optional bias.
///
/// Returns `Ok(None)` when the **weight** is absent — the discriminator
/// llama.cpp uses to select StableLM's parallel-residual topology.
fn load_optional_layer_norm(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
    hidden_size: usize,
    eps: f32,
) -> ArchResult<Option<LayerNorm>> {
    if !model.file.tensors.contains(weight_name) {
        return Ok(None);
    }
    let weight = load_dequant_tensor(model, weight_name)?;
    check_vector(weight_name, weight.len(), hidden_size)?;

    let bias = load_bias(model, bias_name)?;
    if let Some(b) = bias.as_ref() {
        check_vector(bias_name, b.len(), hidden_size)?;
    }

    Ok(Some(LayerNorm::new(weight, bias, eps)))
}

/// Load an optional per-head QK LayerNorm weight.
fn load_optional_head_norm(
    model: &GgufModel,
    name: &str,
    head_dim: usize,
    num_heads: usize,
    eps: f32,
) -> ArchResult<Option<PerHeadLayerNorm>> {
    if !model.file.tensors.contains(name) {
        return Ok(None);
    }
    let weight = load_dequant_tensor(model, name)?;
    PerHeadLayerNorm::new(name, weight, head_dim, num_heads, eps).map(Some)
}

/// Validate a 1-D norm/bias vector's length.
fn check_vector(name: &str, got: usize, expected: usize) -> ArchResult<()> {
    if got == expected {
        return Ok(());
    }
    Err(ArchError::InvalidShape {
        name: name.to_string(),
        expected: vec![expected],
        got: vec![got],
    })
}

/// Validate a projection's `[out_features, in_features]` geometry.
///
/// The GEMV kernels take these from the tensor itself while every scratch
/// buffer is sized from [`ModelConfig`]; a disagreement would otherwise write
/// a truncated result and leave the rest of the buffer stale.
fn check_linear(
    name: &str,
    linear: &QuantLinear,
    out_features: usize,
    in_features: usize,
) -> ArchResult<()> {
    if linear.out_features == out_features && linear.in_features == in_features {
        return Ok(());
    }
    Err(ArchError::InvalidShape {
        name: name.to_string(),
        expected: vec![out_features, in_features],
        got: vec![linear.out_features, linear.in_features],
    })
}
