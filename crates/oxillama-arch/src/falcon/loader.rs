//! GGUF loading for the Falcon architecture.
//!
//! # Reference
//!
//! `~/work/refs/llama.cpp/src/llama-model.cpp`, `case LLM_ARCH_FALCON:` in
//! `load_tensors()` — the exact tensor set this module reads:
//!
//! | GGUF name | llama.cpp flags | Handled by |
//! |-----------|-----------------|------------|
//! | `token_embd.weight` | required | [`load_dequant_tensor`] |
//! | `output_norm.weight` / `.bias` | required | [`load_rms_norm_weight`] / [`load_bias`] |
//! | `output.weight` | `TENSOR_NOT_REQUIRED`, `TENSOR_DUPLICATED` fallback onto `token_embd.weight` | [`load_lm_head`] |
//! | `blk.{i}.attn_norm.weight` / `.bias` | required | [`load_rms_norm_weight`] / [`load_bias`] |
//! | `blk.{i}.attn_norm_2.weight` / `.bias` | `TENSOR_NOT_REQUIRED` (Falcon-40B) | presence-checked, then loaded |
//! | `blk.{i}.attn_qkv.weight` | required, `{n_embd, n_embd + 2*n_embd_gqa}` | [`load_quant_linear_with_bias`] |
//! | `blk.{i}.attn_output.weight` | required, `{n_embd, n_embd}` | [`load_quant_linear_with_bias`] |
//! | `blk.{i}.ffn_up.weight` | required, `{n_embd, n_ff}` | [`load_quant_linear_with_bias`] |
//! | `blk.{i}.ffn_down.weight` | required, `{n_ff, n_embd}` | [`load_quant_linear_with_bias`] |
//!
//! `LLM_TENSOR_NAMES` in `src/llama-arch.cpp` resolves `ATTN_NORM_2` to the
//! literal `"blk.%d.attn_norm_2"`.
//!
//! Falcon creates **no** linear bias tensors — the only biases in the
//! architecture are the LayerNorm shifts.  The `*_with_bias` helpers are used
//! anyway because a missing bias is not an error there, so a third-party
//! conversion that ships one is accepted rather than ignored.
//!
//! Falcon also has **no `ffn_norm`**: `llm_build_falcon` feeds the FFN with
//! `attn_norm`'s output.  The optional `blk.{i}.ffn_norm.*` tensors are still
//! read when present so that a hand-built sequential checkpoint round-trips,
//! but no real Falcon GGUF contains them.

use oxillama_gguf::GgufModel;

use crate::common::layer_norm::LayerNorm;
use crate::common::loader::{
    load_bias, load_dequant_tensor, load_lm_head, load_quant_linear_with_bias, load_rms_norm_weight,
};
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::falcon::config::FalconConfig;
use crate::falcon::forward::{FalconForward, FalconLayer};

/// Load a LayerNorm from GGUF.
///
/// `bias_required` mirrors llama.cpp's `create_tensor` flags: `attn_norm` and
/// `output_norm` pass their bias with flags `0` (required), while
/// `attn_norm_2` passes both weight and bias as `TENSOR_NOT_REQUIRED`.
///
/// A required bias must not degrade to "no shift term": dropping it silently
/// computes a different model, which is the same failure mode as substituting
/// a norm for a missing one.
fn load_layer_norm(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
    hidden_size: usize,
    eps: f32,
    bias_required: bool,
) -> ArchResult<LayerNorm> {
    let weight = load_rms_norm_weight(model, weight_name)?;
    if weight.len() != hidden_size {
        return Err(ArchError::InvalidShape {
            name: weight_name.to_string(),
            expected: vec![hidden_size],
            got: vec![weight.len()],
        });
    }
    if bias_required && !model.file.tensors.contains(bias_name) {
        return Err(ArchError::MissingTensor {
            name: bias_name.to_string(),
        });
    }
    let bias = load_bias(model, bias_name)?;
    if let Some(ref b) = bias {
        if b.len() != hidden_size {
            return Err(ArchError::InvalidShape {
                name: bias_name.to_string(),
                expected: vec![hidden_size],
                got: vec![b.len()],
            });
        }
    }
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Load an optional LayerNorm, returning `None` when its weight is absent.
///
/// Its bias is optional too — that is exactly how llama.cpp declares
/// `attn_norm_2`.
fn load_layer_norm_opt(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
    hidden_size: usize,
    eps: f32,
) -> ArchResult<Option<LayerNorm>> {
    if !model.file.tensors.contains(weight_name) {
        return Ok(None);
    }
    load_layer_norm(model, weight_name, bias_name, hidden_size, eps, false).map(Some)
}

/// Validate a linear projection's `[out_features, in_features]` shape.
fn check_linear_shape(
    name: &str,
    linear: &crate::common::linear::QuantLinear,
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
    Ok(())
}

/// Load a Falcon model from a parsed [`GgufModel`].
///
/// This is the real loading path.  [`FalconArchitecture::build`] cannot be it:
/// the [`ModelArchitecture`] trait only hands `build()` a `TensorStore`, which
/// has no access to the raw tensor payload, so it stays a config-validation
/// path that reports [`ArchError::MissingTensor`] — the same arrangement
/// starcoder and qwen3 use.
///
/// [`FalconArchitecture::build`]: crate::falcon::FalconArchitecture
/// [`ModelArchitecture`]: crate::traits::ModelArchitecture
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] when a required tensor is absent.
/// * [`ArchError::InvalidShape`] when a tensor's shape disagrees with the
///   metadata (including a truncated payload, which the shared loader helpers
///   detect instead of indexing past the end of the mapping).
/// * [`ArchError::ConfigMismatch`] when the head geometry is inconsistent.
pub fn load_falcon_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<FalconForward> {
    let cfg = FalconConfig::from_model_config(config)?;

    let hidden = cfg.hidden_size;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let intermediate = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let eps = cfg.norm_eps;

    // ── Head geometry ────────────────────────────────────────────────────
    //
    // `groups = n_heads / n_kv_heads` drives the GQA mapping in the attention
    // loop, and `wo` is `{n_embd, n_embd}` in the reference — so the
    // concatenated heads must be exactly `hidden_size` wide.  Both are checked
    // here so that a crafted GGUF fails at load instead of dividing by zero or
    // writing past a scratch buffer mid-decode.
    if n_kv == 0 || !n_heads.is_multiple_of(n_kv) {
        return Err(ArchError::ConfigMismatch {
            param: "falcon.attention.head_count_kv".to_string(),
            expected: format!("a non-zero divisor of head_count ({n_heads})"),
            got: n_kv.to_string(),
        });
    }
    if n_heads * head_dim != hidden {
        return Err(ArchError::ConfigMismatch {
            param: "falcon.attention.head_count * head_dim".to_string(),
            expected: format!("embedding_length ({hidden})"),
            got: (n_heads * head_dim).to_string(),
        });
    }

    // ── Token embeddings ─────────────────────────────────────────────────
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;
    let expected_embd = vocab
        .checked_mul(hidden)
        .ok_or_else(|| ArchError::InvalidConfig {
            detail: format!("vocab_size ({vocab}) * hidden_size ({hidden}) overflows"),
        })?;
    if token_embd.len() != expected_embd {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![vocab, hidden],
            got: vec![token_embd.len()],
        });
    }

    // ── Transformer layers ───────────────────────────────────────────────
    //
    // Fused QKV width: `n_embd + 2 * n_embd_gqa` where
    // `n_embd_gqa = n_head_kv * head_dim`.
    let qkv_total = (n_heads + 2 * n_kv) * head_dim;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");

        // `create_tensor(ATTN_NORM "weight"/"bias", i, {n_embd}, 0)` — both
        // required.
        let attn_norm = load_layer_norm(
            model,
            &format!("{prefix}.attn_norm.weight"),
            &format!("{prefix}.attn_norm.bias"),
            hidden,
            eps,
            true,
        )?;

        // Falcon-40B only (`TENSOR_NOT_REQUIRED`).
        let attn_norm_2 = load_layer_norm_opt(
            model,
            &format!("{prefix}.attn_norm_2.weight"),
            &format!("{prefix}.attn_norm_2.bias"),
            hidden,
            eps,
        )?;

        // Not a Falcon tensor in llama.cpp; read when present so that a
        // hand-built sequential checkpoint is not silently mis-normalised.
        let ffn_norm = load_layer_norm_opt(
            model,
            &format!("{prefix}.ffn_norm.weight"),
            &format!("{prefix}.ffn_norm.bias"),
            hidden,
            eps,
        )?;

        let qkv_name = format!("{prefix}.attn_qkv.weight");
        let attn_qkv =
            load_quant_linear_with_bias(model, &qkv_name, &format!("{prefix}.attn_qkv.bias"))?;
        check_linear_shape(&qkv_name, &attn_qkv, qkv_total, hidden)?;

        let out_name = format!("{prefix}.attn_output.weight");
        let attn_out =
            load_quant_linear_with_bias(model, &out_name, &format!("{prefix}.attn_output.bias"))?;
        check_linear_shape(&out_name, &attn_out, hidden, n_heads * head_dim)?;

        let up_name = format!("{prefix}.ffn_up.weight");
        let ffn_up =
            load_quant_linear_with_bias(model, &up_name, &format!("{prefix}.ffn_up.bias"))?;
        check_linear_shape(&up_name, &ffn_up, intermediate, hidden)?;

        let down_name = format!("{prefix}.ffn_down.weight");
        let ffn_down =
            load_quant_linear_with_bias(model, &down_name, &format!("{prefix}.ffn_down.bias"))?;
        check_linear_shape(&down_name, &ffn_down, hidden, intermediate)?;

        layers.push(FalconLayer {
            attn_norm,
            attn_norm_2,
            ffn_norm,
            attn_qkv,
            attn_out,
            ffn_up,
            ffn_down,
        });
    }

    // ── Final norm + LM head ─────────────────────────────────────────────
    // `create_tensor(OUTPUT_NORM "weight"/"bias", {n_embd}, 0)` — both required.
    let output_norm = load_layer_norm(
        model,
        "output_norm.weight",
        "output_norm.bias",
        hidden,
        eps,
        true,
    )?;

    // `output.weight` is `TENSOR_NOT_REQUIRED` and duplicated from
    // `token_embd.weight` when absent.
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;
    check_linear_shape("output.weight", &output, vocab, hidden)?;

    FalconForward::new(config.clone(), cfg, token_embd, layers, output_norm, output)
}
