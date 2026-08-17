//! GGUF loader for the GPT-NeoX (Pythia) architecture.
//!
//! Before this module existed, `gpt_neox` had **no** GGUF path at all: only
//! `GptNeoxModel::new(config, …, layers, …)` with pre-materialised weights, and
//! `GptNeoxArchitecture::build()` returning a `MissingTensor` sentinel.  No
//! checkpoint on disk could reach the forward pass.
//!
//! # Tensors read (and where each name comes from)
//!
//! Every name below is the string `gguf-py`'s `TENSOR_NAMES` maps the
//! `LLM_TENSOR_*` constant to at the `case LLM_ARCH_GPTNEOX:` site in
//! `~/work/refs/llama.cpp/src/llama-model.cpp`'s `load_tensors()`.  Every one
//! is created there with `flags = 0`, so every one is **required** here — a
//! missing bias is [`ArchError::MissingTensor`], not a silent zero.
//!
//! | GGUF name | llama.cpp field | Shape (`ne`, in-features first) |
//! |---|---|---|
//! | `token_embd.weight` | `tok_embd` | `{n_embd, n_vocab}` |
//! | `output_norm.weight` / `.bias` | `output_norm` / `output_norm_b` | `{n_embd}` |
//! | `output.weight` | `output` | `{n_embd, n_vocab}` |
//! | `blk.{i}.attn_norm.weight` / `.bias` | `attn_norm` / `attn_norm_b` | `{n_embd}` |
//! | `blk.{i}.attn_qkv.weight` | `wqkv` | `{n_embd, n_embd + 2*n_embd_gqa}` |
//! | `blk.{i}.attn_qkv.bias` | `bqkv` | `{n_embd + 2*n_embd_gqa}` |
//! | `blk.{i}.attn_output.weight` / `.bias` | `wo` / `bo` | `{n_embd, n_embd}` / `{n_embd}` |
//! | `blk.{i}.ffn_norm.weight` / `.bias` | `ffn_norm` / `ffn_norm_b` | `{n_embd}` |
//! | `blk.{i}.ffn_up.weight` / `.bias` | `ffn_up` / `ffn_up_b` | `{n_embd, n_ff}` / `{n_ff}` |
//! | `blk.{i}.ffn_down.weight` / `.bias` | `ffn_down` / `ffn_down_b` | `{n_ff, n_embd}` / `{n_embd}` |
//!
//! There is **no** `ln1`/`ln2` tensor and there are **no** separate
//! `attn_q`/`attn_k`/`attn_v` tensors anywhere in the GPT-NeoX family; see
//! [`super::tensor_names`].
//!
//! # Hyper-parameters read directly from metadata
//!
//! [`ModelConfig`] carries neither of the two keys this architecture branches
//! on, so both are read here through [`ExtraHparams`]:
//!
//! * `{arch}.rope.dimension_count` → `n_rot`.  Fallback when absent is
//!   `head_dim`, matching `llama-model.cpp`:
//!   `hparams.n_rot = hparams.n_embd_head_k;` followed by the *optional*
//!   `ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false)`.
//!   `convert_hf_to_gguf.py::GPTNeoXModel.set_gguf_parameters` always writes
//!   it as `int(rotary_pct * head_dim)`, so the fallback is only reached by a
//!   hand-built file.
//! * `{arch}.use_parallel_residual` → the block topology.  Fallback `true`,
//!   matching the converter's `hparams.get("use_parallel_residual", True)`.
//! * `{arch}.attention.layer_norm_epsilon` → the plain-LayerNorm epsilon
//!   (`ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps)` in the
//!   `LLM_ARCH_GPTNEOX` hparams case).

use oxillama_gguf::GgufModel;

use crate::common::layer_norm::LayerNorm;
use crate::common::linear::QuantLinear;
use crate::common::loader::{load_dequant_tensor, load_lm_head, load_quant_linear};
use crate::config::{ExtraHparams, ModelConfig};
use crate::error::{ArchError, ArchResult};
use crate::gpt_neox::model::{
    validate_gpt_neox_shapes, validate_rotary_dims, GptNeoxLayer, GptNeoxLayerWeights,
    GptNeoxModel, DEFAULT_USE_PARALLEL_RESIDUAL,
};

/// Load a complete [`GptNeoxModel`] from a parsed GGUF file.
///
/// # Errors
///
/// * [`ArchError::ConfigMismatch`] when the metadata's attention geometry is
///   inconsistent (`n_head * head_dim != n_embd`) or `n_rot` is odd / wider
///   than a head.
/// * [`ArchError::MissingTensor`] naming the first absent tensor — including
///   any of the six per-layer bias tensors, all of which GPT-NeoX requires.
/// * [`ArchError::InvalidShape`] when a tensor's declared shape disagrees with
///   the metadata-derived dimensions.
pub fn load_gpt_neox_from_gguf(
    model: &GgufModel,
    config: &ModelConfig,
) -> ArchResult<GptNeoxModel> {
    // Reject an impossible geometry *before* touching the payload, so the
    // reported error names the metadata that is actually wrong rather than the
    // first tensor whose shape happens to disagree with it.
    validate_gpt_neox_shapes(config)?;

    let arch = if config.architecture.is_empty() {
        "gptneox"
    } else {
        config.architecture.as_str()
    };
    let extra = ExtraHparams::from_metadata(&model.file.metadata, arch);

    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let n_ff = config.intermediate_size;
    let vocab = config.vocab_size;
    let attn_dim = config.num_attention_heads * head_dim;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_total = attn_dim + 2 * kv_dim;

    let rotary_dims = extra.rope_dimension_count.unwrap_or(head_dim);
    validate_rotary_dims(rotary_dims, head_dim)?;

    let use_parallel_residual = extra
        .use_parallel_residual
        .unwrap_or(DEFAULT_USE_PARALLEL_RESIDUAL);

    // GPT-NeoX normalises with a plain LayerNorm, whose epsilon lives under
    // `attention.layer_norm_epsilon`.  `ModelConfig` already falls back to that
    // key, but reading it explicitly keeps the intent visible and survives a
    // checkpoint that carries both spellings.
    let ln_eps = extra.layer_norm_eps.unwrap_or(config.rms_norm_eps);

    // ── Token embeddings ─────────────────────────────────────────────────────
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;
    let expected_embd = vocab
        .checked_mul(hidden)
        .ok_or_else(|| ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![vocab, hidden],
            got: vec![token_embd.len()],
        })?;
    if token_embd.len() != expected_embd {
        return Err(ArchError::InvalidShape {
            name: "token_embd.weight".to_string(),
            expected: vec![expected_embd],
            got: vec![token_embd.len()],
        });
    }

    // ── Blocks ───────────────────────────────────────────────────────────────
    let dispatcher = oxillama_quant::KernelDispatcher::new();
    let mut layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_layer_norm(model, &prefix, "attn_norm", hidden, ln_eps)?;
        let ffn_norm = load_layer_norm(model, &prefix, "ffn_norm", hidden, ln_eps)?;

        let attn_qkv_name = format!("{prefix}.attn_qkv.weight");
        let attn_qkv = load_quant_linear(model, &attn_qkv_name)?;
        check_linear(&attn_qkv_name, &attn_qkv, qkv_total, hidden)?;
        let attn_qkv_bias =
            load_required_vector(model, &format!("{prefix}.attn_qkv.bias"), qkv_total)?;

        let attn_output_name = format!("{prefix}.attn_output.weight");
        let attn_output = load_quant_linear(model, &attn_output_name)?;
        check_linear(&attn_output_name, &attn_output, hidden, attn_dim)?;
        let attn_out_bias =
            load_required_vector(model, &format!("{prefix}.attn_output.bias"), hidden)?;

        let ffn_up_name = format!("{prefix}.ffn_up.weight");
        let ffn_up = load_quant_linear(model, &ffn_up_name)?;
        check_linear(&ffn_up_name, &ffn_up, n_ff, hidden)?;
        let ffn_up_bias = load_required_vector(model, &format!("{prefix}.ffn_up.bias"), n_ff)?;

        let ffn_down_name = format!("{prefix}.ffn_down.weight");
        let ffn_down = load_quant_linear(model, &ffn_down_name)?;
        check_linear(&ffn_down_name, &ffn_down, hidden, n_ff)?;
        let ffn_down_bias =
            load_required_vector(model, &format!("{prefix}.ffn_down.bias"), hidden)?;

        layers.push(GptNeoxLayer::new(
            &dispatcher,
            GptNeoxLayerWeights {
                attn_norm,
                attn_qkv,
                attn_qkv_bias,
                attn_output,
                attn_out_bias,
                ffn_norm,
                ffn_up,
                ffn_up_bias,
                ffn_down,
                ffn_down_bias,
            },
        )?);
    }

    // ── Final norm + LM head ─────────────────────────────────────────────────
    let output_norm_w = load_required_vector(model, "output_norm.weight", hidden)?;
    let output_norm_b = load_required_vector(model, "output_norm.bias", hidden)?;
    let output_norm = LayerNorm::new(output_norm_w, Some(output_norm_b), ln_eps);

    // `LLM_ARCH_GPTNEOX` creates `output` with `flags = 0` and has no explicit
    // tied-embedding fallback, so a well-formed checkpoint always ships
    // `output.weight`.  `load_lm_head` tries that exact name first and only
    // falls back to `token_embd.weight` when it is genuinely absent, so it is a
    // strict superset of the reference behaviour: identical on every real
    // GPT-NeoX file, and it additionally accepts a hand-tied one instead of
    // failing.  Nothing that llama.cpp loads is rejected, and nothing it
    // rejects is silently mis-loaded — the fallback reuses `token_embd`'s own
    // `[n_vocab, n_embd]` matrix, which `check_linear` then verifies.
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;
    check_linear("output.weight", &output, vocab, hidden)?;

    GptNeoxModel::new(
        config.clone(),
        rotary_dims,
        use_parallel_residual,
        token_embd,
        layers,
        output_norm,
        output,
    )
}

/// Load a `{prefix}.{name}.weight` / `.bias` pair as a plain LayerNorm.
///
/// Both halves are required: `attn_norm_b` and `ffn_norm_b` are created with
/// `flags = 0` for `LLM_ARCH_GPTNEOX`.
fn load_layer_norm(
    model: &GgufModel,
    prefix: &str,
    name: &str,
    hidden: usize,
    eps: f32,
) -> ArchResult<LayerNorm> {
    let weight = load_required_vector(model, &format!("{prefix}.{name}.weight"), hidden)?;
    let bias = load_required_vector(model, &format!("{prefix}.{name}.bias"), hidden)?;
    Ok(LayerNorm::new(weight, Some(bias), eps))
}

/// Load a required 1-D tensor and check its length.
///
/// # Errors
///
/// [`ArchError::MissingTensor`] when absent (this is what makes GPT-NeoX's
/// mandatory biases mandatory), [`ArchError::InvalidShape`] on a length
/// mismatch.
fn load_required_vector(model: &GgufModel, name: &str, expected: usize) -> ArchResult<Vec<f32>> {
    let values = load_dequant_tensor(model, name)?;
    if values.len() != expected {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![expected],
            got: vec![values.len()],
        });
    }
    Ok(values)
}

/// Verify a projection's `[out_features, in_features]` against the metadata.
fn check_linear(
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
    Ok(())
}
