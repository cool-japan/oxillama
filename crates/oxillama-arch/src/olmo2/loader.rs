//! GGUF loader for OLMo2.
//!
//! The tensor set is transcribed from `~/work/refs/llama.cpp`,
//! `src/llama-model.cpp`, `case LLM_ARCH_OLMO2:` inside `load_tensors()`:
//!
//! ```text
//! const int64_t n_embd_head = n_embd / n_head;
//! tok_embd    = create_tensor(TOKEN_EMBD  "weight", {n_embd, n_vocab}, 0);
//! output_norm = create_tensor(OUTPUT_NORM "weight", {n_embd},          0);
//! output      = create_tensor(OUTPUT      "weight", {n_embd, n_vocab}, 0);
//! // per layer i:
//! wq              = ATTN_Q         "weight" {n_embd, n_embd}
//! wk              = ATTN_K         "weight" {n_embd, n_embd_gqa}
//! wv              = ATTN_V         "weight" {n_embd, n_embd_gqa}
//! wo              = ATTN_OUT       "weight" {n_embd, n_embd}
//! attn_q_norm     = ATTN_Q_NORM    "weight" {n_embd}
//! attn_k_norm     = ATTN_K_NORM    "weight" {n_head_kv * n_embd_head}
//! attn_post_norm  = ATTN_POST_NORM "weight" {n_embd}
//! ffn_gate        = FFN_GATE       "weight" {n_embd, n_ff}
//! ffn_up          = FFN_UP         "weight" {n_embd, n_ff}
//! ffn_down        = FFN_DOWN       "weight" {n_ff,   n_embd}
//! ffn_post_norm   = FFN_POST_NORM  "weight" {n_embd}
//! ```
//!
//! Every one of those has `flags = 0`, i.e. **required**, and OLMo2 carries no
//! attention or FFN biases at all.
//!
//! The two post-norm enum slots resolve through the shared `LLM_TENSOR_NAMES`
//! table to the literal strings `blk.%d.post_attention_norm` and
//! `blk.%d.post_ffw_norm` (`src/llama-arch.cpp` lines 355 and 359) — see
//! [`crate::olmo2::tensor_names`].

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    load_dequant_tensor, load_lm_head, load_quant_linear, load_rms_norm_weight,
};
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::olmo2::config::Olmo2Config;
use crate::olmo2::forward::{resolve_kernel, Olmo2Forward, Olmo2Layer};
use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

/// Load an OLMo2 model from a parsed GGUF file.
///
/// # Tied embeddings
///
/// llama.cpp creates `output.weight` with `flags = 0` for OLMo2, so the
/// reference treats it as required and no released OLMo2 checkpoint ties its
/// LM head.  This loader nevertheless goes through
/// [`load_lm_head`], which prefers `output.weight` and only falls back to
/// `token_embd.weight` when the exact name is genuinely absent.  That is a
/// strict superset of the reference behaviour: a conforming checkpoint takes
/// the same path, and a re-quantized community checkpoint that dropped the
/// duplicate matrix loads instead of failing.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] for any absent required tensor.
/// * [`ArchError::InvalidShape`] when a norm weight or the embedding table
///   disagrees with the metadata — in particular when `attn_k_norm` is not
///   `n_head_kv * head_dim` elements wide.
pub fn load_olmo2_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<Olmo2Forward> {
    let cfg = Olmo2Config::from_model_config(config)?;
    let dispatcher = KernelDispatcher::new();

    if cfg.n_heads % cfg.n_kv_heads != 0 {
        return Err(ArchError::ConfigMismatch {
            param: "olmo2.attention.head_count_kv".to_string(),
            expected: format!("a divisor of head_count ({})", cfg.n_heads),
            got: cfg.n_kv_heads.to_string(),
        });
    }

    // `token_embd` is dequantized because the forward pass reads it as a plain
    // `[vocab_size * hidden_size]` f32 table.
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let attn_q = load_quant_linear(model, &format!("blk.{i}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("blk.{i}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("blk.{i}.attn_v.weight"))?;
        let attn_out = load_quant_linear(model, &format!("blk.{i}.attn_output.weight"))?;
        let ffn_gate = load_quant_linear(model, &format!("blk.{i}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("blk.{i}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("blk.{i}.ffn_down.weight"))?;

        // Both QK-norm weights are read at the width the *tensor itself*
        // declares (`load_rms_norm_weight` → `dequant_to_f32` →
        // `TensorInfo::n_elements()`).  Assuming `n_embd` for `attn_k_norm`
        // would over-read a GQA checkpoint, where it is `n_embd_gqa` wide.
        let attn_q_norm = RmsNorm::new(
            load_rms_norm_weight(model, &format!("blk.{i}.attn_q_norm.weight"))?,
            cfg.norm_eps,
        );
        let attn_k_norm = RmsNorm::new(
            load_rms_norm_weight(model, &format!("blk.{i}.attn_k_norm.weight"))?,
            cfg.norm_eps,
        );
        let attn_post_norm = RmsNorm::new(
            load_rms_norm_weight(model, &format!("blk.{i}.post_attention_norm.weight"))?,
            cfg.norm_eps,
        );
        let ffn_post_norm = RmsNorm::new(
            load_rms_norm_weight(model, &format!("blk.{i}.post_ffw_norm.weight"))?,
            cfg.norm_eps,
        );

        // Resolve every kernel once, here, so the per-token loop never
        // dispatches on `tensor_type` again.
        let attn_q_kernel = resolve_kernel(&dispatcher, &attn_q)?;
        let attn_k_kernel = resolve_kernel(&dispatcher, &attn_k)?;
        let attn_v_kernel = resolve_kernel(&dispatcher, &attn_v)?;
        let attn_out_kernel = resolve_kernel(&dispatcher, &attn_out)?;
        let ffn_gate_kernel = resolve_kernel(&dispatcher, &ffn_gate)?;
        let ffn_up_kernel = resolve_kernel(&dispatcher, &ffn_up)?;
        let ffn_down_kernel = resolve_kernel(&dispatcher, &ffn_down)?;

        check_linear(
            &format!("blk.{i}.attn_q.weight"),
            &attn_q,
            cfg.n_heads * cfg.head_dim,
            cfg.hidden_size,
        )?;
        check_linear(
            &format!("blk.{i}.attn_k.weight"),
            &attn_k,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.hidden_size,
        )?;
        check_linear(
            &format!("blk.{i}.attn_v.weight"),
            &attn_v,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.hidden_size,
        )?;
        check_linear(
            &format!("blk.{i}.attn_output.weight"),
            &attn_out,
            cfg.hidden_size,
            cfg.n_heads * cfg.head_dim,
        )?;
        check_linear(
            &format!("blk.{i}.ffn_gate.weight"),
            &ffn_gate,
            cfg.intermediate_size,
            cfg.hidden_size,
        )?;
        check_linear(
            &format!("blk.{i}.ffn_up.weight"),
            &ffn_up,
            cfg.intermediate_size,
            cfg.hidden_size,
        )?;
        check_linear(
            &format!("blk.{i}.ffn_down.weight"),
            &ffn_down,
            cfg.hidden_size,
            cfg.intermediate_size,
        )?;

        layers.push(Olmo2Layer {
            attn_q_norm,
            attn_k_norm,
            attn_post_norm,
            ffn_post_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_down,
            attn_q_kernel,
            attn_k_kernel,
            attn_v_kernel,
            attn_out_kernel,
            ffn_gate_kernel,
            ffn_up_kernel,
            ffn_down_kernel,
        });
    }

    let output_norm = RmsNorm::new(
        load_rms_norm_weight(model, "output_norm.weight")?,
        cfg.norm_eps,
    );
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;
    // `QuantLinear::forward` takes the GEMV's row/column counts from the
    // tensor's own shape, so an LM head that declares more rows than
    // `vocab_size` writes past `buf_logits`, and one whose in-features exceed
    // `hidden_size` reads past `buf_hidden` — both from an attacker-supplied
    // GGUF.  The tied fallback passes this identically: `token_embd.weight` is
    // `{n_embd, n_vocab}`, which `gguf_linear_shape` reverses to
    // `[vocab, hidden]` — the same shape a standalone `output.weight` gets.
    check_linear("output.weight", &output, cfg.vocab_size, cfg.hidden_size)?;

    let max_ctx = config.max_context_length;
    Olmo2Forward::new(cfg, token_embd, layers, output_norm, output, max_ctx)
}

/// Validate a loaded projection against the shape the metadata implies.
///
/// `QuantLinear::new` already reverses GGUF's fastest-changing-first `ne` into
/// `[out_features, in_features]`, so a transposed checkpoint shows up here
/// rather than as a GEMV that reads past its activation vector.
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
