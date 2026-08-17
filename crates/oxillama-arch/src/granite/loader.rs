//! GGUF → [`GraniteModel`].
//!
//! # Tensor set
//!
//! Exactly LLaMA's.  `src/llama-arch.cpp` gives `LLM_ARCH_GRANITE` the same
//! twelve `LLM_TENSOR_*` entries as `LLM_ARCH_LLAMA`, and `src/llama-model.cpp`
//! creates them in the shared `case LLM_ARCH_LLAMA: … case LLM_ARCH_GRANITE:`
//! arm, with `output.weight` marked `TENSOR_NOT_REQUIRED` and re-bound to
//! `token_embd.weight` when absent (`TENSOR_DUPLICATED`).
//!
//! ```text
//! token_embd.weight
//! blk.{i}.attn_norm.weight      blk.{i}.ffn_norm.weight
//! blk.{i}.attn_q.weight         blk.{i}.attn_k.weight
//! blk.{i}.attn_v.weight         blk.{i}.attn_output.weight
//! blk.{i}.ffn_gate.weight       blk.{i}.ffn_up.weight
//! blk.{i}.ffn_down.weight
//! output_norm.weight            output.weight   (optional — may be tied)
//! ```
//!
//! # Why the scales are read here and not in `ModelConfig`
//!
//! `crate::config` parses `{arch}.logit_scale` into [`ModelConfig::logit_scale`]
//! for **every** architecture and its only other consumer, Command-R,
//! *multiplies* by it — while Granite *divides*.  It parses none of
//! `embedding_scale` / `residual_scale` / `attention.scale` at all.  This loader
//! therefore reads all four straight out of `model.file.metadata` through
//! [`GraniteScales::from_metadata`]; see that module for the reference
//! citations and for the `config.rs` collision in full.
//!
//! # Bounds checking
//!
//! Every tensor goes through `crate::common::loader`, which validates each
//! payload against its declared element count before decoding a byte.  The
//! per-architecture `dequant_to_f32` copies these helpers replaced sliced
//! `&data[off..off + block_bytes]` unchecked and aborted the process on a
//! truncated GGUF.

use std::sync::Arc;

use oxillama_gguf::GgufModel;
use oxillama_quant::QuantKernel;

use crate::common::linear::QuantLinear;
use crate::common::loader::{
    load_dequant_tensor, load_lm_head, load_quant_linear, load_rms_norm_weight,
};
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::granite::model::{GraniteLayer, GraniteModel};
use crate::granite::scales::GraniteScales;

/// Resolve `linear`'s dequantization kernel once, at load time.
///
/// The decode loop would otherwise walk the tensor-type match ladder — and
/// allocate a fresh `Box<dyn QuantKernel>` — seven times per layer per token.
fn resolve_kernel(linear: &QuantLinear) -> ArchResult<Arc<dyn QuantKernel>> {
    oxillama_quant::global_dispatcher()
        .get_kernel(linear.weight.tensor_type)
        .map_err(ArchError::from)
}

/// Load a Granite-3.x model from a parsed GGUF file.
///
/// `config` supplies the geometry (`ModelConfig::from_metadata` already handles
/// every key Granite shares with LLaMA); the four Granite-specific multipliers
/// come from `model.file.metadata` directly.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] naming the first absent required tensor.
/// * [`ArchError::InvalidShape`] when `token_embd.weight` does not hold
///   `vocab_size × hidden_size` elements, or when any payload is shorter than
///   its declared shape.
/// * [`ArchError::InvalidConfig`] for degenerate attention geometry.
pub fn load_granite_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<GraniteModel> {
    // The scales are namespaced under the *declared* architecture id so a
    // re-quantizer that wrote `granitemoe` or a vendor prefix still resolves;
    // fall back to the canonical `granite` when the config carries no id.
    let arch = if config.architecture.is_empty() {
        "granite"
    } else {
        config.architecture.as_str()
    };
    let scales = GraniteScales::from_metadata(&model.file.metadata, arch);

    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn_gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;

        layers.push(GraniteLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            attn_q_kernel: resolve_kernel(&attn_q)?,
            attn_k_kernel: resolve_kernel(&attn_k)?,
            attn_v_kernel: resolve_kernel(&attn_v)?,
            attn_output_kernel: resolve_kernel(&attn_output)?,
            ffn_gate_kernel: resolve_kernel(&ffn_gate)?,
            ffn_up_kernel: resolve_kernel(&ffn_up)?,
            ffn_down_kernel: resolve_kernel(&ffn_down)?,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let output_norm = RmsNorm::new(
        load_rms_norm_weight(model, "output_norm.weight")?,
        config.rms_norm_eps,
    );
    // Granite-3.0-2B and -3.1-2B tie their embeddings and ship no
    // `output.weight`; llama.cpp falls back to `tok_embd` for exactly this.
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    GraniteModel::new(
        config.clone(),
        scales,
        token_embd,
        layers,
        output_norm,
        output,
    )
}

impl GraniteModel {
    /// Load a Granite model straight from a parsed GGUF file.
    ///
    /// Thin alias for [`load_granite_from_gguf`], matching the
    /// `XxxModel::from_gguf` spelling the rest of the crate uses.
    ///
    /// # Errors
    ///
    /// See [`load_granite_from_gguf`].
    pub fn from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<Self> {
        load_granite_from_gguf(model, config)
    }
}
