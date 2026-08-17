//! GGUF loading for Qwen2-VL.
//!
//! Two rules, both of which the previous loader broke:
//!
//! 1. **No zero fallbacks.** Every tensor inside a tower that is being loaded is
//!    required.  A missing one is [`ArchError::MissingTensor`], never
//!    `unwrap_or_else(|| vec![0.0; …])`.
//! 2. **No fabricated tower.** A checkpoint without `clip.vision.*` metadata
//!    simply has no vision half; `vision` is `None` and
//!    [`crate::qwen2_vl::Qwen2VlModel::encode_image`] says so out loud.
//!
//! The private, unguarded `dequant_to_f32_local` this module used to carry —
//! whose `&data[data_offset..data_offset + block_bytes]` panicked on a
//! truncated GGUF — is gone; everything routes through the bounds-checked
//! [`crate::common::loader`] helpers.

use crate::common::loader::{
    load_bias, load_dequant_tensor, load_lm_head, load_quant_linear, load_quant_linear_with_bias,
    load_rms_norm_weight,
};
use crate::common::rms_norm::RmsNorm;
use crate::config::{ExtraHparams, ModelConfig, VisionConfig};
use crate::error::{ArchError, ArchResult};
use crate::qwen2_vl::model::{MmMerger, Qwen2Layer, Qwen2VlModel, Qwen2VlVision};
use crate::qwen2_vl::vision::{Qwen2VlVisionEncoder, VisionBlock, VisionFfnOp, VisionNorm};
use oxillama_gguf::{GgufModel, MetadataValue};

/// Load a Qwen2-VL model from a `GgufModel`.
///
/// The vision half is loaded only when the file declares it, i.e. when
/// `config.vision_config` is `Some` **and** the file actually carries
/// `v.patch_embd.weight`.  Declaring a tower and then omitting its tensors is an
/// error, not a zero fill.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] — any required backbone or vision tensor.
/// * [`ArchError::InvalidShape`] / [`ArchError::InvalidConfig`] — inconsistent
///   `rope.dimension_sections` or vision geometry.
pub fn load_qwen2vl_from_gguf(model: &GgufModel, config: &ModelConfig) -> ArchResult<Qwen2VlModel> {
    let token_embd = load_dequant_tensor(model, "token_embd.weight")?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");
        layers.push(Qwen2Layer {
            attn_norm: RmsNorm::new(
                load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?,
                config.rms_norm_eps,
            ),
            attn_q: load_quant_linear_with_bias(
                model,
                &format!("{prefix}.attn_q.weight"),
                &format!("{prefix}.attn_q.bias"),
            )?,
            attn_k: load_quant_linear_with_bias(
                model,
                &format!("{prefix}.attn_k.weight"),
                &format!("{prefix}.attn_k.bias"),
            )?,
            attn_v: load_quant_linear_with_bias(
                model,
                &format!("{prefix}.attn_v.weight"),
                &format!("{prefix}.attn_v.bias"),
            )?,
            attn_output: load_quant_linear_with_bias(
                model,
                &format!("{prefix}.attn_output.weight"),
                &format!("{prefix}.attn_output.bias"),
            )?,
            ffn_norm: RmsNorm::new(
                load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?,
                config.rms_norm_eps,
            ),
            ffn_gate: load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?,
            ffn_up: load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?,
            ffn_down: load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?,
        });
    }

    let output_norm = RmsNorm::new(
        load_rms_norm_weight(model, "output_norm.weight")?,
        config.rms_norm_eps,
    );
    // Qwen2-VL-2B ties its LM head to the embedding matrix.
    let output = load_lm_head(model, "output.weight", "token_embd.weight")?;

    let vision = load_vision(model, config)?;

    // `qwen2vl.rope.dimension_sections` — `[16, 24, 24, 0]` for a 128-wide
    // head, which is *not* an equal three-way split of its 64 rotation pairs.
    let extra = ExtraHparams::from_metadata(&model.file.metadata, &config.architecture);
    let rope_sections = extra.rope_sections.clone();

    Qwen2VlModel::new(
        config.clone(),
        vision,
        token_embd,
        layers,
        output_norm,
        output,
        rope_sections.as_deref(),
    )
}

/// Load the vision half, or `None` when the checkpoint has no tower.
fn load_vision(model: &GgufModel, config: &ModelConfig) -> ArchResult<Option<Qwen2VlVision>> {
    // Prefer the parsed config; fall back to reading the metadata directly so a
    // caller that built `ModelConfig` by hand still gets a tower.
    let vis_cfg = match config.vision_config.clone() {
        Some(v) => v,
        None => match VisionConfig::from_metadata(&model.file.metadata) {
            Some(v) => v,
            None => return Ok(None),
        },
    };

    // Metadata declares a tower; from here every tensor is required.
    if !model.file.tensors.contains("v.patch_embd.weight") {
        return Err(ArchError::MissingTensor {
            name: format!(
                "v.patch_embd.weight — the checkpoint declares a {}-block vision tower via \
                 clip.vision.block_count but carries no v.* tensors. Vision weights live in the \
                 separate mmproj GGUF (general.architecture = \"clip\")",
                vis_cfg.num_layers
            ),
        });
    }

    let encoder = load_vision_encoder(model, &vis_cfg)?;
    let merger = load_merger(model, vis_cfg.hidden_size, config.hidden_size)?;
    Ok(Some(Qwen2VlVision { encoder, merger }))
}

/// Load the ViT.
fn load_vision_encoder(model: &GgufModel, cfg: &VisionConfig) -> ArchResult<Qwen2VlVisionEncoder> {
    let meta = &model.file.metadata;

    // Qwen2.5-VL uses RMSNorm; plain Qwen2-VL uses LayerNorm
    // (`models/qwen2vl.cpp:12-15`).  The projector type is the discriminator.
    let projector = meta.get_string("clip.projector_type").unwrap_or("");
    let norm = if projector.contains("2.5") || projector.contains("25") {
        VisionNorm::Rms
    } else {
        VisionNorm::Layer
    };

    // `clip.use_gelu` / `clip.use_silu` carry no `.vision.` segment; neither set
    // means QuickGELU (`clip.cpp:1070-1088`).
    let ffn_op = match (
        meta.get("clip.use_gelu").and_then(MetadataValue::as_bool),
        meta.get("clip.use_silu").and_then(MetadataValue::as_bool),
    ) {
        (Some(true), _) => VisionFfnOp::Gelu,
        (_, Some(true)) => VisionFfnOp::Silu,
        _ => VisionFfnOp::GeluQuick,
    };

    let norm_eps = meta
        .get_f32("clip.vision.attention.layer_norm_epsilon")
        .unwrap_or(1e-6);

    // Qwen2-VL's patch embedding is a temporal-2 conv exported as two tensors
    // applied to the same input and summed (`models/qwen2vl.cpp:20-33`), so the
    // effective weight is their elementwise sum.
    let mut patch_embd_weight = load_dequant_tensor(model, "v.patch_embd.weight")?;
    if model.file.tensors.contains("v.patch_embd.weight.1") {
        let second = load_dequant_tensor(model, "v.patch_embd.weight.1")?;
        if second.len() != patch_embd_weight.len() {
            return Err(ArchError::InvalidShape {
                name: "v.patch_embd.weight.1".to_string(),
                expected: vec![patch_embd_weight.len()],
                got: vec![second.len()],
            });
        }
        for (a, b) in patch_embd_weight.iter_mut().zip(second.iter()) {
            *a += b;
        }
    }

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        let pfx = format!("v.blk.{i}");
        layers.push(VisionBlock {
            ln1_weight: load_dequant_tensor(model, &format!("{pfx}.ln1.weight"))?,
            ln1_bias: optional(model, &format!("{pfx}.ln1.bias"))?,
            ln2_weight: load_dequant_tensor(model, &format!("{pfx}.ln2.weight"))?,
            ln2_bias: optional(model, &format!("{pfx}.ln2.bias"))?,
            attn_q_weight: load_dequant_tensor(model, &format!("{pfx}.attn_q.weight"))?,
            attn_q_bias: optional(model, &format!("{pfx}.attn_q.bias"))?,
            attn_k_weight: load_dequant_tensor(model, &format!("{pfx}.attn_k.weight"))?,
            attn_k_bias: optional(model, &format!("{pfx}.attn_k.bias"))?,
            attn_v_weight: load_dequant_tensor(model, &format!("{pfx}.attn_v.weight"))?,
            attn_v_bias: optional(model, &format!("{pfx}.attn_v.bias"))?,
            attn_out_weight: load_dequant_tensor(model, &format!("{pfx}.attn_out.weight"))?,
            attn_out_bias: optional(model, &format!("{pfx}.attn_out.bias"))?,
            ffn_up_weight: load_dequant_tensor(model, &format!("{pfx}.ffn_up.weight"))?,
            ffn_up_bias: optional(model, &format!("{pfx}.ffn_up.bias"))?,
            ffn_down_weight: load_dequant_tensor(model, &format!("{pfx}.ffn_down.weight"))?,
            ffn_down_bias: optional(model, &format!("{pfx}.ffn_down.bias"))?,
        });
    }

    Ok(Qwen2VlVisionEncoder {
        patch_size: cfg.patch_size,
        window_size: cfg.window_size,
        hidden_size: cfg.hidden_size,
        num_heads: cfg.num_heads,
        norm,
        ffn_op,
        norm_eps,
        rope_base: 10_000.0,
        layers,
        patch_embd_weight,
        post_ln_weight: optional(model, "v.post_ln.weight")?,
        post_ln_bias: optional(model, "v.post_ln.bias")?,
    })
}

/// Load `mm.0.*` / `mm.2.*`.
fn load_merger(
    model: &GgufModel,
    vis_hidden_size: usize,
    llm_hidden_size: usize,
) -> ArchResult<MmMerger> {
    if vis_hidden_size == 0 {
        return Err(ArchError::InvalidConfig {
            detail: "MmMerger: vision hidden size must be > 0".to_string(),
        });
    }
    let fc1_weight = load_dequant_tensor(model, "mm.0.weight")?;
    let fc1_bias = load_bias(model, "mm.0.bias")?.unwrap_or_default();
    let fc2_weight = load_dequant_tensor(model, "mm.2.weight")?;
    let fc2_bias = load_bias(model, "mm.2.bias")?.unwrap_or_default();

    let merged_in_dim = 4 * vis_hidden_size;
    let mm_hidden_size = fc1_weight.len() / merged_in_dim;
    if mm_hidden_size == 0 {
        return Err(ArchError::InvalidShape {
            name: "mm.0.weight".to_string(),
            expected: vec![merged_in_dim],
            got: vec![fc1_weight.len()],
        });
    }
    if fc2_weight.len() < llm_hidden_size * mm_hidden_size {
        return Err(ArchError::InvalidShape {
            name: "mm.2.weight".to_string(),
            expected: vec![llm_hidden_size, mm_hidden_size],
            got: vec![fc2_weight.len()],
        });
    }

    Ok(MmMerger {
        vis_hidden_size,
        llm_hidden_size,
        mm_hidden_size,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    })
}

/// Load a genuinely optional tensor, propagating read errors for one present.
fn optional(model: &GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    if model.file.tensors.contains(name) {
        load_dequant_tensor(model, name)
    } else {
        Ok(Vec::new())
    }
}
