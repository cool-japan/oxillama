//! Canonical GGUF tensor-loading helpers shared by every architecture.
//!
//! Before this module existed, `dequant_to_f32`, `load_dequant_tensor`,
//! `load_quant_linear` and `load_rms_norm_weight` were duplicated in more than
//! ten architecture modules.  Every copy computed its block count from the
//! tensor's *declared* `n_elements` and then did
//!
//! ```ignore
//! let block_data = &data[data_offset..data_offset + block_bytes];
//! ```
//!
//! without ever comparing that range against `data.len()`, so a truncated or
//! malformed GGUF aborted the process during load.  The F32/F16 branches did
//! the opposite and silently zero-padded, and `lora/loader.rs` clamped with
//! `.min(data.len())` — three different behaviours for the same failure.
//!
//! Everything here validates with `data.get(..)` and reports
//! [`ArchError::InvalidShape`] naming the tensor.
//!
//! # Migration
//!
//! Each architecture deletes its private copies and imports these instead:
//!
//! ```ignore
//! use crate::common::loader::{
//!     dequant_to_f32, load_dequant_tensor, load_lm_head, load_quant_linear,
//!     load_quant_linear_opt, load_quant_linear_with_bias, load_rms_norm_weight,
//! };
//! ```

use oxillama_gguf::{GgufModel, GgufTensorType, TensorInfo};
use oxillama_quant::{KernelDispatcher, QuantTensor};

use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::error::{ArchError, ArchResult};

/// Dequantize `n_elements` weights of `tensor_type` out of `data`.
///
/// # Errors
///
/// [`ArchError::InvalidShape`] if `data` is too short for the declared element
/// count (the truncated-GGUF case that used to panic), or
/// [`ArchError::Quant`] if the kernel rejects a block.
pub fn dequant_to_f32_slice(
    name: &str,
    tensor_type: GgufTensorType,
    data: &[u8],
    n_elements: usize,
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let short = |need: usize| ArchError::InvalidShape {
        name: name.to_string(),
        expected: vec![need],
        got: vec![data.len()],
    };

    if tensor_type == GgufTensorType::F32 {
        let need = n_elements.checked_mul(4).ok_or_else(|| short(usize::MAX))?;
        if data.len() < need {
            return Err(short(need));
        }
        let mut out = vec![0.0f32; n_elements];
        for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == GgufTensorType::F16 {
        let need = n_elements.checked_mul(2).ok_or_else(|| short(usize::MAX))?;
        if data.len() < need {
            return Err(short(need));
        }
        let mut out = vec![0.0f32; n_elements];
        for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            *slot = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    if block_size == 0 || block_bytes == 0 {
        return Err(ArchError::InvalidConfig {
            detail: format!("tensor '{name}': quantization type {tensor_type:?} has a zero block"),
        });
    }
    let n_blocks = n_elements.div_ceil(block_size);
    let need = n_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| short(usize::MAX))?;
    if data.len() < need {
        return Err(short(need));
    }

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = data
            .get(data_offset..data_offset + block_bytes)
            .ok_or_else(|| short(data_offset + block_bytes))?;
        let out_end = out_offset.saturating_add(block_size).min(n_elements);
        let out_slice =
            out.get_mut(out_offset..out_end)
                .ok_or_else(|| ArchError::InvalidShape {
                    name: name.to_string(),
                    expected: vec![n_elements],
                    got: vec![out_offset],
                })?;
        if out_slice.len() == block_size {
            kernel.dequant_block(block_data, out_slice)?;
        } else {
            // Ragged tail: dequantize into a full-width scratch block and copy
            // back only the elements that exist.  Handing a short slice to the
            // kernel is what the duplicated copies did, and several kernels
            // index the full block width.
            let mut scratch = vec![0.0f32; block_size];
            kernel.dequant_block(block_data, &mut scratch)?;
            let take = out_slice.len();
            out_slice.copy_from_slice(&scratch[..take]);
        }
    }

    Ok(out)
}

/// Dequantize a tensor described by `info` to f32.
///
/// # Errors
///
/// See [`dequant_to_f32_slice`].
pub fn dequant_to_f32(
    info: &TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    dequant_to_f32_slice(
        &info.name,
        info.tensor_type,
        data,
        info.n_elements() as usize,
        dispatcher,
    )
}

/// Load a tensor from `model` and dequantize it to f32.
///
/// # Errors
///
/// [`ArchError::MissingTensor`] when `name` is absent; otherwise see
/// [`dequant_to_f32_slice`].
pub fn load_dequant_tensor(model: &GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let data = model.tensor_data(name)?;
    let dispatcher = KernelDispatcher::new();
    dequant_to_f32(info, data, &dispatcher)
}

/// Load an RMSNorm / LayerNorm weight vector from GGUF.
///
/// # Errors
///
/// See [`load_dequant_tensor`].
pub fn load_rms_norm_weight(model: &GgufModel, name: &str) -> ArchResult<Vec<f32>> {
    load_dequant_tensor(model, name)
}

/// Load a bias vector if it exists, returning `None` when absent.
///
/// Bias tensors are conventionally F32, but this goes through the shared
/// dequantizer so an F16 bias is handled too.
///
/// # Errors
///
/// Propagates dequantization failures for a bias that *is* present.
pub fn load_bias(model: &GgufModel, name: &str) -> ArchResult<Option<Vec<f32>>> {
    if !model.file.tensors.contains(name) {
        return Ok(None);
    }
    load_dequant_tensor(model, name).map(Some)
}

/// Load a quantized linear layer, keeping the weights in their GGUF format.
///
/// The payload is an mmap-backed [`SharedBytes`](oxillama_gguf::SharedBytes)
/// view, **not** a copy: loading a 4 GB model must not materialise 4 GB of f32.
///
/// # Errors
///
/// [`ArchError::MissingTensor`] when `name` is absent.
pub fn load_quant_linear(model: &GgufModel, name: &str) -> ArchResult<QuantLinear> {
    load_quant_linear_inner(model, name, None)
}

/// Load a quantized linear layer together with its bias, if the checkpoint has
/// one.
///
/// # Errors
///
/// [`ArchError::MissingTensor`] when the *weight* is absent; a missing bias is
/// not an error.
pub fn load_quant_linear_with_bias(
    model: &GgufModel,
    weight_name: &str,
    bias_name: &str,
) -> ArchResult<QuantLinear> {
    load_quant_linear_inner(model, weight_name, Some(bias_name))
}

/// Load a quantized linear layer, returning `None` when the tensor is absent.
///
/// # Errors
///
/// Propagates GGUF read failures for a tensor that *is* present.
pub fn load_quant_linear_opt(model: &GgufModel, name: &str) -> ArchResult<Option<QuantLinear>> {
    if !model.file.tensors.contains(name) {
        return Ok(None);
    }
    load_quant_linear(model, name).map(Some)
}

fn load_quant_linear_inner(
    model: &GgufModel,
    weight_name: &str,
    bias_name: Option<&str>,
) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(weight_name)
        .map_err(|_| ArchError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    let data = model.tensor_bytes(weight_name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);

    let bias = match bias_name {
        Some(b) => load_bias(model, b)?,
        None => None,
    };

    Ok(QuantLinear::new(tensor, bias))
}

/// Load the LM head, falling back to the tied token-embedding matrix.
///
/// Many checkpoints (Gemma, Qwen3, Phi-3-mini, Llama-3.2-1B/3B, …) omit
/// `output.weight` because the output projection is tied to
/// `token_embd.weight`.  Each architecture open-coded this fallback — or
/// forgot to.
///
/// # Errors
///
/// [`ArchError::MissingTensor`] naming *both* candidates when neither exists.
pub fn load_lm_head(
    model: &GgufModel,
    output_name: &str,
    token_embd_name: &str,
) -> ArchResult<QuantLinear> {
    if model.file.tensors.contains(output_name) {
        return load_quant_linear(model, output_name);
    }
    if model.file.tensors.contains(token_embd_name) {
        return load_quant_linear(model, token_embd_name);
    }
    Err(ArchError::MissingTensor {
        name: format!("{output_name} (and no tied fallback '{token_embd_name}')"),
    })
}

/// Split a stacked Mixture-of-Experts weight tensor into per-expert views.
///
/// GGUF stores MoE weights as one 3-D tensor per projection — e.g.
/// `blk.N.ffn_gate_exps.weight` with `ne = [in_features, out_features,
/// n_expert]`.  Expert `e` occupies a contiguous `out_features × in_features`
/// run, and because `in_features` is a whole number of quantization blocks for
/// every real GGUF quant type, every per-expert offset is block-aligned.
///
/// The returned [`QuantTensor`]s are **shared views** over the same mmap, so
/// splitting a Mixtral-8x7B layer costs no extra memory.  This is what makes a
/// quantized expert path possible at all: dequantizing the experts to `f32` at
/// load time needs ~180 GB for Mixtral-8x7B and >500 GB for DBRX.
///
/// # Errors
///
/// * [`ArchError::MissingTensor`] when `name` is absent.
/// * [`ArchError::InvalidShape`] when the tensor is not 3-D, when the expert
///   count disagrees with `expected_experts`, when `in_features` is not a whole
///   number of blocks, or when the payload is shorter than the declared shape.
pub fn load_stacked_experts(
    model: &GgufModel,
    name: &str,
    expected_experts: usize,
) -> ArchResult<Vec<QuantTensor>> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;

    let dims = &info.dimensions;
    if dims.len() != 3 {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![3],
            got: vec![dims.len()],
        });
    }
    // GGUF `ne` is fastest-changing-first.
    let in_features = dims[0] as usize;
    let out_features = dims[1] as usize;
    let n_expert = dims[2] as usize;

    if n_expert != expected_experts {
        return Err(ArchError::InvalidShape {
            name: format!("{name}.n_expert"),
            expected: vec![expected_experts],
            got: vec![n_expert],
        });
    }

    let tensor_type = info.tensor_type;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    if block_size == 0 || block_bytes == 0 {
        return Err(ArchError::InvalidConfig {
            detail: format!("tensor '{name}': quantization type {tensor_type:?} has a zero block"),
        });
    }
    if !in_features.is_multiple_of(block_size) {
        return Err(ArchError::InvalidShape {
            name: format!("{name}.in_features"),
            expected: vec![block_size],
            got: vec![in_features],
        });
    }

    let row_bytes = (in_features / block_size) * block_bytes;
    let expert_bytes = out_features * row_bytes;
    let data = model.tensor_bytes(name)?;
    let need = expert_bytes
        .checked_mul(n_expert)
        .ok_or_else(|| ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![usize::MAX],
            got: vec![data.len()],
        })?;
    if data.len() < need {
        return Err(ArchError::InvalidShape {
            name: name.to_string(),
            expected: vec![need],
            got: vec![data.len()],
        });
    }

    let mut out = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let view =
            data.slice(e * expert_bytes, expert_bytes)
                .ok_or_else(|| ArchError::InvalidShape {
                    name: format!("{name}.expert[{e}]"),
                    expected: vec![expert_bytes],
                    got: vec![data.len()],
                })?;
        out.push(QuantTensor::from_shared(
            view,
            vec![out_features, in_features],
            tensor_type,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatcher() -> KernelDispatcher {
        KernelDispatcher::new()
    }

    #[test]
    fn f32_round_trip() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequant_to_f32_slice("t", GgufTensorType::F32, &bytes, 4, &dispatcher())
            .expect("dequant");
        assert_eq!(out, values);
    }

    /// A truncated F32 payload used to be silently zero-padded.
    #[test]
    fn truncated_f32_is_reported() {
        let bytes = vec![0u8; 8]; // only 2 of the 4 declared elements
        let err = dequant_to_f32_slice("t", GgufTensorType::F32, &bytes, 4, &dispatcher());
        assert!(err.is_err(), "truncated F32 must be rejected");
    }

    /// A truncated quantized payload used to panic with an out-of-range slice.
    #[test]
    fn truncated_quant_block_is_reported() {
        // Q8_0: 32 weights per 34-byte block.  Declare 64 weights (2 blocks)
        // but supply only one block.
        let bytes = vec![0u8; 34];
        let err = dequant_to_f32_slice(
            "blk.0.attn_q.weight",
            GgufTensorType::Q8_0,
            &bytes,
            64,
            &dispatcher(),
        );
        match err {
            Err(ArchError::InvalidShape { name, .. }) => {
                assert_eq!(name, "blk.0.attn_q.weight");
            }
            other => panic!("expected InvalidShape naming the tensor, got {other:?}"),
        }
    }

    #[test]
    fn truncated_f16_is_reported() {
        let bytes = vec![0u8; 2];
        assert!(dequant_to_f32_slice("t", GgufTensorType::F16, &bytes, 4, &dispatcher()).is_err());
    }

    #[test]
    fn q8_0_full_blocks_dequantize() {
        // One Q8_0 block: f16 scale then 32 int8 values.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        for i in 0..32i8 {
            bytes.push(i as u8);
        }
        let out = dequant_to_f32_slice("t", GgufTensorType::Q8_0, &bytes, 32, &dispatcher())
            .expect("dequant");
        assert_eq!(out.len(), 32);
        assert!((out[2] - 1.0).abs() < 1e-3, "got {}", out[2]);
    }
}
