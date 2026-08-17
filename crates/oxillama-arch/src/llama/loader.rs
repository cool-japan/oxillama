//! GGUF loading for the LLaMA family.
//!
//! Split out of [`super::model`] so the forward pass and the weight plumbing can
//! grow independently and neither file approaches the 2000-line ceiling.
//!
//! Everything here is shared with the architectures that build on LLaMA:
//! `command_r`, `starcoder`, `falcon`, `minicpm` and `olmo2` import
//! [`load_quant_linear`] and [`load_rms_norm_weight`] through `crate::llama`,
//! and `llava`/`llava_next` build their language tower with
//! [`load_llama_from_gguf`].  The generic, bounds-checked loaders live in
//! [`crate::common::loader`]; prefer those for new code.

use std::sync::Arc;

use oxillama_quant::{KernelDispatcher, QuantKernel, QuantTensor};

use crate::common::embedding::TokenEmbedding;
use crate::common::linear::{gguf_linear_shape, QuantLinear};
use crate::common::loader::load_stacked_experts;
use crate::common::moe::{QuantExpert, QuantMoeFfn};
use crate::common::rms_norm::RmsNorm;
use crate::config::ModelConfig;
use crate::error::{ArchError, ArchResult};
use crate::llama::model::{DenseFfn, FfnVariant, LlamaLayer, LlamaModel};

/// Load a LLaMA model from a `GgufModel`.
///
/// Handles both the dense SwiGLU layout and the sparse MoE layout Mixtral uses:
/// llama.cpp converts `MixtralForCausalLM` to `MODEL_ARCH.LLAMA`
/// (`convert_hf_to_gguf.py`), so a Mixtral GGUF arrives here with
/// `expert_count > 0` rather than through `crate::mixtral`.
pub fn load_llama_from_gguf(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
) -> ArchResult<LlamaModel> {
    let dispatcher = KernelDispatcher::new();

    let token_embd = load_token_embedding(model, config, &dispatcher)?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm = load_rms_norm_weight(model, &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = load_rms_norm_weight(model, &format!("{prefix}.ffn_norm.weight"))?;

        let attn_q = load_quant_linear(model, &format!("{prefix}.attn_q.weight"))?;
        let attn_k = load_quant_linear(model, &format!("{prefix}.attn_k.weight"))?;
        let attn_v = load_quant_linear(model, &format!("{prefix}.attn_v.weight"))?;
        let attn_output = load_quant_linear(model, &format!("{prefix}.attn_output.weight"))?;

        let ffn = if config.num_experts > 0 {
            load_moe_ffn(model, &prefix, config)?
        } else {
            let gate = load_quant_linear(model, &format!("{prefix}.ffn_gate.weight"))?;
            let up = load_quant_linear(model, &format!("{prefix}.ffn_up.weight"))?;
            let down = load_quant_linear(model, &format!("{prefix}.ffn_down.weight"))?;
            FfnVariant::Dense(Box::new(DenseFfn {
                gate_kernel: resolve_kernel(&dispatcher, &gate)?,
                up_kernel: resolve_kernel(&dispatcher, &up)?,
                down_kernel: resolve_kernel(&dispatcher, &down)?,
                gate,
                up,
                down,
            }))
        };

        layers.push(LlamaLayer {
            attn_norm: RmsNorm::new(attn_norm, config.rms_norm_eps),
            attn_q_kernel: resolve_kernel(&dispatcher, &attn_q)?,
            attn_k_kernel: resolve_kernel(&dispatcher, &attn_k)?,
            attn_v_kernel: resolve_kernel(&dispatcher, &attn_v)?,
            attn_output_kernel: resolve_kernel(&dispatcher, &attn_output)?,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: RmsNorm::new(ffn_norm, config.rms_norm_eps),
            ffn,
        });
    }

    let output_norm_weight = load_rms_norm_weight(model, "output_norm.weight")?;
    let output_norm = RmsNorm::new(output_norm_weight, config.rms_norm_eps);

    let output = load_lm_head(model)?;

    LlamaModel::with_embedding(config.clone(), token_embd, layers, output_norm, output)
}

/// Resolve `linear`'s kernel once, wrapped for cheap sharing.
///
/// `dispatcher.get_kernel` walks the full tensor-type match ladder and returns a
/// fresh `Box<dyn QuantKernel>`.  The decode loop used to do that seven times
/// per layer plus once for the LM head — 225 dispatches and allocations per
/// token on a 32-layer model — so the lookup is hoisted to load time and stored
/// as an `Arc` (`impl From<Box<T>> for Arc<T>`, no extra copy) on the structure
/// that owns the weight.
pub(crate) fn resolve_kernel(
    dispatcher: &KernelDispatcher,
    linear: &QuantLinear,
) -> ArchResult<Arc<dyn QuantKernel>> {
    Ok(dispatcher.get_kernel(linear.weight.tensor_type)?.into())
}

/// Load `token_embd.weight`, keeping it quantized whenever possible.
///
/// Materialising this tensor as f32 was the single largest avoidable allocation
/// in the loader: Llama-3-8B's `[128256, 4096]` table is 525 M elements —
/// **2.10 GB** of anonymous f32 against ~295 MB of `Q4_K` on disk, i.e. more
/// than a third of the whole checkpoint spent on a matrix a forward pass reads
/// exactly one row of per token.
///
/// The table is therefore kept in its GGUF form (a `SharedBytes` view straight
/// into the backing store, so it costs nothing on top of it) and rows are
/// dequantized on lookup.  See [`TokenEmbedding`], which documents the
/// 6.736 GB → 2.684 GB RSS the same change bought for Qwen3-4B.
///
/// Checkpoints whose rows are not a whole number of quantization blocks fall
/// back to the bulk-dequantized [`TokenEmbedding::Dense`], so nothing that used
/// to load stops loading.
pub(crate) fn load_token_embedding(
    model: &oxillama_gguf::GgufModel,
    config: &ModelConfig,
    dispatcher: &KernelDispatcher,
) -> ArchResult<TokenEmbedding> {
    const NAME: &str = "token_embd.weight";

    let info = model
        .file
        .tensors
        .get(NAME)
        .map_err(|_| ArchError::MissingTensor {
            name: NAME.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    let n_elements = info.n_elements() as usize;
    let data = model.tensor_bytes(NAME)?;

    if let Some(embd) =
        TokenEmbedding::quantized(QuantTensor::from_shared(data.clone(), shape, tensor_type))
    {
        return Ok(embd);
    }

    let hidden = if config.hidden_size == 0 {
        n_elements
    } else {
        config.hidden_size
    };
    let dense = dequant_to_f32_slice(tensor_type, &data, n_elements, dispatcher)?;
    Ok(TokenEmbedding::dense(dense, hidden))
}

/// Load the LM head, falling back to the tied input embedding.
///
/// Llama-3.2-1B and Llama-3.2-3B — and the LLaVA builds on top of them — tie
/// their input and output embeddings: the GGUF ships `token_embd.weight` and no
/// `output.weight` at all.  Demanding `output.weight` unconditionally made those
/// checkpoints unloadable.  llama.cpp has the same fallback: it loads `output`
/// with `TENSOR_NOT_REQUIRED` and reuses `tok_embd` when it is absent
/// (`src/llama-model.cpp`, `LLM_ARCH_LLAMA`).
///
/// The reuse goes through [`load_quant_linear`], so the head stays quantized and
/// both views share one payload — tying costs no extra memory.
///
/// When neither tensor is present the error still names `output.weight`.
pub(crate) fn load_lm_head(model: &oxillama_gguf::GgufModel) -> ArchResult<QuantLinear> {
    if !model.file.tensors.contains("output.weight")
        && model.file.tensors.contains("token_embd.weight")
    {
        return load_quant_linear(model, "token_embd.weight");
    }
    load_quant_linear(model, "output.weight")
}

/// Load a MoE FFN for one transformer block from GGUF.
fn load_moe_ffn(
    model: &oxillama_gguf::GgufModel,
    prefix: &str,
    config: &ModelConfig,
) -> ArchResult<FfnVariant> {
    Ok(FfnVariant::Moe(Box::new(load_quant_moe(
        model,
        prefix,
        config.num_experts,
        config.num_experts_used.max(1),
    )?)))
}

/// Build one block's [`QuantMoeFfn`] from the stacked GGUF expert tensors.
///
/// Mixtral-style GGUF stores the whole expert pool as one 3-D tensor per
/// projection:
/// - `blk.{i}.ffn_gate_inp.weight`   — router: `[num_experts, hidden]`
/// - `blk.{i}.ffn_gate_exps.weight`  — `[num_experts, intermediate, hidden]`
/// - `blk.{i}.ffn_up_exps.weight`    — `[num_experts, intermediate, hidden]`
/// - `blk.{i}.ffn_down_exps.weight`  — `[num_experts, hidden, intermediate]`
///
/// [`load_stacked_experts`] hands back **shared views** over the same mapping
/// rather than copies, so a Mixtral-8x7B layer costs its quantized bytes once.
/// Dequantizing the experts to f32 — what this loader used to do — needs ~5.6 GB
/// per layer and made a real Mixtral unloadable on any machine.
///
/// Shared with `crate::mixtral`, which reads the identical tensor set: a
/// checkpoint declaring `general.architecture = "mixtral"` and one declaring
/// `"llama"` with `expert_count > 0` must produce the same weights.
pub(crate) fn load_quant_moe(
    model: &oxillama_gguf::GgufModel,
    prefix: &str,
    num_experts: usize,
    top_k: usize,
) -> ArchResult<QuantMoeFfn> {
    let router = load_quant_linear(model, &format!("{prefix}.ffn_gate_inp.weight"))?;

    let gate = load_stacked_experts(
        model,
        &format!("{prefix}.ffn_gate_exps.weight"),
        num_experts,
    )?;
    let up = load_stacked_experts(model, &format!("{prefix}.ffn_up_exps.weight"), num_experts)?;
    let down = load_stacked_experts(
        model,
        &format!("{prefix}.ffn_down_exps.weight"),
        num_experts,
    )?;

    let mut experts = Vec::with_capacity(num_experts);
    for ((g, u), d) in gate.into_iter().zip(up).zip(down) {
        experts.push(QuantExpert::new(
            QuantLinear::new(g, None),
            QuantLinear::new(u, None),
            QuantLinear::new(d, None),
        )?);
    }

    QuantMoeFfn::new(router, experts, top_k)
}

/// Load a quantized linear layer from GGUF.
///
/// The weight payload is taken as a [`SharedBytes`][oxillama_gguf::SharedBytes]
/// view, not a `to_vec()` copy: the GEMV kernels only ever read `&[u8]` out of
/// it, so a private copy would double the checkpoint's resident cost for
/// nothing.
pub(crate) fn load_quant_linear(
    model: &oxillama_gguf::GgufModel,
    name: &str,
) -> ArchResult<QuantLinear> {
    let info = model
        .file
        .tensors
        .get(name)
        .map_err(|_| ArchError::MissingTensor {
            name: name.to_string(),
        })?;
    let shape = gguf_linear_shape(&info.dimensions);
    let tensor_type = info.tensor_type;
    let data = model.tensor_bytes(name)?;
    let tensor = QuantTensor::from_shared(data, shape, tensor_type);

    Ok(QuantLinear::new(tensor, None))
}

/// Load an RMSNorm weight vector from GGUF (always dequantized to F32).
pub(crate) fn load_rms_norm_weight(
    model: &oxillama_gguf::GgufModel,
    name: &str,
) -> ArchResult<Vec<f32>> {
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

/// Dequantize tensor data to f32.
pub(crate) fn dequant_to_f32(
    info: &oxillama_gguf::TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    dequant_to_f32_slice(
        info.tensor_type,
        data,
        info.n_elements() as usize,
        dispatcher,
    )
}

/// Dequantize `n_elements` weights of `tensor_type` out of `data`.
///
/// `n_elements` comes from the tensor header, i.e. from the *file*, and the
/// payload length is checked against it before a single byte is read.  A
/// truncated or corrupt GGUF used to slice past the end of `data` here and abort
/// the process; it now returns [`ArchError::TensorShapeMismatch`].
pub(crate) fn dequant_to_f32_slice(
    tensor_type: oxillama_gguf::GgufTensorType,
    data: &[u8],
    n_elements: usize,
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    /// Reject a payload that cannot hold the declared element count.
    fn require(data_len: usize, needed: usize, what: &str) -> ArchResult<()> {
        if data_len < needed {
            return Err(ArchError::TensorShapeMismatch {
                tensor: what.to_string(),
                expected: vec![needed],
                got: vec![data_len],
            });
        }
        Ok(())
    }

    if tensor_type == oxillama_gguf::GgufTensorType::F32 {
        require(data.len(), n_elements * 4, "F32 tensor payload")?;
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if tensor_type == oxillama_gguf::GgufTensorType::F16 {
        require(data.len(), n_elements * 2, "F16 tensor payload")?;
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher.get_kernel(tensor_type)?;
    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();
    if block_size == 0 || block_bytes == 0 {
        return Err(ArchError::NotSupported {
            detail: format!("tensor type {tensor_type:?} has no block geometry"),
        });
    }
    let n_blocks = n_elements.div_ceil(block_size);
    let needed = n_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| ArchError::InvalidConfig {
            detail: format!("{tensor_type:?}: block count overflows"),
        })?;
    require(data.len(), needed, "quantized tensor payload")?;

    let mut out = vec![0.0f32; n_elements];
    for blk in 0..n_blocks {
        let data_offset = blk * block_bytes;
        let out_offset = blk * block_size;
        let block_data = &data[data_offset..data_offset + block_bytes];
        let out_slice = &mut out[out_offset..out_offset.saturating_add(block_size).min(n_elements)];
        kernel.dequant_block(block_data, out_slice)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::GgufTensorType;

    /// A truncated quantized payload must be reported, not indexed past.
    ///
    /// Before the length check this sliced `data[0..144]` out of a 16-byte
    /// buffer and aborted the process on any corrupt or partially downloaded
    /// checkpoint.
    #[test]
    fn truncated_quantized_payload_errors_instead_of_panicking() {
        let dispatcher = KernelDispatcher::new();
        // One Q4_K block is 144 bytes and covers 256 weights; supply 16.
        let err = dequant_to_f32_slice(GgufTensorType::Q4K, &[0u8; 16], 256, &dispatcher)
            .expect_err("a 16-byte payload cannot hold a 256-weight Q4_K tensor");
        assert!(
            matches!(err, ArchError::TensorShapeMismatch { .. }),
            "expected TensorShapeMismatch, got {err}"
        );
    }

    /// The same guard covers the float fast paths.
    #[test]
    fn truncated_float_payloads_error() {
        let dispatcher = KernelDispatcher::new();
        assert!(
            dequant_to_f32_slice(GgufTensorType::F32, &[0u8; 7], 4, &dispatcher).is_err(),
            "4 f32 need 16 bytes"
        );
        assert!(
            dequant_to_f32_slice(GgufTensorType::F16, &[0u8; 3], 4, &dispatcher).is_err(),
            "4 f16 need 8 bytes"
        );
    }

    /// An exactly-sized payload still decodes.
    #[test]
    fn exact_payload_still_decodes() {
        let dispatcher = KernelDispatcher::new();
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.0, 0.5, 7.25] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequant_to_f32_slice(GgufTensorType::F32, &bytes, 4, &dispatcher)
            .expect("exact payload must decode");
        assert_eq!(out, vec![1.0, -2.0, 0.5, 7.25]);
    }
}
