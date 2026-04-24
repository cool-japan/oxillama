//! Full-precision f32 weight loader for CI numeric diff testing.
//!
//! This module eagerly dequantizes all tensors in a GGUF model to f32.
//! It is **not for production use** — memory consumption is approximately
//! 4× that of a Q4_0-quantized model.

use std::collections::HashMap;

use oxillama_gguf::GgufModel;
use oxillama_quant::KernelDispatcher;

use crate::error::{ArchError, ArchResult};
use crate::lora::loader::dequant_tensor_to_f32;

/// All model weights dequantized to f32 at load time.
///
/// `tensors` maps tensor name → flat f32 data.
/// `shapes` maps tensor name → dimension list (slowest to fastest, matching
/// the GGUF dimension order: `[rows, cols]` for 2-D tensors).
pub struct ReferenceWeights {
    /// Tensor data as flat f32 vectors, keyed by name.
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes (GGUF dimension order), keyed by name.
    pub shapes: HashMap<String, Vec<usize>>,
}

/// Loader that eagerly dequantizes an entire GGUF model to f32.
pub struct ReferenceLoader;

impl ReferenceLoader {
    /// Dequantize every tensor in `model` to f32.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::Quant`] if any tensor cannot be dequantized, or
    /// [`ArchError::MissingTensor`] if tensor data is inaccessible.
    pub fn dequantize_all(model: &GgufModel) -> ArchResult<ReferenceWeights> {
        let dispatcher = KernelDispatcher::new();
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, info) in model.file.tensors.iter() {
            let data = model
                .tensor_data(name)
                .map_err(|_| ArchError::MissingTensor { name: name.clone() })?;

            let f32_data = dequant_tensor_to_f32(info, data, &dispatcher).map_err(|e| {
                ArchError::ForwardPassError {
                    layer: 0,
                    message: format!("dequantize tensor '{name}': {e}"),
                }
            })?;

            let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.insert(name.clone(), f32_data);
            shapes.insert(name.clone(), shape);
        }

        Ok(ReferenceWeights { tensors, shapes })
    }
}
