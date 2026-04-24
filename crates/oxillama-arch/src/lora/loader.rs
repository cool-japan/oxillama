//! LoRA adapter loading from GGUF files.
//!
//! LoRA adapter GGUF files follow the same binary format as model GGUF files,
//! but contain only the adapter weight matrices (A and B per adapted layer),
//! plus metadata keys for rank and alpha:
//!
//! | Key                  | Fallback key              | Type  | Description                   |
//! |----------------------|---------------------------|-------|-------------------------------|
//! | `lora.r`             | `adapter.lora.r`          | u32   | Low-rank dimension `r`        |
//! | `lora.alpha`         | `adapter.lora.alpha`      | f32   | LoRA alpha (scale numerator)  |
//!
//! ## Tensor naming convention (llama.cpp-compatible)
//!
//! Each adapted linear layer provides two tensors:
//!
//! ```text
//! blk.{i}.attn_q.weight.lora_a      blk.{i}.attn_q.weight.lora_b
//! blk.{i}.attn_k.weight.lora_a      blk.{i}.attn_k.weight.lora_b
//! blk.{i}.attn_v.weight.lora_a      blk.{i}.attn_v.weight.lora_b
//! blk.{i}.attn_output.weight.lora_a blk.{i}.attn_output.weight.lora_b
//! blk.{i}.ffn_gate.weight.lora_a    blk.{i}.ffn_gate.weight.lora_b
//! blk.{i}.ffn_up.weight.lora_a      blk.{i}.ffn_up.weight.lora_b
//! blk.{i}.ffn_down.weight.lora_a    blk.{i}.ffn_down.weight.lora_b
//! ```
//!
//! A tensors have shape `[rank, in_features]`; B tensors have shape
//! `[out_features, rank]`.

use std::collections::HashMap;
use std::sync::Arc;

use oxillama_gguf::{GgufModel, GgufTensorType, TensorInfo};
use oxillama_quant::{KernelDispatcher, LoraAdapter, QuantError};

use crate::error::{ArchError, ArchResult};

pub(super) const LORA_A_SUFFIX: &str = ".lora_a";
pub(super) const LORA_B_SUFFIX: &str = ".lora_b";

/// A fully loaded LoRA adapter, mapping tensor base names to their adapters.
///
/// The key in [`adapters`](Self::adapters) is the tensor base name without the
/// `.lora_a` / `.lora_b` suffix (e.g. `"blk.0.attn_q.weight"`).
///
/// After loading, attach adapters to `QuantLinear` layers via
/// [`LoadedLora::get`] + `QuantLinear::set_lora`.
#[derive(Debug)]
pub struct LoadedLora {
    /// Map from tensor base name → LoRA adapter.
    pub adapters: HashMap<String, Arc<LoraAdapter>>,
    /// The LoRA rank used throughout this adapter file.
    pub rank: usize,
    /// The alpha value (scale numerator).
    pub alpha: f32,
}

impl LoadedLora {
    /// Load a LoRA adapter from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`ArchError::Gguf`] if the file cannot be parsed, or
    /// [`ArchError::Quant`] if dequantization of any adapter tensor fails.
    pub fn load(path: &str) -> ArchResult<Self> {
        let model = GgufModel::load(path)?;
        Self::from_gguf(&model)
    }

    /// Load a LoRA adapter from an already-loaded [`GgufModel`].
    ///
    /// This is the primary construction path, separated from file I/O for
    /// testability.
    pub fn from_gguf(model: &GgufModel) -> ArchResult<Self> {
        // --- Extract rank --------------------------------------------------
        let rank = model
            .file
            .metadata
            .get("lora.r")
            .and_then(|v| v.as_u32())
            .or_else(|| {
                model
                    .file
                    .metadata
                    .get("adapter.lora.r")
                    .and_then(|v| v.as_u32())
            })
            .map(|v| v as usize)
            .unwrap_or(8);

        // --- Extract alpha -------------------------------------------------
        let alpha = model
            .file
            .metadata
            .get("lora.alpha")
            .and_then(|v| v.as_f32())
            .or_else(|| {
                model
                    .file
                    .metadata
                    .get("adapter.lora.alpha")
                    .and_then(|v| v.as_f32())
            })
            .unwrap_or(rank as f32);

        let scale = alpha / rank.max(1) as f32;
        let dispatcher = KernelDispatcher::new();

        // --- Find and pair .lora_a / .lora_b tensors -----------------------
        let tensor_names: Vec<String> = model.file.tensors.names().cloned().collect();
        let mut adapters: HashMap<String, Arc<LoraAdapter>> = HashMap::new();

        for name in &tensor_names {
            if !name.ends_with(LORA_A_SUFFIX) {
                continue;
            }
            let base = &name[..name.len() - LORA_A_SUFFIX.len()];
            let b_name = format!("{base}{LORA_B_SUFFIX}");

            if !model.file.tensors.contains(&b_name) {
                tracing::warn!(
                    tensor = %name,
                    "LoRA tensor has no matching .lora_b partner; skipping"
                );
                continue;
            }

            let a_info = model
                .file
                .tensors
                .get(name)
                .map_err(|_| ArchError::MissingTensor { name: name.clone() })?;
            let a_data = model.tensor_data(name)?;
            let a_f32 = dequant_tensor_to_f32(a_info, a_data, &dispatcher)?;

            let (rank_actual, in_features) = shape_to_rank_in(a_info, rank, a_f32.len());

            let b_info = model
                .file
                .tensors
                .get(&b_name)
                .map_err(|_| ArchError::MissingTensor {
                    name: b_name.clone(),
                })?;
            let b_data = model.tensor_data(&b_name)?;
            let b_f32 = dequant_tensor_to_f32(b_info, b_data, &dispatcher)?;

            let out_features = b_f32.len().checked_div(rank_actual).unwrap_or(0);

            let adapter =
                LoraAdapter::new(a_f32, b_f32, rank_actual, scale, in_features, out_features)
                    .map_err(ArchError::Quant)?;

            adapters.insert(base.to_string(), Arc::new(adapter));
        }

        tracing::debug!(
            rank = rank,
            alpha = alpha,
            adapters = adapters.len(),
            "LoRA adapter loaded from GGUF"
        );

        Ok(Self {
            adapters,
            rank,
            alpha,
        })
    }

    /// Look up the adapter for a named linear weight tensor.
    ///
    /// Returns `None` if this LoRA file does not patch the named tensor.
    pub fn get(&self, tensor_name: &str) -> Option<Arc<LoraAdapter>> {
        self.adapters.get(tensor_name).cloned()
    }

    /// Number of adapted layers in this adapter file.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Infer `(rank_actual, in_features)` from a LoRA-A tensor's shape metadata.
fn shape_to_rank_in(info: &TensorInfo, hint_rank: usize, n_elements: usize) -> (usize, usize) {
    match info.dimensions.as_slice() {
        [in_f, r] => (*r as usize, *in_f as usize),
        [total] => {
            let r = hint_rank.max(1);
            let in_f = (*total as usize) / r;
            (r, in_f)
        }
        _ => {
            let r = hint_rank.max(1);
            (r, n_elements / r)
        }
    }
}

/// Dequantize arbitrary-typed tensor data to a Vec<f32>.
pub(crate) fn dequant_tensor_to_f32(
    info: &TensorInfo,
    data: &[u8],
    dispatcher: &KernelDispatcher,
) -> ArchResult<Vec<f32>> {
    let n_elements = info.n_elements() as usize;

    if info.tensor_type == GgufTensorType::F32 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(4).enumerate().take(n_elements) {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Ok(out);
    }

    if info.tensor_type == GgufTensorType::F16 {
        let mut out = vec![0.0f32; n_elements];
        for (i, chunk) in data.chunks_exact(2).enumerate().take(n_elements) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = half::f16::from_bits(bits).to_f32();
        }
        return Ok(out);
    }

    let kernel = dispatcher
        .get_kernel(info.tensor_type)
        .map_err(ArchError::Quant)?;
    let block_size = kernel.block_size();
    let block_bytes = kernel.block_bytes();

    if block_size == 0 || block_bytes == 0 {
        return Err(ArchError::Quant(QuantError::UnsupportedType {
            quant_type: format!("{:?}", info.tensor_type),
        }));
    }

    let n_blocks = n_elements.div_ceil(block_size);
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block_start = b * block_bytes;
        let out_start = b * block_size;
        let block_end = (block_start + block_bytes).min(data.len());
        let out_end = (out_start + block_size).min(n_elements);

        if block_end <= block_start {
            break;
        }

        kernel
            .dequant_block(&data[block_start..block_end], &mut out[out_start..out_end])
            .map_err(ArchError::Quant)?;
    }

    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loaded_lora_empty_construction() {
        let lora = LoadedLora {
            adapters: HashMap::new(),
            rank: 8,
            alpha: 8.0,
        };
        assert_eq!(lora.num_adapters(), 0);
        assert_eq!(lora.rank, 8);
        assert!(lora.get("blk.0.attn_q.weight").is_none());
    }

    #[test]
    fn test_shape_to_rank_in_2d() {
        let info = TensorInfo {
            name: "test.lora_a".into(),
            n_dims: 2,
            dimensions: vec![64, 8],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        let (r, in_f) = shape_to_rank_in(&info, 8, 64 * 8);
        assert_eq!(r, 8, "rank should be 8 (dims[1])");
        assert_eq!(in_f, 64, "in_features should be 64 (dims[0])");
    }

    #[test]
    fn test_shape_to_rank_in_1d() {
        let info = TensorInfo {
            name: "test.lora_a".into(),
            n_dims: 1,
            dimensions: vec![128],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        let (r, in_f) = shape_to_rank_in(&info, 8, 128);
        assert_eq!(r, 8);
        assert_eq!(in_f, 16);
    }

    #[test]
    fn test_get_missing() {
        let lora = LoadedLora {
            adapters: HashMap::new(),
            rank: 4,
            alpha: 4.0,
        };
        assert!(lora.get("blk.99.ffn_gate.weight").is_none());
    }

    #[test]
    fn test_get_present() {
        let adapter =
            Arc::new(LoraAdapter::new(vec![1.0], vec![1.0], 1, 1.0, 1, 1).expect("valid"));
        let mut adapters = HashMap::new();
        adapters.insert("blk.0.attn_q.weight".to_string(), adapter);

        let lora = LoadedLora {
            adapters,
            rank: 1,
            alpha: 1.0,
        };
        assert!(lora.get("blk.0.attn_q.weight").is_some());
    }

    #[test]
    fn test_loaded_lora_from_gguf_succeeds() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        assert!(lora.rank > 0, "rank must be positive");
        assert!(lora.alpha > 0.0, "alpha must be positive");
        assert!(!lora.adapters.is_empty(), "adapters map must not be empty");
    }

    #[test]
    fn test_loaded_lora_rank_matches_metadata() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        assert_eq!(lora.rank, 4, "rank should match lora.r=4 in synthetic GGUF");
    }

    #[test]
    fn test_loaded_lora_alpha_matches_metadata() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        assert!(
            (lora.alpha - 8.0).abs() < 1e-5,
            "alpha should match lora.alpha=8.0, got {}",
            lora.alpha
        );
    }

    #[test]
    fn test_loaded_lora_contains_expected_adapters() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        assert_eq!(
            lora.adapters.len(),
            3,
            "expected 3 lora adapters (attn_q, attn_v, ffn_gate), got {}",
            lora.adapters.len()
        );
    }

    #[test]
    fn test_loaded_lora_get_returns_adapter() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        let adapter = lora.get("blk.0.attn_q.weight");
        assert!(
            adapter.is_some(),
            "expected to find adapter for blk.0.attn_q.weight"
        );
    }

    #[test]
    fn test_loaded_lora_get_missing_returns_none() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        let adapter = lora.get("nonexistent_layer");
        assert!(adapter.is_none(), "nonexistent layer should return None");
    }

    #[test]
    fn test_loaded_lora_from_base_model_has_empty_adapters() {
        use oxillama_gguf::{test_utils::build_minimal_llama_gguf, GgufModel};
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse base model gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load from base model");
        assert!(
            lora.adapters.is_empty(),
            "base model (no .lora_a tensors) should yield zero adapters"
        );
    }

    #[test]
    fn test_loaded_lora_default_rank_when_missing() {
        use oxillama_gguf::{test_utils::build_minimal_llama_gguf, GgufModel};
        let bytes = build_minimal_llama_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse base model gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load from base model");
        assert_eq!(
            lora.rank, 8,
            "without lora.r key the default rank should be 8"
        );
    }

    #[test]
    fn test_loaded_lora_all_three_adapters_reachable() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        for layer_name in [
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_gate.weight",
        ] {
            assert!(
                lora.get(layer_name).is_some(),
                "adapter for '{layer_name}' must be reachable via get()"
            );
        }
    }

    #[test]
    fn test_loaded_lora_num_adapters_matches_len() {
        use oxillama_gguf::{test_utils::build_minimal_lora_gguf, GgufModel};
        let bytes = build_minimal_lora_gguf();
        let model = GgufModel::from_bytes(bytes).expect("test: parse lora gguf");
        let lora = LoadedLora::from_gguf(&model).expect("test: load lora from gguf");
        assert_eq!(
            lora.num_adapters(),
            lora.adapters.len(),
            "num_adapters() must equal adapters.len()"
        );
    }
}
