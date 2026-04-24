//! Quantize-on-load: convert F16/F32 tensors to Q4_0 or Q8_0 while loading.
//!
//! This module provides inline Q4_0 and Q8_0 encoders that run inside the
//! `oxillama-gguf` crate itself, avoiding any dependency on `oxillama-quant`
//! (which would introduce a circular dependency in the workspace).
//!
//! ## Design
//!
//! [`GgufModel::load_with_quant_plan`] loads a GGUF file then, for every
//! tensor matching the [`QuantPlan`], applies the specified quantization.
//! Quantized tensor data is stored in an in-memory override map so that
//! subsequent calls to [`GgufModel::tensor_data`] transparently return the
//! quantized bytes without any changes to the on-disk file.
//!
//! ## Supported conversions
//!
//! | Source dtype | Target [`QuantTarget`] | Supported |
//! |---|---|---|
//! | `F16` | `Q4_0` | ✓ |
//! | `F16` | `Q8_0` | ✓ |
//! | `F32` | `Q4_0` | ✓ |
//! | `F32` | `Q8_0` | ✓ |
//!
//! Any tensor whose source type is already quantized (not F16 or F32) is
//! rejected with [`GgufError::CannotRequantize`].
//!
//! ## Q4_0 block layout (18 bytes / 32 weights)
//!
//! ```text
//! [ scale: f16 (2 bytes) | packed nibbles: 16 bytes ]
//! ```
//!
//! Each nibble stores a weight value in [0, 15], where the dequantized value
//! is `(nibble - 8) * scale`.
//!
//! ## Q8_0 block layout (34 bytes / 32 weights)
//!
//! ```text
//! [ scale: f16 (2 bytes) | quantized: 32 × i8 ]
//! ```

use std::collections::HashMap;
use std::path::Path;

use half::f16;

use crate::error::{GgufError, GgufResult};
use crate::loader::GgufModel;
use crate::types::GgufTensorType;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Target quantization format for on-load quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTarget {
    /// 4-bit quantization with 1 f16 scale per 32-weight block (18 bytes/block).
    Q4_0,
    /// 8-bit quantization with 1 f16 scale per 32-weight block (34 bytes/block).
    Q8_0,
}

impl QuantTarget {
    /// Returns the corresponding [`GgufTensorType`] for this target.
    pub fn as_tensor_type(self) -> GgufTensorType {
        match self {
            QuantTarget::Q4_0 => GgufTensorType::Q4_0,
            QuantTarget::Q8_0 => GgufTensorType::Q8_0,
        }
    }

    /// Human-readable name for error messages.
    pub fn name(self) -> &'static str {
        match self {
            QuantTarget::Q4_0 => "Q4_0",
            QuantTarget::Q8_0 => "Q8_0",
        }
    }
}

/// Quantization plan governing which tensors to quantize and into which format.
///
/// Build via [`QuantPlan::new`] then optionally add per-tensor overrides via
/// [`QuantPlan::with_override`].
#[derive(Debug, Clone)]
pub struct QuantPlan {
    /// Default target for all F16/F32 tensors not mentioned in `overrides`.
    ///
    /// If `None`, only tensors listed in `overrides` are quantized.
    pub default: Option<QuantTarget>,
    /// Per-tensor target overrides, keyed by exact tensor name.
    pub overrides: HashMap<String, QuantTarget>,
}

impl QuantPlan {
    /// Create a plan that applies `target` to every compatible tensor.
    pub fn uniform(target: QuantTarget) -> Self {
        Self {
            default: Some(target),
            overrides: HashMap::new(),
        }
    }

    /// Create an empty plan (no quantization unless overrides are added).
    pub fn new() -> Self {
        Self {
            default: None,
            overrides: HashMap::new(),
        }
    }

    /// Add a per-tensor override.
    ///
    /// If `name` is also matched by the default target, the override wins.
    pub fn with_override(mut self, name: impl Into<String>, target: QuantTarget) -> Self {
        self.overrides.insert(name.into(), target);
        self
    }

    /// Resolve the target for a given tensor name and source type.
    ///
    /// Returns `None` if the tensor should not be quantized.
    fn target_for(&self, name: &str) -> Option<QuantTarget> {
        if let Some(t) = self.overrides.get(name) {
            return Some(*t);
        }
        self.default
    }
}

impl Default for QuantPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline quantization kernels (no oxillama-quant dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Quantize a slice of `f32` values to Q4_0 blocks.
///
/// Inputs must be a multiple of 32 in length (zero-padded otherwise).
///
/// Each block is 18 bytes: 2-byte f16 scale + 16 bytes of packed nibbles.
/// The mapping is `quant = round(weight / scale) + 8` clamped to `[0, 15]`.
fn encode_q4_0(weights: &[f32]) -> Vec<u8> {
    const BLOCK: usize = 32;

    let n_blocks = weights.len().div_ceil(BLOCK);
    let mut out = Vec::with_capacity(n_blocks * 18);

    for b in 0..n_blocks {
        let start = b * BLOCK;
        let end = (start + BLOCK).min(weights.len());
        let block = &weights[start..end];

        // Compute max absolute value for scale derivation
        let amax = block
            .iter()
            .copied()
            .fold(0.0f32, |acc, w| acc.max(w.abs()));

        // Scale: maps the full block range to [-8, 7] (offset by 8 for nibble storage)
        let scale = if amax == 0.0 { 0.0 } else { amax / 7.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write scale as f16
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Pack 32 nibbles into 16 bytes (two nibbles per byte, low nibble first)
        let mut nibbles = [8u8; BLOCK]; // default to 0-centered (offset 8 = zero)
        for (i, &w) in block.iter().enumerate() {
            let q = (w * inv_scale).round() as i32;
            let q_clamped = (q + 8).clamp(0, 15) as u8;
            nibbles[i] = q_clamped;
        }

        // Pack pairs of nibbles
        for pair in nibbles.chunks(2) {
            let lo = pair[0] & 0x0F;
            let hi = if pair.len() > 1 { pair[1] & 0x0F } else { 8 };
            out.push(lo | (hi << 4));
        }
    }

    out
}

/// Quantize a slice of `f32` values to Q8_0 blocks.
///
/// Each block is 34 bytes: 2-byte f16 scale + 32 bytes of i8 values.
/// The mapping is `quant = round(weight / scale)` clamped to `[-127, 127]`.
fn encode_q8_0(weights: &[f32]) -> Vec<u8> {
    const BLOCK: usize = 32;

    let n_blocks = weights.len().div_ceil(BLOCK);
    let mut out = Vec::with_capacity(n_blocks * 34);

    for b in 0..n_blocks {
        let start = b * BLOCK;
        let end = (start + BLOCK).min(weights.len());
        let block = &weights[start..end];

        let amax = block
            .iter()
            .copied()
            .fold(0.0f32, |acc, w| acc.max(w.abs()));

        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write scale as f16
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Write 32 quantized i8 values (padding with 0 if block is incomplete)
        let mut quants = [0i8; BLOCK];
        for (i, &w) in block.iter().enumerate() {
            let q = (w * inv_scale).round() as i32;
            quants[i] = q.clamp(-127, 127) as i8;
        }
        // SAFETY: transmuting [i8; 32] to [u8; 32] — bitwise identical
        out.extend_from_slice(&quants.map(|q| q as u8));
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// F16 / F32 dequantization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dequantize a raw F16 tensor blob to a `Vec<f32>`.
///
/// The blob must be even-length (2 bytes per f16 value).
fn dequant_f16(blob: &[u8]) -> GgufResult<Vec<f32>> {
    if blob.len() % 2 != 0 {
        return Err(GgufError::InvalidMetadata {
            key: "<f16_blob>".to_string(),
            reason: format!("blob length {} is not a multiple of 2", blob.len()),
        });
    }
    let count = blob.len() / 2;
    let mut out = Vec::with_capacity(count);
    for chunk in blob.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = f16::from_bits(bits);
        out.push(val.to_f32());
    }
    Ok(out)
}

/// Dequantize a raw F32 tensor blob to a `Vec<f32>`.
fn dequant_f32(blob: &[u8]) -> GgufResult<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return Err(GgufError::InvalidMetadata {
            key: "<f32_blob>".to_string(),
            reason: format!("blob length {} is not a multiple of 4", blob.len()),
        });
    }
    let count = blob.len() / 4;
    let mut out = Vec::with_capacity(count);
    for chunk in blob.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(val);
    }
    Ok(out)
}

/// Convert a raw tensor blob (F16 or F32) to `Vec<f32>`.
fn blob_to_f32(blob: &[u8], source_type: GgufTensorType) -> GgufResult<Vec<f32>> {
    match source_type {
        GgufTensorType::F16 => dequant_f16(blob),
        GgufTensorType::F32 => dequant_f32(blob),
        other => Err(GgufError::CannotRequantize {
            name: "<tensor>".to_string(),
            existing: format!("{other:?}"),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The quantized override store embedded inside GgufModel
// ─────────────────────────────────────────────────────────────────────────────

/// Per-tensor quantization override — stores the quantized bytes and the new dtype.
#[derive(Debug)]
pub(crate) struct QuantOverride {
    pub(crate) data: Vec<u8>,
    pub(crate) new_type: GgufTensorType,
}

/// Mutable map from tensor name → quantized bytes + new dtype.
pub(crate) type OverrideMap = HashMap<String, QuantOverride>;

// ─────────────────────────────────────────────────────────────────────────────
// GgufModel extension methods
// ─────────────────────────────────────────────────────────────────────────────

impl GgufModel {
    /// Load a GGUF model and apply on-the-fly quantization according to `plan`.
    ///
    /// For each F16/F32 tensor whose name matches `plan`, the tensor blob is
    /// dequantized to `f32`, re-quantized to the target format, and stored in
    /// an in-memory override map.  Subsequent [`GgufModel::tensor_data`] calls
    /// for those tensors return the quantized bytes transparently.
    ///
    /// Tensors that are already in a quantized format are skipped if they are
    /// not in the plan, or rejected with [`GgufError::CannotRequantize`] if
    /// the plan explicitly targets them.
    ///
    /// # Errors
    ///
    /// - [`GgufError::CannotRequantize`] — tensor is already quantized.
    /// - Any IO / parse errors from the underlying loader.
    pub fn load_with_quant_plan(path: impl AsRef<Path>, plan: &QuantPlan) -> GgufResult<Self> {
        let mut model = GgufModel::load(path)?;
        apply_quant_plan(&mut model, plan)?;
        Ok(model)
    }

    /// Apply a quantization plan to an already-loaded model.
    ///
    /// Same semantics as [`GgufModel::load_with_quant_plan`] but for models already in
    /// memory (e.g., from [`GgufModel::from_bytes`]).
    pub fn apply_quant_plan(&mut self, plan: &QuantPlan) -> GgufResult<()> {
        apply_quant_plan(self, plan)
    }
}

/// Core quantization logic — mutates the model by populating `quant_overrides`.
pub(crate) fn apply_quant_plan(model: &mut GgufModel, plan: &QuantPlan) -> GgufResult<()> {
    // Collect the list of tensors to quantize to avoid borrowing conflicts
    let to_quantize: Vec<(String, GgufTensorType, QuantTarget)> = model
        .file
        .tensors
        .iter()
        .filter_map(|(name, info)| {
            let target = plan.target_for(name)?;
            Some((name.clone(), info.tensor_type, target))
        })
        .collect();

    for (name, source_type, target) in to_quantize {
        // Reject re-quantization of already-quantized tensors
        match source_type {
            GgufTensorType::F16 | GgufTensorType::F32 => {} // allowed
            other => {
                return Err(GgufError::CannotRequantize {
                    name: name.clone(),
                    existing: format!("{other:?}"),
                });
            }
        }

        // If there's already an override for this tensor, respect it
        if model.quant_overrides.contains_key(&name) {
            let existing_type = model.quant_overrides[&name].new_type;
            return Err(GgufError::CannotRequantize {
                name: name.clone(),
                existing: format!("{existing_type:?}"),
            });
        }

        // Get the raw tensor bytes from the original data
        let blob = model.file.tensor_data(model.raw_data(), &name)?.to_vec();

        // Dequantize to f32
        let f32_weights = blob_to_f32(&blob, source_type)?;

        // Quantize to target
        let quantized = match target {
            QuantTarget::Q4_0 => encode_q4_0(&f32_weights),
            QuantTarget::Q8_0 => encode_q8_0(&f32_weights),
        };

        let new_tensor_type = target.as_tensor_type();
        // Keep TensorStore in sync so that downstream readers of
        // `tensor_info.tensor_type` see the post-quantization format, not the
        // original F16/F32 format.
        model.file.tensors.set_type(&name, new_tensor_type);
        model.quant_overrides.insert(
            name,
            QuantOverride {
                data: quantized,
                new_type: new_tensor_type,
            },
        );
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for the inline kernels
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Q8_0 encoder tests ──────────────────────────────────────────────────

    #[test]
    fn q8_0_block_layout_is_34_bytes() {
        let weights = vec![0.0f32; 32];
        let encoded = encode_q8_0(&weights);
        assert_eq!(encoded.len(), 34, "Q8_0: one block = 34 bytes");
    }

    #[test]
    fn q8_0_two_blocks_are_68_bytes() {
        let weights = vec![1.0f32; 64];
        let encoded = encode_q8_0(&weights);
        assert_eq!(encoded.len(), 68, "Q8_0: two blocks = 68 bytes");
    }

    #[test]
    fn q8_0_zero_block_has_zero_scale_and_zero_quants() {
        let weights = vec![0.0f32; 32];
        let encoded = encode_q8_0(&weights);
        // scale bytes (first 2) = f16(0.0) = 0x0000
        assert_eq!(encoded[0], 0);
        assert_eq!(encoded[1], 0);
        // all quant bytes should be 0
        for b in &encoded[2..] {
            assert_eq!(*b, 0, "zero weights should produce zero quants");
        }
    }

    #[test]
    fn q8_0_uniform_positive_block() {
        // All weights = 1.0 → scale = 1.0/127, quants all = 127
        let weights = vec![1.0f32; 32];
        let encoded = encode_q8_0(&weights);
        // scale = f16(1.0/127) — check it's non-zero
        let scale_bits = u16::from_le_bytes([encoded[0], encoded[1]]);
        assert_ne!(scale_bits, 0, "scale should be non-zero");
        // all quants should be 127
        for b in &encoded[2..] {
            assert_eq!(*b as i8, 127, "all quants should be +127");
        }
    }

    #[test]
    fn q8_0_partial_block_gets_padded() {
        // 17 weights — should produce one 34-byte block, padding with 0
        let weights = vec![2.0f32; 17];
        let encoded = encode_q8_0(&weights);
        assert_eq!(
            encoded.len(),
            34,
            "partial block should still produce one block"
        );
    }

    // ── Q4_0 encoder tests ──────────────────────────────────────────────────

    #[test]
    fn q4_0_block_layout_is_18_bytes() {
        let weights = vec![0.0f32; 32];
        let encoded = encode_q4_0(&weights);
        assert_eq!(encoded.len(), 18, "Q4_0: one block = 18 bytes");
    }

    #[test]
    fn q4_0_two_blocks_are_36_bytes() {
        let weights = vec![1.0f32; 64];
        let encoded = encode_q4_0(&weights);
        assert_eq!(encoded.len(), 36, "Q4_0: two blocks = 36 bytes");
    }

    #[test]
    fn q4_0_zero_block_has_zero_scale() {
        let weights = vec![0.0f32; 32];
        let encoded = encode_q4_0(&weights);
        let scale_bits = u16::from_le_bytes([encoded[0], encoded[1]]);
        assert_eq!(scale_bits, 0, "zero weights should produce zero scale");
    }

    #[test]
    fn q4_0_partial_block_produces_one_block() {
        let weights = vec![1.0f32; 7];
        let encoded = encode_q4_0(&weights);
        assert_eq!(
            encoded.len(),
            18,
            "partial block should still produce one 18-byte block"
        );
    }

    // ── F16 / F32 dequantization tests ──────────────────────────────────────

    #[test]
    fn dequant_f32_roundtrips_single_value() {
        let val = std::f32::consts::PI;
        let bytes = val.to_le_bytes().to_vec();
        let result = dequant_f32(&bytes).expect("test: dequant f32");
        assert!((result[0] - val).abs() < 1e-6);
    }

    #[test]
    fn dequant_f16_zero_is_zero() {
        let zero_f16 = f16::ZERO;
        let bytes = zero_f16.to_le_bytes().to_vec();
        let result = dequant_f16(&bytes).expect("test: dequant f16");
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn dequant_f16_rejects_odd_length() {
        let bytes = vec![0u8; 3]; // odd length
        let result = dequant_f16(&bytes);
        assert!(result.is_err(), "odd blob length should be rejected");
    }

    #[test]
    fn dequant_f32_rejects_non_multiple_of_4() {
        let bytes = vec![0u8; 5];
        let result = dequant_f32(&bytes);
        assert!(
            result.is_err(),
            "non-multiple-of-4 blob length should be rejected"
        );
    }

    // ── QuantPlan tests ──────────────────────────────────────────────────────

    #[test]
    fn quant_plan_uniform_applies_to_all() {
        let plan = QuantPlan::uniform(QuantTarget::Q8_0);
        assert_eq!(plan.target_for("any.tensor"), Some(QuantTarget::Q8_0));
        assert_eq!(plan.target_for("other.weight"), Some(QuantTarget::Q8_0));
    }

    #[test]
    fn quant_plan_override_wins_over_default() {
        let plan = QuantPlan::uniform(QuantTarget::Q8_0)
            .with_override("special.weight", QuantTarget::Q4_0);
        assert_eq!(plan.target_for("special.weight"), Some(QuantTarget::Q4_0));
        assert_eq!(plan.target_for("normal.weight"), Some(QuantTarget::Q8_0));
    }

    #[test]
    fn quant_plan_empty_returns_none_without_default() {
        let plan = QuantPlan::new();
        assert_eq!(plan.target_for("any.weight"), None);
    }

    #[test]
    fn quant_plan_override_only_no_default() {
        let plan = QuantPlan::new().with_override("embed.weight", QuantTarget::Q4_0);
        assert_eq!(plan.target_for("embed.weight"), Some(QuantTarget::Q4_0));
        assert_eq!(plan.target_for("other.weight"), None);
    }

    // ── Integration tests with GgufModel ────────────────────────────────────

    /// Build a minimal GGUF with a single F16 tensor.
    fn build_f16_gguf(n_weights: u64) -> Vec<u8> {
        use crate::types::{GgufValueType, GGUF_MAGIC};

        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        // KV: general.architecture = "llama"
        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        // Tensor: "embed.weight", F16, shape [n_weights], offset 0
        let tname = b"embed.weight";
        data.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        data.extend_from_slice(tname);
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&n_weights.to_le_bytes()); // dim[0]
        data.extend_from_slice(&(GgufTensorType::F16 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Pad to 32-byte alignment
        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }

        // F16 tensor data: n_weights × 2 bytes, all set to f16(1.0) = 0x3C00
        for _ in 0..n_weights {
            data.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        }

        data
    }

    /// Build a minimal GGUF with a single F32 tensor.
    fn build_f32_gguf(n_weights: u64) -> Vec<u8> {
        use crate::types::{GgufValueType, GGUF_MAGIC};

        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 KV

        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        let tname = b"embed.weight";
        data.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        data.extend_from_slice(tname);
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&n_weights.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }

        for _ in 0..n_weights {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }

        data
    }

    /// Build a minimal GGUF with a pre-quantized Q8_0 tensor.
    fn build_q8_gguf() -> Vec<u8> {
        use crate::types::{GgufValueType, GGUF_MAGIC};
        const N: u64 = 32;

        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        let tname = b"embed.weight";
        data.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        data.extend_from_slice(tname);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&N.to_le_bytes());
        data.extend_from_slice(&(GgufTensorType::Q8_0 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let align = 32usize;
        let rem = data.len() % align;
        if rem != 0 {
            data.resize(data.len() + align - rem, 0u8);
        }
        // Q8_0: 34 bytes per block
        data.resize(data.len() + 34, 0x00);

        data
    }

    #[test]
    fn quantize_on_load_f16_to_q4_0() {
        let raw = build_f16_gguf(32);
        let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

        let plan = QuantPlan::uniform(QuantTarget::Q4_0);
        model.apply_quant_plan(&plan).expect("test: apply plan");

        // The override should now be set
        let override_data = model
            .tensor_data("embed.weight")
            .expect("test: tensor_data");
        // Q4_0 for 32 weights = 18 bytes
        assert_eq!(
            override_data.len(),
            18,
            "Q4_0 quantized 32 weights = 18 bytes"
        );

        // TensorStore must reflect the new type — critical for downstream consumers
        let info = model
            .file
            .tensors
            .get("embed.weight")
            .expect("test: get info");
        assert_eq!(
            info.tensor_type,
            GgufTensorType::Q4_0,
            "tensor_type in TensorStore should be updated to Q4_0"
        );
    }

    #[test]
    fn quantize_on_load_f16_to_q8_0() {
        let raw = build_f16_gguf(32);
        let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

        let plan = QuantPlan::uniform(QuantTarget::Q8_0);
        model.apply_quant_plan(&plan).expect("test: apply plan");

        let override_data = model
            .tensor_data("embed.weight")
            .expect("test: tensor_data");
        // Q8_0 for 32 weights = 34 bytes
        assert_eq!(
            override_data.len(),
            34,
            "Q8_0 quantized 32 weights = 34 bytes"
        );

        // TensorStore must reflect the new type
        let info = model
            .file
            .tensors
            .get("embed.weight")
            .expect("test: get info");
        assert_eq!(
            info.tensor_type,
            GgufTensorType::Q8_0,
            "tensor_type in TensorStore should be updated to Q8_0"
        );
    }

    #[test]
    fn quantize_on_load_f32_to_q4_0() {
        let raw = build_f32_gguf(64); // 2 blocks
        let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

        let plan = QuantPlan::uniform(QuantTarget::Q4_0);
        model.apply_quant_plan(&plan).expect("test: apply plan");

        let override_data = model
            .tensor_data("embed.weight")
            .expect("test: tensor_data");
        // Q4_0 for 64 weights = 2 blocks × 18 bytes = 36 bytes
        assert_eq!(override_data.len(), 36);

        // TensorStore must reflect the new type
        let info = model
            .file
            .tensors
            .get("embed.weight")
            .expect("test: get info");
        assert_eq!(
            info.tensor_type,
            GgufTensorType::Q4_0,
            "tensor_type in TensorStore should be updated to Q4_0 after F32 quantization"
        );
    }

    #[test]
    fn quantize_on_load_rejects_requantize() {
        let raw = build_q8_gguf();
        let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

        // Trying to Q8_0-quantize an already-Q8_0 tensor should fail
        let plan = QuantPlan::uniform(QuantTarget::Q4_0);
        let result = model.apply_quant_plan(&plan);
        assert!(
            matches!(result, Err(GgufError::CannotRequantize { .. })),
            "re-quantizing Q8_0 tensor should return CannotRequantize"
        );
    }

    #[test]
    fn quantize_on_load_override_per_tensor() {
        let raw = build_f16_gguf(32);
        let mut model = GgufModel::from_bytes(raw).expect("test: from_bytes");

        // Default = Q8_0, but embed.weight override = Q4_0
        let plan =
            QuantPlan::uniform(QuantTarget::Q8_0).with_override("embed.weight", QuantTarget::Q4_0);
        model.apply_quant_plan(&plan).expect("test: apply plan");

        // embed.weight should be Q4_0 (18 bytes) not Q8_0 (34 bytes)
        let data = model
            .tensor_data("embed.weight")
            .expect("test: tensor_data");
        assert_eq!(data.len(), 18, "override Q4_0 for 32 weights = 18 bytes");
    }

    #[test]
    fn load_with_quant_plan_works_from_temp_file() {
        let raw = build_f32_gguf(32);
        let dir = tempfile::TempDir::new().expect("test: tempdir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, &raw).expect("test: write");

        let plan = QuantPlan::uniform(QuantTarget::Q8_0);
        let model =
            GgufModel::load_with_quant_plan(&path, &plan).expect("test: load_with_quant_plan");

        let data = model
            .tensor_data("embed.weight")
            .expect("test: tensor_data");
        assert_eq!(data.len(), 34, "Q8_0 for 32 weights = 34 bytes");
    }
}
