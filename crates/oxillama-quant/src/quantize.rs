//! Quantize-on-the-fly conversion utilities.
//!
//! Enables in-place re-quantization of GGUF checkpoints without
//! external tooling.  Provides public functions to convert FP32 or
//! FP16 weight buffers into quantized block formats (Q4_0, Q8_0)
//! and a generic dequantization helper for any supported type.

use half::f16;
use oxillama_gguf::GgufTensorType;

use crate::dispatch::KernelDispatcher;
use crate::error::{QuantError, QuantResult};

// ── Constants ────────────────────────────────────────────────────────

/// Weights per block for Q4_0 / Q8_0.
const BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (FP16 scale) + 16 (nibble data).
const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per Q8_0 block: 2 (FP16 scale) + 32 (int8 data).
const Q8_0_BLOCK_BYTES: usize = 34;

// ── Public API ───────────────────────────────────────────────────────

/// Quantize FP32 values to Q4_0 format.
///
/// Q4_0 block layout (18 bytes per 32 weights):
///   - 2 bytes: FP16 scale `d`
///   - 16 bytes: 32 unsigned 4-bit nibbles packed pair-wise
///
/// Returns an error when `data.len()` is not a multiple of 32.
pub fn quantize_f32_to_q4_0(data: &[f32]) -> QuantResult<Vec<u8>> {
    if !data.len().is_multiple_of(BLOCK_SIZE) {
        return Err(QuantError::DimensionMismatch {
            expected: (data.len() / BLOCK_SIZE + 1) * BLOCK_SIZE,
            got: data.len(),
        });
    }

    let n_blocks = data.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);

    for blk_idx in 0..n_blocks {
        let blk = &data[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        encode_q4_0_block(blk, &mut out);
    }

    Ok(out)
}

/// Quantize FP32 values to Q8_0 format.
///
/// Q8_0 block layout (34 bytes per 32 weights):
///   - 2 bytes: FP16 scale `d`
///   - 32 bytes: 32 int8 quantized values
///
/// Returns an error when `data.len()` is not a multiple of 32.
pub fn quantize_f32_to_q8_0(data: &[f32]) -> QuantResult<Vec<u8>> {
    if !data.len().is_multiple_of(BLOCK_SIZE) {
        return Err(QuantError::DimensionMismatch {
            expected: (data.len() / BLOCK_SIZE + 1) * BLOCK_SIZE,
            got: data.len(),
        });
    }

    let n_blocks = data.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);

    for blk_idx in 0..n_blocks {
        let blk = &data[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        encode_q8_0_block(blk, &mut out);
    }

    Ok(out)
}

/// Quantize FP16 values (as raw `u16` bits) to Q4_0 format.
///
/// Converts each half-precision value to FP32, then delegates to
/// [`quantize_f32_to_q4_0`].
pub fn quantize_f16_to_q4_0(data: &[u16]) -> QuantResult<Vec<u8>> {
    let f32_data: Vec<f32> = data.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    quantize_f32_to_q4_0(&f32_data)
}

/// Quantize FP16 values (as raw `u16` bits) to Q8_0 format.
///
/// Converts each half-precision value to FP32, then delegates to
/// [`quantize_f32_to_q8_0`].
pub fn quantize_f16_to_q8_0(data: &[u16]) -> QuantResult<Vec<u8>> {
    let f32_data: Vec<f32> = data.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    quantize_f32_to_q8_0(&f32_data)
}

/// Generic dequantization: raw quantized bytes → FP32.
///
/// Uses the [`KernelDispatcher`] to select the best available kernel
/// for `tensor_type`, then dequantizes `n_elements` values.
///
/// # Errors
///
/// * [`QuantError::UnsupportedType`] if no kernel exists for `tensor_type`.
/// * [`QuantError::BlockCountMismatch`] if the byte buffer does not contain
///   the expected number of complete blocks.
pub fn dequantize_to_f32(
    data: &[u8],
    tensor_type: GgufTensorType,
    n_elements: usize,
) -> QuantResult<Vec<f32>> {
    let dispatcher = KernelDispatcher::new();
    let kernel = dispatcher.get_kernel(tensor_type)?;

    let block_size = tensor_type.block_size();
    let block_bytes = tensor_type.block_bytes();

    if n_elements == 0 {
        return Ok(Vec::new());
    }

    let n_blocks = n_elements.div_ceil(block_size);
    let expected_bytes = n_blocks * block_bytes;

    if data.len() < expected_bytes {
        return Err(QuantError::BufferTooSmall {
            needed: expected_bytes,
            available: data.len(),
        });
    }

    let mut output = vec![0.0f32; n_blocks * block_size];

    for blk_idx in 0..n_blocks {
        let byte_offset = blk_idx * block_bytes;
        let block = &data[byte_offset..byte_offset + block_bytes];
        let out_offset = blk_idx * block_size;
        kernel.dequant_block(block, &mut output[out_offset..out_offset + block_size])?;
    }

    // Trim to the exact number of requested elements
    output.truncate(n_elements);
    Ok(output)
}

// ── Block encoders (private) ─────────────────────────────────────────

/// Encode 32 FP32 values into one Q4_0 block, appending to `out`.
///
/// Algorithm (matches llama.cpp `quantize_row_q4_0_reference`):
///   1. Find the element with the largest absolute value → `max` (keeps sign)
///   2. `d = max / -8` (maps that extreme to nibble 0)
///   3. `id = 1/d` (or 0 when `d == 0`)
///   4. `q_i = clamp(round(x_i * id) + 8, 0, 15)`
///   5. Pack pairs as `lo | (hi << 4)`
fn encode_q4_0_block(values: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK_SIZE);

    // 1. Find element with largest absolute value, keeping sign.
    let max_val = values
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| if v.abs() > acc.abs() { v } else { acc });

    // 2-3. Scale: d = max / -8, so that max maps to nibble 0 exactly
    let d = max_val / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    // Encode d as FP16
    let d_fp16 = f16::from_f32(d);
    out.extend_from_slice(&d_fp16.to_bits().to_le_bytes());

    // 4-5. Quantize and pack nibble pairs
    for pair in 0..BLOCK_SIZE / 2 {
        let x0 = values[pair * 2];
        let x1 = values[pair * 2 + 1];

        let q0 = ((x0 * id + 8.5) as i32).clamp(0, 15) as u8;
        let q1 = ((x1 * id + 8.5) as i32).clamp(0, 15) as u8;

        out.push(q0 | (q1 << 4));
    }
}

/// Encode 32 FP32 values into one Q8_0 block, appending to `out`.
///
/// Algorithm (matches llama.cpp `quantize_row_q8_0_ref`):
///   1. `amax` = max absolute value in the block
///   2. `d = amax / 127.0`
///   3. `id = 1/d` (or 0)
///   4. `q_i = clamp(round(x_i * id), -128, 127)` stored as int8
fn encode_q8_0_block(values: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK_SIZE);

    // 1. Max absolute value
    let amax = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

    // 2-3. Scale
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    // Encode d as FP16
    let d_fp16 = f16::from_f32(d);
    out.extend_from_slice(&d_fp16.to_bits().to_le_bytes());

    // 4. Quantize each weight to int8
    for &x in values {
        let q = (x * id).round() as i32;
        let q_clamped = q.clamp(-128, 127) as i8;
        out.push(q_clamped as u8);
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: maximum absolute error between two slices.
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    // ── Q4_0 round-trip ──────────────────────────────────────────────

    #[test]
    fn q4_0_round_trip_small_values() {
        // 32 linearly spaced values in [-1.0, 1.0]
        let data: Vec<f32> = (0..32).map(|i| (i as f32 / 15.5) - 1.0).collect();

        let quantized = quantize_f32_to_q4_0(&data).expect("quantize failed");
        assert_eq!(quantized.len(), Q4_0_BLOCK_BYTES);

        let restored =
            dequantize_to_f32(&quantized, GgufTensorType::Q4_0, 32).expect("dequantize failed");
        assert_eq!(restored.len(), 32);

        // Q4_0 has only 16 levels; error can be up to ~d (scale / 15).
        let err = max_abs_error(&data, &restored);
        assert!(err < 0.15, "Q4_0 round-trip error too large: {err}");
    }

    #[test]
    fn q4_0_round_trip_multiple_blocks() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();

        let quantized = quantize_f32_to_q4_0(&data).expect("quantize failed");
        assert_eq!(quantized.len(), 4 * Q4_0_BLOCK_BYTES);

        let restored =
            dequantize_to_f32(&quantized, GgufTensorType::Q4_0, 128).expect("dequantize failed");
        assert_eq!(restored.len(), 128);

        let err = max_abs_error(&data, &restored);
        assert!(err < 0.15, "Q4_0 multi-block error: {err}");
    }

    // ── Q8_0 round-trip ──────────────────────────────────────────────

    #[test]
    fn q8_0_round_trip_small_values() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 / 15.5) - 1.0).collect();

        let quantized = quantize_f32_to_q8_0(&data).expect("quantize failed");
        assert_eq!(quantized.len(), Q8_0_BLOCK_BYTES);

        let restored =
            dequantize_to_f32(&quantized, GgufTensorType::Q8_0, 32).expect("dequantize failed");
        assert_eq!(restored.len(), 32);

        // Q8_0 has 256 levels — should be very close.
        let err = max_abs_error(&data, &restored);
        assert!(err < 0.02, "Q8_0 round-trip error too large: {err}");
    }

    #[test]
    fn q8_0_round_trip_multiple_blocks() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();

        let quantized = quantize_f32_to_q8_0(&data).expect("quantize failed");
        assert_eq!(quantized.len(), 4 * Q8_0_BLOCK_BYTES);

        let restored =
            dequantize_to_f32(&quantized, GgufTensorType::Q8_0, 128).expect("dequantize failed");
        assert_eq!(restored.len(), 128);

        let err = max_abs_error(&data, &restored);
        assert!(err < 0.01, "Q8_0 multi-block error: {err}");
    }

    // ── F16 → Q4_0 ──────────────────────────────────────────────────

    #[test]
    fn f16_to_q4_0_round_trip() {
        let f32_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let f16_data: Vec<u16> = f32_data
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();

        let q_via_f16 = quantize_f16_to_q4_0(&f16_data).expect("f16 quantize failed");
        let q_via_f32 = quantize_f32_to_q4_0(&f32_data).expect("f32 quantize failed");

        // Both paths should produce the same quantized bytes (within FP16
        // precision — the scale may differ by one ULP).  We compare the
        // dequantized outputs instead for robustness.
        let r16 = dequantize_to_f32(&q_via_f16, GgufTensorType::Q4_0, 32).expect("deq f16 failed");
        let r32 = dequantize_to_f32(&q_via_f32, GgufTensorType::Q4_0, 32).expect("deq f32 failed");

        let err = max_abs_error(&r16, &r32);
        assert!(err < 0.25, "F16 vs F32 path divergence: {err}");
    }

    // ── F16 → Q8_0 ──────────────────────────────────────────────────

    #[test]
    fn f16_to_q8_0_round_trip() {
        let f32_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
        let f16_data: Vec<u16> = f32_data
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();

        let q = quantize_f16_to_q8_0(&f16_data).expect("f16→q8_0 failed");
        let restored = dequantize_to_f32(&q, GgufTensorType::Q8_0, 32).expect("deq failed");

        let err = max_abs_error(&f32_data, &restored);
        // Extra FP16 rounding adds a bit of noise on top of Q8_0 quantization.
        assert!(err < 0.03, "F16→Q8_0 error: {err}");
    }

    // ── Alignment errors ─────────────────────────────────────────────

    #[test]
    fn q4_0_rejects_unaligned_input() {
        let data = vec![0.0f32; 33]; // not a multiple of 32
        let result = quantize_f32_to_q4_0(&data);
        assert!(result.is_err());
        match result {
            Err(QuantError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn q8_0_rejects_unaligned_input() {
        let data = vec![0.0f32; 31];
        let result = quantize_f32_to_q8_0(&data);
        assert!(result.is_err());
        match result {
            Err(QuantError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // ── Zero input ───────────────────────────────────────────────────

    #[test]
    fn q4_0_zero_input() {
        let data = vec![0.0f32; 32];
        let quantized = quantize_f32_to_q4_0(&data).expect("zero quantize failed");

        let restored = dequantize_to_f32(&quantized, GgufTensorType::Q4_0, 32).expect("deq failed");
        for &v in &restored {
            assert!(v.abs() < f32::EPSILON, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn q8_0_zero_input() {
        let data = vec![0.0f32; 32];
        let quantized = quantize_f32_to_q8_0(&data).expect("zero quantize failed");

        let restored = dequantize_to_f32(&quantized, GgufTensorType::Q8_0, 32).expect("deq failed");
        for &v in &restored {
            assert!(v.abs() < f32::EPSILON, "expected 0.0, got {v}");
        }
    }

    // ── Large positive / negative ────────────────────────────────────

    #[test]
    fn q4_0_large_values() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1000.0;
        data[1] = -1000.0;
        data[31] = 500.0;

        let quantized = quantize_f32_to_q4_0(&data).expect("quantize failed");
        let restored = dequantize_to_f32(&quantized, GgufTensorType::Q4_0, 32).expect("deq failed");

        // First element should retain sign and rough magnitude.
        assert!(restored[0] > 0.0, "expected positive, got {}", restored[0]);
        assert!(restored[1] < 0.0, "expected negative, got {}", restored[1]);
        assert!(
            restored[31] > 0.0,
            "expected positive, got {}",
            restored[31]
        );

        // Relative error on the largest element should be bounded.
        let rel_err_0 = (restored[0] - 1000.0).abs() / 1000.0;
        assert!(
            rel_err_0 < 0.15,
            "Q4_0 large-value relative error: {rel_err_0}"
        );
    }

    #[test]
    fn q8_0_large_values() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1000.0;
        data[1] = -1000.0;
        data[31] = 500.0;

        let quantized = quantize_f32_to_q8_0(&data).expect("quantize failed");
        let restored = dequantize_to_f32(&quantized, GgufTensorType::Q8_0, 32).expect("deq failed");

        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
        assert!(restored[31] > 0.0);

        let rel_err_0 = (restored[0] - 1000.0).abs() / 1000.0;
        assert!(
            rel_err_0 < 0.02,
            "Q8_0 large-value relative error: {rel_err_0}"
        );
    }

    // ── Empty input ──────────────────────────────────────────────────

    #[test]
    fn empty_input_produces_empty_output() {
        let empty: Vec<f32> = Vec::new();
        let q4 = quantize_f32_to_q4_0(&empty).expect("empty q4_0 failed");
        assert!(q4.is_empty());

        let q8 = quantize_f32_to_q8_0(&empty).expect("empty q8_0 failed");
        assert!(q8.is_empty());

        let deq = dequantize_to_f32(&[], GgufTensorType::Q4_0, 0).expect("empty deq failed");
        assert!(deq.is_empty());
    }
}
