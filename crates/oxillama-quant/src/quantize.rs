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
/// Bytes per Q5_0 block: 2 (FP16 scale) + 4 (high bits) + 16 (nibble data).
const Q5_0_BLOCK_BYTES: usize = 22;
/// Bytes per Q5_1 block: 2 (scale) + 2 (min) + 4 (high bits) + 16 (nibbles).
const Q5_1_BLOCK_BYTES: usize = 24;
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

/// Quantize FP32 values to Q5_0 format.
///
/// Q5_0 block layout (22 bytes per 32 weights):
///   - 2 bytes: FP16 scale `d`
///   - 4 bytes: `qh` — bit 4 of each of the 32 codes, LSB-first
///   - 16 bytes: 32 low nibbles, split-half packed
///
/// Port of llama.cpp `quantize_row_q5_0_ref`. This exists for the model-level
/// re-quantization path: llama.cpp falls back from Q4_K to Q5_0 for tensors
/// whose row length is not a multiple of 256.
///
/// Returns an error when `data.len()` is not a multiple of 32.
pub fn quantize_f32_to_q5_0(data: &[f32]) -> QuantResult<Vec<u8>> {
    let n_blocks = check_legacy_row(data, "Q5_0")?;
    let mut out = Vec::with_capacity(n_blocks * Q5_0_BLOCK_BYTES);
    for blk_idx in 0..n_blocks {
        encode_q5_0_block(
            &data[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE],
            &mut out,
        );
    }
    Ok(out)
}

/// Quantize FP32 values to Q5_1 format.
///
/// Q5_1 block layout (24 bytes per 32 weights):
///   - 2 bytes: FP16 scale `d`
///   - 2 bytes: FP16 minimum `m`
///   - 4 bytes: `qh` — bit 4 of each of the 32 codes, LSB-first
///   - 16 bytes: 32 low nibbles, split-half packed
///
/// Port of llama.cpp `quantize_row_q5_1_ref`; llama.cpp's fallback for Q5_K
/// on rows that are not a multiple of 256.
///
/// Returns an error when `data.len()` is not a multiple of 32.
pub fn quantize_f32_to_q5_1(data: &[f32]) -> QuantResult<Vec<u8>> {
    let n_blocks = check_legacy_row(data, "Q5_1")?;
    let mut out = Vec::with_capacity(n_blocks * Q5_1_BLOCK_BYTES);
    for blk_idx in 0..n_blocks {
        encode_q5_1_block(
            &data[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE],
            &mut out,
        );
    }
    Ok(out)
}

/// Reject a row that is not a whole number of 32-weight blocks.
fn check_legacy_row(data: &[f32], quant_type: &'static str) -> QuantResult<usize> {
    if !data.len().is_multiple_of(BLOCK_SIZE) {
        return Err(QuantError::RowNotBlockAligned {
            quant_type,
            block_size: BLOCK_SIZE,
            row_len: data.len(),
        });
    }
    Ok(data.len() / BLOCK_SIZE)
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

/// Encode one row of FP32 weights into `target`'s block bytes.
///
/// This is the single entry point the re-quantization pipeline uses; it knows
/// every type OxiLLaMa can *write*, which is a strict subset of the types it
/// can read. Passing a read-only type (an I-quant, Q8_K, …) returns
/// [`QuantError::UnsupportedType`] rather than silently producing garbage.
///
/// `values.len()` must be a whole number of `target.block_size()` weights;
/// see [`QuantError::RowNotBlockAligned`] for why padding is not an option.
pub fn quantize_f32_row(values: &[f32], target: GgufTensorType) -> QuantResult<Vec<u8>> {
    match target {
        GgufTensorType::F32 => Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        GgufTensorType::F16 => Ok(values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect()),
        GgufTensorType::Q4_0 => quantize_f32_to_q4_0(values),
        GgufTensorType::Q5_0 => quantize_f32_to_q5_0(values),
        GgufTensorType::Q5_1 => quantize_f32_to_q5_1(values),
        GgufTensorType::Q8_0 => quantize_f32_to_q8_0(values),
        GgufTensorType::Q2K => crate::kquant::quantize_f32_to_q2_k(values),
        GgufTensorType::Q3K => crate::kquant::quantize_f32_to_q3_k(values),
        GgufTensorType::Q4K => crate::kquant::quantize_f32_to_q4_k(values),
        GgufTensorType::Q5K => crate::kquant::quantize_f32_to_q5_k(values),
        GgufTensorType::Q6K => crate::kquant::quantize_f32_to_q6_k(values),
        other => Err(QuantError::UnsupportedType {
            quant_type: format!("{other} (no encoder; OxiLLaMa can read but not write this type)"),
        }),
    }
}

/// Returns `true` when [`quantize_f32_row`] can encode `target`.
pub fn can_encode(target: GgufTensorType) -> bool {
    matches!(
        target,
        GgufTensorType::F32
            | GgufTensorType::F16
            | GgufTensorType::Q4_0
            | GgufTensorType::Q5_0
            | GgufTensorType::Q5_1
            | GgufTensorType::Q8_0
            | GgufTensorType::Q2K
            | GgufTensorType::Q3K
            | GgufTensorType::Q4K
            | GgufTensorType::Q5K
            | GgufTensorType::Q6K
    )
}

/// Encode a whole 2-D tensor, one row at a time.
///
/// Quantization is **per row**, exactly as ggml's `ggml_quantize_chunk` does
/// it: every row gets its own blocks, so a row must independently be a whole
/// number of blocks. Quantizing the flattened tensor as a single stream would
/// let one row's tail share a block with the next row's head, which the
/// decoder — which indexes rows by `row * row_size` — would then read back
/// misaligned.
///
/// With the `parallel` feature (on by default) rows are encoded on the rayon
/// pool; output order is preserved.
pub fn quantize_f32_rows(
    values: &[f32],
    n_per_row: usize,
    target: GgufTensorType,
) -> QuantResult<Vec<u8>> {
    if n_per_row == 0 {
        return Err(QuantError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if !values.len().is_multiple_of(n_per_row) {
        return Err(QuantError::DimensionMismatch {
            expected: values.len().div_ceil(n_per_row) * n_per_row,
            got: values.len(),
        });
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let rows: Vec<Vec<u8>> = values
            .par_chunks(n_per_row)
            .map(|row| quantize_f32_row(row, target))
            .collect::<QuantResult<Vec<Vec<u8>>>>()?;
        Ok(rows.concat())
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::new();
        for row in values.chunks(n_per_row) {
            out.extend_from_slice(&quantize_f32_row(row, target)?);
        }
        Ok(out)
    }
}

// ── Block encoders (private) ─────────────────────────────────────────

/// Encode 32 FP32 values into one Q4_0 block, appending to `out`.
///
/// Byte-exact port of llama.cpp `quantize_row_q4_0_ref`:
///   1. Find the element with the largest absolute value → `max` (keeps sign)
///   2. `d = max / -8` (maps that extreme to nibble 0)
///   3. `id = 1/d` (or 0 when `d == 0`), computed from the **f32** `d`, before
///      it is rounded to f16 — this is what upstream does, and deriving `id`
///      from the f16-rounded value instead yields different bytes on some
///      inputs.
///   4. `q_j = min(15, trunc(x_j * id + 8.5))`.  The argument is always
///      non-negative (`|x_j · id| ≤ 8`), so `clamp(0, 15)` is identical to
///      upstream's `MIN(15, (int8_t)…)`.
///   5. Pack **split halves**: `qs[j] = q(x[j]) | (q(x[j + 16]) << 4)`.
///
/// Step 5 is the encoder side of the layout documented in
/// [`crate::reference::q4_0`]: byte `j` carries weight `j` and weight
/// `j + 16`, *not* weights `2j` and `2j + 1`.
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

    // 4-5. Quantize and pack split halves
    for j in 0..BLOCK_SIZE / 2 {
        let x0 = values[j];
        let x1 = values[j + BLOCK_SIZE / 2];

        let q0 = ((x0 * id + 8.5) as i32).clamp(0, 15) as u8;
        let q1 = ((x1 * id + 8.5) as i32).clamp(0, 15) as u8;

        out.push(q0 | (q1 << 4));
    }
}

/// Encode 32 FP32 values into one Q5_0 block, appending to `out`.
///
/// Port of llama.cpp `quantize_row_q5_0_ref`: `d = max / -16`, codes
/// `min(31, trunc(x * id + 16.5))`, low nibbles split-half packed and the
/// 5th bit of code `j` written to bit `j` of the little-endian `qh` word.
fn encode_q5_0_block(values: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK_SIZE);

    let max_val = values
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| if v.abs() > acc.abs() { v } else { acc });

    let d = max_val / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let d_fp16 = f16::from_f32(d);
    out.extend_from_slice(&d_fp16.to_bits().to_le_bytes());

    let mut qh: u32 = 0;
    let mut qs = [0u8; BLOCK_SIZE / 2];
    for j in 0..BLOCK_SIZE / 2 {
        // `|x * id| <= 16` by construction, so the argument is >= 0.5 and the
        // C cast's truncation towards zero cannot go negative.
        let xi0 = ((values[j] * id + 16.5) as i32).clamp(0, 31) as u32;
        let xi1 = ((values[j + BLOCK_SIZE / 2] * id + 16.5) as i32).clamp(0, 31) as u32;
        qs[j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
        qh |= ((xi0 & 0x10) >> 4) << j;
        qh |= ((xi1 & 0x10) >> 4) << (j + BLOCK_SIZE / 2);
    }
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
}

/// Encode 32 FP32 values into one Q5_1 block, appending to `out`.
///
/// Port of llama.cpp `quantize_row_q5_1_ref`: affine `d = (max - min) / 31`
/// with the block minimum stored alongside the scale.
fn encode_q5_1_block(values: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK_SIZE);

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&f16::from_f32(min).to_bits().to_le_bytes());

    let mut qh: u32 = 0;
    let mut qs = [0u8; BLOCK_SIZE / 2];
    for j in 0..BLOCK_SIZE / 2 {
        let xi0 = (((values[j] - min) * id + 0.5) as i32).clamp(0, 31) as u32;
        let xi1 = (((values[j + BLOCK_SIZE / 2] - min) * id + 0.5) as i32).clamp(0, 31) as u32;
        qs[j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
        qh |= ((xi0 & 0x10) >> 4) << j;
        qh |= ((xi1 & 0x10) >> 4) << (j + BLOCK_SIZE / 2);
    }
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
}

/// Weights per Q8_0 activation block (public mirror of the private
/// `BLOCK_SIZE`, for callers sizing a [`quantize_activations_q8_0_into`]
/// scratch buffer).
pub const Q8_0_ACT_BLOCK_SIZE: usize = BLOCK_SIZE;

/// Bytes per Q8_0 activation block: 2 (FP16 scale) + 32 (int8 values).
pub const Q8_0_ACT_BLOCK_BYTES: usize = Q8_0_BLOCK_BYTES;

/// Quantize a decode activation vector into exactly `n_blocks` Q8_0 blocks,
/// reusing `out`'s allocation.
///
/// This is the producer for [`crate::traits::QuantKernel::matvec_q8_fused`]: the activation
/// vector is converted to Q8_0 **once per matmul input** and then consumed by
/// every weight row, so the `f32 → i8` conversion cost is amortised over
/// thousands of rows instead of being repeated inside the row loop.
///
/// `n_blocks` is dictated by the *weight* kernel, not by `data.len()`: a Q4_K
/// row pairs each 256-weight block with 8 Q8_0 activation blocks, so a kernel
/// whose K is not a whole number of 256 still asks for `ceil(K/256) * 8`
/// blocks.  Every element past `data.len()` is quantized as `0.0`, which
/// contributes exactly zero to a dot product and to the Q4_K/Q6_K `Σq_a`
/// correction terms — the same value the ragged-tail scalar paths compute by
/// masking.
///
/// `out` is truncated-and-refilled rather than reallocated, so after the first
/// call in a forward pass this performs no allocation.
///
/// Bit-for-bit identical to running [`quantize_f32_to_q8_0`] on `data`
/// zero-padded to `n_blocks * 32` elements.
pub fn quantize_activations_q8_0_into(data: &[f32], n_blocks: usize, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(n_blocks * Q8_0_ACT_BLOCK_BYTES);
    // At most one partially populated block, then all-zero padding blocks.
    append_activations_q8_0(data, n_blocks, out);
}

/// Quantize `m` consecutive activation vectors into one contiguous Q8_0
/// buffer, `n_blocks` blocks per vector.
///
/// This is the batched-prefill producer for
/// [`crate::traits::QuantKernel::matmul_q8_fused`]: `rows` is a row-major
/// `[m][row_stride]` matrix of activations (one token per row) and the result
/// is a `[m][n_blocks]` Q8_0 image whose vector `t` starts at byte
/// `t * n_blocks * `[`Q8_0_ACT_BLOCK_BYTES`].
///
/// Each vector is encoded exactly as [`quantize_activations_q8_0_into`] would
/// encode it on its own — vectors share no scale and no rounding state — so a
/// batched matmul consumes byte-identical activations to the per-token path.
///
/// Only the first `n_cols` values of each row are quantized; a `row_stride`
/// wider than `n_cols` (padded scratch) is permitted and the padding is
/// ignored.
pub fn quantize_activations_q8_0_batch_into(
    rows: &[f32],
    m: usize,
    row_stride: usize,
    n_cols: usize,
    n_blocks: usize,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(m * n_blocks * Q8_0_ACT_BLOCK_BYTES);

    let cols = n_cols.min(row_stride);
    for t in 0..m {
        let start = t * row_stride;
        let row = match rows.get(start..start + cols) {
            Some(r) => r,
            None => &[],
        };
        append_activations_q8_0(row, n_blocks, out);
    }
}

/// Append exactly `n_blocks` Q8_0 blocks encoding `data` (zero-padded) to
/// `out`, without clearing it.
fn append_activations_q8_0(data: &[f32], n_blocks: usize, out: &mut Vec<u8>) {
    let full_blocks = (data.len() / BLOCK_SIZE).min(n_blocks);
    for blk_idx in 0..full_blocks {
        let blk = &data[blk_idx * BLOCK_SIZE..(blk_idx + 1) * BLOCK_SIZE];
        encode_q8_0_block(blk, out);
    }

    let mut padded = [0.0f32; BLOCK_SIZE];
    for blk_idx in full_blocks..n_blocks {
        let start = blk_idx * BLOCK_SIZE;
        let tail = data.get(start..).unwrap_or(&[]);
        let valid = tail.len().min(BLOCK_SIZE);
        padded[..valid].copy_from_slice(&tail[..valid]);
        padded[valid..].fill(0.0);
        encode_q8_0_block(&padded, out);
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

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
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

    #[cfg_attr(miri, ignore)] // half crate aarch64 asm not supported in Miri
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
