//! AVX2+FMA accelerated Q1_0_G128 quantization kernel.
//!
//! Q1_0_G128 block layout (18 bytes per 128 weights):
//! - bytes[0..2]  — FP16 scale `d` (little-endian)
//! - bytes[2..18] — 16 bytes = 128 sign bits (bit=1 → +d, bit=0 → −d)
//!
//! Bit ordering: LSB-first per byte.  Byte 0, bit 0 → weight 0;
//! byte 0, bit 7 → weight 7; byte 1, bit 0 → weight 8; etc.
//!
//! This is PrismML's 1-bit Bonsai quantization format.
//!
//! ## GEMV strategy
//!
//! For each sign byte (8 weights), we XOR the sign bit of the corresponding
//! FP32 inputs: bit==0 means weight=−d, so we flip the input's sign bit;
//! bit==1 means weight=+d, so we leave the input unchanged.
//!
//! XOR mask: `bit==0 → 0x80000000` (flip IEEE-754 sign), `bit==1 → 0x00000000`.
//! Sum all XOR'd inputs, then multiply by `d`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q1_0_G128: 128 weights per block.
pub const BLOCK_SIZE: usize = 128;
/// Bytes per Q1_0_G128 block: 2 (FP16 scale) + 16 (128 sign bits).
pub const BLOCK_BYTES: usize = 18;

/// AVX2+FMA accelerated Q1_0_G128 kernel.
///
/// Requires `avx2` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q1_0G128Avx2;

impl QuantKernel for Q1_0G128Avx2 {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_SIZE,
                available: output.len(),
            });
        }

        // SAFETY: block.len() >= 18 and output.len() >= 128 verified above.
        // CPU avx2 support guaranteed by KernelDispatcher.
        unsafe { dequant_block_avx2(block, output) }
        Ok(())
    }

    fn gemv(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
    ) -> QuantResult<()> {
        let n_rows = quant_matrix.shape[0];
        let n_cols = if quant_matrix.shape.len() > 1 {
            quant_matrix.shape[1]
        } else {
            quant_matrix.n_elements() / n_rows
        };

        if input.len() < n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_cols,
                got: input.len(),
            });
        }
        if output.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: output.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: row/block bounds verified above.
            // CPU avx2 support guaranteed by KernelDispatcher.
            *out = unsafe {
                gemv_row_avx2(
                    &quant_matrix.data[row_start..row_start + row_bytes],
                    input,
                    blocks_per_row,
                    n_cols,
                )
            };
        }

        Ok(())
    }

    fn gemm(
        &self,
        quant_matrix: &QuantTensor,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> QuantResult<()> {
        for row in 0..m {
            let input_row = &input[row * k..(row + 1) * k];
            let output_row = &mut output[row * n..(row + 1) * n];
            self.gemv(quant_matrix, input_row, output_row)?;
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q1_0_G128"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Build an 8-element XOR mask from a sign byte.
///
/// For each bit position `i` in `sign_byte`:
///   - bit==1 → XOR mask = 0x00000000 (leave sign unchanged, weight = +d)
///   - bit==0 → XOR mask = 0x80000000 (flip IEEE-754 sign, weight = −d)
///
/// Stored in natural order: element 0 corresponds to bit 0 (LSB).
#[inline(always)]
fn sign_byte_to_xor_masks(sign_byte: u8) -> [u32; 8] {
    core::array::from_fn(|i| {
        let bit = (sign_byte >> i) & 1;
        // bit XOR 1 == 1 when the weight is negative; shift to sign bit position.
        (bit as u32 ^ 1) << 31
    })
}

/// Dequantize one 18-byte Q1_0_G128 block into 128 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 18`
/// - `output.len() >= 128`
/// - CPU must support `avx2`
#[target_feature(enable = "avx2")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 18 >= 2.
    let d = f16_to_f32(block);

    // Broadcast +d and -d as AVX2 registers for fast blending.
    let vd = _mm256_set1_ps(d);
    let vd_neg = _mm256_set1_ps(-d);

    // 16 sign bytes, each covering 8 weights.
    for byte_idx in 0..16 {
        // SAFETY: block[2 + byte_idx] valid because 2 + 15 = 17 < 18.
        let sign_byte = *block.get_unchecked(2 + byte_idx);
        let masks = sign_byte_to_xor_masks(sign_byte);

        // Load mask as 256-bit integer, then reinterpret as float for blendv.
        // _mm256_blendv_ps selects vd_neg when mask bit 31 is set (i.e., bit==0 → negative).
        // SAFETY: masks is a [u32; 8] = 32 bytes on stack — always aligned enough for loadu.
        let vmask = _mm256_loadu_si256(masks.as_ptr() as *const __m256i);
        let vmask_ps = _mm256_castsi256_ps(vmask);

        // blendv_ps: for each lane, if mask bit 31 == 1 → pick vd_neg, else → pick vd.
        let result = _mm256_blendv_ps(vd, vd_neg, vmask_ps);

        // Store 8 dequantized weights.
        // SAFETY: byte_idx * 8 + 7 < 128; output.len() >= 128.
        _mm256_storeu_ps(output.as_mut_ptr().add(byte_idx * 8), result);
    }
}

/// Compute the dot product of one row of a Q1_0_G128 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx2`
#[target_feature(enable = "avx2")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        // Read FP16 scale.
        // SAFETY: block.len() == 18 >= 2.
        let d = f16_to_f32(block);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 128 weights are valid — process 16 bytes × 8 weights.
            let mut acc = _mm256_setzero_ps();

            for byte_idx in 0..16 {
                // SAFETY: 2 + byte_idx < 18; block.len() == 18.
                let sign_byte = *block.get_unchecked(2 + byte_idx);
                let masks = sign_byte_to_xor_masks(sign_byte);

                // Expand the sign byte into a per-lane XOR mask (0 or 0x80000000).
                // SAFETY: masks is [u32; 8] = 32 bytes on stack.
                let vmask = _mm256_loadu_si256(masks.as_ptr() as *const __m256i);
                let vmask_ps = _mm256_castsi256_ps(vmask);

                // Load 8 input floats.
                // SAFETY: input_offset + byte_idx * 8 + 7 < input_offset + 128 <= n_cols <= input.len().
                let inp = _mm256_loadu_ps(input.as_ptr().add(input_offset + byte_idx * 8));

                // XOR the sign bit: if bit==0 the weight is -d so flip input sign,
                // if bit==1 the weight is +d so leave input unchanged.
                // Then accumulate: acc += signed_input (will be multiplied by d at end).
                let signed_inp = _mm256_xor_ps(inp, vmask_ps);
                acc = _mm256_add_ps(acc, signed_inp);
            }

            row_sum += d * hsum_f32_avx(acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut diff = 0.0f32;

            for byte_idx in 0..16 {
                // SAFETY: 2 + byte_idx < 18; block.len() == 18.
                let sign_byte = *block.get_unchecked(2 + byte_idx);
                for bit_idx in 0..8 {
                    let weight_idx = input_offset + byte_idx * 8 + bit_idx;
                    if weight_idx < n_cols {
                        let bit = (sign_byte >> bit_idx) & 1;
                        if bit == 1 {
                            diff += input[weight_idx];
                        } else {
                            diff -= input[weight_idx];
                        }
                    }
                }
            }

            row_sum += d * diff;
        }
        // remaining == 0: block fully out of bounds, skip.
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q1_0_g128::Q1_0G128Ref;

    fn make_q1_block(scale: f32, bits: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(bits);
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(
            block,
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Q1_0G128,
        )
    }

    #[test]
    fn test_dequant_all_positive() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_q1_block(2.0, &[0xFF; 16]);
        let mut out_avx2 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx2.dequant_block(&block, &mut out_avx2).unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_all_negative() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_q1_block(3.0, &[0x00; 16]);
        let mut out_avx2 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx2.dequant_block(&block, &mut out_avx2).unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_alternating() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 0xAA = 10101010: bits 1,3,5,7 are +d; bits 0,2,4,6 are -d.
        let block = make_q1_block(1.5, &[0xAA; 16]);
        let mut out_avx2 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx2.dequant_block(&block, &mut out_avx2).unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_all_positive() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // All bits = 1 → all weights = +d = +1.0, all inputs = 1.0 → dot = 128.
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_avx2 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.5,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_alternating() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Alternating sign bits: should cancel with uniform input.
        let block = make_q1_block(1.0, &[0xAA; 16]);
        let tensor_avx2 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.5,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_matches_reference_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Non-uniform pattern and non-uniform input.
        let bits: [u8; 16] = [
            0b10110101, 0b01001110, 0b11100010, 0b00011111, 0b10101010, 0b01010101, 0b11001100,
            0b00110011, 0b11110000, 0b00001111, 0b10011001, 0b01100110, 0b11111110, 0b00000001,
            0b10000001, 0b01111110,
        ];
        let block = make_q1_block(0.5, &bits);
        let tensor_avx2 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 1.9).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-3,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 80 columns — one partial block (80 < 128).
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_avx2 = make_tensor(block.clone(), 80);
        let tensor_ref = make_tensor(block, 80);

        let input = vec![1.0f32; 80];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.5,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }
}
