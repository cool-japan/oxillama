//! AVX-512 accelerated Q1_0_G128 quantization kernel.
//!
//! Q1_0_G128 block layout (18 bytes per 128 weights):
//! - bytes[0..2]  — FP16 scale `d` (little-endian)
//! - bytes[2..18] — 16 bytes = 128 sign bits (bit=1 → +d, bit=0 → −d)
//!
//! Bit ordering: LSB-first per byte.  Byte 0, bit 0 → weight 0;
//! byte 0, bit 7 → weight 7; byte 1, bit 0 → weight 8; etc.
//!
//! ## GEMV strategy
//!
//! Same XOR-sign approach as the AVX2 kernel, but we process **2 sign bytes
//! (16 weights) per `__m512` pass** instead of 1 sign byte (8 weights) per
//! `__m256` pass.  This halves the loop trip count.
//!
//! For each pair of consecutive sign bytes (8 weights each):
//! - Expand 16 bits into 16 × u32 XOR masks (0 → 0x80000000, 1 → 0x00000000).
//! - XOR 16 input floats with the mask to apply sign.
//! - Accumulate with `_mm512_add_ps`.
//!
//! At the end, multiply the horizontal sum by `d`.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q1_0_G128: 128 weights per block.
pub const BLOCK_SIZE: usize = 128;
/// Bytes per Q1_0_G128 block: 2 (FP16 scale) + 16 (128 sign bits).
pub const BLOCK_BYTES: usize = 18;

/// AVX-512 accelerated Q1_0_G128 kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
pub struct Q1_0G128Avx512;

impl QuantKernel for Q1_0G128Avx512 {
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
        // CPU avx512f support guaranteed by KernelDispatcher.
        unsafe { dequant_block_avx512(block, output) }
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
            // CPU avx512f support guaranteed by KernelDispatcher.
            *out = unsafe {
                gemv_row_avx512(
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
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a 16-element XOR mask array from two consecutive sign bytes.
///
/// For each of the 16 bit positions (byte0 bits 0..7 then byte1 bits 0..7):
///   - bit==1 → mask = 0x00000000 (leave sign unchanged, weight = +d)
///   - bit==0 → mask = 0x80000000 (flip IEEE-754 sign, weight = −d)
///
/// Stored in natural order: element 0 ← byte0 bit 0 (LSB).
#[inline(always)]
fn two_sign_bytes_to_xor_masks(byte0: u8, byte1: u8) -> [u32; 16] {
    core::array::from_fn(|i| {
        let bit = if i < 8 {
            (byte0 >> i) & 1
        } else {
            (byte1 >> (i - 8)) & 1
        };
        // bit XOR 1 == 1 when the weight is negative; shift to sign-bit position.
        (bit as u32 ^ 1) << 31
    })
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 18-byte Q1_0_G128 block into 128 FP32 values using AVX-512.
///
/// Processes 16 weights (2 sign bytes) per `__m512` pass — 8 passes total.
///
/// # Safety
/// - `block.len() >= 18`
/// - `output.len() >= 128`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 18 >= 2.
    let d = f16_to_f32(block);

    // Broadcast +d: XOR-ing with 0x80000000 mask will flip to −d as needed.
    let vd = _mm512_set1_ps(d);

    // Process pairs of sign bytes: each pair covers 16 weights (one AVX-512 lane).
    // 16 sign bytes / 2 = 8 AVX-512 passes.
    for pair_idx in 0..8 {
        // SAFETY: 2 + pair_idx*2 + 1 <= 17 < 18; block.len() >= 18.
        let byte0 = *block.get_unchecked(2 + pair_idx * 2);
        let byte1 = *block.get_unchecked(2 + pair_idx * 2 + 1);
        let masks = two_sign_bytes_to_xor_masks(byte0, byte1);

        // Load the 16 × u32 mask as a 512-bit int register.
        // SAFETY: masks is [u32; 16] = 64 bytes on stack.
        let vmask = _mm512_loadu_si512(masks.as_ptr() as *const __m512i);

        // XOR vd (as integer) with the mask to flip sign bits where bit 31 == 1.
        // _mm512_xor_ps requires avx512dq; use _mm512_xor_si512 (avx512f) + casts instead.
        // mask lanes: 0x00000000 (XOR no-op, keeps +d) or 0x80000000 (flips sign → −d).
        let result = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(vd), vmask));

        // Store 16 dequantized weights.
        // SAFETY: pair_idx * 16 + 15 < 128; output.len() >= 128.
        _mm512_storeu_ps(output.as_mut_ptr().add(pair_idx * 16), result);
    }
}

/// Compute the dot product of one row of a Q1_0_G128 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
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
            // Fast path: all 128 weights are valid — 8 AVX-512 passes of 16.
            let mut acc = _mm512_setzero_ps();

            for pair_idx in 0..8 {
                // SAFETY: 2 + pair_idx*2 + 1 < 18; block.len() == 18.
                let byte0 = *block.get_unchecked(2 + pair_idx * 2);
                let byte1 = *block.get_unchecked(2 + pair_idx * 2 + 1);
                let masks = two_sign_bytes_to_xor_masks(byte0, byte1);

                // Expand the sign bytes into per-lane XOR masks.
                // SAFETY: masks is [u32; 16] = 64 bytes on stack.
                let vmask = _mm512_loadu_si512(masks.as_ptr() as *const __m512i);

                // Load 16 input floats.
                // SAFETY: input_offset + pair_idx*16 + 15 < input_offset + 128 <= n_cols <= input.len().
                let inp = _mm512_loadu_ps(input.as_ptr().add(input_offset + pair_idx * 16));

                // XOR input sign bits with the mask, then accumulate.
                // bit==0 → flip input sign (weight=−d), bit==1 → keep sign (weight=+d).
                // Use _mm512_xor_si512 (avx512f) instead of _mm512_xor_ps (avx512dq).
                let signed_inp =
                    _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(inp), vmask));
                acc = _mm512_add_ps(acc, signed_inp);
            }

            row_sum += d * hsum_f32_avx512(acc);
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

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_dequant_all_positive() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_q1_block(2.0, &[0xFF; 16]);
        let mut out_avx512 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx512
            .dequant_block(&block, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_dequant_all_negative() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_q1_block(3.0, &[0x00; 16]);
        let mut out_avx512 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx512
            .dequant_block(&block, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_dequant_alternating() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 0xAA = 10101010: bits 1,3,5,7 are +d; bits 0,2,4,6 are -d.
        let block = make_q1_block(1.5, &[0xAA; 16]);
        let mut out_avx512 = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Avx512
            .dequant_block(&block, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_all_positive() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // All bits = 1 → all weights = +d = +1.0, all inputs = 1.0 → dot = 128.
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_avx512 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.5,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_alternating() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Alternating sign bits: should cancel with uniform input.
        let block = make_q1_block(1.0, &[0xAA; 16]);
        let tensor_avx512 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.5,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_matches_reference_random() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let bits: [u8; 16] = [
            0b10110101, 0b01001110, 0b11100010, 0b00011111, 0b10101010, 0b01010101, 0b11001100,
            0b00110011, 0b11110000, 0b00001111, 0b10011001, 0b01100110, 0b11111110, 0b00000001,
            0b10000001, 0b01111110,
        ];
        let block = make_q1_block(0.5, &bits);
        let tensor_avx512 = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 1.9).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-3,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 80 columns — one partial block (80 < 128).
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_avx512 = make_tensor(block.clone(), 80);
        let tensor_ref = make_tensor(block, 80);

        let input = vec![1.0f32; 80];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q1_0G128Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.5,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }
}
