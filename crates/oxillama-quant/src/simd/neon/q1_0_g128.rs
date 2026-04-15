//! NEON (AArch64) accelerated Q1_0_G128 quantization kernel.
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

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q1_0_G128: 128 weights per block.
pub const BLOCK_SIZE: usize = 128;
/// Bytes per Q1_0_G128 block: 2 (FP16 scale) + 16 (128 sign bits).
pub const BLOCK_BYTES: usize = 18;

/// NEON-accelerated Q1_0_G128 kernel (AArch64 only).
pub struct Q1_0G128Neon;

/// Convert an IEEE 754 FP16 value (as raw u16 LE bytes) to f32.
#[inline(always)]
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// Expand a sign byte into two `uint32x4_t` XOR masks.
///
/// For each bit position `i` in `sign_byte`:
///   - bit==1 → XOR mask lane = 0x00000000 (leave sign unchanged, weight = +d)
///   - bit==0 → XOR mask lane = 0x80000000 (flip IEEE-754 sign, weight = −d)
///
/// Elements ordered so element 0 corresponds to bit 0 (LSB).
#[inline(always)]
fn sign_byte_to_neon_masks(sign_byte: u8) -> (uint32x4_t, uint32x4_t) {
    let masks: [u32; 8] = core::array::from_fn(|i| {
        let bit = (sign_byte >> i) & 1;
        // bit XOR 1 == 1 when weight is negative; shift to sign-bit position.
        (bit as u32 ^ 1) << 31
    });
    // SAFETY: vld1q_u32 requires a pointer to 4 valid u32 values.
    // masks is [u32; 8] on the stack, so both lo and hi pointers are valid.
    let lo = unsafe { vld1q_u32(masks.as_ptr()) };
    let hi = unsafe { vld1q_u32(masks.as_ptr().add(4)) };
    (lo, hi)
}

/// Dequantize one 18-byte Q1_0_G128 block into 128 FP32 values using NEON.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES` (18)
/// - `output.len() >= BLOCK_SIZE` (128)
/// - Must be called on AArch64 with NEON
unsafe fn dequant_block_neon(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 2.
    let d = f16_to_f32(block);

    // XOR approach: mask lanes are 0x00000000 (+d) or 0x80000000 (−d).
    // result_bits = vreinterpretq_u32_f32(vd) XOR mask
    let vd = unsafe { vdupq_n_f32(d) };

    for byte_idx in 0..16usize {
        // SAFETY: 2 + byte_idx <= 17 < 18; block.len() >= 18.
        let sign_byte = unsafe { *block.get_unchecked(2 + byte_idx) };
        let (mask_lo, mask_hi) = sign_byte_to_neon_masks(sign_byte);

        // XOR d's bits with the sign mask: mask lanes are already 0x00000000 or 0x80000000.
        let vd_bits = unsafe { vreinterpretq_u32_f32(vd) };

        let lo_bits = unsafe { veorq_u32(vd_bits, mask_lo) };
        let hi_bits = unsafe { veorq_u32(vd_bits, mask_hi) };

        let lo_f32 = unsafe { vreinterpretq_f32_u32(lo_bits) };
        let hi_f32 = unsafe { vreinterpretq_f32_u32(hi_bits) };

        // Store 8 dequantized weights.
        // SAFETY: byte_idx * 8 + 7 < 128; output.len() >= 128.
        let out_ptr = output.as_mut_ptr().add(byte_idx * 8);
        unsafe {
            vst1q_f32(out_ptr, lo_f32);
            vst1q_f32(out_ptr.add(4), hi_f32);
        }
    }
}

/// Compute the dot product of one row of a Q1_0_G128 matrix with an FP32 vector using NEON.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - Must be called on AArch64 with NEON
unsafe fn gemv_row_neon(
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
            // Fast path: all 128 weights are valid.
            // Accumulate the sum of ±input[i] for i in 0..128, then multiply by d.
            let mut acc_lo = unsafe { vdupq_n_f32(0.0f32) };
            let mut acc_hi = unsafe { vdupq_n_f32(0.0f32) };

            for byte_idx in 0..16usize {
                // SAFETY: 2 + byte_idx <= 17 < 18; block.len() == 18.
                let sign_byte = unsafe { *block.get_unchecked(2 + byte_idx) };
                let (mask_lo, mask_hi) = sign_byte_to_neon_masks(sign_byte);

                // Load 8 input floats.
                // SAFETY: input_offset + byte_idx * 8 + 7 < input_offset + 128 <= n_cols <= input.len().
                let inp_ptr = input.as_ptr().add(input_offset + byte_idx * 8);
                let inp_lo = unsafe { vld1q_f32(inp_ptr) };
                let inp_hi = unsafe { vld1q_f32(inp_ptr.add(4)) };

                // XOR the sign bits: if bit==0 the weight is −d so flip input sign,
                // if bit==1 weight is +d so leave unchanged.
                // SAFETY: vreinterpretq_u32_f32 and veorq_u32 are always valid on AArch64.
                let lo_bits = unsafe { veorq_u32(vreinterpretq_u32_f32(inp_lo), mask_lo) };
                let hi_bits = unsafe { veorq_u32(vreinterpretq_u32_f32(inp_hi), mask_hi) };

                let signed_lo = unsafe { vreinterpretq_f32_u32(lo_bits) };
                let signed_hi = unsafe { vreinterpretq_f32_u32(hi_bits) };

                // SAFETY: vaddq_f32 is always valid on AArch64.
                acc_lo = unsafe { vaddq_f32(acc_lo, signed_lo) };
                acc_hi = unsafe { vaddq_f32(acc_hi, signed_hi) };
            }

            // Combine and scale.
            // SAFETY: vaddq_f32 and vaddvq_f32 are always valid on AArch64.
            let acc = unsafe { vaddq_f32(acc_lo, acc_hi) };
            row_sum += d * unsafe { vaddvq_f32(acc) };
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut diff = 0.0f32;

            for byte_idx in 0..16usize {
                // SAFETY: 2 + byte_idx <= 17 < 18; block.len() == 18.
                let sign_byte = unsafe { *block.get_unchecked(2 + byte_idx) };
                for bit_idx in 0..8usize {
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

impl QuantKernel for Q1_0G128Neon {
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
        // AArch64 always has NEON.
        unsafe { dequant_block_neon(block, output) }
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
            // SAFETY: row/block bounds verified above. AArch64 always has NEON.
            *out = unsafe {
                gemv_row_neon(
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
        "Q1_0_G128_NEON"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
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
        let block = make_q1_block(2.0, &[0xFF; 16]);
        let mut out_neon = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Neon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q1_0G128Ref
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_all_negative() {
        let block = make_q1_block(3.0, &[0x00; 16]);
        let mut out_neon = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Neon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q1_0G128Ref
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_alternating() {
        // 0xAA = 10101010: bits 1,3,5,7 are +d; bits 0,2,4,6 are −d.
        let block = make_q1_block(1.5, &[0xAA; 16]);
        let mut out_neon = vec![0.0f32; 128];
        let mut out_ref = vec![0.0f32; 128];

        Q1_0G128Neon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q1_0G128Ref
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_all_positive() {
        // All bits = 1 → all weights = +d = +1.0, all inputs = 1.0 → dot = 128.
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_neon = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q1_0G128Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv all-positive: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_alternating() {
        // Alternating sign bits: should cancel with uniform input.
        let block = make_q1_block(1.0, &[0xAA; 16]);
        let tensor_neon = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input = vec![1.0f32; 128];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q1_0G128Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv alternating: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_matches_reference_random() {
        // Non-uniform bit pattern and non-uniform input.
        let bits: [u8; 16] = [
            0b10110101, 0b01001110, 0b11100010, 0b00011111, 0b10101010, 0b01010101, 0b11001100,
            0b00110011, 0b11110000, 0b00001111, 0b10011001, 0b01100110, 0b11111110, 0b00000001,
            0b10000001, 0b01111110,
        ];
        let block = make_q1_block(0.5, &bits);
        let tensor_neon = make_tensor(block.clone(), 128);
        let tensor_ref = make_tensor(block, 128);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 1.9).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q1_0G128Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv random: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        // 80 columns — one partial block (80 < 128).
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor_neon = make_tensor(block.clone(), 80);
        let tensor_ref = make_tensor(block, 80);

        let input = vec![1.0f32; 80];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q1_0G128Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q1_0G128Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "partial gemv: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_multi_row() {
        let n_rows = 4usize;
        let n_cols = 256usize; // 2 full blocks per row

        let bits: [u8; 16] = [
            0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
            0x44, 0x55,
        ];

        let mut data = Vec::new();
        let scales = [0.5f32, 1.0, 1.5, 2.0];
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        for &sc in &scales {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&make_q1_block(sc, &bits));
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q1_0G128,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q1_0G128,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q1_0G128Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q1_0G128Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "gemv row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }
}
