//! AVX2+FMA accelerated Q4_0 quantization kernel.
//!
//! Q4_0 block layout (18 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..18]  — 16 packed bytes encoding 32 × 4-bit unsigned nibbles
//!
//! Each weight reconstructs as `(nibble − 8) × d`.
//! Nibble order: for byte `b[i]`, `lo = b[i] & 0x0F` → weight `2i`,
//!                                `hi = b[i] >> 4`   → weight `2i+1`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (FP16 scale) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 18;

/// AVX2+FMA accelerated Q4_0 kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q4_0Avx2;

impl QuantKernel for Q4_0Avx2 {
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

        // SAFETY: we verified block.len() >= 18 and output.len() >= 32 above.
        // The CPU features avx2+fma are guaranteed by KernelDispatcher.
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
            // SAFETY: row and block bounds are checked above (n_rows, n_cols).
            // CPU avx2+fma support is guaranteed by KernelDispatcher.
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
        "Q4_0"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 18-byte Q4_0 block to 32 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 18`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 18 ≥ 2 — guaranteed by caller.
    let d = f16_to_f32(block);

    let vd = _mm256_set1_ps(d);

    // Load the 16 nibble bytes.
    // SAFETY: block.ptr + 2 is valid because block.len() >= 18.
    let raw = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);

    // Split each byte into its low and high nibble.
    // Note: there is no _mm_srli_epi8 in x86.  We use _mm_srli_epi16
    // (16-bit right shift) and then mask each byte to 0x0F to strip
    // the cross-byte contamination introduced by the 16-bit shift.
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, mask_lo); // low nibbles in each byte
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo); // high nibbles

    // Interleave: first16 = [lo0,hi0,lo1,hi1,...,lo7,hi7]  (weights 0-15)
    //             last16  = [lo8,hi8,...,lo15,hi15]         (weights 16-31)
    let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
    let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

    // Convert i8→i32→f32 in four groups of 8, subtract 8, scale by d.
    // Groups: first16[0..8], first16[8..16], last16[0..8], last16[8..16]

    let eight_i32 = _mm256_set1_epi32(8);

    // Group A: first 8 weights from first16
    let a_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16), eight_i32);
    let a_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(a_i32), vd);

    // Group B: next 8 weights from first16 (shifted by 8 bytes)
    let first16_hi = _mm_srli_si128(first16, 8);
    let b_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16_hi), eight_i32);
    let b_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i32), vd);

    // Group C: first 8 weights from last16
    let c_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16), eight_i32);
    let c_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(c_i32), vd);

    // Group D: next 8 weights from last16
    let last16_hi = _mm_srli_si128(last16, 8);
    let d_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16_hi), eight_i32);
    let d_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(d_i32), vd);

    // Store all 32 values.
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, a_f32);
    _mm256_storeu_ps(ptr.add(8), b_f32);
    _mm256_storeu_ps(ptr.add(16), c_f32);
    _mm256_storeu_ps(ptr.add(24), d_f32);
}

/// Compute the dot product of one row of a Q4_0 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    let eight_i32 = _mm256_set1_epi32(8);

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 18 ≥ 2.
        let d = f16_to_f32(block);

        // Load 16 nibble bytes.
        // SAFETY: block.ptr + 2 valid because BLOCK_BYTES == 18.
        let raw = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);

        let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_bytes = _mm_and_si128(raw, mask_lo);
        let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo);

        let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
        let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

        // Check whether this block is fully within bounds.
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 32 weights are valid — use 4 AVX2 FMA lanes.
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A (weights 0-7)
            let wa_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16), eight_i32);
            let wa_f32 = _mm256_cvtepi32_ps(wa_i32);
            let ia = _mm256_loadu_ps(inp_ptr);
            let mut acc = _mm256_mul_ps(wa_f32, ia);

            // Group B (weights 8-15)
            let first16_hi = _mm_srli_si128(first16, 8);
            let wb_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(first16_hi), eight_i32);
            let wb_f32 = _mm256_cvtepi32_ps(wb_i32);
            let ib = _mm256_loadu_ps(inp_ptr.add(8));
            acc = _mm256_fmadd_ps(wb_f32, ib, acc);

            // Group C (weights 16-23)
            let wc_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16), eight_i32);
            let wc_f32 = _mm256_cvtepi32_ps(wc_i32);
            let ic = _mm256_loadu_ps(inp_ptr.add(16));
            acc = _mm256_fmadd_ps(wc_f32, ic, acc);

            // Group D (weights 24-31)
            let last16_hi = _mm_srli_si128(last16, 8);
            let wd_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(last16_hi), eight_i32);
            let wd_f32 = _mm256_cvtepi32_ps(wd_i32);
            let id = _mm256_loadu_ps(inp_ptr.add(24));
            acc = _mm256_fmadd_ps(wd_f32, id, acc);

            row_sum += hsum_f32_avx(acc) * d;
        } else if remaining > 0 {
            // Tail path: partial block — fall back to scalar to avoid OOB reads.
            // Reconstruct nibbles from raw bytes in the block.
            let mut partial_sum = 0.0f32;
            for i in 0..BLOCK_SIZE / 2 {
                let byte = *block.get_unchecked(2 + i);
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                let idx = input_offset + i * 2;
                if idx + 1 < n_cols {
                    partial_sum += lo as f32 * input[idx];
                    partial_sum += hi as f32 * input[idx + 1];
                } else if idx < n_cols {
                    partial_sum += lo as f32 * input[idx];
                }
            }
            row_sum += partial_sum * d;
        }
        // remaining == 0: block is fully out of bounds, skip
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q4_0::Q4_0Ref;

    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    /// Build a single-row QuantTensor from a raw block.
    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q4_0)
    }

    #[test]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // skip on machines without AVX2
        }
        let nibbles: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB,
            0xED, 0x0F,
        ];
        let block = make_q4_0_block(0.25, &nibbles);

        let mut out_avx2 = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        let avx2 = Q4_0Avx2;
        let refk = Q4_0Ref;

        avx2.dequant_block(&block, &mut out_avx2).unwrap();
        refk.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let nibbles: [u8; 16] = [
            0x89, 0x7A, 0x6B, 0x5C, 0x4D, 0x3E, 0x2F, 0x10, 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5,
            0x96, 0x87,
        ];
        let scale = 0.5f32;
        let block = make_q4_0_block(scale, &nibbles);
        let tensor_avx2 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        // Use distinct input values to detect permutation bugs.
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-4,
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
        // 20 columns — one partial block of 20 out of 32
        let nibbles = [0x88u8; 16]; // all zeros after -8
        let block = make_q4_0_block(1.0, &nibbles);
        let tensor_avx2 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-4,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemm_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 2 rows, 32 cols, batch of 2 inputs (M=2)
        let nibbles: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB,
            0xED, 0x0F,
        ];
        let block = make_q4_0_block(0.25, &nibbles);
        let two_row_data = [block.as_slice(), block.as_slice()].concat();
        let tensor_avx2 = QuantTensor::new(
            two_row_data.clone(),
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let tensor_ref = QuantTensor::new(
            two_row_data,
            vec![2, 32],
            oxillama_gguf::GgufTensorType::Q4_0,
        );

        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let mut out_avx2 = vec![0.0f32; 4];
        let mut out_ref = vec![0.0f32; 4];

        Q4_0Avx2
            .gemm(&tensor_avx2, &input, &mut out_avx2, 2, 2, 32)
            .unwrap();
        Q4_0Ref
            .gemm(&tensor_ref, &input, &mut out_ref, 2, 2, 32)
            .unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "gemm mismatch at [{i}]: avx2={a}, ref={r}"
            );
        }
    }
}
