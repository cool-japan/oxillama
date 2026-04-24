//! AVX2+FMA accelerated Q4_1 quantization kernel.
//!
//! Q4_1 block layout (20 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 minimum `m` (little-endian)
//! - bytes[4..20]  — 16 packed bytes encoding 32 × 4-bit unsigned nibbles
//!
//! Each weight reconstructs as `nibble × d + m`.
//! Nibble order: for byte `b[i]`, `lo = b[i] & 0x0F` → weight `2i`,
//!                                `hi = b[i] >> 4`   → weight `2i+1`.
//! Nibbles are unsigned (0..=15).

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q4_1 block: 2 (FP16 scale d) + 2 (FP16 min m) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 20;

/// AVX2+FMA accelerated Q4_1 kernel.
///
/// Requires `avx2` and `fma` CPU features. The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q4_1Avx2;

impl QuantKernel for Q4_1Avx2 {
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

        // SAFETY: we verified block.len() >= 20 and output.len() >= 32 above.
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
            // SAFETY: row and block bounds are checked above.
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
        "Q4_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 20-byte Q4_1 block to 32 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 20`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale d and minimum m.
    // SAFETY: block.len() >= 20 ≥ 4 — guaranteed by caller.
    let d = f16_to_f32(block);
    let m = f16_to_f32(&block[2..]);

    let vd = _mm256_set1_ps(d);
    let vm = _mm256_set1_ps(m);

    // Load 16 nibble bytes (qs starts at byte offset 4).
    // SAFETY: block.ptr + 4 valid because block.len() >= 20.
    let raw = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);

    // Split each byte into its low and high nibble.
    // Note: _mm_srli_epi16 shifts each 16-bit lane, so we mask to 0x0F
    // to strip cross-byte contamination.
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, mask_lo); // low nibbles per byte
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo); // high nibbles per byte

    // Interleave: first16 = [lo0,hi0,lo1,hi1,...,lo7,hi7]  (weights 0-15)
    //             last16  = [lo8,hi8,...,lo15,hi15]         (weights 16-31)
    let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
    let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

    // Convert u8→i32→f32 in four groups of 8, scale by d, add m.
    // Groups: first16[0..8], first16[8..16], last16[0..8], last16[8..16]

    // Group A: first 8 weights from first16
    let a_u32 = _mm256_cvtepu8_epi32(first16);
    let a_f32 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a_u32), vd, vm);

    // Group B: next 8 weights from first16 (shift by 8 bytes)
    let first16_hi = _mm_srli_si128(first16, 8);
    let b_u32 = _mm256_cvtepu8_epi32(first16_hi);
    let b_f32 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(b_u32), vd, vm);

    // Group C: first 8 weights from last16
    let c_u32 = _mm256_cvtepu8_epi32(last16);
    let c_f32 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c_u32), vd, vm);

    // Group D: next 8 weights from last16
    let last16_hi = _mm_srli_si128(last16, 8);
    let d_u32 = _mm256_cvtepu8_epi32(last16_hi);
    let d_f32 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d_u32), vd, vm);

    // Store all 32 values.
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, a_f32);
    _mm256_storeu_ps(ptr.add(8), b_f32);
    _mm256_storeu_ps(ptr.add(16), c_f32);
    _mm256_storeu_ps(ptr.add(24), d_f32);
}

/// Compute the dot product of one row of a Q4_1 matrix with an FP32 vector.
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
    let mut row_sum = _mm256_setzero_ps();
    let mut m_sum = _mm256_setzero_ps(); // accumulates input values for m adjustment

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale d and minimum m.
        // SAFETY: block.len() == BLOCK_BYTES == 20 ≥ 4.
        let d = f16_to_f32(block);
        let m = f16_to_f32(&block[2..]);

        let vd = _mm256_set1_ps(d);
        let vm = _mm256_set1_ps(m);

        // Load 16 nibble bytes.
        // SAFETY: block.ptr + 4 valid because BLOCK_BYTES == 20.
        let raw = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);

        let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_bytes = _mm_and_si128(raw, mask_lo);
        let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo);

        let first16 = _mm_unpacklo_epi8(lo_bytes, hi_bytes);
        let last16 = _mm_unpackhi_epi8(lo_bytes, hi_bytes);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 32 weights are valid.
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A (weights 0-7): nibbles × d
            let wa_i32 = _mm256_cvtepu8_epi32(first16);
            let wa_f32 = _mm256_cvtepi32_ps(wa_i32);
            let ia = _mm256_loadu_ps(inp_ptr);
            // acc += (d * w) * i = d * w * i; also acc_m += i * m
            row_sum = _mm256_fmadd_ps(_mm256_mul_ps(wa_f32, vd), ia, row_sum);
            m_sum = _mm256_fmadd_ps(vm, ia, m_sum);

            // Group B (weights 8-15)
            let first16_hi = _mm_srli_si128(first16, 8);
            let wb_i32 = _mm256_cvtepu8_epi32(first16_hi);
            let wb_f32 = _mm256_cvtepi32_ps(wb_i32);
            let ib = _mm256_loadu_ps(inp_ptr.add(8));
            row_sum = _mm256_fmadd_ps(_mm256_mul_ps(wb_f32, vd), ib, row_sum);
            m_sum = _mm256_fmadd_ps(vm, ib, m_sum);

            // Group C (weights 16-23)
            let wc_i32 = _mm256_cvtepu8_epi32(last16);
            let wc_f32 = _mm256_cvtepi32_ps(wc_i32);
            let ic = _mm256_loadu_ps(inp_ptr.add(16));
            row_sum = _mm256_fmadd_ps(_mm256_mul_ps(wc_f32, vd), ic, row_sum);
            m_sum = _mm256_fmadd_ps(vm, ic, m_sum);

            // Group D (weights 24-31)
            let last16_hi = _mm_srli_si128(last16, 8);
            let wd_i32 = _mm256_cvtepu8_epi32(last16_hi);
            let wd_f32 = _mm256_cvtepi32_ps(wd_i32);
            let id = _mm256_loadu_ps(inp_ptr.add(24));
            row_sum = _mm256_fmadd_ps(_mm256_mul_ps(wd_f32, vd), id, row_sum);
            m_sum = _mm256_fmadd_ps(vm, id, m_sum);
        } else {
            // Partial block: scalar fallback for tail.
            let mut partial = [0.0f32; BLOCK_SIZE];
            let partial_ptr = partial.as_mut_ptr();
            let a_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(first16));
            let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(first16, 8)));
            let c_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(last16));
            let d_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(last16, 8)));
            _mm256_storeu_ps(partial_ptr, a_f32);
            _mm256_storeu_ps(partial_ptr.add(8), b_f32);
            _mm256_storeu_ps(partial_ptr.add(16), c_f32);
            _mm256_storeu_ps(partial_ptr.add(24), d_f32);

            let mut scalar_sum = 0.0f32;
            for j in 0..remaining {
                let w = partial[j];
                scalar_sum += (d * w + m) * input[input_offset + j];
            }
            row_sum = _mm256_add_ps(row_sum, _mm256_set1_ps(scalar_sum));
        }
    }

    // m_sum already holds the m * input contributions — they were accumulated above.
    // row_sum holds the nibble*d*input contributions + m*input contributions.
    hsum_f32_avx(row_sum)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d: f32, m: f32, nibbles: &[u8; 32]) -> Vec<u8> {
        let d_f16 = half::f16::from_f32(d);
        let m_f16 = half::f16::from_f32(m);
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&d_f16.to_le_bytes());
        block.extend_from_slice(&m_f16.to_le_bytes());
        // Pack nibbles into 16 bytes, 2 per byte (lo first).
        for i in 0..16 {
            let lo = nibbles[2 * i] & 0x0F;
            let hi = nibbles[2 * i + 1] & 0x0F;
            block.push(lo | (hi << 4));
        }
        block
    }

    #[test]
    fn test_dequant_all_zeros() {
        let nibbles = [0u8; 32];
        let block = make_block(1.0, 0.0, &nibbles);
        let mut output = [0.0f32; BLOCK_SIZE];
        let kernel = Q4_1Avx2;
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v - 0.0).abs() < 1e-6, "expected 0 got {v}");
        }
    }

    #[test]
    fn test_dequant_all_max_nibble() {
        let nibbles = [15u8; 32];
        let block = make_block(1.0, 0.0, &nibbles);
        let mut output = [0.0f32; BLOCK_SIZE];
        let kernel = Q4_1Avx2;
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v - 15.0).abs() < 1e-4, "expected 15 got {v}");
        }
    }

    #[test]
    fn test_dequant_with_bias() {
        let nibbles = [0u8; 32];
        let block = make_block(0.5, 2.0, &nibbles);
        let mut output = [0.0f32; BLOCK_SIZE];
        let kernel = Q4_1Avx2;
        kernel.dequant_block(&block, &mut output).unwrap();
        // w = 0 * 0.5 + 2.0 = 2.0
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-4, "expected 2.0 got {v}");
        }
    }

    #[test]
    fn test_dequant_nibble_ordering() {
        // lo nibble of byte 0 = 3, hi nibble of byte 0 = 7
        let mut nibbles = [0u8; 32];
        nibbles[0] = 3;
        nibbles[1] = 7;
        let block = make_block(1.0, 0.0, &nibbles);
        let mut output = [0.0f32; BLOCK_SIZE];
        let kernel = Q4_1Avx2;
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!((output[0] - 3.0).abs() < 1e-4, "output[0]={}", output[0]);
        assert!((output[1] - 7.0).abs() < 1e-4, "output[1]={}", output[1]);
    }

    #[test]
    fn test_matches_reference_scalar() {
        use crate::reference::Q4_1Ref;

        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = make_block(0.25, 1.0, &nibbles);

        let mut ref_out = [0.0f32; BLOCK_SIZE];
        let mut avx_out = [0.0f32; BLOCK_SIZE];

        Q4_1Ref.dequant_block(&block, &mut ref_out).unwrap();
        Q4_1Avx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!(
                (r - a).abs() < 1e-4,
                "mismatch at index {i}: ref={r} avx={a}"
            );
        }
    }
}
