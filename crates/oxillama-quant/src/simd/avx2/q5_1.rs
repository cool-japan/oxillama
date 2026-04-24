//! AVX2+FMA accelerated Q5_1 quantization kernel.
//!
//! Q5_1 block layout (24 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 minimum `m` (little-endian)
//! - bytes[4..8]   — `qh`: 32 high bits (bit 4 of each 5-bit quant), u32 LE
//! - bytes[8..24]  — `qs`: 32 × lower 4 bits packed (2 per byte)
//!
//! Weight layout (sequential):
//!   output[i]      = d * (qs[i].lo4 | ((qh >> i)      & 1) << 4) + m  for i in 0..16
//!   output[i + 16] = d * (qs[i].hi4 | ((qh >> (i+16)) & 1) << 4) + m  for i in 0..16
//!
//! Q5_1 is UNSIGNED ([0..31]): do NOT apply Q5_0's -16 bias.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q5_1 block.
pub const BLOCK_BYTES: usize = 24;

/// AVX2+FMA accelerated Q5_1 kernel.
#[allow(non_camel_case_types)]
pub struct Q5_1Avx2;

impl QuantKernel for Q5_1Avx2 {
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
        "Q5_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 24-byte Q5_1 block to 32 FP32 values using AVX2.
///
/// Q5_1 is unsigned ([0..31]): the combined 5-bit value is used directly
/// (no -16 centering). Affine formula: `d * q5 + m`.
///
/// # Safety
/// - `block.len() >= 24`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let m = f16_to_f32(&block[2..]);
    let vd = _mm256_set1_ps(d);
    let vm = _mm256_set1_ps(m);

    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = _mm_loadu_si128(block.as_ptr().add(8) as *const __m128i);

    let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_nib = _mm_and_si128(qs, mask4);
    let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4);

    // Expand qh into per-element high bits
    let qh_lo_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> i) & 1) as u8);
    let qh_hi_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> (i + 16)) & 1) as u8);

    let vqh_lo = _mm_loadu_si128(qh_lo_bits.as_ptr() as *const __m128i);
    let vqh_hi = _mm_loadu_si128(qh_hi_bits.as_ptr() as *const __m128i);

    // Shift the 5th bit into position 4
    let vqh_lo_shifted = _mm_and_si128(
        _mm_slli_epi16(_mm_and_si128(vqh_lo, _mm_set1_epi8(1_i8)), 4),
        _mm_set1_epi8(0x10_u8 as i8),
    );
    let vqh_hi_shifted = _mm_and_si128(
        _mm_slli_epi16(_mm_and_si128(vqh_hi, _mm_set1_epi8(1_i8)), 4),
        _mm_set1_epi8(0x10_u8 as i8),
    );

    // Combine low nibble + high bit (unsigned 5-bit, range [0..31])
    let q5_lo = _mm_or_si128(lo_nib, vqh_lo_shifted);
    let q5_hi = _mm_or_si128(hi_nib, vqh_hi_shifted);

    // Process groups of 8: convert u8→u32→f32, then FMA: d*q + m
    let lo_f32_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_lo)), vd, vm);

    let q5_lo_hi8 = _mm_srli_si128(q5_lo, 8);
    let lo_f32_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_lo_hi8)), vd, vm);

    let hi_f32_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_hi)), vd, vm);

    let q5_hi_hi8 = _mm_srli_si128(q5_hi, 8);
    let hi_f32_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_hi_hi8)), vd, vm);

    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, lo_f32_a);
    _mm256_storeu_ps(ptr.add(8), lo_f32_b);
    _mm256_storeu_ps(ptr.add(16), hi_f32_a);
    _mm256_storeu_ps(ptr.add(24), hi_f32_b);
}

/// Compute the dot product of one row of a Q5_1 matrix with an FP32 vector.
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
    let mut acc = _mm256_setzero_ps();

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let col_start = blk * BLOCK_SIZE;
        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let inp = &input[col_start..col_end];

        let d = f16_to_f32(block);
        let m = f16_to_f32(&block[2..]);
        let vd = _mm256_set1_ps(d);

        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = _mm_loadu_si128(block.as_ptr().add(8) as *const __m128i);

        let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
        let lo_nib = _mm_and_si128(qs, mask4);
        let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4);

        let qh_lo_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> i) & 1) as u8);
        let qh_hi_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> (i + 16)) & 1) as u8);

        let vqh_lo = _mm_loadu_si128(qh_lo_bits.as_ptr() as *const __m128i);
        let vqh_hi = _mm_loadu_si128(qh_hi_bits.as_ptr() as *const __m128i);

        let vqh_lo_shifted = _mm_and_si128(
            _mm_slli_epi16(_mm_and_si128(vqh_lo, _mm_set1_epi8(1_i8)), 4),
            _mm_set1_epi8(0x10_u8 as i8),
        );
        let vqh_hi_shifted = _mm_and_si128(
            _mm_slli_epi16(_mm_and_si128(vqh_hi, _mm_set1_epi8(1_i8)), 4),
            _mm_set1_epi8(0x10_u8 as i8),
        );

        let q5_lo = _mm_or_si128(lo_nib, vqh_lo_shifted);
        let q5_hi = _mm_or_si128(hi_nib, vqh_hi_shifted);

        let avail = inp.len();

        // Group A: output[0..8] × input[0..8]
        let n_a = avail.min(8);
        let mut inp_a = [0.0f32; 8];
        inp_a[..n_a].copy_from_slice(&inp[..n_a]);
        let vinp_a = _mm256_loadu_ps(inp_a.as_ptr());
        let vq_a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_lo));
        // acc += (d * q5 + m) * inp = d * q5 * inp + m * inp
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vq_a, vd), vinp_a, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(m), vinp_a, acc);

        // Group B: output[8..16] × input[8..16]
        let n_b = avail.saturating_sub(8).min(8);
        let mut inp_b = [0.0f32; 8];
        if n_b > 0 {
            inp_b[..n_b].copy_from_slice(&inp[8..8 + n_b]);
        }
        let vinp_b = _mm256_loadu_ps(inp_b.as_ptr());
        let q5_lo_hi8 = _mm_srli_si128(q5_lo, 8);
        let vq_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_lo_hi8));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vq_b, vd), vinp_b, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(m), vinp_b, acc);

        // Group C: output[16..24] × input[16..24]
        let n_c = avail.saturating_sub(16).min(8);
        let mut inp_c = [0.0f32; 8];
        if n_c > 0 {
            inp_c[..n_c].copy_from_slice(&inp[16..16 + n_c]);
        }
        let vinp_c = _mm256_loadu_ps(inp_c.as_ptr());
        let vq_c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_hi));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vq_c, vd), vinp_c, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(m), vinp_c, acc);

        // Group D: output[24..32] × input[24..32]
        let n_d = avail.saturating_sub(24).min(8);
        let mut inp_d = [0.0f32; 8];
        if n_d > 0 {
            inp_d[..n_d].copy_from_slice(&inp[24..24 + n_d]);
        }
        let vinp_d = _mm256_loadu_ps(inp_d.as_ptr());
        let q5_hi_hi8 = _mm_srli_si128(q5_hi, 8);
        let vq_d = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q5_hi_hi8));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vq_d, vd), vinp_d, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(m), vinp_d, acc);
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::reference::q5_1::Q5_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    fn avx2_available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn dequant_matches_reference_zeros() {
        if !avx2_available() {
            return;
        }
        let block = make_block(0.0, 0.0, 0, &[0; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn dequant_matches_reference_max() {
        if !avx2_available() {
            return;
        }
        // qh = 0xFFFFFFFF, qs = 0xFF → q5=31; d=1.0, m=0.0 → weight=31.0
        let block = make_block(1.0, 0.0, 0xFFFF_FFFF, &[0xFF; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn dequant_matches_reference_alternating() {
        if !avx2_available() {
            return;
        }
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q5_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r:.6} avx2={a:.6}");
        }
    }

    #[test]
    fn empty_block_errors() {
        let mut out = vec![0.0f32; BLOCK_SIZE];
        let err = Q5_1Avx2.dequant_block(&[], &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn gemv_matches_reference() {
        if !avx2_available() {
            return;
        }
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);

        let tensor = QuantTensor {
            data: block.clone().into(),
            shape: vec![1, BLOCK_SIZE],
            tensor_type: oxillama_gguf::GgufTensorType::Q5_1,
        };

        let input: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let mut ref_out = vec![0.0f32; 1];
        let mut avx2_out = vec![0.0f32; 1];

        Q5_1Ref
            .gemv(&tensor, &input, &mut ref_out)
            .expect("ref gemv");
        Q5_1Avx2
            .gemv(&tensor, &input, &mut avx2_out)
            .expect("avx2 gemv");

        assert!(
            (ref_out[0] - avx2_out[0]).abs() < 1e-3,
            "gemv: ref={} avx2={}",
            ref_out[0],
            avx2_out[0]
        );
    }
}
