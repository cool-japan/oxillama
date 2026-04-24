//! AVX2+FMA accelerated Q8_1 quantization kernel.
//!
//! Q8_1 block layout (36 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 sum `s` = d * Σqs (stored but unused in GEMV)
//! - bytes[4..36]  — 32 × int8 signed quantised values
//!
//! GEMV mirrors `reference::q8_1::Q8_1Ref::gemv` exactly:
//!   plain f32 activations × `d * qs[i]`, `s` is not used.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q8_1 block.
pub const BLOCK_BYTES: usize = 36;

/// AVX2+FMA accelerated Q8_1 kernel.
#[allow(non_camel_case_types)]
pub struct Q8_1Avx2;

impl QuantKernel for Q8_1Avx2 {
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
        "Q8_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 36-byte Q8_1 block to 32 FP32 values using AVX2.
///
/// Mirrors reference: `d * qs[i]`. The `s` field at bytes[2..4] is unused.
///
/// # Safety
/// - `block.len() >= 36`
/// - `output.len() >= 32`
/// - CPU must support `avx2`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let vd = _mm256_set1_ps(d);

    // Load 32 int8 values from bytes[4..36]
    let q_raw = _mm256_loadu_si256(block.as_ptr().add(4) as *const __m256i);

    // Widen i8 → i16 → i32 → f32 in four groups of 8
    // Lower 128-bit lane (bytes 0..15 of q_raw = quant values 0..15)
    let q_lo128 = _mm256_castsi256_si128(q_raw);
    let q_lo_i16 = _mm256_cvtepi8_epi16(q_lo128);
    let q0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo_i16));
    let q1_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo_i16, 1));

    // Upper 128-bit lane (quant values 16..31)
    let q_hi128 = _mm256_extracti128_si256(q_raw, 1);
    let q_hi_i16 = _mm256_cvtepi8_epi16(q_hi128);
    let q2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi_i16));
    let q3_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi_i16, 1));

    let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0_i32), vd);
    let w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_i32), vd);
    let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_i32), vd);
    let w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_i32), vd);

    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, w0);
    _mm256_storeu_ps(ptr.add(8), w1);
    _mm256_storeu_ps(ptr.add(16), w2);
    _mm256_storeu_ps(ptr.add(24), w3);
}

/// Compute the dot product of one row of a Q8_1 matrix with an FP32 vector.
///
/// Mirrors reference exactly: `d * Σ(qs[i] * inp[i])`. The `s` field is unused.
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
        let vd = _mm256_set1_ps(d);

        // Load 32 int8 values from bytes[4..36]
        let q_raw = _mm256_loadu_si256(block.as_ptr().add(4) as *const __m256i);

        let q_lo128 = _mm256_castsi256_si128(q_raw);
        let q_lo_i16 = _mm256_cvtepi8_epi16(q_lo128);
        let q0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_lo_i16));
        let q1_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_lo_i16, 1));

        let q_hi128 = _mm256_extracti128_si256(q_raw, 1);
        let q_hi_i16 = _mm256_cvtepi8_epi16(q_hi128);
        let q2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_hi_i16));
        let q3_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_hi_i16, 1));

        let qf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0_i32), vd);
        let qf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_i32), vd);
        let qf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_i32), vd);
        let qf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_i32), vd);

        let avail = inp.len();

        let n_a = avail.min(8);
        let mut inp_a = [0.0f32; 8];
        inp_a[..n_a].copy_from_slice(&inp[..n_a]);
        let vi0 = _mm256_loadu_ps(inp_a.as_ptr());
        acc = _mm256_fmadd_ps(qf0, vi0, acc);

        let n_b = avail.saturating_sub(8).min(8);
        let mut inp_b = [0.0f32; 8];
        if n_b > 0 {
            inp_b[..n_b].copy_from_slice(&inp[8..8 + n_b]);
        }
        let vi1 = _mm256_loadu_ps(inp_b.as_ptr());
        acc = _mm256_fmadd_ps(qf1, vi1, acc);

        let n_c = avail.saturating_sub(16).min(8);
        let mut inp_c = [0.0f32; 8];
        if n_c > 0 {
            inp_c[..n_c].copy_from_slice(&inp[16..16 + n_c]);
        }
        let vi2 = _mm256_loadu_ps(inp_c.as_ptr());
        acc = _mm256_fmadd_ps(qf2, vi2, acc);

        let n_d = avail.saturating_sub(24).min(8);
        let mut inp_d = [0.0f32; 8];
        if n_d > 0 {
            inp_d[..n_d].copy_from_slice(&inp[24..24 + n_d]);
        }
        let vi3 = _mm256_loadu_ps(inp_d.as_ptr());
        acc = _mm256_fmadd_ps(qf3, vi3, acc);
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::reference::q8_1::Q8_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        block.extend_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
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
        let block = make_block(0.0, &[0; 32]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q8_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q8_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn dequant_matches_reference_mixed() {
        if !avx2_available() {
            return;
        }
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_block(0.5, &qs);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];
        Q8_1Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q8_1Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");
        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r:.6} avx2={a:.6}");
        }
    }

    #[test]
    fn empty_block_errors() {
        let mut out = vec![0.0f32; BLOCK_SIZE];
        let err = Q8_1Avx2.dequant_block(&[], &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn gemv_matches_reference() {
        if !avx2_available() {
            return;
        }
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_block(0.5, &qs);

        let tensor = QuantTensor {
            data: block.clone().into(),
            shape: vec![1, BLOCK_SIZE],
            tensor_type: oxillama_gguf::GgufTensorType::Q8_1,
        };

        let input: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let mut ref_out = vec![0.0f32; 1];
        let mut avx2_out = vec![0.0f32; 1];

        Q8_1Ref
            .gemv(&tensor, &input, &mut ref_out)
            .expect("ref gemv");
        Q8_1Avx2
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
