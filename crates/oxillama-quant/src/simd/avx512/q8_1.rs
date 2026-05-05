//! AVX-512 accelerated Q8_1 quantization kernel.
//!
//! Q8_1 block layout (36 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..4]   — FP16 partial sum `s` = d × Σqs (unused in GEMV, stored for GEMM opt)
//! - bytes[4..36]  — 32 × int8 signed quantized values
//!
//! Each weight reconstructs as `d × qs[i]`.
//!
//! ## AVX-512 strategy
//!
//! Process 32 int8 values in **two** AVX-512 (16-wide) passes, mirroring the
//! Q8_0 AVX-512 kernel:
//!
//! 1. Load first 16 i8 from bytes[4..20] → `_mm512_cvtepi8_epi32` → 16 × i32 → f32 → scale d.
//! 2. Load next 16 i8 from bytes[20..36] → same widening → scale d.
//!
//! Key difference from Q8_0: block stride is 36 bytes (not 34), and the i8
//! quantized values start at offset 4 (not 2), because bytes[2..4] hold the
//! precomputed partial sum `s`.
//!
//! ~2× throughput vs AVX2 on machines with AVX-512F support.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_1: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q8_1 block: 2 (FP16 scale d) + 2 (FP16 sum s) + 32 (i8 data).
pub const BLOCK_BYTES: usize = 36;

/// AVX-512 accelerated Q8_1 kernel.
///
/// Requires the `avx512f` CPU feature. The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q8_1Avx512;

impl QuantKernel for Q8_1Avx512 {
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

        // SAFETY: block.len() >= 36 and output.len() >= 32 verified above.
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
        "Q8_1"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 36-byte Q8_1 block to 32 FP32 values using AVX-512.
///
/// Mirrors reference: `d × qs[i]`. The `s` field at bytes[2..4] is not used
/// in dequantization (it is only exploited in quantized–quantized GEMM).
///
/// # Safety
/// - `block.len() >= 36`
/// - `output.len() >= 32`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 36 >= 2.
    let d = f16_to_f32(block);
    let vd = _mm512_set1_ps(d);

    // Pass 1: first 16 int8 values at bytes[4..20].
    // SAFETY: block.ptr + 4 valid (block.len() >= 36); 16 bytes available.
    let q1 = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);
    let q1_i32 = _mm512_cvtepi8_epi32(q1); // sign-extend 16 × i8 → 16 × i32
    let q1_f32 = _mm512_cvtepi32_ps(q1_i32);
    let w1 = _mm512_mul_ps(q1_f32, vd);

    // Pass 2: next 16 int8 values at bytes[20..36].
    // SAFETY: block.ptr + 20 valid; block.len() >= 36 guarantees bytes 20..36.
    let q2 = _mm_loadu_si128(block.as_ptr().add(20) as *const __m128i);
    let q2_i32 = _mm512_cvtepi8_epi32(q2);
    let q2_f32 = _mm512_cvtepi32_ps(q2_i32);
    let w2 = _mm512_mul_ps(q2_f32, vd);

    // Store results.
    // SAFETY: output.len() >= 32 guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm512_storeu_ps(ptr, w1);
    _mm512_storeu_ps(ptr.add(16), w2);
}

/// Compute the dot product of one row of a Q8_1 matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// Mirrors reference exactly: `d × Σ(qs[i] × inp[i])`.  The `s` field is
/// unused — plain FP32 activations do not benefit from the precomputed sum.
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
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 36 >= 2.
        let d = f16_to_f32(block);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block — 2 AVX-512 passes of 16 i8 values each.
            // SAFETY: block.ptr + 4 valid; block.len() == 36.
            let q1 = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);
            // SAFETY: block.ptr + 20 valid; block.len() == 36.
            let q2 = _mm_loadu_si128(block.as_ptr().add(20) as *const __m128i);

            let vd = _mm512_set1_ps(d);
            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Pass 1: i8[0..16] × input[0..16]
            let w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), vd);
            let i1 = _mm512_loadu_ps(inp_ptr);
            let acc = _mm512_mul_ps(w1, i1);

            // Pass 2: i8[16..32] × input[16..32]
            let w2 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2)), vd);
            let i2 = _mm512_loadu_ps(inp_ptr.add(16));
            let acc = _mm512_fmadd_ps(w2, i2, acc);

            row_sum += hsum_f32_avx512(acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid OOB reads.
            let mut partial_sum = 0.0f32;
            for i in 0..remaining {
                // SAFETY: block[4 + i] valid because remaining <= BLOCK_SIZE == 32
                // and BLOCK_BYTES == 36 = 4 + 32.
                let q = *block.get_unchecked(4 + i) as i8;
                partial_sum += q as f32 * input[input_offset + i];
            }
            row_sum += partial_sum * d;
        }
        // remaining == 0: out of bounds, skip
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::q8_1::Q8_1Ref;

    /// Build a valid Q8_1 block from a scale and 32 int8 quantized values.
    ///
    /// The `s` field is computed as `d × Σqs` per the format spec, though it
    /// is not used in plain-FP32 GEMV.
    fn make_q8_1_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        // s = d × sum(qs), stored as FP16
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        let s_bits = half::f16::from_f32(s).to_bits();
        block.extend_from_slice(&s_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    fn make_tensor(data: Vec<u8>, n_rows: usize, n_cols: usize) -> QuantTensor {
        QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_1,
        )
    }

    fn avx512_available() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    /// Dequantize one full block and compare against the scalar reference.
    /// Tolerance: 1e-6 (FP16 scale rounding dominates over FP32 arithmetic).
    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn avx512_q8_1_dequant_matches_reference() {
        if !avx512_available() {
            return;
        }
        let mut values = [0i8; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as i8).wrapping_sub(16);
        }
        let block = make_q8_1_block(0.5, &values);

        let mut out_avx512 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ref = vec![0.0f32; BLOCK_SIZE];

        Q8_1Avx512.dequant_block(&block, &mut out_avx512).unwrap();
        Q8_1Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-6,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    /// 64×1024 GEMV: 64 rows, 1024 columns (32 full Q8_1 blocks per row).
    /// Compare AVX-512 result against scalar reference with tolerance 1e-5.
    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn avx512_q8_1_matvec_matches_reference() {
        if !avx512_available() {
            return;
        }
        const N_ROWS: usize = 64;
        const N_COLS: usize = 1024;
        const BLOCKS_PER_ROW: usize = N_COLS / BLOCK_SIZE; // == 32

        // Build weight tensor: N_ROWS rows × BLOCKS_PER_ROW blocks, each with
        // a distinct pattern so the test is not trivially degenerate.
        let mut weight_data = Vec::with_capacity(N_ROWS * BLOCKS_PER_ROW * BLOCK_BYTES);
        for row in 0..N_ROWS {
            for blk in 0..BLOCKS_PER_ROW {
                let scale = 0.01 * (row as f32 + 1.0) * (blk as f32 * 0.1 + 0.5);
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (((row * 7 + blk * 3 + i * 11) as i16 % 256) as i16 - 128).clamp(-128, 127)
                        as i8;
                }
                weight_data.extend_from_slice(&make_q8_1_block(scale, &qs));
            }
        }

        let tensor_avx512 = make_tensor(weight_data.clone(), N_ROWS, N_COLS);
        let tensor_ref = make_tensor(weight_data, N_ROWS, N_COLS);

        let input: Vec<f32> = (0..N_COLS).map(|i| (i as f32) * 0.001 - 0.5).collect();

        let mut out_avx512 = vec![0.0f32; N_ROWS];
        let mut out_ref = vec![0.0f32; N_ROWS];

        Q8_1Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q8_1Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "matvec mismatch at row {i}: avx512={a}, ref={r}"
            );
        }
    }

    /// GEMV where the column count is not a multiple of BLOCK_SIZE (32).
    /// Exercises the scalar partial-block tail path.
    /// Tolerance: 1e-5.
    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn avx512_q8_1_partial_block_gemv() {
        if !avx512_available() {
            return;
        }
        // Use n_cols = 1024 + 17 to force a partial tail block.
        const N_ROWS: usize = 4;
        const N_COLS: usize = 1024 + 17; // 33 blocks per row; last has 17 valid elements
        let blocks_per_row = N_COLS.div_ceil(BLOCK_SIZE); // == 33

        let mut weight_data = Vec::with_capacity(N_ROWS * blocks_per_row * BLOCK_BYTES);
        for row in 0..N_ROWS {
            for blk in 0..blocks_per_row {
                let scale = 0.25 * (row as f32 + 1.0);
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (((row * 13 + blk * 5 + i * 7) % 251) as i16 - 64).clamp(-128, 127) as i8;
                }
                weight_data.extend_from_slice(&make_q8_1_block(scale, &qs));
            }
        }

        let tensor_avx512 = make_tensor(weight_data.clone(), N_ROWS, N_COLS);
        let tensor_ref = make_tensor(weight_data, N_ROWS, N_COLS);

        let input: Vec<f32> = (0..N_COLS).map(|i| (i as f32) * 0.002 - 1.0).collect();

        let mut out_avx512 = vec![0.0f32; N_ROWS];
        let mut out_ref = vec![0.0f32; N_ROWS];

        Q8_1Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q8_1Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "partial gemv mismatch at row {i}: avx512={a}, ref={r}"
            );
        }
    }
}
