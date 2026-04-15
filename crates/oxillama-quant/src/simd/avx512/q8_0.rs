//! AVX-512 accelerated Q8_0 quantization kernel.
//!
//! Q8_0 block layout (34 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..34]  — 32 × int8 signed quantized values
//!
//! Each weight reconstructs as `q[i] × d`.
//!
//! ## AVX-512 strategy
//!
//! Process 32 int8 values in **two** AVX-512 (16-wide) passes instead of
//! the AVX2 kernel's four 8-wide passes:
//!
//! 1. Load first 16 i8 → `_mm512_cvtepi8_epi32` → 16 × i32 → f32 → scale by d.
//! 2. Load next 16 i8 → same widening → scale.
//!
//! ~2× throughput improvement vs AVX2 on machines with AVX-512F.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (FP16 scale) + 32 (i8 data).
pub const BLOCK_BYTES: usize = 34;

/// AVX-512 accelerated Q8_0 kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
pub struct Q8_0Avx512;

impl QuantKernel for Q8_0Avx512 {
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

        // SAFETY: block.len() >= 34 and output.len() >= 32 verified above.
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
        "Q8_0"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 34-byte Q8_0 block to 32 FP32 values using AVX-512.
///
/// # Safety
/// - `block.len() >= 34`
/// - `output.len() >= 32`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 34 >= 2.
    let d = f16_to_f32(block);
    let vd = _mm512_set1_ps(d);

    // Pass 1: first 16 int8 values → 16 × i32 → 16 × f32 → scale.
    // SAFETY: block.ptr + 2 is valid; block.len() >= 34 guarantees 32 bytes available.
    let q1 = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i); // 16 bytes
    let q1_i32 = _mm512_cvtepi8_epi32(q1); // sign-extend 16 × i8 → 16 × i32
    let q1_f32 = _mm512_cvtepi32_ps(q1_i32);
    let w1 = _mm512_mul_ps(q1_f32, vd);

    // Pass 2: next 16 int8 values → same widening.
    // SAFETY: block.ptr + 18 is valid; block.len() >= 34 guarantees bytes 18..34.
    let q2 = _mm_loadu_si128(block.as_ptr().add(18) as *const __m128i); // bytes 18..34
    let q2_i32 = _mm512_cvtepi8_epi32(q2);
    let q2_f32 = _mm512_cvtepi32_ps(q2_i32);
    let w2 = _mm512_mul_ps(q2_f32, vd);

    // Store results.
    // SAFETY: output.len() >= 32 guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm512_storeu_ps(ptr, w1);
    _mm512_storeu_ps(ptr.add(16), w2);
}

/// Compute the dot product of one row of a Q8_0 matrix with an FP32 vector.
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
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 34 >= 2.
        let d = f16_to_f32(block);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block — 2 AVX-512 passes of 16.
            // SAFETY: block.ptr + 2 valid; block.len() == 34.
            let q1 = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);
            let q2 = _mm_loadu_si128(block.as_ptr().add(18) as *const __m128i);

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
                // SAFETY: block[2 + i] valid because remaining <= BLOCK_SIZE == 32
                // and BLOCK_BYTES == 34 = 2 + 32.
                let q = *block.get_unchecked(2 + i) as i8;
                partial_sum += q as f32 * input[input_offset + i];
            }
            row_sum += partial_sum * d;
        }
        // remaining == 0: out of bounds, skip
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::q8_0::Q8_0Ref;

    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q8_0)
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut values = [0i8; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as i8).wrapping_sub(16);
        }
        let block = make_q8_0_block(0.5, &values);

        let mut out_avx512 = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        Q8_0Avx512.dequant_block(&block, &mut out_avx512).unwrap();
        Q8_0Ref.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut values = [0i8; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as i32) - 10) as i8;
        }
        let block = make_q8_0_block(0.25, &values);
        let tensor_avx512 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();

        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-4,
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
        let values = [1i8; 32];
        let block = make_q8_0_block(1.0, &values);
        let tensor_avx512 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-4,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_negative_weights() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut values = [0i8; 32];
        values[0] = -128;
        values[31] = 127;
        let block = make_q8_0_block(2.0, &values);
        let tensor_avx512 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let mut input = vec![0.0f32; 32];
        input[0] = 1.0;
        input[31] = 1.0;

        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-2,
            "negative weight gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }
}
