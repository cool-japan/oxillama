//! AVX2+FMA accelerated Q8_0 quantization kernel.
//!
//! Q8_0 block layout (34 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..34]  — 32 × int8 signed quantized values
//!
//! Each weight reconstructs as `q[i] × d`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (FP16 scale) + 32 (i8 data).
pub const BLOCK_BYTES: usize = 34;

/// AVX2+FMA accelerated Q8_0 kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q8_0Avx2;

impl QuantKernel for Q8_0Avx2 {
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
        // CPU avx2+fma support guaranteed by KernelDispatcher.
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
            // CPU avx2+fma support guaranteed by KernelDispatcher.
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
        "Q8_0"
    }

    /// Fused Q8_0 weight × Q8_0 activation GEMV using AVX2+FMA.
    ///
    /// Both weight and activation blocks share the same format (34 bytes).
    /// Accumulates into `out` (caller must zero if a fresh GEMV is desired).
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        let acts_needed = blocks_per_row * BLOCK_BYTES; // Q8_0 acts = same size as Q8_0 weights

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        for row in 0..n_rows {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked above; CPU avx2+fma guaranteed by KernelDispatcher.
            let row_sum = unsafe {
                fused_q8_0_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            out[row] += row_sum;
        }

        Ok(())
    }
}

/// Fused Q8_0 weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q8_0_q8_0_row_avx2(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_off = blk * BLOCK_BYTES;
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let w_block = &row_data[w_off..w_off + BLOCK_BYTES];
        let d_w = f16_to_f32(w_block);

        let a_off = blk * BLOCK_BYTES;
        // SAFETY: acts_q8.len() >= blocks_per_row * BLOCK_BYTES.
        let a_block = &acts_q8[a_off..a_off + BLOCK_BYTES];
        let d_a = f16_to_f32(a_block);

        let scale = d_w * d_a;

        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full 32-weight block.
            // SAFETY: w_block[2..34] = 32 i8 bytes; a_block[2..34] = 32 i8 bytes.
            let w256 = _mm256_loadu_si256(w_block.as_ptr().add(2) as *const __m256i);
            let a256 = _mm256_loadu_si256(a_block.as_ptr().add(2) as *const __m256i);

            let wlo = _mm256_castsi256_si128(w256);
            let whi = _mm256_extracti128_si256(w256, 1);
            let alo = _mm256_castsi256_si128(a256);
            let ahi = _mm256_extracti128_si256(a256, 1);

            // Group A (weights 0-7)
            let w_a = _mm256_cvtepi8_epi32(wlo);
            let a_a = _mm256_cvtepi8_epi32(alo);
            let mut acc = _mm256_mullo_epi32(w_a, a_a);

            // Group B (weights 8-15)
            let w_b = _mm256_cvtepi8_epi32(_mm_srli_si128(wlo, 8));
            let a_b = _mm256_cvtepi8_epi32(_mm_srli_si128(alo, 8));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_b, a_b));

            // Group C (weights 16-23)
            let w_c = _mm256_cvtepi8_epi32(whi);
            let a_c = _mm256_cvtepi8_epi32(ahi);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_c, a_c));

            // Group D (weights 24-31)
            let w_d = _mm256_cvtepi8_epi32(_mm_srli_si128(whi, 8));
            let a_d = _mm256_cvtepi8_epi32(_mm_srli_si128(ahi, 8));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_d, a_d));

            let dot_i32 = hsum_i32_avx(acc);
            row_sum += scale * dot_i32 as f32;
        } else if remaining > 0 {
            // Scalar tail.
            let q_w = &w_block[2..];
            let q_a = &a_block[2..];
            let mut partial = 0.0f32;
            for i in 0..remaining {
                partial += (q_w[i] as i8 as f32) * (q_a[i] as i8 as f32);
            }
            row_sum += scale * partial;
        }
    }

    row_sum
}

/// Horizontal sum of an `__m256i` i32 register.
///
/// # Safety
/// Caller must have `avx2` CPU feature.
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s128 = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(s128, 0b10_11_00_01);
    let sums = _mm_add_epi32(s128, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b00_00_10_10);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 34-byte Q8_0 block to 32 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 34`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 34 >= 2.
    let d = f16_to_f32(block);
    let vd = _mm256_set1_ps(d);

    // Load all 32 i8 values into a single 256-bit register.
    // SAFETY: block.ptr + 2 is valid; block.len() >= 34 guarantees 32 bytes available.
    let raw256 = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);

    // Split into two 128-bit halves for i8→i32 widening.
    let lo128 = _mm256_castsi256_si128(raw256); // bytes 0-15 (weights 0-15)
    let hi128 = _mm256_extracti128_si256(raw256, 1); // bytes 16-31 (weights 16-31)

    // Group A: weights 0-7  (lower 8 bytes of lo128)
    let a_i32 = _mm256_cvtepi8_epi32(lo128); // signed i8 → i32
    let a_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(a_i32), vd);

    // Group B: weights 8-15 (upper 8 bytes of lo128)
    let lo128_hi = _mm_srli_si128(lo128, 8);
    let b_i32 = _mm256_cvtepi8_epi32(lo128_hi);
    let b_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i32), vd);

    // Group C: weights 16-23 (lower 8 bytes of hi128)
    let c_i32 = _mm256_cvtepi8_epi32(hi128);
    let c_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(c_i32), vd);

    // Group D: weights 24-31 (upper 8 bytes of hi128)
    let hi128_hi = _mm_srli_si128(hi128, 8);
    let d_i32 = _mm256_cvtepi8_epi32(hi128_hi);
    let d_f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(d_i32), vd);

    // Store results.
    // SAFETY: output.len() >= 32 guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, a_f32);
    _mm256_storeu_ps(ptr.add(8), b_f32);
    _mm256_storeu_ps(ptr.add(16), c_f32);
    _mm256_storeu_ps(ptr.add(24), d_f32);
}

/// Compute the dot product of one row of a Q8_0 matrix with an FP32 vector.
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

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;

        // Read FP16 scale.
        // SAFETY: block.len() == BLOCK_BYTES == 34 >= 2.
        let d = f16_to_f32(block);

        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block — 4 FMA groups of 8.
            // SAFETY: block has 34 bytes; ptr+2 gives 32 bytes for 256-bit load.
            let raw256 = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
            let lo128 = _mm256_castsi256_si128(raw256);
            let hi128 = _mm256_extracti128_si256(raw256, 1);

            // SAFETY: input_offset + 32 <= n_cols <= input.len().
            let inp_ptr = input.as_ptr().add(input_offset);

            // Group A: i8[0..8] × input[0..8]
            let wa = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128));
            let ia = _mm256_loadu_ps(inp_ptr);
            let mut acc = _mm256_mul_ps(wa, ia);

            // Group B: i8[8..16] × input[8..16]
            let wb = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo128, 8)));
            let ib = _mm256_loadu_ps(inp_ptr.add(8));
            acc = _mm256_fmadd_ps(wb, ib, acc);

            // Group C: i8[16..24] × input[16..24]
            let wc = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128));
            let ic = _mm256_loadu_ps(inp_ptr.add(16));
            acc = _mm256_fmadd_ps(wc, ic, acc);

            // Group D: i8[24..32] × input[24..32]
            let wd = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi128, 8)));
            let id = _mm256_loadu_ps(inp_ptr.add(24));
            acc = _mm256_fmadd_ps(wd, id, acc);

            row_sum += hsum_f32_avx(acc) * d;
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

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
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
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut values = [0i8; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as i8).wrapping_sub(16);
        }
        let block = make_q8_0_block(0.5, &values);

        let mut out_avx2 = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        Q8_0Avx2.dequant_block(&block, &mut out_avx2).unwrap();
        Q8_0Ref.dequant_block(&block, &mut out_ref).unwrap();

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
        let mut values = [0i8; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as i32) - 10) as i8;
        }
        let block = make_q8_0_block(0.25, &values);
        let tensor_avx2 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

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
        // 20 columns — partial block
        let values = [1i8; 32];
        let block = make_q8_0_block(1.0, &values);
        let tensor_avx2 = make_tensor(block.clone(), 20);
        let tensor_ref = make_tensor(block, 20);

        let input = vec![1.0f32; 20];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-4,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_negative_weights() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut values = [0i8; 32];
        values[0] = -128;
        values[31] = 127;
        let block = make_q8_0_block(2.0, &values);
        let tensor_avx2 = make_tensor(block.clone(), 32);
        let tensor_ref = make_tensor(block, 32);

        let mut input = vec![0.0f32; 32];
        input[0] = 1.0;
        input[31] = 1.0;

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Avx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q8_0Ref.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "negative weight gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }
}
