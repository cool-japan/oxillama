//! AVX-512 accelerated Q8_K quantization kernel.
//!
//! Q8_K block layout (292 bytes per 256 weights):
//! - bytes[0..4]    — f32 super-block scale `d` (little-endian)
//! - bytes[4..260]  — 256 × int8 signed quantized values (qs)
//! - bytes[260..292] — 16 × int16 block sums (bsums, unused in gemv)
//!
//! Each weight reconstructs as `qs[i] × d`.
//!
//! Strategy: load 256 i8 values in 16 chunks of 16, widen to i32 via
//! `_mm512_cvtepi8_epi32`, convert to f32, multiply by scale.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::hsum_f32_avx512;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q8_K block: 4 (f32 scale) + 256 (i8 data) + 32 (bsums).
pub const BLOCK_BYTES: usize = 292;
/// Offset of the quantized i8 values within a block.
const QS_OFFSET: usize = 4;

/// AVX-512 accelerated Q8_K kernel.
///
/// Requires the `avx512f` CPU feature. The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q8_KAvx512;

impl QuantKernel for Q8_KAvx512 {
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
        // SAFETY: block.len() >= 292 and output.len() >= 256 verified above.
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
        "Q8_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 292-byte Q8_K block to 256 FP32 values using AVX-512.
///
/// Processes 256 i8 values in 16 chunks of 16 via `_mm512_cvtepi8_epi32`.
/// Each chunk: load 16 i8s from a `__m128i`, widen to 16 × i32, convert to
/// f32, multiply by the super-block scale, and store.
///
/// # Safety
/// - `block.len() >= 292`
/// - `output.len() >= 256`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read f32 scale from bytes[0..4].
    // SAFETY: block.len() >= 292 ≥ 4 — guaranteed by caller.
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let vd = _mm512_set1_ps(d);

    let qs_ptr = block.as_ptr().add(QS_OFFSET);
    let out_ptr = output.as_mut_ptr();

    // Process 256 i8 values in 16 chunks of 16.
    for chunk in 0..16_usize {
        let base = chunk * 16;
        // SAFETY: qs_ptr + base..+16 is within block[4..260] (256 bytes).
        let vi8_128 = _mm_loadu_si128(qs_ptr.add(base) as *const __m128i);
        let vi32 = _mm512_cvtepi8_epi32(vi8_128);
        let vf32 = _mm512_mul_ps(_mm512_cvtepi32_ps(vi32), vd);
        // SAFETY: out_ptr + base..+16 is within output[0..256].
        _mm512_storeu_ps(out_ptr.add(base), vf32);
    }
}

/// Compute dot product of one Q8_K row with an FP32 vector using AVX-512.
///
/// For each block: decode i8 quants in 16-wide chunks, FMA with input,
/// accumulate, then apply the super-block f32 scale.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc_total = 0.0f32;

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let col_base = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - col_base).min(BLOCK_SIZE);

        // SAFETY: block.len() == BLOCK_BYTES ≥ 4.
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);

        let qs_ptr = block.as_ptr().add(QS_OFFSET);
        let mut acc_vec = _mm512_setzero_ps();
        let mut acc_scalar = 0.0f32;

        // Vectorised FMA: 16 values per iteration.
        let full_chunks = cols_in_block / 16;
        for chunk in 0..full_chunks {
            let base = chunk * 16;
            // SAFETY: qs_ptr + base..+16 is within block[4..260].
            let vi8_128 = _mm_loadu_si128(qs_ptr.add(base) as *const __m128i);
            let vi32 = _mm512_cvtepi8_epi32(vi8_128);
            let vf32 = _mm512_cvtepi32_ps(vi32);
            // SAFETY: col_base + base + 16 ≤ n_cols — guaranteed by full_chunks bound.
            let inp = _mm512_loadu_ps(input.as_ptr().add(col_base + base));
            acc_vec = _mm512_fmadd_ps(vf32, inp, acc_vec);
        }

        // Scalar tail.
        for i in (full_chunks * 16)..cols_in_block {
            // SAFETY: QS_OFFSET + i is within block[4..260].
            let q = *block.get_unchecked(QS_OFFSET + i) as i8;
            acc_scalar += q as f32 * input[col_base + i];
        }

        // Scale the accumulated block dot product by d.
        // SAFETY: avx512f — guaranteed by target_feature.
        acc_total += d * (hsum_f32_avx512(acc_vec) + acc_scalar);
    }

    acc_total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(qs_val: i8, d: f32) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        block[0..4].copy_from_slice(&d.to_le_bytes());
        for b in block[QS_OFFSET..QS_OFFSET + BLOCK_SIZE].iter_mut() {
            *b = qs_val as u8;
        }
        block
    }

    fn ref_decode(block: &[u8]) -> [f32; 256] {
        use crate::reference::Q8KRef;
        use crate::traits::QuantKernel;
        let mut out = [0.0f32; 256];
        Q8KRef
            .dequant_block(block, &mut out)
            .expect("ref decode failed");
        out
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_k_avx512_zero_qs() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_block(0, 1.0);
        let mut out = [0.0f32; 256];
        Q8_KAvx512
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-6, "output[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_k_avx512_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_block(42, 0.125);
        let ref_out = ref_decode(&block);
        let mut avx_out = [0.0f32; 256];
        Q8_KAvx512
            .dequant_block(&block, &mut avx_out)
            .expect("avx512 dequant failed");
        for (i, (&r, &a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!(
                (r - a).abs() < 1e-5,
                "mismatch at [{i}]: ref={r}, avx512={a}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_k_avx512_negative_qs() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_block(-7, 2.0);
        let ref_out = ref_decode(&block);
        let mut avx_out = [0.0f32; 256];
        Q8_KAvx512
            .dequant_block(&block, &mut avx_out)
            .expect("avx512 dequant failed");
        for (i, (&r, &a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!(
                (r - a).abs() < 1e-5,
                "mismatch at [{i}]: ref={r}, avx512={a}"
            );
        }
    }
}
