//! AVX-512 accelerated Q5_0 quantization kernel.
//!
//! Q5_0 block layout (22 bytes per 32 weights):
//! - bytes[0..2]   — FP16 scale `d` (little-endian)
//! - bytes[2..6]   — `qh`: 32 high bits (bit 4 of each 5-bit quant), u32 LE
//! - bytes[6..22]  — `qs`: 32 × lower 4 bits packed (2 per byte)
//!
//! Weight layout (element ordering):
//!   output[i]      = d * ((qs[i].lo4 | ((qh >> i)      & 1) << 4) - 16)  for i in 0..16
//!   output[i + 16] = d * ((qs[i].hi4 | ((qh >> (i+16)) & 1) << 4) - 16)  for i in 0..16
//!
//! Strategy: scalar decode of 32 elements into `[i32; 32]`, then AVX-512 i32→f32 multiply.
//! With only 32 elements per block, one AVX-512 pass covers both halves.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q5_0 block: 2 (FP16 scale) + 4 (qh) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 22;

/// AVX-512 accelerated Q5_0 kernel.
///
/// Requires the `avx512f` CPU feature. The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q5_0Avx512;

impl QuantKernel for Q5_0Avx512 {
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
        // SAFETY: block.len() >= 22 and output.len() >= 32 verified above.
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
        "Q5_0"
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Decode one Q5_0 block into a `[i32; 32]` of centered 5-bit values.
///
/// Elements 0..16 come from low nibbles of qs bytes, bits from qh[0..16].
/// Elements 16..32 come from high nibbles of qs bytes, bits from qh[16..32].
///
/// # Safety
/// `block.len() >= BLOCK_BYTES` — guaranteed by callers.
#[inline(always)]
unsafe fn decode_vals(block: &[u8]) -> [i32; BLOCK_SIZE] {
    // Read qh as a u32 little-endian.
    // SAFETY: bytes 2..6 within block (len >= 22).
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let mut vals = [0i32; BLOCK_SIZE];
    // qs bytes start at offset 6; 16 bytes for 32 nibbles.
    for i in 0..16_usize {
        let byte = block[6 + i];
        let lo4 = (byte & 0x0F) as u32;
        let hi4 = ((byte >> 4) & 0x0F) as u32;
        let bit_lo = (qh >> i) & 1;
        let bit_hi = (qh >> (i + 16)) & 1;
        vals[i] = (lo4 | (bit_lo << 4)) as i32 - 16;
        vals[i + 16] = (hi4 | (bit_hi << 4)) as i32 - 16;
    }
    vals
}

/// Dequantize one 22-byte Q5_0 block to 32 FP32 values using AVX-512.
///
/// Two AVX-512 passes of 16 elements each cover the full 32-element block.
///
/// # Safety
/// - `block.len() >= 22`
/// - `output.len() >= 32`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 22 ≥ 2 — guaranteed by caller.
    let d = f16_to_f32(block);
    let vscale = _mm512_set1_ps(d);

    // SAFETY: block.len() >= BLOCK_BYTES — guaranteed by caller.
    let vals = decode_vals(block);

    // Pass 1: elements 0..16.
    // SAFETY: vals.len() == 32 ≥ 16; output.len() >= 32.
    let vi32_lo = _mm512_loadu_si512(vals.as_ptr() as *const __m512i);
    let vf32_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(vi32_lo), vscale);
    _mm512_storeu_ps(output.as_mut_ptr(), vf32_lo);

    // Pass 2: elements 16..32.
    // SAFETY: vals.as_ptr().add(16) and output.as_mut_ptr().add(16) are within bounds.
    let vi32_hi = _mm512_loadu_si512(vals.as_ptr().add(16) as *const __m512i);
    let vf32_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(vi32_hi), vscale);
    _mm512_storeu_ps(output.as_mut_ptr().add(16), vf32_hi);
}

/// Compute one row of a gemv for Q5_0 using AVX-512.
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
    let mut acc_vec = _mm512_setzero_ps();
    let mut acc_scalar = 0.0f32;

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let col_base = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - col_base).min(BLOCK_SIZE);

        // SAFETY: block.len() == BLOCK_BYTES ≥ 2 — guaranteed by caller.
        let d = f16_to_f32(block);
        let vscale = _mm512_set1_ps(d);

        // SAFETY: block.len() >= BLOCK_BYTES.
        let vals = decode_vals(block);

        // BLOCK_SIZE = 32; AVX-512 is 16-wide. Two passes when full block present.
        let full_chunks = cols_in_block / 16;
        for chunk in 0..full_chunks {
            let base = chunk * 16;
            // SAFETY: base + 16 ≤ cols_in_block ≤ BLOCK_SIZE = vals.len().
            let vi32 = _mm512_loadu_si512(vals.as_ptr().add(base) as *const __m512i);
            let wf32 = _mm512_mul_ps(_mm512_cvtepi32_ps(vi32), vscale);
            // SAFETY: col_base + base + 16 ≤ n_cols — guaranteed by full_chunks bound.
            let inp = _mm512_loadu_ps(input.as_ptr().add(col_base + base));
            acc_vec = _mm512_fmadd_ps(wf32, inp, acc_vec);
        }

        // Scalar tail.
        for i in (full_chunks * 16)..cols_in_block {
            acc_scalar += vals[i] as f32 * d * input[col_base + i];
        }
    }

    // SAFETY: avx512f — guaranteed by target_feature.
    hsum_f32_avx512(acc_vec) + acc_scalar
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(qs_byte: u8, qh: u32) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        // d = 1.0f16 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        let qh_bytes = qh.to_le_bytes();
        block[2..6].copy_from_slice(&qh_bytes);
        block[6..22].fill(qs_byte);
        block
    }

    fn ref_decode(block: &[u8]) -> [f32; 32] {
        use crate::reference::Q5_0Ref;
        use crate::traits::QuantKernel;
        let mut out = [0.0f32; 32];
        Q5_0Ref
            .dequant_block(block, &mut out)
            .expect("ref decode failed");
        out
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q5_0_avx512_zero_nibbles_zero_qh() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // All nibbles = 0, all qh bits = 0 → each element = 0 - 16 = -16; d=1.0 → -16.0
        let block = make_block(0x00, 0x0000_0000);
        let mut out = [0.0f32; 32];
        Q5_0Avx512
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - (-16.0f32)).abs() < 1e-6,
                "output[{i}] = {v}, expected -16.0"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q5_0_avx512_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_block(0xAB, 0xDEAD_BEEF);
        let ref_out = ref_decode(&block);
        let mut avx_out = [0.0f32; 32];
        Q5_0Avx512
            .dequant_block(&block, &mut avx_out)
            .expect("avx512 dequant failed");
        for (i, (&r, &a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!(
                (r - a).abs() < 1e-4,
                "mismatch at [{i}]: ref={r}, avx512={a}"
            );
        }
    }
}
