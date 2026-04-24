//! AVX2+FMA accelerated TQ1_0 quantization kernel.
//!
//! TQ1_0 block format (54 bytes per 256 weights):
//! - bytes[0..48]  — `qs`: 240 ternary values packed in base-3 (5 values/byte).
//! - bytes[48..52] — `qh`: 16 ternary values packed as 2-bit codes (4 values/byte).
//! - bytes[52..54] — FP16 scale `d` (little-endian).
//!
//! Ternary encoding: `digit - 1` where digit ∈ {0, 1, 2} → {-1, 0, +1}.
//!
//! The AVX2 strategy:
//! 1. LUT-based scalar decode of all 256 ternary values into `[i8; 256]`.
//! 2. AVX2 vectorised i8→f32 conversion and multiply by scale.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for TQ1_0: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per TQ1_0 block: 48 (qs) + 4 (qh) + 2 (d).
pub const BLOCK_BYTES: usize = 54;
/// Number of `qs` bytes.
const QS_BYTES: usize = 48;
/// Number of `qh` bytes.
const QH_BYTES: usize = 4;
/// Byte offset of `qh`.
const QH_OFFSET: usize = QS_BYTES;
/// Byte offset of FP16 scale `d`.
const D_OFFSET: usize = QS_BYTES + QH_BYTES;

// ---------------------------------------------------------------------------
// Compile-time LUT for qs byte → 5 ternary values
// ---------------------------------------------------------------------------

/// Precomputed LUT: `qs_byte` → 5 ternary values as `i8` ∈ {-1, 0, +1}.
///
/// For byte `b`, the five base-3 digits are computed as:
/// `digit[i] = (b / 3^i) % 3`, and the ternary value is `digit - 1`.
static TQ1_0_QS_LUT: [[i8; 5]; 256] = {
    let mut table = [[0i8; 5]; 256];
    let mut b = 0u16;
    while b < 256 {
        let mut q = b;
        let mut i = 0usize;
        while i < 5 {
            table[b as usize][i] = (q % 3) as i8 - 1;
            q /= 3;
            i += 1;
        }
        b += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Public kernel struct
// ---------------------------------------------------------------------------

/// AVX2+FMA accelerated TQ1_0 kernel.
pub struct Tq1_0Avx2;

impl QuantKernel for Tq1_0Avx2 {
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
        // SAFETY: bounds checked above; CPU feature guaranteed by KernelDispatcher.
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
            // SAFETY: bounds checked above; CPU feature guaranteed by KernelDispatcher.
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
        "TQ1_0"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Decode all 256 ternary values from one TQ1_0 block into an `[i8; 256]` scratch buffer.
///
/// # Safety
/// `block.len() >= BLOCK_BYTES` — guaranteed by callers.
#[inline(always)]
unsafe fn decode_vals(block: &[u8]) -> [i8; BLOCK_SIZE] {
    let mut vals = [0i8; BLOCK_SIZE];

    // qs: 48 bytes → 240 ternary values via LUT.
    for (i, &byte) in block[..QS_BYTES].iter().enumerate() {
        let row = &TQ1_0_QS_LUT[byte as usize];
        let base = i * 5;
        vals[base] = row[0];
        vals[base + 1] = row[1];
        vals[base + 2] = row[2];
        vals[base + 3] = row[3];
        vals[base + 4] = row[4];
    }

    // qh: 4 bytes → 16 ternary values (positions 240..256).
    for (i, &byte) in block[QH_OFFSET..QH_OFFSET + QH_BYTES].iter().enumerate() {
        let base = 240 + i * 4;
        vals[base] = (byte & 0x03) as i8 - 1;
        vals[base + 1] = ((byte >> 2) & 0x03) as i8 - 1;
        vals[base + 2] = ((byte >> 4) & 0x03) as i8 - 1;
        vals[base + 3] = ((byte >> 6) & 0x03) as i8 - 1;
    }

    vals
}

/// Dequantize one 54-byte TQ1_0 block to 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 54`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= BLOCK_BYTES — guaranteed by caller.
    let d = f16_to_f32(&block[D_OFFSET..]);
    let scale = _mm256_set1_ps(d);

    // Step 1: decode all 256 ternary values via LUT.
    let vals = decode_vals(block);

    // Step 2: AVX2 i8 → f32 multiply: 32 chunks of 8 values.
    for chunk in 0..32usize {
        let base = chunk * 8;
        // SAFETY: base + 8 ≤ 256 = vals.len() — loop invariant.
        let vi8 = _mm_loadl_epi64(vals.as_ptr().add(base) as *const __m128i);
        let vi32 = _mm256_cvtepi8_epi32(vi8);
        let vf32 = _mm256_cvtepi32_ps(vi32);
        let result = _mm256_mul_ps(vf32, scale);
        // SAFETY: base + 8 ≤ 256 = output.len() — guaranteed by caller & loop.
        _mm256_storeu_ps(output.as_mut_ptr().add(base), result);
    }
}

/// Compute one row of a gemv for TQ1_0.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc_vec = _mm256_setzero_ps();
    let mut acc_scalar = 0.0f32;

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let col_base = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - col_base).min(BLOCK_SIZE);

        // SAFETY: block.len() == BLOCK_BYTES ≥ D_OFFSET + 2.
        let d = f16_to_f32(&block[D_OFFSET..]);
        let scale = _mm256_set1_ps(d);

        // Decode via LUT.
        let vals = decode_vals(block);

        // Vectorised FMA: 8 values per iteration.
        let full_chunks = cols_in_block / 8;
        for chunk in 0..full_chunks {
            let base = chunk * 8;
            // SAFETY: base + 8 ≤ cols_in_block ≤ BLOCK_SIZE = vals.len().
            let vi8 = _mm_loadl_epi64(vals.as_ptr().add(base) as *const __m128i);
            let vi32 = _mm256_cvtepi8_epi32(vi8);
            let vf32 = _mm256_cvtepi32_ps(vi32);
            let wf32 = _mm256_mul_ps(vf32, scale);
            // SAFETY: col_base + base + 8 ≤ n_cols — guaranteed by full_chunks bound.
            let inp = _mm256_loadu_ps(input.as_ptr().add(col_base + base));
            acc_vec = _mm256_fmadd_ps(wf32, inp, acc_vec);
        }

        // Scalar tail for remaining elements.
        for i in (full_chunks * 8)..cols_in_block {
            acc_scalar += vals[i] as f32 * d * input[col_base + i];
        }
    }

    // SAFETY: avx feature — guaranteed by target_feature.
    hsum_f32_avx(acc_vec) + acc_scalar
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic block with qs/qh set to a given pattern byte and d=1.0.
    fn make_block(qs_byte: u8, qh_byte: u8) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        block[..QS_BYTES].fill(qs_byte);
        block[QH_OFFSET..QH_OFFSET + QH_BYTES].fill(qh_byte);
        // d = 1.0f16 = 0x3C00
        block[D_OFFSET] = 0x00;
        block[D_OFFSET + 1] = 0x3C;
        block
    }

    /// Decode block via scalar reference for cross-validation.
    fn ref_decode(block: &[u8]) -> [f32; 256] {
        use crate::reference::Tq1_0Ref;
        use crate::traits::QuantKernel;
        let mut out = [0.0f32; 256];
        Tq1_0Ref
            .dequant_block(block, &mut out)
            .expect("ref decode failed");
        out
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tq1_0_avx2_zero_qs_zero_qh() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // qs=0: byte 0 in base-3 → all 5 digits = 0 → ternary -1.
        // qh=0: all 4 codes = 0 → ternary -1.
        let block = make_block(0x00, 0x00);
        let mut out = [0.0f32; 256];
        Tq1_0Avx2
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - (-1.0f32)).abs() < 1e-6,
                "output[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tq1_0_avx2_qs_1_qh_0x55_produces_expected() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // qs=1: byte 1 = base-3 digits [1,0,0,0,0] → ternary [0,-1,-1,-1,-1].
        // qh=0x55 = 0b01010101 → all 2-bit codes = 01 = 1 → ternary 0.
        let block = make_block(0x01, 0x55);
        let ref_out = ref_decode(&block);
        let mut avx_out = [0.0f32; 256];
        Tq1_0Avx2
            .dequant_block(&block, &mut avx_out)
            .expect("avx dequant failed");
        for (i, (&r, &a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-6, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tq1_0_avx2_matches_reference_varied() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut block = [0u8; BLOCK_BYTES];
        // Varied qs pattern (some bytes will be ≥ 243 = 3^5, wrapping naturally).
        for (i, b) in block[..QS_BYTES].iter_mut().enumerate() {
            *b = ((i * 17 + 3) % 243) as u8; // keep within valid base-3 range [0, 242]
        }
        for (i, b) in block[QH_OFFSET..QH_OFFSET + QH_BYTES]
            .iter_mut()
            .enumerate()
        {
            *b = ((i * 31 + 7) % 256) as u8;
        }
        // d ≈ 1.5 in f16 (0x3E00)
        block[D_OFFSET] = 0x00;
        block[D_OFFSET + 1] = 0x3E;

        let ref_out = ref_decode(&block);
        let mut avx_out = [0.0f32; 256];
        Tq1_0Avx2
            .dequant_block(&block, &mut avx_out)
            .expect("avx dequant failed");

        for (i, (&r, &a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }
}
