//! AVX2+FMA accelerated Q5_0 quantization kernel.
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
//! This matches the reference implementation in `crate::reference::q5_0`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_0: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per Q5_0 block: 2 (FP16 scale) + 4 (qh) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 22;

/// AVX2+FMA accelerated Q5_0 kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Q5_0Avx2;

impl QuantKernel for Q5_0Avx2 {
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

        // SAFETY: we verified block.len() >= 22 and output.len() >= 32 above.
        // CPU features avx2+fma are guaranteed by KernelDispatcher.
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
        "Q5_0"
    }

    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        self.matvec_q8_fused_avx2(weights, acts_q8, out, n_rows, n_cols)
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 22-byte Q5_0 block to 32 FP32 values using AVX2.
///
/// The output ordering matches the reference scalar kernel:
///   output[0..16]  ← elements whose low nibble is the even-nibble of each qs byte
///   output[16..32] ← elements whose low nibble is the odd-nibble of each qs byte
///
/// # Safety
/// - `block.len() >= 22`
/// - `output.len() >= 32`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    // SAFETY: block.len() >= 22 ≥ 2 — guaranteed by caller.
    let d = f16_to_f32(block);
    let vd = _mm256_set1_ps(d);

    // Read qh (32 high bits, one per element).
    // SAFETY: bytes 2..6 are within block.
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    // Load the 16 packed nibble bytes (qs).
    // SAFETY: bytes 6..22 are within block (len >= 22).
    let qs = _mm_loadu_si128(block.as_ptr().add(6) as *const __m128i);

    // Split lo and hi nibbles from each qs byte.
    let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_nib = _mm_and_si128(qs, mask4); // lower nibble → elements 0..15
    let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4); // upper nibble → elements 16..31

    // Expand qh into per-element high bits using scalar extraction, then
    // form two __m128i vectors of the 5th bit for elements 0..15 and 16..31.
    //
    // We extract bit `i` from qh for the lower group and bit `i+16` for the upper group.
    let qh_lo_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> i) & 1) as u8);
    let qh_hi_bits: [u8; 16] = core::array::from_fn(|i| ((qh >> (i + 16)) & 1) as u8);

    let vqh_lo = _mm_loadu_si128(qh_lo_bits.as_ptr() as *const __m128i);
    let vqh_hi = _mm_loadu_si128(qh_hi_bits.as_ptr() as *const __m128i);

    // Shift the 5th bit into position 4 (multiply by 16 = shift left 4).
    let shift4 = _mm_set1_epi8(4_i8);
    let vqh_lo_shifted = _mm_slli_epi16(_mm_and_si128(vqh_lo, _mm_set1_epi8(1_i8)), 4);
    // _mm_slli_epi16 shifts in 16-bit lanes; mask afterwards to avoid bleed.
    let vqh_lo_shifted = _mm_and_si128(vqh_lo_shifted, _mm_set1_epi8(0x10_u8 as i8));
    let vqh_hi_shifted = _mm_slli_epi16(_mm_and_si128(vqh_hi, _mm_set1_epi8(1_i8)), 4);
    let vqh_hi_shifted = _mm_and_si128(vqh_hi_shifted, _mm_set1_epi8(0x10_u8 as i8));

    // 5-bit unsigned quant for each element:
    //   q5_lo[i] = lo_nib[i] | (qh_lo_bit[i] << 4)
    //   q5_hi[i] = hi_nib[i] | (qh_hi_bit[i] << 4)
    let q5_lo = _mm_or_si128(lo_nib, vqh_lo_shifted); // u8 in [0,31]
    let q5_hi = _mm_or_si128(hi_nib, vqh_hi_shifted); // u8 in [0,31]

    // Center: subtract 16.  We widen to i32 first via cvtepu8_epi32 (unsigned).
    let sixteen = _mm256_set1_epi32(16);

    // Process elements 0..15 (q5_lo) in two groups of 8:
    let lo_a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo), sixteen);
    let lo_f32_a = _mm256_mul_ps(_mm256_cvtepi32_ps(lo_a), vd);

    let q5_lo_hi8 = _mm_srli_si128(q5_lo, 8);
    let lo_b = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo_hi8), sixteen);
    let lo_f32_b = _mm256_mul_ps(_mm256_cvtepi32_ps(lo_b), vd);

    // Process elements 16..31 (q5_hi) in two groups of 8:
    let hi_a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi), sixteen);
    let hi_f32_a = _mm256_mul_ps(_mm256_cvtepi32_ps(hi_a), vd);

    let q5_hi_hi8 = _mm_srli_si128(q5_hi, 8);
    let hi_b = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi_hi8), sixteen);
    let hi_f32_b = _mm256_mul_ps(_mm256_cvtepi32_ps(hi_b), vd);

    // Store: output[0..8], output[8..16], output[16..24], output[24..32].
    // SAFETY: output.len() >= 32 — guaranteed by caller.
    let ptr = output.as_mut_ptr();
    _mm256_storeu_ps(ptr, lo_f32_a);
    _mm256_storeu_ps(ptr.add(8), lo_f32_b);
    _mm256_storeu_ps(ptr.add(16), hi_f32_a);
    _mm256_storeu_ps(ptr.add(24), hi_f32_b);

    // Suppress unused-variable warning for `shift4` (used conceptually but
    // replaced by immediate-shift intrinsics).
    let _ = shift4;
}

/// Compute the dot product of one row of a Q5_0 matrix with an FP32 vector.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= blocks_per_row * BLOCK_SIZE`
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

        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = _mm_loadu_si128(block.as_ptr().add(6) as *const __m128i);

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

        let sixteen = _mm256_set1_epi32(16);

        // Load input vectors: 32 f32s in 4 groups of 8.
        // For partial last block, fall back to scalar for remaining elements.
        let avail = inp.len();

        // Group A: output[0..8] × input[0..8]
        let n_a = avail.min(8);
        let mut inp_a = [0.0f32; 8];
        inp_a[..n_a].copy_from_slice(&inp[..n_a]);
        let vinp_a = _mm256_loadu_ps(inp_a.as_ptr());
        let qi_a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo), sixteen);
        let vq_a = _mm256_mul_ps(_mm256_cvtepi32_ps(qi_a), vd);
        acc = _mm256_fmadd_ps(vq_a, vinp_a, acc);

        // Group B: output[8..16] × input[8..16]
        let n_b = avail.saturating_sub(8).min(8);
        let mut inp_b = [0.0f32; 8];
        if n_b > 0 {
            inp_b[..n_b].copy_from_slice(&inp[8..8 + n_b]);
        }
        let vinp_b = _mm256_loadu_ps(inp_b.as_ptr());
        let q5_lo_hi8 = _mm_srli_si128(q5_lo, 8);
        let qi_b = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo_hi8), sixteen);
        let vq_b = _mm256_mul_ps(_mm256_cvtepi32_ps(qi_b), vd);
        acc = _mm256_fmadd_ps(vq_b, vinp_b, acc);

        // Group C: output[16..24] × input[16..24]
        let n_c = avail.saturating_sub(16).min(8);
        let mut inp_c = [0.0f32; 8];
        if n_c > 0 {
            inp_c[..n_c].copy_from_slice(&inp[16..16 + n_c]);
        }
        let vinp_c = _mm256_loadu_ps(inp_c.as_ptr());
        let qi_c = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi), sixteen);
        let vq_c = _mm256_mul_ps(_mm256_cvtepi32_ps(qi_c), vd);
        acc = _mm256_fmadd_ps(vq_c, vinp_c, acc);

        // Group D: output[24..32] × input[24..32]
        let n_d = avail.saturating_sub(24).min(8);
        let mut inp_d = [0.0f32; 8];
        if n_d > 0 {
            inp_d[..n_d].copy_from_slice(&inp[24..24 + n_d]);
        }
        let vinp_d = _mm256_loadu_ps(inp_d.as_ptr());
        let q5_hi_hi8 = _mm_srli_si128(q5_hi, 8);
        let qi_d = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi_hi8), sixteen);
        let vq_d = _mm256_mul_ps(_mm256_cvtepi32_ps(qi_d), vd);
        acc = _mm256_fmadd_ps(vq_d, vinp_d, acc);
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Fused matvec: Q5_0 weights × Q8_0 activations
// ---------------------------------------------------------------------------

/// Fused dequant+dot for Q5_0 weights × Q8_0 activations.
///
/// # Safety
/// - `weights_row.len() >= blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 34`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q5_0_q8_0_row_avx2(
    weights_row: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    const Q8_BLOCK_BYTES: usize = 34;
    let mut acc = _mm256_setzero_ps();

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &weights_row[bo..bo + BLOCK_BYTES];
        let col_start = blk * BLOCK_SIZE;

        let d_w = f16_to_f32(block);
        let vd_w = _mm256_set1_ps(d_w);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = _mm_loadu_si128(block.as_ptr().add(6) as *const __m128i);

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

        // Decode Q8_0 activation block.
        let ab = blk * Q8_BLOCK_BYTES;
        let a_block = &acts_q8[ab..ab + Q8_BLOCK_BYTES];
        let d_a = f16_to_f32(a_block);
        let vd_a = _mm256_set1_ps(d_a);

        let q8_raw = _mm256_loadu_si256(a_block.as_ptr().add(2) as *const __m256i);
        let q8_lo128 = _mm256_castsi256_si128(q8_raw);
        let q8_hi128 = _mm256_extracti128_si256(q8_raw, 1);
        let q8_lo_i16 = _mm256_cvtepi8_epi16(q8_lo128);
        let q8_hi_i16 = _mm256_cvtepi8_epi16(q8_hi128);
        let qa0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q8_lo_i16)));
        let qa1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(
            q8_lo_i16, 1,
        )));
        let qa2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q8_hi_i16)));
        let qa3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(
            q8_hi_i16, 1,
        )));
        // Scale activations by d_a.
        let act0 = _mm256_mul_ps(qa0, vd_a);
        let act1 = _mm256_mul_ps(qa1, vd_a);
        let act2 = _mm256_mul_ps(qa2, vd_a);
        let act3 = _mm256_mul_ps(qa3, vd_a);

        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let avail = col_end - col_start;
        let sixteen = _mm256_set1_epi32(16);

        // Group A: weight[0..8] × act[0..8]
        let n_a = avail.min(8);
        let mut buf_a = [0.0f32; 8];
        let wq_a = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo), sixteen));
        let wf_a = _mm256_mul_ps(wq_a, vd_w);
        let mut act_a = [0.0f32; 8];
        _mm256_storeu_ps(act_a.as_mut_ptr(), act0);
        buf_a[..n_a].copy_from_slice(&act_a[..n_a]);
        acc = _mm256_fmadd_ps(wf_a, _mm256_loadu_ps(buf_a.as_ptr()), acc);

        // Group B: weight[8..16] × act[8..16]
        let n_b = avail.saturating_sub(8).min(8);
        let mut buf_b = [0.0f32; 8];
        let q5_lo_hi8 = _mm_srli_si128(q5_lo, 8);
        let wq_b = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_lo_hi8), sixteen));
        let wf_b = _mm256_mul_ps(wq_b, vd_w);
        let mut act_b = [0.0f32; 8];
        _mm256_storeu_ps(act_b.as_mut_ptr(), act1);
        if n_b > 0 {
            buf_b[..n_b].copy_from_slice(&act_b[..n_b]);
        }
        acc = _mm256_fmadd_ps(wf_b, _mm256_loadu_ps(buf_b.as_ptr()), acc);

        // Group C: weight[16..24] × act[16..24]
        let n_c = avail.saturating_sub(16).min(8);
        let mut buf_c = [0.0f32; 8];
        let wq_c = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi), sixteen));
        let wf_c = _mm256_mul_ps(wq_c, vd_w);
        let mut act_c = [0.0f32; 8];
        _mm256_storeu_ps(act_c.as_mut_ptr(), act2);
        if n_c > 0 {
            buf_c[..n_c].copy_from_slice(&act_c[..n_c]);
        }
        acc = _mm256_fmadd_ps(wf_c, _mm256_loadu_ps(buf_c.as_ptr()), acc);

        // Group D: weight[24..32] × act[24..32]
        let n_d = avail.saturating_sub(24).min(8);
        let mut buf_d = [0.0f32; 8];
        let q5_hi_hi8 = _mm_srli_si128(q5_hi, 8);
        let wq_d = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q5_hi_hi8), sixteen));
        let wf_d = _mm256_mul_ps(wq_d, vd_w);
        let mut act_d = [0.0f32; 8];
        _mm256_storeu_ps(act_d.as_mut_ptr(), act3);
        if n_d > 0 {
            buf_d[..n_d].copy_from_slice(&act_d[..n_d]);
        }
        acc = _mm256_fmadd_ps(wf_d, _mm256_loadu_ps(buf_d.as_ptr()), acc);
    }

    hsum_f32_avx(acc)
}

impl Q5_0Avx2 {
    /// Override of `matvec_q8_fused` using AVX2+FMA intrinsics.
    ///
    /// Computes `out[row] += Σ_block (d_w * q5_0 · d_a * q8_0)` in a single pass.
    pub fn matvec_q8_fused_avx2(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> crate::error::QuantResult<()> {
        use crate::error::QuantError;

        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        let q8_block_bytes: usize = 34;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < blocks_per_row * q8_block_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: blocks_per_row * q8_block_bytes,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: bounds verified above; avx2+fma guaranteed by dispatcher.
            let partial = unsafe {
                fused_q5_0_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            *out_val += partial;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::q5_0::Q5_0Ref;
    use crate::traits::QuantKernel;

    fn make_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    fn avx2_available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn avx2_matches_reference_zeros() {
        if !avx2_available() {
            return;
        }
        let block = make_block(0.0, 0, &[0; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];

        Q5_0Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_0Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");

        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_all_bits_set() {
        if !avx2_available() {
            return;
        }
        // qh = 0xFFFFFFFF → hi bit = 1 for all; qs = 0xFF → nibbles all 0xF
        // q5 = 0xF | (1<<4) = 31 → 31 - 16 = 15; weight = 1.0 * 15 = 15.0
        let block = make_block(1.0, 0xFFFF_FFFF, &[0xFF; 16]);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];

        Q5_0Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_0Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");

        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r} avx2={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_alternating() {
        if !avx2_available() {
            return;
        }
        let qh: u32 = 0xA5A5_A5A5;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, qh, &qs);
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        let mut avx2_out = vec![0.0f32; BLOCK_SIZE];

        Q5_0Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant");
        Q5_0Avx2
            .dequant_block(&block, &mut avx2_out)
            .expect("avx2 dequant");

        for (i, (r, a)) in ref_out.iter().zip(avx2_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "elem[{i}]: ref={r:.6} avx2={a:.6}");
        }
    }

    #[test]
    fn empty_block_errors() {
        let mut out = vec![0.0f32; BLOCK_SIZE];
        let err = Q5_0Avx2.dequant_block(&[], &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn output_too_small_errors() {
        if !avx2_available() {
            return;
        }
        let block = make_block(1.0, 0, &[0; 16]);
        let mut out = vec![0.0f32; 16]; // too small
        let err = Q5_0Avx2.dequant_block(&block, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn gemv_matches_reference() {
        if !avx2_available() {
            return;
        }
        let qh: u32 = 0x1234_5678;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 13 + 7) & 0xFF) as u8;
        }
        let block = make_block(0.25, qh, &qs);

        // Build QuantTensor with a single block row.
        use crate::types::QuantTensor;
        use oxillama_gguf::GgufTensorType;

        let tensor = QuantTensor {
            data: block.clone().into(),
            shape: vec![1, BLOCK_SIZE],
            tensor_type: GgufTensorType::Q5_0,
        };

        let input: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let mut ref_out = vec![0.0f32; 1];
        let mut avx2_out = vec![0.0f32; 1];

        Q5_0Ref
            .gemv(&tensor, &input, &mut ref_out)
            .expect("ref gemv");
        Q5_0Avx2
            .gemv(&tensor, &input, &mut avx2_out)
            .expect("avx2 gemv");

        assert!(
            (ref_out[0] - avx2_out[0]).abs() < 1e-3,
            "gemv: ref={} avx2={}",
            ref_out[0],
            avx2_out[0]
        );
    }

    #[test]
    fn fused_q5_0_avx2_matches_reference() {
        if !avx2_available() {
            return;
        }

        let qh: u32 = 0xA5A5_A5A5;
        let mut qs_w = [0u8; 16];
        for (i, v) in qs_w.iter_mut().enumerate() {
            *v = ((i * 11 + 5) & 0xFF) as u8;
        }
        let weight_block = make_block(0.5, qh, &qs_w);

        // Build a Q8_0 activation block (34 bytes): 2-byte f16 scale + 32 i8 values
        let d_a = 0.25f32;
        let mut acts_raw = [0i8; 32];
        for (i, v) in acts_raw.iter_mut().enumerate() {
            *v = ((i as i16 * 5 - 40).clamp(-128, 127)) as i8;
        }
        let mut acts_block = Vec::with_capacity(34);
        acts_block.extend_from_slice(&half::f16::from_f32(d_a).to_bits().to_le_bytes());
        for &v in &acts_raw {
            acts_block.push(v as u8);
        }

        // Reference: dequant weight, dot with dequanted acts
        let mut w_dequant = vec![0.0f32; BLOCK_SIZE];
        Q5_0Ref
            .dequant_block(&weight_block, &mut w_dequant)
            .expect("ref dequant w");
        let acts_f32: Vec<f32> = acts_raw.iter().map(|&v| v as f32 * d_a).collect();
        let expected: f32 = w_dequant
            .iter()
            .zip(acts_f32.iter())
            .map(|(w, a)| w * a)
            .sum();

        // AVX2 fused (additive — start with 0)
        let mut out_avx2 = vec![0.0f32; 1];
        Q5_0Avx2
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_avx2, 1, BLOCK_SIZE)
            .expect("avx2 fused");

        assert!(
            (out_avx2[0] - expected).abs() < 0.1,
            "fused_q5_0_avx2: got={} expected={}",
            out_avx2[0],
            expected
        );

        // Also verify against reference Q5_0Ref::matvec_q8_fused
        use crate::reference::q5_0::Q5_0Ref as Q5_0RefInner;
        let mut out_ref = vec![0.0f32; 1];
        Q5_0RefInner
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_ref, 1, BLOCK_SIZE)
            .expect("ref fused");
        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "fused: avx2={} ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }
}
