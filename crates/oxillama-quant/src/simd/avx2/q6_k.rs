//! AVX2+FMA accelerated Q6_K quantization kernel.
//!
//! Q6_K block layout (210 bytes per 256 weights):
//! - bytes[0..128]   — `ql` — lower 4 bits of each 6-bit quant (2 per byte)
//! - bytes[128..192] — `qh` — upper 2 bits of each 6-bit quant (4 per byte)
//! - bytes[192..208] — 16 × int8 signed scales (one per 16-weight sub-block)
//! - bytes[208..210] — FP16 super-block scale `d` (little-endian)
//!
//! Block structure: 16 sub-blocks of 16 weights, each with its own int8 scale.
//! Q6_K is a symmetric ("type-0") format — no minimum offset term.
//!
//! Weight formula: `w = d * scale_i * (q6 - 32)` where `q6 = ql_lo | (qh_hi << 4)`.
//!
//! The ql/qh bit extraction uses the same 128-bit masking pattern as Q4_K for
//! the lower nibbles, and an `_mm_srli_epi16` + narrow mask for the 2-bit qh
//! values (safe because the narrow 0x03 mask eliminates cross-byte spillover).

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q6_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q6_K block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (FP16 d).
pub const BLOCK_BYTES: usize = 210;

/// AVX2+FMA accelerated Q6_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q6_KAvx2;

impl QuantKernel for Q6_KAvx2 {
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

        // SAFETY: block.len() >= 210 and output.len() >= 256 verified above.
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
        "Q6_K"
    }

    /// Fused Q6_K weight × Q8_0 activation GEMV.
    ///
    /// Each Q6_K super-block (256 weights, 210 bytes) maps to 8 Q8_0 activation blocks.
    /// Column `col` within the super-block maps to Q8_0 block `blk*8 + col/32` and
    /// lane `col % 32` within that block.
    /// Accumulates into `out` (ACCUMULATE semantics).
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
        let q8_blocks_per_row = blocks_per_row * 8;
        let acts_needed = q8_blocks_per_row * Q8_0_BLOCK_BYTES;

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
                fused_q6_k_q8_0_row_avx2(
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

/// Q8_0 block bytes for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Fused Q6_K weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q6_k_q8_0_row_avx2(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let block = &row_data[bo..bo + BLOCK_BYTES];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(&block[208..]);

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        // Q6_K: 2 groups of 128 weights; each group has 4 sub-blocks of 32.
        // Within each sub-block of 32, use Q8_0 block at: blk*8 + (in_off+col)/32
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let in_off = group * 128;

            for l in 0..32 {
                let is = l / 16; // sub-block index within group (0 or 1)

                let q1 = ((ql[ql_off + l] & 0x0F)
                    | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F)
                    | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4)
                    | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4)
                    | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let s0 = d * scales[sc_off + is] as i8 as f32;
                let s1 = d * scales[sc_off + is + 2] as i8 as f32;
                let s2 = d * scales[sc_off + is + 4] as i8 as f32;
                let s3 = d * scales[sc_off + is + 6] as i8 as f32;

                let c0 = in_off + l;
                let c1 = in_off + l + 32;
                let c2 = in_off + l + 64;
                let c3 = in_off + l + 96;

                // For each column, resolve the corresponding Q8_0 sample.
                // Column `col` within the super-block → Q8_0 block `blk*8 + col/32`,
                // lane `col % 32`.
                let sample_q8 = |col: usize| -> Option<f32> {
                    if col >= cols_in_block {
                        return None;
                    }
                    let q8_blk = blk * 8 + col / 32;
                    let q8_lane = col % 32;
                    // SAFETY: acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES.
                    let ab = &acts_q8[q8_blk * Q8_0_BLOCK_BYTES..(q8_blk + 1) * Q8_0_BLOCK_BYTES];
                    let d_a = f16_to_f32(ab);
                    let q_a = ab[2 + q8_lane] as i8 as f32;
                    Some(d_a * q_a)
                };

                if let Some(a0) = sample_q8(c0) {
                    row_sum += s0 * q1 as f32 * a0;
                }
                if let Some(a1) = sample_q8(c1) {
                    row_sum += s1 * q2 as f32 * a1;
                }
                if let Some(a2) = sample_q8(c2) {
                    row_sum += s2 * q3 as f32 * a2;
                }
                if let Some(a3) = sample_q8(c3) {
                    row_sum += s3 * q4 as f32 * a3;
                }
            }
        }
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Decode 32 × 6-bit quants from one row of 32 bytes in `ql` and the
/// corresponding 32-byte row in `qh` (providing 4 sets of 2-bit upper parts).
///
/// The reference layout per `l` in 0..32:
/// - `q1 = (ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)` → weight at out_off + l
/// - `q2 = (ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)` → out_off + l + 32
/// - `q3 = (ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)` → out_off + l + 64
/// - `q4 = (ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)` → out_off + l + 96
///
/// Returns four `__m128i` registers: (q1_bytes, q2_bytes, q3_bytes, q4_bytes)
/// where each byte is the 6-bit unsigned quant (0..63).
///
/// # Safety
/// Requires AVX2.  Pointers `ql_ptr` and `qh_ptr` must each be valid for
/// loading 32 bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_q6_group(
    ql_ptr: *const u8,
    qh_ptr: *const u8,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // Load two 16-byte halves of ql (positions 0..15 and 32..47 for q1/q3)
    // and (positions 16..31 and 48..63 for q2/q4).
    // SAFETY: caller ensures 32 valid bytes at ql_ptr and ql_ptr+32, 32 at qh_ptr.
    let ql0 = _mm_loadu_si128(ql_ptr as *const __m128i); // ql[0..15]
    let ql1 = _mm_loadu_si128(ql_ptr.add(32) as *const __m128i); // ql[32..47] (for q2/q4)

    let qh_raw = _mm_loadu_si128(qh_ptr as *const __m128i); // qh[0..15]

    let mask4 = _mm_set1_epi8(0x0F_u8 as i8);
    let mask2 = _mm_set1_epi8(0x03_u8 as i8);

    // Lower nibbles from ql.
    let ql0_lo = _mm_and_si128(ql0, mask4); // ql[0..15] & 0x0F → q1 low bits
    let ql1_lo = _mm_and_si128(ql1, mask4); // ql[32..47] & 0x0F → q2 low bits

    // Upper nibbles from ql.
    // SAFETY: _mm_srli_epi16 shifts 16-bit lanes right; subsequent mask4 AND
    // removes any cross-byte contamination, leaving only the upper nibble in each byte.
    let ql0_hi = _mm_and_si128(_mm_srli_epi16(ql0, 4), mask4); // ql[0..15] >> 4 → q3 low bits
    let ql1_hi = _mm_and_si128(_mm_srli_epi16(ql1, 4), mask4); // ql[32..47] >> 4 → q4 low bits

    // 2-bit upper parts from qh, extracted via _mm_srli_epi16 + mask.
    // The 0x03 mask is narrow enough that no cross-byte contamination reaches
    // the kept bits for any of the four shift amounts (0, 2, 4, 6).
    let qh_sh0 = _mm_and_si128(qh_raw, mask2); // bits 1:0 → q1 high part
    let qh_sh2 = _mm_and_si128(_mm_srli_epi16(qh_raw, 2), mask2); // bits 3:2 → q2 high part
    let qh_sh4 = _mm_and_si128(_mm_srli_epi16(qh_raw, 4), mask2); // bits 5:4 → q3 high part
    let qh_sh6 = _mm_and_si128(_mm_srli_epi16(qh_raw, 6), mask2); // bits 7:6 → q4 high part

    // Shift qh 2-bit parts left by 4 to occupy bit positions 5:4.
    // SAFETY: _mm_slli_epi16 shifts 16-bit lanes; value is at most 3 << 4 = 48,
    // which fits in the low byte with no overflow into the adjacent byte.
    let qh_hi0 = _mm_slli_epi16(qh_sh0, 4); // 0x00 or 0x10 or 0x20 or 0x30
    let qh_hi2 = _mm_slli_epi16(qh_sh2, 4);
    let qh_hi4 = _mm_slli_epi16(qh_sh4, 4);
    let qh_hi6 = _mm_slli_epi16(qh_sh6, 4);

    // Combine: q = ql_low | qh_high → 6-bit quant in 0..63.
    let q1 = _mm_or_si128(ql0_lo, qh_hi0);
    let q2 = _mm_or_si128(ql1_lo, qh_hi2);
    let q3 = _mm_or_si128(ql0_hi, qh_hi4);
    let q4 = _mm_or_si128(ql1_hi, qh_hi6);

    (q1, q2, q3, q4)
}

/// Convert a `__m128i` of unsigned byte quants to signed int32 by subtracting 32,
/// then convert to float32.  Returns two `__m256` registers covering the first 8
/// and last 8 bytes respectively (= 16 floats).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q6_bytes_to_f32_pair(q: __m128i) -> (__m256, __m256) {
    let offset = _mm256_set1_epi32(32);

    let q_lo = _mm256_cvtepu8_epi32(q);
    let f_lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_lo, offset));

    let q_hi_bytes = _mm_srli_si128(q, 8);
    let q_hi = _mm256_cvtepu8_epi32(q_hi_bytes);
    let f_hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_hi, offset));

    (f_lo, f_hi)
}

/// Dequantize one 210-byte Q6_K block into 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 210`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    // SAFETY: block.len() >= 210 >= 210.
    let d = f16_to_f32(&block[208..]);

    // Process 2 groups of 128 weights each.
    for group in 0..2usize {
        let ql_off = group * 64; // ql base for this group's 32 positions
        let qh_off = group * 32; // qh base for this group's 32 positions
        let sc_off = group * 8; // scales base for 8 sub-block scales
        let out_off = group * 128; // output base

        // SAFETY: ql_off + 32 + 32 = ql_off + 64 <= 128; qh_off + 16 <= 64.
        // The second half (positions 16..31) is loaded inside the inner loop.
        //
        // We process the 32-position row in two 16-element halves because
        // `decode_q6_group` loads 16 bytes from ql[0..15] and ql[32..47]
        // plus 16 bytes from qh[0..15].

        // -------- First half: positions 0..15 --------
        let (q1_a, q2_a, q3_a, q4_a) =
            decode_q6_group(ql.as_ptr().add(ql_off), qh.as_ptr().add(qh_off));

        // -------- Second half: positions 16..31 --------
        let (q1_b, q2_b, q3_b, q4_b) =
            decode_q6_group(ql.as_ptr().add(ql_off + 16), qh.as_ptr().add(qh_off + 16));

        // Scale lookup: for positions 0..15 `is = l/16 = 0`, for 16..31 `is = 1`.
        // Four quant streams map to sub-blocks at sc_off + 0, +2, +4, +6 (for is=0).
        // SAFETY: scales.len() == 16; sc_off + 7 <= 15.
        let s0a = d * (*scales.get_unchecked(sc_off)) as i8 as f32;
        let s1a = d * (*scales.get_unchecked(sc_off + 2)) as i8 as f32;
        let s2a = d * (*scales.get_unchecked(sc_off + 4)) as i8 as f32;
        let s3a = d * (*scales.get_unchecked(sc_off + 6)) as i8 as f32;

        let s0b = d * (*scales.get_unchecked(sc_off + 1)) as i8 as f32;
        let s1b = d * (*scales.get_unchecked(sc_off + 3)) as i8 as f32;
        let s2b = d * (*scales.get_unchecked(sc_off + 5)) as i8 as f32;
        let s3b = d * (*scales.get_unchecked(sc_off + 7)) as i8 as f32;

        let vs0a = _mm256_set1_ps(s0a);
        let vs1a = _mm256_set1_ps(s1a);
        let vs2a = _mm256_set1_ps(s2a);
        let vs3a = _mm256_set1_ps(s3a);
        let vs0b = _mm256_set1_ps(s0b);
        let vs1b = _mm256_set1_ps(s1b);
        let vs2b = _mm256_set1_ps(s2b);
        let vs3b = _mm256_set1_ps(s3b);

        // q1: weights at out_off + 0..31
        // q2: weights at out_off + 32..63
        // q3: weights at out_off + 64..95
        // q4: weights at out_off + 96..127
        // _a halves = positions 0..15, _b halves = positions 16..31

        // --- q1 stream (out_off + 0..31) ---
        let (q1a_lo, q1a_hi) = q6_bytes_to_f32_pair(q1_a);
        let (q1b_lo, q1b_hi) = q6_bytes_to_f32_pair(q1_b);

        let ptr_q1 = output.as_mut_ptr().add(out_off);
        // SAFETY: out_off + 32 <= group*128 + 32 <= 256; output.len() >= 256.
        _mm256_storeu_ps(ptr_q1, _mm256_mul_ps(vs0a, q1a_lo));
        _mm256_storeu_ps(ptr_q1.add(8), _mm256_mul_ps(vs0a, q1a_hi));
        _mm256_storeu_ps(ptr_q1.add(16), _mm256_mul_ps(vs0b, q1b_lo));
        _mm256_storeu_ps(ptr_q1.add(24), _mm256_mul_ps(vs0b, q1b_hi));

        // --- q2 stream (out_off + 32..63) ---
        let (q2a_lo, q2a_hi) = q6_bytes_to_f32_pair(q2_a);
        let (q2b_lo, q2b_hi) = q6_bytes_to_f32_pair(q2_b);

        let ptr_q2 = output.as_mut_ptr().add(out_off + 32);
        // SAFETY: out_off + 64 <= 256.
        _mm256_storeu_ps(ptr_q2, _mm256_mul_ps(vs1a, q2a_lo));
        _mm256_storeu_ps(ptr_q2.add(8), _mm256_mul_ps(vs1a, q2a_hi));
        _mm256_storeu_ps(ptr_q2.add(16), _mm256_mul_ps(vs1b, q2b_lo));
        _mm256_storeu_ps(ptr_q2.add(24), _mm256_mul_ps(vs1b, q2b_hi));

        // --- q3 stream (out_off + 64..95) ---
        let (q3a_lo, q3a_hi) = q6_bytes_to_f32_pair(q3_a);
        let (q3b_lo, q3b_hi) = q6_bytes_to_f32_pair(q3_b);

        let ptr_q3 = output.as_mut_ptr().add(out_off + 64);
        // SAFETY: out_off + 96 <= 256.
        _mm256_storeu_ps(ptr_q3, _mm256_mul_ps(vs2a, q3a_lo));
        _mm256_storeu_ps(ptr_q3.add(8), _mm256_mul_ps(vs2a, q3a_hi));
        _mm256_storeu_ps(ptr_q3.add(16), _mm256_mul_ps(vs2b, q3b_lo));
        _mm256_storeu_ps(ptr_q3.add(24), _mm256_mul_ps(vs2b, q3b_hi));

        // --- q4 stream (out_off + 96..127) ---
        let (q4a_lo, q4a_hi) = q6_bytes_to_f32_pair(q4_a);
        let (q4b_lo, q4b_hi) = q6_bytes_to_f32_pair(q4_b);

        let ptr_q4 = output.as_mut_ptr().add(out_off + 96);
        // SAFETY: out_off + 128 <= 256.
        _mm256_storeu_ps(ptr_q4, _mm256_mul_ps(vs3a, q4a_lo));
        _mm256_storeu_ps(ptr_q4.add(8), _mm256_mul_ps(vs3a, q4a_hi));
        _mm256_storeu_ps(ptr_q4.add(16), _mm256_mul_ps(vs3b, q4b_lo));
        _mm256_storeu_ps(ptr_q4.add(24), _mm256_mul_ps(vs3b, q4b_hi));
    }
}

/// Compute the dot product of one row of a Q6_K matrix with an FP32 vector.
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
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];
        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        // SAFETY: block.len() == 210 >= 210.
        let d = f16_to_f32(&block[208..]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.
            let mut block_acc = _mm256_setzero_ps();

            for group in 0..2usize {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let w_off = input_offset + group * 128;

                // SAFETY: sc_off + 7 <= 15; scales.len() == 16.
                let s0a = d * (*scales.get_unchecked(sc_off)) as i8 as f32;
                let s1a = d * (*scales.get_unchecked(sc_off + 2)) as i8 as f32;
                let s2a = d * (*scales.get_unchecked(sc_off + 4)) as i8 as f32;
                let s3a = d * (*scales.get_unchecked(sc_off + 6)) as i8 as f32;
                let s0b = d * (*scales.get_unchecked(sc_off + 1)) as i8 as f32;
                let s1b = d * (*scales.get_unchecked(sc_off + 3)) as i8 as f32;
                let s2b = d * (*scales.get_unchecked(sc_off + 5)) as i8 as f32;
                let s3b = d * (*scales.get_unchecked(sc_off + 7)) as i8 as f32;

                let vs0a = _mm256_set1_ps(s0a);
                let vs1a = _mm256_set1_ps(s1a);
                let vs2a = _mm256_set1_ps(s2a);
                let vs3a = _mm256_set1_ps(s3a);
                let vs0b = _mm256_set1_ps(s0b);
                let vs1b = _mm256_set1_ps(s1b);
                let vs2b = _mm256_set1_ps(s2b);
                let vs3b = _mm256_set1_ps(s3b);

                // First half: positions 0..15
                let (q1_a, q2_a, q3_a, q4_a) =
                    decode_q6_group(ql.as_ptr().add(ql_off), qh.as_ptr().add(qh_off));
                // Second half: positions 16..31
                let (q1_b, q2_b, q3_b, q4_b) =
                    decode_q6_group(ql.as_ptr().add(ql_off + 16), qh.as_ptr().add(qh_off + 16));

                let off32 = _mm256_set1_epi32(32);

                // Helper inline for (q_bytes, scale_vec, inp_ptr, block_acc):
                // q1 stream: w_off + 0..31
                let inp_q1 = input.as_ptr().add(w_off);
                // SAFETY: w_off + 32 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let q1a_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q1_a), off32);
                let q1a_lo_f = _mm256_cvtepi32_ps(q1a_lo_i);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs0a, q1a_lo_f),
                    _mm256_loadu_ps(inp_q1),
                    block_acc,
                );

                let q1a_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q1_a, 8)), off32);
                let q1a_hi_f = _mm256_cvtepi32_ps(q1a_hi_i);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs0a, q1a_hi_f),
                    _mm256_loadu_ps(inp_q1.add(8)),
                    block_acc,
                );

                let q1b_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q1_b), off32);
                let q1b_lo_f = _mm256_cvtepi32_ps(q1b_lo_i);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs0b, q1b_lo_f),
                    _mm256_loadu_ps(inp_q1.add(16)),
                    block_acc,
                );

                let q1b_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q1_b, 8)), off32);
                let q1b_hi_f = _mm256_cvtepi32_ps(q1b_hi_i);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs0b, q1b_hi_f),
                    _mm256_loadu_ps(inp_q1.add(24)),
                    block_acc,
                );

                // q2 stream: w_off + 32..63
                let inp_q2 = input.as_ptr().add(w_off + 32);
                let q2a_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q2_a), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs1a, _mm256_cvtepi32_ps(q2a_lo_i)),
                    _mm256_loadu_ps(inp_q2),
                    block_acc,
                );
                let q2a_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q2_a, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs1a, _mm256_cvtepi32_ps(q2a_hi_i)),
                    _mm256_loadu_ps(inp_q2.add(8)),
                    block_acc,
                );
                let q2b_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q2_b), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs1b, _mm256_cvtepi32_ps(q2b_lo_i)),
                    _mm256_loadu_ps(inp_q2.add(16)),
                    block_acc,
                );
                let q2b_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q2_b, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs1b, _mm256_cvtepi32_ps(q2b_hi_i)),
                    _mm256_loadu_ps(inp_q2.add(24)),
                    block_acc,
                );

                // q3 stream: w_off + 64..95
                let inp_q3 = input.as_ptr().add(w_off + 64);
                let q3a_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q3_a), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs2a, _mm256_cvtepi32_ps(q3a_lo_i)),
                    _mm256_loadu_ps(inp_q3),
                    block_acc,
                );
                let q3a_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q3_a, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs2a, _mm256_cvtepi32_ps(q3a_hi_i)),
                    _mm256_loadu_ps(inp_q3.add(8)),
                    block_acc,
                );
                let q3b_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q3_b), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs2b, _mm256_cvtepi32_ps(q3b_lo_i)),
                    _mm256_loadu_ps(inp_q3.add(16)),
                    block_acc,
                );
                let q3b_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q3_b, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs2b, _mm256_cvtepi32_ps(q3b_hi_i)),
                    _mm256_loadu_ps(inp_q3.add(24)),
                    block_acc,
                );

                // q4 stream: w_off + 96..127
                let inp_q4 = input.as_ptr().add(w_off + 96);
                let q4a_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q4_a), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs3a, _mm256_cvtepi32_ps(q4a_lo_i)),
                    _mm256_loadu_ps(inp_q4),
                    block_acc,
                );
                let q4a_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q4_a, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs3a, _mm256_cvtepi32_ps(q4a_hi_i)),
                    _mm256_loadu_ps(inp_q4.add(8)),
                    block_acc,
                );
                let q4b_lo_i = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q4_b), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs3b, _mm256_cvtepi32_ps(q4b_lo_i)),
                    _mm256_loadu_ps(inp_q4.add(16)),
                    block_acc,
                );
                let q4b_hi_i =
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q4_b, 8)), off32);
                block_acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(vs3b, _mm256_cvtepi32_ps(q4b_hi_i)),
                    _mm256_loadu_ps(inp_q4.add(24)),
                    block_acc,
                );
            }

            row_sum += hsum_f32_avx(block_acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut partial_sum = 0.0f32;

            for group in 0..2usize {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let in_off = input_offset + group * 128;

                for l in 0..32 {
                    let is = l / 16;

                    // SAFETY: ql_off + l + 32 < 128; qh_off + l < 64; sc_off + is + 6 < 16.
                    let ql_l = *ql.get_unchecked(ql_off + l);
                    let ql_l32 = *ql.get_unchecked(ql_off + l + 32);
                    let qh_l = *qh.get_unchecked(qh_off + l);

                    let q1 = ((ql_l & 0x0F) | ((qh_l & 3) << 4)) as i32 - 32;
                    let q2 = ((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) as i32 - 32;

                    let s0 = d * (*scales.get_unchecked(sc_off + is)) as i8 as f32;
                    let s1 = d * (*scales.get_unchecked(sc_off + is + 2)) as i8 as f32;
                    let s2 = d * (*scales.get_unchecked(sc_off + is + 4)) as i8 as f32;
                    let s3 = d * (*scales.get_unchecked(sc_off + is + 6)) as i8 as f32;

                    let idx0 = in_off + l;
                    let idx1 = in_off + l + 32;
                    let idx2 = in_off + l + 64;
                    let idx3 = in_off + l + 96;

                    if idx0 < n_cols {
                        partial_sum += s0 * q1 as f32 * input[idx0];
                    }
                    if idx1 < n_cols {
                        partial_sum += s1 * q2 as f32 * input[idx1];
                    }
                    if idx2 < n_cols {
                        partial_sum += s2 * q3 as f32 * input[idx2];
                    }
                    if idx3 < n_cols {
                        partial_sum += s3 * q4 as f32 * input[idx3];
                    }
                }
            }

            row_sum += partial_sum;
        }
        // remaining == 0: block fully out of bounds, skip.
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q6_k::Q6KRef;

    fn make_q6k_block(d: f32, ql: &[u8; 128], qh: &[u8; 64], scales: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(ql);
        block.extend_from_slice(qh);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> crate::types::QuantTensor {
        crate::types::QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q6K)
    }

    #[test]
    fn test_q6k_avx2_dequant_matches_reference_zero() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // d=0 → all weights zero.
        let block = make_q6k_block(0.0, &[0; 128], &[0; 64], &[0; 16]);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q6KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "dequant mismatch [zero] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q6k_avx2_dequant_matches_reference_quant32() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // All quants = 32 (q-32 = 0) → all weights = 0 regardless of scales.
        // ql lower nibbles = 0, qh 2-bit parts = 2 (0b10) → quant = 0 | (2 << 4) = 32.
        // qh packed: each byte encodes 4 × 2-bit = 0b10101010 = 0xAA.
        let qh = [0xAAu8; 64];
        let scales: [u8; 16] = [1; 16];

        let block = make_q6k_block(1.0, &[0; 128], &qh, &scales);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q6KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [quant32] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q6k_avx2_dequant_matches_reference_varied() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as i8 * 3 - 8) as u8;
        }

        let block = make_q6k_block(0.5, &ql, &qh, &scales);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q6KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [varied] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q6k_avx2_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as i8 * 3 - 8) as u8;
        }

        let block = make_q6k_block(0.5, &ql, &qh, &scales);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q6_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv");
        Q6KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q6k_avx2_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 200 columns — partial block.
        let scales = [1i8 as u8; 16];
        let block = make_q6k_block(1.0, &[0; 128], &[0xAAu8; 64], &scales);
        let tensor_avx2 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q6_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv partial");
        Q6KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }
}
