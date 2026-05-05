//! AVX2+FMA accelerated Q3_K quantization kernel.
//!
//! Q3_K block layout (110 bytes per 256 weights):
//! - bytes\[0..32\]   — hmask: 1 bit per weight; if set → subtract 0; if clear → subtract 4
//! - bytes\[32..96\]  — qs: lower 2 bits of each 3-bit quant (4 per byte via shifts)
//! - bytes\[96..108\] — 12 bytes packed scales (16 × 6-bit unsigned, decode then
//!   subtract 32 for signed range -32..31)
//! - bytes\[108..110\] — FP16 super-block scale `d` (little-endian)
//!
//! Q3_K is a symmetric ("type-0") format — NO minimum offset.
//! Weight formula: `w = d * scale_i * (q_lo - (hmask_bit ? 0 : 4))`
//!   where q_lo is 2-bit (0..3), giving effective range -4..3.
//!
//! 16 sub-blocks of 16 weights each (2 groups × 4 shifts × 2 sub-blocks).
//! The hmask bit selector `m` rotates through bits 0..7 across the 8 passes.
//!
//! AVX2 strategy: extract 2-bit q_lo via `_mm_srli_epi16` + AND 0x03; extract
//! hmask correction via `_mm_cmpeq_epi8` + AND(4); widen both to i32, subtract,
//! convert to f32, multiply by `d * scale`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q3_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q3_K block: 32 (hmask) + 64 (qs) + 12 (scales) + 2 (FP16 d).
pub const BLOCK_BYTES: usize = 110;

/// AVX2+FMA accelerated Q3_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q3_KAvx2;

/// Decode 16 signed 6-bit scales from the 12-byte packed representation.
///
/// The packing uses 16 × 6-bit values stored across 12 bytes.
/// The decoded values are unsigned 0..63, then we subtract 32 to get signed -32..31.
fn decode_scales(scales_raw: &[u8]) -> [f32; 16] {
    let mut sc = [0u32; 16];

    // Scales 0..3: lower 6 bits of bytes 0..3
    for j in 0..4 {
        sc[j] = (scales_raw[j] & 0x3F) as u32;
    }
    // Scales 4..7: lower 6 bits of bytes 4..7
    for j in 0..4 {
        sc[4 + j] = (scales_raw[4 + j] & 0x3F) as u32;
    }
    // Scales 8..11: lo 4 bits from bytes 8..11, hi 2 bits from upper bits of bytes 0..3
    for j in 0..4 {
        let lo = (scales_raw[8 + j] & 0x0F) as u32;
        let hi = ((scales_raw[j] >> 6) & 0x03) as u32;
        sc[8 + j] = lo | (hi << 4);
    }
    // Scales 12..15: hi 4 bits from bytes 8..11, hi 2 bits from upper bits of bytes 4..7
    for j in 0..4 {
        let lo = ((scales_raw[8 + j] >> 4) & 0x0F) as u32;
        let hi = ((scales_raw[4 + j] >> 6) & 0x03) as u32;
        sc[12 + j] = lo | (hi << 4);
    }

    // Convert to signed: subtract 32
    let mut result = [0.0f32; 16];
    for i in 0..16 {
        result[i] = (sc[i] as i32 - 32) as f32;
    }
    result
}

/// Extract 2-bit values from 16 packed bytes using the given bit-shift.
///
/// Same technique as Q2_K: `_mm_srli_epi16` shifts 16-bit lanes but cross-byte
/// leakage is always a multiple of 4, which vanishes under the 0x03 mask.
///
/// # Safety
/// Requires `avx2` CPU feature.  `shift` must be one of 0, 2, 4, 6.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_2bit_16(raw: __m128i, shift: u32, mask: __m128i) -> __m128i {
    // SAFETY: each branch uses a compile-time const generic for _mm_srli_epi16.
    let shifted = match shift {
        0 => raw,
        2 => _mm_srli_epi16::<2>(raw),
        4 => _mm_srli_epi16::<4>(raw),
        _ => _mm_srli_epi16::<6>(raw),
    };
    _mm_and_si128(shifted, mask)
}

/// Compute the hmask correction vector for 16 positions.
///
/// For each byte position: if the corresponding hmask bit is SET → correction
/// is 0 (subtract nothing); if CLEAR → correction is 4 (subtract 4).
///
/// Uses `_mm_cmpeq_epi8(masked, zero)` which yields 0xFF where the bit was
/// clear (i.e., masked byte == 0).  AND with 4 gives the correction value.
///
/// # Safety
/// Requires `avx2` CPU feature.  `m` selects which bit to test (one-hot byte).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmask_correction_16(hmask_half: __m128i, m_vec: __m128i) -> __m128i {
    // SAFETY: all intrinsics require avx2, guaranteed by target_feature.
    let masked = _mm_and_si128(hmask_half, m_vec);
    let is_zero = _mm_cmpeq_epi8(masked, _mm_setzero_si128());
    // is_zero: 0xFF where bit was clear (subtract 4), 0x00 where set (subtract 0)
    _mm_and_si128(is_zero, _mm_set1_epi8(4))
}

impl QuantKernel for Q3_KAvx2 {
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

        // SAFETY: block.len() >= 110 and output.len() >= 256 verified above.
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

    /// Fused Q3_K weight × Q8_0 activation GEMV using AVX2+FMA.
    ///
    /// Each Q3_K super-block (256 weights, 110 bytes) maps to 8 Q8_0 activation
    /// blocks (32 weights each, 34 bytes each).  The 256 weights are organized as
    /// 16 sub-blocks of 16 weights each.  Two sub-blocks (at positions `sb*16`
    /// and `sb*16+16`) share one Q8_0 block of 32 activations.
    ///
    /// Fused formula per sub-block:  `dl * Σ(q3_i * q8_i) * d_a`
    /// where `q3_i = q_lo_i - correction_i` (correction = 4 if hmask bit clear).
    /// No intermediate f32 scratch buffer is allocated.
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
        // One Q3_K super-block → 8 Q8_0 blocks (256 weights / 32 per Q8_0 block)
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

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked above; CPU avx2+fma guaranteed by KernelDispatcher.
            let row_sum = unsafe {
                fused_q3k_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            *out_val += row_sum;
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
        "Q3_K"
    }
}

/// Q8_0 block bytes for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Fused Q3_K weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// Q3_K layout (per 256-weight block): hmask[0..32], qs[32..96], scales[96..108], d[108..110].
/// 256 weights = 16 sub-blocks × 16 weights. Two consecutive sub-blocks map to one Q8_0 block
/// of 32 activations.  Sub-block `sb` (0..16) → Q8_0 block `blk*8 + sb/2`.
///
/// For each sub-block `sb` with signed scale `dl = d * sc[sb]`:
///   contribution = dl * Σ(q3_i * q8_i) * d_a
/// where q3_i = q_lo_i - correction_i (correction = 4 if hmask bit clear, else 0).
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q3k_q8_0_row_avx2(
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

        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        // SAFETY: block.len() >= 110.
        let d = f16_to_f32(&block[108..]);
        let sc = decode_scales(scales_raw);

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        if cols_in_block < BLOCK_SIZE {
            // Scalar tail path — partial block.
            // Mirrors the same iteration order as the fast path to keep index alignment.
            let mut is = 0usize;
            let mut col_off = 0usize;
            let mut m_bit: u8 = 1;

            for group in 0..2usize {
                let qs_base = group * 32;
                for _shift_idx in 0..4u32 {
                    let shift = _shift_idx * 2;
                    // Sub-block A (hmask[0..16])
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_vals = &a_block[2..];
                        let mut dot = 0.0f32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q_lo = ((qs[qs_base + l] >> shift) & 3) as i32;
                                let sub = if hmask[l] & m_bit != 0 { 0 } else { 4 };
                                let q3 = (q_lo - sub) as f32;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                dot += q3 * q_a;
                            }
                        }
                        row_sum += dl * dot * d_a;
                        col_off += 16;
                    }
                    // Sub-block B (hmask[16..32])
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_vals = &a_block[2..];
                        let mut dot = 0.0f32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q_lo = ((qs[qs_base + 16 + l] >> shift) & 3) as i32;
                                let sub = if hmask[16 + l] & m_bit != 0 { 0 } else { 4 };
                                let q3 = (q_lo - sub) as f32;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                dot += q3 * q_a;
                            }
                        }
                        row_sum += dl * dot * d_a;
                        col_off += 16;
                    }
                    m_bit = m_bit.wrapping_shl(1);
                }
            }
        } else {
            // Fast path: all 256 weights in bounds.
            // SAFETY: hmask.len() == 32.
            let hmask_lo = _mm_loadu_si128(hmask.as_ptr() as *const __m128i);
            let hmask_hi = _mm_loadu_si128(hmask.as_ptr().add(16) as *const __m128i);
            let mask_2bit = _mm_set1_epi8(0x03);

            let mut is = 0usize;
            let mut col_off = 0usize;

            for group in 0..2usize {
                let qs_base = group * 32;
                // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
                let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
                let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

                for shift_idx in 0..4u32 {
                    let shift = shift_idx * 2;
                    let bit_pos = (group as u32) * 4 + shift_idx;
                    // SAFETY: bit_pos is 0..7, so 1u8 << bit_pos is in range 1..128.
                    let m: u8 = 1u8 << bit_pos;
                    let m_vec = _mm_set1_epi8(m as i8);

                    // Sub-block A: 16 weights from hmask[0..16], qs[qs_base..qs_base+16]
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const __m128i;

                        // SAFETY: extract_2bit_16 requires avx2, shift in {0,2,4,6}.
                        let q2_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                        // SAFETY: hmask_correction_16 requires avx2.
                        let corr_bytes = hmask_correction_16(hmask_lo, m_vec);

                        // SAFETY: q8_ptr points to 16 valid i8 bytes.
                        let qa_bytes = _mm_loadu_si128(q8_ptr);

                        // q3_lo = q2 - correction (signed i32 arithmetic)
                        let q2_lo = _mm256_cvtepu8_epi32(q2_bytes);
                        let q2_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q2_bytes, 8));
                        let c_lo = _mm256_cvtepu8_epi32(corr_bytes);
                        let c_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(corr_bytes, 8));
                        let q3_lo = _mm256_sub_epi32(q2_lo, c_lo);
                        let q3_hi = _mm256_sub_epi32(q2_hi, c_hi);

                        // widen q8 i8 → i32
                        let qa_lo = _mm256_cvtepi8_epi32(qa_bytes);
                        let qa_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(qa_bytes, 8));

                        // dot = Σ(q3 * q8)
                        let dot_acc = _mm256_add_epi32(
                            _mm256_mullo_epi32(q3_lo, qa_lo),
                            _mm256_mullo_epi32(q3_hi, qa_hi),
                        );

                        let dot = hsum_i32_avx2_q3k(dot_acc) as f32;
                        row_sum += dl * dot * d_a;
                        col_off += 16;
                    }

                    // Sub-block B: 16 weights from hmask[16..32], qs[qs_base+16..qs_base+32]
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const __m128i;

                        let q2_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                        let corr_bytes = hmask_correction_16(hmask_hi, m_vec);

                        let qa_bytes = _mm_loadu_si128(q8_ptr);

                        let q2_lo = _mm256_cvtepu8_epi32(q2_bytes);
                        let q2_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q2_bytes, 8));
                        let c_lo = _mm256_cvtepu8_epi32(corr_bytes);
                        let c_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(corr_bytes, 8));
                        let q3_lo = _mm256_sub_epi32(q2_lo, c_lo);
                        let q3_hi = _mm256_sub_epi32(q2_hi, c_hi);

                        let qa_lo = _mm256_cvtepi8_epi32(qa_bytes);
                        let qa_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(qa_bytes, 8));

                        let dot_acc = _mm256_add_epi32(
                            _mm256_mullo_epi32(q3_lo, qa_lo),
                            _mm256_mullo_epi32(q3_hi, qa_hi),
                        );

                        let dot = hsum_i32_avx2_q3k(dot_acc) as f32;
                        row_sum += dl * dot * d_a;
                        col_off += 16;
                    }
                }
            }
        }
    }

    row_sum
}

/// Horizontal sum of a 256-bit i32 register (AVX2).
///
/// # Safety
/// Requires `avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2_q3k(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(s, 0b10_11_00_01);
    let s2 = _mm_add_epi32(s, shuf);
    let shuf2 = _mm_shuffle_epi32(s2, 0b00_00_10_10);
    let s3 = _mm_add_epi32(s2, shuf2);
    _mm_cvtsi128_si32(s3)
}

/// Dequantize one 110-byte Q3_K block into 256 FP32 values using AVX2.
///
/// Processes 2 groups of 128 weights.  Within each group, the same 32 qs
/// bytes are re-used with shifts 0, 2, 4, 6 and the same 32 hmask bytes
/// are tested with a rotating 1-bit selector.
///
/// # Safety
/// - `block.len() >= 110`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales_raw = &block[96..108];

    // SAFETY: block.len() >= 110, so byte offsets 108..110 are valid.
    let d = f16_to_f32(&block[108..]);

    let sc = decode_scales(scales_raw);

    // Pre-load both halves of the hmask (persistent across all passes).
    // SAFETY: hmask.len() == 32.
    let hmask_lo = _mm_loadu_si128(hmask.as_ptr() as *const __m128i);
    let hmask_hi = _mm_loadu_si128(hmask.as_ptr().add(16) as *const __m128i);

    let mask_2bit = _mm_set1_epi8(0x03);

    let mut is = 0usize;
    let mut out_off = 0usize;

    for group in 0..2usize {
        let qs_base = group * 32;

        // Pre-load the 32 qs bytes for this group.
        // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
        let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
        let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

        for shift_idx in 0..4u32 {
            let shift = shift_idx * 2;
            let bit_pos = (group as u32) * 4 + shift_idx;
            // SAFETY: bit_pos is 0..7, so 1u8 << bit_pos is in range 1..128.
            let m: u8 = 1u8 << bit_pos;
            let m_vec = _mm_set1_epi8(m as i8);

            // --- Sub-block 0: 16 weights, hmask[0..16] ---
            {
                let dl = d * sc[is];
                is += 1;
                let vdl = _mm256_set1_ps(dl);

                // SAFETY: extract_2bit_16 requires avx2 and shift in {0,2,4,6}.
                let q_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                // SAFETY: hmask_correction_16 requires avx2.
                let correction = hmask_correction_16(hmask_lo, m_vec);

                // First 8 weights
                // SAFETY: _mm256_cvtepu8_epi32 reads from the low 8 bytes.
                let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                let c_i32 = _mm256_cvtepu8_epi32(correction);
                let val_i32 = _mm256_sub_epi32(q_i32, c_i32);
                let val_f32 = _mm256_cvtepi32_ps(val_i32);
                let w0 = _mm256_mul_ps(vdl, val_f32);

                // Next 8 weights
                let q_bytes_hi8 = _mm_srli_si128(q_bytes, 8);
                let correction_hi8 = _mm_srli_si128(correction, 8);
                let q_i32_2 = _mm256_cvtepu8_epi32(q_bytes_hi8);
                let c_i32_2 = _mm256_cvtepu8_epi32(correction_hi8);
                let val_i32_2 = _mm256_sub_epi32(q_i32_2, c_i32_2);
                let val_f32_2 = _mm256_cvtepi32_ps(val_i32_2);
                let w1 = _mm256_mul_ps(vdl, val_f32_2);

                // SAFETY: out_off + 16 <= 256; output.len() >= 256.
                let ptr = output.as_mut_ptr().add(out_off);
                _mm256_storeu_ps(ptr, w0);
                _mm256_storeu_ps(ptr.add(8), w1);
                out_off += 16;
            }

            // --- Sub-block 1: 16 weights, hmask[16..32] ---
            {
                let dl = d * sc[is];
                is += 1;
                let vdl = _mm256_set1_ps(dl);

                let q_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                let correction = hmask_correction_16(hmask_hi, m_vec);

                // First 8 weights
                let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                let c_i32 = _mm256_cvtepu8_epi32(correction);
                let val_i32 = _mm256_sub_epi32(q_i32, c_i32);
                let val_f32 = _mm256_cvtepi32_ps(val_i32);
                let w2 = _mm256_mul_ps(vdl, val_f32);

                // Next 8 weights
                let q_bytes_hi8 = _mm_srli_si128(q_bytes, 8);
                let correction_hi8 = _mm_srli_si128(correction, 8);
                let q_i32_2 = _mm256_cvtepu8_epi32(q_bytes_hi8);
                let c_i32_2 = _mm256_cvtepu8_epi32(correction_hi8);
                let val_i32_2 = _mm256_sub_epi32(q_i32_2, c_i32_2);
                let val_f32_2 = _mm256_cvtepi32_ps(val_i32_2);
                let w3 = _mm256_mul_ps(vdl, val_f32_2);

                // SAFETY: out_off + 16 <= 256; output.len() >= 256.
                let ptr = output.as_mut_ptr().add(out_off);
                _mm256_storeu_ps(ptr, w2);
                _mm256_storeu_ps(ptr.add(8), w3);
                out_off += 16;
            }
        }
    }
}

/// Compute the dot product of one row of a Q3_K matrix with an FP32 vector.
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

        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];

        // SAFETY: block.len() == 110 >= 110.
        let d = f16_to_f32(&block[108..]);
        let sc = decode_scales(scales_raw);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.

            // SAFETY: hmask.len() == 32.
            let hmask_lo = _mm_loadu_si128(hmask.as_ptr() as *const __m128i);
            let hmask_hi = _mm_loadu_si128(hmask.as_ptr().add(16) as *const __m128i);

            let mask_2bit = _mm_set1_epi8(0x03);
            let mut block_acc = _mm256_setzero_ps();
            let mut is = 0usize;
            let mut w_off = input_offset;

            for group in 0..2usize {
                let qs_base = group * 32;

                // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
                let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
                let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

                for shift_idx in 0..4u32 {
                    let shift = shift_idx * 2;
                    let bit_pos = (group as u32) * 4 + shift_idx;
                    // SAFETY: bit_pos is 0..7.
                    let m: u8 = 1u8 << bit_pos;
                    let m_vec = _mm_set1_epi8(m as i8);

                    // --- Sub-block 0: 16 weights, hmask bytes 0..15 ---
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let vdl = _mm256_set1_ps(dl);

                        let q_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                        let correction = hmask_correction_16(hmask_lo, m_vec);

                        // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                        let inp_ptr = input.as_ptr().add(w_off);

                        // First 8
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                        let c_i32 = _mm256_cvtepu8_epi32(correction);
                        let val_f32 = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_i32, c_i32));
                        let w0 = _mm256_mul_ps(vdl, val_f32);
                        let i0 = _mm256_loadu_ps(inp_ptr);
                        block_acc = _mm256_fmadd_ps(w0, i0, block_acc);

                        // Next 8
                        let q_hi8 = _mm_srli_si128(q_bytes, 8);
                        let c_hi8 = _mm_srli_si128(correction, 8);
                        let q_i32_2 = _mm256_cvtepu8_epi32(q_hi8);
                        let c_i32_2 = _mm256_cvtepu8_epi32(c_hi8);
                        let val_f32_2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_i32_2, c_i32_2));
                        let w1 = _mm256_mul_ps(vdl, val_f32_2);
                        let i1 = _mm256_loadu_ps(inp_ptr.add(8));
                        block_acc = _mm256_fmadd_ps(w1, i1, block_acc);

                        w_off += 16;
                    }

                    // --- Sub-block 1: 16 weights, hmask bytes 16..31 ---
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let vdl = _mm256_set1_ps(dl);

                        let q_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                        let correction = hmask_correction_16(hmask_hi, m_vec);

                        // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                        let inp_ptr = input.as_ptr().add(w_off);

                        // First 8
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                        let c_i32 = _mm256_cvtepu8_epi32(correction);
                        let val_f32 = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_i32, c_i32));
                        let w2 = _mm256_mul_ps(vdl, val_f32);
                        let i2 = _mm256_loadu_ps(inp_ptr);
                        block_acc = _mm256_fmadd_ps(w2, i2, block_acc);

                        // Next 8
                        let q_hi8 = _mm_srli_si128(q_bytes, 8);
                        let c_hi8 = _mm_srli_si128(correction, 8);
                        let q_i32_2 = _mm256_cvtepu8_epi32(q_hi8);
                        let c_i32_2 = _mm256_cvtepu8_epi32(c_hi8);
                        let val_f32_2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(q_i32_2, c_i32_2));
                        let w3 = _mm256_mul_ps(vdl, val_f32_2);
                        let i3 = _mm256_loadu_ps(inp_ptr.add(8));
                        block_acc = _mm256_fmadd_ps(w3, i3, block_acc);

                        w_off += 16;
                    }
                }
            }

            row_sum += hsum_f32_avx(block_acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut partial_sum = 0.0f32;
            let mut is = 0usize;
            let mut in_off = input_offset;
            let mut m_bit: u8 = 1;

            for group in 0..2usize {
                let qs_base = group * 32;

                for shift in (0u32..8).step_by(2) {
                    for n in 0..2usize {
                        let dl = d * sc[is];
                        is += 1;

                        for l in 0..16 {
                            let idx = in_off + l;
                            if idx < n_cols {
                                let qs_idx = qs_base + n * 16 + l;
                                // SAFETY: qs_idx < 64; hmask index n*16+l < 32.
                                let q_lo = ((*qs.get_unchecked(qs_idx) >> shift) & 3) as i32;
                                let subtract = if *hmask.get_unchecked(n * 16 + l) & m_bit != 0 {
                                    0
                                } else {
                                    4
                                };
                                partial_sum += dl * (q_lo - subtract) as f32 * input[idx];
                            }
                        }
                        in_off += 16;
                    }
                    m_bit = m_bit.wrapping_shl(1);
                }
            }

            row_sum += partial_sum;
        }
        // remaining == 0: block fully out of bounds, skip.
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q3_k::Q3KRef;

    fn make_q3k_block(d: f32, scales: &[u8; 12], hmask: &[u8; 32], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(hmask);
        block.extend_from_slice(qs);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q3K)
    }

    #[test]
    fn test_dequant_matches_reference_zeros() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // d=0 → all weights = 0
        let block = make_q3k_block(0.0, &[0; 12], &[0; 32], &[0; 64]);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "dequant mismatch [zeros] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_hmask_all_set() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // hmask all set → subtract 0.  qs all 0 → q_lo = 0.
        // Value = 0 - 0 = 0.  Weight = d * scale * 0 = 0.
        let hmask = [0xFFu8; 32];
        let qs = [0x00u8; 64];
        // Scales = 33 (raw) → signed = 1
        let mut scales = [0u8; 12];
        scales[..8].fill(0x21);

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let mut out_avx2 = vec![99.0f32; 256];
        let mut out_ref = vec![99.0f32; 256];

        Q3_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [hmask_set] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_hmask_all_clear() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // hmask all clear → subtract 4.  qs all 0 → q_lo = 0.
        // Value = 0 - 4 = -4.  Weight = d * scale * (-4).
        // d=1.0, all scales = signed +1 (raw 33).
        // Weight = 1.0 * 1 * (-4) = -4.0
        let hmask = [0x00u8; 32];
        let qs = [0x00u8; 64];
        // All 16 scales = raw 33 → signed 1.
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [hmask_clear] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_varied() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0x3F) as u8;
        }
        let mut hmask = [0u8; 32];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let block = make_q3k_block(0.5, &scales, &hmask, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [varied] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0x3F) as u8;
        }
        let mut hmask = [0u8; 32];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let block = make_q3k_block(0.5, &scales, &hmask, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
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
        // 200 columns — partial block.
        let hmask = [0xAAu8; 32]; // alternating bits
        let qs = [0x55u8; 64];
        let mut scales = [0u8; 12];
        scales[..8].fill(0x21);

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv partial");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_alternating_hmask() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 11 + 5) & 0x3F) as u8;
        }
        let mut hmask = [0u8; 32];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = if i % 2 == 0 { 0xAA } else { 0x55 };
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 3) & 0xFF) as u8;
        }

        let block = make_q3k_block(0.75, &scales, &hmask, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv alternating");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv alternating");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "alternating gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_buffer_too_small_block() {
        let block = vec![0u8; 10]; // too small
        let mut output = vec![0.0f32; 256];
        assert!(Q3_KAvx2.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_buffer_too_small_output() {
        let block = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; 10]; // too small
        assert!(Q3_KAvx2.dequant_block(&block, &mut output).is_err());
    }

    // ── matvec_q8_fused Q3_K ─────────────────────────────────────────────

    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    fn make_q8_acts(n_q8_blocks: usize, scale: f32, values: &[i8; 32]) -> Vec<u8> {
        make_q8_0_block(scale, values).repeat(n_q8_blocks)
    }

    #[test]
    fn test_q3k_avx2_fused_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // One Q3_K super-block (256 weights) needs 8 Q8_0 activation blocks.
        let mut scales_raw = [0u8; 12];
        for (i, s) in scales_raw.iter_mut().enumerate() {
            *s = ((i * 11 + 3) & 0x3F) as u8;
        }
        let mut hmask = [0u8; 32];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 3) & 0xFF) as u8;
        }

        let w_block = make_q3k_block(0.5, &scales_raw, &hmask, &qs);
        let act_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 0, 1, -1, 2, -2, 3, -3,
            4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let acts = make_q8_acts(8, 0.1, &act_vals);

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out_avx2, 1, 256)
            .expect("avx2 fused q3k single block");
        Q3KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, 256)
            .expect("ref fused q3k single block");

        let err = (out_avx2[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "q3k_avx2_fused_matches_reference: avx2={} ref={} err={}",
            out_avx2[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q3k_avx2_fused_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 3 rows × 512 cols = 2 Q3_K super-blocks per row → 16 Q8_0 act blocks.
        let n_rows = 3usize;
        let n_cols = 512usize;
        let blocks_per_row = 2usize;
        let q8_blocks_per_row = blocks_per_row * 8; // 16

        let mut all_weights = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut scales_raw = [0u8; 12];
                for (i, s) in scales_raw.iter_mut().enumerate() {
                    *s = ((r * 11 + b * 7 + i * 5 + 3) & 0x3F) as u8;
                }
                let mut hmask = [0u8; 32];
                for (i, h) in hmask.iter_mut().enumerate() {
                    *h = ((r * 13 + b * 17 + i * 7) & 0xFF) as u8;
                }
                let mut qs = [0u8; 64];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((r * 5 + b * 23 + i * 11) & 0xFF) as u8;
                }
                all_weights.extend(make_q3k_block(
                    0.5 + r as f32 * 0.1,
                    &scales_raw,
                    &hmask,
                    &qs,
                ));
            }
        }

        let act_vals: [i8; 32] = [
            2, -3, 5, -7, 1, -1, 4, -4, 6, -6, 3, -3, 2, -2, 1, -1, 8, -8, 7, -7, 6, -6, 5, -5, 4,
            -4, 3, -3, 2, -2, 1, -1,
        ];
        let acts = make_q8_acts(q8_blocks_per_row, 0.05, &act_vals);

        let mut out_avx2 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q3_KAvx2
            .matvec_q8_fused(&all_weights, &acts, &mut out_avx2, n_rows, n_cols)
            .expect("avx2 fused q3k multi-row");
        Q3KRef
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused q3k multi-row");

        for i in 0..n_rows {
            let err = (out_avx2[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "q3k fused multi-row row {i}: avx2={} ref={} err={}",
                out_avx2[i],
                out_ref[i],
                err
            );
        }
    }
}
