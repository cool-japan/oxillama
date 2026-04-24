//! AVX2+FMA accelerated Q4_K quantization kernel.
//!
//! Q4_K block layout (144 bytes per 256 weights):
//! - bytes[0..2]   — FP16 super-block scale `d` (little-endian)
//! - bytes[2..4]   — FP16 super-block minimum `dmin` (little-endian)
//! - bytes[4..16]  — 12 bytes encoding 8 sub-block scales + 8 sub-block mins,
//!                   6 bits each, packed (see `decode_scales_mins`)
//! - bytes[16..144] — 128 packed nibble bytes (256 × 4-bit unsigned values)
//!
//! Block structure: 8 sub-blocks of 32 weights each (4 groups of 2 sub-blocks).
//!
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 4-bit (0..15).
//! The nibble layout separates lo nibbles (first 32 weights of the group) from hi
//! nibbles (second 32 weights), unlike Q4_0 which interleaves them per byte.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K block: 2 (FP16 d) + 2 (FP16 dmin) + 12 (packed scales/mins) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 144;

/// AVX2+FMA accelerated Q4_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q4_KAvx2;

/// Decode the 6-bit packed scales and mins from the 12-byte header of a Q4_K block.
///
/// Returns `(scales[8], mins[8])` where each element is a 6-bit unsigned value.
/// Scale unpacking is kept scalar because the bit-manipulation pattern is irregular
/// and would not benefit from SIMD vectorization.
fn decode_scales_mins(scales_raw: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // Sub-blocks 0..3: straightforward 6-bit extraction from bytes 0..3 and 4..7.
    for j in 0..4 {
        sc[j] = scales_raw[j] & 0x3F;
        mn[j] = scales_raw[j + 4] & 0x3F;
    }

    // Sub-blocks 4..7: assembled from high bits of bytes 0..3/4..7 and bytes 8..11.
    for j in 4..8 {
        let lo_sc = scales_raw[j + 4] & 0x0F;
        let hi_sc = (scales_raw[j - 4] >> 6) & 0x03;
        sc[j] = lo_sc | (hi_sc << 4);

        let lo_mn = (scales_raw[j + 4] >> 4) & 0x0F;
        let hi_mn = (scales_raw[j] >> 6) & 0x03;
        mn[j] = lo_mn | (hi_mn << 4);
    }

    (sc, mn)
}

impl QuantKernel for Q4_KAvx2 {
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

        // SAFETY: block.len() >= 144 and output.len() >= 256 verified above.
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

    /// Fused Q4_K weight × Q8_0 activation GEMV using AVX2+FMA.
    ///
    /// Overrides the broken trait default. One Q4_K block (256 weights) =
    /// 8 Q8_0 activation blocks (32 each).
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
                fused_q4k_q8_0_row_avx2(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            out[row] += row_sum; // ACCUMULATE
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
        "Q4_K"
    }
}

/// Q8_0 block constants for the fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Compute fused Q4_K weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q4k_q8_0_row_avx2(
    row_data: &[u8],
    acts_q8: &[u8],
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

        // SAFETY: block.len() >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);

        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds.
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
            let qs = &block[16..144];

            let mut block_sum = 0.0f32;
            let mut is = 0usize;
            let mut qs_off = 0usize;

            for _group in 0..4 {
                // Sub-block `is` (lo nibbles).
                let a_idx_lo = blk * 8 + is;
                let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
                // SAFETY: acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES.
                let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
                let d_a_lo = f16_to_f32(a_block_lo);
                let q8_lo_ptr = a_block_lo.as_ptr().add(2) as *const __m128i;

                // Sub-block `is+1` (hi nibbles).
                let a_idx_hi = blk * 8 + is + 1;
                let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
                // SAFETY: same.
                let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
                let d_a_hi = f16_to_f32(a_block_hi);
                let q8_hi_ptr = a_block_hi.as_ptr().add(2) as *const __m128i;

                let da_lo = d * sc[is] as f32;
                let m_lo = dmin * mn[is] as f32;
                let da_hi = d * sc[is + 1] as f32;
                let m_hi = dmin * mn[is + 1] as f32;

                // Load 32 nibble bytes.
                // SAFETY: qs_off + 32 <= 128.
                let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

                let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo);
                let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo);
                let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
                let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

                // Load Q8_0 activations (32 i8 = two __m128i).
                // SAFETY: q8_lo_ptr / q8_hi_ptr each point to 32 valid i8 bytes.
                let qa_lo_0 = _mm_loadu_si128(q8_lo_ptr);
                let qa_lo_1 = _mm_loadu_si128(q8_lo_ptr.add(1));
                let qa_hi_0 = _mm_loadu_si128(q8_hi_ptr);
                let qa_hi_1 = _mm_loadu_si128(q8_hi_ptr.add(1));

                // Compute dot(lo_nibbles, q8_lo) and sum(q8_lo) using AVX2.
                // Extend u8 nibbles to i32, extend i8 activations to i32, multiply.
                let dot_lo = {
                    let mut acc = _mm256_setzero_si256();
                    // First 8 lo nibbles × first 8 i8 activations.
                    let w0 = _mm256_cvtepu8_epi32(lo_nibbles_0);
                    let a0 = _mm256_cvtepi8_epi32(qa_lo_0);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w0, a0));
                    let lo0_hi = _mm_srli_si128(lo_nibbles_0, 8);
                    let qa_lo0_hi = _mm_srli_si128(qa_lo_0, 8);
                    let w1 = _mm256_cvtepu8_epi32(lo0_hi);
                    let a1 = _mm256_cvtepi8_epi32(qa_lo0_hi);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w1, a1));
                    let w2 = _mm256_cvtepu8_epi32(lo_nibbles_1);
                    let a2 = _mm256_cvtepi8_epi32(qa_lo_1);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w2, a2));
                    let lo1_hi = _mm_srli_si128(lo_nibbles_1, 8);
                    let qa_lo1_hi = _mm_srli_si128(qa_lo_1, 8);
                    let w3 = _mm256_cvtepu8_epi32(lo1_hi);
                    let a3 = _mm256_cvtepi8_epi32(qa_lo1_hi);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w3, a3));
                    hsum_i32_avx256(acc)
                };

                let sum_a_lo = {
                    let mut acc = _mm256_setzero_si256();
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(qa_lo_0));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(_mm_srli_si128(qa_lo_0, 8)));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(qa_lo_1));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(_mm_srli_si128(qa_lo_1, 8)));
                    hsum_i32_avx256(acc)
                };

                block_sum += (da_lo * dot_lo as f32 - m_lo * sum_a_lo as f32) * d_a_lo;

                let dot_hi = {
                    let mut acc = _mm256_setzero_si256();
                    let w0 = _mm256_cvtepu8_epi32(hi_nibbles_0);
                    let a0 = _mm256_cvtepi8_epi32(qa_hi_0);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w0, a0));
                    let hi0_hi = _mm_srli_si128(hi_nibbles_0, 8);
                    let qa_hi0_hi = _mm_srli_si128(qa_hi_0, 8);
                    let w1 = _mm256_cvtepu8_epi32(hi0_hi);
                    let a1 = _mm256_cvtepi8_epi32(qa_hi0_hi);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w1, a1));
                    let w2 = _mm256_cvtepu8_epi32(hi_nibbles_1);
                    let a2 = _mm256_cvtepi8_epi32(qa_hi_1);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w2, a2));
                    let hi1_hi = _mm_srli_si128(hi_nibbles_1, 8);
                    let qa_hi1_hi = _mm_srli_si128(qa_hi_1, 8);
                    let w3 = _mm256_cvtepu8_epi32(hi1_hi);
                    let a3 = _mm256_cvtepi8_epi32(qa_hi1_hi);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w3, a3));
                    hsum_i32_avx256(acc)
                };

                let sum_a_hi = {
                    let mut acc = _mm256_setzero_si256();
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(qa_hi_0));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(_mm_srli_si128(qa_hi_0, 8)));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(qa_hi_1));
                    acc = _mm256_add_epi32(acc, _mm256_cvtepi8_epi32(_mm_srli_si128(qa_hi_1, 8)));
                    hsum_i32_avx256(acc)
                };

                block_sum += (da_hi * dot_hi as f32 - m_hi * sum_a_hi as f32) * d_a_hi;

                is += 2;
                qs_off += 32;
            }

            row_sum += block_sum;
        } else if remaining > 0 {
            // Scalar tail path.
            let qs = &block[16..144];
            let mut partial_sum = 0.0f32;
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for _group in 0..4 {
                let a_idx_lo = blk * 8 + is;
                let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
                // SAFETY: bounds validated.
                let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
                let d_a_lo = f16_to_f32(a_block_lo);
                let q8_lo = &a_block_lo[2..];

                let a_idx_hi = blk * 8 + is + 1;
                let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
                let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
                let d_a_hi = f16_to_f32(a_block_hi);
                let q8_hi = &a_block_hi[2..];

                let da_lo = d * sc[is] as f32;
                let m_lo = dmin * mn[is] as f32;
                let da_hi = d * sc[is + 1] as f32;
                let m_hi = dmin * mn[is + 1] as f32;

                let mut dot_lo = 0.0f32;
                let mut sum_a_lo = 0.0f32;
                for l in 0..32 {
                    let idx = w_off + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128.
                        let q_w = (*qs.get_unchecked(qs_off + l) & 0x0F) as f32;
                        let q_a = q8_lo[l] as i8 as f32;
                        dot_lo += q_w * q_a;
                        sum_a_lo += q_a;
                    }
                }
                partial_sum += (da_lo * dot_lo - m_lo * sum_a_lo) * d_a_lo;

                let mut dot_hi = 0.0f32;
                let mut sum_a_hi = 0.0f32;
                for l in 0..32 {
                    let idx = w_off + 32 + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128.
                        let q_w = ((*qs.get_unchecked(qs_off + l) >> 4) & 0x0F) as f32;
                        let q_a = q8_hi[l] as i8 as f32;
                        dot_hi += q_w * q_a;
                        sum_a_hi += q_a;
                    }
                }
                partial_sum += (da_hi * dot_hi - m_hi * sum_a_hi) * d_a_hi;

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            row_sum += partial_sum;
        }
        // remaining == 0: skip.
    }

    row_sum
}

/// Horizontal sum of a `__m256i` i32 register using AVX2.
///
/// # Safety
/// Caller must have `avx2` CPU feature.
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx256(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(sum128, 0b10_11_00_01);
    let sums = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b00_00_10_10);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 144-byte Q4_K block into 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 144`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read super-block FP16 scales.
    // SAFETY: block.len() >= 144 >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    // Decode the 12-byte packed scale/min header (scalar — irregular bit layout).
    let scales_raw = &block[4..16];
    let (sc, mn) = decode_scales_mins(scales_raw);

    // nibble data: 128 bytes starting at offset 16.
    let qs = &block[16..144];
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

    // Process 4 groups; each group has 32 lo-nibble weights (sub-block `is`) and
    // 32 hi-nibble weights (sub-block `is+1`).  The 32 nibble bytes for each group
    // start at qs_off and supply lo nibbles for the first 32 weights and hi nibbles
    // for the second 32 weights.
    let mut is = 0usize;
    let mut qs_off = 0usize;
    let mut out_off = 0usize;

    for _group in 0..4 {
        // Pre-compute scalar per-sub-block factors.
        let a_lo = d * sc[is] as f32; // d * scale for lo sub-block
        let b_lo = dmin * mn[is] as f32; // dmin * min for lo sub-block
        let a_hi = d * sc[is + 1] as f32; // d * scale for hi sub-block
        let b_hi = dmin * mn[is + 1] as f32; // dmin * min for hi sub-block

        let va_lo = _mm256_set1_ps(a_lo);
        let vb_lo = _mm256_set1_ps(b_lo);
        let va_hi = _mm256_set1_ps(a_hi);
        let vb_hi = _mm256_set1_ps(b_hi);

        // Load the 32 nibble bytes for this group (each byte encodes 2 nibbles:
        // lo nibble → lo sub-block weight, hi nibble → hi sub-block weight).
        // We split into two 16-byte loads.
        // SAFETY: qs_off + 32 <= 128 (4 groups × 32 bytes); block ensures validity.
        let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
        let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

        // Extract lo nibbles (weights 0..15 and 16..31 of the lo sub-block).
        let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo); // bytes 0..15 low nibbles
        let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo); // bytes 16..31 low nibbles

        // Extract hi nibbles (weights 0..15 and 16..31 of the hi sub-block).
        let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
        let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

        // --- Lo sub-block: 32 weights = a_lo * q - b_lo ---

        // First 8 lo nibbles → f32.
        // SAFETY: _mm256_cvtepu8_epi32 reads 8 bytes from the 128-bit source.
        let q0_i32 = _mm256_cvtepu8_epi32(lo_nibbles_0);
        let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
        // w = a_lo * q - b_lo  using FMA: fmadd(a_lo, q, -b_lo)
        let w0 = _mm256_fmsub_ps(va_lo, q0_f32, vb_lo);

        // Next 8 lo nibbles (bytes 8..15 of raw_lo).
        let lo_nibbles_0_hi = _mm_srli_si128(lo_nibbles_0, 8);
        let q1_i32 = _mm256_cvtepu8_epi32(lo_nibbles_0_hi);
        let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
        let w1 = _mm256_fmsub_ps(va_lo, q1_f32, vb_lo);

        // First 8 lo nibbles from second half (bytes 0..7 of raw_hi lo nibbles).
        let q2_i32 = _mm256_cvtepu8_epi32(lo_nibbles_1);
        let q2_f32 = _mm256_cvtepi32_ps(q2_i32);
        let w2 = _mm256_fmsub_ps(va_lo, q2_f32, vb_lo);

        // Last 8 lo nibbles (bytes 8..15 of raw_hi lo nibbles).
        let lo_nibbles_1_hi = _mm_srli_si128(lo_nibbles_1, 8);
        let q3_i32 = _mm256_cvtepu8_epi32(lo_nibbles_1_hi);
        let q3_f32 = _mm256_cvtepi32_ps(q3_i32);
        let w3 = _mm256_fmsub_ps(va_lo, q3_f32, vb_lo);

        // Store 32 lo-sub-block weights.
        // SAFETY: out_off + 32 <= 256; output.len() >= 256.
        let ptr = output.as_mut_ptr().add(out_off);
        _mm256_storeu_ps(ptr, w0);
        _mm256_storeu_ps(ptr.add(8), w1);
        _mm256_storeu_ps(ptr.add(16), w2);
        _mm256_storeu_ps(ptr.add(24), w3);

        // --- Hi sub-block: 32 weights = a_hi * q - b_hi ---

        let q4_i32 = _mm256_cvtepu8_epi32(hi_nibbles_0);
        let q4_f32 = _mm256_cvtepi32_ps(q4_i32);
        let w4 = _mm256_fmsub_ps(va_hi, q4_f32, vb_hi);

        let hi_nibbles_0_hi = _mm_srli_si128(hi_nibbles_0, 8);
        let q5_i32 = _mm256_cvtepu8_epi32(hi_nibbles_0_hi);
        let q5_f32 = _mm256_cvtepi32_ps(q5_i32);
        let w5 = _mm256_fmsub_ps(va_hi, q5_f32, vb_hi);

        let q6_i32 = _mm256_cvtepu8_epi32(hi_nibbles_1);
        let q6_f32 = _mm256_cvtepi32_ps(q6_i32);
        let w6 = _mm256_fmsub_ps(va_hi, q6_f32, vb_hi);

        let hi_nibbles_1_hi = _mm_srli_si128(hi_nibbles_1, 8);
        let q7_i32 = _mm256_cvtepu8_epi32(hi_nibbles_1_hi);
        let q7_f32 = _mm256_cvtepi32_ps(q7_i32);
        let w7 = _mm256_fmsub_ps(va_hi, q7_f32, vb_hi);

        // Store 32 hi-sub-block weights.
        // SAFETY: out_off + 32 + 32 <= 256; output.len() >= 256.
        let ptr2 = output.as_mut_ptr().add(out_off + 32);
        _mm256_storeu_ps(ptr2, w4);
        _mm256_storeu_ps(ptr2.add(8), w5);
        _mm256_storeu_ps(ptr2.add(16), w6);
        _mm256_storeu_ps(ptr2.add(24), w7);

        is += 2;
        qs_off += 32;
        out_off += 64;
    }
}

/// Compute the dot product of one row of a Q4_K matrix with an FP32 vector.
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

        // Read FP16 super-block scales.
        // SAFETY: block.len() == 144 >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);

        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
            let qs = &block[16..144];

            let mut block_acc = _mm256_setzero_ps();
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for _group in 0..4 {
                let a_lo = d * sc[is] as f32;
                let b_lo = dmin * mn[is] as f32;
                let a_hi = d * sc[is + 1] as f32;
                let b_hi = dmin * mn[is + 1] as f32;

                let va_lo = _mm256_set1_ps(a_lo);
                let vb_lo = _mm256_set1_ps(b_lo);
                let va_hi = _mm256_set1_ps(a_hi);
                let vb_hi = _mm256_set1_ps(b_hi);

                // Load 32 nibble bytes for this group.
                // SAFETY: qs_off + 32 <= 128; qs.len() == 128.
                let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

                let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo);
                let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo);
                let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
                let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

                // SAFETY: w_off + 32 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let inp_ptr_lo = input.as_ptr().add(w_off);
                let inp_ptr_hi = input.as_ptr().add(w_off + 32);

                // Lo sub-block: 32 weights at inp_ptr_lo.
                let q0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_nibbles_0));
                let i0 = _mm256_loadu_ps(inp_ptr_lo);
                let w0 = _mm256_fmsub_ps(va_lo, q0, vb_lo); // a_lo*q - b_lo
                block_acc = _mm256_fmadd_ps(w0, i0, block_acc);

                let lo_nibbles_0_hi = _mm_srli_si128(lo_nibbles_0, 8);
                let q1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_nibbles_0_hi));
                let i1 = _mm256_loadu_ps(inp_ptr_lo.add(8));
                let w1 = _mm256_fmsub_ps(va_lo, q1, vb_lo);
                block_acc = _mm256_fmadd_ps(w1, i1, block_acc);

                let q2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_nibbles_1));
                let i2 = _mm256_loadu_ps(inp_ptr_lo.add(16));
                let w2 = _mm256_fmsub_ps(va_lo, q2, vb_lo);
                block_acc = _mm256_fmadd_ps(w2, i2, block_acc);

                let lo_nibbles_1_hi = _mm_srli_si128(lo_nibbles_1, 8);
                let q3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_nibbles_1_hi));
                let i3 = _mm256_loadu_ps(inp_ptr_lo.add(24));
                let w3 = _mm256_fmsub_ps(va_lo, q3, vb_lo);
                block_acc = _mm256_fmadd_ps(w3, i3, block_acc);

                // Hi sub-block: 32 weights at inp_ptr_hi.
                let q4 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_nibbles_0));
                let i4 = _mm256_loadu_ps(inp_ptr_hi);
                let w4 = _mm256_fmsub_ps(va_hi, q4, vb_hi);
                block_acc = _mm256_fmadd_ps(w4, i4, block_acc);

                let hi_nibbles_0_hi = _mm_srli_si128(hi_nibbles_0, 8);
                let q5 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_nibbles_0_hi));
                let i5 = _mm256_loadu_ps(inp_ptr_hi.add(8));
                let w5 = _mm256_fmsub_ps(va_hi, q5, vb_hi);
                block_acc = _mm256_fmadd_ps(w5, i5, block_acc);

                let q6 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_nibbles_1));
                let i6 = _mm256_loadu_ps(inp_ptr_hi.add(16));
                let w6 = _mm256_fmsub_ps(va_hi, q6, vb_hi);
                block_acc = _mm256_fmadd_ps(w6, i6, block_acc);

                let hi_nibbles_1_hi = _mm_srli_si128(hi_nibbles_1, 8);
                let q7 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_nibbles_1_hi));
                let i7 = _mm256_loadu_ps(inp_ptr_hi.add(24));
                let w7 = _mm256_fmsub_ps(va_hi, q7, vb_hi);
                block_acc = _mm256_fmadd_ps(w7, i7, block_acc);

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            row_sum += hsum_f32_avx(block_acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let qs = &block[16..144];
            let mut partial_sum = 0.0f32;
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for _group in 0..4 {
                let d1 = d * sc[is] as f32;
                let m1 = dmin * mn[is] as f32;
                let d2 = d * sc[is + 1] as f32;
                let m2 = dmin * mn[is + 1] as f32;

                // Lo nibbles → first 32 weights of this group.
                for l in 0..32 {
                    let idx = w_off + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128 because qs_off < 128 and l < 32.
                        let q = (*qs.get_unchecked(qs_off + l) & 0x0F) as f32;
                        partial_sum += (d1 * q - m1) * input[idx];
                    }
                }
                // Hi nibbles → next 32 weights.
                for l in 0..32 {
                    let idx = w_off + 32 + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128.
                        let q = ((*qs.get_unchecked(qs_off + l) >> 4) & 0x0F) as f32;
                        partial_sum += (d2 * q - m2) * input[idx];
                    }
                }

                is += 2;
                qs_off += 32;
                w_off += 64;
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
    use crate::reference::q4_k::Q4KRef;

    fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q4K)
    }

    #[test]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[0] = 5;
        scales[1] = 3;
        scales[2] = 7;
        scales[3] = 2;
        scales[4] = 4; // mins 0..3
        scales[5] = 6;
        scales[6] = 1;
        scales[7] = 3;
        scales[8] = 9; // sub-blocks 4..7 lo nibbles
        scales[9] = 11;
        scales[10] = 13;
        scales[11] = 15;

        // Alternating nibble pattern: lo=5, hi=10 → byte 0xA5
        let qs = [0xA5u8; 128];

        let block = make_q4_k_block(0.5, 0.1, &scales, &qs);

        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q4_KAvx2.dequant_block(&block, &mut out_avx2).unwrap();
        Q4KRef.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1); // sub-blocks 0..3 scale=1
        scales[8..12].fill(1); // sub-blocks 4..7 scale=1 (lo nibble of bytes 8..11)

        // All nibbles = 8: lo=8, hi=8 → byte 0x88
        let qs = [0x88u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KAvx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4KRef.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

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
        // 200 columns — one partial block (200 < 256).
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x11u8; 128]; // all weights = 1 (lo=1, hi=1)

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KAvx2.gemv(&tensor_avx2, &input, &mut out_avx2).unwrap();
        Q4KRef.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_uniform_all_ones() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // d=1.0, dmin=0.0, all scales=1, all nibbles=1 → weight=1.0, sum=256.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x11u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor = make_tensor(block, 256);
        let input = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 1];

        Q4_KAvx2.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - 256.0).abs() < 1.0,
            "expected ~256.0, got {}",
            out[0]
        );
    }

    // ── matvec_q8_fused Q4_K ─────────────────────────────────────────────

    fn make_q8_0_block(scale: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn avx2_fused_matches_reference_q4k() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n_cols = 256usize;
        let mut scales = [0u8; 12];
        scales[0] = 5;
        scales[1] = 3;
        scales[2] = 7;
        scales[3] = 2;
        scales[4] = 4;
        scales[5] = 6;
        scales[6] = 1;
        scales[7] = 3;
        scales[8] = 9;
        scales[9] = 11;
        scales[10] = 13;
        scales[11] = 15;
        let qs_nibbles = [0xA5u8; 128];
        let d_w = 0.5f32;
        let dmin_w = 0.1f32;
        let w_block = make_q4_k_block(d_w, dmin_w, &scales, &qs_nibbles);

        let d_a = 0.25f32;
        let q8_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16,
        ];

        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out_avx2, 1, n_cols)
            .expect("avx2 fused q4k");
        Q4KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, n_cols)
            .expect("ref fused q4k");

        let err = (out_avx2[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "avx2_fused_matches_reference_q4k: avx2={} ref={} err={}",
            out_avx2[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn avx2_fused_q4k_accumulates() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let w_block = make_q4_k_block(0.0, 0.0, &[0u8; 12], &[0u8; 128]);
        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(1.0, &[0i8; 32]));
        }

        let mut out = vec![42.0f32; 1];
        Q4_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out, 1, 256)
            .expect("avx2 fused q4k accumulate");
        assert!(
            (out[0] - 42.0).abs() < 1e-5,
            "accumulation broken: got {}",
            out[0]
        );
    }

    #[test]
    fn avx2_fused_q4k_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n_rows = 4usize;
        let n_cols = 256usize;
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x55u8; 128];
        let d_a = 0.5f32;
        let q8_vals: [i8; 32] = [
            2, -1, 4, -3, 6, -5, 8, -7, 1, -2, 3, -4, 5, -6, 7, -8, -2, 1, -4, 3, -6, 5, -8, 7, -1,
            2, -3, 4, -5, 6, -7, 8,
        ];
        let scales_w = [0.5f32, 1.0f32, 0.25f32, 0.1f32];

        let mut weights: Vec<u8> = Vec::new();
        for &s in &scales_w {
            weights.extend_from_slice(&make_q4_k_block(s, 0.0, &scales, &qs));
        }

        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        let mut out_avx2 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_KAvx2
            .matvec_q8_fused(&weights, &acts, &mut out_avx2, n_rows, n_cols)
            .expect("avx2 fused q4k multi-row");
        Q4KRef
            .matvec_q8_fused(&weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused q4k multi-row");

        for i in 0..n_rows {
            let err = (out_avx2[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "row {i}: avx2={} ref={} err={}",
                out_avx2[i],
                out_ref[i],
                err
            );
        }
    }
}
