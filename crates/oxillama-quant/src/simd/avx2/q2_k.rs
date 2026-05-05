//! AVX2+FMA accelerated Q2_K quantization kernel.
//!
//! Q2_K block layout (84 bytes per 256 weights):
//! - bytes\[0..16\]  — 16 scale bytes (lo 4 bits = scale, hi 4 bits = min)
//! - bytes\[16..80\] — 64 qs bytes (256 × 2-bit packed, 4 per byte via shifts 0,2,4,6)
//! - bytes\[80..82\] — FP16 super-block scale `d` (little-endian)
//! - bytes\[82..84\] — FP16 super-block minimum `dmin` (little-endian)
//!
//! NOTE: In Q2_K, d/dmin come AFTER scales and qs in memory.
//!
//! 16 sub-blocks of 16 weights each (2 groups of 128, each group processes
//! the same 32 qs bytes with 4 different shift amounts).
//!
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 2-bit (0..3).
//!
//! AVX2 strategy: for each shift, extract 2-bit values from pre-loaded qs bytes
//! via `_mm_srli_epi16` + AND 0x03, widen via `_mm256_cvtepu8_epi32`, convert
//! to f32, then apply `_mm256_fmsub_ps` for `d*scale*q - dmin*min`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q2_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q2_K block: 16 (scales) + 64 (qs) + 2 (FP16 d) + 2 (FP16 dmin).
pub const BLOCK_BYTES: usize = 84;

/// AVX2+FMA accelerated Q2_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q2_KAvx2;

/// Extract 2-bit values from 16 packed bytes using the given bit-shift.
///
/// Each source byte contains four 2-bit values at positions 0..1, 2..3, 4..5,
/// 6..7.  The `shift` parameter (0, 2, 4, or 6) selects which 2-bit field to
/// extract.  `_mm_srli_epi16` shifts 16-bit lanes but the subsequent AND with
/// 0x03 discards any cross-byte contamination (because the leak is always a
/// multiple of 4, which vanishes under the 0x03 mask).
///
/// # Safety
/// Requires `avx2` CPU feature.  `shift` must be one of 0, 2, 4, 6.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_2bit_16(raw: __m128i, shift: u32, mask: __m128i) -> __m128i {
    // SAFETY: each branch uses a compile-time const generic for _mm_srli_epi16.
    // The runtime match selects the correct shift amount; cross-byte leakage from
    // the 16-bit shift is harmless because it is always a multiple of 4, eliminated
    // by the AND with 0x03.
    let shifted = match shift {
        0 => raw,
        2 => _mm_srli_epi16::<2>(raw),
        4 => _mm_srli_epi16::<4>(raw),
        _ => _mm_srli_epi16::<6>(raw),
    };
    _mm_and_si128(shifted, mask)
}

impl QuantKernel for Q2_KAvx2 {
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

        // SAFETY: block.len() >= 84 and output.len() >= 256 verified above.
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

    /// Fused Q2_K weight × Q8_0 activation GEMV using AVX2+FMA.
    ///
    /// Each Q2_K super-block (256 weights, 84 bytes) maps to 8 Q8_0 activation blocks
    /// (32 weights each, 34 bytes each).  Sub-block `sb` at weight position `sb*16`
    /// uses Q8_0 block `blk*8 + sb*16/32 = blk*8 + sb/2`.
    ///
    /// The fused computation: for each sub-block, compute
    ///   Σ (dl * q - ml) * d_a * q8_a  =  dl * Σ(q * q8_a) * d_a  -  ml * Σ(q8_a) * d_a
    /// avoiding any intermediate f32 scratch buffer.
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
        // One Q2_K block → 8 Q8_0 blocks (256 weights / 32 per Q8_0 block)
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
                fused_q2k_q8_0_row_avx2(
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
        "Q2_K"
    }
}

/// Q8_0 block bytes for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Fused Q2_K weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// Q2_K layout: 16 scales (nibble-packed), 64 qs (2-bit × 4 per byte), then d/dmin.
/// 256 weights split into 16 sub-blocks of 16 weights.  Sub-block `s` (0..16) maps
/// to Q8_0 block at index `blk*8 + s/2` (two sub-blocks share one Q8_0 block).
///
/// For sub-block `s` with Q8_0 block `q8_blk` at lane `q8_lane` (0 or 16):
///   contribution = (dl * Σ(q2_i * q8_i) - ml * Σ(q8_i)) * d_a
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q2k_q8_0_row_avx2(
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

        let scales = &block[0..16];
        let qs = &block[16..80];
        // SAFETY: block.len() >= 84.
        let d = f16_to_f32(&block[80..]);
        let dmin = f16_to_f32(&block[82..]);

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        if cols_in_block < BLOCK_SIZE {
            // Scalar tail path — partial block.
            let mut is = 0usize;
            let mut col_off = 0usize;

            for _group in 0..2 {
                let qs_base = _group * 32;
                for shift in (0u32..8).step_by(2) {
                    // Sub-block A: columns col_off..col_off+16
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8 bounds checked at caller level.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_vals = &a_block[2..];
                        let mut dot = 0.0f32;
                        let mut sum_a = 0.0f32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q2 = (qs[qs_base + l] >> shift) & 3;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                dot += q2 as f32 * q_a;
                                sum_a += q_a;
                            }
                        }
                        row_sum += (dl * dot - ml * sum_a) * d_a;
                    }
                    col_off += 16;

                    // Sub-block B: columns col_off..col_off+16
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8 bounds checked at caller level.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_vals = &a_block[2..];
                        let mut dot = 0.0f32;
                        let mut sum_a = 0.0f32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q2 = (qs[qs_base + 16 + l] >> shift) & 3;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                dot += q2 as f32 * q_a;
                                sum_a += q_a;
                            }
                        }
                        row_sum += (dl * dot - ml * sum_a) * d_a;
                    }
                    col_off += 16;
                }
            }
        } else {
            // Fast path: all 256 weights in bounds.
            // 256 weights = 8 Q8_0 blocks × 32 weights.
            // Q2_K organizes as 2 groups × 4 shifts × 2 sub-blocks (A+B) × 16 weights.
            // Sub-block index `sb` = is/2 (0..8 pairs per block → 16 sub-blocks total).
            // Q8_0 block index: sb_flat/2 where sb_flat = 0..16.
            // We track sb_flat (= col_off/16) to map to Q8_0 blocks.
            let mask_2bit = _mm_set1_epi8(0x03);

            let mut is = 0usize;
            let mut col_off = 0usize; // 0..256, steps of 16

            for group in 0..2usize {
                let qs_base = group * 32;
                // Pre-load 32 qs bytes for this group (two 16-byte halves).
                // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
                let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
                let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

                for &shift in &[0u32, 2, 4, 6] {
                    // Sub-block A (16 weights): Q8_0 block at index col_off/32, lane col_off%32
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const __m128i;

                        // SAFETY: extract_2bit_16 requires avx2 and shift in {0,2,4,6}.
                        let q2_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                        // SAFETY: q8_ptr points to 16 valid i8 bytes.
                        let qa_bytes = _mm_loadu_si128(q8_ptr);

                        // dot = Σ(q2_i * q8_i); sum_a = Σ(q8_i)
                        // Widen q2 (u8→i32), q8 (i8→i32), multiply, sum.
                        let q2_lo = _mm256_cvtepu8_epi32(q2_bytes);
                        let q2_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q2_bytes, 8));
                        let qa_lo = _mm256_cvtepi8_epi32(qa_bytes);
                        let qa_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(qa_bytes, 8));

                        let dot_acc = _mm256_add_epi32(
                            _mm256_mullo_epi32(q2_lo, qa_lo),
                            _mm256_mullo_epi32(q2_hi, qa_hi),
                        );
                        let sum_a_acc = _mm256_add_epi32(qa_lo, qa_hi);

                        let dot = hsum_i32_avx2_q2k(dot_acc) as f32;
                        let sum_a = hsum_i32_avx2_q2k(sum_a_acc) as f32;
                        row_sum += (dl * dot - ml * sum_a) * d_a;
                        col_off += 16;
                    }

                    // Sub-block B (16 weights): Q8_0 block at index col_off/32, lane col_off%32
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(a_block);
                        let q8_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const __m128i;

                        let q2_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                        // SAFETY: q8_ptr points to 16 valid i8 bytes.
                        let qa_bytes = _mm_loadu_si128(q8_ptr);

                        let q2_lo = _mm256_cvtepu8_epi32(q2_bytes);
                        let q2_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(q2_bytes, 8));
                        let qa_lo = _mm256_cvtepi8_epi32(qa_bytes);
                        let qa_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(qa_bytes, 8));

                        let dot_acc = _mm256_add_epi32(
                            _mm256_mullo_epi32(q2_lo, qa_lo),
                            _mm256_mullo_epi32(q2_hi, qa_hi),
                        );
                        let sum_a_acc = _mm256_add_epi32(qa_lo, qa_hi);

                        let dot = hsum_i32_avx2_q2k(dot_acc) as f32;
                        let sum_a = hsum_i32_avx2_q2k(sum_a_acc) as f32;
                        row_sum += (dl * dot - ml * sum_a) * d_a;
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
unsafe fn hsum_i32_avx2_q2k(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(s, 0b10_11_00_01);
    let s2 = _mm_add_epi32(s, shuf);
    let shuf2 = _mm_shuffle_epi32(s2, 0b00_00_10_10);
    let s3 = _mm_add_epi32(s2, shuf2);
    _mm_cvtsi128_si32(s3)
}

/// Dequantize one 84-byte Q2_K block into 256 FP32 values using AVX2.
///
/// Processes 2 groups of 128 weights.  Within each group, the same 32 qs
/// bytes are re-used with 4 different shift amounts (0, 2, 4, 6) to extract
/// all four 2-bit fields per byte.
///
/// # Safety
/// - `block.len() >= 84`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let scales = &block[0..16];
    let qs = &block[16..80];

    // SAFETY: block.len() >= 84, so byte offsets 80..84 are valid.
    let d = f16_to_f32(&block[80..]);
    let dmin = f16_to_f32(&block[82..]);

    let mask_2bit = _mm_set1_epi8(0x03);

    let mut is = 0usize;
    let mut out_off = 0usize;

    for group in 0..2usize {
        let qs_base = group * 32;

        // Pre-load the 32 qs bytes for this group (two 16-byte halves).
        // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
        let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
        let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

        for &shift in &[0u32, 2, 4, 6] {
            // --- Sub-block A: 16 weights from qs[qs_base..qs_base+16] ---
            let sc_byte_a = scales[is];
            is += 1;
            let dl_a = d * (sc_byte_a & 0x0F) as f32;
            let ml_a = dmin * (sc_byte_a >> 4) as f32;
            let vdl_a = _mm256_set1_ps(dl_a);
            let vml_a = _mm256_set1_ps(ml_a);

            // SAFETY: extract_2bit_16 requires avx2 and shift in {0,2,4,6}.
            let q_bytes_a = extract_2bit_16(raw_a, shift, mask_2bit);

            // First 8 weights: widen bytes 0..7 to i32, convert to f32.
            // SAFETY: _mm256_cvtepu8_epi32 reads from the low 8 bytes of q_bytes_a.
            let q0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_a));
            let w0 = _mm256_fmsub_ps(vdl_a, q0_f32, vml_a); // dl*q - ml

            // Next 8 weights: shift the 128-bit register right by 8 bytes.
            let q_bytes_a_hi = _mm_srli_si128(q_bytes_a, 8);
            let q1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_a_hi));
            let w1 = _mm256_fmsub_ps(vdl_a, q1_f32, vml_a);

            // SAFETY: out_off + 16 <= 256; output.len() >= 256.
            let ptr_a = output.as_mut_ptr().add(out_off);
            _mm256_storeu_ps(ptr_a, w0);
            _mm256_storeu_ps(ptr_a.add(8), w1);
            out_off += 16;

            // --- Sub-block B: 16 weights from qs[qs_base+16..qs_base+32] ---
            let sc_byte_b = scales[is];
            is += 1;
            let dl_b = d * (sc_byte_b & 0x0F) as f32;
            let ml_b = dmin * (sc_byte_b >> 4) as f32;
            let vdl_b = _mm256_set1_ps(dl_b);
            let vml_b = _mm256_set1_ps(ml_b);

            let q_bytes_b = extract_2bit_16(raw_b, shift, mask_2bit);

            let q2_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_b));
            let w2 = _mm256_fmsub_ps(vdl_b, q2_f32, vml_b);

            let q_bytes_b_hi = _mm_srli_si128(q_bytes_b, 8);
            let q3_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_b_hi));
            let w3 = _mm256_fmsub_ps(vdl_b, q3_f32, vml_b);

            // SAFETY: out_off + 16 <= 256; output.len() >= 256.
            let ptr_b = output.as_mut_ptr().add(out_off);
            _mm256_storeu_ps(ptr_b, w2);
            _mm256_storeu_ps(ptr_b.add(8), w3);
            out_off += 16;
        }
    }
}

/// Compute the dot product of one row of a Q2_K matrix with an FP32 vector.
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

        let scales = &block[0..16];
        let qs = &block[16..80];

        // SAFETY: block.len() == 84 >= 84.
        let d = f16_to_f32(&block[80..]);
        let dmin = f16_to_f32(&block[82..]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.
            let mask_2bit = _mm_set1_epi8(0x03);
            let mut block_acc = _mm256_setzero_ps();
            let mut is = 0usize;
            let mut w_off = input_offset;

            for group in 0..2usize {
                let qs_base = group * 32;

                // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
                let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
                let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

                for &shift in &[0u32, 2, 4, 6] {
                    // --- Sub-block A ---
                    let sc_byte_a = scales[is];
                    is += 1;
                    let dl_a = d * (sc_byte_a & 0x0F) as f32;
                    let ml_a = dmin * (sc_byte_a >> 4) as f32;
                    let vdl_a = _mm256_set1_ps(dl_a);
                    let vml_a = _mm256_set1_ps(ml_a);

                    let q_bytes_a = extract_2bit_16(raw_a, shift, mask_2bit);

                    // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                    let inp_ptr_a = input.as_ptr().add(w_off);

                    // First 8
                    let q0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_a));
                    let w0 = _mm256_fmsub_ps(vdl_a, q0, vml_a);
                    let i0 = _mm256_loadu_ps(inp_ptr_a);
                    block_acc = _mm256_fmadd_ps(w0, i0, block_acc);

                    // Next 8
                    let q_bytes_a_hi = _mm_srli_si128(q_bytes_a, 8);
                    let q1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_a_hi));
                    let w1 = _mm256_fmsub_ps(vdl_a, q1, vml_a);
                    let i1 = _mm256_loadu_ps(inp_ptr_a.add(8));
                    block_acc = _mm256_fmadd_ps(w1, i1, block_acc);

                    w_off += 16;

                    // --- Sub-block B ---
                    let sc_byte_b = scales[is];
                    is += 1;
                    let dl_b = d * (sc_byte_b & 0x0F) as f32;
                    let ml_b = dmin * (sc_byte_b >> 4) as f32;
                    let vdl_b = _mm256_set1_ps(dl_b);
                    let vml_b = _mm256_set1_ps(ml_b);

                    let q_bytes_b = extract_2bit_16(raw_b, shift, mask_2bit);

                    // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                    let inp_ptr_b = input.as_ptr().add(w_off);

                    let q2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_b));
                    let w2 = _mm256_fmsub_ps(vdl_b, q2, vml_b);
                    let i2 = _mm256_loadu_ps(inp_ptr_b);
                    block_acc = _mm256_fmadd_ps(w2, i2, block_acc);

                    let q_bytes_b_hi = _mm_srli_si128(q_bytes_b, 8);
                    let q3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_bytes_b_hi));
                    let w3 = _mm256_fmsub_ps(vdl_b, q3, vml_b);
                    let i3 = _mm256_loadu_ps(inp_ptr_b.add(8));
                    block_acc = _mm256_fmadd_ps(w3, i3, block_acc);

                    w_off += 16;
                }
            }

            row_sum += hsum_f32_avx(block_acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut partial_sum = 0.0f32;
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut in_off = input_offset;

            for _group in 0..2 {
                for shift in (0u32..8).step_by(2) {
                    // Sub-block A: qs[qs_off..qs_off+16]
                    let sc_byte = scales[is];
                    let dl = d * (sc_byte & 0x0F) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;

                    for l in 0..16 {
                        let idx = in_off + l;
                        if idx < n_cols {
                            // SAFETY: qs_off + l < 64; shift in {0,2,4,6}.
                            let q = (*qs.get_unchecked(qs_off + l) >> shift) & 3;
                            partial_sum += (dl * q as f32 - ml) * input[idx];
                        }
                    }
                    in_off += 16;

                    // Sub-block B: qs[qs_off+16..qs_off+32]
                    let sc_byte = scales[is];
                    let dl = d * (sc_byte & 0x0F) as f32;
                    let ml = dmin * (sc_byte >> 4) as f32;
                    is += 1;

                    for l in 0..16 {
                        let idx = in_off + l;
                        if idx < n_cols {
                            // SAFETY: qs_off + 16 + l < 64.
                            let q = (*qs.get_unchecked(qs_off + 16 + l) >> shift) & 3;
                            partial_sum += (dl * q as f32 - ml) * input[idx];
                        }
                    }
                    in_off += 16;
                }
                qs_off += 32;
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
    use crate::reference::q2_k::Q2KRef;

    fn make_q2k_block(d: f32, dmin: f32, scales: &[u8; 16], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q2K)
    }

    #[test]
    fn test_dequant_matches_reference_zeros() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_q2k_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q2KRef
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
    fn test_dequant_matches_reference_uniform() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // All scales = 0x01 (scale=1, min=0), all qs = 0xFF (all 2-bit = 3)
        let block = make_q2k_block(1.0, 0.0, &[0x01; 16], &[0xFF; 64]);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q2KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [uniform] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_with_min() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // d=2.0, dmin=1.0, scales=0x11 (scale=1, min=1), qs=0x00 (all q=0)
        // Weight = 2.0 * 1 * 0 - 1.0 * 1 = -1.0
        let block = make_q2k_block(2.0, 1.0, &[0x11; 16], &[0x00; 64]);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q2KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [with_min] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_varied() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (0x21 + i as u8) & 0xFF;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }

        let block = make_q2k_block(0.5, 0.25, &scales, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q2KRef
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
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (0x21 + i as u8) & 0xFF;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }
        let block = make_q2k_block(0.5, 0.25, &scales, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv");
        Q2KRef
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
        let scales = [0x11u8; 16];
        let qs = [0xAAu8; 64];

        let block = make_q2k_block(1.0, 0.5, &scales, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv partial");
        Q2KRef
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
    fn test_gemv_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let block = make_q2k_block(0.75, 0.3, &scales, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv varied");
        Q2KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv varied");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "varied gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_buffer_too_small_block() {
        let block = vec![0u8; 10]; // too small
        let mut output = vec![0.0f32; 256];
        assert!(Q2_KAvx2.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_buffer_too_small_output() {
        let block = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; 10]; // too small
        assert!(Q2_KAvx2.dequant_block(&block, &mut output).is_err());
    }

    // ── matvec_q8_fused Q2_K ─────────────────────────────────────────────

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
    fn test_q2k_avx2_fused_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // One Q2_K super-block (256 weights) needs 8 Q8_0 activation blocks.
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 5) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 3) & 0xFF) as u8;
        }

        let w_block = make_q2k_block(0.5, 0.25, &scales, &qs);
        let act_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 0, 1, -1, 2, -2, 3, -3,
            4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let acts = make_q8_acts(8, 0.1, &act_vals);

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out_avx2, 1, 256)
            .expect("avx2 fused q2k single block");
        Q2KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, 256)
            .expect("ref fused q2k single block");

        let err = (out_avx2[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "q2k_avx2_fused_matches_reference: avx2={} ref={} err={}",
            out_avx2[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q2k_avx2_fused_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 3 rows × 512 cols = 2 Q2_K super-blocks per row → 16 Q8_0 act blocks.
        let n_rows = 3usize;
        let n_cols = 512usize;
        let blocks_per_row = 2usize;
        let q8_blocks_per_row = blocks_per_row * 8; // 16

        let mut all_weights = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut scales = [0u8; 16];
                for (i, s) in scales.iter_mut().enumerate() {
                    *s = ((r * 17 + b * 11 + i * 7 + 5) & 0xFF) as u8;
                }
                let mut qs = [0u8; 64];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((r * 5 + b * 23 + i * 11) & 0xFF) as u8;
                }
                all_weights.extend(make_q2k_block(
                    0.5 + r as f32 * 0.1,
                    0.1 + b as f32 * 0.05,
                    &scales,
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

        Q2_KAvx2
            .matvec_q8_fused(&all_weights, &acts, &mut out_avx2, n_rows, n_cols)
            .expect("avx2 fused q2k multi-row");
        Q2KRef
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused q2k multi-row");

        for i in 0..n_rows {
            let err = (out_avx2[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "q2k fused multi-row row {i}: avx2={} ref={} err={}",
                out_avx2[i],
                out_ref[i],
                err
            );
        }
    }
}
