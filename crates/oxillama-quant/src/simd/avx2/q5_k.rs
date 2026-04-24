//! AVX2+FMA accelerated Q5_K quantization kernel.
//!
//! Q5_K block layout (176 bytes per 256 weights):
//! - bytes[0..2]   — FP16 super-block scale `d` (little-endian)
//! - bytes[2..4]   — FP16 super-block minimum `dmin` (little-endian)
//! - bytes[4..16]  — 12 bytes encoding 8 sub-block scales + 8 sub-block mins,
//!                   6 bits each, packed (same packing as Q4_K)
//! - bytes[16..48] — 32 bytes `qh` — the high (5th) bit of each 5-bit quant,
//!                   bit `j` of byte `qh[l]` is the high bit of weight
//!                   (group * 32 + l) in the lo sub-block (j < 4) or
//!                   (group * 32 + l) in the hi sub-block (j >= 4).
//! - bytes[48..176] — 128 packed nibble bytes (256 × 4-bit unsigned lo values)
//!
//! Block structure: 8 sub-blocks of 32 weights each (4 groups of 2 sub-blocks).
//!
//! Weight formula: `w = d * scale_i * q5 - dmin * min_i`
//! where `q5 = nibble | (high_bit << 4)` (range 0..31).
//!
//! The high-bit extraction uses a compare-equal pattern (not a shift) to avoid
//! cross-byte contamination from `_mm_srli_epi16` when the bit position changes
//! per group.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q5_K block: 2+2+12 (header) + 32 (qh) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 176;

/// AVX2+FMA accelerated Q5_K kernel.
///
/// Requires `avx2` and `fma` CPU features.  The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q5_KAvx2;

/// Decode the 6-bit packed scales and mins from the 12-byte header.
///
/// Identical packing to Q4_K — returns `(scales[8], mins[8])`.
fn decode_scales_mins(scales_raw: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for j in 0..4 {
        sc[j] = scales_raw[j] & 0x3F;
        mn[j] = scales_raw[j + 4] & 0x3F;
    }

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

impl QuantKernel for Q5_KAvx2 {
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

        // SAFETY: block.len() >= 176 and output.len() >= 256 verified above.
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
        "Q5_K"
    }

    /// Fused Q5_K weight × Q8_0 activation GEMV.
    ///
    /// Each Q5_K super-block (256 weights, 176 bytes) maps to 8 Q8_0 activation blocks.
    /// Sub-block `is` (0..8) of weight block `blk` uses Q8_0 at index `blk*8 + is`.
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
                fused_q5_k_q8_0_row_avx2(
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

/// Fused Q5_K weight × Q8_0 activation dot product for one row using AVX2+FMA.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_q5_k_q8_0_row_avx2(
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

        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);
        let (sc, mn) = decode_scales_mins(&block[4..16]);
        let qh = &block[16..48]; // 32 high-bit bytes
        let qs = &block[48..176]; // 128 nibble bytes

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        let mut is = 0usize;
        let mut qs_off = 0usize;
        let mut w_off = 0usize;

        for group in 0..4 {
            // Sub-block `is` (lo nibbles) — Q8_0 act block index `blk*8 + is`.
            let a_idx_lo = blk * 8 + is;
            let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
            // SAFETY: acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES.
            let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
            let d_a_lo = f16_to_f32(a_block_lo);
            let q8_lo = &a_block_lo[2..]; // 32 i8 values

            let da_lo = d * sc[is] as f32;
            let m_lo = dmin * mn[is] as f32;

            // Sub-block `is+1` (hi nibbles) — Q8_0 act block index `blk*8 + is + 1`.
            let a_idx_hi = blk * 8 + is + 1;
            let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
            // SAFETY: same guarantee as above.
            let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
            let d_a_hi = f16_to_f32(a_block_hi);
            let q8_hi = &a_block_hi[2..]; // 32 i8 values

            let da_hi = d * sc[is + 1] as f32;
            let m_hi = dmin * mn[is + 1] as f32;

            // Vectorised inner loops: compute Σ(q5_lo * q8_lo) and Σ(q8_lo) in 4-wide chunks.
            // Using i32 accumulators for the integer dot products.
            let mut dot_lo_i32 = _mm256_setzero_si256();
            let mut sum_a_lo_i32 = _mm256_setzero_si256();
            let mut dot_hi_i32 = _mm256_setzero_si256();
            let mut sum_a_hi_i32 = _mm256_setzero_si256();

            // Process groups of 8 within the sub-block.
            let valid_lo = cols_in_block.saturating_sub(w_off).min(32);
            let valid_hi = cols_in_block.saturating_sub(w_off + 32).min(32);

            // Lo sub-block (weights w_off..w_off+32).
            if valid_lo >= 8 {
                let full_iters_lo = valid_lo / 8;
                for chunk in 0..full_iters_lo {
                    let l = chunk * 8;
                    let q5_ptr = qs_off + l;

                    // Extract 8 lo nibbles from qs bytes (two bytes each cover 2 weights).
                    // Byte qs_off + l/2 has lo = weight 2*(l/2), hi = weight 2*(l/2)+1
                    // But we process in the nibble-interleaved order stored by Q5_K.
                    // For a clean SIMD approach: expand 4 bytes into 8 nibbles.
                    // Actually Q5_K lo nibbles: for l in 0..32, qs[qs_off+l] & 0x0F.
                    // We have 8 consecutive positions here.
                    let mut nib_buf = [0i8; 8];
                    let mut q8_buf = [0i8; 8];
                    for j in 0..8 {
                        let qh_bit = (qh[l + j] >> group) & 1;
                        nib_buf[j] = ((qs[q5_ptr + j] & 0x0F) | (qh_bit << 4)) as i8;
                        q8_buf[j] = q8_lo[l + j] as i8;
                    }

                    // SAFETY: nib_buf and q8_buf are stack arrays; loads are valid.
                    let vw =
                        _mm256_cvtepi8_epi32(_mm_loadl_epi64(nib_buf.as_ptr() as *const __m128i));
                    let va =
                        _mm256_cvtepi8_epi32(_mm_loadl_epi64(q8_buf.as_ptr() as *const __m128i));
                    dot_lo_i32 = _mm256_add_epi32(dot_lo_i32, _mm256_mullo_epi32(vw, va));
                    sum_a_lo_i32 = _mm256_add_epi32(sum_a_lo_i32, va);
                }
            }
            // Scalar tail for lo sub-block.
            let mut dot_lo_scalar = 0.0f32;
            let mut sum_a_lo_scalar = 0.0f32;
            let lo_simd_done = (valid_lo / 8) * 8;
            for l in lo_simd_done..valid_lo {
                let qh_bit = (qh[l] >> group) & 1;
                let q_w = ((qs[qs_off + l] & 0x0F) | (qh_bit << 4)) as f32;
                let q_a = q8_lo[l] as i8 as f32;
                dot_lo_scalar += q_w * q_a;
                sum_a_lo_scalar += q_a;
            }
            // Combine SIMD and scalar results for lo sub-block.
            let dot_lo_total = hsum_i32_avx2(dot_lo_i32) as f32 + dot_lo_scalar;
            let sum_a_lo_total = hsum_i32_avx2(sum_a_lo_i32) as f32 + sum_a_lo_scalar;
            row_sum += (da_lo * dot_lo_total - m_lo * sum_a_lo_total) * d_a_lo;

            // Hi sub-block (weights w_off+32..w_off+64).
            if valid_hi >= 8 {
                let full_iters_hi = valid_hi / 8;
                for chunk in 0..full_iters_hi {
                    let l = chunk * 8;
                    let q5_ptr = qs_off + l;

                    let mut nib_buf = [0i8; 8];
                    let mut q8_buf = [0i8; 8];
                    for j in 0..8 {
                        let qh_bit = (qh[l + j] >> (group + 4)) & 1;
                        nib_buf[j] = (((qs[q5_ptr + j] >> 4) & 0x0F) | (qh_bit << 4)) as i8;
                        q8_buf[j] = q8_hi[l + j] as i8;
                    }

                    let vw =
                        _mm256_cvtepi8_epi32(_mm_loadl_epi64(nib_buf.as_ptr() as *const __m128i));
                    let va =
                        _mm256_cvtepi8_epi32(_mm_loadl_epi64(q8_buf.as_ptr() as *const __m128i));
                    dot_hi_i32 = _mm256_add_epi32(dot_hi_i32, _mm256_mullo_epi32(vw, va));
                    sum_a_hi_i32 = _mm256_add_epi32(sum_a_hi_i32, va);
                }
            }
            let mut dot_hi_scalar = 0.0f32;
            let mut sum_a_hi_scalar = 0.0f32;
            let hi_simd_done = (valid_hi / 8) * 8;
            for l in hi_simd_done..valid_hi {
                let qh_bit = (qh[l] >> (group + 4)) & 1;
                let q_w = (((qs[qs_off + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
                let q_a = q8_hi[l] as i8 as f32;
                dot_hi_scalar += q_w * q_a;
                sum_a_hi_scalar += q_a;
            }
            let dot_hi_total = hsum_i32_avx2(dot_hi_i32) as f32 + dot_hi_scalar;
            let sum_a_hi_total = hsum_i32_avx2(sum_a_hi_i32) as f32 + sum_a_hi_scalar;
            row_sum += (da_hi * dot_hi_total - m_hi * sum_a_hi_total) * d_a_hi;

            is += 2;
            qs_off += 32;
            w_off += 64;
        }
    }

    row_sum
}

/// Horizontal sum of a 256-bit i32 register.
///
/// # Safety
/// Requires `avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(hi, lo);
    let shuf = _mm_shuffle_epi32(s, 0b10_11_00_01);
    let s2 = _mm_add_epi32(s, shuf);
    let shuf2 = _mm_shuffle_epi32(s2, 0b00_00_10_10);
    let s3 = _mm_add_epi32(s2, shuf2);
    _mm_cvtsi128_si32(s3)
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Expand the high-bit vector for one group pass.
///
/// For each of the 32 positions in the lo or hi sub-block, the 5th bit of the
/// 5-bit quant comes from bit `bit_pos` of `qh[0..32]`.  We process positions
/// 0..16 (lo_half) and 16..32 (hi_half) separately (one `__m128i` each).
///
/// The function returns a `__m128i` where each byte is either `0x10` (bit set)
/// or `0x00` (bit clear), ready to be OR-ed into the nibble value.
///
/// # Safety
/// Requires AVX2.  `bit_pos` must be in 0..8.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_high_bit(qh_half: __m128i, bit_pos: u32) -> __m128i {
    // SAFETY: bit_pos in 0..8 guaranteed by caller; 1u8 << bit_pos is always
    // a valid byte mask.
    let mask_byte = (1u8 << bit_pos) as i8;
    let mask_vec = _mm_set1_epi8(mask_byte);

    // Isolate the target bit in each byte.
    let masked = _mm_and_si128(qh_half, mask_vec);

    // masked == 0 ⟹ zero vector (bit is clear), masked ≠ 0 ⟹ all-ones (bit is set).
    let is_zero = _mm_cmpeq_epi8(masked, _mm_setzero_si128());

    // _mm_andnot_si128(a, b) = (!a) & b.
    // When is_zero=0xFF (bit was clear) → result=0x00.
    // When is_zero=0x00 (bit was set)   → result=0x10 (high nibble).
    _mm_andnot_si128(is_zero, _mm_set1_epi8(0x10_u8 as i8))
}

/// Merge `__m128i` nibbles (0..15) with high bits (0x00 or 0x10), then widen
/// one 8-element slice (from the first 8 bytes) to 8 × `i32`, and convert to
/// 8 × `f32`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn nibbles_or_high_to_f32(nibbles: __m128i, high: __m128i) -> (__m256, __m256) {
    let q5 = _mm_or_si128(nibbles, high); // q5 in each byte: 0..31 (unsigned)

    // First 8 bytes → 8 × i32 → 8 × f32.
    let lo_i32 = _mm256_cvtepu8_epi32(q5);
    let lo_f32 = _mm256_cvtepi32_ps(lo_i32);

    // Second 8 bytes (shift by 8).
    let q5_hi = _mm_srli_si128(q5, 8);
    let hi_i32 = _mm256_cvtepu8_epi32(q5_hi);
    let hi_f32 = _mm256_cvtepi32_ps(hi_i32);

    (lo_f32, hi_f32)
}

/// Dequantize one 176-byte Q5_K block into 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 176`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 176 >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    let (sc, mn) = decode_scales_mins(&block[4..16]);

    // qh: 32 bytes at offset 16, covering all 256 weight positions.
    // qs: 128 nibble bytes at offset 48.
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

    let mut is = 0usize; // sub-block index
    let mut qs_off = 0usize; // nibble byte offset (0, 32, 64, 96)
    let mut out_off = 0usize; // output float offset (0, 64, 128, 192)

    for group in 0..4u32 {
        let a_lo = d * sc[is] as f32;
        let b_lo = dmin * mn[is] as f32;
        let a_hi = d * sc[is + 1] as f32;
        let b_hi = dmin * mn[is + 1] as f32;

        let va_lo = _mm256_set1_ps(a_lo);
        let vb_lo = _mm256_set1_ps(b_lo);
        let va_hi = _mm256_set1_ps(a_hi);
        let vb_hi = _mm256_set1_ps(b_hi);

        // Load 32 nibble bytes (= 64 nibbles = 32 lo-sub + 32 hi-sub weights).
        // SAFETY: qs_off + 32 <= 128; qs.len() == 128.
        let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
        let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

        // Separate lo and hi nibbles (identical to Q4_K).
        let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo); // lo-sub weights 0..15
        let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo); // lo-sub weights 16..31
        let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo); // hi-sub weights 0..15
        let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo); // hi-sub weights 16..31

        // Load qh for positions 0..16 and 16..32.
        // SAFETY: qh.len() == 32; positions 0..16 and 16..32.
        let qh_0 = _mm_loadu_si128(qh.as_ptr() as *const __m128i); // bytes 0..15
        let qh_1 = _mm_loadu_si128(qh.as_ptr().add(16) as *const __m128i); // bytes 16..31

        // High bits for lo sub-block: bit `group` of qh (positions 0..32).
        let hb_lo_0 = extract_high_bit(qh_0, group); // positions 0..15
        let hb_lo_1 = extract_high_bit(qh_1, group); // positions 16..31

        // High bits for hi sub-block: bit `group+4` of qh.
        let hb_hi_0 = extract_high_bit(qh_0, group + 4); // positions 0..15
        let hb_hi_1 = extract_high_bit(qh_1, group + 4); // positions 16..31

        // --- Lo sub-block: 32 weights = a_lo * q5 - b_lo ---

        // Positions 0..15 of lo sub-block.
        let (q0_lo, q0_hi) = nibbles_or_high_to_f32(lo_nibbles_0, hb_lo_0);
        let w0 = _mm256_fmsub_ps(va_lo, q0_lo, vb_lo);
        let w1 = _mm256_fmsub_ps(va_lo, q0_hi, vb_lo);

        // Positions 16..31 of lo sub-block.
        let (q1_lo, q1_hi) = nibbles_or_high_to_f32(lo_nibbles_1, hb_lo_1);
        let w2 = _mm256_fmsub_ps(va_lo, q1_lo, vb_lo);
        let w3 = _mm256_fmsub_ps(va_lo, q1_hi, vb_lo);

        // Store 32 lo-sub-block weights.
        // SAFETY: out_off + 32 <= 256; output.len() >= 256.
        let ptr_lo = output.as_mut_ptr().add(out_off);
        _mm256_storeu_ps(ptr_lo, w0);
        _mm256_storeu_ps(ptr_lo.add(8), w1);
        _mm256_storeu_ps(ptr_lo.add(16), w2);
        _mm256_storeu_ps(ptr_lo.add(24), w3);

        // --- Hi sub-block: 32 weights = a_hi * q5 - b_hi ---

        // Positions 0..15 of hi sub-block.
        let (q2_lo, q2_hi) = nibbles_or_high_to_f32(hi_nibbles_0, hb_hi_0);
        let w4 = _mm256_fmsub_ps(va_hi, q2_lo, vb_hi);
        let w5 = _mm256_fmsub_ps(va_hi, q2_hi, vb_hi);

        // Positions 16..31 of hi sub-block.
        let (q3_lo, q3_hi) = nibbles_or_high_to_f32(hi_nibbles_1, hb_hi_1);
        let w6 = _mm256_fmsub_ps(va_hi, q3_lo, vb_hi);
        let w7 = _mm256_fmsub_ps(va_hi, q3_hi, vb_hi);

        // Store 32 hi-sub-block weights.
        // SAFETY: out_off + 64 <= 256; output.len() >= 256.
        let ptr_hi = output.as_mut_ptr().add(out_off + 32);
        _mm256_storeu_ps(ptr_hi, w4);
        _mm256_storeu_ps(ptr_hi.add(8), w5);
        _mm256_storeu_ps(ptr_hi.add(16), w6);
        _mm256_storeu_ps(ptr_hi.add(24), w7);

        is += 2;
        qs_off += 32;
        out_off += 64;
    }
}

/// Compute the dot product of one row of a Q5_K matrix with an FP32 vector.
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

        // SAFETY: block.len() == 176 >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);
        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds.
            let qh = &block[16..48];
            let qs = &block[48..176];
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

            let mut block_acc = _mm256_setzero_ps();
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for group in 0..4u32 {
                let a_lo = d * sc[is] as f32;
                let b_lo = dmin * mn[is] as f32;
                let a_hi = d * sc[is + 1] as f32;
                let b_hi = dmin * mn[is + 1] as f32;

                let va_lo = _mm256_set1_ps(a_lo);
                let vb_lo = _mm256_set1_ps(b_lo);
                let va_hi = _mm256_set1_ps(a_hi);
                let vb_hi = _mm256_set1_ps(b_hi);

                // SAFETY: qs_off + 32 <= 128.
                let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

                let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo);
                let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo);
                let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
                let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

                // SAFETY: qh.len() == 32.
                let qh_0 = _mm_loadu_si128(qh.as_ptr() as *const __m128i);
                let qh_1 = _mm_loadu_si128(qh.as_ptr().add(16) as *const __m128i);

                let hb_lo_0 = extract_high_bit(qh_0, group);
                let hb_lo_1 = extract_high_bit(qh_1, group);
                let hb_hi_0 = extract_high_bit(qh_0, group + 4);
                let hb_hi_1 = extract_high_bit(qh_1, group + 4);

                // SAFETY: w_off + 64 <= input_offset + BLOCK_SIZE <= n_cols.
                let inp_lo = input.as_ptr().add(w_off);
                let inp_hi = input.as_ptr().add(w_off + 32);

                // Lo sub-block
                let (q0_lo, q0_hi) = nibbles_or_high_to_f32(lo_nibbles_0, hb_lo_0);
                let w0 = _mm256_fmsub_ps(va_lo, q0_lo, vb_lo);
                block_acc = _mm256_fmadd_ps(w0, _mm256_loadu_ps(inp_lo), block_acc);

                let w1 = _mm256_fmsub_ps(va_lo, q0_hi, vb_lo);
                block_acc = _mm256_fmadd_ps(w1, _mm256_loadu_ps(inp_lo.add(8)), block_acc);

                let (q1_lo, q1_hi) = nibbles_or_high_to_f32(lo_nibbles_1, hb_lo_1);
                let w2 = _mm256_fmsub_ps(va_lo, q1_lo, vb_lo);
                block_acc = _mm256_fmadd_ps(w2, _mm256_loadu_ps(inp_lo.add(16)), block_acc);

                let w3 = _mm256_fmsub_ps(va_lo, q1_hi, vb_lo);
                block_acc = _mm256_fmadd_ps(w3, _mm256_loadu_ps(inp_lo.add(24)), block_acc);

                // Hi sub-block
                let (q2_lo, q2_hi) = nibbles_or_high_to_f32(hi_nibbles_0, hb_hi_0);
                let w4 = _mm256_fmsub_ps(va_hi, q2_lo, vb_hi);
                block_acc = _mm256_fmadd_ps(w4, _mm256_loadu_ps(inp_hi), block_acc);

                let w5 = _mm256_fmsub_ps(va_hi, q2_hi, vb_hi);
                block_acc = _mm256_fmadd_ps(w5, _mm256_loadu_ps(inp_hi.add(8)), block_acc);

                let (q3_lo, q3_hi) = nibbles_or_high_to_f32(hi_nibbles_1, hb_hi_1);
                let w6 = _mm256_fmsub_ps(va_hi, q3_lo, vb_hi);
                block_acc = _mm256_fmadd_ps(w6, _mm256_loadu_ps(inp_hi.add(16)), block_acc);

                let w7 = _mm256_fmsub_ps(va_hi, q3_hi, vb_hi);
                block_acc = _mm256_fmadd_ps(w7, _mm256_loadu_ps(inp_hi.add(24)), block_acc);

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            row_sum += hsum_f32_avx(block_acc);
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let qh = &block[16..48];
            let qs = &block[48..176];
            let mut partial_sum = 0.0f32;
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for group in 0..4usize {
                let d1 = d * sc[is] as f32;
                let m1 = dmin * mn[is] as f32;
                let d2 = d * sc[is + 1] as f32;
                let m2 = dmin * mn[is + 1] as f32;

                for l in 0..32 {
                    let idx = w_off + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128; qh index l < 32.
                        let lo_nib = (*qs.get_unchecked(qs_off + l) & 0x0F) as u32;
                        let hi_bit = ((*qh.get_unchecked(l) >> group) & 1) as u32;
                        let q = (lo_nib | (hi_bit << 4)) as f32;
                        partial_sum += (d1 * q - m1) * input[idx];
                    }
                }

                for l in 0..32 {
                    let idx = w_off + 32 + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128; qh index l < 32.
                        let hi_nib = ((*qs.get_unchecked(qs_off + l) >> 4) & 0x0F) as u32;
                        let hi_bit = ((*qh.get_unchecked(l) >> (group + 4)) & 1) as u32;
                        let q = (hi_nib | (hi_bit << 4)) as f32;
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::q5_k::Q5KRef;

    fn make_q5k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 12],
        qh: &[u8; 32],
        qs: &[u8; 128],
    ) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block.extend_from_slice(scales);
        block.extend_from_slice(qh);
        block.extend_from_slice(qs);
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> crate::types::QuantTensor {
        crate::types::QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q5K)
    }

    #[test]
    fn test_q5k_avx2_dequant_matches_reference_zero_high_bits() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x00u8; 32];
        let qs = [0x88u8; 128]; // lo=8, hi=8

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [zero-high-bits] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q5k_avx2_dequant_matches_reference_all_high_bits() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xFFu8; 32]; // all high bits set
        let qs = [0x00u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [all-high-bits] at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q5k_avx2_dequant_matches_reference_varied() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Varied scales, alternating high bits, non-trivial nibbles.
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0x3F) as u8;
        }
        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let block = make_q5k_block(0.5, 0.25, &scales, &qh, &qs);
        let mut out_avx2 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant");
        Q5KRef
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
    fn test_q5k_avx2_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xAAu8; 32]; // alternating bits
        let qs = [0x5Au8; 128];

        let block = make_q5k_block(0.5, 0.1, &scales, &qh, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv");
        Q5KRef
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
    fn test_q5k_avx2_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 200 columns — partial block.
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x55u8; 32];
        let qs = [0x11u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv partial");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "partial gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q5k_avx2_gemv_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0x3F) as u8;
        }
        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let block = make_q5k_block(0.5, 0.25, &scales, &qh, &qs);
        let tensor_avx2 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx2
            .gemv(&tensor_avx2, &input, &mut out_avx2)
            .expect("avx2 gemv varied");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv varied");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "varied gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    // ── matvec_q8_fused ───────────────────────────────────────────────────

    /// Build a minimal valid Q8_0 activation block (34 bytes: 2-byte f16 scale + 32 i8).
    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    /// Build `n_q8_blocks` identical Q8_0 blocks.
    fn make_q8_acts(n_q8_blocks: usize, scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let single = make_q8_0_block(scale, values);
        single.repeat(n_q8_blocks)
    }

    #[test]
    fn test_q5k_avx2_fused_matches_reference_single_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // One Q5_K super-block (256 weights) needs 8 Q8_0 activation blocks.
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 7 + 5) & 0x3F) as u8;
        }
        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 13 + 3) & 0xFF) as u8;
        }
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }

        let w_block = make_q5k_block(0.5, 0.25, &scales, &qh, &qs);
        // 8 Q8_0 activation blocks for one Q5_K super-block.
        let act_vals = [
            2i8, -3, 5, -7, 1, -1, 4, -4, 6, -6, 3, -3, 2, -2, 1, -1, 8, -8, 7, -7, 6, -6, 5, -5,
            4, -4, 3, -3, 2, -2, 1, -1,
        ];
        let acts = make_q8_acts(8, 0.1, &act_vals);

        let mut out_avx2 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out_avx2, 1, 256)
            .expect("avx2 fused single block");
        Q5KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, 256)
            .expect("ref fused single block");

        let err = (out_avx2[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "fused single-block mismatch: avx2={} ref={} err={}",
            out_avx2[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q5k_avx2_fused_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 3 rows × 512 cols = 2 Q5_K super-blocks per row → 16 Q8_0 act blocks per row.
        let n_rows = 3usize;
        let n_cols = 512usize;
        let blocks_per_row = 2usize;
        let q8_blocks_per_row = blocks_per_row * 8; // = 16

        let mut all_weights = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut scales = [0u8; 12];
                for (i, s) in scales.iter_mut().enumerate() {
                    *s = ((r * 17 + b * 11 + i * 7 + 5) & 0x3F) as u8;
                }
                let mut qh = [0u8; 32];
                for (i, h) in qh.iter_mut().enumerate() {
                    *h = ((r * 13 + b * 19 + i * 3) & 0xFF) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((r * 5 + b * 23 + i * 11) & 0xFF) as u8;
                }
                all_weights.extend(make_q5k_block(
                    0.5 + r as f32 * 0.1,
                    0.1 + b as f32 * 0.05,
                    &scales,
                    &qh,
                    &qs,
                ));
            }
        }

        let act_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 0, 1, -1, 2, -2, 3, -3,
            4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let acts = make_q8_acts(q8_blocks_per_row, 0.05, &act_vals);

        let mut out_avx2 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q5_KAvx2
            .matvec_q8_fused(&all_weights, &acts, &mut out_avx2, n_rows, n_cols)
            .expect("avx2 fused multi-row");
        Q5KRef
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused multi-row");

        for i in 0..n_rows {
            let err = (out_avx2[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "fused multi-row row {i}: avx2={} ref={} err={}",
                out_avx2[i],
                out_ref[i],
                err
            );
        }
    }

    #[test]
    fn test_q5k_avx2_fused_accumulate_semantics() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // d=0 → all zero contribution; pre-existing out value must be preserved.
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let w_block = make_q5k_block(0.0, 0.0, &scales, &[0u8; 32], &[0u8; 128]);
        let acts = make_q8_acts(8, 0.0, &[0i8; 32]);

        let mut out = vec![99.0f32; 1];
        Q5_KAvx2
            .matvec_q8_fused(&w_block, &acts, &mut out, 1, 256)
            .expect("avx2 fused accumulate");

        assert!(
            (out[0] - 99.0).abs() < 1e-5,
            "accumulate semantics broken: expected 99.0, got {}",
            out[0]
        );
    }
}
