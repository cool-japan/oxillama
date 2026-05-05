//! AVX-512 accelerated Q3_K quantization kernel.
//!
//! Q3_K block layout (110 bytes per 256 weights):
//! - bytes[0..32]   — hmask: 1 bit per weight; if set → subtract 0; if clear → subtract 4
//! - bytes[32..96]  — qs: lower 2 bits of each 3-bit quant (4 per byte via shifts)
//! - bytes[96..108] — 12 bytes packed scales (16 × 6-bit unsigned, decode then
//!   subtract 32 for signed range -32..31)
//! - bytes[108..110] — FP16 super-block scale `d` (little-endian)
//!
//! Q3_K is a symmetric ("type-0") format — NO minimum offset.
//! Weight formula: `w = d * scale_i * (q_lo - (hmask_bit ? 0 : 4))`
//!   where q_lo is 2-bit (0..3), giving effective range -4..3.
//!
//! 16 sub-blocks of 16 weights each (2 groups × 4 shifts × 2 sub-blocks).
//! The hmask bit selector `m` rotates through bits 0..7 across the 8 passes.
//!
//! ## AVX-512 strategy
//!
//! For each sub-block of 16 weights, use **one** AVX-512 (ZMM, 16-wide) pass
//! instead of AVX2's two 8-wide passes:
//!
//! 1. Extract 2-bit q_lo via `_mm_srli_epi16` + AND 0x03.
//! 2. Compute hmask correction: AND(hmask_half, m_vec), cmpeq-zero → 0xFF/0x00, AND(4).
//! 3. Widen q_lo and correction to i32 with `_mm512_cvtepu8_epi32`, subtract.
//! 4. Convert to f32, multiply by `d * scale`.
//!
//! This yields 2× the output width vs the AVX2 kernel in one SIMD instruction.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q3_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q3_K block: 32 (hmask) + 64 (qs) + 12 (scales) + 2 (FP16 d).
pub const BLOCK_BYTES: usize = 110;

/// AVX-512 accelerated Q3_K kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q3_KAvx512;

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
/// Requires `avx512f` CPU feature (SSE/AVX subset).  `shift` must be one of 0, 2, 4, 6.
#[target_feature(enable = "avx512f")]
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
/// clear.  AND with 4 gives the correction value.
///
/// # Safety
/// Requires `avx512f` CPU feature (SSE/AVX subset).  `m` selects which bit to test.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hmask_correction_16(hmask_half: __m128i, m_vec: __m128i) -> __m128i {
    // SAFETY: all intrinsics require SSE2 (subset of avx512f), guaranteed by target_feature.
    let masked = _mm_and_si128(hmask_half, m_vec);
    let is_zero = _mm_cmpeq_epi8(masked, _mm_setzero_si128());
    // is_zero: 0xFF where bit was clear (subtract 4), 0x00 where set (subtract 0)
    _mm_and_si128(is_zero, _mm_set1_epi8(4))
}

impl QuantKernel for Q3_KAvx512 {
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
        "Q3_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 110-byte Q3_K block into 256 FP32 values using AVX-512.
///
/// Processes 2 groups of 128 weights.  Within each group, the same 32 qs
/// bytes are re-used with shifts 0, 2, 4, 6 and the same 32 hmask bytes
/// are tested with a rotating 1-bit selector.  Each sub-block of 16 weights
/// is processed with a single 16-wide AVX-512 (ZMM) pass.
///
/// # Safety
/// - `block.len() >= 110`
/// - `output.len() >= 256`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
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

            // --- Sub-block 0: 16 weights using hmask[0..16] ---
            // One AVX-512 pass handles all 16 values (vs AVX2's two 8-wide passes).
            {
                let dl = d * sc[is];
                is += 1;
                let vdl = _mm512_set1_ps(dl);

                // SAFETY: extract_2bit_16 requires avx512f and shift in {0,2,4,6}.
                let q_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                // SAFETY: hmask_correction_16 requires avx512f (SSE2 subset).
                let correction = hmask_correction_16(hmask_lo, m_vec);

                // Widen q and correction to 16 × i32, subtract, then convert to f32.
                // SAFETY: _mm512_cvtepu8_epi32 reads from the low 16 bytes of __m128i.
                let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                let c_i32 = _mm512_cvtepu8_epi32(correction);
                let val_i32 = _mm512_sub_epi32(q_i32, c_i32);
                let val_f32 = _mm512_cvtepi32_ps(val_i32);
                let w = _mm512_mul_ps(vdl, val_f32);

                // SAFETY: out_off + 16 <= 256; output.len() >= 256.
                _mm512_storeu_ps(output.as_mut_ptr().add(out_off), w);
                out_off += 16;
            }

            // --- Sub-block 1: 16 weights using hmask[16..32] ---
            {
                let dl = d * sc[is];
                is += 1;
                let vdl = _mm512_set1_ps(dl);

                let q_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                let correction = hmask_correction_16(hmask_hi, m_vec);

                let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                let c_i32 = _mm512_cvtepu8_epi32(correction);
                let val_i32 = _mm512_sub_epi32(q_i32, c_i32);
                let val_f32 = _mm512_cvtepi32_ps(val_i32);
                let w = _mm512_mul_ps(vdl, val_f32);

                // SAFETY: out_off + 16 <= 256; output.len() >= 256.
                _mm512_storeu_ps(output.as_mut_ptr().add(out_off), w);
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
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
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
            // Fast path: all 256 weights in bounds — fully vectorized with AVX-512.

            // SAFETY: hmask.len() == 32.
            let hmask_lo = _mm_loadu_si128(hmask.as_ptr() as *const __m128i);
            let hmask_hi = _mm_loadu_si128(hmask.as_ptr().add(16) as *const __m128i);

            let mask_2bit = _mm_set1_epi8(0x03);
            let mut block_acc = _mm512_setzero_ps();
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
                        let vdl = _mm512_set1_ps(dl);

                        let q_bytes = extract_2bit_16(raw_a, shift, mask_2bit);
                        let correction = hmask_correction_16(hmask_lo, m_vec);

                        let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                        let c_i32 = _mm512_cvtepu8_epi32(correction);
                        let val_f32 = _mm512_cvtepi32_ps(_mm512_sub_epi32(q_i32, c_i32));
                        let w = _mm512_mul_ps(vdl, val_f32);

                        // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                        let i_vec = _mm512_loadu_ps(input.as_ptr().add(w_off));
                        block_acc = _mm512_fmadd_ps(w, i_vec, block_acc);
                        w_off += 16;
                    }

                    // --- Sub-block 1: 16 weights, hmask bytes 16..31 ---
                    {
                        let dl = d * sc[is];
                        is += 1;
                        let vdl = _mm512_set1_ps(dl);

                        let q_bytes = extract_2bit_16(raw_b, shift, mask_2bit);
                        let correction = hmask_correction_16(hmask_hi, m_vec);

                        let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                        let c_i32 = _mm512_cvtepu8_epi32(correction);
                        let val_f32 = _mm512_cvtepi32_ps(_mm512_sub_epi32(q_i32, c_i32));
                        let w = _mm512_mul_ps(vdl, val_f32);

                        // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                        let i_vec = _mm512_loadu_ps(input.as_ptr().add(w_off));
                        block_acc = _mm512_fmadd_ps(w, i_vec, block_acc);
                        w_off += 16;
                    }
                }
            }

            row_sum += hsum_f32_avx512(block_acc);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
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

    // -----------------------------------------------------------------------
    // Dequant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q3k_avx512_dequant_matches_reference_short() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 1 block, d=0 → all weights = 0.
        let block = make_q3k_block(0.0, &[0; 12], &[0; 32], &[0; 64]);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "dequant mismatch [short/zeros] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_dequant_matches_reference_long() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 16 super-blocks worth of data — test the first block from a large buffer.
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

        let single = make_q3k_block(0.5, &scales, &hmask, &qs);
        // Concatenate 16 copies to simulate a large buffer
        let data: Vec<u8> = single
            .iter()
            .cloned()
            .cycle()
            .take(BLOCK_BYTES * 16)
            .collect();

        let mut out_avx512 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ref = vec![0.0f32; BLOCK_SIZE];

        Q3_KAvx512
            .dequant_block(&data[0..BLOCK_BYTES], &mut out_avx512)
            .expect("avx512 dequant");
        Q3KRef
            .dequant_block(&data[0..BLOCK_BYTES], &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [long] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_dequant_hmask_all_set() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // hmask all set → subtract 0.  qs all 0 → q_lo = 0.
        // Value = 0 - 0 = 0.  Weight = d * scale * 0 = 0.
        let hmask = [0xFFu8; 32];
        let qs = [0x00u8; 64];
        let mut scales = [0u8; 12];
        scales[..8].fill(0x21);

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let mut out_avx512 = vec![99.0f32; 256];
        let mut out_ref = vec![99.0f32; 256];

        Q3_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [hmask_set] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_dequant_hmask_all_clear() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // hmask all clear → subtract 4.  qs all 0 → q_lo = 0.
        // Value = 0 - 4 = -4.  Weight = d * scale * (-4).
        let hmask = [0x00u8; 32];
        let qs = [0x00u8; 64];
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [hmask_clear] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_dequant_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q3_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q3KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [varied] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_zero_block_all_zeros() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // d=0 → zero weights regardless of hmask/qs.
        let block = make_q3k_block(0.0, &[0; 12], &[0xFF; 32], &[0xFF; 64]);
        let mut out = vec![1.0f32; 256];
        Q3_KAvx512.dequant_block(&block, &mut out).expect("dequant");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "expected 0 at [{i}], got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // GEMV tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q3k_avx512_matvec_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "matvec mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q3k_avx512_odd_rows_no_panic() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 5 rows (non-power-of-2), 256 cols
        let block = make_q3k_block(1.0, &[0x21; 12], &[0xAAu8; 32], &[0x55u8; 64]);
        let mut data = Vec::new();
        for _ in 0..5 {
            data.extend_from_slice(&block);
        }
        let tensor = QuantTensor::new(data, vec![5, 256], oxillama_gguf::GgufTensorType::Q3K);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 5];
        Q3_KAvx512
            .gemv(&tensor, &input, &mut output)
            .expect("odd rows gemv");
        // All 5 outputs should be identical (same block)
        for i in 1..5 {
            assert!(
                (output[0] - output[i]).abs() < 1e-5,
                "rows 0 and {i} should match"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 200 columns — partial block.
        let hmask = [0xAAu8; 32];
        let qs = [0x55u8; 64];
        let mut scales = [0u8; 12];
        scales[..8].fill(0x21);

        let block = make_q3k_block(1.0, &scales, &hmask, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv partial");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q3k_avx512_gemv_alternating_hmask() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv alternating");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv alternating");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "alternating gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q3k_avx512_gemv_multiple_rows() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 4 rows with different blocks
        let block0 = make_q3k_block(0.5, &[0x20; 12], &[0xFFu8; 32], &[0x55u8; 64]);
        let block1 = make_q3k_block(1.0, &[0x00; 12], &[0x00u8; 32], &[0xFFu8; 64]);
        let block2 = make_q3k_block(0.25, &[0x3F; 12], &[0xAAu8; 32], &[0xAAu8; 64]);
        let block3 = make_q3k_block(0.75, &[0x15; 12], &[0x55u8; 32], &[0x33u8; 64]);

        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        data.extend_from_slice(&block2);
        data.extend_from_slice(&block3);

        let tensor_avx512 = QuantTensor::new(
            data.clone(),
            vec![4, 256],
            oxillama_gguf::GgufTensorType::Q3K,
        );
        let tensor_ref = QuantTensor::new(data, vec![4, 256], oxillama_gguf::GgufTensorType::Q3K);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 4];
        let mut out_ref = vec![0.0f32; 4];

        Q3_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv 4-rows");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv 4-rows");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 0.1,
                "multi-row gemv mismatch row {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q3k_avx512_block_boundary_alignment() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Exactly 2 complete blocks (512 elements).
        let block = make_q3k_block(0.5, &[0x25; 12], &[0xAAu8; 32], &[0x55u8; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);

        let tensor_avx512 = QuantTensor::new(
            data.clone(),
            vec![1, 512],
            oxillama_gguf::GgufTensorType::Q3K,
        );
        let tensor_ref = QuantTensor::new(data, vec![1, 512], oxillama_gguf::GgufTensorType::Q3K);

        let input = vec![1.0f32; 512];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q3_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv 2-blocks");
        Q3KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv 2-blocks");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "2-block gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    // -----------------------------------------------------------------------
    // Dispatch test
    // -----------------------------------------------------------------------

    #[test]
    fn test_q3k_dispatcher_routes_to_avx512_when_available() {
        use crate::dispatch::global_dispatcher;
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let kernel = global_dispatcher()
            .get_kernel(oxillama_gguf::GgufTensorType::Q3K)
            .expect("dispatcher Q3K");
        // When AVX-512 is detected, the kernel name should be Q3_K.
        assert_eq!(kernel.name(), "Q3_K");
    }

    // -----------------------------------------------------------------------
    // Error path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q3k_avx512_buffer_too_small_block() {
        let block = vec![0u8; 10]; // too small
        let mut output = vec![0.0f32; 256];
        assert!(Q3_KAvx512.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_q3k_avx512_buffer_too_small_output() {
        let block = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; 10]; // too small
        assert!(Q3_KAvx512.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_q3k_avx512_gemv_empty_input_error() {
        let block = make_q3k_block(1.0, &[0x21; 12], &[0; 32], &[0; 64]);
        let tensor = make_tensor(block, 256);
        let input = vec![]; // empty — too short
        let mut output = vec![0.0f32; 1];
        assert!(Q3_KAvx512.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_q3k_avx512_kernel_metadata() {
        assert_eq!(Q3_KAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Q3_KAvx512.block_bytes(), BLOCK_BYTES);
        assert_eq!(Q3_KAvx512.name(), "Q3_K");
    }
}
