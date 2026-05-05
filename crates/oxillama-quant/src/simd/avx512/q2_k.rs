//! AVX-512 accelerated Q2_K quantization kernel.
//!
//! Q2_K block layout (84 bytes per 256 weights):
//! - bytes[0..16]  — 16 scale bytes (lo 4 bits = scale, hi 4 bits = min)
//! - bytes[16..80] — 64 qs bytes (256 × 2-bit packed, 4 per byte via shifts 0,2,4,6)
//! - bytes[80..82] — FP16 super-block scale `d` (little-endian)
//! - bytes[82..84] — FP16 super-block minimum `dmin` (little-endian)
//!
//! NOTE: In Q2_K, d/dmin come AFTER scales and qs in memory.
//!
//! 16 sub-blocks of 16 weights each (2 groups of 128, each group processes
//! the same 32 qs bytes with 4 different shift amounts).
//!
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 2-bit (0..3).
//!
//! ## AVX-512 strategy
//!
//! For each of the 16 sub-blocks (16 weights each), use **one** AVX-512
//! (16-wide) pass instead of AVX2's two 8-wide passes:
//!
//! 1. Extract 2-bit values from pre-loaded qs bytes via `_mm_srli_epi16` + AND 0x03.
//! 2. Widen via `_mm512_cvtepu8_epi32` (16 u8 → 16 i32).
//! 3. Convert to f32 and apply `_mm512_fmsub_ps(va, q_f32, vb)` for `d*scale*q - dmin*min`.
//!
//! This yields 2× the output width vs the AVX2 kernel in one SIMD instruction.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q2_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q2_K block: 16 (scales) + 64 (qs) + 2 (FP16 d) + 2 (FP16 dmin).
pub const BLOCK_BYTES: usize = 84;

/// AVX-512 accelerated Q2_K kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q2_KAvx512;

/// Extract 2-bit values from 16 packed bytes using the given bit-shift.
///
/// Each source byte contains four 2-bit values at positions 0..1, 2..3, 4..5,
/// 6..7.  The `shift` parameter (0, 2, 4, or 6) selects which 2-bit field to
/// extract.  `_mm_srli_epi16` shifts 16-bit lanes but the subsequent AND with
/// 0x03 discards any cross-byte contamination.
///
/// # Safety
/// Requires `avx512f` CPU feature.  `shift` must be one of 0, 2, 4, 6.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_2bit_16(raw: __m128i, shift: u32, mask: __m128i) -> __m128i {
    // SAFETY: each branch uses a compile-time const generic for _mm_srli_epi16.
    // The runtime match selects the correct shift amount; cross-byte leakage from
    // the 16-bit shift is always a multiple of 4 and is eliminated by AND 0x03.
    let shifted = match shift {
        0 => raw,
        2 => _mm_srli_epi16::<2>(raw),
        4 => _mm_srli_epi16::<4>(raw),
        _ => _mm_srli_epi16::<6>(raw),
    };
    _mm_and_si128(shifted, mask)
}

impl QuantKernel for Q2_KAvx512 {
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
        "Q2_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 84-byte Q2_K block into 256 FP32 values using AVX-512.
///
/// Processes 2 groups of 128 weights.  Within each group, the same 32 qs
/// bytes are re-used with 4 different shift amounts (0, 2, 4, 6) to extract
/// all four 2-bit fields per byte.  Each sub-block of 16 weights is processed
/// with a single AVX-512 (ZMM, 16-wide) pass instead of AVX2's two 8-wide passes.
///
/// # Safety
/// - `block.len() >= 84`
/// - `output.len() >= 256`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
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

        // Pre-load the 32 qs bytes for this group as two 16-byte halves.
        // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
        let raw_a = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
        let raw_b = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);

        for &shift in &[0u32, 2, 4, 6] {
            // --- Sub-block A: 16 weights from qs[qs_base..qs_base+16] ---
            // One AVX-512 pass handles all 16 values (vs AVX2's two 8-wide passes).
            let sc_byte_a = scales[is];
            is += 1;
            let dl_a = d * (sc_byte_a & 0x0F) as f32;
            let ml_a = dmin * (sc_byte_a >> 4) as f32;
            let va_dl = _mm512_set1_ps(dl_a);
            let va_ml = _mm512_set1_ps(ml_a);

            // SAFETY: extract_2bit_16 requires avx512f (SSE/AVX subset) and shift in {0,2,4,6}.
            let q_bytes_a = extract_2bit_16(raw_a, shift, mask_2bit);

            // Widen 16 unsigned bytes to 16 × i32, then to 16 × f32.
            // SAFETY: _mm512_cvtepu8_epi32 reads from the low 16 bytes of q_bytes_a.
            let q_a_i32 = _mm512_cvtepu8_epi32(q_bytes_a);
            let q_a_f32 = _mm512_cvtepi32_ps(q_a_i32);

            // Apply fmsub: dl * q - ml
            let w_a = _mm512_fmsub_ps(va_dl, q_a_f32, va_ml);

            // SAFETY: out_off + 16 <= 256; output.len() >= 256.
            _mm512_storeu_ps(output.as_mut_ptr().add(out_off), w_a);
            out_off += 16;

            // --- Sub-block B: 16 weights from qs[qs_base+16..qs_base+32] ---
            let sc_byte_b = scales[is];
            is += 1;
            let dl_b = d * (sc_byte_b & 0x0F) as f32;
            let ml_b = dmin * (sc_byte_b >> 4) as f32;
            let vb_dl = _mm512_set1_ps(dl_b);
            let vb_ml = _mm512_set1_ps(ml_b);

            let q_bytes_b = extract_2bit_16(raw_b, shift, mask_2bit);

            let q_b_i32 = _mm512_cvtepu8_epi32(q_bytes_b);
            let q_b_f32 = _mm512_cvtepi32_ps(q_b_i32);

            let w_b = _mm512_fmsub_ps(vb_dl, q_b_f32, vb_ml);

            // SAFETY: out_off + 16 <= 256; output.len() >= 256.
            _mm512_storeu_ps(output.as_mut_ptr().add(out_off), w_b);
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

        let scales = &block[0..16];
        let qs = &block[16..80];

        // SAFETY: block.len() == 84 >= 84.
        let d = f16_to_f32(&block[80..]);
        let dmin = f16_to_f32(&block[82..]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized with AVX-512.
            let mask_2bit = _mm_set1_epi8(0x03);
            let mut block_acc = _mm512_setzero_ps();
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
                    let va_dl = _mm512_set1_ps(dl_a);
                    let va_ml = _mm512_set1_ps(ml_a);

                    let q_bytes_a = extract_2bit_16(raw_a, shift, mask_2bit);
                    let q_a_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q_bytes_a));
                    let w_a = _mm512_fmsub_ps(va_dl, q_a_f32, va_ml);

                    // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                    let i_a = _mm512_loadu_ps(input.as_ptr().add(w_off));
                    block_acc = _mm512_fmadd_ps(w_a, i_a, block_acc);
                    w_off += 16;

                    // --- Sub-block B ---
                    let sc_byte_b = scales[is];
                    is += 1;
                    let dl_b = d * (sc_byte_b & 0x0F) as f32;
                    let ml_b = dmin * (sc_byte_b >> 4) as f32;
                    let vb_dl = _mm512_set1_ps(dl_b);
                    let vb_ml = _mm512_set1_ps(ml_b);

                    let q_bytes_b = extract_2bit_16(raw_b, shift, mask_2bit);
                    let q_b_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q_bytes_b));
                    let w_b = _mm512_fmsub_ps(vb_dl, q_b_f32, vb_ml);

                    // SAFETY: w_off + 16 <= input_offset + BLOCK_SIZE <= n_cols.
                    let i_b = _mm512_loadu_ps(input.as_ptr().add(w_off));
                    block_acc = _mm512_fmadd_ps(w_b, i_b, block_acc);
                    w_off += 16;
                }
            }

            row_sum += hsum_f32_avx512(block_acc);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
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

    // -----------------------------------------------------------------------
    // Dequant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q2k_avx512_dequant_matches_reference_short() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 1 super-block, all zeros: d=0, dmin=0 → all weights = 0.
        let block = make_q2k_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
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
    fn test_q2k_avx512_dequant_matches_reference_long() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 16 super-blocks worth of data, varied pattern.
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 5) & 0xFF) as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 11) & 0xFF) as u8;
        }
        // Build 16 blocks concatenated
        let single = make_q2k_block(0.5, 0.25, &scales, &qs);
        let data: Vec<u8> = single
            .iter()
            .cloned()
            .cycle()
            .take(BLOCK_BYTES * 16)
            .collect();

        let mut out_avx512 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ref = vec![0.0f32; BLOCK_SIZE];

        // Test each block individually — they are all identical here
        Q2_KAvx512
            .dequant_block(&data[0..BLOCK_BYTES], &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
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
    fn test_q2k_avx512_dequant_uniform_scale_no_min() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // d=1.0, dmin=0.0, all scales=0x01 (scale=1, min=0), all qs=0xFF (all 2-bit = 3)
        // Expected weight = 1.0 * 1 * 3 - 0 = 3.0
        let block = make_q2k_block(1.0, 0.0, &[0x01; 16], &[0xFF; 64]);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [uniform_no_min] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q2k_avx512_dequant_with_min() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // d=2.0, dmin=1.0, scales=0x11 (scale=1, min=1), qs=0x00 (all q=0)
        // Weight = 2.0 * 1 * 0 - 1.0 * 1 = -1.0
        let block = make_q2k_block(2.0, 1.0, &[0x11; 16], &[0x00; 64]);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [with_min] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q2k_avx512_dequant_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0x21_u8.wrapping_add(i as u8);
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }

        let block = make_q2k_block(0.5, 0.25, &scales, &qs);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
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
    fn test_q2k_avx512_zero_block_all_zeros() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Zero weights → zero output.
        let block = make_q2k_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let mut out = vec![1.0f32; 256];
        Q2_KAvx512.dequant_block(&block, &mut out).expect("dequant");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "expected 0 at [{i}], got {v}");
        }
    }

    #[test]
    fn test_q2k_avx512_scale_correctness() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // d=1.0, dmin=0.5, all scales=0x12 (scale=2, min=1 → min factor = 0.5*1 = 0.5)
        // All qs=0xAA: 2-bit values at shift 0 → 0b10 = 2
        // Weight = 1.0 * 2 * 2 - 0.5 * 1 = 4.0 - 0.5 = 3.5
        let block = make_q2k_block(1.0, 0.5, &[0x12; 16], &[0xAA; 64]);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q2_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q2KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "scale_correctness mismatch at [{i}]: avx512={a}, ref={r}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // GEMV tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q2k_avx512_matvec_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0x21_u8.wrapping_add(i as u8);
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }
        let block = make_q2k_block(0.5, 0.25, &scales, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Q2KRef
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
    fn test_q2k_avx512_odd_rows_no_panic() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 3 rows (non-power-of-2), 256 cols
        let block = make_q2k_block(0.5, 0.25, &[0x21; 16], &[0xAAu8; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let tensor = QuantTensor::new(data, vec![3, 256], oxillama_gguf::GgufTensorType::Q2K);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 3];
        Q2_KAvx512
            .gemv(&tensor, &input, &mut output)
            .expect("odd rows gemv");
        // All 3 outputs should be identical (same block)
        assert!(
            (output[0] - output[1]).abs() < 1e-5,
            "rows 0 and 1 should match"
        );
        assert!(
            (output[1] - output[2]).abs() < 1e-5,
            "rows 1 and 2 should match"
        );
    }

    #[test]
    fn test_q2k_avx512_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 200 columns — partial block.
        let block = make_q2k_block(1.0, 0.5, &[0x11u8; 16], &[0xAAu8; 64]);
        let tensor_avx512 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv partial");
        Q2KRef
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
    fn test_q2k_avx512_gemv_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv varied");
        Q2KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv varied");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "varied gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q2k_avx512_gemv_multiple_rows() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 4 rows, 256 cols — verify each row independently
        let block0 = make_q2k_block(0.5, 0.1, &[0x21; 16], &[0x55u8; 64]);
        let block1 = make_q2k_block(1.0, 0.0, &[0x01; 16], &[0xFFu8; 64]);
        let block2 = make_q2k_block(0.75, 0.25, &[0x11; 16], &[0xAAu8; 64]);
        let block3 = make_q2k_block(0.25, 0.5, &[0x22; 16], &[0x33u8; 64]);

        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        data.extend_from_slice(&block2);
        data.extend_from_slice(&block3);

        let tensor_avx512 = QuantTensor::new(
            data.clone(),
            vec![4, 256],
            oxillama_gguf::GgufTensorType::Q2K,
        );
        let tensor_ref = QuantTensor::new(data, vec![4, 256], oxillama_gguf::GgufTensorType::Q2K);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 4];
        let mut out_ref = vec![0.0f32; 4];

        Q2_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv 4-rows");
        Q2KRef
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
    fn test_q2k_avx512_block_boundary_alignment() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Test exactly 2 complete blocks (512 elements)
        let block = make_q2k_block(0.5, 0.25, &[0x31; 16], &[0x55u8; 64]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);

        let tensor_avx512 = QuantTensor::new(
            data.clone(),
            vec![1, 512],
            oxillama_gguf::GgufTensorType::Q2K,
        );
        let tensor_ref = QuantTensor::new(data, vec![1, 512], oxillama_gguf::GgufTensorType::Q2K);

        let input = vec![1.0f32; 512];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv 2-blocks");
        Q2KRef
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
    fn test_q2k_dispatcher_routes_to_avx512_when_available() {
        use crate::dispatch::global_dispatcher;
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let kernel = global_dispatcher()
            .get_kernel(oxillama_gguf::GgufTensorType::Q2K)
            .expect("dispatcher Q2K");
        // When AVX-512 is detected, the kernel name should still be Q2_K
        // (both avx512 and reference share the same logical name).
        assert_eq!(kernel.name(), "Q2_K");
    }

    // -----------------------------------------------------------------------
    // Error path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_q2k_avx512_buffer_too_small_block() {
        let block = vec![0u8; 10]; // too small
        let mut output = vec![0.0f32; 256];
        assert!(Q2_KAvx512.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_q2k_avx512_buffer_too_small_output() {
        let block = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; 10]; // too small
        assert!(Q2_KAvx512.dequant_block(&block, &mut output).is_err());
    }

    #[test]
    fn test_q2k_avx512_gemv_empty_input_error() {
        let block = make_q2k_block(1.0, 0.0, &[0x01; 16], &[0; 64]);
        let tensor = make_tensor(block, 256);
        let input = vec![]; // empty — too short
        let mut output = vec![0.0f32; 1];
        assert!(Q2_KAvx512.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_q2k_avx512_kernel_metadata() {
        assert_eq!(Q2_KAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Q2_KAvx512.block_bytes(), BLOCK_BYTES);
        assert_eq!(Q2_KAvx512.name(), "Q2_K");
    }
}
