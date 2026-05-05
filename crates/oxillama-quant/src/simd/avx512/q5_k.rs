//! AVX-512 accelerated Q5_K quantization kernel.
//!
//! Q5_K block layout (176 bytes per 256 weights):
//! - bytes[0..2]   — FP16 super-block scale `d` (little-endian)
//! - bytes[2..4]   — FP16 super-block minimum `dmin` (little-endian)
//! - bytes[4..16]  — 12 bytes encoding 8 sub-block scales + 8 sub-block mins,
//!   6 bits each, packed (same packing as Q4_K)
//! - bytes[16..48] — 32 bytes `qh` — the high (5th) bit of each 5-bit quant,
//!   bit `j` of byte `qh[l]` is the high bit of weight
//!   (group * 32 + l) in the lo sub-block (j < 4) or
//!   (group * 32 + l) in the hi sub-block (j >= 4).
//! - bytes[48..176] — 128 packed nibble bytes (256 × 4-bit unsigned lo values)
//!
//! Block structure: 8 sub-blocks of 32 weights each (4 groups of 2 sub-blocks).
//!
//! Weight formula: `w = d * scale_i * q5 - dmin * min_i`
//! where `q5 = nibble | (high_bit << 4)` (range 0..31).
//!
//! ## AVX-512 strategy
//!
//! For each of the 4 groups (64 weights: 32 lo-sub + 32 hi-sub), use **two**
//! AVX-512 (16-wide) passes per 32-weight sub-block instead of AVX2's four
//! 8-wide passes:
//!
//! 1. Load 16 nibble bytes, extract lo/hi nibbles.
//! 2. Load qh bytes (128-bit), broadcast bit-position mask, AND + cmpeq to
//!    produce `0x10` or `0x00` per byte, then OR with nibbles.
//! 3. Widen with `_mm512_cvtepu8_epi32` (16 u8 → 16 i32).
//! 4. Apply `_mm512_fmsub_ps(va, q_f32, vb)`.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q5_K block: 2+2+12 (header) + 32 (qh) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 176;

/// AVX-512 accelerated Q5_K kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q5_KAvx512;

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

/// Extract the high (5th) bit from a 16-byte slice of `qh` at the given
/// `bit_pos`, producing a `__m128i` where each byte is `0x10` (bit set) or
/// `0x00` (bit clear).
///
/// Uses a compare-equal pattern (not shift) to avoid cross-byte contamination.
///
/// # Safety
/// Requires AVX-512F (SSE/AVX subset).  `bit_pos` must be in 0..8.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_high_bit(qh_half: __m128i, bit_pos: u32) -> __m128i {
    // SAFETY: bit_pos in 0..8 guaranteed by caller; 1u8 << bit_pos is valid.
    let mask_byte = (1u8 << bit_pos) as i8;
    let mask_vec = _mm_set1_epi8(mask_byte);

    // Isolate the target bit in each byte.
    let masked = _mm_and_si128(qh_half, mask_vec);

    // masked == 0 ⟹ zero vector (bit clear), masked ≠ 0 ⟹ all-ones (bit set).
    let is_zero = _mm_cmpeq_epi8(masked, _mm_setzero_si128());

    // _mm_andnot_si128(a, b) = (!a) & b.
    // is_zero=0xFF (bit clear) → 0x00; is_zero=0x00 (bit set) → 0x10.
    _mm_andnot_si128(is_zero, _mm_set1_epi8(0x10_u8 as i8))
}

impl QuantKernel for Q5_KAvx512 {
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
        "Q5_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 176-byte Q5_K block into 256 FP32 values using AVX-512.
///
/// # Safety
/// - `block.len() >= 176`
/// - `output.len() >= 256`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 176 >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    let (sc, mn) = decode_scales_mins(&block[4..16]);

    // qh: 32 bytes at offset 16, covering all 256 weight positions.
    let qh = &block[16..48];
    // qs: 128 nibble bytes at offset 48.
    let qs = &block[48..176];

    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

    // Load qh halves once (shared across all 4 groups).
    // SAFETY: qh.len() == 32.
    let qh_0 = _mm_loadu_si128(qh.as_ptr() as *const __m128i); // bytes 0..15
    let qh_1 = _mm_loadu_si128(qh.as_ptr().add(16) as *const __m128i); // bytes 16..31

    let mut is = 0usize;
    let mut qs_off = 0usize;
    let mut out_off = 0usize;

    for group in 0..4u32 {
        let a_lo = d * sc[is] as f32;
        let b_lo = dmin * mn[is] as f32;
        let a_hi = d * sc[is + 1] as f32;
        let b_hi = dmin * mn[is + 1] as f32;

        let va_lo = _mm512_set1_ps(a_lo);
        let vb_lo = _mm512_set1_ps(b_lo);
        let va_hi = _mm512_set1_ps(a_hi);
        let vb_hi = _mm512_set1_ps(b_hi);

        // Load 32 nibble bytes for this group as two 16-byte chunks.
        // SAFETY: qs_off + 32 <= 128; qs.len() == 128.
        let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
        let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

        // Extract lo nibbles (bits 3:0) — weights for lo sub-block.
        let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo); // bytes 0..15 lo nibbles
        let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo); // bytes 16..31 lo nibbles

        // Extract hi nibbles (bits 7:4) — weights for hi sub-block.
        let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
        let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

        // High bits for lo sub-block: bit `group` of qh (positions 0..31).
        let hb_lo_0 = extract_high_bit(qh_0, group); // positions 0..15
        let hb_lo_1 = extract_high_bit(qh_1, group); // positions 16..31

        // High bits for hi sub-block: bit `group+4` of qh.
        let hb_hi_0 = extract_high_bit(qh_0, group + 4); // positions 0..15
        let hb_hi_1 = extract_high_bit(qh_1, group + 4); // positions 16..31

        // --- Lo sub-block: 32 weights = a_lo * q5 - b_lo ---
        // Two AVX-512 passes of 16 instead of AVX2's four passes of 8.

        // OR nibbles with high bits to form q5 (0..31) in each byte.
        let q5_lo_0 = _mm_or_si128(lo_nibbles_0, hb_lo_0); // positions 0..15
        let q5_lo_1 = _mm_or_si128(lo_nibbles_1, hb_lo_1); // positions 16..31

        // First 16 q5 bytes → 16 × i32 → 16 × f32.
        // SAFETY: _mm512_cvtepu8_epi32 reads the low 16 bytes of the __m128i.
        let q0_i32 = _mm512_cvtepu8_epi32(q5_lo_0);
        let q0_f32 = _mm512_cvtepi32_ps(q0_i32);
        let w0 = _mm512_fmsub_ps(va_lo, q0_f32, vb_lo); // a_lo * q5 - b_lo

        // Second 16 q5 bytes.
        let q1_i32 = _mm512_cvtepu8_epi32(q5_lo_1);
        let q1_f32 = _mm512_cvtepi32_ps(q1_i32);
        let w1 = _mm512_fmsub_ps(va_lo, q1_f32, vb_lo);

        // Store 32 lo-sub-block weights.
        // SAFETY: out_off + 32 <= 256; output.len() >= 256.
        let ptr = output.as_mut_ptr().add(out_off);
        _mm512_storeu_ps(ptr, w0);
        _mm512_storeu_ps(ptr.add(16), w1);

        // --- Hi sub-block: 32 weights = a_hi * q5 - b_hi ---

        let q5_hi_0 = _mm_or_si128(hi_nibbles_0, hb_hi_0);
        let q5_hi_1 = _mm_or_si128(hi_nibbles_1, hb_hi_1);

        let q2_i32 = _mm512_cvtepu8_epi32(q5_hi_0);
        let q2_f32 = _mm512_cvtepi32_ps(q2_i32);
        let w2 = _mm512_fmsub_ps(va_hi, q2_f32, vb_hi);

        let q3_i32 = _mm512_cvtepu8_epi32(q5_hi_1);
        let q3_f32 = _mm512_cvtepi32_ps(q3_i32);
        let w3 = _mm512_fmsub_ps(va_hi, q3_f32, vb_hi);

        // Store 32 hi-sub-block weights.
        // SAFETY: out_off + 32 + 32 <= 256; output.len() >= 256.
        let ptr2 = output.as_mut_ptr().add(out_off + 32);
        _mm512_storeu_ps(ptr2, w2);
        _mm512_storeu_ps(ptr2.add(16), w3);

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

        // SAFETY: block.len() == 176 >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);
        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.
            let qh = &block[16..48];
            let qs = &block[48..176];
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

            // Load qh halves once for this block.
            // SAFETY: qh.len() == 32.
            let qh_0 = _mm_loadu_si128(qh.as_ptr() as *const __m128i);
            let qh_1 = _mm_loadu_si128(qh.as_ptr().add(16) as *const __m128i);

            let mut block_acc = _mm512_setzero_ps();
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for group in 0..4u32 {
                let a_lo = d * sc[is] as f32;
                let b_lo = dmin * mn[is] as f32;
                let a_hi = d * sc[is + 1] as f32;
                let b_hi = dmin * mn[is + 1] as f32;

                let va_lo = _mm512_set1_ps(a_lo);
                let vb_lo = _mm512_set1_ps(b_lo);
                let va_hi = _mm512_set1_ps(a_hi);
                let vb_hi = _mm512_set1_ps(b_hi);

                // SAFETY: qs_off + 32 <= 128.
                let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

                let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo);
                let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo);
                let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
                let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

                let hb_lo_0 = extract_high_bit(qh_0, group);
                let hb_lo_1 = extract_high_bit(qh_1, group);
                let hb_hi_0 = extract_high_bit(qh_0, group + 4);
                let hb_hi_1 = extract_high_bit(qh_1, group + 4);

                // SAFETY: w_off + 64 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let inp_lo = input.as_ptr().add(w_off);
                let inp_hi = input.as_ptr().add(w_off + 32);

                // Lo sub-block: first 16 q5 values.
                let q5_0 = _mm_or_si128(lo_nibbles_0, hb_lo_0);
                let q0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_0));
                let w0 = _mm512_fmsub_ps(va_lo, q0, vb_lo);
                // SAFETY: inp_lo points to 16 valid floats (w_off + 16 <= input.len()).
                block_acc = _mm512_fmadd_ps(w0, _mm512_loadu_ps(inp_lo), block_acc);

                // Lo sub-block: second 16 q5 values.
                let q5_1 = _mm_or_si128(lo_nibbles_1, hb_lo_1);
                let q1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_1));
                let w1 = _mm512_fmsub_ps(va_lo, q1, vb_lo);
                // SAFETY: inp_lo + 16 points to 16 valid floats.
                block_acc = _mm512_fmadd_ps(w1, _mm512_loadu_ps(inp_lo.add(16)), block_acc);

                // Hi sub-block: first 16 q5 values.
                let q5_2 = _mm_or_si128(hi_nibbles_0, hb_hi_0);
                let q2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_2));
                let w2 = _mm512_fmsub_ps(va_hi, q2, vb_hi);
                // SAFETY: inp_hi points to 16 valid floats (w_off + 32 + 16 <= input.len()).
                block_acc = _mm512_fmadd_ps(w2, _mm512_loadu_ps(inp_hi), block_acc);

                // Hi sub-block: second 16 q5 values.
                let q5_3 = _mm_or_si128(hi_nibbles_1, hb_hi_1);
                let q3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(q5_3));
                let w3 = _mm512_fmsub_ps(va_hi, q3, vb_hi);
                // SAFETY: inp_hi + 16 points to 16 valid floats.
                block_acc = _mm512_fmadd_ps(w3, _mm512_loadu_ps(inp_hi.add(16)), block_acc);

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            row_sum += hsum_f32_avx512(block_acc);
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

                // Lo nibbles → first 32 weights of this group.
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

                // Hi nibbles → next 32 weights.
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
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_dequant_matches_reference_zero_high_bits() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x00u8; 32];
        let qs = [0x88u8; 128]; // lo=8, hi=8

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [zero-high-bits] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_dequant_matches_reference_all_high_bits() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xFFu8; 32]; // all high bits set
        let qs = [0x00u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch [all-high-bits] at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_dequant_matches_reference_varied() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Q5KRef
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xAAu8; 32]; // alternating bits
        let qs = [0x5Au8; 128];

        let block = make_q5k_block(0.5, 0.1, &scales, &qh, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-2,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 200 columns — partial block.
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x55u8; 32];
        let qs = [0x11u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv partial");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-2,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_q5k_avx512_gemv_varied_data() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv varied");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv varied");

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 1e-2,
            "varied gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }
}
