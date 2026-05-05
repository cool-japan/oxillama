//! AVX-512 accelerated Q4_K quantization kernel.
//!
//! Q4_K block layout (144 bytes per 256 weights):
//! - bytes[0..2]   — FP16 super-block scale `d` (little-endian)
//! - bytes[2..4]   — FP16 super-block minimum `dmin` (little-endian)
//! - bytes[4..16]  — 12 bytes encoding 8 sub-block scales + 8 sub-block mins,
//!   6 bits each, packed (see `decode_scales_mins`)
//! - bytes[16..144] — 128 packed nibble bytes (256 × 4-bit unsigned values)
//!
//! Block structure: 8 sub-blocks of 32 weights each (4 groups of 2 sub-blocks).
//!
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 4-bit (0..15).
//! The nibble layout separates lo nibbles (first 32 weights of the group) from hi
//! nibbles (second 32 weights), unlike Q4_0 which interleaves them per byte.
//!
//! ## AVX-512 strategy
//!
//! For each of the 8 sub-blocks (32 weights each), use **two** AVX-512 (16-wide)
//! passes instead of the AVX2 kernel's four 8-wide passes:
//!
//! 1. Load 16 nibble bytes for this half of the sub-block.
//! 2. Extract lo/hi nibbles, widen to 16 × i32 with `_mm512_cvtepu8_epi32`.
//! 3. Apply `_mm512_fmsub_ps(va, q_f32, vb)` where `va = d * scale`, `vb = d * min`.

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K block: 2 (FP16 d) + 2 (FP16 dmin) + 12 (packed scales/mins) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 144;

/// AVX-512 accelerated Q4_K kernel.
///
/// Requires the `avx512f` CPU feature.  The [`crate::dispatch::KernelDispatcher`]
/// checks for this at runtime before constructing this kernel.
#[allow(non_camel_case_types)]
pub struct Q4_KAvx512;

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

impl QuantKernel for Q4_KAvx512 {
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
        "Q4_K"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one 144-byte Q4_K block into 256 FP32 values using AVX-512.
///
/// # Safety
/// - `block.len() >= 144`
/// - `output.len() >= 256`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    // Read super-block FP16 scales.
    // SAFETY: block.len() >= 144 >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    // Decode the 12-byte packed scale/min header (scalar — irregular bit layout).
    let scales_raw = &block[4..16];
    let (sc, mn) = decode_scales_mins(scales_raw);

    // Nibble data: 128 bytes starting at offset 16.
    let qs = &block[16..144];
    let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);

    // Process 4 groups; each group has 32 lo-nibble weights (sub-block `is`) and
    // 32 hi-nibble weights (sub-block `is+1`).  Each group uses 32 nibble bytes.
    let mut is = 0usize;
    let mut qs_off = 0usize;
    let mut out_off = 0usize;

    for _group in 0..4 {
        // Pre-compute scalar per-sub-block factors.
        let a_lo = d * sc[is] as f32; // d * scale for lo sub-block
        let b_lo = dmin * mn[is] as f32; // dmin * min for lo sub-block
        let a_hi = d * sc[is + 1] as f32; // d * scale for hi sub-block
        let b_hi = dmin * mn[is + 1] as f32; // dmin * min for hi sub-block

        let va_lo = _mm512_set1_ps(a_lo);
        let vb_lo = _mm512_set1_ps(b_lo);
        let va_hi = _mm512_set1_ps(a_hi);
        let vb_hi = _mm512_set1_ps(b_hi);

        // Load the 32 nibble bytes for this group as two 16-byte chunks.
        // SAFETY: qs_off + 32 <= 128 (4 groups × 32 bytes).
        let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
        let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

        // Extract lo nibbles (bits 3:0) — weights for lo sub-block.
        let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo); // bytes 0..15 lo nibbles
        let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo); // bytes 16..31 lo nibbles

        // Extract hi nibbles (bits 7:4) — weights for hi sub-block.
        let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
        let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

        // --- Lo sub-block: 32 weights = a_lo * q - b_lo ---
        // Two AVX-512 passes of 16 instead of AVX2's four passes of 8.

        // First 16 lo nibbles → f32, apply fmsub.
        // SAFETY: _mm512_cvtepu8_epi32 reads 16 bytes from the 128-bit source.
        let q0_i32 = _mm512_cvtepu8_epi32(lo_nibbles_0);
        let q0_f32 = _mm512_cvtepi32_ps(q0_i32);
        let w0 = _mm512_fmsub_ps(va_lo, q0_f32, vb_lo); // a_lo * q - b_lo

        // Second 16 lo nibbles.
        let q1_i32 = _mm512_cvtepu8_epi32(lo_nibbles_1);
        let q1_f32 = _mm512_cvtepi32_ps(q1_i32);
        let w1 = _mm512_fmsub_ps(va_lo, q1_f32, vb_lo);

        // Store 32 lo-sub-block weights.
        // SAFETY: out_off + 32 <= 256; output.len() >= 256.
        let ptr = output.as_mut_ptr().add(out_off);
        _mm512_storeu_ps(ptr, w0);
        _mm512_storeu_ps(ptr.add(16), w1);

        // --- Hi sub-block: 32 weights = a_hi * q - b_hi ---

        let q2_i32 = _mm512_cvtepu8_epi32(hi_nibbles_0);
        let q2_f32 = _mm512_cvtepi32_ps(q2_i32);
        let w2 = _mm512_fmsub_ps(va_hi, q2_f32, vb_hi);

        let q3_i32 = _mm512_cvtepu8_epi32(hi_nibbles_1);
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

/// Compute the dot product of one row of a Q4_K matrix with an FP32 vector.
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

        // Read FP16 super-block scales.
        // SAFETY: block.len() == 144 >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);

        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorized.
            let mask_lo = _mm_set1_epi8(0x0F_u8 as i8);
            let qs = &block[16..144];

            let mut block_acc = _mm512_setzero_ps();
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for _group in 0..4 {
                let a_lo = d * sc[is] as f32;
                let b_lo = dmin * mn[is] as f32;
                let a_hi = d * sc[is + 1] as f32;
                let b_hi = dmin * mn[is + 1] as f32;

                let va_lo = _mm512_set1_ps(a_lo);
                let vb_lo = _mm512_set1_ps(b_lo);
                let va_hi = _mm512_set1_ps(a_hi);
                let vb_hi = _mm512_set1_ps(b_hi);

                // Load 32 nibble bytes for this group.
                // SAFETY: qs_off + 32 <= 128; qs.len() == 128.
                let raw_lo = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let raw_hi = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);

                let lo_nibbles_0 = _mm_and_si128(raw_lo, mask_lo);
                let lo_nibbles_1 = _mm_and_si128(raw_hi, mask_lo);
                let hi_nibbles_0 = _mm_and_si128(_mm_srli_epi16(raw_lo, 4), mask_lo);
                let hi_nibbles_1 = _mm_and_si128(_mm_srli_epi16(raw_hi, 4), mask_lo);

                // SAFETY: w_off + 64 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let inp_ptr_lo = input.as_ptr().add(w_off);
                let inp_ptr_hi = input.as_ptr().add(w_off + 32);

                // Lo sub-block: first 16 nibbles.
                let q0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_nibbles_0));
                let i0 = _mm512_loadu_ps(inp_ptr_lo);
                let w0 = _mm512_fmsub_ps(va_lo, q0, vb_lo);
                block_acc = _mm512_fmadd_ps(w0, i0, block_acc);

                // Lo sub-block: second 16 nibbles.
                let q1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_nibbles_1));
                let i1 = _mm512_loadu_ps(inp_ptr_lo.add(16));
                let w1 = _mm512_fmsub_ps(va_lo, q1, vb_lo);
                block_acc = _mm512_fmadd_ps(w1, i1, block_acc);

                // Hi sub-block: first 16 nibbles.
                let q2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_nibbles_0));
                let i2 = _mm512_loadu_ps(inp_ptr_hi);
                let w2 = _mm512_fmsub_ps(va_hi, q2, vb_hi);
                block_acc = _mm512_fmadd_ps(w2, i2, block_acc);

                // Hi sub-block: second 16 nibbles.
                let q3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_nibbles_1));
                let i3 = _mm512_loadu_ps(inp_ptr_hi.add(16));
                let w3 = _mm512_fmsub_ps(va_hi, q3, vb_hi);
                block_acc = _mm512_fmadd_ps(w3, i3, block_acc);

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            row_sum += hsum_f32_avx512(block_acc);
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

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
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

        // Alternating nibble pattern: lo=5, hi=10 → byte 0xA5
        let qs = [0xA5u8; 128];

        let block = make_q4_k_block(0.5, 0.1, &scales, &qs);

        let mut out_avx512 = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q4_KAvx512.dequant_block(&block, &mut out_avx512).unwrap();
        Q4KRef.dequant_block(&block, &mut out_ref).unwrap();

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch at index {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut scales = [0u8; 12];
        scales[..4].fill(1); // sub-blocks 0..3 scale=1
        scales[8..12].fill(1); // sub-blocks 4..7 scale=1 (lo nibble of bytes 8..11)

        // All nibbles = 8: lo=8, hi=8 → byte 0x88
        let qs = [0x88u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q4KRef.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_partial_block() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 200 columns — one partial block (200 < 256).
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x11u8; 128]; // all weights = 1 (lo=1, hi=1)

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_avx512 = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_avx512 = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .unwrap();
        Q4KRef.gemv(&tensor_ref, &input, &mut out_ref).unwrap();

        assert!(
            (out_avx512[0] - out_ref[0]).abs() < 0.1,
            "partial gemv mismatch: avx512={}, ref={}",
            out_avx512[0],
            out_ref[0]
        );
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_gemv_uniform_all_ones() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
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

        Q4_KAvx512.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - 256.0).abs() < 1.0,
            "expected ~256.0, got {}",
            out[0]
        );
    }
}
