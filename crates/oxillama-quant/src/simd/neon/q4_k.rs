//! NEON (AArch64) accelerated Q4_K quantization kernel.
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

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K block: 2 (FP16 d) + 2 (FP16 dmin) + 12 (packed scales/mins) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 144;

/// NEON-accelerated Q4_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q4_KNeon;

/// Convert an IEEE 754 FP16 value (as raw u16 LE bytes) to f32.
#[inline(always)]
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// Horizontal sum of a `float32x4_t` register using `vaddvq_f32`.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    // SAFETY: vaddvq_f32 is always valid on AArch64.
    unsafe { vaddvq_f32(v) }
}

/// Decode the 6-bit packed scales and mins from the 12-byte header of a Q4_K block.
///
/// Returns `(scales[8], mins[8])` where each element is a 6-bit unsigned value.
/// Scale unpacking is kept scalar because the bit-manipulation pattern is irregular
/// and does not benefit from SIMD vectorization.
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

/// Widen 4 u8 nibbles (low lane of a uint8x8_t) to f32x4.
///
/// Chain: u8x8 → u16x8 → u32x4 → f32x4.
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn nibbles_low_to_f32(nibbles: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_low_u8(nibbles));
        let u32_4 = vmovl_u16(vget_low_u16(u16_8));
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 nibbles (high lane of a uint8x8_t) to f32x4 (indices 4..7 of the low u8x8).
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn nibbles_mid_to_f32(nibbles: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_high_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_low_u8(nibbles));
        let u32_4 = vmovl_high_u16(u16_8);
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 nibbles (low lane of the high u8x8 half) to f32x4 (indices 8..11).
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn nibbles_hi_low_to_f32(nibbles: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_high_u8(nibbles));
        let u32_4 = vmovl_u16(vget_low_u16(u16_8));
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 nibbles (high lane of the high u8x8 half) to f32x4 (indices 12..15).
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn nibbles_hi_high_to_f32(nibbles: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_high_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_high_u8(nibbles));
        let u32_4 = vmovl_high_u16(u16_8);
        vcvtq_f32_u32(u32_4)
    }
}

/// Compute `a * q - b` for a vector of f32 nibble values.
///
/// Uses FMA pattern: `vfmaq_f32(vnegq_f32(b_vec), a_vec, q_vec)` = `-b + a*q`.
/// SAFETY: all operations are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn scale_nibbles(q: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: vfmaq_f32 and vnegq_f32 are always valid on AArch64.
    unsafe { vfmaq_f32(vnegq_f32(b), a, q) }
}

/// Dequantize one 144-byte Q4_K block into 256 FP32 values using NEON.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES` (144)
/// - `output.len() >= BLOCK_SIZE` (256)
/// - Must be called on AArch64 with NEON
unsafe fn dequant_block_neon(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    let (sc, mn) = decode_scales_mins(&block[4..16]);

    let qs = &block[16..144];
    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    let mut is = 0usize;
    let mut qs_off = 0usize;
    let mut out_off = 0usize;

    for _group in 0..4 {
        // Pre-compute scalar per-sub-block factors.
        let a_lo = d * sc[is] as f32;
        let b_lo = dmin * mn[is] as f32;
        let a_hi = d * sc[is + 1] as f32;
        let b_hi = dmin * mn[is + 1] as f32;

        let va_lo = unsafe { vdupq_n_f32(a_lo) };
        let vb_lo = unsafe { vdupq_n_f32(b_lo) };
        let va_hi = unsafe { vdupq_n_f32(a_hi) };
        let vb_hi = unsafe { vdupq_n_f32(b_hi) };

        // Load 16 nibble bytes for the first half of this group (16 bytes = 16 lo + 16 hi weights).
        // SAFETY: qs_off + 16 <= 128 (4 groups × 32 bytes total).
        let raw_lo = unsafe { vld1q_u8(qs.as_ptr().add(qs_off)) };
        // Load 16 nibble bytes for the second half of this group.
        // SAFETY: qs_off + 32 <= 128.
        let raw_hi = unsafe { vld1q_u8(qs.as_ptr().add(qs_off + 16)) };

        // Extract lo nibbles (0x0F mask) and hi nibbles (shift right 4).
        // SAFETY: vandq_u8 and vshrq_n_u8 are always valid on AArch64.
        let lo0 = unsafe { vandq_u8(raw_lo, mask_lo) };
        let lo1 = unsafe { vandq_u8(raw_hi, mask_lo) };
        let hi0 = unsafe { vshrq_n_u8::<4>(raw_lo) };
        let hi1 = unsafe { vshrq_n_u8::<4>(raw_hi) };

        // --- Lo sub-block: 32 weights = a_lo * q - b_lo ---
        // lo0 provides first 16 lo nibbles, lo1 provides second 16 lo nibbles.
        // Each produces 4 groups of 4 f32s.

        // SAFETY: all widening/conversion intrinsics are valid on AArch64.
        let w0 = unsafe { scale_nibbles(nibbles_low_to_f32(lo0), va_lo, vb_lo) };
        let w1 = unsafe { scale_nibbles(nibbles_mid_to_f32(lo0), va_lo, vb_lo) };
        let w2 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(lo0), va_lo, vb_lo) };
        let w3 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(lo0), va_lo, vb_lo) };

        let w4 = unsafe { scale_nibbles(nibbles_low_to_f32(lo1), va_lo, vb_lo) };
        let w5 = unsafe { scale_nibbles(nibbles_mid_to_f32(lo1), va_lo, vb_lo) };
        let w6 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(lo1), va_lo, vb_lo) };
        let w7 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(lo1), va_lo, vb_lo) };

        // Store 32 lo-sub-block weights.
        // SAFETY: out_off + 32 <= 256; output.len() >= 256.
        let ptr = output.as_mut_ptr().add(out_off);
        unsafe {
            vst1q_f32(ptr, w0);
            vst1q_f32(ptr.add(4), w1);
            vst1q_f32(ptr.add(8), w2);
            vst1q_f32(ptr.add(12), w3);
            vst1q_f32(ptr.add(16), w4);
            vst1q_f32(ptr.add(20), w5);
            vst1q_f32(ptr.add(24), w6);
            vst1q_f32(ptr.add(28), w7);
        }

        // --- Hi sub-block: 32 weights = a_hi * q - b_hi ---
        // SAFETY: all widening/conversion intrinsics are valid on AArch64.
        let wh0 = unsafe { scale_nibbles(nibbles_low_to_f32(hi0), va_hi, vb_hi) };
        let wh1 = unsafe { scale_nibbles(nibbles_mid_to_f32(hi0), va_hi, vb_hi) };
        let wh2 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(hi0), va_hi, vb_hi) };
        let wh3 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(hi0), va_hi, vb_hi) };

        let wh4 = unsafe { scale_nibbles(nibbles_low_to_f32(hi1), va_hi, vb_hi) };
        let wh5 = unsafe { scale_nibbles(nibbles_mid_to_f32(hi1), va_hi, vb_hi) };
        let wh6 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(hi1), va_hi, vb_hi) };
        let wh7 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(hi1), va_hi, vb_hi) };

        // Store 32 hi-sub-block weights.
        // SAFETY: out_off + 32 + 32 <= 256; output.len() >= 256.
        let ptr2 = output.as_mut_ptr().add(out_off + 32);
        unsafe {
            vst1q_f32(ptr2, wh0);
            vst1q_f32(ptr2.add(4), wh1);
            vst1q_f32(ptr2.add(8), wh2);
            vst1q_f32(ptr2.add(12), wh3);
            vst1q_f32(ptr2.add(16), wh4);
            vst1q_f32(ptr2.add(20), wh5);
            vst1q_f32(ptr2.add(24), wh6);
            vst1q_f32(ptr2.add(28), wh7);
        }

        is += 2;
        qs_off += 32;
        out_off += 64;
    }
}

/// Compute the dot product of one row of a Q4_K matrix with an FP32 vector using NEON.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - Must be called on AArch64 with NEON
unsafe fn gemv_row_neon(
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
            // Fast path: all 256 weights in bounds — fully vectorised.
            let qs = &block[16..144];
            let mask_lo = unsafe { vdupq_n_u8(0x0F) };

            let mut block_acc = unsafe { vdupq_n_f32(0.0f32) };
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for _group in 0..4 {
                let a_lo = d * sc[is] as f32;
                let b_lo = dmin * mn[is] as f32;
                let a_hi = d * sc[is + 1] as f32;
                let b_hi = dmin * mn[is + 1] as f32;

                let va_lo = unsafe { vdupq_n_f32(a_lo) };
                let vb_lo = unsafe { vdupq_n_f32(b_lo) };
                let va_hi = unsafe { vdupq_n_f32(a_hi) };
                let vb_hi = unsafe { vdupq_n_f32(b_hi) };

                // Load 32 nibble bytes for this group (16 + 16).
                // SAFETY: qs_off + 32 <= 128; qs.len() == 128.
                let raw_lo = unsafe { vld1q_u8(qs.as_ptr().add(qs_off)) };
                let raw_hi = unsafe { vld1q_u8(qs.as_ptr().add(qs_off + 16)) };

                // Extract lo and hi nibbles.
                // SAFETY: bitwise ops on NEON registers always valid.
                let lo0 = unsafe { vandq_u8(raw_lo, mask_lo) };
                let lo1 = unsafe { vandq_u8(raw_hi, mask_lo) };
                let hi0 = unsafe { vshrq_n_u8::<4>(raw_lo) };
                let hi1 = unsafe { vshrq_n_u8::<4>(raw_hi) };

                // Input pointers for lo and hi sub-blocks.
                // SAFETY: w_off + 32 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let inp_lo = input.as_ptr().add(w_off);
                let inp_hi = input.as_ptr().add(w_off + 32);

                // --- Lo sub-block: 32 weights at inp_lo ---
                // Process in 8 groups of 4.
                // SAFETY: widening/load intrinsics valid on AArch64.
                let i0 = unsafe { vld1q_f32(inp_lo) };
                let i1 = unsafe { vld1q_f32(inp_lo.add(4)) };
                let i2 = unsafe { vld1q_f32(inp_lo.add(8)) };
                let i3 = unsafe { vld1q_f32(inp_lo.add(12)) };
                let i4 = unsafe { vld1q_f32(inp_lo.add(16)) };
                let i5 = unsafe { vld1q_f32(inp_lo.add(20)) };
                let i6 = unsafe { vld1q_f32(inp_lo.add(24)) };
                let i7 = unsafe { vld1q_f32(inp_lo.add(28)) };

                let w0 = unsafe { scale_nibbles(nibbles_low_to_f32(lo0), va_lo, vb_lo) };
                let w1 = unsafe { scale_nibbles(nibbles_mid_to_f32(lo0), va_lo, vb_lo) };
                let w2 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(lo0), va_lo, vb_lo) };
                let w3 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(lo0), va_lo, vb_lo) };
                let w4 = unsafe { scale_nibbles(nibbles_low_to_f32(lo1), va_lo, vb_lo) };
                let w5 = unsafe { scale_nibbles(nibbles_mid_to_f32(lo1), va_lo, vb_lo) };
                let w6 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(lo1), va_lo, vb_lo) };
                let w7 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(lo1), va_lo, vb_lo) };

                // SAFETY: vfmaq_f32 is always valid on AArch64.
                block_acc = unsafe { vfmaq_f32(block_acc, w0, i0) };
                block_acc = unsafe { vfmaq_f32(block_acc, w1, i1) };
                block_acc = unsafe { vfmaq_f32(block_acc, w2, i2) };
                block_acc = unsafe { vfmaq_f32(block_acc, w3, i3) };
                block_acc = unsafe { vfmaq_f32(block_acc, w4, i4) };
                block_acc = unsafe { vfmaq_f32(block_acc, w5, i5) };
                block_acc = unsafe { vfmaq_f32(block_acc, w6, i6) };
                block_acc = unsafe { vfmaq_f32(block_acc, w7, i7) };

                // --- Hi sub-block: 32 weights at inp_hi ---
                // SAFETY: widening/load intrinsics valid on AArch64.
                let i8 = unsafe { vld1q_f32(inp_hi) };
                let i9 = unsafe { vld1q_f32(inp_hi.add(4)) };
                let i10 = unsafe { vld1q_f32(inp_hi.add(8)) };
                let i11 = unsafe { vld1q_f32(inp_hi.add(12)) };
                let i12 = unsafe { vld1q_f32(inp_hi.add(16)) };
                let i13 = unsafe { vld1q_f32(inp_hi.add(20)) };
                let i14 = unsafe { vld1q_f32(inp_hi.add(24)) };
                let i15 = unsafe { vld1q_f32(inp_hi.add(28)) };

                let wh0 = unsafe { scale_nibbles(nibbles_low_to_f32(hi0), va_hi, vb_hi) };
                let wh1 = unsafe { scale_nibbles(nibbles_mid_to_f32(hi0), va_hi, vb_hi) };
                let wh2 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(hi0), va_hi, vb_hi) };
                let wh3 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(hi0), va_hi, vb_hi) };
                let wh4 = unsafe { scale_nibbles(nibbles_low_to_f32(hi1), va_hi, vb_hi) };
                let wh5 = unsafe { scale_nibbles(nibbles_mid_to_f32(hi1), va_hi, vb_hi) };
                let wh6 = unsafe { scale_nibbles(nibbles_hi_low_to_f32(hi1), va_hi, vb_hi) };
                let wh7 = unsafe { scale_nibbles(nibbles_hi_high_to_f32(hi1), va_hi, vb_hi) };

                // SAFETY: vfmaq_f32 is always valid on AArch64.
                block_acc = unsafe { vfmaq_f32(block_acc, wh0, i8) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh1, i9) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh2, i10) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh3, i11) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh4, i12) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh5, i13) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh6, i14) };
                block_acc = unsafe { vfmaq_f32(block_acc, wh7, i15) };

                is += 2;
                qs_off += 32;
                w_off += 64;
            }

            // SAFETY: hsum_f32x4 calls vaddvq_f32 which is valid on AArch64.
            row_sum += unsafe { hsum_f32x4(block_acc) };
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
                        let q = (unsafe { *qs.get_unchecked(qs_off + l) } & 0x0F) as f32;
                        partial_sum += (d1 * q - m1) * input[idx];
                    }
                }
                // Hi nibbles → next 32 weights.
                for l in 0..32 {
                    let idx = w_off + 32 + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128.
                        let q = ((unsafe { *qs.get_unchecked(qs_off + l) } >> 4) & 0x0F) as f32;
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

impl QuantKernel for Q4_KNeon {
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
        // AArch64 always has NEON.
        unsafe { dequant_block_neon(block, output) }
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
            // SAFETY: row/block bounds verified above. AArch64 always has NEON.
            *out = unsafe {
                gemv_row_neon(
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
        "Q4_K_NEON"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
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

        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q4_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q4KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "dequant mismatch at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_all_nibbles_8() {
        // All nibbles = 8 and dmin=0 → weight = d * scale * 8
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x88u8; 128];
        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);

        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q4_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q4KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "dequant max error {max_err}");
    }

    #[test]
    fn test_gemv_matches_reference() {
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);

        let qs = [0x88u8; 128];
        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_neon = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv mismatch: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_uniform_all_ones() {
        // d=1.0, dmin=0.0, all scales=1, all nibbles=1 → weight=1.0, sum=256.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x11u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor = make_tensor(block, 256);
        let input = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 1];

        Q4_KNeon.gemv(&tensor, &input, &mut out).expect("neon gemv");

        assert!(
            (out[0] - 256.0).abs() < 1.0,
            "expected ~256.0, got {}",
            out[0]
        );
    }

    #[test]
    fn test_gemv_partial_block() {
        // 200 columns — one partial block (200 < 256).
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x11u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor_neon = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 0.1,
            "partial gemv mismatch: neon={}, ref={}, err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_multi_row() {
        let n_rows = 3usize;
        let n_cols = 512usize; // 2 full blocks per row

        let mut scales = [0u8; 12];
        scales[..4].fill(2);
        scales[4..8].fill(1);
        scales[8..12].fill(3);
        let qs: Vec<u8> = (0..128u8).collect();
        let qs_arr: [u8; 128] = qs.try_into().expect("slice conversion");

        let block = make_q4_k_block(0.5, 0.05, &scales, &qs_arr);
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let mut data = Vec::new();
        for _ in 0..n_rows {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&block);
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.001).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4K,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4K,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "gemv row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }
}
