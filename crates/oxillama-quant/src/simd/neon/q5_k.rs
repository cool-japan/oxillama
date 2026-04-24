//! NEON (AArch64) accelerated Q5_K quantization kernel.
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

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q5_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q5_K block: 2 (FP16 d) + 2 (FP16 dmin) + 12 (packed scales/mins) + 32 (qh) + 128 (nibbles).
pub const BLOCK_BYTES: usize = 176;

/// NEON-accelerated Q5_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q5_KNeon;

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

/// Decode the 6-bit packed scales and mins from the 12-byte header.
///
/// Returns `(scales[8], mins[8])` where each element is a 6-bit unsigned value.
/// Identical packing to Q4_K.
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

/// Widen 4 u8 values (low lane of a uint8x8_t) to f32x4.
///
/// Chain: u8x8 → u16x8 → u32x4 → f32x4.
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn widen_low_to_f32(v: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_low_u8(v));
        let u32_4 = vmovl_u16(vget_low_u16(u16_8));
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 values (indices 4..7 of the low u8x8 half) to f32x4.
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn widen_mid_to_f32(v: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_high_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_low_u8(v));
        let u32_4 = vmovl_high_u16(u16_8);
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 values (low lane of the high u8x8 half) to f32x4 (indices 8..11).
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn widen_hi_low_to_f32(v: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_high_u8(v));
        let u32_4 = vmovl_u16(vget_low_u16(u16_8));
        vcvtq_f32_u32(u32_4)
    }
}

/// Widen 4 u8 values (high lane of the high u8x8 half) to f32x4 (indices 12..15).
///
/// SAFETY: all widening intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn widen_hi_high_to_f32(v: uint8x16_t) -> float32x4_t {
    // SAFETY: vmovl_u8 / vmovl_high_u16 / vcvtq_f32_u32 are always valid on AArch64.
    unsafe {
        let u16_8 = vmovl_u8(vget_high_u8(v));
        let u32_4 = vmovl_high_u16(u16_8);
        vcvtq_f32_u32(u32_4)
    }
}

/// Compute `a * q5 - b` for a vector of f32 quantized values.
///
/// Uses FMA pattern: `vfmaq_f32(vnegq_f32(b_vec), a_vec, q_vec)` = `-b + a*q`.
/// SAFETY: all operations are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn scale_nibbles_q5(q: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: vfmaq_f32 and vnegq_f32 are always valid on AArch64.
    unsafe { vfmaq_f32(vnegq_f32(b), a, q) }
}

/// Extract the high (5th) bit for 16 weight positions from `qh` at a given bit position.
///
/// Returns a `uint8x16_t` where each byte is either `0x10` (bit set) or `0x00` (bit clear),
/// ready to be OR-ed into the 4-bit nibble to form a 5-bit value (0..31).
///
/// # Safety
/// - Must be called on AArch64 with NEON.
/// - `bit_pos` must be in 0..8.
#[inline(always)]
unsafe fn extract_high_bit_neon(qh_half: uint8x16_t, bit_pos: u32) -> uint8x16_t {
    // SAFETY: bit_pos is in 0..8 guaranteed by caller; operations are valid on AArch64.
    unsafe {
        // Create a mask with only the target bit set in each byte.
        let mask = vdupq_n_u8(1u8 << bit_pos);

        // Isolate the target bit in each byte.
        let masked = vandq_u8(qh_half, mask);

        // Compare: masked != 0 means the bit is set.
        let zero = vdupq_n_u8(0);
        let is_nonzero = vcgtq_u8(masked, zero); // 0xFF if bit set, 0x00 if clear

        // Select 0x10 (bit set) or 0x00 (bit clear).
        vbslq_u8(is_nonzero, vdupq_n_u8(0x10), zero)
    }
}

/// OR high bits into 4-bit nibbles to produce 5-bit values, then widen to f32x4.
///
/// Takes a `uint8x16_t` of nibbles (0..15) and a `uint8x16_t` of high bits
/// (each byte 0x00 or 0x10), ORs them together (0..31), and returns 4 groups
/// of 4 f32 values via widening.
///
/// # Safety
/// Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn q5_widen_low(nibbles: uint8x16_t, high: uint8x16_t) -> float32x4_t {
    // SAFETY: vorrq_u8 is always valid; widen_low_to_f32 returns valid f32x4.
    unsafe {
        let q5 = vorrq_u8(nibbles, high);
        widen_low_to_f32(q5)
    }
}

#[inline(always)]
unsafe fn q5_widen_mid(nibbles: uint8x16_t, high: uint8x16_t) -> float32x4_t {
    // SAFETY: vorrq_u8 is always valid; widen_mid_to_f32 returns valid f32x4.
    unsafe {
        let q5 = vorrq_u8(nibbles, high);
        widen_mid_to_f32(q5)
    }
}

#[inline(always)]
unsafe fn q5_widen_hi_low(nibbles: uint8x16_t, high: uint8x16_t) -> float32x4_t {
    // SAFETY: vorrq_u8 is always valid; widen_hi_low_to_f32 returns valid f32x4.
    unsafe {
        let q5 = vorrq_u8(nibbles, high);
        widen_hi_low_to_f32(q5)
    }
}

#[inline(always)]
unsafe fn q5_widen_hi_high(nibbles: uint8x16_t, high: uint8x16_t) -> float32x4_t {
    // SAFETY: vorrq_u8 is always valid; widen_hi_high_to_f32 returns valid f32x4.
    unsafe {
        let q5 = vorrq_u8(nibbles, high);
        widen_hi_high_to_f32(q5)
    }
}

/// Dequantize one 176-byte Q5_K block into 256 FP32 values using NEON.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES` (176)
/// - `output.len() >= BLOCK_SIZE` (256)
/// - Must be called on AArch64 with NEON
unsafe fn dequant_block_neon(block: &[u8], output: &mut [f32]) {
    // SAFETY: block.len() >= 176 >= 4.
    let d = f16_to_f32(block);
    let dmin = f16_to_f32(&block[2..]);

    let (sc, mn) = decode_scales_mins(&block[4..16]);

    let qh = &block[16..48];
    let qs = &block[48..176];

    // SAFETY: vandq_u8 / vshrq_n_u8 / vld1q_u8 are always valid on AArch64.
    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    // Load all 32 bytes of qh as two 16-byte vectors.
    // SAFETY: qh.len() == 32.
    let qh_0 = unsafe { vld1q_u8(qh.as_ptr()) }; // bytes 0..15
    let qh_1 = unsafe { vld1q_u8(qh.as_ptr().add(16)) }; // bytes 16..31

    let mut is = 0usize;
    let mut qs_off = 0usize;
    let mut out_off = 0usize;

    for group in 0..4u32 {
        // Pre-compute scalar per-sub-block factors.
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

        // Extract high bits for lo sub-block (bit = group) and hi sub-block (bit = group + 4).
        // SAFETY: group is 0..3, so group is in 0..8 and group+4 is in 4..8.
        let hb_lo_0 = unsafe { extract_high_bit_neon(qh_0, group) }; // positions 0..15
        let hb_lo_1 = unsafe { extract_high_bit_neon(qh_1, group) }; // positions 16..31
        let hb_hi_0 = unsafe { extract_high_bit_neon(qh_0, group + 4) }; // positions 0..15
        let hb_hi_1 = unsafe { extract_high_bit_neon(qh_1, group + 4) }; // positions 16..31

        // --- Lo sub-block: 32 weights = a_lo * q5 - b_lo ---
        // lo0 has 16 nibbles at positions 0..15, lo1 has positions 16..31.
        // Each is OR-ed with high bits, widened to f32, then scaled.

        // SAFETY: q5_widen_* + scale_nibbles_q5 are valid on AArch64.
        let w0 = unsafe { scale_nibbles_q5(q5_widen_low(lo0, hb_lo_0), va_lo, vb_lo) };
        let w1 = unsafe { scale_nibbles_q5(q5_widen_mid(lo0, hb_lo_0), va_lo, vb_lo) };
        let w2 = unsafe { scale_nibbles_q5(q5_widen_hi_low(lo0, hb_lo_0), va_lo, vb_lo) };
        let w3 = unsafe { scale_nibbles_q5(q5_widen_hi_high(lo0, hb_lo_0), va_lo, vb_lo) };

        let w4 = unsafe { scale_nibbles_q5(q5_widen_low(lo1, hb_lo_1), va_lo, vb_lo) };
        let w5 = unsafe { scale_nibbles_q5(q5_widen_mid(lo1, hb_lo_1), va_lo, vb_lo) };
        let w6 = unsafe { scale_nibbles_q5(q5_widen_hi_low(lo1, hb_lo_1), va_lo, vb_lo) };
        let w7 = unsafe { scale_nibbles_q5(q5_widen_hi_high(lo1, hb_lo_1), va_lo, vb_lo) };

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

        // --- Hi sub-block: 32 weights = a_hi * q5 - b_hi ---
        // SAFETY: q5_widen_* + scale_nibbles_q5 are valid on AArch64.
        let wh0 = unsafe { scale_nibbles_q5(q5_widen_low(hi0, hb_hi_0), va_hi, vb_hi) };
        let wh1 = unsafe { scale_nibbles_q5(q5_widen_mid(hi0, hb_hi_0), va_hi, vb_hi) };
        let wh2 = unsafe { scale_nibbles_q5(q5_widen_hi_low(hi0, hb_hi_0), va_hi, vb_hi) };
        let wh3 = unsafe { scale_nibbles_q5(q5_widen_hi_high(hi0, hb_hi_0), va_hi, vb_hi) };

        let wh4 = unsafe { scale_nibbles_q5(q5_widen_low(hi1, hb_hi_1), va_hi, vb_hi) };
        let wh5 = unsafe { scale_nibbles_q5(q5_widen_mid(hi1, hb_hi_1), va_hi, vb_hi) };
        let wh6 = unsafe { scale_nibbles_q5(q5_widen_hi_low(hi1, hb_hi_1), va_hi, vb_hi) };
        let wh7 = unsafe { scale_nibbles_q5(q5_widen_hi_high(hi1, hb_hi_1), va_hi, vb_hi) };

        // Store 32 hi-sub-block weights.
        // SAFETY: out_off + 64 <= 256; output.len() >= 256.
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

/// Compute the dot product of one row of a Q5_K matrix with an FP32 vector using NEON.
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
        // SAFETY: block.len() == 176 >= 4.
        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);

        let (sc, mn) = decode_scales_mins(&block[4..16]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorised.
            let qh = &block[16..48];
            let qs = &block[48..176];
            let mask_lo = unsafe { vdupq_n_u8(0x0F) };

            // Load all 32 bytes of qh as two 16-byte vectors.
            // SAFETY: qh.len() == 32.
            let qh_0 = unsafe { vld1q_u8(qh.as_ptr()) };
            let qh_1 = unsafe { vld1q_u8(qh.as_ptr().add(16)) };

            let mut block_acc = unsafe { vdupq_n_f32(0.0f32) };
            let mut is = 0usize;
            let mut qs_off = 0usize;
            let mut w_off = input_offset;

            for group in 0..4u32 {
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

                // Extract high bits for lo and hi sub-blocks.
                // SAFETY: group is 0..3, so bit positions are in 0..8.
                let hb_lo_0 = unsafe { extract_high_bit_neon(qh_0, group) };
                let hb_lo_1 = unsafe { extract_high_bit_neon(qh_1, group) };
                let hb_hi_0 = unsafe { extract_high_bit_neon(qh_0, group + 4) };
                let hb_hi_1 = unsafe { extract_high_bit_neon(qh_1, group + 4) };

                // Input pointers for lo and hi sub-blocks.
                // SAFETY: w_off + 64 <= input_offset + BLOCK_SIZE <= n_cols <= input.len().
                let inp_lo = input.as_ptr().add(w_off);
                let inp_hi = input.as_ptr().add(w_off + 32);

                // --- Lo sub-block: 32 weights at inp_lo ---
                // Process in 8 groups of 4 f32 values.
                // SAFETY: widening/load intrinsics valid on AArch64.
                let i0 = unsafe { vld1q_f32(inp_lo) };
                let i1 = unsafe { vld1q_f32(inp_lo.add(4)) };
                let i2 = unsafe { vld1q_f32(inp_lo.add(8)) };
                let i3 = unsafe { vld1q_f32(inp_lo.add(12)) };
                let i4 = unsafe { vld1q_f32(inp_lo.add(16)) };
                let i5 = unsafe { vld1q_f32(inp_lo.add(20)) };
                let i6 = unsafe { vld1q_f32(inp_lo.add(24)) };
                let i7 = unsafe { vld1q_f32(inp_lo.add(28)) };

                let w0 = unsafe { scale_nibbles_q5(q5_widen_low(lo0, hb_lo_0), va_lo, vb_lo) };
                let w1 = unsafe { scale_nibbles_q5(q5_widen_mid(lo0, hb_lo_0), va_lo, vb_lo) };
                let w2 = unsafe { scale_nibbles_q5(q5_widen_hi_low(lo0, hb_lo_0), va_lo, vb_lo) };
                let w3 = unsafe { scale_nibbles_q5(q5_widen_hi_high(lo0, hb_lo_0), va_lo, vb_lo) };
                let w4 = unsafe { scale_nibbles_q5(q5_widen_low(lo1, hb_lo_1), va_lo, vb_lo) };
                let w5 = unsafe { scale_nibbles_q5(q5_widen_mid(lo1, hb_lo_1), va_lo, vb_lo) };
                let w6 = unsafe { scale_nibbles_q5(q5_widen_hi_low(lo1, hb_lo_1), va_lo, vb_lo) };
                let w7 = unsafe { scale_nibbles_q5(q5_widen_hi_high(lo1, hb_lo_1), va_lo, vb_lo) };

                // Accumulate dot products for lo sub-block.
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

                let wh0 = unsafe { scale_nibbles_q5(q5_widen_low(hi0, hb_hi_0), va_hi, vb_hi) };
                let wh1 = unsafe { scale_nibbles_q5(q5_widen_mid(hi0, hb_hi_0), va_hi, vb_hi) };
                let wh2 = unsafe { scale_nibbles_q5(q5_widen_hi_low(hi0, hb_hi_0), va_hi, vb_hi) };
                let wh3 = unsafe { scale_nibbles_q5(q5_widen_hi_high(hi0, hb_hi_0), va_hi, vb_hi) };
                let wh4 = unsafe { scale_nibbles_q5(q5_widen_low(hi1, hb_hi_1), va_hi, vb_hi) };
                let wh5 = unsafe { scale_nibbles_q5(q5_widen_mid(hi1, hb_hi_1), va_hi, vb_hi) };
                let wh6 = unsafe { scale_nibbles_q5(q5_widen_hi_low(hi1, hb_hi_1), va_hi, vb_hi) };
                let wh7 = unsafe { scale_nibbles_q5(q5_widen_hi_high(hi1, hb_hi_1), va_hi, vb_hi) };

                // Accumulate dot products for hi sub-block.
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

                // Lo nibbles + high bits → first 32 weights of this group.
                for l in 0..32 {
                    let idx = w_off + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128 because qs_off < 128 and l < 32.
                        // qh index l < 32.
                        let lo_nib = (unsafe { *qs.get_unchecked(qs_off + l) } & 0x0F) as u32;
                        let hi_bit = ((unsafe { *qh.get_unchecked(l) } >> group) & 1) as u32;
                        let q = (lo_nib | (hi_bit << 4)) as f32;
                        partial_sum += (d1 * q - m1) * input[idx];
                    }
                }

                // Hi nibbles + high bits → next 32 weights.
                for l in 0..32 {
                    let idx = w_off + 32 + l;
                    if idx < n_cols {
                        // SAFETY: qs_off + l < 128; qh index l < 32.
                        let hi_nib =
                            ((unsafe { *qs.get_unchecked(qs_off + l) } >> 4) & 0x0F) as u32;
                        let hi_bit = ((unsafe { *qh.get_unchecked(l) } >> (group + 4)) & 1) as u32;
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

impl QuantKernel for Q5_KNeon {
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
        "Q5_K_NEON"
    }

    /// Fused Q5_K weight × Q8_0 activation GEMV using NEON.
    ///
    /// Each Q5_K super-block maps to 8 Q8_0 activation blocks.
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
            // SAFETY: bounds checked above.
            let row_sum = unsafe {
                fused_q5_k_q8_0_row_neon(
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

/// Q8_0 activation block byte count.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Fused Q5_K weight × Q8_0 activation dot product for one row using NEON.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - Must run on AArch64 with NEON.
unsafe fn fused_q5_k_q8_0_row_neon(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        // SAFETY: blk < blocks_per_row; row_data.len() == blocks_per_row * BLOCK_BYTES.
        let block = &row_data[bo..bo + BLOCK_BYTES];

        let d = f16_to_f32(block);
        let dmin = f16_to_f32(&block[2..]);
        let (sc, mn) = decode_scales_mins(&block[4..16]);
        let qh = &block[16..48];
        let qs = &block[48..176];

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        let mut is = 0usize;
        let mut qs_off = 0usize;
        let mut w_off = 0usize;

        for group in 0..4_u32 {
            // Sub-block `is` (lo nibbles).
            let a_idx_lo = blk * 8 + is;
            let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
            // SAFETY: acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES.
            let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
            let d_a_lo = f16_to_f32(a_block_lo);
            let q8_lo = &a_block_lo[2..];

            let da_lo = d * sc[is] as f32;
            let m_lo = dmin * mn[is] as f32;

            // Sub-block `is+1` (hi nibbles).
            let a_idx_hi = blk * 8 + is + 1;
            let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
            // SAFETY: same guarantee.
            let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
            let d_a_hi = f16_to_f32(a_block_hi);
            let q8_hi = &a_block_hi[2..];

            let da_hi = d * sc[is + 1] as f32;
            let m_hi = dmin * mn[is + 1] as f32;

            let valid_lo = cols_in_block.saturating_sub(w_off).min(32);
            let valid_hi = cols_in_block.saturating_sub(w_off + 32).min(32);

            // Process lo sub-block: Σ(q5_lo * q8_lo) and Σ(q8_lo).
            let mut dot_lo = 0.0f32;
            let mut sum_a_lo = 0.0f32;
            for l in 0..valid_lo {
                let qh_bit = (qh[l] >> group) & 1;
                let q_w = ((qs[qs_off + l] & 0x0F) | (qh_bit << 4)) as f32;
                let q_a = q8_lo[l] as i8 as f32;
                dot_lo += q_w * q_a;
                sum_a_lo += q_a;
            }
            row_sum += (da_lo * dot_lo - m_lo * sum_a_lo) * d_a_lo;

            // Process hi sub-block.
            let mut dot_hi = 0.0f32;
            let mut sum_a_hi = 0.0f32;
            for l in 0..valid_hi {
                let qh_bit = (qh[l] >> (group + 4)) & 1;
                let q_w = (((qs[qs_off + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
                let q_a = q8_hi[l] as i8 as f32;
                dot_hi += q_w * q_a;
                sum_a_hi += q_a;
            }
            row_sum += (da_hi * dot_hi - m_hi * sum_a_hi) * d_a_hi;

            is += 2;
            qs_off += 32;
            w_off += 64;
        }
    }

    row_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
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

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q5K)
    }

    #[test]
    fn test_q5k_neon_dequant_matches_reference_zero_high_bits() {
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x00u8; 32];
        let qs = [0x88u8; 128]; // lo=8, hi=8

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [zero-high-bits] at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q5k_neon_dequant_matches_reference_all_high_bits() {
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xFFu8; 32]; // all high bits set
        let qs = [0x00u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [all-high-bits] at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q5k_neon_dequant_matches_reference_varied() {
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
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q5_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q5KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [varied] at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q5k_neon_gemv_matches_reference() {
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0xAAu8; 32]; // alternating bits
        let qs = [0x5Au8; 128];

        let block = make_q5k_block(0.5, 0.1, &scales, &qh, &qs);
        let tensor_neon = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        assert!(
            (out_neon[0] - out_ref[0]).abs() < 1e-2,
            "gemv mismatch: neon={}, ref={}",
            out_neon[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q5k_neon_gemv_partial_block() {
        // 200 columns — partial block.
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qh = [0x55u8; 32];
        let qs = [0x11u8; 128];

        let block = make_q5k_block(1.0, 0.0, &scales, &qh, &qs);
        let tensor_neon = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv partial");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv partial");

        assert!(
            (out_neon[0] - out_ref[0]).abs() < 1e-2,
            "partial gemv mismatch: neon={}, ref={}",
            out_neon[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q5k_neon_gemv_varied_data() {
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
        let tensor_neon = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q5_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv varied");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv varied");

        assert!(
            (out_neon[0] - out_ref[0]).abs() < 1e-2,
            "varied gemv mismatch: neon={}, ref={}",
            out_neon[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_q5k_neon_gemv_multi_row() {
        let n_rows = 3usize;
        let n_cols = 512usize; // 2 full blocks per row

        let mut scales = [0u8; 12];
        scales[..4].fill(2);
        scales[4..8].fill(1);
        scales[8..12].fill(3);

        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 7 + 3) & 0xFF) as u8;
        }

        let qs: Vec<u8> = (0..128u8).collect();
        let qs_arr: [u8; 128] = {
            let mut arr = [0u8; 128];
            arr.copy_from_slice(&qs);
            arr
        };

        let block = make_q5k_block(0.5, 0.05, &scales, &qh, &qs_arr);
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
            oxillama_gguf::GgufTensorType::Q5K,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q5K,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q5_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q5KRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-2,
                "gemv row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }
}
