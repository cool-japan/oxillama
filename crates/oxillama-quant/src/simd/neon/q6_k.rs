//! NEON (AArch64) accelerated Q6_K quantization kernel.
//!
//! Q6_K block layout (210 bytes per 256 weights):
//! - bytes[0..128]   — `ql` — lower 4 bits of each 6-bit quant (2 per byte)
//! - bytes[128..192] — `qh` — upper 2 bits of each 6-bit quant (4 per byte)
//! - bytes[192..208] — 16 × int8 signed scales (one per 16-weight sub-block)
//! - bytes[208..210] — FP16 super-block scale `d` (little-endian)
//!
//! Block structure: 16 sub-blocks of 16 weights, each with its own int8 scale.
//! Q6_K is a symmetric ("type-0") format — no minimum offset term.
//!
//! Weight formula: `w = d * scale_i * (q6 - 32)` where `q6 = ql_lo | (qh_hi << 4)`.
//!
//! The ql/qh bit extraction mirrors the AVX2 kernel but uses true per-byte
//! NEON shifts (`vshrq_n_u8`) instead of 16-bit lane shifts, eliminating the
//! need for cross-byte contamination masking on the shift results.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q6_K: 256 weights per block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q6_K block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (FP16 d).
pub const BLOCK_BYTES: usize = 210;

/// NEON-accelerated Q6_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q6_KNeon;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

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

/// Decode 16 × 6-bit quants using the Q6_K bit-extraction pattern.
///
/// Loads 16 bytes from `ql[0..15]`, 16 bytes from `ql[32..47]`, and 16 bytes
/// from `qh[0..15]`, producing four `uint8x16_t` registers:
/// - q1: `(ql[l] & 0x0F) | ((qh[l] & 3) << 4)`         → out + 0..15
/// - q2: `(ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)` → out + 32..47
/// - q3: `(ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)`     → out + 64..79
/// - q4: `(ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)`  → out + 96..111
///
/// # Safety
/// - `ql_ptr` must be valid for reads at byte offsets 0..15 and 32..47.
/// - `qh_ptr` must be valid for reads at byte offsets 0..15.
/// - Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn decode_q6_half_neon(
    ql_ptr: *const u8,
    qh_ptr: *const u8,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    // SAFETY: caller ensures valid pointers for the required ranges.
    unsafe {
        let ql0 = vld1q_u8(ql_ptr); // ql[0..15]
        let ql1 = vld1q_u8(ql_ptr.add(32)); // ql[32..47]
        let qh_raw = vld1q_u8(qh_ptr); // qh[0..15]

        let mask4 = vdupq_n_u8(0x0F);
        let mask2 = vdupq_n_u8(0x03);

        // Lower nibbles from ql.
        let ql0_lo = vandq_u8(ql0, mask4); // ql[0..15] & 0x0F
        let ql1_lo = vandq_u8(ql1, mask4); // ql[32..47] & 0x0F

        // Upper nibbles from ql (true per-byte shift — no cross-byte leak).
        let ql0_hi = vshrq_n_u8::<4>(ql0); // ql[0..15] >> 4
        let ql1_hi = vshrq_n_u8::<4>(ql1); // ql[32..47] >> 4

        // 2-bit upper parts from qh, extracted via shift + mask.
        // The mask is technically redundant for shift-by-6 with per-byte ops,
        // but kept for uniformity and documentation clarity.
        let qh_sh0 = vandq_u8(qh_raw, mask2); // bits 1:0
        let qh_sh2 = vandq_u8(vshrq_n_u8::<2>(qh_raw), mask2); // bits 3:2
        let qh_sh4 = vandq_u8(vshrq_n_u8::<4>(qh_raw), mask2); // bits 5:4
        let qh_sh6 = vandq_u8(vshrq_n_u8::<6>(qh_raw), mask2); // bits 7:6

        // Shift 2-bit parts left by 4 to occupy bit positions 5:4.
        // Max value is 3 << 4 = 48 — fits in a byte with no overflow.
        let qh_hi0 = vshlq_n_u8::<4>(qh_sh0);
        let qh_hi2 = vshlq_n_u8::<4>(qh_sh2);
        let qh_hi4 = vshlq_n_u8::<4>(qh_sh4);
        let qh_hi6 = vshlq_n_u8::<4>(qh_sh6);

        // Combine: q = ql_part | qh_shifted → 6-bit quant in 0..63.
        let q1 = vorrq_u8(ql0_lo, qh_hi0);
        let q2 = vorrq_u8(ql1_lo, qh_hi2);
        let q3 = vorrq_u8(ql0_hi, qh_hi4);
        let q4 = vorrq_u8(ql1_hi, qh_hi6);

        (q1, q2, q3, q4)
    }
}

/// Widen 16 unsigned 6-bit quant bytes to 4 groups of 4 float32 values,
/// subtracting 32 from each to center the symmetric quants around zero.
///
/// Returns `(f0, f1, f2, f3)` covering elements `[0..3]`, `[4..7]`, `[8..11]`,
/// `[12..15]`.
///
/// # Safety
/// All widening/conversion intrinsics are unconditionally valid on AArch64.
#[inline(always)]
unsafe fn q6_bytes_to_f32_quad(
    q: uint8x16_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    // SAFETY: vmovl_u8, vmovl_u16, vmovl_high_u16, vcvtq_f32_u32, vsubq_f32
    // are all unconditionally valid on AArch64.
    unsafe {
        let offset = vdupq_n_f32(32.0);

        // Widen low 8 bytes: u8 → u16 → u32 → f32 − 32
        let u16_lo = vmovl_u8(vget_low_u8(q));
        let u32_0 = vmovl_u16(vget_low_u16(u16_lo)); // elements 0..3
        let f0 = vsubq_f32(vcvtq_f32_u32(u32_0), offset);
        let u32_1 = vmovl_high_u16(u16_lo); // elements 4..7
        let f1 = vsubq_f32(vcvtq_f32_u32(u32_1), offset);

        // Widen high 8 bytes
        let u16_hi = vmovl_u8(vget_high_u8(q));
        let u32_2 = vmovl_u16(vget_low_u16(u16_hi)); // elements 8..11
        let f2 = vsubq_f32(vcvtq_f32_u32(u32_2), offset);
        let u32_3 = vmovl_high_u16(u16_hi); // elements 12..15
        let f3 = vsubq_f32(vcvtq_f32_u32(u32_3), offset);

        (f0, f1, f2, f3)
    }
}

/// Store 16 scaled `(q6 − 32)` float32 values: `output[ptr..ptr+16] = scale * (q6 - 32)`.
///
/// # Safety
/// - `ptr` must be valid for 16 consecutive f32 writes.
/// - Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn store_scaled_q6(q: uint8x16_t, scale: float32x4_t, ptr: *mut f32) {
    // SAFETY: q6_bytes_to_f32_quad, vmulq_f32, vst1q_f32 are valid on AArch64.
    unsafe {
        let (f0, f1, f2, f3) = q6_bytes_to_f32_quad(q);
        vst1q_f32(ptr, vmulq_f32(scale, f0));
        vst1q_f32(ptr.add(4), vmulq_f32(scale, f1));
        vst1q_f32(ptr.add(8), vmulq_f32(scale, f2));
        vst1q_f32(ptr.add(12), vmulq_f32(scale, f3));
    }
}

/// Accumulate 16 scaled dot products into `acc`:
/// `acc += scale * (q6 - 32) * input[inp_ptr..inp_ptr+16]`.
///
/// # Safety
/// - `inp_ptr` must be valid for 16 consecutive f32 reads.
/// - Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn acc_scaled_q6(
    q: uint8x16_t,
    scale: float32x4_t,
    inp_ptr: *const f32,
    acc: float32x4_t,
) -> float32x4_t {
    // SAFETY: q6_bytes_to_f32_quad, vmulq_f32, vld1q_f32, vfmaq_f32 are
    // all unconditionally valid on AArch64.
    unsafe {
        let (f0, f1, f2, f3) = q6_bytes_to_f32_quad(q);
        let w0 = vmulq_f32(scale, f0);
        let w1 = vmulq_f32(scale, f1);
        let w2 = vmulq_f32(scale, f2);
        let w3 = vmulq_f32(scale, f3);

        let i0 = vld1q_f32(inp_ptr);
        let i1 = vld1q_f32(inp_ptr.add(4));
        let i2 = vld1q_f32(inp_ptr.add(8));
        let i3 = vld1q_f32(inp_ptr.add(12));

        let mut a = vfmaq_f32(acc, w0, i0);
        a = vfmaq_f32(a, w1, i1);
        a = vfmaq_f32(a, w2, i2);
        vfmaq_f32(a, w3, i3)
    }
}

// ---------------------------------------------------------------------------
// Internal NEON kernels
// ---------------------------------------------------------------------------

/// Dequantize one 210-byte Q6_K block into 256 FP32 values using NEON.
///
/// Processes 2 groups of 128 weights.  Each group is split into two 16-element
/// halves (A = positions 0..15, B = positions 16..31) with four quant streams
/// (q1..q4) producing 4 × 32 = 128 output weights.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES` (210)
/// - `output.len() >= BLOCK_SIZE` (256)
/// - Must be called on AArch64 with NEON
unsafe fn dequant_block_neon(block: &[u8], output: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    // SAFETY: block.len() >= 210.
    let d = f16_to_f32(&block[208..]);

    let out_ptr = output.as_mut_ptr();

    // Process 2 groups of 128 weights each.
    for group in 0..2usize {
        let ql_off = group * 64; // ql base for this group
        let qh_off = group * 32; // qh base for this group
        let sc_off = group * 8; // scales base (8 per group)
        let out_off = group * 128; // output base

        // -------- Half A: positions 0..15 --------
        // SAFETY: ql_off + 47 < 128; qh_off + 15 < 64.
        let (q1_a, q2_a, q3_a, q4_a) =
            unsafe { decode_q6_half_neon(ql.as_ptr().add(ql_off), qh.as_ptr().add(qh_off)) };

        // -------- Half B: positions 16..31 --------
        // SAFETY: ql_off + 16 + 47 <= 127 for group 0; ql_off + 16 = 80, 80 + 47 = 127 < 128.
        //         qh_off + 16 + 15 <= 63 for group 0; qh_off + 16 = 48, 48 + 15 = 63 < 64.
        let (q1_b, q2_b, q3_b, q4_b) = unsafe {
            decode_q6_half_neon(ql.as_ptr().add(ql_off + 16), qh.as_ptr().add(qh_off + 16))
        };

        // Scale factors: signed int8 cast (note: scales[] bytes are raw, cast as i8).
        // Half A uses even-indexed scales; half B uses odd-indexed.
        // SAFETY: sc_off + 7 <= 15; scales.len() == 16.
        let s0a = d * unsafe { *scales.get_unchecked(sc_off) } as i8 as f32;
        let s1a = d * unsafe { *scales.get_unchecked(sc_off + 2) } as i8 as f32;
        let s2a = d * unsafe { *scales.get_unchecked(sc_off + 4) } as i8 as f32;
        let s3a = d * unsafe { *scales.get_unchecked(sc_off + 6) } as i8 as f32;

        let s0b = d * unsafe { *scales.get_unchecked(sc_off + 1) } as i8 as f32;
        let s1b = d * unsafe { *scales.get_unchecked(sc_off + 3) } as i8 as f32;
        let s2b = d * unsafe { *scales.get_unchecked(sc_off + 5) } as i8 as f32;
        let s3b = d * unsafe { *scales.get_unchecked(sc_off + 7) } as i8 as f32;

        // SAFETY: vdupq_n_f32 is always valid on AArch64.
        let vs0a = unsafe { vdupq_n_f32(s0a) };
        let vs1a = unsafe { vdupq_n_f32(s1a) };
        let vs2a = unsafe { vdupq_n_f32(s2a) };
        let vs3a = unsafe { vdupq_n_f32(s3a) };
        let vs0b = unsafe { vdupq_n_f32(s0b) };
        let vs1b = unsafe { vdupq_n_f32(s1b) };
        let vs2b = unsafe { vdupq_n_f32(s2b) };
        let vs3b = unsafe { vdupq_n_f32(s3b) };

        // --- q1 stream: out_off + 0..31 (half A → 0..15, half B → 16..31) ---
        // SAFETY: out_off + 31 <= group*128 + 31 <= 255 < 256 = output.len().
        unsafe {
            store_scaled_q6(q1_a, vs0a, out_ptr.add(out_off));
            store_scaled_q6(q1_b, vs0b, out_ptr.add(out_off + 16));
        }

        // --- q2 stream: out_off + 32..63 ---
        // SAFETY: out_off + 63 <= 255.
        unsafe {
            store_scaled_q6(q2_a, vs1a, out_ptr.add(out_off + 32));
            store_scaled_q6(q2_b, vs1b, out_ptr.add(out_off + 48));
        }

        // --- q3 stream: out_off + 64..95 ---
        // SAFETY: out_off + 95 <= 255.
        unsafe {
            store_scaled_q6(q3_a, vs2a, out_ptr.add(out_off + 64));
            store_scaled_q6(q3_b, vs2b, out_ptr.add(out_off + 80));
        }

        // --- q4 stream: out_off + 96..127 ---
        // SAFETY: out_off + 127 <= 255.
        unsafe {
            store_scaled_q6(q4_a, vs3a, out_ptr.add(out_off + 96));
            store_scaled_q6(q4_b, vs3b, out_ptr.add(out_off + 112));
        }
    }
}

/// Compute the dot product of one row of a Q6_K matrix with an FP32 vector
/// using NEON.
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

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        // SAFETY: block.len() == 210 >= 210.
        let d = f16_to_f32(&block[208..]);

        if remaining >= BLOCK_SIZE {
            // Fast path: all 256 weights in bounds — fully vectorised.
            // SAFETY: vdupq_n_f32 is always valid on AArch64.
            let mut block_acc = unsafe { vdupq_n_f32(0.0f32) };

            for group in 0..2usize {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let w_off = input_offset + group * 128;

                // SAFETY: sc_off + 7 <= 15; scales.len() == 16.
                let s0a = d * unsafe { *scales.get_unchecked(sc_off) } as i8 as f32;
                let s1a = d * unsafe { *scales.get_unchecked(sc_off + 2) } as i8 as f32;
                let s2a = d * unsafe { *scales.get_unchecked(sc_off + 4) } as i8 as f32;
                let s3a = d * unsafe { *scales.get_unchecked(sc_off + 6) } as i8 as f32;
                let s0b = d * unsafe { *scales.get_unchecked(sc_off + 1) } as i8 as f32;
                let s1b = d * unsafe { *scales.get_unchecked(sc_off + 3) } as i8 as f32;
                let s2b = d * unsafe { *scales.get_unchecked(sc_off + 5) } as i8 as f32;
                let s3b = d * unsafe { *scales.get_unchecked(sc_off + 7) } as i8 as f32;

                // SAFETY: vdupq_n_f32 is always valid on AArch64.
                let vs0a = unsafe { vdupq_n_f32(s0a) };
                let vs1a = unsafe { vdupq_n_f32(s1a) };
                let vs2a = unsafe { vdupq_n_f32(s2a) };
                let vs3a = unsafe { vdupq_n_f32(s3a) };
                let vs0b = unsafe { vdupq_n_f32(s0b) };
                let vs1b = unsafe { vdupq_n_f32(s1b) };
                let vs2b = unsafe { vdupq_n_f32(s2b) };
                let vs3b = unsafe { vdupq_n_f32(s3b) };

                // Decode half A (positions 0..15) and half B (positions 16..31).
                // SAFETY: ql_off + 47 < 128; qh_off + 15 < 64.
                let (q1_a, q2_a, q3_a, q4_a) = unsafe {
                    decode_q6_half_neon(ql.as_ptr().add(ql_off), qh.as_ptr().add(qh_off))
                };
                // SAFETY: ql_off + 16 + 47 <= 127; qh_off + 31 <= 63.
                let (q1_b, q2_b, q3_b, q4_b) = unsafe {
                    decode_q6_half_neon(ql.as_ptr().add(ql_off + 16), qh.as_ptr().add(qh_off + 16))
                };

                let inp = input.as_ptr();

                // q1 stream: w_off + 0..31
                // SAFETY: w_off + 31 < input_offset + BLOCK_SIZE <= n_cols <= input.len().
                block_acc = unsafe { acc_scaled_q6(q1_a, vs0a, inp.add(w_off), block_acc) };
                block_acc = unsafe { acc_scaled_q6(q1_b, vs0b, inp.add(w_off + 16), block_acc) };

                // q2 stream: w_off + 32..63
                // SAFETY: w_off + 63 <= input_offset + BLOCK_SIZE - 1.
                block_acc = unsafe { acc_scaled_q6(q2_a, vs1a, inp.add(w_off + 32), block_acc) };
                block_acc = unsafe { acc_scaled_q6(q2_b, vs1b, inp.add(w_off + 48), block_acc) };

                // q3 stream: w_off + 64..95
                // SAFETY: w_off + 95 <= input_offset + BLOCK_SIZE - 1.
                block_acc = unsafe { acc_scaled_q6(q3_a, vs2a, inp.add(w_off + 64), block_acc) };
                block_acc = unsafe { acc_scaled_q6(q3_b, vs2b, inp.add(w_off + 80), block_acc) };

                // q4 stream: w_off + 96..127
                // SAFETY: w_off + 127 <= input_offset + BLOCK_SIZE - 1.
                block_acc = unsafe { acc_scaled_q6(q4_a, vs3a, inp.add(w_off + 96), block_acc) };
                block_acc = unsafe { acc_scaled_q6(q4_b, vs3b, inp.add(w_off + 112), block_acc) };
            }

            // SAFETY: hsum_f32x4 calls vaddvq_f32 which is valid on AArch64.
            row_sum += unsafe { hsum_f32x4(block_acc) };
        } else if remaining > 0 {
            // Tail path: partial block — scalar fallback to avoid out-of-bounds reads.
            let mut partial_sum = 0.0f32;

            for group in 0..2usize {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let in_off = input_offset + group * 128;

                for l in 0..32 {
                    let is = l / 16;

                    // SAFETY: ql_off + l + 32 < 128; qh_off + l < 64; sc_off + is + 6 < 16.
                    let ql_l = unsafe { *ql.get_unchecked(ql_off + l) };
                    let ql_l32 = unsafe { *ql.get_unchecked(ql_off + l + 32) };
                    let qh_l = unsafe { *qh.get_unchecked(qh_off + l) };

                    let q1 = ((ql_l & 0x0F) | ((qh_l & 3) << 4)) as i32 - 32;
                    let q2 = ((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) as i32 - 32;

                    let s0 = d * unsafe { *scales.get_unchecked(sc_off + is) } as i8 as f32;
                    let s1 = d * unsafe { *scales.get_unchecked(sc_off + is + 2) } as i8 as f32;
                    let s2 = d * unsafe { *scales.get_unchecked(sc_off + is + 4) } as i8 as f32;
                    let s3 = d * unsafe { *scales.get_unchecked(sc_off + is + 6) } as i8 as f32;

                    let idx0 = in_off + l;
                    let idx1 = in_off + l + 32;
                    let idx2 = in_off + l + 64;
                    let idx3 = in_off + l + 96;

                    if idx0 < n_cols {
                        partial_sum += s0 * q1 as f32 * input[idx0];
                    }
                    if idx1 < n_cols {
                        partial_sum += s1 * q2 as f32 * input[idx1];
                    }
                    if idx2 < n_cols {
                        partial_sum += s2 * q3 as f32 * input[idx2];
                    }
                    if idx3 < n_cols {
                        partial_sum += s3 * q4 as f32 * input[idx3];
                    }
                }
            }

            row_sum += partial_sum;
        }
        // remaining == 0: block fully out of bounds, skip.
    }

    row_sum
}

// ---------------------------------------------------------------------------
// QuantKernel trait implementation
// ---------------------------------------------------------------------------

impl QuantKernel for Q6_KNeon {
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

        // SAFETY: block.len() >= 210 and output.len() >= 256 verified above.
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
        "Q6_K_NEON"
    }

    /// Fused Q6_K weight × Q8_0 activation GEMV using NEON.
    ///
    /// Each Q6_K super-block (256 weights) maps to 8 Q8_0 activation blocks.
    /// Column `col` → Q8_0 block `blk*8 + col/32`, lane `col % 32`.
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

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked above.
            let row_sum = unsafe {
                fused_q6_k_q8_0_row_neon(
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
}

/// Q8_0 activation block byte count.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Convert two raw LE bytes to f32 via FP16.
#[inline(always)]
fn f16_bytes_to_f32_neon(bytes: [u8; 2]) -> f32 {
    half::f16::from_le_bytes(bytes).to_f32()
}

/// Fused Q6_K weight × Q8_0 activation dot product for one row using NEON.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - Must run on AArch64 with NEON.
unsafe fn fused_q6_k_q8_0_row_neon(
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

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_bytes_to_f32_neon([block[208], block[209]]);

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let in_off = group * 128;

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql[ql_off + l] & 0x0F)
                    | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F)
                    | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4)
                    | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4)
                    | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let s0 = d * scales[sc_off + is] as i8 as f32;
                let s1 = d * scales[sc_off + is + 2] as i8 as f32;
                let s2 = d * scales[sc_off + is + 4] as i8 as f32;
                let s3 = d * scales[sc_off + is + 6] as i8 as f32;

                let c0 = in_off + l;
                let c1 = in_off + l + 32;
                let c2 = in_off + l + 64;
                let c3 = in_off + l + 96;

                let sample_q8 = |col: usize| -> Option<f32> {
                    if col >= cols_in_block {
                        return None;
                    }
                    let q8_blk = blk * 8 + col / 32;
                    let q8_lane = col % 32;
                    // SAFETY: acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES.
                    let ab = &acts_q8[q8_blk * Q8_0_BLOCK_BYTES..(q8_blk + 1) * Q8_0_BLOCK_BYTES];
                    let d_a = f16_bytes_to_f32_neon([ab[0], ab[1]]);
                    let q_a = ab[2 + q8_lane] as i8 as f32;
                    Some(d_a * q_a)
                };

                if let Some(a0) = sample_q8(c0) {
                    row_sum += s0 * q1 as f32 * a0;
                }
                if let Some(a1) = sample_q8(c1) {
                    row_sum += s1 * q2 as f32 * a1;
                }
                if let Some(a2) = sample_q8(c2) {
                    row_sum += s2 * q3 as f32 * a2;
                }
                if let Some(a3) = sample_q8(c3) {
                    row_sum += s3 * q4 as f32 * a3;
                }
            }
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
    use crate::reference::q6_k::Q6KRef;

    fn make_q6k_block(d: f32, ql: &[u8; 128], qh: &[u8; 64], scales: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(ql);
        block.extend_from_slice(qh);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q6K)
    }

    #[test]
    fn test_q6k_neon_dequant_matches_reference_zero() {
        // d=0 → all weights zero.
        let block = make_q6k_block(0.0, &[0; 128], &[0; 64], &[0; 16]);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q6KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-5,
                "dequant mismatch [zero] at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q6k_neon_dequant_matches_reference_quant32() {
        // All quants = 32 (q-32 = 0) → all weights = 0 regardless of scales.
        // ql lower nibbles = 0, qh 2-bit parts = 2 (0b10) → quant = 0 | (2 << 4) = 32.
        // qh packed: each byte encodes 4 × 2-bit = 0b10101010 = 0xAA.
        let qh = [0xAAu8; 64];
        let scales: [u8; 16] = [1; 16];

        let block = make_q6k_block(1.0, &[0; 128], &qh, &scales);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q6KRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        for (i, (&a, &r)) in out_neon.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch [quant32] at index {i}: neon={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_q6k_neon_dequant_matches_reference_varied() {
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as i8 * 3 - 8) as u8;
        }

        let block = make_q6k_block(0.5, &ql, &qh, &scales);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];

        Q6_KNeon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q6KRef
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
    fn test_q6k_neon_gemv_matches_reference() {
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as i8 * 3 - 8) as u8;
        }

        let block = make_q6k_block(0.5, &ql, &qh, &scales);
        let tensor_neon = make_tensor(block.clone(), 256);
        let tensor_ref = make_tensor(block, 256);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q6_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q6KRef
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
    fn test_q6k_neon_gemv_partial_block() {
        // 200 columns — partial block triggers scalar tail.
        let scales = [1i8 as u8; 16];
        let block = make_q6k_block(1.0, &[0; 128], &[0xAAu8; 64], &scales);
        let tensor_neon = make_tensor(block.clone(), 200);
        let tensor_ref = make_tensor(block, 200);

        let input = vec![1.0f32; 200];
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q6_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv partial");
        Q6KRef
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
    fn test_q6k_neon_gemm_delegates_to_gemv() {
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = ((i * 11 + 1) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = ((i * 7 + 2) & 0xFF) as u8;
        }
        let scales: [u8; 16] = [3; 16];

        // 2 rows × 256 cols → need 2 blocks.
        let block = make_q6k_block(0.25, &ql, &qh, &scales);
        let mut data = Vec::with_capacity(BLOCK_BYTES * 2);
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);

        let tensor = QuantTensor::new(data, vec![2, 256], oxillama_gguf::GgufTensorType::Q6K);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();

        // gemv on each row individually.
        let mut out_gemv = vec![0.0f32; 2];
        Q6_KNeon.gemv(&tensor, &input, &mut out_gemv).expect("gemv");

        // gemm with m=1 should give same results.
        // (Row delegation: gemm calls gemv per input row.)
        let mut out_gemm = vec![0.0f32; 2];
        Q6_KNeon
            .gemm(&tensor, &input, &mut out_gemm, 1, 2, 256)
            .expect("gemm");

        for (i, (&g, &v)) in out_gemm.iter().zip(out_gemv.iter()).enumerate() {
            assert!(
                (g - v).abs() < 1e-5,
                "gemm/gemv mismatch at row {i}: gemm={g}, gemv={v}"
            );
        }
    }

    // ── matvec_q8_fused ───────────────────────────────────────────────────

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
    fn test_q6k_neon_fused_matches_reference_single_block() {
        let mut ql = [0u8; 128];
        for (i, q) in ql.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let scales: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let w_block = make_q6k_block(0.01, &ql, &qh, &scales);
        let act_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
            0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let acts = make_q8_acts(8, 0.1, &act_vals);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q6_KNeon
            .matvec_q8_fused(&w_block, &acts, &mut out_neon, 1, 256)
            .expect("neon fused single block");
        Q6KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, 256)
            .expect("ref fused single block");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "fused single-block mismatch: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q6k_neon_fused_multi_row() {
        let n_rows = 3usize;
        let n_cols = 512usize;
        let blocks_per_row = 2usize;
        let q8_blocks_per_row = blocks_per_row * 8;

        let mut all_weights = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut ql = [0u8; 128];
                for (i, q) in ql.iter_mut().enumerate() {
                    *q = ((r * 17 + b * 11 + i * 5) & 0xFF) as u8;
                }
                let mut qh = [0u8; 64];
                for (i, h) in qh.iter_mut().enumerate() {
                    *h = ((r * 13 + b * 19 + i * 3) & 0xFF) as u8;
                }
                let scales: [u8; 16] = core::array::from_fn(|i| {
                    ((r * 7 + b * 11 + i * 13 + 5) & 0xFF) as u8
                });
                all_weights.extend(make_q6k_block(
                    0.01 + r as f32 * 0.005,
                    &ql,
                    &qh,
                    &scales,
                ));
            }
        }

        let act_vals: [i8; 32] = [
            2, -3, 5, -7, 1, -1, 4, -4, 6, -6, 3, -3, 2, -2, 1, -1,
            8, -8, 7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1,
        ];
        let acts = make_q8_acts(q8_blocks_per_row, 0.05, &act_vals);

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q6_KNeon
            .matvec_q8_fused(&all_weights, &acts, &mut out_neon, n_rows, n_cols)
            .expect("neon fused multi-row");
        Q6KRef
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused multi-row");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "fused multi-row row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }

    #[test]
    fn test_q6k_neon_fused_accumulate_semantics() {
        let w_block = make_q6k_block(0.0, &[0u8; 128], &[0u8; 64], &[0u8; 16]);
        let acts = make_q8_acts(8, 0.0, &[0i8; 32]);

        let mut out = vec![77.0f32; 1];
        Q6_KNeon
            .matvec_q8_fused(&w_block, &acts, &mut out, 1, 256)
            .expect("neon fused accumulate");

        assert!(
            (out[0] - 77.0).abs() < 1e-5,
            "accumulate semantics broken: expected 77.0, got {}",
            out[0]
        );
    }
}
