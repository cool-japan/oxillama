//! Q4_0 NEON (AArch64) SIMD kernel.
//!
//! Q4_0 block format (18 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..18]: 16 packed nibble bytes (32 × 4-bit values)
//!
//! Nibble layout per byte: lo = byte & 0x0F (weight 2i), hi = byte >> 4 (weight 2i+1)
//! Weight reconstruction: `(nibble - 8) * d`

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q4_0 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q4_0 block.
pub const BLOCK_BYTES: usize = 18;

/// NEON-accelerated Q4_0 kernel (AArch64 only).
pub struct Q4_0Neon;

/// Convert an IEEE 754 FP16 value (as raw u16 LE bytes) to f32.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Horizontal sum of a `float32x4_t` register.
///
/// Uses `vaddvq_f32` which is a single instruction on NEON.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    // SAFETY: caller guarantees AArch64 context; vaddvq_f32 is always valid.
    unsafe { vaddvq_f32(v) }
}

/// Dequantize one Q4_0 block using NEON intrinsics.
///
/// Produces 32 f32 values in interleaved order [lo0, hi0, lo1, hi1, ...].
///
/// # Safety
/// Must be called on AArch64 with NEON. Pointer `nibbles` must point
/// to at least 16 initialised bytes.
#[inline]
unsafe fn dequant_block_neon(nibbles: *const u8, d: f32, output: &mut [f32]) {
    // SAFETY: caller ensures nibbles points to 16 valid bytes.
    let raw = unsafe { vld1q_u8(nibbles) };

    // Split nibbles: lo = lower 4 bits, hi = upper 4 bits
    let mask = unsafe { vdupq_n_u8(0x0F) };
    // SAFETY: vandq_u8 and vshrq_n_u8 are safe on valid u8x16 registers.
    let lo = unsafe { vandq_u8(raw, mask) };
    let hi = unsafe { vshrq_n_u8::<4>(raw) };

    // Offset vector: subtract 8 from each nibble to centre on zero
    let offset = unsafe { vdupq_n_s16(8) };
    let d_vec = unsafe { vdupq_n_f32(d) };

    // Process lo nibbles (weight indices 0, 2, 4, ..., 30)
    // SAFETY: vmovl_u8 / vreinterpretq_s16_u16 / vsubq_s16 are always valid on AArch64.
    let lo16_low = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo))), offset) };
    let lo16_high = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo))), offset) };

    // Process hi nibbles (weight indices 1, 3, 5, ..., 31)
    // SAFETY: same as above.
    let hi16_low = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi))), offset) };
    let hi16_high = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi))), offset) };

    // Convert to f32 and multiply by d (4 groups of 4 per lane)
    // lo_low: weights [0, 2, 4, 6, 8, 10, 12, 14]
    // lo_high: weights [16, 18, 20, 22, 24, 26, 28, 30]
    // hi_low: weights [1, 3, 5, 7, 9, 11, 13, 15]
    // hi_high: weights [17, 19, 21, 23, 25, 27, 29, 31]

    // SAFETY: vcvtq_f32_s32 / vmovl_s16 / vmulq_f32 are valid on AArch64.
    let lo_f0 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_low))), d_vec) };
    let lo_f1 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo16_low)), d_vec) };
    let lo_f2 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_high))), d_vec) };
    let lo_f3 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo16_high)), d_vec) };

    let hi_f0 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_low))), d_vec) };
    let hi_f1 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi16_low)), d_vec) };
    let hi_f2 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_high))), d_vec) };
    let hi_f3 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi16_high)), d_vec) };

    // Interleave lo/hi pairs into output: [lo0,hi0,lo1,hi1, lo2,hi2,lo3,hi3, ...]
    // Each vzip group handles 4 lo + 4 hi → 8 interleaved values.
    // SAFETY: vzipq_f32 is always valid on AArch64.
    let zip0 = unsafe { vzipq_f32(lo_f0, hi_f0) };
    let zip1 = unsafe { vzipq_f32(lo_f1, hi_f1) };
    let zip2 = unsafe { vzipq_f32(lo_f2, hi_f2) };
    let zip3 = unsafe { vzipq_f32(lo_f3, hi_f3) };

    // Store interleaved results
    // zip0.0 = [lo0,hi0,lo1,hi1], zip0.1 = [lo2,hi2,lo3,hi3]
    // zip1.0 = [lo4,hi4,lo5,hi5], zip1.1 = [lo6,hi6,lo7,hi7]
    // etc.
    // SAFETY: vst1q_f32 requires a valid pointer to 4 f32 values.
    unsafe { vst1q_f32(output.as_mut_ptr(), zip0.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), zip0.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), zip1.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), zip1.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), zip2.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), zip2.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), zip3.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), zip3.1) };
}

/// Compute the dot product between one dequantized Q4_0 block and 32 f32 inputs.
///
/// Returns `d * Σ (nibble_i - 8) * input_i`.
///
/// The key insight: nibble layout is interleaved [lo0,hi0,lo1,hi1,...] matching
/// input[0], input[1], input[2], input[3]...  We split the input into even/odd
/// streams and dot each with the corresponding lo/hi vectors.
///
/// # Safety
/// Must be called on AArch64 with NEON. `nibbles` must point to 16 valid bytes,
/// `input` must have exactly 32 elements.
#[inline]
unsafe fn dot_block_neon(nibbles: *const u8, d: f32, input: &[f32]) -> f32 {
    // SAFETY: caller ensures nibbles points to 16 valid bytes.
    let raw = unsafe { vld1q_u8(nibbles) };

    let mask = unsafe { vdupq_n_u8(0x0F) };
    // SAFETY: bitwise AND and shift on NEON vector registers.
    let lo = unsafe { vandq_u8(raw, mask) };
    let hi = unsafe { vshrq_n_u8::<4>(raw) };

    let offset = unsafe { vdupq_n_s16(8) };

    // Extend nibbles to i16 and subtract 8
    // SAFETY: vmovl_u8, vreinterpretq_s16_u16, vsubq_s16 are valid on AArch64.
    let lo16_low = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo))), offset) };
    let lo16_high = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo))), offset) };
    let hi16_low = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi))), offset) };
    let hi16_high = unsafe { vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi))), offset) };

    // Convert to f32
    // SAFETY: vcvtq_f32_s32, vmovl_s16, vmovl_high_s16 are valid on AArch64.
    let lo_f0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_low))) };
    let lo_f1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(lo16_low)) };
    let lo_f2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_high))) };
    let lo_f3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(lo16_high)) };

    let hi_f0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_low))) };
    let hi_f1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(hi16_low)) };
    let hi_f2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_high))) };
    let hi_f3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(hi16_high)) };

    // Load input in deinterleaved form: even indices (0,2,4,...) and odd (1,3,5,...).
    // We use vld2q_f32 which loads two interleaved f32 streams.
    // input[0..8]:   vld2q_f32 → val0.0 = [in0,in2,in4,in6], val0.1 = [in1,in3,in5,in7]
    // input[8..16]:  vld2q_f32 → val1.0, val1.1
    // input[16..24]: vld2q_f32 → val2.0, val2.1
    // input[24..32]: vld2q_f32 → val3.0, val3.1
    //
    // SAFETY: vld2q_f32 requires pointer to 8 contiguous f32 (32 bytes).
    // input has 32 elements; offsets 0, 8, 16, 24 each have 8 f32 remaining.
    let ip = input.as_ptr();
    let val0 = unsafe { vld2q_f32(ip) };
    let val1 = unsafe { vld2q_f32(ip.add(8)) };
    let val2 = unsafe { vld2q_f32(ip.add(16)) };
    let val3 = unsafe { vld2q_f32(ip.add(24)) };

    // val0.0 = even inputs [in0,in2,in4,in6] → paired with lo nibbles [lo0,lo1,lo2,lo3]
    // val0.1 = odd  inputs [in1,in3,in5,in7] → paired with hi nibbles [hi0,hi1,hi2,hi3]
    // etc.

    // Dot products: Σ lo_f * even_input + Σ hi_f * odd_input
    // SAFETY: vfmaq_f32 / vmulq_f32 are always valid on AArch64.
    let mut acc = unsafe { vmulq_f32(lo_f0, val0.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f0, val0.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f1, val1.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f1, val1.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f2, val2.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f2, val2.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f3, val3.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f3, val3.1) };

    // SAFETY: hsum_f32x4 calls vaddvq_f32, valid on AArch64.
    d * unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q4_0Neon {
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

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: block has at least 18 bytes; bytes[2..18] = 16 valid nibble bytes.
        // output has at least 32 f32 slots. We pass a slice of exactly 32 to dequant_block_neon.
        unsafe { dequant_block_neon(block.as_ptr().add(2), d, &mut output[..BLOCK_SIZE]) };

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
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let input_offset = blk * BLOCK_SIZE;
                let block_input_end = (input_offset + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_offset;

                if block_input_len == BLOCK_SIZE {
                    // Full block: use NEON fast path.
                    // SAFETY: block has 18 bytes; nibbles at block[2..18] = 16 bytes.
                    // input slice has exactly 32 elements.
                    sum += unsafe {
                        dot_block_neon(
                            block.as_ptr().add(2),
                            d,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Partial block at the tail: scalar fallback.
                    for i in 0..block_input_len {
                        let byte = block[2 + i / 2];
                        let nibble = if i % 2 == 0 {
                            (byte & 0x0F) as i32 - 8
                        } else {
                            ((byte >> 4) & 0x0F) as i32 - 8
                        };
                        sum += nibble as f32 * d * input[input_offset + i];
                    }
                }
            }
            *out = sum;
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

    /// Fused Q4_0 weight × Q8_0 activation GEMV using NEON intrinsics.
    ///
    /// Computes `out[row] += Σ_block (q4_0_weight · q8_0_act)` with ACCUMULATE semantics.
    ///
    /// Activation blocks (Q8_0) are 34 bytes each: 2-byte FP16 scale + 32 i8 values.
    /// Weight blocks (Q4_0) are 18 bytes each: 2-byte FP16 scale + 16 nibble bytes.
    /// One weight block maps 1-to-1 to one Q8_0 activation block.
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
        let acts_needed = blocks_per_row * Q8_0_BLOCK_BYTES;

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
            // SAFETY: all bounds checked above; AArch64 always has NEON.
            let row_sum = unsafe {
                fused_q4_0_q8_0_row_neon(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            *out_val += row_sum; // ACCUMULATE
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
        "Q4_0_Neon"
    }
}

/// Q8_0 block constants for the fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Compute fused Q4_0 weight × Q8_0 activation dot product for one row using NEON.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * Q8_0_BLOCK_BYTES`
/// - Must be called on AArch64 with NEON
unsafe fn fused_q4_0_q8_0_row_neon(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        // Weight block.
        let w_off = blk * BLOCK_BYTES;
        // SAFETY: row_data.len() == blocks_per_row * BLOCK_BYTES; blk < blocks_per_row.
        let w_block = &row_data[w_off..w_off + BLOCK_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([w_block[0], w_block[1]]));

        // Q8_0 activation block (1:1 with weight block).
        let a_off = blk * Q8_0_BLOCK_BYTES;
        // SAFETY: acts_q8.len() >= blocks_per_row * Q8_0_BLOCK_BYTES.
        let a_block = &acts_q8[a_off..a_off + Q8_0_BLOCK_BYTES];
        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));
        let scale = d_w * d_a;

        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full block. Use NEON to compute Σ (q4_w × q8_a).
            // SAFETY: w_block has 18 bytes; nibbles at [2..18] = 16 bytes. a_block has 34 bytes; q8 at [2..34] = 32 i8s.
            let nibble_ptr = w_block.as_ptr().add(2);
            let q8_ptr = a_block.as_ptr().add(2) as *const i8;

            // Load 16 nibble bytes.
            // SAFETY: nibble_ptr points to 16 valid bytes.
            let raw = unsafe { vld1q_u8(nibble_ptr) };
            let mask = unsafe { vdupq_n_u8(0x0F) };
            // SAFETY: bitwise ops on NEON registers.
            let lo = unsafe { vandq_u8(raw, mask) };
            let hi = unsafe { vshrq_n_u8::<4>(raw) };

            // Subtract 8 from nibbles → signed weights in [-8, 7].
            let offset_u8 = unsafe { vdupq_n_u8(8) };
            // Reinterpret as i8 and subtract 8 (unsigned subtract keeps values).
            // Use vreinterpretq_s8_u8 then vsubq_s8 with constant.
            let offset_s8 = unsafe { vdupq_n_s8(8) };
            let lo_s8 = unsafe { vsubq_s8(vreinterpretq_s8_u8(lo), offset_s8) };
            let hi_s8 = unsafe { vsubq_s8(vreinterpretq_s8_u8(hi), offset_s8) };

            // Load Q8_0 activation bytes (32 i8s = two int8x16_t).
            // SAFETY: q8_ptr points to 32 valid i8 values.
            let qa_lo = unsafe { vld1q_s8(q8_ptr) };
            let qa_hi = unsafe { vld1q_s8(q8_ptr.add(16)) };

            // Q4 nibbles interleave: byte i → lo nibble = weight 2i, hi nibble = weight 2i+1.
            // Q8 activations are sequential: index 0,1,2,3,...,31.
            // So lo_s8[0..16] pairs with qa[0,2,4,...,30] and hi_s8[0..16] pairs with qa[1,3,5,...,31].
            // We need to deinterleave qa into even and odd lanes.

            // Deinterleave: even (qa[0,2,4,...]) and odd (qa[1,3,5,...]).
            // SAFETY: vuzp1q_s8 / vuzp2q_s8 are always valid on AArch64.
            let qa_even = unsafe { vuzp1q_s8(qa_lo, qa_hi) }; // qa[0,2,4,...,30]
            let qa_odd = unsafe { vuzp2q_s8(qa_lo, qa_hi) }; // qa[1,3,5,...,31]

            // Multiply lo nibble weights × even activations.
            // SAFETY: vmull_s8 / vmlal_s8 are always valid on AArch64.
            let mut acc_lo = unsafe { vmull_s8(vget_low_s8(lo_s8), vget_low_s8(qa_even)) };
            acc_lo = unsafe { vmlal_s8(acc_lo, vget_high_s8(lo_s8), vget_high_s8(qa_even)) };

            // Multiply hi nibble weights × odd activations.
            // SAFETY: vmull_s8 / vmlal_s8 are always valid on AArch64.
            let mut acc_hi = unsafe { vmull_s8(vget_low_s8(hi_s8), vget_low_s8(qa_odd)) };
            acc_hi = unsafe { vmlal_s8(acc_hi, vget_high_s8(hi_s8), vget_high_s8(qa_odd)) };

            // acc_lo + acc_hi is now a int16x8_t. Widen to i32 and sum.
            let combined = unsafe { vaddq_s16(acc_lo, acc_hi) };
            // SAFETY: vpaddlq_s16 / vaddvq_s32 are always valid on AArch64.
            let i32_sum = unsafe { vpaddlq_s16(combined) };
            let dot: i32 = unsafe { vaddvq_s32(i32_sum) };

            row_sum += scale * dot as f32;

            let _ = offset_u8; // suppress unused warning
        } else if remaining > 0 {
            // Scalar tail path.
            let q8_bytes = &a_block[2..];
            let valid = remaining;

            for i in 0..(valid / 2) {
                let byte = w_block[2 + i];
                let q_lo = (byte & 0x0F) as i32 - 8;
                let q_hi = ((byte >> 4) & 0x0F) as i32 - 8;
                let a_lo = q8_bytes[i * 2] as i8 as i32;
                let a_hi = q8_bytes[i * 2 + 1] as i8 as i32;
                row_sum += scale * (q_lo * a_lo + q_hi * a_hi) as f32;
            }
            if valid % 2 == 1 {
                let i = valid / 2;
                let byte = w_block[2 + i];
                let q_lo = (byte & 0x0F) as i32 - 8;
                let a_lo = q8_bytes[i * 2] as i8 as i32;
                row_sum += scale * (q_lo * a_lo) as f32;
            }
        }
    }

    row_sum
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q4_0::{matvec_q8_fused_reference, Q4_0Ref};
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    /// Build a Q4_0 block from a scale and 16 nibble bytes.
    fn make_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    /// Fixed pseudo-random nibble pattern for reproducible tests.
    fn fixed_nibbles() -> [u8; 16] {
        // Each byte encodes two nibbles: lo = byte & 0x0F, hi = byte >> 4
        // Values span [0, 15] to exercise the full centred range [-8, 7].
        [
            0x5A, 0xF0, 0x13, 0x7E, 0xC2, 0x48, 0x9D, 0x6B, 0xA3, 0x2F, 0x71, 0xE4, 0x0C, 0x58,
            0xB6, 0xD9,
        ]
    }

    /// Fixed input vector for reproducible tests (32 elements).
    fn fixed_input() -> [f32; 32] {
        [
            0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, 1.6,
            -0.1, -0.2, 0.3, -0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5,
            -1.6,
        ]
    }

    // ── dequant_block ─────────────────────────────────────────────────────

    #[test]
    fn test_dequant_block_zeros() {
        // All nibbles = 8 → each weight = (8 − 8) * d = 0
        let block = make_block(1.0, &[0x88; 16]);
        let neon = Q4_0Neon;
        let mut out_neon = vec![0.0f32; 32];
        neon.dequant_block(&block, &mut out_neon).expect("dequant");
        for &v in &out_neon {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_matches_reference() {
        let nibbles = fixed_nibbles();
        let block = make_block(0.5, &nibbles);

        let neon = Q4_0Neon;
        let ref_k = Q4_0Ref;

        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        neon.dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        ref_k
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");

        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "dequant_block max error {max_err} >= 1e-3");
    }

    #[test]
    fn test_dequant_block_extreme_nibbles() {
        // lo nibble = 0 (→ −8), hi nibble = 15 (→ 7), all bytes = 0xF0
        let block = make_block(1.0, &[0xF0; 16]);
        let neon = Q4_0Neon;
        let ref_k = Q4_0Ref;

        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        neon.dequant_block(&block, &mut out_neon).expect("neon");
        ref_k.dequant_block(&block, &mut out_ref).expect("ref");

        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "extreme nibble error {max_err}");
    }

    // ── gemv ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gemv_zeros_output() {
        // All-zero weights: output must be zero regardless of input
        let block = make_block(1.0, &[0x88; 16]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_0);
        let input = vec![1.0f32; 32];
        let mut output = vec![9.9f32; 1];
        Q4_0Neon.gemv(&tensor, &input, &mut output).expect("gemv");
        assert!(output[0].abs() < 1e-5, "expected 0, got {}", output[0]);
    }

    #[test]
    fn test_gemv_matches_reference_single_row() {
        let nibbles = fixed_nibbles();
        let block = make_block(0.25, &nibbles);
        let input = fixed_input();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_0);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4_0Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv single row: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_gemv_matches_reference_multi_row() {
        let nibbles = fixed_nibbles();
        let n_rows = 4usize;
        let n_cols = 64usize; // 2 blocks per row
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let mut data = Vec::with_capacity(n_rows * blocks_per_row * BLOCK_BYTES);

        // Different scale per row for variety
        let scales = [0.1f32, 0.25, 0.5, 1.0];
        for &s in &scales {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&make_block(s, &nibbles));
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 - 32.0) * 0.05).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_0,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4_0Ref
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

    #[test]
    fn test_gemv_partial_block() {
        // n_cols not divisible by BLOCK_SIZE to exercise the scalar tail path
        let n_rows = 1usize;
        let n_cols = 48usize; // 1.5 blocks → 1 full block + 16 scalars
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let nibbles = fixed_nibbles();
        let mut data = Vec::with_capacity(blocks_per_row * BLOCK_BYTES);
        for _ in 0..blocks_per_row {
            data.extend_from_slice(&make_block(0.5, &nibbles));
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.1).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_0,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_0,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q4_0Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "partial block: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    // ── matvec_q8_fused ───────────────────────────────────────────────────

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
    fn neon_fused_matches_reference_single_block() {
        // Single row, single block: NEON fused must match scalar oracle.
        let nibbles = fixed_nibbles();
        let w_block = make_block(0.25, &nibbles);
        let q8_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16,
        ];
        let a_block = make_q8_0_block(0.5, &q8_vals);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_0Neon
            .matvec_q8_fused(&w_block, &a_block, &mut out_neon, 1, 32)
            .expect("neon fused");
        matvec_q8_fused_reference(&w_block, &a_block, &mut out_ref, 1, 32).expect("ref fused");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "neon_fused_matches_reference: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn neon_fused_matches_reference_multi_row() {
        // 4 rows × 64 cols (2 blocks each): NEON fused vs scalar oracle.
        let n_rows = 4usize;
        let n_cols = 64usize;
        let blocks_per_row = 2usize;
        let nibbles = fixed_nibbles();
        let scales = [0.1f32, 0.25f32, 0.5f32, 1.0f32];
        let d_a = 0.5f32;
        let q8_vals: [i8; 32] = [
            2, 4, -6, 8, -10, 12, -14, 16, 1, -3, 5, -7, 9, -11, 13, -15, 0, 1, -2, 3, -4, 5, -6,
            7, -8, 9, -10, 11, -12, 13, -14, 15,
        ];

        let mut weights: Vec<u8> = Vec::new();
        for &s in &scales {
            for _ in 0..blocks_per_row {
                weights.extend_from_slice(&make_block(s, &nibbles));
            }
        }

        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..blocks_per_row {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q4_0Neon
            .matvec_q8_fused(&weights, &acts, &mut out_neon, n_rows, n_cols)
            .expect("neon fused multi-row");
        matvec_q8_fused_reference(&weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused multi-row");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }

    #[test]
    fn neon_fused_accumulate_semantics() {
        // Out must be ADDED to (not overwritten).
        let w_block = make_block(1.0, &[0x88u8; 16]); // zero weights
        let a_block = make_q8_0_block(1.0, &[0i8; 32]);

        let mut out = vec![42.0f32; 1];
        Q4_0Neon
            .matvec_q8_fused(&w_block, &a_block, &mut out, 1, 32)
            .expect("neon fused accumulate");
        assert!(
            (out[0] - 42.0).abs() < 1e-5,
            "accumulation broken: got {}",
            out[0]
        );
    }
}
