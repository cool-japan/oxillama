//! Q5_0 NEON (AArch64) SIMD kernel.
//!
//! Q5_0 block format (22 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..6]: `qh` — 32 high bits (bit 4 of each 5-bit quant), u32 LE
//! - bytes[6..22]: `qs` — 32 × lower 4 bits packed (2 per byte)
//!
//! Weight layout (sequential, not interleaved):
//!   output\[i\]      = d * ((qs\[i\].lo4 | ((qh >> i)      & 1) << 4) - 16)  for i in 0..16
//!   output\[i + 16\] = d * ((qs\[i\].hi4 | ((qh >> (i+16)) & 1) << 4) - 16)  for i in 0..16
//!
//! Q5_0 is signed: after combining, subtract 16 to get range [-16..15].

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q5_0 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q5_0 block.
pub const BLOCK_BYTES: usize = 22;

/// NEON-accelerated Q5_0 kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q5_0Neon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Expand `qh` into two `[u8; 16]` arrays for the lo and hi halves.
///
/// `qh_lo[i]` = bit `i` of `qh` (the 5th bit for lo-half weights).
/// `qh_hi[i]` = bit `i+16` of `qh` (the 5th bit for hi-half weights).
#[inline(always)]
fn expand_qh(qh: u32) -> ([u8; 16], [u8; 16]) {
    let lo: [u8; 16] = core::array::from_fn(|i| ((qh >> i) & 1) as u8);
    let hi: [u8; 16] = core::array::from_fn(|i| ((qh >> (i + 16)) & 1) as u8);
    (lo, hi)
}

/// Dequantize one Q5_0 block using NEON intrinsics.
///
/// Produces 32 f32 values in sequential order:
///   output[0..16]  = lo-half weights (from lo nibbles + qh bits 0..15)
///   output[16..32] = hi-half weights (from hi nibbles + qh bits 16..31)
///
/// # Safety
/// Must be called on AArch64 with NEON. `qs_ptr` must point to 16 valid bytes.
/// `qh_lo` and `qh_hi` must each be 16-element arrays of 0 or 1.
/// `output` must have at least 32 slots.
#[inline]
unsafe fn dequant_block_neon(
    qs_ptr: *const u8,
    qh_lo: &[u8; 16],
    qh_hi: &[u8; 16],
    d: f32,
    output: &mut [f32],
) {
    let d_vec = unsafe { vdupq_n_f32(d) };
    let sixteen_s32 = unsafe { vdupq_n_s32(16) };

    // Load 16 packed nibble bytes
    let raw = unsafe { vld1q_u8(qs_ptr) };
    let mask = unsafe { vdupq_n_u8(0x0F) };
    let lo_nib = unsafe { vandq_u8(raw, mask) };
    let hi_nib = unsafe { vshrq_n_u8::<4>(raw) };

    // Load qh high-bit arrays
    let vqh_lo = unsafe { vld1q_u8(qh_lo.as_ptr()) };
    let vqh_hi = unsafe { vld1q_u8(qh_hi.as_ptr()) };

    // Shift high bit into position 4: qh_bit << 4
    let shift4 = unsafe { vdupq_n_u8(4) };
    let qh_lo_shifted = unsafe { vshlq_u8(vqh_lo, vreinterpretq_s8_u8(shift4)) };
    let qh_hi_shifted = unsafe { vshlq_u8(vqh_hi, vreinterpretq_s8_u8(shift4)) };

    // Combine: 5-bit unsigned quant for each element
    let q5_lo = unsafe { vorrq_u8(lo_nib, qh_lo_shifted) };
    let q5_hi = unsafe { vorrq_u8(hi_nib, qh_hi_shifted) };

    // Widen to u16 then u32, subtract 16 (signed), convert to f32, scale by d
    // --- lo-half (output[0..16]) ---
    let q5_lo_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_lo)) };
    let q5_lo_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_lo)) };

    let a = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_low))),
            sixteen_s32,
        )
    };
    let b = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_low)),
            sixteen_s32,
        )
    };
    let c = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_high))),
            sixteen_s32,
        )
    };
    let e = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_high)),
            sixteen_s32,
        )
    };

    let fa = unsafe { vmulq_f32(vcvtq_f32_s32(a), d_vec) };
    let fb = unsafe { vmulq_f32(vcvtq_f32_s32(b), d_vec) };
    let fc = unsafe { vmulq_f32(vcvtq_f32_s32(c), d_vec) };
    let fe = unsafe { vmulq_f32(vcvtq_f32_s32(e), d_vec) };

    // --- hi-half (output[16..32]) ---
    let q5_hi_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_hi)) };
    let q5_hi_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_hi)) };

    let g = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_low))),
            sixteen_s32,
        )
    };
    let h = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_low)),
            sixteen_s32,
        )
    };
    let j = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_high))),
            sixteen_s32,
        )
    };
    let k = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_high)),
            sixteen_s32,
        )
    };

    let fg = unsafe { vmulq_f32(vcvtq_f32_s32(g), d_vec) };
    let fh = unsafe { vmulq_f32(vcvtq_f32_s32(h), d_vec) };
    let fj = unsafe { vmulq_f32(vcvtq_f32_s32(j), d_vec) };
    let fk = unsafe { vmulq_f32(vcvtq_f32_s32(k), d_vec) };

    // Store: sequential layout, lo-half first then hi-half
    unsafe { vst1q_f32(output.as_mut_ptr(), fa) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), fb) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), fc) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), fe) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), fg) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), fh) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), fj) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), fk) };
}

/// Compute dot product for one Q5_0 block against a contiguous f32 input slice.
///
/// Input layout matches the sequential output layout: input[0..16] pairs with
/// lo-half weights, input[16..32] pairs with hi-half weights.
///
/// # Safety
/// Must be called on AArch64 with NEON. `qs_ptr` must point to 16 valid bytes.
/// `input` must have exactly 32 elements.
#[inline]
unsafe fn dot_block_neon(
    qs_ptr: *const u8,
    qh_lo: &[u8; 16],
    qh_hi: &[u8; 16],
    d: f32,
    input: &[f32],
) -> f32 {
    let d_vec = unsafe { vdupq_n_f32(d) };
    let sixteen_s32 = unsafe { vdupq_n_s32(16) };

    let raw = unsafe { vld1q_u8(qs_ptr) };
    let mask = unsafe { vdupq_n_u8(0x0F) };
    let lo_nib = unsafe { vandq_u8(raw, mask) };
    let hi_nib = unsafe { vshrq_n_u8::<4>(raw) };

    let vqh_lo = unsafe { vld1q_u8(qh_lo.as_ptr()) };
    let vqh_hi = unsafe { vld1q_u8(qh_hi.as_ptr()) };

    let shift4 = unsafe { vdupq_n_u8(4) };
    let qh_lo_shifted = unsafe { vshlq_u8(vqh_lo, vreinterpretq_s8_u8(shift4)) };
    let qh_hi_shifted = unsafe { vshlq_u8(vqh_hi, vreinterpretq_s8_u8(shift4)) };

    let q5_lo = unsafe { vorrq_u8(lo_nib, qh_lo_shifted) };
    let q5_hi = unsafe { vorrq_u8(hi_nib, qh_hi_shifted) };

    let q5_lo_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_lo)) };
    let q5_lo_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_lo)) };
    let q5_hi_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_hi)) };
    let q5_hi_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_hi)) };

    let a = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_low))),
            sixteen_s32,
        )
    };
    let b = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_low)),
            sixteen_s32,
        )
    };
    let c = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_high))),
            sixteen_s32,
        )
    };
    let e = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_high)),
            sixteen_s32,
        )
    };
    let g = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_low))),
            sixteen_s32,
        )
    };
    let h = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_low)),
            sixteen_s32,
        )
    };
    let j = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_high))),
            sixteen_s32,
        )
    };
    let k = unsafe {
        vsubq_s32(
            vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_high)),
            sixteen_s32,
        )
    };

    let qf_a = unsafe { vcvtq_f32_s32(a) };
    let qf_b = unsafe { vcvtq_f32_s32(b) };
    let qf_c = unsafe { vcvtq_f32_s32(c) };
    let qf_e = unsafe { vcvtq_f32_s32(e) };
    let qf_g = unsafe { vcvtq_f32_s32(g) };
    let qf_h = unsafe { vcvtq_f32_s32(h) };
    let qf_j = unsafe { vcvtq_f32_s32(j) };
    let qf_k = unsafe { vcvtq_f32_s32(k) };

    // Sequential input layout: [0..4, 4..8, 8..12, 12..16] for lo-half,
    //                          [16..20, 20..24, 24..28, 28..32] for hi-half
    let ip = input.as_ptr();
    let i0 = unsafe { vld1q_f32(ip) };
    let i1 = unsafe { vld1q_f32(ip.add(4)) };
    let i2 = unsafe { vld1q_f32(ip.add(8)) };
    let i3 = unsafe { vld1q_f32(ip.add(12)) };
    let i4 = unsafe { vld1q_f32(ip.add(16)) };
    let i5 = unsafe { vld1q_f32(ip.add(20)) };
    let i6 = unsafe { vld1q_f32(ip.add(24)) };
    let i7 = unsafe { vld1q_f32(ip.add(28)) };

    let mut acc = unsafe { vmulq_f32(qf_a, i0) };
    acc = unsafe { vfmaq_f32(acc, qf_b, i1) };
    acc = unsafe { vfmaq_f32(acc, qf_c, i2) };
    acc = unsafe { vfmaq_f32(acc, qf_e, i3) };
    acc = unsafe { vfmaq_f32(acc, qf_g, i4) };
    acc = unsafe { vfmaq_f32(acc, qf_h, i5) };
    acc = unsafe { vfmaq_f32(acc, qf_j, i6) };
    acc = unsafe { vfmaq_f32(acc, qf_k, i7) };

    let _ = d_vec;
    d * unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q5_0Neon {
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
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let (qh_lo, qh_hi) = expand_qh(qh);
        unsafe {
            dequant_block_neon(
                block.as_ptr().add(6),
                &qh_lo,
                &qh_hi,
                d,
                &mut output[..BLOCK_SIZE],
            )
        };
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
                let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                let (qh_lo, qh_hi) = expand_qh(qh);
                let input_offset = blk * BLOCK_SIZE;
                let block_input_end = (input_offset + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_offset;

                if block_input_len == BLOCK_SIZE {
                    sum += unsafe {
                        dot_block_neon(
                            block.as_ptr().add(6),
                            &qh_lo,
                            &qh_hi,
                            d,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Scalar tail for partial blocks (sequential layout)
                    for i in 0..block_input_len {
                        let byte = block[6 + i / 2];
                        let nibble = if i < 16 {
                            byte & 0x0F
                        } else {
                            (byte >> 4) & 0x0F
                        };
                        let hi_bit = ((qh >> i) & 1) as u8;
                        let q5 = (nibble | (hi_bit << 4)) as i32 - 16;
                        sum += q5 as f32 * d * input[input_offset + i];
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

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q5_0_Neon"
    }

    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> crate::error::QuantResult<()> {
        use crate::error::QuantError;

        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        let q8_block_bytes: usize = 34;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < blocks_per_row * q8_block_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: blocks_per_row * q8_block_bytes,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let partial = unsafe {
                fused_q5_0_q8_0_row_neon(
                    &weights[row_start..row_start + row_bytes],
                    acts_q8,
                    blocks_per_row,
                    n_cols,
                )
            };
            *out_val += partial;
        }

        Ok(())
    }
}

/// Fused Q5_0 weight × Q8_0 activation dot product for one matrix row.
///
/// Decodes each Q5_0 block's 5-bit signed weights and the corresponding Q8_0
/// activation block (2-byte FP16 scale + 32 i8 values), then accumulates using
/// NEON vector multiply-accumulate.
///
/// # Safety
/// Must be called on AArch64 with NEON. All slice bounds must be pre-validated.
unsafe fn fused_q5_0_q8_0_row_neon(
    weights_row: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    const Q8_BLOCK_BYTES: usize = 34;
    let mut acc = vdupq_n_f32(0.0f32);

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &weights_row[bo..bo + BLOCK_BYTES];
        let col_start = blk * BLOCK_SIZE;

        let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let (qh_lo, qh_hi) = expand_qh(qh);

        // Decode Q8_0 activation block
        let ab = blk * Q8_BLOCK_BYTES;
        let a_block = &acts_q8[ab..ab + Q8_BLOCK_BYTES];
        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));

        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let avail = col_end - col_start;

        if avail == BLOCK_SIZE {
            // Full block: use NEON for both halves.
            // Decode Q5_0 to scratch buffer, then dot with scaled activations.
            let qs_ptr = block.as_ptr().add(6);
            let raw = vld1q_u8(qs_ptr);
            let mask = vdupq_n_u8(0x0F);
            let lo_nib = vandq_u8(raw, mask);
            let hi_nib = vshrq_n_u8::<4>(raw);

            let vqh_lo = vld1q_u8(qh_lo.as_ptr());
            let vqh_hi = vld1q_u8(qh_hi.as_ptr());
            let shift4 = vdupq_n_u8(4);
            let qh_lo_sh = vshlq_u8(vqh_lo, vreinterpretq_s8_u8(shift4));
            let qh_hi_sh = vshlq_u8(vqh_hi, vreinterpretq_s8_u8(shift4));
            let q5_lo = vorrq_u8(lo_nib, qh_lo_sh);
            let q5_hi = vorrq_u8(hi_nib, qh_hi_sh);

            let sixteen_s32 = vdupq_n_s32(16);
            let q5_lo_u16_low = vmovl_u8(vget_low_u8(q5_lo));
            let q5_lo_u16_high = vmovl_u8(vget_high_u8(q5_lo));
            let q5_hi_u16_low = vmovl_u8(vget_low_u8(q5_hi));
            let q5_hi_u16_high = vmovl_u8(vget_high_u8(q5_hi));

            // Signed int32 weight quants (subtract 16)
            let wa = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_low))),
                sixteen_s32,
            );
            let wb = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_low)),
                sixteen_s32,
            );
            let wc = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_high))),
                sixteen_s32,
            );
            let we = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_high_u16(q5_lo_u16_high)),
                sixteen_s32,
            );
            let wg = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_low))),
                sixteen_s32,
            );
            let wh = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_low)),
                sixteen_s32,
            );
            let wj = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_high))),
                sixteen_s32,
            );
            let wk = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_high_u16(q5_hi_u16_high)),
                sixteen_s32,
            );

            // Float weight groups scaled by d_w
            let d_w_vec = vdupq_n_f32(d_w);
            let wfa = vmulq_f32(vcvtq_f32_s32(wa), d_w_vec);
            let wfb = vmulq_f32(vcvtq_f32_s32(wb), d_w_vec);
            let wfc = vmulq_f32(vcvtq_f32_s32(wc), d_w_vec);
            let wfe = vmulq_f32(vcvtq_f32_s32(we), d_w_vec);
            let wfg = vmulq_f32(vcvtq_f32_s32(wg), d_w_vec);
            let wfh = vmulq_f32(vcvtq_f32_s32(wh), d_w_vec);
            let wfj = vmulq_f32(vcvtq_f32_s32(wj), d_w_vec);
            let wfk = vmulq_f32(vcvtq_f32_s32(wk), d_w_vec);

            // Load and scale Q8_0 activation i8 values by d_a
            let q8_ptr = a_block.as_ptr().add(2) as *const i8;
            let aq0 = vld1q_s8(q8_ptr);
            let aq1 = vld1q_s8(q8_ptr.add(16));

            let aq0_lo = vmovl_s8(vget_low_s8(aq0));
            let aq0_hi = vmovl_s8(vget_high_s8(aq0));
            let aq1_lo = vmovl_s8(vget_low_s8(aq1));
            let aq1_hi = vmovl_s8(vget_high_s8(aq1));

            let d_a_vec = vdupq_n_f32(d_a);
            let aa0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(aq0_lo))), d_a_vec);
            let aa1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(aq0_lo)), d_a_vec);
            let aa2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(aq0_hi))), d_a_vec);
            let aa3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(aq0_hi)), d_a_vec);
            let aa4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(aq1_lo))), d_a_vec);
            let aa5 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(aq1_lo)), d_a_vec);
            let aa6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(aq1_hi))), d_a_vec);
            let aa7 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(aq1_hi)), d_a_vec);

            // FMA: acc += w * a (sequential weight layout: lo-half first, then hi-half)
            acc = vfmaq_f32(acc, wfa, aa0);
            acc = vfmaq_f32(acc, wfb, aa1);
            acc = vfmaq_f32(acc, wfc, aa2);
            acc = vfmaq_f32(acc, wfe, aa3);
            acc = vfmaq_f32(acc, wfg, aa4);
            acc = vfmaq_f32(acc, wfh, aa5);
            acc = vfmaq_f32(acc, wfj, aa6);
            acc = vfmaq_f32(acc, wfk, aa7);
        } else {
            // Scalar tail for partial blocks
            let qs = &block[6..22];
            let q8_vals = &a_block[2..];
            let mut block_dot = 0.0f32;
            for i in 0..avail {
                let byte = qs[i / 2];
                let nibble = if i < 16 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let hi_bit = ((qh >> i) & 1) as u8;
                let q5 = (nibble | (hi_bit << 4)) as i32 - 16;
                let a_val = (q8_vals[i] as i8) as f32 * d_a;
                block_dot += d_w * q5 as f32 * a_val;
            }
            let scalar_vec = vdupq_n_f32(block_dot);
            // Accumulate into lane 0 only via scalar
            let lane0 = vgetq_lane_f32::<0>(acc) + block_dot;
            acc = vdupq_n_f32(0.0f32);
            acc = vsetq_lane_f32::<0>(lane0, acc);
            let _ = scalar_vec;
        }
    }

    hsum_f32x4(acc)
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q5_0::Q5_0Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, 0, &[0; 16]);
        let mut out = vec![0.0f32; 32];
        Q5_0Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_all_bits_set() {
        // qh=0xFFFFFFFF, qs=0xFF → q5 = 31 → 31-16 = 15; weight = d*15
        let block = make_block(1.0, 0xFFFF_FFFF, &[0xFF; 16]);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q5_0Neon.dequant_block(&block, &mut out_neon).expect("neon");
        Q5_0Ref.dequant_block(&block, &mut out_ref).expect("ref");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "max err {max_err}");
    }

    #[test]
    fn test_dequant_matches_reference() {
        let qh: u32 = 0xA5A5_A5A5;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, qh, &qs);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q5_0Neon.dequant_block(&block, &mut out_neon).expect("neon");
        Q5_0Ref.dequant_block(&block, &mut out_ref).expect("ref");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "dequant max error {max_err}; neon[0]={} ref[0]={}",
            out_neon[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_matches_reference() {
        let qh: u32 = 0x1234_5678;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 13 + 7) & 0xFF) as u8;
        }
        let block = make_block(0.25, qh, &qs);
        let n_cols = BLOCK_SIZE;
        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Q5_0,
        );
        let tensor_ref =
            QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q5_0);
        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];
        Q5_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q5_0Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref");
        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "gemv: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn fused_q5_0_neon_matches_reference() {
        let qh: u32 = 0xA5A5_A5A5;
        let mut qs_w = [0u8; 16];
        for (i, v) in qs_w.iter_mut().enumerate() {
            *v = ((i * 11 + 5) & 0xFF) as u8;
        }
        let weight_block = make_block(0.5, qh, &qs_w);

        // Build Q8_0 activation block (34 bytes)
        let d_a = 0.25f32;
        let mut acts_raw = [0i8; 32];
        for (i, v) in acts_raw.iter_mut().enumerate() {
            *v = ((i as i16 * 5 - 40).clamp(-128, 127)) as i8;
        }
        let mut acts_block = Vec::with_capacity(34);
        acts_block.extend_from_slice(&half::f16::from_f32(d_a).to_bits().to_le_bytes());
        for &v in &acts_raw {
            acts_block.push(v as u8);
        }

        // Reference: dequant weight + dot with scaled activations
        let mut w_dequant = vec![0.0f32; BLOCK_SIZE];
        Q5_0Ref
            .dequant_block(&weight_block, &mut w_dequant)
            .expect("ref dequant");
        let acts_f32: Vec<f32> = acts_raw.iter().map(|&v| v as f32 * d_a).collect();
        let expected: f32 = w_dequant
            .iter()
            .zip(acts_f32.iter())
            .map(|(w, a)| w * a)
            .sum();

        // NEON fused
        let mut out_neon = vec![0.0f32; 1];
        Q5_0Neon
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_neon, 1, BLOCK_SIZE)
            .expect("neon fused");

        let err = (out_neon[0] - expected).abs();
        assert!(
            err < 0.1,
            "fused_q5_0_neon: got={} expected={} err={}",
            out_neon[0],
            expected,
            err
        );

        // Also verify against reference Q5_0Ref::matvec_q8_fused
        let mut out_ref = vec![0.0f32; 1];
        Q5_0Ref
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_ref, 1, BLOCK_SIZE)
            .expect("ref fused");
        let ref_err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            ref_err < 0.1,
            "fused: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            ref_err
        );
    }
}
