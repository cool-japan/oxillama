//! Q2_K NEON (AArch64) SIMD kernel.
//!
//! Q2_K super-block format (84 bytes per 256 weights):
//! - bytes[0..16]:  scales — 16 sub-block bytes (lo 4 bits = scale, hi 4 bits = min)
//! - bytes[16..80]: qs    — 64 bytes, 256 × 2-bit packed (4 per byte via shifts 0,2,4,6)
//! - bytes[80..82]: FP16 super-block scale `d`
//! - bytes[82..84]: FP16 super-block minimum `dmin`
//!
//! 16 sub-blocks of 16 weights each (2 groups of 128, each group re-uses the
//! same 32 qs bytes with shift 0,2,4,6 to extract all 2-bit fields).
//! Weight formula: `w = d * scale_i * q - dmin * min_i`, q ∈ [0..3].

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q2_K super-block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q2_K super-block.
pub const BLOCK_BYTES: usize = 84;

/// NEON-accelerated Q2_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q2_KNeon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Extract 2-bit fields from 16 packed bytes using the given bit-shift.
///
/// Each source byte packs 4 × 2-bit fields at positions 0..1, 2..3, 4..5, 6..7.
/// `shift` selects which field (0, 2, 4, or 6). Result bytes are in [0..3].
#[inline(always)]
unsafe fn extract_2bit_neon(raw: uint8x16_t, shift: u32) -> uint8x16_t {
    let mask = unsafe { vdupq_n_u8(0x03) };
    let shifted = match shift {
        0 => raw,
        2 => unsafe { vshrq_n_u8::<2>(raw) },
        4 => unsafe { vshrq_n_u8::<4>(raw) },
        _ => unsafe { vshrq_n_u8::<6>(raw) },
    };
    unsafe { vandq_u8(shifted, mask) }
}

/// Dequantize 16 weights from a 2-bit sub-block using NEON.
///
/// Writes 16 f32 values: `dl * q - ml` for each weight.
///
/// # Safety
/// `qs_ptr` must point to 16 valid bytes. `out_ptr` must have at least 16 f32 slots.
#[inline]
unsafe fn dequant_16_weights(qs_raw: uint8x16_t, shift: u32, dl: f32, ml: f32, out_ptr: *mut f32) {
    let q_bytes = unsafe { extract_2bit_neon(qs_raw, shift) };

    let dl_vec = unsafe { vdupq_n_f32(dl) };
    let ml_vec = unsafe { vdupq_n_f32(ml) };

    // Widen u8 → u16 → u32 → f32 (two halves)
    let q_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q_bytes)) };
    let q_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q_bytes)) };

    // First 4 weights
    let q0 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo_u16))) };
    // Second 4 weights
    let q1 = unsafe { vcvtq_f32_u32(vmovl_high_u16(q_lo_u16)) };
    // Third 4 weights
    let q2 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi_u16))) };
    // Fourth 4 weights
    let q3 = unsafe { vcvtq_f32_u32(vmovl_high_u16(q_hi_u16)) };

    // w = dl * q - ml
    let w0 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q0), ml_vec) };
    let w1 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q1), ml_vec) };
    let w2 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q2), ml_vec) };
    let w3 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q3), ml_vec) };

    unsafe { vst1q_f32(out_ptr, w0) };
    unsafe { vst1q_f32(out_ptr.add(4), w1) };
    unsafe { vst1q_f32(out_ptr.add(8), w2) };
    unsafe { vst1q_f32(out_ptr.add(12), w3) };
}

/// Dot-product contribution from 16 weights × 16 inputs using NEON.
///
/// Returns `Σ (dl * q[i] - ml) * inp[i]`.
///
/// # Safety
/// `inp_ptr` must point to 16 valid f32 values. `qs_raw` is a loaded u8x16.
#[inline]
unsafe fn dot_16_weights(
    qs_raw: uint8x16_t,
    shift: u32,
    dl: f32,
    ml: f32,
    inp_ptr: *const f32,
) -> f32 {
    let q_bytes = unsafe { extract_2bit_neon(qs_raw, shift) };

    let dl_vec = unsafe { vdupq_n_f32(dl) };
    let ml_vec = unsafe { vdupq_n_f32(ml) };

    let q_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q_bytes)) };
    let q_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q_bytes)) };

    let q0 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo_u16))) };
    let q1 = unsafe { vcvtq_f32_u32(vmovl_high_u16(q_lo_u16)) };
    let q2 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi_u16))) };
    let q3 = unsafe { vcvtq_f32_u32(vmovl_high_u16(q_hi_u16)) };

    // weights = dl * q - ml
    let w0 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q0), ml_vec) };
    let w1 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q1), ml_vec) };
    let w2 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q2), ml_vec) };
    let w3 = unsafe { vsubq_f32(vmulq_f32(dl_vec, q3), ml_vec) };

    // Load inputs
    let i0 = unsafe { vld1q_f32(inp_ptr) };
    let i1 = unsafe { vld1q_f32(inp_ptr.add(4)) };
    let i2 = unsafe { vld1q_f32(inp_ptr.add(8)) };
    let i3 = unsafe { vld1q_f32(inp_ptr.add(12)) };

    // Dot product
    let mut acc = unsafe { vmulq_f32(w0, i0) };
    acc = unsafe { vfmaq_f32(acc, w1, i1) };
    acc = unsafe { vfmaq_f32(acc, w2, i2) };
    acc = unsafe { vfmaq_f32(acc, w3, i3) };

    unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q2_KNeon {
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

        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut is = 0usize;
        let mut out_off = 0usize;

        for group in 0..2usize {
            let qs_base = group * 32;
            let raw_a = unsafe { vld1q_u8(qs.as_ptr().add(qs_base)) };
            let raw_b = unsafe { vld1q_u8(qs.as_ptr().add(qs_base + 16)) };

            for shift in [0u32, 2, 4, 6] {
                let sc_a = scales[is];
                is += 1;
                let dl_a = d * (sc_a & 0x0F) as f32;
                let ml_a = dmin * (sc_a >> 4) as f32;

                unsafe {
                    dequant_16_weights(raw_a, shift, dl_a, ml_a, output.as_mut_ptr().add(out_off))
                };
                out_off += 16;

                let sc_b = scales[is];
                is += 1;
                let dl_b = d * (sc_b & 0x0F) as f32;
                let ml_b = dmin * (sc_b >> 4) as f32;

                unsafe {
                    dequant_16_weights(raw_b, shift, dl_b, ml_b, output.as_mut_ptr().add(out_off))
                };
                out_off += 16;
            }
        }

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
                let scales = &block[0..16];
                let qs = &block[16..80];
                let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
                let input_offset = blk * BLOCK_SIZE;
                let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

                if cols_in_block == BLOCK_SIZE {
                    let mut is = 0usize;
                    let mut w_off = input_offset;

                    for group in 0..2usize {
                        let qs_base = group * 32;
                        let raw_a = unsafe { vld1q_u8(qs.as_ptr().add(qs_base)) };
                        let raw_b = unsafe { vld1q_u8(qs.as_ptr().add(qs_base + 16)) };

                        for shift in [0u32, 2, 4, 6] {
                            let sc_a = scales[is];
                            is += 1;
                            let dl_a = d * (sc_a & 0x0F) as f32;
                            let ml_a = dmin * (sc_a >> 4) as f32;
                            sum += unsafe {
                                dot_16_weights(raw_a, shift, dl_a, ml_a, input.as_ptr().add(w_off))
                            };
                            w_off += 16;

                            let sc_b = scales[is];
                            is += 1;
                            let dl_b = d * (sc_b & 0x0F) as f32;
                            let ml_b = dmin * (sc_b >> 4) as f32;
                            sum += unsafe {
                                dot_16_weights(raw_b, shift, dl_b, ml_b, input.as_ptr().add(w_off))
                            };
                            w_off += 16;
                        }
                    }
                } else {
                    // Scalar tail for partial blocks
                    let inp = &input[input_offset..];
                    let mut is = 0usize;
                    let mut qs_off = 0usize;
                    let mut in_off = 0usize;

                    for _n in 0..2 {
                        for shift in (0..8usize).step_by(2) {
                            let sc_a = scales[is];
                            is += 1;
                            let dl_a = d * (sc_a & 0x0F) as f32;
                            let ml_a = dmin * (sc_a >> 4) as f32;
                            for l in 0..16 {
                                if in_off + l < cols_in_block {
                                    let q = (qs[qs_off + l] >> shift) & 3;
                                    sum += (dl_a * q as f32 - ml_a) * inp[in_off + l];
                                }
                            }
                            in_off += 16;

                            let sc_b = scales[is];
                            is += 1;
                            let dl_b = d * (sc_b & 0x0F) as f32;
                            let ml_b = dmin * (sc_b >> 4) as f32;
                            for l in 0..16 {
                                if in_off + l < cols_in_block {
                                    let q = (qs[qs_off + 16 + l] >> shift) & 3;
                                    sum += (dl_b * q as f32 - ml_b) * inp[in_off + l];
                                }
                            }
                            in_off += 16;
                        }
                        qs_off += 32;
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

    /// Fused Q2_K weight × Q8_0 activation GEMV using NEON integer dot products.
    ///
    /// Each Q2_K super-block (256 weights, 84 bytes) maps to 8 Q8_0 activation blocks
    /// (32 weights each, 34 bytes each).  The 256 weights are organized as 16 sub-blocks
    /// of 16 weights.  Two consecutive sub-blocks share one Q8_0 block of 32 activations.
    ///
    /// Fused formula per sub-block:
    ///   contribution = (dl * Σ(q2_i * q8_i) - ml * Σ(q8_i)) * d_a
    /// avoiding any intermediate f32 scratch buffer.
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
            // SAFETY: bounds checked above; AArch64 always has NEON.
            let row_sum = unsafe {
                fused_q2k_q8_0_row_neon(
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

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q2_K_Neon"
    }
}

/// Q8_0 block bytes for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Fused Q2_K weight × Q8_0 activation dot product for one row using NEON integer arithmetic.
///
/// Q2_K layout: 16 scales (nibble-packed), 64 qs (2-bit × 4 per byte), then d/dmin.
/// 256 weights split into 16 sub-blocks of 16 weights.  Sub-block `s` (0..16) maps
/// to Q8_0 block at index `blk*8 + s/2` (two sub-blocks share one Q8_0 block).
///
/// For sub-block `s` with Q8_0 block `q8_blk`:
///   contribution = (dl * Σ(q2_i * q8_i) - ml * Σ(q8_i)) * d_a
///
/// Uses NEON vmull_u8/vmlal_u8 for integer dot products, avoiding f32 materialization.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * 8 * Q8_0_BLOCK_BYTES`
/// - Must be called on AArch64 with NEON
unsafe fn fused_q2k_q8_0_row_neon(
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

        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let input_offset = blk * BLOCK_SIZE;
        let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

        if cols_in_block < BLOCK_SIZE {
            // Scalar tail path — partial block.
            let mut is = 0usize;
            let mut col_off = 0usize;

            for _group in 0..2usize {
                let qs_base = _group * 32;
                for shift in [0u32, 2, 4, 6] {
                    // Sub-block A
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));
                        let q8_vals = &a_block[2..];
                        let mut dot = 0i32;
                        let mut sum_a = 0i32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q2 = ((qs[qs_base + l] >> shift) & 3) as i32;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as i32;
                                dot += q2 * q_a;
                                sum_a += q_a;
                            }
                        }
                        row_sum += (dl * dot as f32 - ml * sum_a as f32) * d_a;
                        col_off += 16;
                    }
                    // Sub-block B
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));
                        let q8_vals = &a_block[2..];
                        let mut dot = 0i32;
                        let mut sum_a = 0i32;
                        for l in 0..16 {
                            if col_off + l < cols_in_block {
                                let q2 = ((qs[qs_base + 16 + l] >> shift) & 3) as i32;
                                let q_a = q8_vals[q8_lane_base + l] as i8 as i32;
                                dot += q2 * q_a;
                                sum_a += q_a;
                            }
                        }
                        row_sum += (dl * dot as f32 - ml * sum_a as f32) * d_a;
                        col_off += 16;
                    }
                }
            }
        } else {
            // Fast path: full 256-weight block.
            // Process 16 sub-blocks of 16 weights each.
            // Two sub-blocks (A+B) share one Q8_0 block of 32 activations.
            // We pair (A, B) so each pair operates on one 32-element Q8_0 block.
            // Sub-block A occupies q8_lane 0..16; sub-block B occupies lane 16..32.
            let mut is = 0usize;
            let mut col_off = 0usize;

            for group in 0..2usize {
                let qs_base = group * 32;
                // Pre-load 32 qs bytes (two 16-byte halves).
                // SAFETY: qs_base + 32 <= 64; qs.len() == 64.
                let raw_a = unsafe { vld1q_u8(qs.as_ptr().add(qs_base)) };
                let raw_b = unsafe { vld1q_u8(qs.as_ptr().add(qs_base + 16)) };

                for shift in [0u32, 2, 4, 6] {
                    // SAFETY: extract_2bit_neon valid on AArch64.
                    let q2_a = unsafe { extract_2bit_neon(raw_a, shift) }; // u8x16, values 0..3
                    let q2_b = unsafe { extract_2bit_neon(raw_b, shift) }; // u8x16, values 0..3

                    // Sub-block A: lanes 0..16 of Q8_0 block at col_off/32
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));

                        // Load 16 Q8_0 i8 activations at lane base.
                        // SAFETY: a_block has 34 bytes; 2 + q8_lane_base + 16 <= 34.
                        let qa_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const i8;
                        let qa_i8 = unsafe { vld1q_s8(qa_ptr) };

                        // Widen q2 u8 → i16, widen q8 i8 → i16, integer dot product.
                        let q2_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q2_a)) };
                        let q2_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q2_a)) };
                        let q2_lo_i16 = unsafe { vreinterpretq_s16_u16(q2_lo_u16) };
                        let q2_hi_i16 = unsafe { vreinterpretq_s16_u16(q2_hi_u16) };
                        let qa_lo_i16 = unsafe { vmovl_s8(vget_low_s8(qa_i8)) };
                        let qa_hi_i16 = unsafe { vmovl_s8(vget_high_s8(qa_i8)) };

                        // dot = Σ(q2 * q8) as i32
                        let p0 =
                            unsafe { vmull_s16(vget_low_s16(q2_lo_i16), vget_low_s16(qa_lo_i16)) };
                        let p1 = unsafe {
                            vmlal_s16(p0, vget_high_s16(q2_lo_i16), vget_high_s16(qa_lo_i16))
                        };
                        let p2 = unsafe {
                            vmlal_s16(p1, vget_low_s16(q2_hi_i16), vget_low_s16(qa_hi_i16))
                        };
                        let p3 = unsafe {
                            vmlal_s16(p2, vget_high_s16(q2_hi_i16), vget_high_s16(qa_hi_i16))
                        };
                        let dot = unsafe { vaddvq_s32(p3) };

                        // sum_a = Σ(q8) as i32
                        let s0 =
                            unsafe { vaddl_s16(vget_low_s16(qa_lo_i16), vget_high_s16(qa_lo_i16)) };
                        let s1 =
                            unsafe { vaddl_s16(vget_low_s16(qa_hi_i16), vget_high_s16(qa_hi_i16)) };
                        let sum_a = unsafe { vaddvq_s32(vaddq_s32(s0, s1)) };

                        row_sum += (dl * dot as f32 - ml * sum_a as f32) * d_a;
                        col_off += 16;
                    }

                    // Sub-block B: lanes 16..32 of Q8_0 block at col_off/32
                    {
                        let sc_byte = scales[is];
                        is += 1;
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        let q8_blk_idx = blk * 8 + col_off / 32;
                        let q8_lane_base = col_off % 32;
                        let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                        // SAFETY: acts_q8.len() >= blocks_per_row*8*Q8_0_BLOCK_BYTES.
                        let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));

                        let qa_ptr = a_block.as_ptr().add(2 + q8_lane_base) as *const i8;
                        let qa_i8 = unsafe { vld1q_s8(qa_ptr) };

                        let q2_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q2_b)) };
                        let q2_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q2_b)) };
                        let q2_lo_i16 = unsafe { vreinterpretq_s16_u16(q2_lo_u16) };
                        let q2_hi_i16 = unsafe { vreinterpretq_s16_u16(q2_hi_u16) };
                        let qa_lo_i16 = unsafe { vmovl_s8(vget_low_s8(qa_i8)) };
                        let qa_hi_i16 = unsafe { vmovl_s8(vget_high_s8(qa_i8)) };

                        let p0 =
                            unsafe { vmull_s16(vget_low_s16(q2_lo_i16), vget_low_s16(qa_lo_i16)) };
                        let p1 = unsafe {
                            vmlal_s16(p0, vget_high_s16(q2_lo_i16), vget_high_s16(qa_lo_i16))
                        };
                        let p2 = unsafe {
                            vmlal_s16(p1, vget_low_s16(q2_hi_i16), vget_low_s16(qa_hi_i16))
                        };
                        let p3 = unsafe {
                            vmlal_s16(p2, vget_high_s16(q2_hi_i16), vget_high_s16(qa_hi_i16))
                        };
                        let dot = unsafe { vaddvq_s32(p3) };

                        let s0 =
                            unsafe { vaddl_s16(vget_low_s16(qa_lo_i16), vget_high_s16(qa_lo_i16)) };
                        let s1 =
                            unsafe { vaddl_s16(vget_low_s16(qa_hi_i16), vget_high_s16(qa_hi_i16)) };
                        let sum_a = unsafe { vaddvq_s32(vaddq_s32(s0, s1)) };

                        row_sum += (dl * dot as f32 - ml * sum_a as f32) * d_a;
                        col_off += 16;
                    }
                }
            }
        }
    }

    row_sum
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q2_k::Q2KRef;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, dmin: f32, scales: &[u8; 16], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let mut out = vec![0.0f32; 256];
        Q2_KNeon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_matches_reference() {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0x21 + i as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, &scales, &qs);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];
        Q2_KNeon.dequant_block(&block, &mut out_neon).expect("neon");
        Q2KRef.dequant_block(&block, &mut out_ref).expect("ref");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "dequant max error {max_err}; neon[0]={} ref[0]={}",
            out_neon[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_dequant_uniform() {
        // d=1.0, dmin=0.0, scales=0x01 (scale=1, min=0), qs=0xFF → all q=3 → w=3.0
        let block = make_block(1.0, 0.0, &[0x01; 16], &[0xFF; 64]);
        let mut out = vec![0.0f32; 256];
        Q2_KNeon.dequant_block(&block, &mut out).expect("dequant");
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_gemv_zeros() {
        let block = make_block(1.0, 0.0, &[0; 16], &[0; 64]);
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q2K);
        let input = vec![1.0f32; 256];
        let mut out = vec![9.9f32; 1];
        Q2_KNeon.gemv(&tensor, &input, &mut out).expect("gemv");
        assert!(out[0].abs() < 1e-3, "expected ~0, got {}", out[0]);
    }

    #[test]
    fn test_gemv_matches_reference() {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0x21 + i as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, &scales, &qs);
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Q2K,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q2K);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];
        Q2_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q2KRef.gemv(&tensor_ref, &input, &mut out_ref).expect("ref");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 0.1,
            "gemv: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    // ── matvec_q8_fused Q2_K (NEON) ──────────────────────────────────────

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
    fn test_q2k_neon_fused_matches_reference() {
        // One Q2_K super-block (256 weights) needs 8 Q8_0 activation blocks.
        let mut scales = [0u8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 5) & 0xFF) as u8;
        }
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 3) & 0xFF) as u8;
        }

        let w_block = make_block(0.5, 0.25, &scales, &qs);
        let act_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 0, 1, -1, 2, -2, 3, -3,
            4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let acts = make_q8_acts(8, 0.1, &act_vals);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q2_KNeon
            .matvec_q8_fused(&w_block, &acts, &mut out_neon, 1, 256)
            .expect("neon fused q2k single block");
        Q2KRef
            .matvec_q8_fused(&w_block, &acts, &mut out_ref, 1, 256)
            .expect("ref fused q2k single block");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "q2k_neon_fused_matches_reference: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q2k_neon_fused_multi_row() {
        // 3 rows × 512 cols = 2 Q2_K super-blocks per row → 16 Q8_0 act blocks.
        let n_rows = 3usize;
        let n_cols = 512usize;
        let blocks_per_row = 2usize;
        let q8_blocks_per_row = blocks_per_row * 8; // 16

        let mut all_weights = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut scales = [0u8; 16];
                for (i, s) in scales.iter_mut().enumerate() {
                    *s = ((r * 17 + b * 11 + i * 7 + 5) & 0xFF) as u8;
                }
                let mut qs = [0u8; 64];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((r * 5 + b * 23 + i * 11) & 0xFF) as u8;
                }
                all_weights.extend(make_block(
                    0.5 + r as f32 * 0.1,
                    0.1 + b as f32 * 0.05,
                    &scales,
                    &qs,
                ));
            }
        }

        let act_vals: [i8; 32] = [
            2, -3, 5, -7, 1, -1, 4, -4, 6, -6, 3, -3, 2, -2, 1, -1, 8, -8, 7, -7, 6, -6, 5, -5, 4,
            -4, 3, -3, 2, -2, 1, -1,
        ];
        let acts = make_q8_acts(q8_blocks_per_row, 0.05, &act_vals);

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q2_KNeon
            .matvec_q8_fused(&all_weights, &acts, &mut out_neon, n_rows, n_cols)
            .expect("neon fused q2k multi-row");
        Q2KRef
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused q2k multi-row");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "q2k neon fused multi-row row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }
}
