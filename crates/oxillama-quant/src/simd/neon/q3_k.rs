//! Q3_K NEON (AArch64) SIMD kernel.
//!
//! Q3_K super-block format (110 bytes per 256 weights):
//! - bytes[0..32]:   hmask  — 1 bit per weight; if set → subtract 0; if clear → subtract 4
//! - bytes[32..96]:  qs     — lower 2 bits of each 3-bit quant (4 per byte via shifts 0,2,4,6)
//! - bytes[96..108]: scales — 16 × 6-bit signed sub-block scales, packed into 12 bytes
//! - bytes[108..110]: FP16 super-block scale `d`
//!
//! Symmetric format (no minimum offset). Weight formula:
//!   `w = d * scale_i * (q_lo | (hmask_bit ? 0 : -4))`
//! where q_lo ∈ [0..3], giving effective range [-4..3].

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q3_K super-block.
pub const BLOCK_SIZE: usize = 256;
/// Bytes per Q3_K super-block.
pub const BLOCK_BYTES: usize = 110;

/// NEON-accelerated Q3_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q3_KNeon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Decode 16 signed 6-bit sub-block scales from the 12-byte packed representation.
///
/// Mirrors `reference::q3_k::decode_scales` exactly.
/// Returns unsigned 6-bit values minus 32 → signed range [-32..31].
pub fn unpack_scales(scales_raw: &[u8; 12]) -> [i8; 16] {
    let mut sc = [0u32; 16];

    for j in 0..4 {
        sc[j] = (scales_raw[j] & 0x3F) as u32;
    }
    for j in 0..4 {
        sc[4 + j] = (scales_raw[4 + j] & 0x3F) as u32;
    }
    for j in 0..4 {
        let lo = (scales_raw[8 + j] & 0x0F) as u32;
        let hi = ((scales_raw[j] >> 6) & 0x03) as u32;
        sc[8 + j] = lo | (hi << 4);
    }
    for j in 0..4 {
        let lo = ((scales_raw[8 + j] >> 4) & 0x0F) as u32;
        let hi = ((scales_raw[4 + j] >> 6) & 0x03) as u32;
        sc[12 + j] = lo | (hi << 4);
    }

    let mut result = [0i8; 16];
    for i in 0..16 {
        result[i] = (sc[i] as i32 - 32) as i8;
    }
    result
}

/// Extract 2-bit fields from 16 bytes, selecting field by shift (0, 2, 4, 6).
#[inline(always)]
unsafe fn extract_2bit(raw: uint8x16_t, shift: u32) -> uint8x16_t {
    let mask = unsafe { vdupq_n_u8(0x03) };
    let shifted = match shift {
        0 => raw,
        2 => unsafe { vshrq_n_u8::<2>(raw) },
        4 => unsafe { vshrq_n_u8::<4>(raw) },
        _ => unsafe { vshrq_n_u8::<6>(raw) },
    };
    unsafe { vandq_u8(shifted, mask) }
}

/// Compute per-element correction from hmask bytes: 0 if bit set, 4 if bit clear.
///
/// `m_bit` is the single-bit selector (a power of 2 ≤ 128).
/// Returns a u8x16 with values in {0, 4}.
#[inline(always)]
unsafe fn hmask_correction(hmask_chunk: uint8x16_t, m_bit: u8) -> uint8x16_t {
    let m_vec = unsafe { vdupq_n_u8(m_bit) };
    let four = unsafe { vdupq_n_u8(4) };
    let zero = unsafe { vdupq_n_u8(0) };
    // Mask the bit: if (hmask & m_bit) != 0 → bit present
    let masked = unsafe { vandq_u8(hmask_chunk, m_vec) };
    // If masked == m_bit → correction 0, else correction 4
    // Use vcgtq_u8: masked > 0 → all-ones lane, else all-zeros.
    // Then select: 0 where bit set, 4 where bit clear.
    let is_set = unsafe { vcgtq_u8(masked, vdupq_n_u8(0)) };
    unsafe { vbslq_u8(is_set, zero, four) }
}

/// Dequantize 16 weights from a Q3_K sub-block using NEON.
///
/// `qs_raw` = 16-byte chunk of qs at appropriate base.
/// `hmask_chunk` = corresponding 16-byte slice of hmask.
/// Returns nothing; writes to `out_ptr`.
///
/// # Safety
/// `out_ptr` must have at least 16 valid f32 slots.
#[inline]
unsafe fn dequant_16_weights(
    qs_raw: uint8x16_t,
    hmask_chunk: uint8x16_t,
    shift: u32,
    m_bit: u8,
    dl: f32,
    out_ptr: *mut f32,
) {
    let q_bytes = unsafe { extract_2bit(qs_raw, shift) };
    let corr = unsafe { hmask_correction(hmask_chunk, m_bit) };

    let dl_vec = unsafe { vdupq_n_f32(dl) };

    // q_signed = q_lo - correction (4 when hmask bit clear → subtracts 4 for signed range)
    // Widen u8 to i16, subtract correction (u8→i16 too, then sub)
    let q_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q_bytes)) };
    let q_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q_bytes)) };
    let c_lo_u16 = unsafe { vmovl_u8(vget_low_u8(corr)) };
    let c_hi_u16 = unsafe { vmovl_u8(vget_high_u8(corr)) };

    let qs_lo = unsafe { vreinterpretq_s16_u16(vsubq_u16(q_lo_u16, c_lo_u16)) };
    let qs_hi = unsafe { vreinterpretq_s16_u16(vsubq_u16(q_hi_u16, c_hi_u16)) };

    // Widen to s32 → f32, scale by dl
    let q0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs_lo))) };
    let q1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(qs_lo)) };
    let q2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs_hi))) };
    let q3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(qs_hi)) };

    let w0 = unsafe { vmulq_f32(dl_vec, q0) };
    let w1 = unsafe { vmulq_f32(dl_vec, q1) };
    let w2 = unsafe { vmulq_f32(dl_vec, q2) };
    let w3 = unsafe { vmulq_f32(dl_vec, q3) };

    unsafe { vst1q_f32(out_ptr, w0) };
    unsafe { vst1q_f32(out_ptr.add(4), w1) };
    unsafe { vst1q_f32(out_ptr.add(8), w2) };
    unsafe { vst1q_f32(out_ptr.add(12), w3) };
}

/// Dot product: 16 Q3_K weights × 16 f32 inputs.
///
/// # Safety
/// `inp_ptr` must point to 16 valid f32 values.
#[inline]
unsafe fn dot_16_weights(
    qs_raw: uint8x16_t,
    hmask_chunk: uint8x16_t,
    shift: u32,
    m_bit: u8,
    dl: f32,
    inp_ptr: *const f32,
) -> f32 {
    let q_bytes = unsafe { extract_2bit(qs_raw, shift) };
    let corr = unsafe { hmask_correction(hmask_chunk, m_bit) };

    let dl_vec = unsafe { vdupq_n_f32(dl) };

    let q_lo_u16 = unsafe { vmovl_u8(vget_low_u8(q_bytes)) };
    let q_hi_u16 = unsafe { vmovl_u8(vget_high_u8(q_bytes)) };
    let c_lo_u16 = unsafe { vmovl_u8(vget_low_u8(corr)) };
    let c_hi_u16 = unsafe { vmovl_u8(vget_high_u8(corr)) };

    let qs_lo = unsafe { vreinterpretq_s16_u16(vsubq_u16(q_lo_u16, c_lo_u16)) };
    let qs_hi = unsafe { vreinterpretq_s16_u16(vsubq_u16(q_hi_u16, c_hi_u16)) };

    let q0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs_lo))) };
    let q1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(qs_lo)) };
    let q2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs_hi))) };
    let q3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(qs_hi)) };

    let w0 = unsafe { vmulq_f32(dl_vec, q0) };
    let w1 = unsafe { vmulq_f32(dl_vec, q1) };
    let w2 = unsafe { vmulq_f32(dl_vec, q2) };
    let w3 = unsafe { vmulq_f32(dl_vec, q3) };

    let i0 = unsafe { vld1q_f32(inp_ptr) };
    let i1 = unsafe { vld1q_f32(inp_ptr.add(4)) };
    let i2 = unsafe { vld1q_f32(inp_ptr.add(8)) };
    let i3 = unsafe { vld1q_f32(inp_ptr.add(12)) };

    let mut acc = unsafe { vmulq_f32(w0, i0) };
    acc = unsafe { vfmaq_f32(acc, w1, i1) };
    acc = unsafe { vfmaq_f32(acc, w2, i2) };
    acc = unsafe { vfmaq_f32(acc, w3, i3) };

    unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q3_KNeon {
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

        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw: &[u8; 12] = block[96..108].try_into().unwrap_or(&[0u8; 12]);
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let sc = unpack_scales(scales_raw);

        let hmask_lo = unsafe { vld1q_u8(hmask.as_ptr()) };
        let hmask_hi = unsafe { vld1q_u8(hmask.as_ptr().add(16)) };

        let mut is = 0usize;
        let mut out_off = 0usize;

        for group in 0..2usize {
            let qs_base = group * 32;
            let raw_a = unsafe { vld1q_u8(qs.as_ptr().add(qs_base)) };
            let raw_b = unsafe { vld1q_u8(qs.as_ptr().add(qs_base + 16)) };

            for shift_idx in 0..4u32 {
                let shift = shift_idx * 2;
                let bit_pos = (group as u32) * 4 + shift_idx;
                let m_bit: u8 = 1u8 << bit_pos;

                let dl_a = d * sc[is] as f32;
                is += 1;
                unsafe {
                    dequant_16_weights(
                        raw_a,
                        hmask_lo,
                        shift,
                        m_bit,
                        dl_a,
                        output.as_mut_ptr().add(out_off),
                    )
                };
                out_off += 16;

                let dl_b = d * sc[is] as f32;
                is += 1;
                unsafe {
                    dequant_16_weights(
                        raw_b,
                        hmask_hi,
                        shift,
                        m_bit,
                        dl_b,
                        output.as_mut_ptr().add(out_off),
                    )
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
                let hmask = &block[0..32];
                let qs = &block[32..96];
                let scales_raw: &[u8; 12] = block[96..108].try_into().unwrap_or(&[0u8; 12]);
                let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
                let sc = unpack_scales(scales_raw);
                let input_offset = blk * BLOCK_SIZE;
                let cols_in_block = (n_cols - input_offset).min(BLOCK_SIZE);

                let hmask_lo = unsafe { vld1q_u8(hmask.as_ptr()) };
                let hmask_hi = unsafe { vld1q_u8(hmask.as_ptr().add(16)) };

                if cols_in_block == BLOCK_SIZE {
                    let mut is = 0usize;
                    let mut w_off = input_offset;

                    for group in 0..2usize {
                        let qs_base = group * 32;
                        let raw_a = unsafe { vld1q_u8(qs.as_ptr().add(qs_base)) };
                        let raw_b = unsafe { vld1q_u8(qs.as_ptr().add(qs_base + 16)) };

                        for shift_idx in 0..4u32 {
                            let shift = shift_idx * 2;
                            let bit_pos = (group as u32) * 4 + shift_idx;
                            let m_bit: u8 = 1u8 << bit_pos;

                            let dl_a = d * sc[is] as f32;
                            is += 1;
                            sum += unsafe {
                                dot_16_weights(
                                    raw_a,
                                    hmask_lo,
                                    shift,
                                    m_bit,
                                    dl_a,
                                    input.as_ptr().add(w_off),
                                )
                            };
                            w_off += 16;

                            let dl_b = d * sc[is] as f32;
                            is += 1;
                            sum += unsafe {
                                dot_16_weights(
                                    raw_b,
                                    hmask_hi,
                                    shift,
                                    m_bit,
                                    dl_b,
                                    input.as_ptr().add(w_off),
                                )
                            };
                            w_off += 16;
                        }
                    }
                } else {
                    // Scalar tail for partial blocks
                    let inp = &input[input_offset..];
                    let mut is = 0usize;
                    let mut in_off = 0usize;
                    let mut m_bit: u8 = 1;

                    for group in 0..2 {
                        let qs_base = group * 32;
                        for shift in (0..8usize).step_by(2) {
                            for n in 0..2 {
                                let dl = d * sc[is] as f32;
                                is += 1;
                                for l in 0..16 {
                                    if in_off + l < cols_in_block {
                                        let qs_idx = qs_base + n * 16 + l;
                                        let q_lo = ((qs[qs_idx] >> shift) & 3) as i32;
                                        let sub =
                                            if hmask[n * 16 + l] & m_bit != 0 { 0 } else { 4 };
                                        sum += dl * (q_lo - sub) as f32 * inp[in_off + l];
                                    }
                                }
                                in_off += 16;
                            }
                            m_bit = m_bit.wrapping_shl(1);
                        }
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
        "Q3_K_Neon"
    }
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q3_k::Q3KRef;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, scales: &[u8; 12], hmask: &[u8; 32], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(hmask);
        block.extend_from_slice(qs);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    fn all_one_scales() -> [u8; 12] {
        // All 16 sub-block scales = signed +1 (raw = 33)
        [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ]
    }

    #[test]
    fn test_unpack_scales_all_one() {
        let scales = all_one_scales();
        let decoded = unpack_scales(&scales);
        for (i, &s) in decoded.iter().enumerate() {
            assert_eq!(s, 1, "scale[{i}] = {s}, expected 1");
        }
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, &[0; 12], &[0; 32], &[0; 64]);
        let mut out = vec![0.0f32; 256];
        Q3_KNeon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_matches_reference() {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 7 + 3) & 0xFF) as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 11 + 5) & 0xFF) as u8;
        }
        let scales = all_one_scales();
        let block = make_block(0.5, &scales, &hmask, &qs);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];
        Q3_KNeon.dequant_block(&block, &mut out_neon).expect("neon");
        Q3KRef.dequant_block(&block, &mut out_ref).expect("ref");
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
    fn test_dequant_hmask_set_q3() {
        // hmask all set → subtract 0, all qs=0xFF → q_lo=3, scale=+1, d=1.0 → weight=3.0
        let hmask = [0xFFu8; 32];
        let qs = [0xFFu8; 64];
        let scales = all_one_scales();
        let block = make_block(1.0, &scales, &hmask, &qs);
        let mut out_neon = vec![0.0f32; 256];
        let mut out_ref = vec![0.0f32; 256];
        Q3_KNeon.dequant_block(&block, &mut out_neon).expect("neon");
        Q3KRef.dequant_block(&block, &mut out_ref).expect("ref");
        for i in 0..256 {
            assert!(
                (out_neon[i] - out_ref[i]).abs() < 1e-4,
                "weight[{i}]: neon={} ref={}",
                out_neon[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference() {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 7 + 3) & 0xFF) as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 11 + 5) & 0xFF) as u8;
        }
        let scales = all_one_scales();
        let block = make_block(0.5, &scales, &hmask, &qs);

        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Q3K,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q3K);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];
        Q3_KNeon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q3KRef.gemv(&tensor_ref, &input, &mut out_ref).expect("ref");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 0.1,
            "gemv: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }
}
