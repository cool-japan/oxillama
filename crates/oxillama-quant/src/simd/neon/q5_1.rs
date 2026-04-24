//! Q5_1 NEON (AArch64) SIMD kernel.
//!
//! Q5_1 block format (24 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..4]: FP16 minimum `m` (little-endian)
//! - bytes[4..8]: `qh` — 32 high bits (bit 4 of each 5-bit quant), u32 LE
//! - bytes[8..24]: `qs` — 32 × lower 4 bits packed (2 per byte)
//!
//! Weight layout (sequential, not interleaved):
//!   output\[i\]      = d * (qs\[i\].lo4 | ((qh >> i) & 1) << 4) + m  for i in 0..16
//!   output\[i + 16\] = d * (qs\[i\].hi4 | ((qh >> (i+16)) & 1) << 4) + m  for i in 0..16
//!
//! Q5_1 is UNSIGNED: range [0..31], no -16 bias. Affine: d * q + m.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q5_1 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q5_1 block.
pub const BLOCK_BYTES: usize = 24;

/// NEON-accelerated Q5_1 kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q5_1Neon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Expand `qh` into two `[u8; 16]` arrays for the lo and hi halves.
#[inline(always)]
fn expand_qh(qh: u32) -> ([u8; 16], [u8; 16]) {
    let lo: [u8; 16] = core::array::from_fn(|i| ((qh >> i) & 1) as u8);
    let hi: [u8; 16] = core::array::from_fn(|i| ((qh >> (i + 16)) & 1) as u8);
    (lo, hi)
}

/// Dequantize one Q5_1 block using NEON intrinsics.
///
/// Produces 32 f32 values in sequential layout:
///   output[0..16]  = lo-half (from lo nibbles + qh bits 0..15)
///   output[16..32] = hi-half (from hi nibbles + qh bits 16..31)
/// Formula: `d * q5 + m` where q5 ∈ [0..31] (unsigned, no centering).
///
/// # Safety
/// Must be called on AArch64 with NEON. `qs_ptr` must point to 16 valid bytes.
/// `output` must have at least 32 slots.
#[inline]
unsafe fn dequant_block_neon(
    qs_ptr: *const u8,
    qh_lo: &[u8; 16],
    qh_hi: &[u8; 16],
    d: f32,
    m: f32,
    output: &mut [f32],
) {
    let d_vec = unsafe { vdupq_n_f32(d) };
    let m_vec = unsafe { vdupq_n_f32(m) };

    let raw = unsafe { vld1q_u8(qs_ptr) };
    let mask = unsafe { vdupq_n_u8(0x0F) };
    let lo_nib = unsafe { vandq_u8(raw, mask) };
    let hi_nib = unsafe { vshrq_n_u8::<4>(raw) };

    let vqh_lo = unsafe { vld1q_u8(qh_lo.as_ptr()) };
    let vqh_hi = unsafe { vld1q_u8(qh_hi.as_ptr()) };

    // Shift high bit into bit-position 4
    let shift4 = unsafe { vdupq_n_u8(4) };
    let qh_lo_shifted = unsafe { vshlq_u8(vqh_lo, vreinterpretq_s8_u8(shift4)) };
    let qh_hi_shifted = unsafe { vshlq_u8(vqh_hi, vreinterpretq_s8_u8(shift4)) };

    // 5-bit unsigned quants
    let q5_lo = unsafe { vorrq_u8(lo_nib, qh_lo_shifted) };
    let q5_hi = unsafe { vorrq_u8(hi_nib, qh_hi_shifted) };

    // Widen to u16 → u32 → f32, apply affine: m + d * q (no centering bias)
    let q5_lo_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_lo)) };
    let q5_lo_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_lo)) };
    let q5_hi_u16_low = unsafe { vmovl_u8(vget_low_u8(q5_hi)) };
    let q5_hi_u16_high = unsafe { vmovl_u8(vget_high_u8(q5_hi)) };

    let fa = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_low))),
            d_vec,
        )
    };
    let fb = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(q5_lo_u16_low)), d_vec) };
    let fc = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_high))),
            d_vec,
        )
    };
    let fe = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(q5_lo_u16_high)), d_vec) };
    let fg = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_low))),
            d_vec,
        )
    };
    let fh = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(q5_hi_u16_low)), d_vec) };
    let fj = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_high))),
            d_vec,
        )
    };
    let fk = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(q5_hi_u16_high)), d_vec) };

    unsafe { vst1q_f32(output.as_mut_ptr(), fa) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), fb) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), fc) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), fe) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), fg) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), fh) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), fj) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), fk) };
}

/// Compute dot product for one Q5_1 block against a contiguous f32 input slice.
///
/// Mirrors `reference::q5_1::Q5_1Ref::gemv`: computes Σ d*q*inp[i] + m*Σinp[i].
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
    m: f32,
    input: &[f32],
) -> f32 {
    let d_vec = unsafe { vdupq_n_f32(d) };

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

    let qf_a = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_low))) };
    let qf_b = unsafe { vcvtq_f32_u32(vmovl_high_u16(q5_lo_u16_low)) };
    let qf_c = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_lo_u16_high))) };
    let qf_e = unsafe { vcvtq_f32_u32(vmovl_high_u16(q5_lo_u16_high)) };
    let qf_g = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_low))) };
    let qf_h = unsafe { vcvtq_f32_u32(vmovl_high_u16(q5_hi_u16_low)) };
    let qf_j = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(q5_hi_u16_high))) };
    let qf_k = unsafe { vcvtq_f32_u32(vmovl_high_u16(q5_hi_u16_high)) };

    // Sequential input: [0..4, 4..8, 8..12, 12..16] for lo, [16..20, 20..24, 24..28, 28..32] for hi
    let ip = input.as_ptr();
    let i0 = unsafe { vld1q_f32(ip) };
    let i1 = unsafe { vld1q_f32(ip.add(4)) };
    let i2 = unsafe { vld1q_f32(ip.add(8)) };
    let i3 = unsafe { vld1q_f32(ip.add(12)) };
    let i4 = unsafe { vld1q_f32(ip.add(16)) };
    let i5 = unsafe { vld1q_f32(ip.add(20)) };
    let i6 = unsafe { vld1q_f32(ip.add(24)) };
    let i7 = unsafe { vld1q_f32(ip.add(28)) };

    // Accumulate d * q * inp
    let mut acc = unsafe { vmulq_f32(qf_a, i0) };
    acc = unsafe { vfmaq_f32(acc, qf_b, i1) };
    acc = unsafe { vfmaq_f32(acc, qf_c, i2) };
    acc = unsafe { vfmaq_f32(acc, qf_e, i3) };
    acc = unsafe { vfmaq_f32(acc, qf_g, i4) };
    acc = unsafe { vfmaq_f32(acc, qf_h, i5) };
    acc = unsafe { vfmaq_f32(acc, qf_j, i6) };
    acc = unsafe { vfmaq_f32(acc, qf_k, i7) };

    // Accumulate input sum for m correction: m * Σ inp[i]
    let mut inp_sum = unsafe { vaddq_f32(i0, i1) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(i2, i3)) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(i4, i5)) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(i6, i7)) };

    let _ = d_vec;
    d * unsafe { hsum_f32x4(acc) } + m * unsafe { hsum_f32x4(inp_sum) }
}

impl QuantKernel for Q5_1Neon {
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
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let (qh_lo, qh_hi) = expand_qh(qh);
        unsafe {
            dequant_block_neon(
                block.as_ptr().add(8),
                &qh_lo,
                &qh_hi,
                d,
                m,
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
                let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
                let (qh_lo, qh_hi) = expand_qh(qh);
                let input_offset = blk * BLOCK_SIZE;
                let block_input_end = (input_offset + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_offset;

                if block_input_len == BLOCK_SIZE {
                    sum += unsafe {
                        dot_block_neon(
                            block.as_ptr().add(8),
                            &qh_lo,
                            &qh_hi,
                            d,
                            m,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Scalar tail (sequential layout, unsigned, affine)
                    let qs = &block[8..24];
                    let inp = &input[input_offset..];
                    let mut input_sum = 0.0f32;
                    for i in 0..block_input_len {
                        let byte = qs[i % 16];
                        let lo_nibble = byte & 0x0F;
                        let hi_nibble = (byte >> 4) & 0x0F;
                        let hi_bit_lo = ((qh >> i) & 1) as u8;
                        let hi_bit_hi = ((qh >> (i + 16)) & 1) as u8;
                        let q0 = (lo_nibble | (hi_bit_lo << 4)) as f32;
                        let q1 = (hi_nibble | (hi_bit_hi << 4)) as f32;
                        if i < 16 {
                            sum += d * q0 * inp[i];
                            input_sum += inp[i];
                        } else {
                            sum += d * q1 * inp[i];
                            input_sum += inp[i];
                        }
                    }
                    sum += m * input_sum;
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
        "Q5_1_Neon"
    }
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q5_1::Q5_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, 0.0, 0, &[0; 16]);
        let mut out = vec![0.0f32; 32];
        Q5_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_min_only() {
        // d=0, m=3.0, all q=0 → w = 3.0
        let block = make_block(0.0, 3.0, 0, &[0; 16]);
        let mut out = vec![0.0f32; 32];
        Q5_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-4, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_max() {
        // d=1.0, m=0, qh=0xFFFFFFFF, qs=0xFF → q5=31 → w=31
        let block = make_block(1.0, 0.0, 0xFFFF_FFFF, &[0xFF; 16]);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q5_1Neon.dequant_block(&block, &mut out_neon).expect("neon");
        Q5_1Ref.dequant_block(&block, &mut out_ref).expect("ref");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "max err {max_err}");
    }

    #[test]
    fn test_dequant_matches_reference() {
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q5_1Neon.dequant_block(&block, &mut out_neon).expect("neon");
        Q5_1Ref.dequant_block(&block, &mut out_ref).expect("ref");
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
    fn test_gemv_matches_reference() {
        let qh: u32 = 0x5A5A_5A5A;
        let mut qs = [0u8; 16];
        for (i, v) in qs.iter_mut().enumerate() {
            *v = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_block(0.5, 0.25, qh, &qs);
        let n_cols = BLOCK_SIZE;
        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Q5_1,
        );
        let tensor_ref =
            QuantTensor::new(block, vec![1, n_cols], oxillama_gguf::GgufTensorType::Q5_1);
        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];
        Q5_1Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q5_1Ref
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
}
