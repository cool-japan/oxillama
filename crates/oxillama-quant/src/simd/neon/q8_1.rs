//! Q8_1 NEON (AArch64) SIMD kernel.
//!
//! Q8_1 block format (36 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..4]: FP16 sum `s` = d * Σqs (stored but unused in GEMV)
//! - bytes[4..36]: 32 × int8 signed quantised values
//!
//! Weight reconstruction: `d * qs[i]` (signed int8 scaled by d).
//! GEMV: plain f32 input × dequantised weights; `s` is not used.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q8_1 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q8_1 block.
pub const BLOCK_BYTES: usize = 36;

/// NEON-accelerated Q8_1 kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q8_1Neon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Dequantize one Q8_1 block using NEON intrinsics.
///
/// Produces 32 f32 values: `output[i] = d * qs[i]`.
///
/// # Safety
/// Must be called on AArch64 with NEON. `quant_ptr` must point to 32 valid i8 bytes.
/// `output` must have at least 32 slots.
#[inline]
unsafe fn dequant_block_neon(quant_ptr: *const i8, d: f32, output: &mut [f32]) {
    let d_vec = unsafe { vdupq_n_f32(d) };

    let q0 = unsafe { vld1q_s8(quant_ptr) };
    let q1 = unsafe { vld1q_s8(quant_ptr.add(16)) };

    let q0_lo = unsafe { vmovl_s8(vget_low_s8(q0)) };
    let q0_hi = unsafe { vmovl_s8(vget_high_s8(q0)) };
    let q1_lo = unsafe { vmovl_s8(vget_low_s8(q1)) };
    let q1_hi = unsafe { vmovl_s8(vget_high_s8(q1)) };

    let f0 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo))), d_vec) };
    let f1 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q0_lo)), d_vec) };
    let f2 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi))), d_vec) };
    let f3 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q0_hi)), d_vec) };
    let f4 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo))), d_vec) };
    let f5 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q1_lo)), d_vec) };
    let f6 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi))), d_vec) };
    let f7 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q1_hi)), d_vec) };

    unsafe { vst1q_f32(output.as_mut_ptr(), f0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), f1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), f2) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), f3) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), f4) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), f5) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), f6) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), f7) };
}

/// Compute the dot product between one Q8_1 block and 32 f32 inputs.
///
/// Mirrors `reference::q8_1::Q8_1Ref::gemv`: plain `d * Σ(qs[i] * inp[i])`.
/// The `s` field is not used (matches reference oracle exactly).
///
/// # Safety
/// Must be called on AArch64 with NEON. `quant_ptr` must point to 32 valid i8 bytes.
/// `input` must have exactly 32 elements.
#[inline]
unsafe fn dot_block_neon(quant_ptr: *const i8, d: f32, input: &[f32]) -> f32 {
    let q0 = unsafe { vld1q_s8(quant_ptr) };
    let q1 = unsafe { vld1q_s8(quant_ptr.add(16)) };

    let q0_lo = unsafe { vmovl_s8(vget_low_s8(q0)) };
    let q0_hi = unsafe { vmovl_s8(vget_high_s8(q0)) };
    let q1_lo = unsafe { vmovl_s8(vget_low_s8(q1)) };
    let q1_hi = unsafe { vmovl_s8(vget_high_s8(q1)) };

    let qf0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo))) };
    let qf1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q0_lo)) };
    let qf2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi))) };
    let qf3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q0_hi)) };
    let qf4 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo))) };
    let qf5 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q1_lo)) };
    let qf6 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi))) };
    let qf7 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q1_hi)) };

    let ip = input.as_ptr();
    let inp0 = unsafe { vld1q_f32(ip) };
    let inp1 = unsafe { vld1q_f32(ip.add(4)) };
    let inp2 = unsafe { vld1q_f32(ip.add(8)) };
    let inp3 = unsafe { vld1q_f32(ip.add(12)) };
    let inp4 = unsafe { vld1q_f32(ip.add(16)) };
    let inp5 = unsafe { vld1q_f32(ip.add(20)) };
    let inp6 = unsafe { vld1q_f32(ip.add(24)) };
    let inp7 = unsafe { vld1q_f32(ip.add(28)) };

    let mut acc = unsafe { vmulq_f32(qf0, inp0) };
    acc = unsafe { vfmaq_f32(acc, qf1, inp1) };
    acc = unsafe { vfmaq_f32(acc, qf2, inp2) };
    acc = unsafe { vfmaq_f32(acc, qf3, inp3) };
    acc = unsafe { vfmaq_f32(acc, qf4, inp4) };
    acc = unsafe { vfmaq_f32(acc, qf5, inp5) };
    acc = unsafe { vfmaq_f32(acc, qf6, inp6) };
    acc = unsafe { vfmaq_f32(acc, qf7, inp7) };

    d * unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q8_1Neon {
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
        // block[2..4] is `s` (sum), not needed for dequant — mirrors reference
        let quant_ptr = unsafe { block.as_ptr().add(4) as *const i8 };
        unsafe { dequant_block_neon(quant_ptr, d, &mut output[..BLOCK_SIZE]) };
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
                    let quant_ptr = unsafe { block.as_ptr().add(4) as *const i8 };
                    sum += unsafe {
                        dot_block_neon(
                            quant_ptr,
                            d,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Scalar tail
                    let qs = &block[4..36];
                    let mut block_sum = 0.0f32;
                    for i in 0..block_input_len {
                        block_sum += (qs[i] as i8) as f32 * input[input_offset + i];
                    }
                    sum += d * block_sum;
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
        "Q8_1_Neon"
    }
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q8_1::Q8_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        block.extend_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    fn fixed_qs() -> [i8; 32] {
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        qs
    }

    fn fixed_input() -> [f32; 32] {
        let mut inp = [0.0f32; 32];
        for (i, v) in inp.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 1.6;
        }
        inp
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, &[0; 32]);
        let mut out = vec![0.0f32; 32];
        Q8_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_matches_reference() {
        let qs = fixed_qs();
        let block = make_block(0.5, &qs);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q8_1Neon.dequant_block(&block, &mut out_neon).expect("neon");
        Q8_1Ref.dequant_block(&block, &mut out_ref).expect("ref");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "dequant max error {max_err}");
    }

    #[test]
    fn test_gemv_zeros() {
        let block = make_block(1.0, &[0; 32]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_1);
        let input = vec![1.0f32; 32];
        let mut out = vec![9.9f32; 1];
        Q8_1Neon.gemv(&tensor, &input, &mut out).expect("gemv");
        assert!(out[0].abs() < 1e-4, "expected ~0, got {}", out[0]);
    }

    #[test]
    fn test_gemv_matches_reference() {
        let qs = fixed_qs();
        let block = make_block(0.5, &qs);
        let input = fixed_input();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q8_1,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_1);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];
        Q8_1Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q8_1Ref
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
    fn test_gemv_multi_row() {
        let qs = fixed_qs();
        let n_rows = 4usize;
        let n_cols = 64usize;
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let scales = [0.25f32, 0.5, 1.0, 2.0];
        let mut data = Vec::new();
        for &s in &scales {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&make_block(s, &qs));
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 - 32.0) * 0.05).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_1,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_1,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];
        Q8_1Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q8_1Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref");

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
}
