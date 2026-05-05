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
                fused_q8_1_q8_0_row_neon(
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

/// Fused Q8_1 weight × Q8_0 activation dot product for one matrix row.
///
/// Q8_1 layout: d (f16) at [0..2], s (f16, unused) at [2..4], qs (i8×32) at [4..36].
/// Formula: `d_w * Σ(qs_weight_i * d_a * q8_0_act_i)`.
///
/// # Safety
/// Must be called on AArch64 with NEON. All slice bounds must be pre-validated.
unsafe fn fused_q8_1_q8_0_row_neon(
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

        // Q8_1 weight: scale at [0..2], qs (int8) at [4..36]
        let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        // Decode Q8_0 activation block
        let ab = blk * Q8_BLOCK_BYTES;
        let a_block = &acts_q8[ab..ab + Q8_BLOCK_BYTES];
        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));

        let col_end = (col_start + BLOCK_SIZE).min(n_cols);
        let avail = col_end - col_start;

        if avail == BLOCK_SIZE {
            // Full block: NEON decode of both weight and activation
            let w_ptr = block.as_ptr().add(4) as *const i8;
            let wq0 = vld1q_s8(w_ptr);
            let wq1 = vld1q_s8(w_ptr.add(16));

            let wq0_lo = vmovl_s8(vget_low_s8(wq0));
            let wq0_hi = vmovl_s8(vget_high_s8(wq0));
            let wq1_lo = vmovl_s8(vget_low_s8(wq1));
            let wq1_hi = vmovl_s8(vget_high_s8(wq1));

            let d_w_vec = vdupq_n_f32(d_w);
            let wf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wq0_lo))), d_w_vec);
            let wf1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(wq0_lo)), d_w_vec);
            let wf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wq0_hi))), d_w_vec);
            let wf3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(wq0_hi)), d_w_vec);
            let wf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wq1_lo))), d_w_vec);
            let wf5 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(wq1_lo)), d_w_vec);
            let wf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wq1_hi))), d_w_vec);
            let wf7 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(wq1_hi)), d_w_vec);

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

            acc = vfmaq_f32(acc, wf0, aa0);
            acc = vfmaq_f32(acc, wf1, aa1);
            acc = vfmaq_f32(acc, wf2, aa2);
            acc = vfmaq_f32(acc, wf3, aa3);
            acc = vfmaq_f32(acc, wf4, aa4);
            acc = vfmaq_f32(acc, wf5, aa5);
            acc = vfmaq_f32(acc, wf6, aa6);
            acc = vfmaq_f32(acc, wf7, aa7);
        } else {
            // Scalar tail for partial blocks
            let qs = &block[4..36];
            let q8_vals = &a_block[2..];
            let mut block_dot = 0.0f32;
            for i in 0..avail {
                let w = (qs[i] as i8) as f32 * d_w;
                let a_val = (q8_vals[i] as i8) as f32 * d_a;
                block_dot += w * a_val;
            }
            let lane0 = vgetq_lane_f32::<0>(acc) + block_dot;
            acc = vdupq_n_f32(0.0f32);
            acc = vsetq_lane_f32::<0>(lane0, acc);
        }
    }

    hsum_f32x4(acc)
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

    #[test]
    fn fused_q8_1_neon_matches_reference() {
        let qs = fixed_qs();
        let weight_block = make_block(0.5, &qs);

        // Build Q8_0 activation block (34 bytes)
        let d_a = 0.25f32;
        let acts_raw = fixed_input();
        let acts_i8: Vec<i8> = acts_raw
            .iter()
            .map(|&v| (v / d_a).clamp(-128.0, 127.0) as i8)
            .collect();
        let mut acts_block = Vec::with_capacity(34);
        acts_block.extend_from_slice(&half::f16::from_f32(d_a).to_bits().to_le_bytes());
        for &v in &acts_i8 {
            acts_block.push(v as u8);
        }

        // Reference: dequant weight + dot with scaled activations
        let mut w_dequant = vec![0.0f32; BLOCK_SIZE];
        Q8_1Ref
            .dequant_block(&weight_block, &mut w_dequant)
            .expect("ref dequant");
        let acts_f32: Vec<f32> = acts_i8.iter().map(|&v| v as f32 * d_a).collect();
        let expected: f32 = w_dequant
            .iter()
            .zip(acts_f32.iter())
            .map(|(w, a)| w * a)
            .sum();

        // NEON fused
        let mut out_neon = vec![0.0f32; 1];
        Q8_1Neon
            .matvec_q8_fused(&weight_block, &acts_block, &mut out_neon, 1, BLOCK_SIZE)
            .expect("neon fused");

        let err = (out_neon[0] - expected).abs();
        assert!(
            err < 0.1,
            "fused_q8_1_neon: got={} expected={} err={}",
            out_neon[0],
            expected,
            err
        );

        // Verify against reference Q8_1Ref::matvec_q8_fused
        let mut out_ref = vec![0.0f32; 1];
        Q8_1Ref
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
