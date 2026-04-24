//! Q4_1 NEON (AArch64) SIMD kernel.
//!
//! Q4_1 block format (20 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..4]: FP16 minimum `m` (little-endian)
//! - bytes[4..20]: 16 packed nibble bytes (32 × 4-bit unsigned values)
//!
//! Each byte: lo = byte & 0x0F (weight 2i), hi = byte >> 4 (weight 2i+1).
//! Weight reconstruction: `d * nibble + m`, nibble in [0..15] (unsigned).

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q4_1 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q4_1 block.
pub const BLOCK_BYTES: usize = 20;

/// NEON-accelerated Q4_1 kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q4_1Neon;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Dequantize one Q4_1 block using NEON intrinsics.
///
/// Produces 32 f32 values in interleaved order [lo0, hi0, lo1, hi1, ...].
/// Weight formula: `d * nibble + m` (unsigned nibbles, no centering bias).
///
/// # Safety
/// Must be called on AArch64 with NEON. `nibbles` must point to 16 valid bytes.
/// `output` must have at least 32 slots.
#[inline]
unsafe fn dequant_block_neon(nibbles: *const u8, d: f32, m: f32, output: &mut [f32]) {
    let raw = unsafe { vld1q_u8(nibbles) };

    let mask = unsafe { vdupq_n_u8(0x0F) };
    let lo = unsafe { vandq_u8(raw, mask) };
    let hi = unsafe { vshrq_n_u8::<4>(raw) };

    let d_vec = unsafe { vdupq_n_f32(d) };
    let m_vec = unsafe { vdupq_n_f32(m) };

    // Widen lo nibbles to u16 → u32 → f32, then FMA: m + d * q
    let lo16_low = unsafe { vmovl_u8(vget_low_u8(lo)) };
    let lo16_high = unsafe { vmovl_u8(vget_high_u8(lo)) };
    let hi16_low = unsafe { vmovl_u8(vget_low_u8(hi)) };
    let hi16_high = unsafe { vmovl_u8(vget_high_u8(hi)) };

    // Convert to f32 and apply affine: m + d * q
    let lo_f0 = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_low))),
            d_vec,
        )
    };
    let lo_f1 = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(lo16_low)), d_vec) };
    let lo_f2 = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_high))),
            d_vec,
        )
    };
    let lo_f3 = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(lo16_high)), d_vec) };

    let hi_f0 = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_low))),
            d_vec,
        )
    };
    let hi_f1 = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(hi16_low)), d_vec) };
    let hi_f2 = unsafe {
        vfmaq_f32(
            m_vec,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_high))),
            d_vec,
        )
    };
    let hi_f3 = unsafe { vfmaq_f32(m_vec, vcvtq_f32_u32(vmovl_high_u16(hi16_high)), d_vec) };

    // Interleave lo/hi pairs: [lo0,hi0,lo1,hi1, lo2,hi2,lo3,hi3, ...]
    let zip0 = unsafe { vzipq_f32(lo_f0, hi_f0) };
    let zip1 = unsafe { vzipq_f32(lo_f1, hi_f1) };
    let zip2 = unsafe { vzipq_f32(lo_f2, hi_f2) };
    let zip3 = unsafe { vzipq_f32(lo_f3, hi_f3) };

    unsafe { vst1q_f32(output.as_mut_ptr(), zip0.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), zip0.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), zip1.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), zip1.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), zip2.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), zip2.1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), zip3.0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), zip3.1) };
}

/// Compute the dot product between one Q4_1 block and 32 f32 inputs.
///
/// Returns `Σ (d * nibble_i + m) * input_i`.
///
/// # Safety
/// Must be called on AArch64 with NEON. `nibbles` must point to 16 valid bytes.
/// `input` must have exactly 32 elements.
#[inline]
unsafe fn dot_block_neon(nibbles: *const u8, d: f32, m: f32, input: &[f32]) -> f32 {
    let raw = unsafe { vld1q_u8(nibbles) };

    let mask = unsafe { vdupq_n_u8(0x0F) };
    let lo = unsafe { vandq_u8(raw, mask) };
    let hi = unsafe { vshrq_n_u8::<4>(raw) };

    // Widen to u16 → u32 → f32
    let lo16_low = unsafe { vmovl_u8(vget_low_u8(lo)) };
    let lo16_high = unsafe { vmovl_u8(vget_high_u8(lo)) };
    let hi16_low = unsafe { vmovl_u8(vget_low_u8(hi)) };
    let hi16_high = unsafe { vmovl_u8(vget_high_u8(hi)) };

    let lo_f0 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_low))) };
    let lo_f1 = unsafe { vcvtq_f32_u32(vmovl_high_u16(lo16_low)) };
    let lo_f2 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_high))) };
    let lo_f3 = unsafe { vcvtq_f32_u32(vmovl_high_u16(lo16_high)) };

    let hi_f0 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_low))) };
    let hi_f1 = unsafe { vcvtq_f32_u32(vmovl_high_u16(hi16_low)) };
    let hi_f2 = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_high))) };
    let hi_f3 = unsafe { vcvtq_f32_u32(vmovl_high_u16(hi16_high)) };

    // Load input: vld2q_f32 deinterleaves into even (lo weights) and odd (hi weights)
    let ip = input.as_ptr();
    let val0 = unsafe { vld2q_f32(ip) };
    let val1 = unsafe { vld2q_f32(ip.add(8)) };
    let val2 = unsafe { vld2q_f32(ip.add(16)) };
    let val3 = unsafe { vld2q_f32(ip.add(24)) };

    // Accumulate: d * q * inp (deferred m correction via sum of inputs)
    let mut acc = unsafe { vmulq_f32(lo_f0, val0.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f0, val0.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f1, val1.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f1, val1.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f2, val2.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f2, val2.1) };
    acc = unsafe { vfmaq_f32(acc, lo_f3, val3.0) };
    acc = unsafe { vfmaq_f32(acc, hi_f3, val3.1) };

    // Sum of all inputs for the m correction: Σ input_i
    let mut inp_sum = unsafe { vaddq_f32(val0.0, val0.1) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(val1.0, val1.1)) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(val2.0, val2.1)) };
    inp_sum = unsafe { vaddq_f32(inp_sum, vaddq_f32(val3.0, val3.1)) };

    d * unsafe { hsum_f32x4(acc) } + m * unsafe { hsum_f32x4(inp_sum) }
}

impl QuantKernel for Q4_1Neon {
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
        unsafe { dequant_block_neon(block.as_ptr().add(4), d, m, &mut output[..BLOCK_SIZE]) };
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
                let input_offset = blk * BLOCK_SIZE;
                let block_input_end = (input_offset + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_offset;

                if block_input_len == BLOCK_SIZE {
                    sum += unsafe {
                        dot_block_neon(
                            block.as_ptr().add(4),
                            d,
                            m,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Scalar tail for partial blocks
                    for i in 0..block_input_len {
                        let byte = block[4 + i / 2];
                        let nibble = if i % 2 == 0 {
                            (byte & 0x0F) as f32
                        } else {
                            ((byte >> 4) & 0x0F) as f32
                        };
                        sum += (d * nibble + m) * input[input_offset + i];
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
        "Q4_1_Neon"
    }
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q4_1::Q4_1Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    fn make_block(d: f32, m: f32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    fn fixed_qs() -> [u8; 16] {
        [
            0x5A, 0xF0, 0x13, 0x7E, 0xC2, 0x48, 0x9D, 0x6B, 0xA3, 0x2F, 0x71, 0xE4, 0x0C, 0x58,
            0xB6, 0xD9,
        ]
    }

    fn fixed_input() -> [f32; 32] {
        [
            0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, 1.6,
            -0.1, -0.2, 0.3, -0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5,
            -1.6,
        ]
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0, m=0, qs=0 → all 0
        let block = make_block(0.0, 0.0, &[0; 16]);
        let mut out = vec![0.0f32; 32];
        Q4_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_min_only() {
        // d=0, m=3.0 → all outputs = 3.0
        let block = make_block(0.0, 3.0, &[0; 16]);
        let mut out = vec![0.0f32; 32];
        Q4_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-4, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_matches_reference() {
        let qs = fixed_qs();
        let block = make_block(0.5, 0.25, &qs);
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];
        Q4_1Neon
            .dequant_block(&block, &mut out_neon)
            .expect("neon dequant");
        Q4_1Ref
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");
        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "dequant max error {max_err}");
    }

    #[test]
    fn test_dequant_unsigned_nibbles() {
        // All nibbles = 15 (max), d=1.0, m=0.0 → weights = 15.0
        let block = make_block(1.0, 0.0, &[0xFF; 16]);
        let mut out = vec![0.0f32; 32];
        Q4_1Neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!((v - 15.0).abs() < 1e-4, "expected 15.0, got {v}");
        }
    }

    #[test]
    fn test_gemv_zeros() {
        let block = make_block(1.0, 0.0, &[0; 16]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_1);
        let input = vec![1.0f32; 32];
        let mut out = vec![9.9f32; 1];
        Q4_1Neon.gemv(&tensor, &input, &mut out).expect("gemv");
        assert!(out[0].abs() < 1e-4, "expected ~0, got {}", out[0]);
    }

    #[test]
    fn test_gemv_matches_reference() {
        let qs = fixed_qs();
        let block = make_block(0.25, 0.1, &qs);
        let input = fixed_input();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q4_1,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_1);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q4_1Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon gemv");
        Q4_1Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

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
        let scales = [0.1f32, 0.25, 0.5, 1.0];
        let mut data = Vec::new();
        for &s in &scales {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&make_block(s, 0.1, &qs));
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 - 32.0) * 0.05).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_1,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q4_1,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];
        Q4_1Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q4_1Ref
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
