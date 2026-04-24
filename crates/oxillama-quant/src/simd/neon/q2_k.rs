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
}
