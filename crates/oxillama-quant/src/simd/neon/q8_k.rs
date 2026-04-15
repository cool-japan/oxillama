//! Q8_K NEON-optimized kernel.
//!
//! Q8_K block format (292 bytes per 256 weights):
//! - 4 bytes: f32 super-block scale (d)
//! - 256 bytes: qs — 256 × int8 signed quants
//! - 32 bytes: bsums — 16 × int16 block sums
//!
//! Weight formula: `w = d * qs[i]` where `qs[i]` is signed int8.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q8_K block.
pub const BLOCK_SIZE: usize = 256;
/// Number of bytes per Q8_K block (4 + 256 + 32).
pub const BLOCK_BYTES: usize = 292;

/// Offset of the quant data within a block.
const QS_OFFSET: usize = 4;

/// NEON-accelerated Q8_K kernel (AArch64 only).
#[allow(non_camel_case_types)]
pub struct Q8_KNeon;

/// Horizontal sum of a `float32x4_t` register.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    // SAFETY: caller guarantees AArch64 context; vaddvq_f32 is always valid.
    unsafe { vaddvq_f32(v) }
}

/// Dequantize 16 int8 quants to 16 f32 values, storing at `out_ptr`.
///
/// # Safety
/// `qs_ptr` must point to at least 16 valid i8 bytes.
/// `out_ptr` must be valid for writing 16 f32 values.
/// Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn dequant_16_neon(qs_ptr: *const i8, d_vec: float32x4_t, out_ptr: *mut f32) {
    // SAFETY: caller guarantees valid pointers and AArch64.
    unsafe {
        let q = vld1q_s8(qs_ptr);

        let lo_s16 = vmovl_s8(vget_low_s8(q));
        let hi_s16 = vmovl_s8(vget_high_s8(q));

        let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16))), d_vec);
        let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo_s16)), d_vec);
        let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16))), d_vec);
        let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi_s16)), d_vec);

        vst1q_f32(out_ptr, f0);
        vst1q_f32(out_ptr.add(4), f1);
        vst1q_f32(out_ptr.add(8), f2);
        vst1q_f32(out_ptr.add(12), f3);
    }
}

/// Compute the dot product of 16 int8 quants with 16 f32 inputs, accumulating
/// into `acc`.
///
/// # Safety
/// `qs_ptr` must point to at least 16 valid i8 bytes.
/// `inp_ptr` must point to at least 16 valid f32 values.
/// Must be called on AArch64 with NEON.
#[inline(always)]
unsafe fn dot_16_neon(qs_ptr: *const i8, inp_ptr: *const f32, acc: float32x4_t) -> float32x4_t {
    // SAFETY: caller guarantees valid pointers and AArch64.
    unsafe {
        let q = vld1q_s8(qs_ptr);

        let lo_s16 = vmovl_s8(vget_low_s8(q));
        let hi_s16 = vmovl_s8(vget_high_s8(q));

        let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16)));
        let w1 = vcvtq_f32_s32(vmovl_high_s16(lo_s16));
        let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16)));
        let w3 = vcvtq_f32_s32(vmovl_high_s16(hi_s16));

        let i0 = vld1q_f32(inp_ptr);
        let i1 = vld1q_f32(inp_ptr.add(4));
        let i2 = vld1q_f32(inp_ptr.add(8));
        let i3 = vld1q_f32(inp_ptr.add(12));

        let a = vfmaq_f32(acc, w0, i0);
        let a = vfmaq_f32(a, w1, i1);
        let a = vfmaq_f32(a, w2, i2);
        vfmaq_f32(a, w3, i3)
    }
}

impl QuantKernel for Q8_KNeon {
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

        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);

        // SAFETY: block has at least 292 bytes; block[4..260] = 256 valid i8 bytes.
        // output has at least 256 f32 slots. We are on AArch64 with NEON.
        unsafe {
            let d_vec = vdupq_n_f32(d);
            let qs_base = block.as_ptr().add(QS_OFFSET).cast::<i8>();
            let out_base = output.as_mut_ptr();

            // 256 values / 16 per iteration = 16 iterations
            for chunk in 0..16 {
                let offset = chunk * 16;
                dequant_16_neon(qs_base.add(offset), d_vec, out_base.add(offset));
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
                let data = &quant_matrix.data;
                let d = f32::from_le_bytes([
                    data[block_offset],
                    data[block_offset + 1],
                    data[block_offset + 2],
                    data[block_offset + 3],
                ]);
                let input_base = blk * BLOCK_SIZE;
                let block_input_end = (input_base + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_base;

                if block_input_len == BLOCK_SIZE {
                    // Full block: NEON fast path — process 256 quants in 16 chunks of 16.
                    // SAFETY: block has 292 bytes starting at block_offset;
                    // data[block_offset+4..block_offset+260] = 256 valid i8 bytes.
                    // input[input_base..input_base+256] has 256 valid f32 values.
                    unsafe {
                        let qs_base = data.as_ptr().add(block_offset + QS_OFFSET).cast::<i8>();
                        let inp_base = input.as_ptr().add(input_base);
                        let mut acc = vdupq_n_f32(0.0);

                        for chunk in 0..16 {
                            let offset = chunk * 16;
                            acc = dot_16_neon(qs_base.add(offset), inp_base.add(offset), acc);
                        }

                        sum += d * hsum_f32x4(acc);
                    }
                } else {
                    // Partial tail block: scalar fallback.
                    let qs_start = block_offset + QS_OFFSET;
                    let mut block_sum = 0.0f32;
                    for i in 0..block_input_len {
                        let q = data[qs_start + i] as i8;
                        block_sum += q as f32 * input[input_base + i];
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
        "Q8_K (NEON)"
    }
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q8_k::Q8KRef;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    /// Build a Q8_K block from scale and 256 i8 values.
    fn make_block(d: f32, qs: &[i8; 256]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&d.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        // bsums: 16 × int16 = 32 zero bytes
        block.extend_from_slice(&[0u8; 32]);
        block
    }

    /// Build a multi-row QuantTensor from a list of blocks.
    fn make_tensor(blocks: &[Vec<u8>], n_rows: usize, n_cols: usize) -> QuantTensor {
        let mut data = Vec::new();
        for b in blocks {
            data.extend_from_slice(b);
        }
        QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8K,
        )
    }

    // -----------------------------------------------------------------------
    // 1. dequant_block: d=0 → all zeros
    // -----------------------------------------------------------------------
    #[test]
    fn test_dequant_zeros() {
        let block = make_block(0.0, &[42; 256]);
        let kernel = Q8_KNeon;
        let mut output = vec![f32::NAN; 256];
        kernel.dequant_block(&block, &mut output).expect("dequant");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-9, "weight[{i}] = {v}, expected 0.0");
        }
    }

    // -----------------------------------------------------------------------
    // 2. dequant_block: d=0.25, qs=40 → all 10.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_dequant_positive() {
        let block = make_block(0.25, &[40; 256]);
        let kernel = Q8_KNeon;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).expect("dequant");
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 10.0).abs() < 1e-4, "weight[{i}] = {v}, expected 10.0");
        }
    }

    // -----------------------------------------------------------------------
    // 3. dequant_block: d=1.0, qs=-100 → all -100.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_dequant_negative() {
        let block = make_block(1.0, &[-100; 256]);
        let kernel = Q8_KNeon;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).expect("dequant");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-100.0)).abs() < 1e-4,
                "weight[{i}] = {v}, expected -100.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. dequant_block: ascending int8 values
    // -----------------------------------------------------------------------
    #[test]
    fn test_dequant_mixed() {
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            // Map 0..255 to -128..127
            *q = (i as i16 - 128) as i8;
        }
        let d = 0.5f32;
        let block = make_block(d, &qs);
        let kernel = Q8_KNeon;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).expect("dequant");
        for (i, &v) in output.iter().enumerate() {
            let expected = d * qs[i] as f32;
            assert!(
                (v - expected).abs() < 1e-4,
                "weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. gemv: single block (1×256)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemv_single_block() {
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 3 - 128).clamp(-128, 127)) as i8;
        }
        let d = 0.25f32;
        let block = make_block(d, &qs);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // Reference scalar result
        let expected: f32 = qs
            .iter()
            .zip(input.iter())
            .map(|(&q, &x)| d * q as f32 * x)
            .sum();

        let tensor = make_tensor(&[block], 1, 256);
        let kernel = Q8_KNeon;
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).expect("gemv");

        assert!(
            (output[0] - expected).abs() < 0.5,
            "gemv={}, expected={}",
            output[0],
            expected
        );
    }

    // -----------------------------------------------------------------------
    // 6. gemv: multi-row (4×256)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemv_multi_row() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let mut blocks = Vec::new();
        let mut expected = [0.0f32; 4];

        for (row, exp) in expected.iter_mut().enumerate() {
            let mut qs = [0i8; 256];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = (((i + row * 7) as i16 * 5 - 300).clamp(-128, 127)) as i8;
            }
            let d = 0.1 * (row as f32 + 1.0);
            blocks.push(make_block(d, &qs));

            *exp = qs
                .iter()
                .zip(input.iter())
                .map(|(&q, &x)| d * q as f32 * x)
                .sum();
        }

        let tensor = make_tensor(&blocks, 4, 256);
        let kernel = Q8_KNeon;
        let mut output = vec![0.0f32; 4];
        kernel.gemv(&tensor, &input, &mut output).expect("gemv");

        for (row, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1.0,
                "row {row}: gemv={got}, expected={exp}",
            );
        }
    }

    // -----------------------------------------------------------------------
    // 7. gemv: cross-validate NEON vs Q8KRef within 1e-4
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemv_cross_validate() {
        // Deterministic pseudo-random-ish data
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            // Simple LCG-like pattern
            let v = ((i as u32).wrapping_mul(2654435761) >> 24) as i8;
            *q = v;
        }
        let d = 0.125f32;
        let block = make_block(d, &qs);
        let input: Vec<f32> = (0..256)
            .map(|i| {
                let bits = (i as u32).wrapping_mul(1597334677);
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let tensor_neon = make_tensor(std::slice::from_ref(&block), 1, 256);
        let tensor_ref = make_tensor(std::slice::from_ref(&block), 1, 256);

        let neon_kernel = Q8_KNeon;
        let ref_kernel = Q8KRef;

        let mut neon_out = vec![0.0f32; 1];
        let mut ref_out = vec![0.0f32; 1];

        neon_kernel
            .gemv(&tensor_neon, &input, &mut neon_out)
            .expect("neon gemv");
        ref_kernel
            .gemv(&tensor_ref, &input, &mut ref_out)
            .expect("ref gemv");

        let diff = (neon_out[0] - ref_out[0]).abs();
        assert!(
            diff < 1e-4,
            "NEON vs Ref mismatch: neon={}, ref={}, diff={diff}",
            neon_out[0],
            ref_out[0]
        );
    }

    // -----------------------------------------------------------------------
    // 8. block_size and name constants
    // -----------------------------------------------------------------------
    #[test]
    fn test_block_size_and_name() {
        let kernel = Q8_KNeon;
        assert_eq!(kernel.block_size(), 256);
        assert_eq!(kernel.block_bytes(), 292);
        assert_eq!(kernel.name(), "Q8_K (NEON)");
    }
}
