//! Q8_0 NEON (AArch64) SIMD kernel.
//!
//! Q8_0 block format (34 bytes per 32 weights):
//! - bytes[0..2]: FP16 scale `d` (little-endian)
//! - bytes[2..34]: 32 × int8 quantized values
//!
//! Weight reconstruction: `q[i] * d`

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Number of weights per Q8_0 block.
pub const BLOCK_SIZE: usize = 32;
/// Number of bytes per Q8_0 block.
pub const BLOCK_BYTES: usize = 34;

/// NEON-accelerated Q8_0 kernel (AArch64 only).
pub struct Q8_0Neon;

/// Convert an IEEE 754 FP16 value (as raw u16 LE bytes) to f32.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Horizontal sum of a `float32x4_t` register.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    // SAFETY: caller guarantees AArch64 context; vaddvq_f32 is always valid.
    unsafe { vaddvq_f32(v) }
}

/// Dequantize one Q8_0 block using NEON intrinsics.
///
/// Produces 32 f32 values: output[i] = quant[i] * d.
///
/// # Safety
/// Must be called on AArch64 with NEON. `quant_ptr` must point to at least
/// 32 initialised i8 bytes. `output` must have at least 32 slots.
#[inline]
unsafe fn dequant_block_neon(quant_ptr: *const i8, d: f32, output: &mut [f32]) {
    let d_vec = unsafe { vdupq_n_f32(d) };

    // Load two 16-byte i8 vectors covering all 32 quant values.
    // SAFETY: quant_ptr points to 32 valid i8 bytes.
    let q0 = unsafe { vld1q_s8(quant_ptr) };
    let q1 = unsafe { vld1q_s8(quant_ptr.add(16)) };

    // q0: first 16 i8s → expand to 4 × float32x4_t
    // SAFETY: vmovl_s8, vmovl_s16, vmovl_high_s16, vcvtq_f32_s32 are valid on AArch64.
    let q0_lo_s16 = unsafe { vmovl_s8(vget_low_s8(q0)) }; // 8 × i16
    let q0_hi_s16 = unsafe { vmovl_s8(vget_high_s8(q0)) }; // 8 × i16

    let f0 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo_s16))), d_vec) };
    let f1 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q0_lo_s16)), d_vec) };
    let f2 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi_s16))), d_vec) };
    let f3 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q0_hi_s16)), d_vec) };

    // q1: next 16 i8s → expand to 4 × float32x4_t
    let q1_lo_s16 = unsafe { vmovl_s8(vget_low_s8(q1)) };
    let q1_hi_s16 = unsafe { vmovl_s8(vget_high_s8(q1)) };

    let f4 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo_s16))), d_vec) };
    let f5 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q1_lo_s16)), d_vec) };
    let f6 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi_s16))), d_vec) };
    let f7 = unsafe { vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q1_hi_s16)), d_vec) };

    // Store 32 f32 results sequentially.
    // SAFETY: vst1q_f32 requires a valid pointer to 4 f32 (16 bytes).
    // output has at least 32 slots; offsets 0..28 are all in range.
    unsafe { vst1q_f32(output.as_mut_ptr(), f0) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(4), f1) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(8), f2) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(12), f3) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(16), f4) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(20), f5) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(24), f6) };
    unsafe { vst1q_f32(output.as_mut_ptr().add(28), f7) };
}

/// Compute the dot product between one Q8_0 block and 32 f32 inputs.
///
/// Returns `d * Σ q[i] * input[i]`.
///
/// # Safety
/// Must be called on AArch64 with NEON. `quant_ptr` must point to 32 valid i8 bytes.
/// `input` must have exactly 32 elements.
#[inline]
unsafe fn dot_block_neon(quant_ptr: *const i8, d: f32, input: &[f32]) -> f32 {
    // Load 32 i8 quantised weights.
    // SAFETY: quant_ptr points to 32 valid bytes.
    let q0 = unsafe { vld1q_s8(quant_ptr) };
    let q1 = unsafe { vld1q_s8(quant_ptr.add(16)) };

    // Convert i8 blocks to f32
    // SAFETY: vmovl_s8, vmovl_s16, vmovl_high_s16, vcvtq_f32_s32 are valid on AArch64.
    let q0_lo_s16 = unsafe { vmovl_s8(vget_low_s8(q0)) };
    let q0_hi_s16 = unsafe { vmovl_s8(vget_high_s8(q0)) };
    let q1_lo_s16 = unsafe { vmovl_s8(vget_low_s8(q1)) };
    let q1_hi_s16 = unsafe { vmovl_s8(vget_high_s8(q1)) };

    let w0 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo_s16))) };
    let w1 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q0_lo_s16)) };
    let w2 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi_s16))) };
    let w3 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q0_hi_s16)) };
    let w4 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo_s16))) };
    let w5 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q1_lo_s16)) };
    let w6 = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi_s16))) };
    let w7 = unsafe { vcvtq_f32_s32(vmovl_high_s16(q1_hi_s16)) };

    // Load 32 input f32 values in 8 groups of 4.
    let ip = input.as_ptr();
    // SAFETY: input.len() == 32; offsets 0..28 each have 4 f32 values remaining.
    let i0 = unsafe { vld1q_f32(ip) };
    let i1 = unsafe { vld1q_f32(ip.add(4)) };
    let i2 = unsafe { vld1q_f32(ip.add(8)) };
    let i3 = unsafe { vld1q_f32(ip.add(12)) };
    let i4 = unsafe { vld1q_f32(ip.add(16)) };
    let i5 = unsafe { vld1q_f32(ip.add(20)) };
    let i6 = unsafe { vld1q_f32(ip.add(24)) };
    let i7 = unsafe { vld1q_f32(ip.add(28)) };

    // Accumulate dot products with FMA.
    // SAFETY: vmulq_f32 and vfmaq_f32 are valid on AArch64.
    let mut acc = unsafe { vmulq_f32(w0, i0) };
    acc = unsafe { vfmaq_f32(acc, w1, i1) };
    acc = unsafe { vfmaq_f32(acc, w2, i2) };
    acc = unsafe { vfmaq_f32(acc, w3, i3) };
    acc = unsafe { vfmaq_f32(acc, w4, i4) };
    acc = unsafe { vfmaq_f32(acc, w5, i5) };
    acc = unsafe { vfmaq_f32(acc, w6, i6) };
    acc = unsafe { vfmaq_f32(acc, w7, i7) };

    // SAFETY: hsum_f32x4 calls vaddvq_f32, valid on AArch64.
    d * unsafe { hsum_f32x4(acc) }
}

impl QuantKernel for Q8_0Neon {
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
        // SAFETY: block has at least 34 bytes; block[2..34] = 32 valid i8 bytes.
        // output has at least 32 f32 slots.
        unsafe {
            dequant_block_neon(
                block.as_ptr().add(2).cast::<i8>(),
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
                let input_offset = blk * BLOCK_SIZE;
                let block_input_end = (input_offset + BLOCK_SIZE).min(n_cols);
                let block_input_len = block_input_end - input_offset;

                if block_input_len == BLOCK_SIZE {
                    // Full block: NEON fast path.
                    // SAFETY: block has 34 bytes; quant data at block[2..34] = 32 i8 bytes.
                    // input slice has exactly 32 f32 elements.
                    sum += unsafe {
                        dot_block_neon(
                            block.as_ptr().add(2).cast::<i8>(),
                            d,
                            &input[input_offset..input_offset + BLOCK_SIZE],
                        )
                    };
                } else {
                    // Partial tail block: scalar fallback.
                    for i in 0..block_input_len {
                        let q = block[2 + i] as i8;
                        sum += q as f32 * d * input[input_offset + i];
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
        "Q8_0_Neon"
    }

    /// Fused Q8_0 weight × Q8_0 activation GEMV using NEON.
    ///
    /// Both blocks share the 34-byte format.  Accumulates into `out`.
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
        let acts_needed = blocks_per_row * BLOCK_BYTES;

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
            // SAFETY: bounds checked above.
            let row_sum = unsafe {
                fused_q8_0_q8_0_row_neon(
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
}

/// Fused Q8_0 weight × Q8_0 activation dot product for one row using NEON.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `acts_q8.len() >= blocks_per_row * BLOCK_BYTES`
/// - Must run on AArch64 with NEON.
unsafe fn fused_q8_0_q8_0_row_neon(
    row_data: &[u8],
    acts_q8: &[u8],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut row_sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let w_off = blk * BLOCK_BYTES;
        // SAFETY: blk < blocks_per_row; row_data.len() == blocks_per_row * BLOCK_BYTES.
        let w_block = &row_data[w_off..w_off + BLOCK_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([w_block[0], w_block[1]]));

        let a_off = blk * BLOCK_BYTES;
        // SAFETY: acts_q8.len() >= blocks_per_row * BLOCK_BYTES.
        let a_block = &acts_q8[a_off..a_off + BLOCK_BYTES];
        let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));

        let scale = d_w * d_a;

        let input_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(input_offset);

        if remaining >= BLOCK_SIZE {
            // Fast path: full 32-weight block using NEON i8 dot products.
            // SAFETY: w_block[2..34] and a_block[2..34] are 32 valid i8 bytes.
            let wptr = w_block.as_ptr().add(2) as *const i8;
            let aptr = a_block.as_ptr().add(2) as *const i8;

            // Load 4×8-lane i8 vectors.
            let w0 = vld1_s8(wptr);
            let w1 = vld1_s8(wptr.add(8));
            let w2 = vld1_s8(wptr.add(16));
            let w3 = vld1_s8(wptr.add(24));
            let a0 = vld1_s8(aptr);
            let a1 = vld1_s8(aptr.add(8));
            let a2 = vld1_s8(aptr.add(16));
            let a3 = vld1_s8(aptr.add(24));

            // i8 → i16 multiply-add using vmull_s8 (8 lanes × 2 = 16-lane i16).
            let prod0 = vmull_s8(w0, a0);
            let prod1 = vmull_s8(w1, a1);
            let prod2 = vmull_s8(w2, a2);
            let prod3 = vmull_s8(w3, a3);

            // Pair-sum to i32 (vpaddlq_s16: add adjacent pairs).
            let acc0 = vpaddlq_s16(prod0);
            let acc1 = vpaddlq_s16(prod1);
            let acc2 = vpaddlq_s16(prod2);
            let acc3 = vpaddlq_s16(prod3);

            // Sum all four accumulators.
            let total = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            let dot_i32 = vaddvq_s32(total);
            row_sum += scale * dot_i32 as f32;
        } else if remaining > 0 {
            // Scalar tail.
            let q_w = &w_block[2..];
            let q_a = &a_block[2..];
            let mut partial = 0.0f32;
            for i in 0..remaining {
                partial += (q_w[i] as i8 as f32) * (q_a[i] as i8 as f32);
            }
            row_sum += scale * partial;
        }
    }

    row_sum
}

#[cfg(all(test, feature = "simd-neon", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::reference::q8_0::Q8_0Ref;
    use crate::traits::QuantKernel;
    use crate::types::QuantTensor;

    /// Build a Q8_0 block from scale and 32 i8 values.
    fn make_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    /// Fixed reproducible quant values.
    fn fixed_quants() -> [i8; 32] {
        [
            10, -20, 30, -40, 50, -60, 70, -80, 90, -90, 80, -70, 60, -50, 40, -30, 20, -10, 5,
            -15, 25, -35, 45, -55, 65, -75, 85, -95, 100, 127, -127, 0,
        ]
    }

    /// Fixed input vector.
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
        let block = make_block(1.0, &[0i8; 32]);
        let neon = Q8_0Neon;
        let mut out = vec![0.0f32; 32];
        neon.dequant_block(&block, &mut out).expect("dequant");
        for &v in &out {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_matches_reference() {
        let values = fixed_quants();
        let block = make_block(0.5, &values);

        let neon = Q8_0Neon;
        let ref_k = Q8_0Ref;
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
        assert!(max_err < 1e-3, "dequant_block max error {max_err}");
    }

    #[test]
    fn test_dequant_block_extreme_values() {
        let mut values = [0i8; 32];
        values[0] = 127;
        values[1] = -128;
        values[31] = 1;
        let block = make_block(0.1, &values);

        let neon = Q8_0Neon;
        let ref_k = Q8_0Ref;
        let mut out_neon = vec![0.0f32; 32];
        let mut out_ref = vec![0.0f32; 32];

        neon.dequant_block(&block, &mut out_neon).expect("neon");
        ref_k.dequant_block(&block, &mut out_ref).expect("ref");

        let max_err = out_neon
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "extreme values max error {max_err}");
    }

    // ── gemv ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gemv_zeros_output() {
        let block = make_block(1.0, &[0i8; 32]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);
        let input = vec![1.0f32; 32];
        let mut output = vec![9.9f32; 1];
        Q8_0Neon.gemv(&tensor, &input, &mut output).expect("gemv");
        assert!(output[0].abs() < 1e-5, "expected 0, got {}", output[0]);
    }

    #[test]
    fn test_gemv_matches_reference_single_row() {
        let values = fixed_quants();
        let block = make_block(0.125, &values);
        let input = fixed_input();

        let tensor_neon = QuantTensor::new(
            block.clone(),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let tensor_ref = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q8_0Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref");

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
        let n_rows = 4usize;
        let n_cols = 64usize;
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let values = fixed_quants();
        let scales = [0.1f32, 0.25, 0.5, 1.0];
        let mut data = Vec::with_capacity(n_rows * blocks_per_row * BLOCK_BYTES);
        for &s in &scales {
            for _ in 0..blocks_per_row {
                data.extend_from_slice(&make_block(s, &values));
            }
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 - 32.0) * 0.05).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q8_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q8_0Ref
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref");

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
        let n_rows = 1usize;
        let n_cols = 48usize; // 1 full block + 16 scalar tail
        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let values = fixed_quants();
        let mut data = Vec::with_capacity(blocks_per_row * BLOCK_BYTES);
        for _ in 0..blocks_per_row {
            data.extend_from_slice(&make_block(0.5, &values));
        }

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.1).collect();

        let tensor_neon = QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );
        let tensor_ref = QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Q8_0,
        );

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q8_0Neon
            .gemv(&tensor_neon, &input, &mut out_neon)
            .expect("neon");
        Q8_0Ref
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

    #[test]
    fn test_gemv_known_value() {
        // Simple known-value test: 1 row, 32 cols, one non-zero weight
        let mut values = [0i8; 32];
        values[0] = 1;
        let block = make_block(2.0, &values);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);

        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        let mut output = vec![0.0f32; 1];
        Q8_0Neon.gemv(&tensor, &input, &mut output).expect("gemv");

        // weight[0] = 1 * 2.0 = 2.0; dot = 2.0 * 3.0 = 6.0
        assert!(
            (output[0] - 6.0).abs() < 0.1,
            "expected 6.0, got {}",
            output[0]
        );
    }

    // ── matvec_q8_fused ───────────────────────────────────────────────────

    fn make_q8_0_block_fused(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    #[test]
    fn test_q8_0_neon_fused_matches_reference_single_block() {
        let w_vals: [i8; 32] = [
            10, -20, 30, -40, 50, -60, 70, -80, 90, -100, 110, -120, 100, -90, 80, -70, 60, -50,
            40, -30, 20, -10, 5, -15, 25, -35, 45, -55, 65, -75, 85, -95,
        ];
        let a_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 0, 1, -1, 2, -2, 3, -3,
            4, -4, 5, -5, 6, -6, 7, -7, 8,
        ];
        let w_block = make_q8_0_block_fused(0.5, &w_vals);
        let a_block = make_q8_0_block_fused(0.1, &a_vals);

        let mut out_neon = vec![0.0f32; 1];
        let mut out_ref = vec![0.0f32; 1];

        Q8_0Neon
            .matvec_q8_fused(&w_block, &a_block, &mut out_neon, 1, 32)
            .expect("neon fused single block");
        Q8_0Ref
            .matvec_q8_fused(&w_block, &a_block, &mut out_ref, 1, 32)
            .expect("ref fused single block");

        let err = (out_neon[0] - out_ref[0]).abs();
        assert!(
            err < 1e-3,
            "fused single-block mismatch: neon={} ref={} err={}",
            out_neon[0],
            out_ref[0],
            err
        );
    }

    #[test]
    fn test_q8_0_neon_fused_multi_row() {
        let n_rows = 3usize;
        let n_cols = 64usize;
        let blocks_per_row = 2usize;

        let w_vals_a: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16,
        ];
        let w_vals_b: [i8; 32] = [
            20, -30, 40, -50, 60, -70, 80, -90, 100, -110, 120, -127, 100, -90, 80, -70, 50, -40,
            30, -20, 10, -5, 15, -25, 35, -45, 55, -65, 75, -85, 95, -100,
        ];
        let scales = [0.25f32, 0.5f32, 1.0f32];
        let mut all_weights = Vec::new();
        for &s in &scales {
            all_weights.extend(make_q8_0_block_fused(s, &w_vals_a));
            all_weights.extend(make_q8_0_block_fused(s * 0.5, &w_vals_b));
        }

        let a_vals: [i8; 32] = [
            2, -3, 5, -7, 1, -1, 4, -4, 6, -6, 3, -3, 2, -2, 1, -1, 8, -8, 7, -7, 6, -6, 5, -5, 4,
            -4, 3, -3, 2, -2, 1, -1,
        ];
        let mut acts = Vec::new();
        for _ in 0..blocks_per_row {
            acts.extend(make_q8_0_block_fused(0.05, &a_vals));
        }

        let mut out_neon = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Q8_0Neon
            .matvec_q8_fused(&all_weights, &acts, &mut out_neon, n_rows, n_cols)
            .expect("neon fused multi-row");
        Q8_0Ref
            .matvec_q8_fused(&all_weights, &acts, &mut out_ref, n_rows, n_cols)
            .expect("ref fused multi-row");

        for i in 0..n_rows {
            let err = (out_neon[i] - out_ref[i]).abs();
            assert!(
                err < 1e-3,
                "fused multi-row row {i}: neon={} ref={} err={}",
                out_neon[i],
                out_ref[i],
                err
            );
        }
    }

    #[test]
    fn test_q8_0_neon_fused_accumulate_semantics() {
        let w_block = make_q8_0_block_fused(1.0, &[0i8; 32]);
        let a_block = make_q8_0_block_fused(1.0, &[0i8; 32]);

        let mut out = vec![55.0f32; 1];
        Q8_0Neon
            .matvec_q8_fused(&w_block, &a_block, &mut out, 1, 32)
            .expect("neon fused accumulate");

        assert!(
            (out[0] - 55.0).abs() < 1e-5,
            "accumulate semantics broken: expected 55.0, got {}",
            out[0]
        );
    }
}
