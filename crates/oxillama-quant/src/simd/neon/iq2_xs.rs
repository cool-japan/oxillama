//! IQ2_XS NEON-optimised kernel.
//!
//! Block format (74 bytes / 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d`
//! - bytes[2..66]:  32 × u16 packed quants (64 bytes), lower 9 bits = grid idx, upper 7 = sign idx
//! - bytes\[66..74\]: 8 scale bytes, one per super-block (low nibble → db\[0\], high nibble → db\[1\])

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 74;
const N_SUPERBLOCKS: usize = 8;
const SUPER_SIZE: usize = 32;
const GROUPS_PER_SUPER: usize = 4;
const GROUP_SIZE: usize = 8;
const QS_OFFSET: usize = 2;
const SCALES_OFFSET: usize = 66;

/// NEON-accelerated IQ2_XS kernel.
pub struct Iq2XsNeon;

fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs_bytes = &block[QS_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    for (ib32, &scale_byte) in scales.iter().enumerate().take(N_SUPERBLOCKS) {
        let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
        let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
        let weight_base = ib32 * SUPER_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let byte_off = ib32 * 8 + l * 2;
            let qs_val = u16::from_le_bytes([qs_bytes[byte_off], qs_bytes[byte_off + 1]]);
            let grid_idx = (qs_val & 511) as usize;
            let sign_idx = (qs_val >> 9) as usize;

            let magnitudes: [u8; 8] = IQ2XS_GRID[grid_idx].to_le_bytes();
            let sign_byte = KSIGNS_IQ2XS[sign_idx];
            let dl = if l < 2 { db0 } else { db1 };
            let group_base = weight_base + l * GROUP_SIZE;

            for j in 0..GROUP_SIZE {
                let mag = magnitudes[j] as f32;
                let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                output[group_base + j] = dl * mag * sign;
            }
        }
    }
}

impl QuantKernel for Iq2XsNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "IQ2_XS-NEON"
    }

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
        decode_block(block, output);
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
        let mut scratch = [0.0f32; BLOCK_SIZE];

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: AArch64 with NEON.
            let mut sum = unsafe { vdupq_n_f32(0.0) };

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * BLOCK_BYTES;
                let block = &quant_matrix.data[bo..bo + BLOCK_BYTES];
                let input_base = blk * BLOCK_SIZE;
                let block_input_len = BLOCK_SIZE.min(n_cols.saturating_sub(input_base));

                decode_block(block, &mut scratch);

                // SAFETY: scratch and input are valid; we are on AArch64 with NEON.
                unsafe {
                    let w_ptr = scratch.as_ptr();
                    let i_ptr = input.as_ptr().add(input_base);
                    let lanes = block_input_len / 4;
                    for k in 0..lanes {
                        let off = k * 4;
                        let wv = vld1q_f32(w_ptr.add(off));
                        let iv = vld1q_f32(i_ptr.add(off));
                        sum = vfmaq_f32(sum, wv, iv);
                    }
                    for k in (lanes * 4)..block_input_len {
                        let s: f32 = scratch[k] * input[input_base + k];
                        sum = vaddq_f32(sum, vdupq_n_f32(s));
                    }
                }
            }

            // SAFETY: AArch64 with NEON.
            *out = unsafe { vaddvq_f32(sum) };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::iq2_xs::Iq2XsRef;
    use oxillama_gguf::GgufTensorType;

    fn make_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = (d_bits & 0xff) as u8;
        block[1] = (d_bits >> 8) as u8;
        block
    }

    #[test]
    fn test_dequant_block_basic() {
        let block = make_zero_block();
        let mut out = vec![0.0f32; BLOCK_SIZE];
        Iq2XsNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        block[2] = 0xAB;
        block[3] = 0x01; // grid idx < 512, sign idx valid
        block[66] = 0x5A; // scale byte

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq2XsNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon dequant failed");
        Iq2XsRef
            .dequant_block(&block, &mut ref_out)
            .expect("ref dequant failed");

        for (i, (&n, &r)) in neon_out.iter().zip(ref_out.iter()).enumerate() {
            assert!((n - r).abs() < 1e-5, "mismatch at {i}: neon={n} ref={r}");
        }
    }

    #[test]
    fn test_gemv_single_row() {
        let block = make_zero_block();
        let data = block.clone();
        let tensor = QuantTensor {
            data,
            shape: vec![1, BLOCK_SIZE],
            tensor_type: GgufTensorType::Iq2Xs,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq2XsNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
