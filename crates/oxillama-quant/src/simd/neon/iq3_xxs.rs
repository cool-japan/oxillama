//! IQ3_XXS NEON-optimised kernel.
//!
//! Block format (98 bytes / 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d`
//! - bytes[2..66]:  `qs_grid[64]` — grid base indices (2 per 8-weight group)
//! - bytes[66..98]: `qs_signs[32]` — packed scale + sign data (4 bytes per super-block)
//!
//! Scalar decode fills a `[f32; 256]` scratch buffer;
//! NEON `vfmaq_f32` is used for the dot-product in gemv.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 98;
const N_SUPERBLOCKS: usize = 8;
const SUPER_BLOCK_SIZE: usize = 32;
const GROUPS_PER_SUPER: usize = 4;
const WEIGHTS_PER_GROUP: usize = 8;
const SIGNS_OFFSET: usize = 64; // within the `qs` region (bytes 2..98)

/// NEON-accelerated IQ3_XXS kernel.
pub struct Iq3XxsNeon;

/// Decode one IQ3_XXS block (256 weights) into `output` using scalar arithmetic.
fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..BLOCK_BYTES];
    let qs_grid = &qs[..SIGNS_OFFSET];
    let qs_signs = &qs[SIGNS_OFFSET..];

    for ib32 in 0..N_SUPERBLOCKS {
        let signs_base = ib32 * 4;
        let aux32 = u32::from_le_bytes([
            qs_signs[signs_base],
            qs_signs[signs_base + 1],
            qs_signs[signs_base + 2],
            qs_signs[signs_base + 3],
        ]);

        let scale_bits = (aux32 >> 28) as f32;
        let db = d * (0.5 + scale_bits) * 0.5;

        let grid_base = ib32 * 8;
        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let g1 = qs_grid[grid_base + 2 * l] as usize;
            let g2 = qs_grid[grid_base + 2 * l + 1] as usize;
            let mags1: [u8; 4] = IQ3XXS_GRID[g1].to_le_bytes();
            let mags2: [u8; 4] = IQ3XXS_GRID[g2].to_le_bytes();

            let sign_idx = ((aux32 >> (7 * l)) & 0x7F) as usize;
            let sign_byte = KSIGNS_IQ2XS[sign_idx];

            let group_base = weight_base + l * WEIGHTS_PER_GROUP;
            for j in 0..4 {
                let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                output[group_base + j] = db * mags1[j] as f32 * sign1;

                let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                output[group_base + j + 4] = db * mags2[j] as f32 * sign2;
            }
        }
    }
}

impl QuantKernel for Iq3XxsNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ3_XXS-NEON"
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
            // SAFETY: AArch64 with NEON; vdupq_n_f32 is always safe on this target.
            let mut sum = unsafe { vdupq_n_f32(0.0) };

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * BLOCK_BYTES;
                let block = &quant_matrix.data[bo..bo + BLOCK_BYTES];
                let input_base = blk * BLOCK_SIZE;
                let block_input_len = BLOCK_SIZE.min(n_cols.saturating_sub(input_base));

                decode_block(block, &mut scratch);

                // SAFETY: scratch has BLOCK_SIZE valid f32s; input slice is valid;
                //         AArch64 NEON is guaranteed by the cfg gate.
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
    use crate::reference::iq3_xxs::Iq3XxsRef;
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
        Iq3XxsNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        block[2] = 0x05;
        block[3] = 0x17;
        block[66] = 0x3F; // qs_signs[0]
        block[67] = 0x00;
        block[68] = 0x00;
        block[69] = 0x10; // scale_bits = (0x10000000 >> 28) = 1

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq3XxsNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon dequant failed");
        Iq3XxsRef
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
            tensor_type: GgufTensorType::Iq3Xxs,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq3XxsNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
