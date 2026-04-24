//! IQ3_S NEON-optimised kernel.
//!
//! Block format (110 bytes / 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d`
//! - bytes\[2..66\]:   qs\[64\] — grid base indices (8 per super-block)
//! - bytes\[66..74\]:  qh\[8\]  — high bits for grid indices
//! - bytes\[74..106\]: signs\[32\] — per-group sign masks
//! - bytes\[106..110\]: scales\[4\] — nibble-packed scales, two super-blocks per byte

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 110;
const N_SUPERBLOCKS: usize = 8;
const SUPER_SIZE: usize = 32;
const GROUPS_PER_SUPER: usize = 4;
const QS_OFFSET: usize = 2;
const QS_BYTES: usize = 64;
const QH_OFFSET: usize = 66;
const QH_BYTES: usize = 8;
const SIGNS_OFFSET: usize = 74;
const SIGNS_BYTES: usize = 32;
const SCALES_OFFSET: usize = 106;

/// NEON-accelerated IQ3_S kernel.
pub struct Iq3SNeon;

/// Dequantize one IQ3_S super-block into `out[0..32]`.
#[inline]
fn dequant_superblock(qs8: &[u8], qh_byte: u8, signs4: &[u8], db: f32, out: &mut [f32]) {
    for l in 0..GROUPS_PER_SUPER {
        let idx1 = qs8[2 * l] as usize | (((qh_byte as usize) << (8 - 2 * l)) & 256);
        let idx2 = qs8[2 * l + 1] as usize | (((qh_byte as usize) << (7 - 2 * l)) & 256);

        let g1: [u8; 4] = IQ3S_GRID[idx1].to_le_bytes();
        let g2: [u8; 4] = IQ3S_GRID[idx2].to_le_bytes();
        let sign_byte = signs4[l];
        let group_base = l * 8;

        for j in 0..4 {
            let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            out[group_base + j] = db * g1[j] as f32 * sign1;
            out[group_base + j + 4] = db * g2[j] as f32 * sign2;
        }
    }
}

fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
    let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
    let signs = &block[SIGNS_OFFSET..SIGNS_OFFSET + SIGNS_BYTES];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    let mut ib32 = 0usize;
    while ib32 < N_SUPERBLOCKS {
        let pair = ib32 / 2;
        let scale_byte = scales[pair];
        let db1 = d * (1.0 + 2.0 * (scale_byte & 0xf) as f32);
        let db2 = d * (1.0 + 2.0 * (scale_byte >> 4) as f32);

        dequant_superblock(
            &qs[8 * ib32..8 * ib32 + 8],
            qh[ib32],
            &signs[4 * ib32..4 * ib32 + 4],
            db1,
            &mut output[ib32 * SUPER_SIZE..(ib32 + 1) * SUPER_SIZE],
        );

        let ib32b = ib32 + 1;
        dequant_superblock(
            &qs[8 * ib32b..8 * ib32b + 8],
            qh[ib32b],
            &signs[4 * ib32b..4 * ib32b + 4],
            db2,
            &mut output[ib32b * SUPER_SIZE..(ib32b + 1) * SUPER_SIZE],
        );

        ib32 += 2;
    }
}

impl QuantKernel for Iq3SNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "IQ3_S-NEON"
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

                // SAFETY: scratch has BLOCK_SIZE valid f32s; input is valid; AArch64 NEON.
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
    use crate::reference::iq3_s::Iq3SRef;
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
        Iq3SNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        block[2] = 0x55;
        block[66] = 0xAA;
        block[106] = 0x3C;

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq3SNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon failed");
        Iq3SRef
            .dequant_block(&block, &mut ref_out)
            .expect("ref failed");

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
            tensor_type: GgufTensorType::Iq3S,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq3SNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
