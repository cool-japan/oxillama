//! IQ4_XS NEON-optimised kernel.
//!
//! Block format (136 bytes / 256 weights):
//! - bytes[0..2]:   FP16 delta `d`
//! - bytes[2..4]:   `scales_h` — 2-bit high parts of the 8 sub-block scales
//! - bytes[4..8]:   `scales_l` — 4-bit low parts of the 8 sub-block scales (two per byte)
//! - bytes[8..136]: 128 nibble-bytes (256 four-bit weights)
//!
//! Dequant: `w = d * ls_signed * KVALUES_IQ4NL[nibble]`

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_shared::KVALUES_IQ4NL;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 136;
const N_SUPERBLOCKS: usize = 8;
const SUB_SIZE: usize = 32;

/// NEON-accelerated IQ4_XS kernel.
pub struct Iq4XsNeon;

#[inline]
fn unpack_sub_scale(scales_h_u16: u16, scales_l: &[u8], i: usize) -> i32 {
    let ls_low: u8 = (scales_l[i / 2] >> (4 * (i & 1))) & 0x0F;
    let ls_high: u8 = (scales_h_u16 >> (2 * i)) as u8 & 0x03;
    let ls: u8 = ls_low | (ls_high << 4);
    (ls as i32).wrapping_sub(32)
}

fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let nibbles = &block[8..136];

    for sub in 0..N_SUPERBLOCKS {
        let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
        let scale = d * ls_signed as f32;
        let nibble_offset = sub * (SUB_SIZE / 2);
        let weight_offset = sub * SUB_SIZE;

        for i in 0..(SUB_SIZE / 2) {
            let byte = nibbles[nibble_offset + i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            output[weight_offset + i * 2] = scale * KVALUES_IQ4NL[lo] as f32;
            output[weight_offset + i * 2 + 1] = scale * KVALUES_IQ4NL[hi] as f32;
        }
    }
}

impl QuantKernel for Iq4XsNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "IQ4_XS-NEON"
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

                // SAFETY: scratch and input are valid; AArch64 with NEON.
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
    use crate::reference::iq4_xs::Iq4XsRef;
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
        Iq4XsNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        block[2] = 0xFF;
        block[3] = 0x0F;
        block[4] = 0xAB;
        block[8] = 0x9F;
        block[9] = 0x52;

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq4XsNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon failed");
        Iq4XsRef
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
            tensor_type: GgufTensorType::Iq4Xs,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq4XsNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
