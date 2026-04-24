//! IQ2_XXS NEON-optimised kernel.
//!
//! Block format (66 bytes / 256 weights, QK_K = 256):
//! - bytes[0..2]:  FP16 scale `d`
//! - bytes[2..66]: 32 × u16 packed quants (64 bytes)
//!   Each 8-byte super-block: aux32\[0\] = 4 grid indices, aux32\[1\] = scale + signs.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 66;
const N_SUPERBLOCKS: usize = 8;
const SUPER_SIZE: usize = 32;
const GROUP_SIZE: usize = 8;

/// NEON-accelerated IQ2_XXS kernel.
pub struct Iq2XxsNeon;

/// Decode one IQ2_XXS block into `output[0..256]` (scalar decode).
fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let base = ib32 * 8;
        let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
        let aux32_1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

        let scale_factor = (aux32_1 >> 28) as f32;
        let db = d * (0.5 + scale_factor) * 0.25;

        let aux8: [u8; 4] = aux32_0.to_le_bytes();
        let weight_base = ib32 * SUPER_SIZE;

        for (l, &grid_byte) in aux8.iter().enumerate() {
            let magnitudes: [u8; 8] = IQ2XXS_GRID[grid_byte as usize].to_le_bytes();
            let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
            let sign_byte = KSIGNS_IQ2XS[sign_idx];
            let group_base = weight_base + l * GROUP_SIZE;

            for j in 0..GROUP_SIZE {
                let mag = magnitudes[j] as f32;
                let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                output[group_base + j] = db * mag * sign;
            }
        }
    }
}

impl QuantKernel for Iq2XxsNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "IQ2_XXS-NEON"
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
            // SAFETY: we are on AArch64 with NEON; acc is a valid float32x4_t.
            let mut sum = unsafe { vdupq_n_f32(0.0) };

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * BLOCK_BYTES;
                let block = &quant_matrix.data[bo..bo + BLOCK_BYTES];
                let input_base = blk * BLOCK_SIZE;
                let block_input_len = BLOCK_SIZE.min(n_cols.saturating_sub(input_base));

                decode_block(block, &mut scratch);

                // NEON dot product over decoded weights.
                // SAFETY: scratch has BLOCK_SIZE = 256 valid f32s; input slice is valid.
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
                    // Scalar tail
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
    use crate::reference::iq2_xxs::Iq2XxsRef;
    use oxillama_gguf::GgufTensorType;

    fn make_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        // d = 1.0 as f16
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = (d_bits & 0xff) as u8;
        block[1] = (d_bits >> 8) as u8;
        block
    }

    #[test]
    fn test_dequant_block_basic() {
        let block = make_zero_block();
        let mut out = vec![0.0f32; BLOCK_SIZE];
        Iq2XxsNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        // With zero qs, all grid indices are 0 → deterministic
        // Just check no panic and length OK.
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        // Set some non-trivial qs bytes
        block[2] = 0xAB;
        block[3] = 0x34;
        block[6] = 0x12;
        block[10] = 0xFF;

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq2XxsNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon dequant failed");
        Iq2XxsRef
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
            tensor_type: GgufTensorType::Iq2Xxs,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq2XxsNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
