//! IQ1_S NEON-optimised kernel.
//!
//! Block format (50 bytes / 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d`
//! - bytes[2..34]:  `qs[32]` — lower 8 bits of eleven-bit grid indices
//! - bytes[34..50]: `qh[8]`  — 8 × u16 sub-block headers
//!
//! Scalar decode produces a `[f32; 256]` scratch buffer;
//! NEON `vfmaq_f32` is used for the dot-product in gemv.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq1s_grid::IQ1S_GRID;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 50;
const QS_OFFSET: usize = 2;
const QH_OFFSET: usize = 34;
const N_SUBBLOCKS: usize = 8;
const SUB_BLOCK_SIZE: usize = BLOCK_SIZE / N_SUBBLOCKS; // 32
const GROUPS_PER_SUB: usize = 4;
const WEIGHTS_PER_GROUP: usize = 8;
const DELTA: f32 = 0.125;

/// NEON-accelerated IQ1_S kernel.
pub struct Iq1SNeon;

/// Decode one IQ1_S block (256 weights) into `output` using scalar arithmetic.
fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[QS_OFFSET..QH_OFFSET];
    let qh_bytes = &block[QH_OFFSET..BLOCK_BYTES];

    for ib in 0..N_SUBBLOCKS {
        let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);

        let scale_bits = ((qh_val >> 12) & 0x7) as f32;
        let dl = d * (2.0 * scale_bits + 1.0);

        let delta = if qh_val & 0x8000 != 0 { -DELTA } else { DELTA };

        let qs_base = ib * GROUPS_PER_SUB;
        let output_base = ib * SUB_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUB {
            let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
            let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);
            let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

            let group_base = output_base + l * WEIGHTS_PER_GROUP;
            for j in 0..WEIGHTS_PER_GROUP {
                let gv = grid_raw[j] as i8 as f32;
                output[group_base + j] = dl * (gv + delta);
            }
        }
    }
}

impl QuantKernel for Iq1SNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ1_S-NEON"
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
    use crate::reference::iq1_s::Iq1SRef;
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
        Iq1SNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        // Set some non-trivial qs bytes (keep within valid grid range)
        block[2] = 0x12;
        block[3] = 0x34;
        block[34] = 0x10; // qh[0] low
        block[35] = 0x30; // qh[0] high — scale_bits=3, delta positive

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq1SNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon dequant failed");
        Iq1SRef
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
            tensor_type: GgufTensorType::Iq1S,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq1SNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
