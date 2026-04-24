//! IQ1_M NEON-optimised kernel.
//!
//! Block format (56 bytes / 256 weights, QK_K = 256):
//! - bytes[0..32]:  `qs[32]`      — lower 8 bits of eleven-bit grid indices
//! - bytes[32..48]: `qh[16]`      — 16 bytes of sub-block headers
//! - bytes[48..56]: `scales[8]`   — 8 bytes of packed sub-block scales (4 × u16 LE)
//!
//! The global scale `d` is reconstructed from the upper nibbles of the four
//! `scales` u16 words. Scalar decode fills a `[f32; 256]` scratch buffer;
//! NEON `vfmaq_f32` is used for the dot-product in gemv.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq1s_grid::IQ1S_GRID;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 56;
const QS_OFFSET: usize = 0;
const QH_OFFSET: usize = 32;
const SCALES_OFFSET: usize = 48;
const N_SUBBLOCKS: usize = 8;
const SUB_BLOCK_SIZE: usize = BLOCK_SIZE / N_SUBBLOCKS; // 32
const GROUPS_PER_SUB: usize = 4;
const WEIGHTS_PER_GROUP: usize = 8;
const DELTA: f32 = 0.125;

/// NEON-accelerated IQ1_M kernel.
pub struct Iq1MNeon;

/// Reconstruct the global FP16 scale `d` from the IQ1_M scales field.
#[inline]
fn reconstruct_d(scales: &[u8]) -> f32 {
    let sc0 = u16::from_le_bytes([scales[0], scales[1]]);
    let sc1 = u16::from_le_bytes([scales[2], scales[3]]);
    let sc2 = u16::from_le_bytes([scales[4], scales[5]]);
    let sc3 = u16::from_le_bytes([scales[6], scales[7]]);
    let d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    half::f16::from_bits(d_bits).to_f32()
}

/// Decode one IQ1_M block (256 weights) into `output` using scalar arithmetic.
fn decode_block(block: &[u8], output: &mut [f32]) {
    let qs = &block[QS_OFFSET..QH_OFFSET];
    let qh = &block[QH_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    let d = reconstruct_d(scales);

    let sc: [u16; 4] = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];

    for ib in 0..N_SUBBLOCKS {
        let sc_pair = sc[ib / 2];
        let sc_shift_base = 6 * (ib % 2);

        let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
        let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

        let qs_base = ib * GROUPS_PER_SUB;
        let qh_base = ib * 2;

        let qh0 = qh[qh_base] as usize;
        let qh1 = qh[qh_base + 1] as usize;

        let idx: [usize; 4] = [
            (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
            (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
            (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
            (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
        ];

        let delta: [f32; 4] = [
            if qh[qh_base] & 0x08 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base] & 0x80 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base + 1] & 0x08 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base + 1] & 0x80 != 0 {
                -DELTA
            } else {
                DELTA
            },
        ];

        let output_base = ib * SUB_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUB {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();
            let group_base = output_base + l * WEIGHTS_PER_GROUP;
            for j in 0..WEIGHTS_PER_GROUP {
                let gv = grid_raw[j] as i8 as f32;
                output[group_base + j] = dl * (gv + delta[l]);
            }
        }
    }
}

impl QuantKernel for Iq1MNeon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ1_M-NEON"
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
    use crate::reference::iq1_m::Iq1MRef;
    use oxillama_gguf::GgufTensorType;

    fn make_zero_block() -> Vec<u8> {
        // IQ1_M has no explicit `d` field — scales encodes it via upper nibbles.
        // With all-zero scales, d = f16::from_bits(0) = 0.0, output is all zeros.
        vec![0u8; BLOCK_BYTES]
    }

    #[test]
    fn test_dequant_block_basic() {
        let block = make_zero_block();
        let mut out = vec![0.0f32; BLOCK_SIZE];
        Iq1MNeon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        // Encode d ≈ 1.0 into scales nibbles: set sc3 = 0x3C00 (f16 = 1.0)
        block[48] = 0x00;
        block[49] = 0x00;
        block[50] = 0x00;
        block[51] = 0x00;
        block[52] = 0x00;
        block[53] = 0x00;
        block[54] = 0x00; // sc3 low
        block[55] = 0x3C; // sc3 high → d_bits |= sc3 & 0xf000 = 0x3C00 = f16(1.0)

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Iq1MNeon
            .dequant_block(&block, &mut neon_out)
            .expect("neon dequant failed");
        Iq1MRef
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
            tensor_type: GgufTensorType::Iq1M,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq1MNeon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
