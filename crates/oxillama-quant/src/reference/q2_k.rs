//! Q2_K reference (naive) implementation.
//!
//! Q2_K block format (84 bytes per 256 weights):
//! - 16 bytes: scales — 16 sub-blocks, each byte: lo 4 bits = scale, hi 4 bits = min
//! - 64 bytes: qs — 256 × 2-bit weights packed (4 per byte)
//! - 2 bytes: FP16 super-block scale (d)
//! - 2 bytes: FP16 super-block minimum (dmin)
//!
//! NOTE: In Q2_K, d/dmin come AFTER scales and qs in memory.
//!
//! 16 sub-blocks of 16 weights each.
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 2-bit (0..3).
//!
//! Effective: 2.625 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q2_K_BLOCK_SIZE: usize = 256;
const Q2_K_BLOCK_BYTES: usize = 84;

/// Reference (naive scalar) Q2_K kernel.
pub struct Q2KRef;

impl QuantKernel for Q2KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q2_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q2_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q2_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q2_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut is = 0usize; // sub-block index
        let mut qs_off = 0usize;
        let mut out_off = 0usize;

        // Process in 2 groups of 128 weights
        for _n in 0..2 {
            // Within each group, iterate through 4 shifts (0, 2, 4, 6)
            for shift in (0..8).step_by(2) {
                // First 16 weights of this sub-block
                let sc_byte = scales[is];
                let dl = d * (sc_byte & 0x0F) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;

                for l in 0..16 {
                    let q = (qs[qs_off + l] >> shift) & 3;
                    output[out_off + l] = dl * q as f32 - ml;
                }
                out_off += 16;

                // Second 16 weights of this sub-block
                let sc_byte = scales[is];
                let dl = d * (sc_byte & 0x0F) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;

                for l in 0..16 {
                    let q = (qs[qs_off + 16 + l] >> shift) & 3;
                    output[out_off + l] = dl * q as f32 - ml;
                }
                out_off += 16;
            }
            qs_off += 32;
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

        let blocks_per_row = n_cols.div_ceil(Q2_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q2_K_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let scales = &data[bo..bo + 16];
                let qs = &data[bo + 16..bo + 80];
                let d = f16_to_f32(u16::from_le_bytes([data[bo + 80], data[bo + 81]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[bo + 82], data[bo + 83]]));
                let inp = &input[blk * Q2_K_BLOCK_SIZE..];
                let cols_in_block = (n_cols - blk * Q2_K_BLOCK_SIZE).min(Q2_K_BLOCK_SIZE);

                // Inline dot product: extract 2-bit quants on-the-fly
                let mut is = 0usize;
                let mut qs_off = 0usize;
                let mut in_off = 0usize;

                for _n in 0..2 {
                    for shift in (0..8).step_by(2) {
                        let sc_byte = scales[is];
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        is += 1;
                        for l in 0..16 {
                            if in_off + l < cols_in_block {
                                let q = (qs[qs_off + l] >> shift) & 3;
                                sum += (dl * q as f32 - ml) * inp[in_off + l];
                            }
                        }
                        in_off += 16;

                        let sc_byte = scales[is];
                        let dl = d * (sc_byte & 0x0F) as f32;
                        let ml = dmin * (sc_byte >> 4) as f32;
                        is += 1;
                        for l in 0..16 {
                            if in_off + l < cols_in_block {
                                let q = (qs[qs_off + 16 + l] >> shift) & 3;
                                sum += (dl * q as f32 - ml) * inp[in_off + l];
                            }
                        }
                        in_off += 16;
                    }
                    qs_off += 32;
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
        Q2_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q2_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q2_K"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q2_k_block(d: f32, dmin: f32, scales: &[u8; 16], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0, dmin=0 → all weights = 0
        let block = make_q2_k_block(0.0, 0.0, &[0; 16], &[0; 64]);
        let kernel = Q2KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_uniform() {
        // d=1.0, dmin=0.0, all scales=0x01 (scale=1, min=0), all qs=0xFF (all 2-bit = 3)
        // Weight = 1.0 * 1 * 3 - 0 = 3.0
        let scales = [0x01u8; 16]; // lo=1 (scale), hi=0 (min)
        let qs = [0xFFu8; 64]; // all 2-bit values = 3

        let block = make_q2_k_block(1.0, 0.0, &scales, &qs);
        let kernel = Q2KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_dequant_with_min() {
        // d=2.0, dmin=1.0, all scales=0x11 (scale=1, min=1), all qs=0x00 (all 2-bit = 0)
        // Weight = 2.0 * 1 * 0 - 1.0 * 1 = -1.0
        let scales = [0x11u8; 16]; // lo=1 (scale), hi=1 (min)
        let qs = [0x00u8; 64]; // all 2-bit values = 0

        let block = make_q2_k_block(2.0, 1.0, &scales, &qs);
        let kernel = Q2KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_gemv_q2_k() {
        // Build a block with varied data
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0x21 + i as u8; // varied scale/min
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 3 + 7) & 0xFF) as u8;
        }
        let block = make_q2_k_block(0.5, 0.25, &scales, &qs);

        let kernel = Q2KRef;

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q2K);
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        assert!(
            (output[0] - expected).abs() < 0.1,
            "gemv={}, expected={}",
            output[0],
            expected
        );
    }
}
