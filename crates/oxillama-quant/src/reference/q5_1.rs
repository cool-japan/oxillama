//! Q5_1 reference (naive) implementation.
//!
//! Q5_1 block format (24 bytes per 32 weights):
//! - 2 bytes: FP16 block scale (d)
//! - 2 bytes: FP16 block minimum (m)
//! - 4 bytes: qh — 32 high bits (bit 4 of each 5-bit quant)
//! - 16 bytes: qs — 32 × lower 4 bits packed (2 per byte)
//!
//! Weight formula: `w = d * q_5bit + m` where q is 5-bit unsigned (0..31).
//!
//! Effective: 6.0 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q5_1_BLOCK_SIZE: usize = 32;
const Q5_1_BLOCK_BYTES: usize = 24;

/// Reference (naive scalar) Q5_1 kernel.
pub struct Q5_1Ref;

impl QuantKernel for Q5_1Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q5_1_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_1_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q5_1_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_1_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];

        for i in 0..16 {
            let lo_nibble = qs[i] & 0x0F;
            let hi_nibble = (qs[i] >> 4) & 0x0F;

            let hi_bit_0 = ((qh >> i) & 1) as u8;
            let hi_bit_1 = ((qh >> (i + 16)) & 1) as u8;

            let q0 = (lo_nibble | (hi_bit_0 << 4)) as f32;
            let q1 = (hi_nibble | (hi_bit_1 << 4)) as f32;

            output[i] = d * q0 + m;
            output[i + 16] = d * q1 + m;
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

        let blocks_per_row = n_cols.div_ceil(Q5_1_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q5_1_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q5_1_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([data[bo], data[bo + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));
                let qh =
                    u32::from_le_bytes([data[bo + 4], data[bo + 5], data[bo + 6], data[bo + 7]]);
                let qs = &data[bo + 8..bo + 24];
                let input_offset = blk * Q5_1_BLOCK_SIZE;
                let n_remaining = n_cols.saturating_sub(input_offset).min(Q5_1_BLOCK_SIZE);
                let inp = &input[input_offset..input_offset + n_remaining];

                let mut input_sum = 0.0f32;
                for i in 0..16 {
                    let lo_nibble = qs[i] & 0x0F;
                    let hi_nibble = qs[i] >> 4;
                    let hi_bit_0 = ((qh >> i) & 1) as u8;
                    let hi_bit_1 = ((qh >> (i + 16)) & 1) as u8;

                    let q0 = (lo_nibble | (hi_bit_0 << 4)) as f32;
                    let q1 = (hi_nibble | (hi_bit_1 << 4)) as f32;

                    if i < n_remaining {
                        sum += d * q0 * inp[i];
                        input_sum += inp[i];
                    }
                    if i + 16 < n_remaining {
                        sum += d * q1 * inp[i + 16];
                        input_sum += inp[i + 16];
                    }
                }
                sum += m * input_sum;
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
        Q5_1_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q5_1_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q5_1"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q5_1_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_1_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0, m=0 → all 0
        let block = make_q5_1_block(0.0, 0.0, 0, &[0; 16]);
        let kernel = Q5_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_min_only() {
        // d=0, m=3.0, all q=0 → w = 0 + 3.0 = 3.0
        let block = make_q5_1_block(0.0, 3.0, 0, &[0; 16]);
        let kernel = Q5_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_dequant_with_high_bit() {
        // d=1.0, m=0, qh all set, qs all 0
        // q_5bit = 16 (lo=0, hi=1) → w = 1.0 * 16 + 0 = 16.0
        let block = make_q5_1_block(1.0, 0.0, 0xFFFFFFFF, &[0; 16]);
        let kernel = Q5_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 16.0).abs() < 0.01, "weight[{i}] = {v}, expected 16.0");
        }
    }

    #[test]
    fn test_dequant_max() {
        // d=1.0, m=0, qh all set, qs all 0xFF (nibbles=15)
        // q_5bit = 31 (lo=15, hi=1) → w = 31.0
        let block = make_q5_1_block(1.0, 0.0, 0xFFFFFFFF, &[0xFF; 16]);
        let kernel = Q5_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 31.0).abs() < 0.01, "weight[{i}] = {v}, expected 31.0");
        }
    }

    #[test]
    fn test_gemv_q5_1() {
        // Build a block with varied data
        let qh: u32 = 0x5A5A5A5A;
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 9 + 3) & 0xFF) as u8;
        }
        let block = make_q5_1_block(0.5, 0.25, qh, &qs);

        let kernel = Q5_1Ref;

        // Dequant reference
        let mut dequant = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q5_1);
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
