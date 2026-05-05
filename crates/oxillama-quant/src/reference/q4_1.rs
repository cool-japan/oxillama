//! Q4_1 reference (naive) implementation.
//!
//! Q4_1 block format (20 bytes per 32 weights):
//! - 2 bytes: FP16 block scale (d)
//! - 2 bytes: FP16 block minimum (m)
//! - 16 bytes: 32 × 4-bit unsigned nibbles packed (2 per byte)
//!
//! Weight formula: `w = d * nibble + m` where nibble is 4-bit (0..15).
//!
//! Effective: 5.0 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q4_1_BLOCK_SIZE: usize = 32;
const Q4_1_BLOCK_BYTES: usize = 20;

/// Reference (naive scalar) Q4_1 kernel.
pub struct Q4_1Ref;

impl QuantKernel for Q4_1Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q4_1_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_1_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q4_1_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_1_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        for i in 0..Q4_1_BLOCK_SIZE / 2 {
            let byte = block[4 + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            output[i * 2] = d * lo + m;
            output[i * 2 + 1] = d * hi + m;
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

        let blocks_per_row = n_cols.div_ceil(Q4_1_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q4_1_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([data[bo], data[bo + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));
                let qs = &data[bo + 4..bo + 20];
                let input_offset = blk * Q4_1_BLOCK_SIZE;
                let n_remaining = n_cols.saturating_sub(input_offset).min(Q4_1_BLOCK_SIZE);
                let inp = &input[input_offset..input_offset + n_remaining];

                let mut input_sum = 0.0f32;
                for (i, &q) in qs.iter().enumerate().take(16) {
                    let lo = (q & 0x0F) as f32;
                    let hi = (q >> 4) as f32;
                    let j0 = i * 2;
                    let j1 = i * 2 + 1;
                    if j0 < n_remaining {
                        sum += d * lo * inp[j0];
                        input_sum += inp[j0];
                    }
                    if j1 < n_remaining {
                        sum += d * hi * inp[j1];
                        input_sum += inp[j1];
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
        Q4_1_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q4_1_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_1"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4_1_block(d: f32, m: f32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_1_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0, m=0, qs=0 → w = 0*0 + 0 = 0
        let block = make_q4_1_block(0.0, 0.0, &[0; 16]);
        let kernel = Q4_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_min_only() {
        // d=0, m=5.0, qs=0 → w = 0*0 + 5.0 = 5.0
        let block = make_q4_1_block(0.0, 5.0, &[0; 16]);
        let kernel = Q4_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 5.0).abs() < 0.01, "weight[{i}] = {v}, expected 5.0");
        }
    }

    #[test]
    fn test_dequant_scale_and_min() {
        // d=1.0, m=2.0, all nibbles=8 → w = 1.0*8 + 2.0 = 10.0
        // byte = 0x88 (lo=8, hi=8)
        let block = make_q4_1_block(1.0, 2.0, &[0x88; 16]);
        let kernel = Q4_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 10.0).abs() < 0.01, "weight[{i}] = {v}, expected 10.0");
        }
    }

    #[test]
    fn test_gemv_q4_1() {
        // Build a block with varied nibbles
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 7 + 3) & 0xFF) as u8;
        }
        let block = make_q4_1_block(0.5, 0.25, &qs);

        let kernel = Q4_1Ref;

        // Dequant reference
        let mut dequant = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_1);
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
