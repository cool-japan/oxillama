//! Q8_1 reference (naive) implementation.
//!
//! Q8_1 block format (36 bytes per 32 weights):
//! - 2 bytes: FP16 block scale (d)
//! - 2 bytes: FP16 block sum (s) — d * sum(qs), used for GEMM optimization
//! - 32 bytes: qs — 32 × int8 signed quants
//!
//! Weight formula: `w = d * qs[i]` where `qs[i]` is signed int8.
//!
//! Effective: 9.0 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

/// Reference (naive scalar) Q8_1 kernel.
pub struct Q8_1Ref;

impl QuantKernel for Q8_1Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q8_1_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_1_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q8_1_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_1_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // block[2..4] is the sum (s), not needed for dequant
        let qs = &block[4..36];

        for (i, &q) in qs.iter().enumerate() {
            output[i] = d * (q as i8) as f32;
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

        let blocks_per_row = n_cols.div_ceil(Q8_1_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q8_1_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q8_1_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([data[bo], data[bo + 1]]));
                // s (sum) at bo+2..bo+4 — not needed for dot product
                let qs = &data[bo + 4..bo + 36];
                let inp = &input[blk * Q8_1_BLOCK_SIZE..];

                // Direct int8 dot product
                let mut block_sum = 0.0f32;
                for (i, &q) in qs.iter().enumerate() {
                    block_sum += (q as i8) as f32 * inp[i];
                }
                sum += d * block_sum;
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
        Q8_1_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q8_1_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q8_1"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8_1_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_1_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(d).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        // s = d * sum(qs)
        let s: f32 = d * qs.iter().map(|&q| q as f32).sum::<f32>();
        let s_bits = half::f16::from_f32(s).to_bits();
        block.extend_from_slice(&s_bits.to_le_bytes());
        // qs as unsigned bytes
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_q8_1_block(0.0, &[0; 32]);
        let kernel = Q8_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_positive() {
        // d=0.5, all qs=10 → w = 0.5 * 10 = 5.0
        let block = make_q8_1_block(0.5, &[10; 32]);
        let kernel = Q8_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 5.0).abs() < 0.01, "weight[{i}] = {v}, expected 5.0");
        }
    }

    #[test]
    fn test_dequant_negative() {
        // d=2.0, all qs=-5 → w = 2.0 * (-5) = -10.0
        let block = make_q8_1_block(2.0, &[-5; 32]);
        let kernel = Q8_1Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-10.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -10.0"
            );
        }
    }

    #[test]
    fn test_gemv_q8_1() {
        // Build a block with varied int8 values
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 7 - 64).clamp(-128, 127)) as i8;
        }
        let block = make_q8_1_block(0.5, &qs);

        let kernel = Q8_1Ref;

        // Dequant reference
        let mut dequant = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_1);
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
