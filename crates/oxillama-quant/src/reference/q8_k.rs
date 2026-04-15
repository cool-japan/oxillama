//! Q8_K reference (naive) implementation.
//!
//! Q8_K block format (292 bytes per 256 weights):
//! - 4 bytes: f32 super-block scale (d)
//! - 256 bytes: qs — 256 × int8 signed quants
//! - 32 bytes: bsums — 16 × int16 block sums (for GEMM optimization, not used in dequant)
//!
//! Q8_K is only used as an intermediate format for dot products;
//! it is the quantized-input counterpart to the K-quant family.
//!
//! Weight formula: `w = d * qs[i]` where `qs[i]` is signed int8.
//!
//! Effective: 9.125 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q8_K_BLOCK_SIZE: usize = 256;
const Q8_K_BLOCK_BYTES: usize = 292;

/// Reference (naive scalar) Q8_K kernel.
pub struct Q8KRef;

impl QuantKernel for Q8KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q8_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q8_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];
        // bsums at block[260..292] — not needed for dequant

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

        let blocks_per_row = n_cols.div_ceil(Q8_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q8_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q8_K_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f32::from_le_bytes([data[bo], data[bo + 1], data[bo + 2], data[bo + 3]]);
                let qs = &data[bo + 4..bo + 260];
                let inp = &input[blk * Q8_K_BLOCK_SIZE..];

                // Direct dot product: extract int8, multiply with input on-the-fly
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
        Q8_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q8_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q8_K"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8_k_block(d: f32, qs: &[i8; 256]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_K_BLOCK_BYTES);
        block.extend_from_slice(&d.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        // bsums: 16 × int16 (32 bytes), zero for testing
        block.extend_from_slice(&[0u8; 32]);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        let block = make_q8_k_block(0.0, &[0; 256]);
        let kernel = Q8KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_positive() {
        // d=0.25, all qs=40 → w = 0.25 * 40 = 10.0
        let block = make_q8_k_block(0.25, &[40; 256]);
        let kernel = Q8KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 10.0).abs() < 0.01, "weight[{i}] = {v}, expected 10.0");
        }
    }

    #[test]
    fn test_dequant_negative() {
        // d=1.0, all qs=-100 → w = -100.0
        let block = make_q8_k_block(1.0, &[-100; 256]);
        let kernel = Q8KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-100.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -100.0"
            );
        }
    }

    #[test]
    fn test_gemv_q8_k() {
        // Build a block with varied int8 values
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i as i16 * 3 - 128).clamp(-128, 127)) as i8;
        }
        let block = make_q8_k_block(0.25, &qs);

        let kernel = Q8KRef;

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q8K);
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
