//! Q8_0 reference (naive) implementation.
//!
//! Q8_0 block format (34 bytes per 32 weights):
//! - 2 bytes: FP16 scale (d)
//! - 32 bytes: 32 × int8 quantized values
//!
//! Each weight is reconstructed as: `q[i] * d`

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q8_0: 32 weights per block.
const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (scale) + 32 (data).
const Q8_0_BLOCK_BYTES: usize = 34;

/// Reference (naive scalar) Q8_0 kernel.
pub struct Q8_0Ref;

impl QuantKernel for Q8_0Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q8_0_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_0_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q8_0_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q8_0_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        for i in 0..Q8_0_BLOCK_SIZE {
            let q = block[2 + i] as i8;
            output[i] = q as f32 * d;
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

        let blocks_per_row = n_cols.div_ceil(Q8_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q8_0_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + Q8_0_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let input_offset = blk * Q8_0_BLOCK_SIZE;

                for i in 0..Q8_0_BLOCK_SIZE {
                    let idx = input_offset + i;
                    if idx < n_cols {
                        let q = block[2 + i] as i8;
                        sum += q as f32 * d * input[idx];
                    }
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
        Q8_0_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q8_0_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q8_0"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        block
    }

    #[test]
    fn test_dequant_block_zeros() {
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let kernel = Q8_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_block_simple() {
        let mut values = [0i8; 32];
        values[0] = 10;
        values[1] = -5;
        values[31] = 127;
        let block = make_q8_0_block(0.5, &values);
        let kernel = Q8_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();

        assert!((output[0] - 5.0).abs() < 0.01); // 10 * 0.5
        assert!((output[1] - (-2.5)).abs() < 0.01); // -5 * 0.5
        assert!((output[31] - 63.5).abs() < 0.1); // 127 * 0.5
    }

    #[test]
    fn test_gemv_q8_0() {
        let kernel = Q8_0Ref;
        let mut values = [0i8; 32];
        values[0] = 1;
        let block = make_q8_0_block(2.0, &values);

        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);

        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // weight[0] = 1 * 2.0 = 2.0, input[0] = 3.0 → dot = 6.0
        assert!((output[0] - 6.0).abs() < 0.1, "got {}", output[0]);
    }

    #[test]
    fn test_gemm_batched_q8_0() {
        let kernel = Q8_0Ref;
        // 2-row weight matrix with all-zero weights
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let tensor = QuantTensor::new(data, vec![2, 32], oxillama_gguf::GgufTensorType::Q8_0);

        // 2 input rows × 32 cols
        let input = vec![1.0f32; 64];
        // 2 input rows × 2 weight rows = 4 outputs
        let mut output = vec![0.0f32; 4];
        kernel
            .gemm(&tensor, &input, &mut output, 2, 2, 32)
            .expect("test: q8_0 gemm");
        // All weights zero → all outputs must be zero
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-5, "output[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_gemm_nonzero_q8_0() {
        let kernel = Q8_0Ref;
        // 1-row weight matrix: values[0]=4, scale=0.5 → weight[0]=2.0
        let mut values = [0i8; 32];
        values[0] = 4;
        let block = make_q8_0_block(0.5, &values);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);

        // 1 input row, input[0]=3.0 → expected output = 2.0 * 3.0 = 6.0
        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        let mut output = vec![0.0f32; 1];
        kernel
            .gemm(&tensor, &input, &mut output, 1, 1, 32)
            .expect("test: q8_0 gemm nonzero");
        assert!((output[0] - 6.0).abs() < 0.1, "got {}", output[0]);
    }

    #[test]
    fn test_gemv_input_too_small_errors() {
        let kernel = Q8_0Ref;
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q8_0);
        let input = vec![0.0f32; 4]; // need 32
        let mut output = vec![0.0f32; 1];
        assert!(
            kernel.gemv(&tensor, &input, &mut output).is_err(),
            "too-small input should error"
        );
    }

    #[test]
    fn test_gemv_output_too_small_errors() {
        let kernel = Q8_0Ref;
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let tensor = QuantTensor::new(data, vec![2, 32], oxillama_gguf::GgufTensorType::Q8_0);
        let input = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 1]; // need 2
        assert!(
            kernel.gemv(&tensor, &input, &mut output).is_err(),
            "too-small output should error"
        );
    }

    #[test]
    fn test_q8_0_kernel_metadata() {
        let kernel = Q8_0Ref;
        assert_eq!(kernel.block_size(), Q8_0_BLOCK_SIZE);
        assert_eq!(kernel.block_bytes(), Q8_0_BLOCK_BYTES);
        assert_eq!(kernel.name(), "Q8_0");
    }

    #[test]
    fn test_dequant_block_too_small_errors() {
        let kernel = Q8_0Ref;
        let mut output = vec![0.0f32; 32];
        assert!(
            kernel.dequant_block(&[0u8; 4], &mut output).is_err(),
            "block too small should error"
        );
    }

    #[test]
    fn test_dequant_output_too_small_errors() {
        let kernel = Q8_0Ref;
        let block = make_q8_0_block(1.0, &[0i8; 32]);
        let mut output = vec![0.0f32; 1]; // need 32
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "output too small should error"
        );
    }
}
