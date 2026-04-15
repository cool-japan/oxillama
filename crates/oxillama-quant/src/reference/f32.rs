//! F32 (single-precision float) pass-through kernel.
//!
//! F32 block format (4 bytes per 1 weight):
//! - 4 bytes: IEEE 754 single-precision float
//!
//! No quantization — simply copies f32 values.
//! Used for embedding tables, norm weights, and other unquantized tensors.
//!
//! Effective: 32 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const F32_BLOCK_SIZE: usize = 1;
const F32_BLOCK_BYTES: usize = 4;

/// F32 pass-through kernel (no dequant needed).
pub struct F32Ref;

impl QuantKernel for F32Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < F32_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: F32_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: F32_BLOCK_SIZE,
                available: 0,
            });
        }

        output[0] = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
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

        let row_bytes = n_cols * F32_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for (col, &inp) in input.iter().enumerate().take(n_cols) {
                let offset = row_start + col * F32_BLOCK_BYTES;
                let w = f32::from_le_bytes([
                    quant_matrix.data[offset],
                    quant_matrix.data[offset + 1],
                    quant_matrix.data[offset + 2],
                    quant_matrix.data[offset + 3],
                ]);
                sum += w * inp;
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
        F32_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        F32_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "F32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_dequant() {
        let val = 42.5f32;
        let block = val.to_le_bytes();
        let kernel = F32Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!((output[0] - val).abs() < 1e-6);
    }

    #[test]
    fn test_f32_zero() {
        let block = [0u8; 4];
        let kernel = F32Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_f32_negative() {
        let val = -123.456f32;
        let block = val.to_le_bytes();
        let kernel = F32Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!((output[0] - val).abs() < 1e-3);
    }

    #[test]
    fn test_f32_gemv() {
        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let tensor = QuantTensor::new(data, vec![2, 3], oxillama_gguf::GgufTensorType::F32);
        let input = vec![1.0f32, 1.0, 1.0];
        let mut output = vec![0.0f32; 2];
        let kernel = F32Ref;
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: f32 gemv");
        assert!((output[0] - 6.0).abs() < 1e-5, "row0: {}", output[0]); // 1+2+3
        assert!((output[1] - 15.0).abs() < 1e-5, "row1: {}", output[1]); // 4+5+6
    }

    #[test]
    fn test_f32_gemv_input_too_small_errors() {
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let tensor = QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F32);
        let input = vec![1.0f32]; // only 1 element, need 2
        let mut output = vec![0.0f32; 2];
        assert!(F32Ref.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_f32_gemv_output_too_small_errors() {
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let tensor = QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F32);
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 1]; // only 1 slot, need 2
        assert!(F32Ref.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_f32_gemm_batched() {
        // 2x2 identity matrix, 2 input rows
        let mut data = Vec::new();
        for v in &[1.0f32, 0.0, 0.0, 1.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let tensor = QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F32);
        let input = vec![3.0f32, 5.0, 7.0, 11.0]; // [2 input rows × 2 cols]
        let mut output = vec![0.0f32; 4];
        F32Ref
            .gemm(&tensor, &input, &mut output, 2, 2, 2)
            .expect("test: f32 gemm");
        // identity * each row is the row unchanged
        assert!((output[0] - 3.0).abs() < 1e-5, "output[0]: {}", output[0]);
        assert!((output[1] - 5.0).abs() < 1e-5, "output[1]: {}", output[1]);
        assert!((output[2] - 7.0).abs() < 1e-5, "output[2]: {}", output[2]);
        assert!((output[3] - 11.0).abs() < 1e-5, "output[3]: {}", output[3]);
    }

    #[test]
    fn test_f32_kernel_metadata() {
        assert_eq!(F32Ref.block_size(), 1);
        assert_eq!(F32Ref.block_bytes(), 4);
        assert_eq!(F32Ref.name(), "F32");
    }

    #[test]
    fn test_f32_dequant_block_too_small_errors() {
        let mut output = [0.0f32; 1];
        assert!(F32Ref.dequant_block(&[0u8; 2], &mut output).is_err());
    }

    #[test]
    fn test_f32_dequant_output_too_small_errors() {
        let mut output: [f32; 0] = [];
        assert!(F32Ref.dequant_block(&[0u8; 4], &mut output).is_err());
    }
}
