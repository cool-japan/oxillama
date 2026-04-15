//! F16 (half-precision float) pass-through kernel.
//!
//! F16 block format (2 bytes per 1 weight):
//! - 2 bytes: FP16 value
//!
//! No quantization — simply converts FP16 → FP32.
//!
//! Effective: 16 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const F16_BLOCK_SIZE: usize = 1;
const F16_BLOCK_BYTES: usize = 2;

/// F16 pass-through kernel (dequant is just FP16 → FP32 conversion).
pub struct F16Ref;

impl QuantKernel for F16Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < F16_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: F16_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: F16_BLOCK_SIZE,
                available: 0,
            });
        }

        output[0] = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
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

        let row_bytes = n_cols * F16_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for (col, &inp) in input.iter().enumerate().take(n_cols) {
                let offset = row_start + col * F16_BLOCK_BYTES;
                let w = half::f16::from_bits(u16::from_le_bytes([
                    quant_matrix.data[offset],
                    quant_matrix.data[offset + 1],
                ]))
                .to_f32();
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
        F16_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        F16_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "F16"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_dequant() {
        let val = 3.125f32; // exact in FP16
        let bits = half::f16::from_f32(val).to_bits();
        let block = bits.to_le_bytes();
        let kernel = F16Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!(
            (output[0] - val).abs() < 0.01,
            "expected ~{val}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_f16_zero() {
        let block = [0u8; 2];
        let kernel = F16Ref;
        let mut output = [0.0f32; 1];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: f16 zero");
        assert!((output[0]).abs() < 1e-5);
    }

    #[test]
    fn test_f16_dequant_block_too_small_errors() {
        let kernel = F16Ref;
        let mut output = [0.0f32; 1];
        assert!(
            kernel.dequant_block(&[0u8; 0], &mut output).is_err(),
            "empty block should error"
        );
    }

    #[test]
    fn test_f16_dequant_output_too_small_errors() {
        let kernel = F16Ref;
        let block = [0u8; 2];
        let mut output: [f32; 0] = [];
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "empty output should error"
        );
    }

    #[test]
    fn test_f16_gemv_2x2() {
        let kernel = F16Ref;
        let vals = [2.0f32, 3.0, 5.0, 7.0];
        let mut data = Vec::new();
        for &v in &vals {
            data.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F16);
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: f16 gemv");
        // row0 = 2+3 = 5, row1 = 5+7 = 12
        assert!((output[0] - 5.0).abs() < 0.1, "row0: {}", output[0]);
        assert!((output[1] - 12.0).abs() < 0.1, "row1: {}", output[1]);
    }

    #[test]
    fn test_f16_gemv_input_too_small_errors() {
        let kernel = F16Ref;
        let data = vec![0u8; 4];
        let tensor =
            crate::types::QuantTensor::new(data, vec![1, 2], oxillama_gguf::GgufTensorType::F16);
        let input = vec![1.0f32];
        let mut output = vec![0.0f32; 1];
        assert!(kernel.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_f16_gemv_output_too_small_errors() {
        let kernel = F16Ref;
        let data = vec![0u8; 8];
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F16);
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 1];
        assert!(kernel.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_f16_gemm_identity() {
        let kernel = F16Ref;
        // 2x2 identity in f16
        let vals = [1.0f32, 0.0, 0.0, 1.0];
        let mut data = Vec::new();
        for &v in &vals {
            data.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::F16);
        // 2 input rows of length 2
        let input = vec![4.0f32, 6.0, 8.0, 10.0];
        let mut output = vec![0.0f32; 4];
        kernel
            .gemm(&tensor, &input, &mut output, 2, 2, 2)
            .expect("test: f16 gemm");
        assert!((output[0] - 4.0).abs() < 0.1, "output[0]: {}", output[0]);
        assert!((output[1] - 6.0).abs() < 0.1, "output[1]: {}", output[1]);
    }

    #[test]
    fn test_f16_kernel_metadata() {
        let kernel = F16Ref;
        assert_eq!(kernel.block_size(), 1);
        assert_eq!(kernel.block_bytes(), 2);
        assert_eq!(kernel.name(), "F16");
    }
}
