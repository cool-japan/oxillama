//! BF16 (Brain Float 16) pass-through kernel.
//!
//! BF16 block format (2 bytes per 1 weight):
//! - 2 bytes: BF16 value
//!
//! No quantization — simply converts BF16 → FP32.
//! BF16 has the same exponent range as FP32 but only 8 bits of mantissa.
//!
//! Effective: 16 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BF16_BLOCK_SIZE: usize = 1;
const BF16_BLOCK_BYTES: usize = 2;

/// BF16 pass-through kernel (dequant is just BF16 → FP32 conversion).
pub struct Bf16Ref;

/// Convert BF16 bits to f32.
///
/// BF16 is just the upper 16 bits of an IEEE 754 float32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl QuantKernel for Bf16Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < BF16_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: BF16_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.is_empty() {
            return Err(QuantError::BufferTooSmall {
                needed: BF16_BLOCK_SIZE,
                available: 0,
            });
        }

        output[0] = bf16_to_f32(u16::from_le_bytes([block[0], block[1]]));
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

        let row_bytes = n_cols * BF16_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for (col, &inp) in input.iter().enumerate().take(n_cols) {
                let offset = row_start + col * BF16_BLOCK_BYTES;
                let w = bf16_to_f32(u16::from_le_bytes([
                    quant_matrix.data[offset],
                    quant_matrix.data[offset + 1],
                ]));
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
        BF16_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BF16_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "BF16"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_dequant() {
        // 1.0 in BF16: same upper 16 bits as f32 1.0
        // f32 1.0 = 0x3F800000 → BF16 = 0x3F80
        let block = 0x3F80u16.to_le_bytes();
        let kernel = Bf16Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_bf16_zero() {
        let block = [0u8; 2];
        let kernel = Bf16Ref;
        let mut output = [0.0f32; 1];
        kernel.dequant_block(&block, &mut output).unwrap();
        assert!((output[0]).abs() < 1e-5);
    }

    #[test]
    fn test_bf16_negative() {
        // -2.0 in f32: 0xC0000000 → BF16: 0xC000
        let block = 0xC000u16.to_le_bytes();
        let kernel = Bf16Ref;
        let mut output = [0.0f32; 1];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: bf16 negative");
        assert!(
            (output[0] - (-2.0)).abs() < 1e-5,
            "expected -2.0, got {}",
            output[0]
        );
    }

    fn bf16_bytes(val: f32) -> [u8; 2] {
        // BF16 = upper 16 bits of f32
        let bits = val.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        bf16_bits.to_le_bytes()
    }

    #[test]
    fn test_bf16_dequant_block_too_small_errors() {
        let kernel = Bf16Ref;
        let mut output = [0.0f32; 1];
        assert!(
            kernel.dequant_block(&[0u8; 0], &mut output).is_err(),
            "empty block should error"
        );
    }

    #[test]
    fn test_bf16_dequant_output_too_small_errors() {
        let kernel = Bf16Ref;
        let block = [0u8; 2];
        let mut output: [f32; 0] = [];
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "empty output should error"
        );
    }

    #[test]
    fn test_bf16_gemv_2x2() {
        // 2x2 matrix of bf16 values [[1.0, 2.0], [3.0, 4.0]]
        let kernel = Bf16Ref;
        let mut data = Vec::new();
        for &v in &[1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&bf16_bytes(v));
        }
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::Bf16);
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: bf16 gemv");
        // row0 = 1+2 = 3, row1 = 3+4 = 7
        assert!((output[0] - 3.0).abs() < 0.02, "row0: {}", output[0]);
        assert!((output[1] - 7.0).abs() < 0.02, "row1: {}", output[1]);
    }

    #[test]
    fn test_bf16_gemv_input_too_small_errors() {
        let kernel = Bf16Ref;
        let data = vec![0u8; 4]; // 1 row, 2 cols
        let tensor =
            crate::types::QuantTensor::new(data, vec![1, 2], oxillama_gguf::GgufTensorType::Bf16);
        let input = vec![1.0f32]; // only 1 element, need 2
        let mut output = vec![0.0f32; 1];
        assert!(kernel.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_bf16_gemv_output_too_small_errors() {
        let kernel = Bf16Ref;
        let data = vec![0u8; 8]; // 2 rows, 2 cols
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::Bf16);
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 1]; // need 2
        assert!(kernel.gemv(&tensor, &input, &mut output).is_err());
    }

    #[test]
    fn test_bf16_gemm_batched() {
        // gemm: 2 input rows * [2x2 matrix]
        let kernel = Bf16Ref;
        let mut data = Vec::new();
        for &v in &[1.0f32, 0.0, 0.0, 1.0] {
            // identity
            data.extend_from_slice(&bf16_bytes(v));
        }
        let tensor =
            crate::types::QuantTensor::new(data, vec![2, 2], oxillama_gguf::GgufTensorType::Bf16);
        let input = vec![3.0f32, 5.0, 7.0, 11.0]; // [2 x 2]
        let mut output = vec![0.0f32; 4]; // [2 x 2]
        kernel
            .gemm(&tensor, &input, &mut output, 2, 2, 2)
            .expect("test: bf16 gemm");
        // identity * [3,5] = [3,5], identity * [7,11] = [7,11]
        assert!((output[0] - 3.0).abs() < 0.02, "output[0]: {}", output[0]);
        assert!((output[1] - 5.0).abs() < 0.02, "output[1]: {}", output[1]);
        assert!((output[2] - 7.0).abs() < 0.02, "output[2]: {}", output[2]);
        assert!((output[3] - 11.0).abs() < 0.02, "output[3]: {}", output[3]);
    }

    #[test]
    fn test_bf16_kernel_metadata() {
        let kernel = Bf16Ref;
        assert_eq!(kernel.block_size(), 1);
        assert_eq!(kernel.block_bytes(), 2);
        assert_eq!(kernel.name(), "BF16");
    }
}
