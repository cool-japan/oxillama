//! Q4_0 reference (naive) implementation.
//!
//! Q4_0 block format (18 bytes per 32 weights):
//! - 2 bytes: FP16 scale (d)
//! - 16 bytes: 32 × 4-bit unsigned nibbles packed into 16 bytes
//!
//! Each weight is reconstructed as: `(nibble - 8) * d`

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q4_0: 32 weights per block.
const Q4_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (scale) + 16 (data).
const Q4_0_BLOCK_BYTES: usize = 18;

/// Reference (naive scalar) Q4_0 kernel.
pub struct Q4_0Ref;

impl QuantKernel for Q4_0Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q4_0_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_0_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q4_0_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_0_BLOCK_SIZE,
                available: output.len(),
            });
        }

        // Read FP16 scale
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        // Dequantize 32 nibbles
        for i in 0..Q4_0_BLOCK_SIZE / 2 {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            output[i * 2] = lo as f32 * d;
            output[i * 2 + 1] = hi as f32 * d;
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

        let blocks_per_row = n_cols.div_ceil(Q4_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q4_0_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + Q4_0_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let input_offset = blk * Q4_0_BLOCK_SIZE;

                for i in 0..Q4_0_BLOCK_SIZE / 2 {
                    let byte = block[2 + i];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    let idx = input_offset + i * 2;
                    if idx + 1 < n_cols {
                        sum += lo as f32 * d * input[idx];
                        sum += hi as f32 * d * input[idx + 1];
                    } else if idx < n_cols {
                        sum += lo as f32 * d * input[idx];
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
        // Implement GEMM as M independent GEMVs for the reference implementation
        for row in 0..m {
            let input_row = &input[row * k..(row + 1) * k];
            let output_row = &mut output[row * n..(row + 1) * n];
            self.gemv(quant_matrix, input_row, output_row)?;
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        Q4_0_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q4_0_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_0"
    }
}

/// Convert an IEEE 754 FP16 half-precision value to FP32.
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4_0_block(scale: f32, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_0_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    #[test]
    fn test_dequant_block_zeros() {
        // All nibbles = 8 (zero after subtracting 8)
        let block = make_q4_0_block(1.0, &[0x88; 16]);
        let kernel = Q4_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_simple() {
        // First byte: lo=0 (val=-8*d), hi=15 (val=7*d)
        let mut nibbles = [0x88u8; 16];
        nibbles[0] = 0xF0; // lo=0 → -8, hi=15 → 7
        let block = make_q4_0_block(0.5, &nibbles);
        let kernel = Q4_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();

        assert!((output[0] - (-4.0)).abs() < 0.01, "got {}", output[0]); // -8 * 0.5
        assert!((output[1] - 3.5).abs() < 0.01, "got {}", output[1]); // 7 * 0.5
    }

    #[test]
    fn test_gemv_identity_like() {
        let kernel = Q4_0Ref;
        // Create a 1-row, 32-col quantized matrix
        let block = make_q4_0_block(1.0, &[0x88; 16]); // all zeros
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_0);

        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // All weights are zero, so output should be zero
        assert!((output[0]).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_batched_q4_0() {
        let kernel = Q4_0Ref;
        // 2-row weight matrix, each row is 1 block of 32 cols with all-zero weights
        let block = make_q4_0_block(1.0, &[0x88; 16]); // all zero weights
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let tensor = QuantTensor::new(data, vec![2, 32], oxillama_gguf::GgufTensorType::Q4_0);

        // 2 input rows × 32 cols
        let input = vec![1.0f32; 64];
        // 2 input rows × 2 weight rows = 4 outputs
        let mut output = vec![0.0f32; 4];
        kernel
            .gemm(&tensor, &input, &mut output, 2, 2, 32)
            .expect("test: q4_0 gemm");
        // All weights are zero → all outputs must be zero
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-5, "output[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_gemv_input_too_small_errors() {
        let kernel = Q4_0Ref;
        let block = make_q4_0_block(1.0, &[0x88; 16]);
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q4_0);
        let input = vec![0.0f32; 4]; // need 32
        let mut output = vec![0.0f32; 1];
        assert!(
            kernel.gemv(&tensor, &input, &mut output).is_err(),
            "too-small input should error"
        );
    }

    #[test]
    fn test_gemv_output_too_small_errors() {
        let kernel = Q4_0Ref;
        let block = make_q4_0_block(1.0, &[0x88; 16]);
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let tensor = QuantTensor::new(data, vec![2, 32], oxillama_gguf::GgufTensorType::Q4_0);
        let input = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 1]; // need 2
        assert!(
            kernel.gemv(&tensor, &input, &mut output).is_err(),
            "too-small output should error"
        );
    }

    #[test]
    fn test_q4_0_kernel_metadata() {
        let kernel = Q4_0Ref;
        assert_eq!(kernel.block_size(), Q4_0_BLOCK_SIZE);
        assert_eq!(kernel.block_bytes(), Q4_0_BLOCK_BYTES);
        assert_eq!(kernel.name(), "Q4_0");
    }

    #[test]
    fn test_dequant_block_too_small_errors() {
        let kernel = Q4_0Ref;
        let mut output = vec![0.0f32; 32];
        assert!(
            kernel.dequant_block(&[0u8; 4], &mut output).is_err(),
            "block too small should error"
        );
    }

    #[test]
    fn test_dequant_output_too_small_errors() {
        let kernel = Q4_0Ref;
        let block = make_q4_0_block(1.0, &[0x88; 16]);
        let mut output = vec![0.0f32; 1]; // need 32
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "output too small should error"
        );
    }
}
