//! Q1_0_G128 reference (naive) implementation — from OxiBonsai.
//!
//! Q1_0_G128 block format (18 bytes per 128 weights):
//! - 2 bytes: FP16 scale (d)
//! - 16 bytes: 128 sign bits (bit=1 → +d, bit=0 → -d)
//!
//! This is PrismML's 1-bit Bonsai quantization format where every weight
//! is represented as a single sign bit with a shared group scale factor.
//!
//! Effective: 1.125 bits/weight (1 sign + 16 bits / 128 amortized).

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for Q1_0_G128: 128 weights per block.
const Q1_0_G128_BLOCK_SIZE: usize = 128;
/// Bytes per Q1_0_G128 block: 2 (FP16 scale) + 16 (128 bits).
const Q1_0_G128_BLOCK_BYTES: usize = 18;

/// Reference (naive scalar) Q1_0_G128 kernel.
///
/// This is the 1-bit quantization kernel from PrismML's Bonsai format.
/// Each weight is either +d or -d based on its sign bit.
pub struct Q1_0G128Ref;

impl QuantKernel for Q1_0G128Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q1_0_G128_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q1_0_G128_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q1_0_G128_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q1_0_G128_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        // 128 sign bits packed in 16 bytes (little-endian bit order)
        for byte_idx in 0..16 {
            let byte = block[2 + byte_idx];
            for bit_idx in 0..8 {
                let weight_idx = byte_idx * 8 + bit_idx;
                let bit = (byte >> bit_idx) & 1;
                // bit=1 → +d, bit=0 → -d
                output[weight_idx] = if bit == 1 { d } else { -d };
            }
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

        let blocks_per_row = n_cols.div_ceil(Q1_0_G128_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q1_0_G128_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q1_0_G128_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + Q1_0_G128_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let input_offset = blk * Q1_0_G128_BLOCK_SIZE;

                // Optimized 1-bit GEMV: accumulate sum_positive - sum_negative
                // For bit=1: weight = +d → contributes d * input[i]
                // For bit=0: weight = -d → contributes -d * input[i]
                // Total = d * (sum_positive - sum_negative)
                let mut diff = 0.0f32;

                for byte_idx in 0..16 {
                    let byte = block[2 + byte_idx];
                    for bit_idx in 0..8 {
                        let weight_idx = input_offset + byte_idx * 8 + bit_idx;
                        if weight_idx < n_cols {
                            let bit = (byte >> bit_idx) & 1;
                            if bit == 1 {
                                diff += input[weight_idx];
                            } else {
                                diff -= input[weight_idx];
                            }
                        }
                    }
                }

                sum += d * diff;
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
        Q1_0_G128_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q1_0_G128_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q1_0_G128"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q1_block(scale: f32, bits: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q1_0_G128_BLOCK_BYTES);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(bits);
        block
    }

    #[test]
    fn test_dequant_all_positive() {
        // All bits = 1 → all weights = +d
        let block = make_q1_block(2.0, &[0xFF; 16]);
        let kernel = Q1_0G128Ref;
        let mut output = vec![0.0f32; 128];
        kernel.dequant_block(&block, &mut output).unwrap();

        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected +2.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_all_negative() {
        // All bits = 0 → all weights = -d
        let block = make_q1_block(3.0, &[0x00; 16]);
        let kernel = Q1_0G128Ref;
        let mut output = vec![0.0f32; 128];
        kernel.dequant_block(&block, &mut output).unwrap();

        for &v in &output {
            assert!((v - (-3.0)).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_mixed() {
        // First byte: 0b10101010 → bits 1,3,5,7 are set
        let mut bits = [0x00u8; 16];
        bits[0] = 0xAA; // 10101010
        let block = make_q1_block(1.0, &bits);
        let kernel = Q1_0G128Ref;
        let mut output = vec![0.0f32; 128];
        kernel.dequant_block(&block, &mut output).unwrap();

        // bit 0 = 0 → -1.0
        assert!((output[0] - (-1.0)).abs() < 0.01);
        // bit 1 = 1 → +1.0
        assert!((output[1] - 1.0).abs() < 0.01);
        // bit 2 = 0 → -1.0
        assert!((output[2] - (-1.0)).abs() < 0.01);
        // bit 3 = 1 → +1.0
        assert!((output[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gemv_1bit() {
        let kernel = Q1_0G128Ref;
        // All bits = 1 → all weights = +d = +1.0
        let block = make_q1_block(1.0, &[0xFF; 16]);
        let tensor = QuantTensor::new(block, vec![1, 128], oxillama_gguf::GgufTensorType::Q1_0G128);

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // All weights = +1, all inputs = 1 → dot product = 128
        assert!(
            (output[0] - 128.0).abs() < 0.5,
            "expected 128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gemv_1bit_alternating() {
        let kernel = Q1_0G128Ref;
        // Alternating bits: half +d, half -d → cancel with uniform input
        let block = make_q1_block(1.0, &[0xAA; 16]); // 10101010 per byte
        let tensor = QuantTensor::new(block, vec![1, 128], oxillama_gguf::GgufTensorType::Q1_0G128);

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // 64 positive + 64 negative with uniform input → should be 0
        assert!(output[0].abs() < 0.5, "expected ~0.0, got {}", output[0]);
    }
}
