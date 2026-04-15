//! TQ2_0 reference (naive) implementation.
//!
//! TQ2_0 block format (66 bytes per 256 weights):
//! - 64 bytes: qs — 256 × 2-bit ternary codes packed (4 per byte)
//! - 2 bytes: FP16 scale (d)
//!
//! Each 2-bit code encodes a ternary value: 0→-1, 1→0, 2→+1.
//! Weight formula: `w = d * (q_2bit - 1)`

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for TQ2_0: 256 weights per block.
const TQ2_0_BLOCK_SIZE: usize = 256;
/// Bytes per TQ2_0 block: 64 (qs) + 2 (d).
const TQ2_0_BLOCK_BYTES: usize = 66;

/// Reference (naive scalar) TQ2_0 kernel.
pub struct Tq2_0Ref;

impl QuantKernel for Tq2_0Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < TQ2_0_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: TQ2_0_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < TQ2_0_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: TQ2_0_BLOCK_SIZE,
                available: output.len(),
            });
        }

        // qs is the first 64 bytes, d is bytes[64..66]
        let qs = &block[0..64];
        let d = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));

        // Each byte holds 4 × 2-bit codes
        for (i, &byte) in qs.iter().enumerate() {
            let v0 = (byte & 3) as i32 - 1;
            let v1 = ((byte >> 2) & 3) as i32 - 1;
            let v2 = ((byte >> 4) & 3) as i32 - 1;
            let v3 = ((byte >> 6) & 3) as i32 - 1;
            output[i * 4] = d * v0 as f32;
            output[i * 4 + 1] = d * v1 as f32;
            output[i * 4 + 2] = d * v2 as f32;
            output[i * 4 + 3] = d * v3 as f32;
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

        let blocks_per_row = n_cols.div_ceil(TQ2_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * TQ2_0_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * TQ2_0_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let qs = &data[bo..bo + 64];
                let d = f16_to_f32(u16::from_le_bytes([data[bo + 64], data[bo + 65]]));
                let inp = &input[blk * TQ2_0_BLOCK_SIZE..];

                // Inline dequant + dot product
                for (i, &byte) in qs.iter().enumerate() {
                    let v0 = (byte & 3) as i32 - 1;
                    let v1 = ((byte >> 2) & 3) as i32 - 1;
                    let v2 = ((byte >> 4) & 3) as i32 - 1;
                    let v3 = ((byte >> 6) & 3) as i32 - 1;
                    let base = i * 4;
                    if base + 3 < n_cols.saturating_sub(blk * TQ2_0_BLOCK_SIZE) {
                        sum += d * v0 as f32 * inp[base];
                        sum += d * v1 as f32 * inp[base + 1];
                        sum += d * v2 as f32 * inp[base + 2];
                        sum += d * v3 as f32 * inp[base + 3];
                    } else {
                        let remaining = n_cols.saturating_sub(blk * TQ2_0_BLOCK_SIZE);
                        let vals = [v0, v1, v2, v3];
                        for (j, &v) in vals.iter().enumerate() {
                            if base + j < remaining {
                                sum += d * v as f32 * inp[base + j];
                            }
                        }
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
        TQ2_0_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        TQ2_0_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "TQ2_0"
    }
}

/// Convert an IEEE 754 FP16 half-precision value to FP32.
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Build a TQ2_0 block from 64 qs bytes and a scale value.
#[cfg(test)]
fn make_tq2_0_block(scale: f32, qs: &[u8; 64]) -> Vec<u8> {
    let mut block = Vec::with_capacity(TQ2_0_BLOCK_BYTES);
    block.extend_from_slice(qs);
    let d_bits = half::f16::from_f32(scale).to_bits();
    block.extend_from_slice(&d_bits.to_le_bytes());
    block
}

/// Pack four 2-bit ternary codes into a single byte.
/// v0 is bits [1:0], v1 is bits [3:2], v2 is bits [5:4], v3 is bits [7:6].
#[cfg(test)]
fn pack_2bit(v0: u8, v1: u8, v2: u8, v3: u8) -> u8 {
    (v0 & 3) | ((v1 & 3) << 2) | ((v2 & 3) << 4) | ((v3 & 3) << 6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_zeros() {
        // d = 0.0 → all weights should be zero regardless of qs values
        let qs = [0xFFu8; 64]; // arbitrary non-zero qs
        let block = make_tq2_0_block(0.0, &qs);
        let kernel = Tq2_0Ref;
        let mut output = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant_zeros");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-7, "output[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_dequant_all_positive() {
        // All 2-bit codes = 2 → ternary +1 → all weights = d * 1.0 = 1.0
        // pack_2bit(2,2,2,2) = 2 | (2<<2) | (2<<4) | (2<<6) = 2+8+32+128 = 0xAA
        let qs = [pack_2bit(2, 2, 2, 2); 64];
        assert_eq!(qs[0], 0xAA);
        let block = make_tq2_0_block(1.0, &qs);
        let kernel = Tq2_0Ref;
        let mut output = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant_all_positive");
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "output[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn test_dequant_all_negative() {
        // All 2-bit codes = 0 → ternary -1 → all weights = d * (-1.0) = -1.0
        // pack_2bit(0,0,0,0) = 0x00
        let qs = [0x00u8; 64];
        let block = make_tq2_0_block(1.0, &qs);
        let kernel = Tq2_0Ref;
        let mut output = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant_all_negative");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-3,
                "output[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_all_zero() {
        // All 2-bit codes = 1 → ternary 0 → all weights = d * 0 = 0.0
        // pack_2bit(1,1,1,1) = 1 | (1<<2) | (1<<4) | (1<<6) = 1+4+16+64 = 0x55
        let qs = [0x55u8; 64];
        let block = make_tq2_0_block(1.0, &qs);
        let kernel = Tq2_0Ref;
        let mut output = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant_all_zero_ternary");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-5, "output[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_dequant_mixed() {
        // First byte: pack_2bit(0,1,2,0) = 0 | (1<<2) | (2<<4) | (0<<6) = 0+4+32+0 = 36 = 0x24
        // → ternary values: -1, 0, +1, -1
        // d = 2.0 → weights: -2.0, 0.0, 2.0, -2.0
        let mut qs = [0x55u8; 64]; // default all ternary-zero
        qs[0] = pack_2bit(0, 1, 2, 0);
        qs[1] = pack_2bit(2, 2, 0, 1);
        let block = make_tq2_0_block(2.0, &qs);
        let kernel = Tq2_0Ref;
        let mut output = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant_mixed");

        // Byte 0: codes 0,1,2,0 → ternary -1,0,+1,-1 → -2.0, 0.0, 2.0, -2.0
        assert!((output[0] - (-2.0)).abs() < 1e-3, "got {}", output[0]);
        assert!((output[1]).abs() < 1e-3, "got {}", output[1]);
        assert!((output[2] - 2.0).abs() < 1e-3, "got {}", output[2]);
        assert!((output[3] - (-2.0)).abs() < 1e-3, "got {}", output[3]);

        // Byte 1: codes 2,2,0,1 → ternary +1,+1,-1,0 → 2.0, 2.0, -2.0, 0.0
        assert!((output[4] - 2.0).abs() < 1e-3, "got {}", output[4]);
        assert!((output[5] - 2.0).abs() < 1e-3, "got {}", output[5]);
        assert!((output[6] - (-2.0)).abs() < 1e-3, "got {}", output[6]);
        assert!((output[7]).abs() < 1e-3, "got {}", output[7]);

        // Remaining bytes are all 0x55 (code 1 → ternary 0 → 0.0)
        for (i, val) in output.iter().enumerate().take(256).skip(8) {
            assert!(val.abs() < 1e-5, "output[{i}] = {}", val);
        }
    }

    #[test]
    fn test_gemv_tq2_0() {
        let kernel = Tq2_0Ref;

        // Build a 1-row, 256-col matrix with known ternary values
        // Use a pattern: first 4 weights = +1, next 4 = -1, rest = 0
        let mut qs = [0x55u8; 64]; // all ternary 0
        qs[0] = pack_2bit(2, 2, 2, 2); // indices 0..4 → ternary +1
        qs[1] = pack_2bit(0, 0, 0, 0); // indices 4..8 → ternary -1
        let scale = 0.5f32;
        let block = make_tq2_0_block(scale, &qs);
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Tq2_0);

        // Input: all 1.0 for the first 8, rest 1.0 too (but ternary 0 → no contribution)
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: gemv_tq2_0");

        // Expected: 4 * (0.5 * 1) * 1.0  +  4 * (0.5 * (-1)) * 1.0 = 2.0 - 2.0 = 0.0
        assert!(
            output[0].abs() < 1e-3,
            "output[0] = {}, expected ~0.0",
            output[0]
        );

        // Now try with asymmetric input to get a non-zero result
        let mut input2 = vec![0.0f32; 256];
        // First 4 weights = +0.5 (+1 ternary * 0.5 scale), next 4 = -0.5 (-1 ternary * 0.5)
        input2[0] = 2.0;
        input2[1] = 2.0;
        input2[2] = 2.0;
        input2[3] = 2.0;
        // Indices 4..8 have ternary -1, input = 0 → contribute 0
        let mut output2 = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input2, &mut output2)
            .expect("test: gemv_tq2_0 asymmetric");

        // Expected: 4 * 0.5 * 1.0 * 2.0 = 4.0 (ternary +1 * scale 0.5 * input 2.0 × 4)
        assert!(
            (output2[0] - 4.0).abs() < 1e-2,
            "output2[0] = {}, expected ~4.0",
            output2[0]
        );
    }

    #[test]
    fn test_gemv_tq2_0_reference_comparison() {
        // Verify GEMV produces the same result as manual dequant + dot product
        let kernel = Tq2_0Ref;

        let mut qs = [0u8; 64];
        // Create a varied pattern
        for (i, byte) in qs.iter_mut().enumerate() {
            let v0 = (i % 3) as u8;
            let v1 = ((i + 1) % 3) as u8;
            let v2 = ((i + 2) % 3) as u8;
            let v3 = (i % 2) as u8;
            *byte = pack_2bit(v0, v1, v2, v3);
        }
        let scale = 1.5f32;
        let block = make_tq2_0_block(scale, &qs);

        // Dequantize to get reference weights
        let mut weights = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut weights)
            .expect("test: reference dequant");

        // Create input
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        // Reference dot product
        let ref_dot: f32 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV result
        let tensor = QuantTensor::new(
            block.clone(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Tq2_0,
        );
        let mut output = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: gemv reference");

        assert!(
            (output[0] - ref_dot).abs() < 1e-2,
            "gemv={}, ref_dot={}, diff={}",
            output[0],
            ref_dot,
            (output[0] - ref_dot).abs()
        );
    }

    #[test]
    fn test_dequant_buffer_too_small() {
        let kernel = Tq2_0Ref;
        let block = vec![0u8; 40]; // too small (need 66)
        let mut output = vec![0.0f32; 256];
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "should error on small block"
        );
    }

    #[test]
    fn test_dequant_output_too_small() {
        let kernel = Tq2_0Ref;
        let block = vec![0u8; 66];
        let mut output = vec![0.0f32; 100]; // need 256
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "should error on small output"
        );
    }

    #[test]
    fn test_block_metadata() {
        let kernel = Tq2_0Ref;
        assert_eq!(kernel.block_size(), 256);
        assert_eq!(kernel.block_bytes(), 66);
        assert_eq!(kernel.name(), "TQ2_0");
    }
}
