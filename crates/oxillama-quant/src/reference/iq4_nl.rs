//! IQ4_NL reference (naive) implementation.
//!
//! IQ4_NL block format (18 bytes per 32 weights):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..18]:  16 nibble-bytes encoding 32 four-bit weights.
//!   Low nibble  = `weight[2i]`,
//!   High nibble = `weight[2i+1]`.
//!
//! Dequantisation: `w = d * KVALUES_IQ4NL[nibble]`
//!
//! This is a non-linear quantisation: the 4-bit index is not subtracted
//! from an offset but used directly as a lookup into [`KVALUES_IQ4NL`].

use super::iq_shared::KVALUES_IQ4NL;
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ4_NL: 32 weights per block.
const IQ4_NL_BLOCK_SIZE: usize = 32;
/// Bytes per IQ4_NL block: 2 (FP16 scale) + 16 (nibble data).
const IQ4_NL_BLOCK_BYTES: usize = 18;

/// Reference (naive scalar) IQ4_NL kernel.
pub struct Iq4NlRef;

impl QuantKernel for Iq4NlRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ4_NL_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ4_NL_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ4_NL_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ4_NL_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        for i in 0..(IQ4_NL_BLOCK_SIZE / 2) {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            output[i * 2] = d * KVALUES_IQ4NL[lo] as f32;
            output[i * 2 + 1] = d * KVALUES_IQ4NL[hi] as f32;
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

        let blocks_per_row = n_cols.div_ceil(IQ4_NL_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ4_NL_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ4_NL_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ4_NL_BLOCK_BYTES];
                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let input_offset = blk * IQ4_NL_BLOCK_SIZE;

                for i in 0..(IQ4_NL_BLOCK_SIZE / 2) {
                    let byte = block[2 + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;
                    let idx = input_offset + i * 2;
                    if idx + 1 < n_cols {
                        sum += KVALUES_IQ4NL[lo] as f32 * d * input[idx];
                        sum += KVALUES_IQ4NL[hi] as f32 * d * input[idx + 1];
                    } else if idx < n_cols {
                        sum += KVALUES_IQ4NL[lo] as f32 * d * input[idx];
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
        IQ4_NL_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ4_NL_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ4_NL"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Helper: build a well-formed IQ4_NL block from a scale and 16 nibble bytes.
    fn make_iq4_nl_block(scale: f32, nibble_bytes: [u8; 16]) -> [u8; IQ4_NL_BLOCK_BYTES] {
        let mut block = [0u8; IQ4_NL_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block[2..18].copy_from_slice(&nibble_bytes);
        block
    }

    #[test]
    fn test_dequant_block_known_values() {
        // d = 1.0, first nibble byte = 0x50
        //   lo = 0 → KVALUES[0] = -127
        //   hi = 5 → KVALUES[5] = -35
        let mut nibbles = [0x00u8; 16];
        nibbles[0] = 0x50;
        let block = make_iq4_nl_block(1.0, nibbles);
        let mut out = [0.0f32; 32];
        Iq4NlRef.dequant_block(&block, &mut out).unwrap();

        assert!(
            (out[0] - (-127.0f32)).abs() < 0.1,
            "out[0] = {}, expected -127",
            out[0]
        );
        assert!(
            (out[1] - (-35.0f32)).abs() < 0.1,
            "out[1] = {}, expected -35",
            out[1]
        );
    }

    #[test]
    fn test_dequant_block_all_zero_index() {
        // d = 2.0, all nibbles = 0 → all weights = 2.0 * KVALUES[0] = -254.0
        let block = make_iq4_nl_block(2.0, [0x00u8; 16]);
        let mut out = [0.0f32; 32];
        Iq4NlRef.dequant_block(&block, &mut out).unwrap();
        for &v in &out {
            assert!((v - (-254.0f32)).abs() < 0.5, "expected -254, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_all_max_index() {
        // all nibbles = 0xF → KVALUES[15] = 113
        let block = make_iq4_nl_block(1.0, [0xFFu8; 16]);
        let mut out = [0.0f32; 32];
        Iq4NlRef.dequant_block(&block, &mut out).unwrap();
        for &v in &out {
            assert!((v - 113.0f32).abs() < 0.1, "expected 113, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 10];
        let mut out = [0.0f32; 32];
        let result = Iq4NlRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_iq4_nl_block(1.0, [0x00u8; 16]);
        let mut out = [0.0f32; 10];
        let result = Iq4NlRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_gemv_sum_equals_dequant_dot_ones() {
        // 1-row tensor with one block; gemv with input=ones must equal
        // the sum of all dequantised weights.
        // Use d=1.0, all nibbles=0x0A (lo=0→-127, hi=10→25 alternating)
        // but simpler: use all nibbles = 0x88 → both nibbles = 8 → KVALUES[8]=1
        // so each weight = 1.0 * 1 = 1.0, sum = 32.0
        let block = make_iq4_nl_block(1.0, [0x88u8; 16]);
        let tensor = QuantTensor::new(
            block.to_vec(),
            vec![1, 32],
            oxillama_gguf::GgufTensorType::Iq4Nl,
        );

        // Compute expected sum from dequant
        let mut dequant = [0.0f32; 32];
        Iq4NlRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let input = vec![1.0f32; 32];
        let mut out = [0.0f32; 1];
        Iq4NlRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-4,
            "gemv={}, expected sum={}",
            out[0],
            expected
        );
    }

    #[test]
    fn test_gemv_two_rows() {
        // 2-row × 32-col tensor; each row has one block.
        // Row 0: all nibbles 0x88 → all weights = KVALUES[8] = 1 → sum with ones = 32
        // Row 1: all nibbles 0xFF → all weights = KVALUES[15] = 113 → sum with ones = 3616
        let block0 = make_iq4_nl_block(1.0, [0x88u8; 16]);
        let block1 = make_iq4_nl_block(1.0, [0xFFu8; 16]);
        let mut data = Vec::with_capacity(IQ4_NL_BLOCK_BYTES * 2);
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);

        let tensor = QuantTensor::new(data, vec![2, 32], oxillama_gguf::GgufTensorType::Iq4Nl);

        let input = vec![1.0f32; 32];
        let mut out = [0.0f32; 2];
        Iq4NlRef.gemv(&tensor, &input, &mut out).unwrap();

        // KVALUES[8] = 1 → 32 * 1 * 1.0 = 32
        assert!((out[0] - 32.0f32).abs() < 0.1, "row0={}", out[0]);
        // KVALUES[15] = 113 → 32 * 113 * 1.0 = 3616
        assert!((out[1] - 3616.0f32).abs() < 0.1, "row1={}", out[1]);
    }

    #[test]
    fn test_block_size_and_bytes() {
        assert_eq!(Iq4NlRef.block_size(), 32);
        assert_eq!(Iq4NlRef.block_bytes(), 18);
        assert_eq!(Iq4NlRef.name(), "IQ4_NL");
    }
}
