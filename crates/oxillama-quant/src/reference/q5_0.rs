//! Q5_0 reference (naive) implementation.
//!
//! Q5_0 block format (22 bytes per 32 weights):
//! - 2 bytes: FP16 block scale (d)
//! - 4 bytes: qh — 32 high bits (bit 4 of each 5-bit quant)
//! - 16 bytes: qs — 32 × lower 4 bits packed (2 per byte)
//!
//! Q5_0 is a symmetric ("type-0") format: no minimum offset.
//! Weight formula: `w = d * (q_5bit - 16)` where q is 5-bit unsigned (0..31).
//!
//! Effective: 5.5 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q5_0_BLOCK_SIZE: usize = 32;
const Q5_0_BLOCK_BYTES: usize = 22;

/// Reference (naive scalar) Q5_0 kernel.
pub struct Q5_0Ref;

impl QuantKernel for Q5_0Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q5_0_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_0_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q5_0_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_0_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];

        for i in 0..16 {
            let lo_nibble = qs[i] & 0x0F;
            let hi_nibble = (qs[i] >> 4) & 0x0F;

            let hi_bit_0 = ((qh >> i) & 1) as u8;
            let hi_bit_1 = ((qh >> (i + 16)) & 1) as u8;

            let q0 = (lo_nibble | (hi_bit_0 << 4)) as i32 - 16;
            let q1 = (hi_nibble | (hi_bit_1 << 4)) as i32 - 16;

            output[i] = d * q0 as f32;
            output[i + 16] = d * q1 as f32;
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

        let blocks_per_row = n_cols.div_ceil(Q5_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q5_0_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([data[bo], data[bo + 1]]));
                let qh =
                    u32::from_le_bytes([data[bo + 2], data[bo + 3], data[bo + 4], data[bo + 5]]);
                let qs = &data[bo + 6..bo + 22];
                let inp = &input[blk * Q5_0_BLOCK_SIZE..];

                // Inline dot product: extract 5-bit quants on-the-fly
                for i in 0..16 {
                    let lo_nibble = qs[i] & 0x0F;
                    let hi_nibble = qs[i] >> 4;
                    let hi_bit_0 = ((qh >> i) & 1) as u8;
                    let hi_bit_1 = ((qh >> (i + 16)) & 1) as u8;

                    let q0 = (lo_nibble | (hi_bit_0 << 4)) as i32 - 16;
                    let q1 = (hi_nibble | (hi_bit_1 << 4)) as i32 - 16;

                    sum += d * q0 as f32 * inp[i];
                    sum += d * q1 as f32 * inp[i + 16];
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
        Q5_0_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q5_0_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q5_0"
    }

    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> crate::error::QuantResult<()> {
        use crate::error::QuantError;

        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }
        if Q5_0_BLOCK_SIZE == 0 {
            return Err(QuantError::KernelError {
                message: "block_size is 0".to_string(),
            });
        }
        let blocks_per_row = n_cols.div_ceil(Q5_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;
        let q8_block_bytes: usize = 34; // 2 (f16 d) + 32 (i8 values)

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < blocks_per_row * q8_block_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: blocks_per_row * q8_block_bytes,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q5_0_BLOCK_BYTES;
                let block = &weights[bo..bo + Q5_0_BLOCK_BYTES];
                let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                let qs = &block[6..22];

                // Q8_0 activation block.
                let ab = blk * q8_block_bytes;
                let a_block = &acts_q8[ab..ab + q8_block_bytes];
                let d_a = half::f16::from_le_bytes([a_block[0], a_block[1]]).to_f32();
                let q8_vals = &a_block[2..];

                let w_off = blk * Q5_0_BLOCK_SIZE;
                let valid = (n_cols - w_off).min(Q5_0_BLOCK_SIZE);

                let mut dot = 0.0f32;
                for i in 0..16 {
                    if i >= valid && i + 16 >= valid {
                        break;
                    }
                    let lo_nibble = qs[i] & 0x0F;
                    let hi_nibble = (qs[i] >> 4) & 0x0F;
                    let hi_bit_0 = ((qh >> i) & 1) as u8;
                    let hi_bit_1 = ((qh >> (i + 16)) & 1) as u8;
                    let q0 = (lo_nibble | (hi_bit_0 << 4)) as i32 - 16;
                    let q1 = (hi_nibble | (hi_bit_1 << 4)) as i32 - 16;
                    if i < valid {
                        let a0 = (q8_vals[i] as i8) as f32 * d_a;
                        dot += d_w * q0 as f32 * a0;
                    }
                    if i + 16 < valid {
                        let a1 = (q8_vals[i + 16] as i8) as f32 * d_a;
                        dot += d_w * q1 as f32 * a1;
                    }
                }
                sum += dot;
            }

            *out_val += sum;
        }

        Ok(())
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q5_0_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_0_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0 → all weights = 0
        let block = make_q5_0_block(0.0, 0, &[0; 16]);
        let kernel = Q5_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_center() {
        // All 5-bit quants = 16 → q-16 = 0 → weight = 0
        // q=16: lower 4 bits = 0, high bit = 1
        // qh: all 32 bits set → 0xFFFFFFFF
        // qs: all nibbles = 0
        let block = make_q5_0_block(1.0, 0xFFFFFFFF, &[0; 16]);
        let kernel = Q5_0Ref;
        let mut output = vec![99.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v).abs() < 0.01, "weight[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_dequant_max_positive() {
        // All 5-bit quants = 31 → q-16 = 15
        // q=31: lower 4 bits = 15 = 0xF, high bit = 1
        // qh: all bits set → 0xFFFFFFFF
        // qs: all nibbles = 0xF → each byte = 0xFF
        // Weight = 1.0 * 15 = 15.0
        let block = make_q5_0_block(1.0, 0xFFFFFFFF, &[0xFF; 16]);
        let kernel = Q5_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 15.0).abs() < 0.01, "weight[{i}] = {v}, expected 15.0");
        }
    }

    #[test]
    fn test_dequant_max_negative() {
        // All 5-bit quants = 0 → q-16 = -16
        // qh = 0, qs = all 0
        // Weight = 2.0 * (-16) = -32.0
        let block = make_q5_0_block(2.0, 0, &[0; 16]);
        let kernel = Q5_0Ref;
        let mut output = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-32.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -32.0"
            );
        }
    }

    #[test]
    fn test_gemv_q5_0() {
        // Build a block with varied data
        let qh: u32 = 0xA5A5A5A5; // alternating pattern
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 11 + 5) & 0xFF) as u8;
        }
        let block = make_q5_0_block(0.5, qh, &qs);

        let kernel = Q5_0Ref;

        // Dequant reference
        let mut dequant = vec![0.0f32; 32];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 32], oxillama_gguf::GgufTensorType::Q5_0);
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
