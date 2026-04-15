//! Q6_K reference (naive) implementation.
//!
//! Q6_K block format (210 bytes per 256 weights):
//! - 128 bytes: ql — lower 4 bits of 6-bit quants (2 per byte)
//! - 64 bytes: qh — upper 2 bits of 6-bit quants (4 per byte)
//! - 16 bytes: scales — 16 × int8 signed scales (one per 16-weight sub-block)
//! - 2 bytes: FP16 super-block scale (d)
//!
//! Q6_K is a symmetric ("type-0") format: no minimum offset.
//! Weight formula: `w = d * scale_i * (q_6bit - 32)`
//!
//! Effective: 6.5625 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

/// Reference (naive scalar) Q6_K kernel.
pub struct Q6KRef;

impl QuantKernel for Q6KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q6_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q6_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q6_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q6_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        // Process in 2 groups of 128 weights
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let out_off = group * 128;

            for l in 0..32 {
                let is = l / 16; // sub-block index within group: 0 or 1

                // Assemble 6-bit quants from ql (4 low bits) and qh (2 high bits)
                let q1 = ((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let s0 = scales[sc_off + is] as i8 as f32;
                let s1 = scales[sc_off + is + 2] as i8 as f32;
                let s2 = scales[sc_off + is + 4] as i8 as f32;
                let s3 = scales[sc_off + is + 6] as i8 as f32;

                output[out_off + l] = d * s0 * q1 as f32;
                output[out_off + l + 32] = d * s1 * q2 as f32;
                output[out_off + l + 64] = d * s2 * q3 as f32;
                output[out_off + l + 96] = d * s3 * q4 as f32;
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

        let blocks_per_row = n_cols.div_ceil(Q6_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q6_K_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let ql = &data[bo..bo + 128];
                let qh = &data[bo + 128..bo + 192];
                let scales = &data[bo + 192..bo + 208];
                let d = f16_to_f32(u16::from_le_bytes([data[bo + 208], data[bo + 209]]));
                let inp = &input[blk * Q6_K_BLOCK_SIZE..];
                // Number of valid columns in this (possibly partial) block.
                let cols_in_block = (n_cols - blk * Q6_K_BLOCK_SIZE).min(Q6_K_BLOCK_SIZE);

                // Inline dot product: extract 6-bit quants on-the-fly
                for group in 0..2 {
                    let ql_off = group * 64;
                    let qh_off = group * 32;
                    let sc_off = group * 8;
                    let in_off = group * 128;

                    for l in 0..32 {
                        let is = l / 16;
                        let q1 =
                            ((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                            as i32
                            - 32;
                        let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4))
                            as i32
                            - 32;
                        let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                            as i32
                            - 32;

                        let s0 = d * scales[sc_off + is] as i8 as f32;
                        let s1 = d * scales[sc_off + is + 2] as i8 as f32;
                        let s2 = d * scales[sc_off + is + 4] as i8 as f32;
                        let s3 = d * scales[sc_off + is + 6] as i8 as f32;

                        let c0 = in_off + l;
                        let c1 = in_off + l + 32;
                        let c2 = in_off + l + 64;
                        let c3 = in_off + l + 96;

                        if c0 < cols_in_block {
                            sum += s0 * q1 as f32 * inp[c0];
                        }
                        if c1 < cols_in_block {
                            sum += s1 * q2 as f32 * inp[c1];
                        }
                        if c2 < cols_in_block {
                            sum += s2 * q3 as f32 * inp[c2];
                        }
                        if c3 < cols_in_block {
                            sum += s3 * q4 as f32 * inp[c3];
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
        Q6_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q6_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q6_K"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_zero_scale() {
        let block = vec![0u8; Q6_K_BLOCK_BYTES];
        let kernel = Q6KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        // d=0 → all weights = 0 (q-32 doesn't matter since d=0)
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_simple() {
        // Construct a block where all 6-bit quants = 32 (so q-32 = 0)
        // ql: lower 4 bits of 32 = 0, qh: upper 2 bits of 32 = 2 (binary: 10_0000)
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];

        // ql = all 0x00 (lower 4 bits of quant value 32 is 0)
        // qh: each byte stores 4 values' upper 2 bits
        // value 32 = 0b100000, upper 2 bits = 0b10 = 2
        // qh[l] should have bits: q1_hi=2, q2_hi=2, q3_hi=2, q4_hi=2
        // packed: (2<<0)|(2<<2)|(2<<4)|(2<<6) = 2 + 8 + 32 + 128 = 0xAA
        for i in 0..64 {
            block[128 + i] = 0xAA;
        }

        // scales = 1 for all sub-blocks
        for i in 0..16 {
            block[192 + i] = 1;
        }

        // d = 1.0
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[208] = (d_bits & 0xFF) as u8;
        block[209] = ((d_bits >> 8) & 0xFF) as u8;

        let kernel = Q6KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        // All quants = 32, so q-32 = 0, weight = d * scale * 0 = 0
        for (i, &v) in output.iter().enumerate() {
            assert!((v).abs() < 0.01, "weight[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_gemv_q6_k() {
        // Build a 1x256 matrix: random-ish block data
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        // Set some ql values
        for (i, b) in block.iter_mut().enumerate().take(128) {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        // Set some qh values
        for i in 0..64 {
            block[128 + i] = ((i * 13 + 5) & 0xFF) as u8;
        }
        // Set scales
        for i in 0..16 {
            block[192 + i] = (i as i8 * 3 - 8) as u8;
        }
        // d = 0.5
        let d_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        block[208] = d_bits[0];
        block[209] = d_bits[1];

        let kernel = Q6KRef;

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        // Input vector
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // Reference: manual dot product
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q6K);
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
