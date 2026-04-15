//! Q4_K reference (naive) implementation.
//!
//! Q4_K block format (144 bytes per 256 weights):
//! - 2 bytes: FP16 super-block scale (d)
//! - 2 bytes: FP16 super-block minimum (dmin)
//! - 12 bytes: 8 sub-block scales + 8 sub-block mins, 6-bit each, packed
//! - 128 bytes: 256 × 4-bit unsigned nibbles packed (2 per byte)
//!
//! 8 sub-blocks of 32 weights each.
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 4-bit (0..15).
//!
//! Effective: 4.5 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

/// Reference (naive scalar) Q4_K kernel.
pub struct Q4KRef;

/// Decode the 6-bit packed scales and mins for Q4_K.
///
/// Returns (scales[8], mins[8]) where each is a 6-bit value.
fn decode_scales_mins(scales_raw: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // Sub-blocks 0..3: straightforward 6-bit extraction
    for j in 0..4 {
        sc[j] = scales_raw[j] & 0x3F;
        mn[j] = scales_raw[j + 4] & 0x3F;
    }

    // Sub-blocks 4..7: assembled from high bits of bytes 0..3/4..7 and bytes 8..11
    for j in 4..8 {
        let lo_sc = scales_raw[j + 4] & 0x0F;
        let hi_sc = (scales_raw[j - 4] >> 6) & 0x03;
        sc[j] = lo_sc | (hi_sc << 4);

        let lo_mn = (scales_raw[j + 4] >> 4) & 0x0F;
        let hi_mn = (scales_raw[j] >> 6) & 0x03;
        mn[j] = lo_mn | (hi_mn << 4);
    }

    (sc, mn)
}

impl QuantKernel for Q4KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q4_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q4_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales_raw = &block[4..16];
        let qs = &block[16..144]; // 128 bytes of nibble data

        let (sc, mn) = decode_scales_mins(scales_raw);

        // Process 4 groups of 64 weights (2 sub-blocks of 32 per group)
        let mut is = 0usize; // sub-block index
        let mut qs_offset = 0usize;
        let mut out_offset = 0usize;

        for _group in 0..4 {
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            // Low nibbles → first 32 weights (sub-block `is`)
            for l in 0..32 {
                let q = (qs[qs_offset + l] & 0x0F) as f32;
                output[out_offset + l] = d1 * q - m1;
            }

            // High nibbles → next 32 weights (sub-block `is+1`)
            for l in 0..32 {
                let q = ((qs[qs_offset + l] >> 4) & 0x0F) as f32;
                output[out_offset + 32 + l] = d2 * q - m2;
            }

            is += 2;
            qs_offset += 32;
            out_offset += 64;
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

        let blocks_per_row = n_cols.div_ceil(Q4_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q4_K_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + Q4_K_BLOCK_BYTES];

                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales_raw = &block[4..16];
                let qs = &block[16..144];
                let input_offset = blk * Q4_K_BLOCK_SIZE;

                let (sc, mn) = decode_scales_mins(scales_raw);

                let mut is = 0usize;
                let mut qs_off = 0usize;
                let mut w_off = input_offset;

                for _group in 0..4 {
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * mn[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * mn[is + 1] as f32;

                    for l in 0..32 {
                        let idx = w_off + l;
                        if idx < n_cols {
                            let q = (qs[qs_off + l] & 0x0F) as f32;
                            sum += (d1 * q - m1) * input[idx];
                        }
                    }
                    for l in 0..32 {
                        let idx = w_off + 32 + l;
                        if idx < n_cols {
                            let q = ((qs[qs_off + l] >> 4) & 0x0F) as f32;
                            sum += (d2 * q - m2) * input[idx];
                        }
                    }

                    is += 2;
                    qs_off += 32;
                    w_off += 64;
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
        Q4_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q4_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_K"
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_K_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        // d=0, dmin=0 → all weights should be 0
        let block = make_q4_k_block(0.0, 0.0, &[0; 12], &[0; 128]);
        let kernel = Q4KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_uniform() {
        // d=1.0, dmin=0.0, all scales=1 (sub-blocks 0..3), all nibbles=8
        // Weight = 1.0 * 1 * 8 - 0 = 8.0
        let mut scales = [0u8; 12];
        // Set sub-block scales 0..3 to 1 (lower 6 bits of bytes 0..3)
        scales[0] = 1;
        scales[1] = 1;
        scales[2] = 1;
        scales[3] = 1;
        // Sub-block scales 4..7: stored in bytes 8..11 lower 4 bits, with high bits from bytes 0..3 upper 2 bits
        scales[8] = 1;
        scales[9] = 1;
        scales[10] = 1;
        scales[11] = 1;

        // All nibbles = 8: byte = 0x88 (lo=8, hi=8)
        let qs = [0x88u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let kernel = Q4KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        // All weights should be 1.0 * 1 * 8 = 8.0
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 8.0).abs() < 0.01, "weight[{i}] = {v}, expected 8.0");
        }
    }

    #[test]
    fn test_gemv_q4_k() {
        // Create a simple 1-row, 256-col Q4_K tensor
        // d=1.0, dmin=0, all scales=1, all nibbles=1
        // Weight = 1.0 * 1 * 1 - 0 = 1.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1); // sub-blocks 0..3 scale=1
        scales[8..12].fill(1); // sub-blocks 4..7 scale=1

        // All nibbles = 1: lo=1, hi=1 → byte = 0x11
        let qs = [0x11u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q4K);

        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        let kernel = Q4KRef;
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // All 256 weights = 1.0, all inputs = 1.0 → dot = 256.0
        assert!(
            (output[0] - 256.0).abs() < 1.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }
}
