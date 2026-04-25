//! Q5_K reference (naive) implementation.
//!
//! Q5_K block format (176 bytes per 256 weights):
//! - 2 bytes: FP16 super-block scale (d)
//! - 2 bytes: FP16 super-block minimum (dmin)
//! - 12 bytes: 8 sub-block scales + 8 sub-block mins, 6-bit each, packed
//! - 32 bytes: qh — high bit of each 5-bit quant (8 per byte)
//! - 128 bytes: qs — lower 4 bits of each 5-bit quant (2 per byte)
//!
//! 8 sub-blocks of 32 weights each.
//! Weight formula: `w = d * scale_i * q_5bit - dmin * min_i`
//!
//! Effective: 5.5 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q5_K_BLOCK_SIZE: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;

/// Reference (naive scalar) Q5_K kernel.
pub struct Q5KRef;

/// Decode the 6-bit packed scales and mins for Q5_K.
/// Same packing as Q4_K: returns (scales[8], mins[8]).
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

impl QuantKernel for Q5KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q5_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q5_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q5_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales_raw = &block[4..16];
        let qh = &block[16..48]; // 32 bytes: high bits
        let qs = &block[48..176]; // 128 bytes: low 4 bits

        let (sc, mn) = decode_scales_mins(scales_raw);

        // Process 4 groups of 64 weights (2 sub-blocks of 32 per group)
        let mut is = 0usize;
        let mut qs_offset = 0usize;
        let mut out_offset = 0usize;

        for group in 0..4 {
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            // Low nibbles → first 32 weights (sub-block `is`)
            for l in 0..32 {
                let qh_bit = (qh[l] >> group) & 1;
                let q = ((qs[qs_offset + l] & 0x0F) | (qh_bit << 4)) as f32;
                output[out_offset + l] = d1 * q - m1;
            }

            // High nibbles → next 32 weights (sub-block `is+1`)
            for l in 0..32 {
                let qh_bit = (qh[l] >> (group + 4)) & 1;
                let q = (((qs[qs_offset + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
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

        let blocks_per_row = n_cols.div_ceil(Q5_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q5_K_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([data[bo], data[bo + 1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));
                let scales_raw = &data[bo + 4..bo + 16];
                let qh = &data[bo + 16..bo + 48];
                let qs = &data[bo + 48..bo + 176];
                let (sc, mn) = decode_scales_mins(scales_raw);
                let inp = &input[blk * Q5_K_BLOCK_SIZE..];
                // Number of valid columns in this (possibly partial) block.
                let cols_in_block = (n_cols - blk * Q5_K_BLOCK_SIZE).min(Q5_K_BLOCK_SIZE);

                // Inline dot product: extract 5-bit quants on-the-fly
                let mut is = 0usize;
                let mut qs_offset = 0usize;
                let mut in_off = 0usize;

                for group in 0..4 {
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * mn[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * mn[is + 1] as f32;

                    for l in 0..32 {
                        let col = in_off + l;
                        if col >= cols_in_block {
                            break;
                        }
                        let qh_bit = (qh[l] >> group) & 1;
                        let q = ((qs[qs_offset + l] & 0x0F) | (qh_bit << 4)) as f32;
                        sum += (d1 * q - m1) * inp[col];
                    }
                    for l in 0..32 {
                        let col = in_off + 32 + l;
                        if col >= cols_in_block {
                            break;
                        }
                        let qh_bit = (qh[l] >> (group + 4)) & 1;
                        let q = (((qs[qs_offset + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
                        sum += (d2 * q - m2) * inp[col];
                    }

                    is += 2;
                    qs_offset += 32;
                    in_off += 64;
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
        Q5_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q5_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q5_K"
    }

    /// Override of `matvec_q8_fused` required because the trait default is wrong for Q5_K.
    ///
    /// # Why the default is broken
    /// The trait default assumes one Q8_0 block per weight block.  For Q5_K (block_size=256),
    /// each weight super-block spans 8 Q8_0 activation blocks (32 activations each).
    ///
    /// # Block mapping
    /// 1 Q5_K weight block (176 bytes, 256 weights) ↔ 8 Q8_0 activation blocks.
    /// Sub-block `s` (0..8) uses Q8_0 activation at index `blk * 8 + s`.
    ///
    /// # Formula per sub-block `s`
    /// `contrib_s = d_a * (d * sc[s] * Σ(q5 * q_a) − dmin * mn[s] * Σ(q_a))`
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(crate::error::QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(Q5_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
        // Each Q5_K super-block maps to 8 Q8_0 activation blocks.
        let q8_blocks_per_row = blocks_per_row * 8;
        let acts_needed = q8_blocks_per_row * Q8_0_BLOCK_BYTES;

        if weights.len() < n_rows * row_bytes {
            return Err(crate::error::QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(crate::error::QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q5_K_BLOCK_BYTES;
                let block = &weights[bo..bo + Q5_K_BLOCK_BYTES];

                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let (sc, mn) = decode_scales_mins(&block[4..16]);
                let qh = &block[16..48]; // 32 bytes: high bit per weight
                let qs = &block[48..176]; // 128 bytes: lo 4-bit quants

                let input_offset = blk * Q5_K_BLOCK_SIZE;
                let cols_in_block = (n_cols - input_offset).min(Q5_K_BLOCK_SIZE);

                let mut is = 0usize;
                let mut qs_off = 0usize;
                let mut w_off = 0usize;

                for group in 0..4 {
                    // Sub-block `is` (lo nibbles)
                    let a_idx_lo = blk * 8 + is;
                    let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
                    let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
                    let d_a_lo = f16_to_f32(u16::from_le_bytes([a_block_lo[0], a_block_lo[1]]));
                    let q8_lo = &a_block_lo[2..]; // 32 i8 values

                    let da_lo = d * sc[is] as f32;
                    let m_lo = dmin * mn[is] as f32;

                    // Sub-block `is+1` (hi nibbles)
                    let a_idx_hi = blk * 8 + is + 1;
                    let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
                    let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
                    let d_a_hi = f16_to_f32(u16::from_le_bytes([a_block_hi[0], a_block_hi[1]]));
                    let q8_hi = &a_block_hi[2..]; // 32 i8 values

                    let da_hi = d * sc[is + 1] as f32;
                    let m_hi = dmin * mn[is + 1] as f32;

                    // Lo nibbles → first 32 weights of this group
                    let mut dot_lo = 0.0f32;
                    let mut sum_a_lo = 0.0f32;
                    for l in 0..32 {
                        let col = w_off + l;
                        if col < cols_in_block {
                            let qh_bit = (qh[l] >> group) & 1;
                            let q_w = ((qs[qs_off + l] & 0x0F) | (qh_bit << 4)) as f32;
                            let q_a = q8_lo[l] as i8 as f32;
                            dot_lo += q_w * q_a;
                            sum_a_lo += q_a;
                        }
                    }
                    sum += (da_lo * dot_lo - m_lo * sum_a_lo) * d_a_lo;

                    // Hi nibbles → next 32 weights
                    let mut dot_hi = 0.0f32;
                    let mut sum_a_hi = 0.0f32;
                    for l in 0..32 {
                        let col = w_off + 32 + l;
                        if col < cols_in_block {
                            let qh_bit = (qh[l] >> (group + 4)) & 1;
                            let q_w = (((qs[qs_off + l] >> 4) & 0x0F) | (qh_bit << 4)) as f32;
                            let q_a = q8_hi[l] as i8 as f32;
                            dot_hi += q_w * q_a;
                            sum_a_hi += q_a;
                        }
                    }
                    sum += (da_hi * dot_hi - m_hi * sum_a_hi) * d_a_hi;

                    is += 2;
                    qs_off += 32;
                    w_off += 64;
                }
            }

            *out_val += sum;
        }

        Ok(())
    }
}

/// Q8_0 activation block byte count used in fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q5_k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 12],
        qh: &[u8; 32],
        qs: &[u8; 128],
    ) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block.extend_from_slice(scales);
        block.extend_from_slice(qh);
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        // d=0, dmin=0 → all weights = 0
        let block = make_q5_k_block(0.0, 0.0, &[0; 12], &[0; 32], &[0; 128]);
        let kernel = Q5KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_uniform() {
        // d=1.0, dmin=0.0, all scales=1, qh all 0 (5th bit=0), all nibbles=8
        // q_5bit = 8 (lower 4 bits = 8, high bit = 0)
        // Weight = 1.0 * 1 * 8 - 0 = 8.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1); // sub-blocks 0..3 scale=1
        scales[8..12].fill(1); // sub-blocks 4..7 scale=1

        let qh = [0x00u8; 32]; // no high bits
        let qs = [0x88u8; 128]; // lo=8, hi=8

        let block = make_q5_k_block(1.0, 0.0, &scales, &qh, &qs);
        let kernel = Q5KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!((v - 8.0).abs() < 0.01, "weight[{i}] = {v}, expected 8.0");
        }
    }

    #[test]
    fn test_dequant_with_high_bit() {
        // d=1.0, dmin=0.0, all scales=1, qh all 0xFF (all high bits set), all nibbles=0
        // q_5bit = 16 (lower 4 bits = 0, high bit = 1)
        // Weight = 1.0 * 1 * 16 - 0 = 16.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);

        let qh = [0xFFu8; 32]; // all high bits set
        let qs = [0x00u8; 128]; // all low nibbles 0

        let block = make_q5_k_block(1.0, 0.0, &scales, &qh, &qs);
        let kernel = Q5KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!((v - 16.0).abs() < 0.01, "weight[{i}] = {v}, expected 16.0");
        }
    }

    #[test]
    fn test_gemv_q5_k() {
        // Build a block with varied data
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0xFF) as u8;
        }
        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = ((i * 13 + 7) & 0xFF) as u8;
        }
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 5 + 11) & 0xFF) as u8;
        }
        let block = make_q5_k_block(0.5, 0.25, &scales, &qh, &qs);

        let kernel = Q5KRef;

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q5K);
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
