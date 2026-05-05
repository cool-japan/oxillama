//! Q3_K reference (naive) implementation.
//!
//! Q3_K block format (110 bytes per 256 weights):
//! - 32 bytes: hmask — one bit per weight; if set, q_hi=0; if clear, q_lo -= 4
//! - 64 bytes: qs — lower 2 bits of each 3-bit quant (4 per byte via bit shifts)
//! - 12 bytes: scales — 16 sub-block scales, 6-bit unsigned, packed (signed = raw - 32)
//! - 2 bytes: FP16 super-block scale (d)
//!
//! Q3_K is a symmetric ("type-0") format: no minimum offset.
//! Weight formula: `w = d * scale_i * (q_lo - (hmask_bit ? 0 : 4))`
//!   where q_lo is 2-bit (0..3), giving effective range -4..3.
//!
//! 16 sub-blocks of 16 weights each.
//! Effective: 3.4375 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q3_K_BLOCK_SIZE: usize = 256;
const Q3_K_BLOCK_BYTES: usize = 110;

/// Reference (naive scalar) Q3_K kernel.
pub struct Q3KRef;

/// Decode 16 signed 6-bit scales from the 12-byte packed representation.
///
/// The packing uses 16 × 6-bit values stored across 12 bytes.
/// The decoded values are unsigned 0..63, then we subtract 32 to get signed -32..31.
fn decode_scales(scales_raw: &[u8]) -> [f32; 16] {
    let mut sc = [0u32; 16];

    // Scales 0..3: lower 6 bits of bytes 0..3
    for j in 0..4 {
        sc[j] = (scales_raw[j] & 0x3F) as u32;
    }
    // Scales 4..7: lower 6 bits of bytes 4..7
    for j in 0..4 {
        sc[4 + j] = (scales_raw[4 + j] & 0x3F) as u32;
    }
    // Scales 8..11: low 4 bits from bytes 8..11, high 2 bits from upper bits of bytes 0..3
    for j in 0..4 {
        let lo = (scales_raw[8 + j] & 0x0F) as u32;
        let hi = ((scales_raw[j] >> 6) & 0x03) as u32;
        sc[8 + j] = lo | (hi << 4);
    }
    // Scales 12..15: high 4 bits from bytes 8..11, high 2 bits from upper bits of bytes 4..7
    for j in 0..4 {
        let lo = ((scales_raw[8 + j] >> 4) & 0x0F) as u32;
        let hi = ((scales_raw[4 + j] >> 6) & 0x03) as u32;
        sc[12 + j] = lo | (hi << 4);
    }

    // Convert to signed: subtract 32
    let mut result = [0.0f32; 16];
    for i in 0..16 {
        result[i] = (sc[i] as i32 - 32) as f32;
    }
    result
}

impl QuantKernel for Q3KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q3_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q3_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q3_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q3_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        let sc = decode_scales(scales_raw);

        // Dequantize following llama.cpp layout:
        //
        // Two groups of 128 weights. Each group uses the same 32 qs bytes
        // (with 4 different shift values extracting 2 bits each = 8 bits total)
        // and the same 32 hmask bytes (with different bit selectors).
        //
        // `m` is a rotating bit selector for hmask, starting at bit 0.
        // It advances once per shift (4 shifts per group × 2 groups = 8 total bits used).
        //
        // hmask[byte] & m: if SET → subtract 0; if CLEAR → subtract 4.
        // This gives effective range: -4..3 for the 3-bit quant.

        let mut is = 0usize; // scale index
        let mut out_off = 0usize;
        let mut m: u8 = 1; // hmask bit selector

        for group in 0..2 {
            let qs_base = group * 32;

            for shift in (0..8).step_by(2) {
                // Two sub-blocks of 16 weights each
                for n in 0..2 {
                    let dl = d * sc[is];
                    is += 1;

                    for l in 0..16 {
                        let qs_idx = qs_base + n * 16 + l;
                        let q_lo = ((qs[qs_idx] >> shift) & 3) as i32;
                        let subtract = if hmask[n * 16 + l] & m != 0 { 0 } else { 4 };
                        output[out_off + l] = dl * (q_lo - subtract) as f32;
                    }
                    out_off += 16;
                }
                m <<= 1;
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

        let blocks_per_row = n_cols.div_ceil(Q3_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q3_K_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let hmask = &data[bo..bo + 32];
                let qs = &data[bo + 32..bo + 96];
                let scales_raw = &data[bo + 96..bo + 108];
                let d = f16_to_f32(u16::from_le_bytes([data[bo + 108], data[bo + 109]]));
                let sc = decode_scales(scales_raw);
                let inp = &input[blk * Q3_K_BLOCK_SIZE..];
                let cols_in_block = (n_cols - blk * Q3_K_BLOCK_SIZE).min(Q3_K_BLOCK_SIZE);

                // Inline dot product: extract 3-bit quants on-the-fly
                let mut is = 0usize;
                let mut in_off = 0usize;
                let mut m_bit: u8 = 1;

                for group in 0..2 {
                    let qs_base = group * 32;
                    for shift in (0..8).step_by(2) {
                        for n in 0..2 {
                            let dl = d * sc[is];
                            is += 1;
                            for l in 0..16 {
                                if in_off + l < cols_in_block {
                                    let qs_idx = qs_base + n * 16 + l;
                                    let q_lo = ((qs[qs_idx] >> shift) & 3) as i32;
                                    let subtract =
                                        if hmask[n * 16 + l] & m_bit != 0 { 0 } else { 4 };
                                    sum += dl * (q_lo - subtract) as f32 * inp[in_off + l];
                                }
                            }
                            in_off += 16;
                        }
                        m_bit <<= 1;
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

    /// Override required: the trait default assumes 1 Q8_0 block per weight block,
    /// but Q3_K has 256 weights per block → 8 Q8_0 blocks (32 weights each).
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(Q3_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
        let q8_blocks_per_row = blocks_per_row * 8;
        let acts_needed = q8_blocks_per_row * Q8_0_BLOCK_BYTES;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * Q3_K_BLOCK_BYTES;
                let block = &weights[bo..bo + Q3_K_BLOCK_BYTES];
                let hmask = &block[0..32];
                let qs = &block[32..96];
                let scales_raw = &block[96..108];
                let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
                let sc = decode_scales(scales_raw);
                let cols_in_block =
                    (n_cols.saturating_sub(blk * Q3_K_BLOCK_SIZE)).min(Q3_K_BLOCK_SIZE);

                let mut is = 0usize;
                let mut col_off = 0usize;
                let mut m_bit: u8 = 1;

                for group in 0..2 {
                    let qs_base = group * 32;
                    for shift in (0..8usize).step_by(2) {
                        // Sub-block A (hmask[0..16])
                        {
                            let dl = d * sc[is];
                            is += 1;
                            let q8_blk_idx = blk * 8 + col_off / 32;
                            let q8_lane_base = col_off % 32;
                            let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                            let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                            let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));
                            let q8_vals = &a_block[2..];
                            let mut dot = 0.0f32;
                            for l in 0..16 {
                                if col_off + l < cols_in_block {
                                    let q_lo = ((qs[qs_base + l] >> shift) & 3) as i32;
                                    let sub = if hmask[l] & m_bit != 0 { 0 } else { 4 };
                                    let q3 = (q_lo - sub) as f32;
                                    let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                    dot += q3 * q_a;
                                }
                            }
                            sum += dl * dot * d_a;
                            col_off += 16;
                        }
                        // Sub-block B (hmask[16..32])
                        {
                            let dl = d * sc[is];
                            is += 1;
                            let q8_blk_idx = blk * 8 + col_off / 32;
                            let q8_lane_base = col_off % 32;
                            let a_start = q8_blk_idx * Q8_0_BLOCK_BYTES;
                            let a_block = &acts_q8[a_start..a_start + Q8_0_BLOCK_BYTES];
                            let d_a = f16_to_f32(u16::from_le_bytes([a_block[0], a_block[1]]));
                            let q8_vals = &a_block[2..];
                            let mut dot = 0.0f32;
                            for l in 0..16 {
                                if col_off + l < cols_in_block {
                                    let q_lo = ((qs[qs_base + 16 + l] >> shift) & 3) as i32;
                                    let sub = if hmask[16 + l] & m_bit != 0 { 0 } else { 4 };
                                    let q3 = (q_lo - sub) as f32;
                                    let q_a = q8_vals[q8_lane_base + l] as i8 as f32;
                                    dot += q3 * q_a;
                                }
                            }
                            sum += dl * dot * d_a;
                            col_off += 16;
                        }
                        m_bit = m_bit.wrapping_shl(1);
                    }
                }
            }

            *out_val += sum;
        }

        Ok(())
    }

    fn block_size(&self) -> usize {
        Q3_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q3_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q3_K"
    }
}

/// Q8_0 bytes per block for fused GEMV.
const Q8_0_BLOCK_BYTES: usize = 34;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q3_k_block(d: f32, scales: &[u8; 12], hmask: &[u8; 32], qs: &[u8; 64]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q3_K_BLOCK_BYTES);
        block.extend_from_slice(hmask);
        block.extend_from_slice(qs);
        block.extend_from_slice(scales);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0 → all weights = 0
        let block = make_q3_k_block(0.0, &[0; 12], &[0; 32], &[0; 64]);
        let kernel = Q3KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_hmask_set_q0() {
        // hmask all set → subtract 0. qs all 0 → q_lo = 0.
        // Value = q_lo - 0 = 0. Weight = d * scale * 0 = 0.
        // Scale doesn't matter since value is 0.
        let hmask = [0xFFu8; 32];
        let qs = [0x00u8; 64];

        // Scales = 33 (raw) → signed = 1
        let mut scales = [0u8; 12];
        scales[..8].fill(0x21); // 0x21 = 33, lower 6 bits = 33

        let block = make_q3_k_block(1.0, &scales, &hmask, &qs);
        let kernel = Q3KRef;
        let mut output = vec![99.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!((v).abs() < 0.01, "weight[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_dequant_hmask_clear() {
        // hmask all clear → subtract 4. qs all 0 → q_lo = 0.
        // Value = 0 - 4 = -4. Weight = d * scale * (-4).
        // d=1.0, all scales decode to signed +1 (raw 33, since 33-32=1)
        // Weight = 1.0 * 1 * (-4) = -4.0
        let hmask = [0x00u8; 32];
        let qs = [0x00u8; 64];

        // All 16 scales = raw 33 → signed 1
        // Bytes 0..3: 0x21 (lower 6 bits = 33), upper 2 bits = 0
        // Bytes 4..7: 0x21
        // Bytes 8..11: sc[8..11] = lo4 | (hi2 << 4). Want 33 = 1 | (2<<4).
        //   lo4 from byte[8+j] & 0xF = 1, hi2 from byte[j] >> 6 = 2
        //   → byte[j] needs upper 2 bits = 2: byte[0..3] = 0x21 | (2<<6) = 0xA1
        //   → byte[8+j] & 0xF = 1
        // sc[12..15] = (byte[8+j]>>4)&0xF | ((byte[4+j]>>6)&3)<<4. Want 33 = 1|(2<<4).
        //   → (byte[8+j]>>4) = 1, (byte[4+j]>>6) = 2
        //   → byte[8+j] = 0x11, byte[4+j] = 0x21 | (2<<6) = 0xA1
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];

        let block = make_q3_k_block(1.0, &scales, &hmask, &qs);
        let kernel = Q3KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-4.0)).abs() < 0.01,
                "weight[{i}] = {v}, expected -4.0"
            );
        }
    }

    #[test]
    fn test_dequant_hmask_set_q3() {
        // hmask all set → subtract 0. All q_lo = 3.
        // Value = 3 - 0 = 3. Weight = d * scale * 3.
        // d=1.0, all scales = signed +1 → Weight = 3.0
        //
        // qs: each byte stores 4 × 2-bit values via shifts 0,2,4,6.
        // To have all q_lo = 3: all qs bytes = 0xFF (11_11_11_11 binary).
        let hmask = [0xFFu8; 32];
        let qs = [0xFFu8; 64];

        // Same scale encoding as above: all 16 scales = signed +1
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];

        let block = make_q3_k_block(1.0, &scales, &hmask, &qs);
        let kernel = Q3KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!((v - 3.0).abs() < 0.01, "weight[{i}] = {v}, expected 3.0");
        }
    }

    #[test]
    fn test_gemv_q3_k() {
        // Build a deterministic block
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        for (i, h) in hmask.iter_mut().enumerate() {
            *h = ((i * 7 + 3) & 0xFF) as u8;
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((i * 11 + 5) & 0xFF) as u8;
        }
        // All 16 scales = signed +1 (raw 33)
        let scales: [u8; 12] = [
            0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0xA1, 0x11, 0x11, 0x11, 0x11,
        ];
        let block = make_q3_k_block(0.5, &scales, &hmask, &qs);

        let kernel = Q3KRef;

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut dequant).unwrap();

        // Input vector
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // Reference dot product
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q3K);
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
