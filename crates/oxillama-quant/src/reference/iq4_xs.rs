//! IQ4_XS reference (naive) implementation.
//!
//! IQ4_XS block format (136 bytes per 256 weights):
//! - bytes[0..2]:   FP16 delta `d` (little-endian)
//! - bytes[2..4]:   `scales_h` — 2-byte little-endian u16 holding the
//!   2-bit high parts of the 8 sub-block scales (bits `[5:4]`).
//! - bytes[4..8]:   `scales_l` — 4 bytes holding the 4-bit low parts of
//!   the 8 sub-block scales (bits `[3:0]`), packed two per byte.
//! - bytes[8..136]: 128 nibble-bytes encoding 256 four-bit weights.
//!   Low nibble = `weight[2i]`, high nibble = `weight[2i+1]`.
//!
//! Each block is divided into 8 sub-blocks of 32 weights.  Sub-block `i`
//! has its own sub-scale `ls` derived from `scales_h` and `scales_l`:
//!
//! ```text
//! ls_low  = (scales_l[i/2] >> (4 * (i & 1))) & 0x0F
//! ls_high = (scales_h_u16 >> (2 * i)) as u8 & 0x03
//! ls      = ls_low | (ls_high << 4)          // 6-bit value [0..63]
//! ls_signed = ls wrapping_sub 32              // centred on 0 → [-32..31]
//! ```
//!
//! Dequantisation: `w = d * ls_signed * KVALUES_IQ4NL[nibble]`

use super::iq_shared::KVALUES_IQ4NL;
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ4_XS: 256 weights per block.
const IQ4_XS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ4_XS block: 2 (d) + 2 (scales_h) + 4 (scales_l) + 128 (nibbles).
const IQ4_XS_BLOCK_BYTES: usize = 136;
/// Number of sub-blocks per IQ4_XS block.
const IQ4_XS_N_SUPERBLOCKS: usize = 8;
/// Weights per sub-block.
const IQ4_XS_SUB_BLOCK_SIZE: usize = IQ4_XS_BLOCK_SIZE / IQ4_XS_N_SUPERBLOCKS; // 32

/// Unpack the sub-scale for sub-block `i` from the raw block bytes.
///
/// # Arguments
/// * `scales_h_u16` — `u16` read from bytes[2..4] as little-endian.
/// * `scales_l`     — slice of 4 bytes from bytes[4..8].
/// * `i`            — sub-block index in [0, 8).
///
/// Returns the signed sub-scale value centred around 0 (range −32..=31).
#[inline]
fn unpack_sub_scale(scales_h_u16: u16, scales_l: &[u8], i: usize) -> i32 {
    let ls_low: u8 = (scales_l[i / 2] >> (4 * (i & 1))) & 0x0F;
    let ls_high: u8 = (scales_h_u16 >> (2 * i)) as u8 & 0x03;
    let ls: u8 = ls_low | (ls_high << 4);
    // Centre around zero: ls is in [0, 63], subtract 32 → [−32, 31].
    (ls as i32).wrapping_sub(32)
}

/// Reference (naive scalar) IQ4_XS kernel.
pub struct Iq4XsRef;

impl QuantKernel for Iq4XsRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ4_XS_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ4_XS_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ4_XS_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ4_XS_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        // Nibble data starts at byte 8.
        let nibbles = &block[8..136];

        for sub in 0..IQ4_XS_N_SUPERBLOCKS {
            let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
            let scale = d * ls_signed as f32;

            // Each sub-block is 32 weights → 16 nibble-bytes.
            let nibble_offset = sub * (IQ4_XS_SUB_BLOCK_SIZE / 2); // 16 bytes per sub-block
            let weight_offset = sub * IQ4_XS_SUB_BLOCK_SIZE;

            for i in 0..(IQ4_XS_SUB_BLOCK_SIZE / 2) {
                let byte = nibbles[nibble_offset + i];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                output[weight_offset + i * 2] = scale * KVALUES_IQ4NL[lo] as f32;
                output[weight_offset + i * 2 + 1] = scale * KVALUES_IQ4NL[hi] as f32;
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

        let blocks_per_row = n_cols.div_ceil(IQ4_XS_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ4_XS_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ4_XS_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
                let scales_l = &block[4..8];
                let nibbles = &block[8..136];

                for sub in 0..IQ4_XS_N_SUPERBLOCKS {
                    let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
                    let scale = d * ls_signed as f32;

                    let nibble_offset = sub * (IQ4_XS_SUB_BLOCK_SIZE / 2);
                    // Absolute column offset for the first weight in this sub-block.
                    let col_offset = blk * IQ4_XS_BLOCK_SIZE + sub * IQ4_XS_SUB_BLOCK_SIZE;

                    for i in 0..(IQ4_XS_SUB_BLOCK_SIZE / 2) {
                        let byte = nibbles[nibble_offset + i];
                        let lo = (byte & 0x0F) as usize;
                        let hi = ((byte >> 4) & 0x0F) as usize;
                        let idx = col_offset + i * 2;

                        if idx + 1 < n_cols {
                            sum += KVALUES_IQ4NL[lo] as f32 * scale * input[idx];
                            sum += KVALUES_IQ4NL[hi] as f32 * scale * input[idx + 1];
                        } else if idx < n_cols {
                            sum += KVALUES_IQ4NL[lo] as f32 * scale * input[idx];
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
        IQ4_XS_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ4_XS_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ4_XS"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Build a minimal IQ4_XS block.
    ///
    /// * `scale`     — FP16 delta `d`.
    /// * `sub_scales` — 8 unsigned sub-scale values in [0, 63] (will be centred at decode
    ///   time by subtracting 32).
    /// * `nibbles`   — 128-byte nibble data (256 weights packed as 4-bit indices).
    fn make_iq4_xs_block(
        scale: f32,
        sub_scales: [u8; 8], // values in [0..63]
        nibbles: [u8; 128],
    ) -> [u8; IQ4_XS_BLOCK_BYTES] {
        let mut block = [0u8; IQ4_XS_BLOCK_BYTES];

        // bytes[0..2]: FP16 d
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];

        // Encode sub_scales into scales_h (2-bit high) and scales_l (4-bit low).
        // sub_scale[i] = ls_low[i] | (ls_high[i] << 4)
        // ls_low  → bits[3:0], packed two per byte in scales_l[4]
        // ls_high → bits[5:4] == high 2 bits, packed in scales_h u16 (2 bits each)
        let mut scales_h_u16: u16 = 0;
        let mut scales_l = [0u8; 4];
        for i in 0..IQ4_XS_N_SUPERBLOCKS {
            let v = sub_scales[i] & 0x3F; // clamp to 6 bits
            let ls_low = v & 0x0F;
            let ls_high = (v >> 4) & 0x03;
            // Pack ls_low into scales_l: 2 per byte, nibble i goes to
            // the appropriate nibble position.
            if i & 1 == 0 {
                scales_l[i / 2] |= ls_low;
            } else {
                scales_l[i / 2] |= ls_low << 4;
            }
            // Pack ls_high into scales_h_u16: 2 bits per sub-block at position 2*i.
            scales_h_u16 |= (ls_high as u16) << (2 * i);
        }
        let sh = scales_h_u16.to_le_bytes();
        block[2] = sh[0];
        block[3] = sh[1];
        block[4..8].copy_from_slice(&scales_l);
        block[8..136].copy_from_slice(&nibbles);
        block
    }

    #[test]
    fn test_dequant_block_known_sub_scale() {
        // d = 1.0, all sub_scales = 32 (centred → ls_signed = 0) → all weights = 0.
        let block = make_iq4_xs_block(1.0, [32u8; 8], [0xAAu8; 128]);
        let mut out = [0.0f32; 256];
        Iq4XsRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "weight[{i}] = {v}, expected 0 (ls_signed=0)"
            );
        }
    }

    #[test]
    fn test_dequant_block_unit_sub_scale() {
        // d = 1.0, sub_scale[0] = 33 → ls_signed = 1.
        // All nibbles in sub-block 0 = 0x88 → lo=hi=8 → KVALUES[8] = 1
        // → each weight in sub-block 0 = 1.0 * 1 * 1 = 1.0
        // All other sub_scales = 32 → ls_signed=0 → weights = 0
        let mut sub_scales = [32u8; 8];
        sub_scales[0] = 33; // ls_signed = 1
        let block = make_iq4_xs_block(1.0, sub_scales, [0x88u8; 128]);
        let mut out = [0.0f32; 256];
        Iq4XsRef.dequant_block(&block, &mut out).unwrap();

        // First 32 weights should be 1.0
        for (i, &val) in out.iter().enumerate().take(32) {
            assert!((val - 1.0f32).abs() < 0.01, "out[{i}]={val}, expected 1.0",);
        }
        // Remaining 224 weights should be 0.0 (ls_signed=0)
        for (i, &val) in out.iter().enumerate().skip(32) {
            assert!(val.abs() < 1e-6, "out[{i}]={val}, expected 0",);
        }
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 50];
        let mut out = [0.0f32; 256];
        let result = Iq4XsRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_iq4_xs_block(1.0, [32u8; 8], [0x00u8; 128]);
        let mut out = [0.0f32; 100];
        let result = Iq4XsRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_gemv_sum_equals_dequant_dot_ones() {
        // 1-row × 256-col tensor; gemv with input=ones must equal sum of dequant.
        // sub_scale=33 (ls_signed=1), d=1.0, nibbles=0x88 (KVALUES[8]=1)
        // expected: 256 * 1.0 * 1 * 1 = 256.0
        let mut sub_scales = [32u8; 8];
        sub_scales.iter_mut().for_each(|s| *s = 33);
        let block = make_iq4_xs_block(1.0, sub_scales, [0x88u8; 128]);

        // Verify via dequant
        let mut dequant = [0.0f32; 256];
        Iq4XsRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq4Xs,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq4XsRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }

    #[test]
    fn test_gemv_two_rows() {
        // 2-row × 256-col tensor; two distinct configurations.
        // Row 0: sub_scales all 32 → ls_signed=0 → all weights=0 → dot=0
        // Row 1: sub_scales all 33 → ls_signed=1, d=1.0, nibbles=0x88 (KVALUES[8]=1)
        //        → each weight=1.0 → dot(ones)=256
        let block0 = make_iq4_xs_block(1.0, [32u8; 8], [0x88u8; 128]);
        let block1 = make_iq4_xs_block(1.0, [33u8; 8], [0x88u8; 128]);
        let mut data = Vec::with_capacity(IQ4_XS_BLOCK_BYTES * 2);
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);

        let tensor = QuantTensor::new(data, vec![2, 256], oxillama_gguf::GgufTensorType::Iq4Xs);
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 2];
        Iq4XsRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(out[0].abs() < 1e-4, "row0={}", out[0]);
        // KVALUES[8] = 1; 256 weights * 1.0 * 1 * 1 = 256
        assert!((out[1] - 256.0f32).abs() < 0.5, "row1={}", out[1]);
    }

    #[test]
    fn test_block_size_and_bytes() {
        assert_eq!(Iq4XsRef.block_size(), 256);
        assert_eq!(Iq4XsRef.block_bytes(), 136);
        assert_eq!(Iq4XsRef.name(), "IQ4_XS");
    }

    #[test]
    fn test_sub_scale_roundtrip() {
        // Verify unpack_sub_scale correctly inverts make_iq4_xs_block's encoding
        // for all 8 sub-blocks with a variety of sub_scale values.
        let sub_scales = [32u8, 0, 63, 1, 31, 33, 16, 48];
        let block = make_iq4_xs_block(1.0, sub_scales, [0x00u8; 128]);
        let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];

        for (i, &raw_scale) in sub_scales.iter().enumerate() {
            let got = unpack_sub_scale(scales_h_u16, scales_l, i);
            let expected = (raw_scale as i32).wrapping_sub(32);
            assert_eq!(
                got, expected,
                "sub-block {i}: got ls_signed={got}, expected {expected}"
            );
        }
    }
}
