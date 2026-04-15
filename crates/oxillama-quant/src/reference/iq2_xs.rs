//! IQ2_XS reference (naive) implementation.
//!
//! IQ2_XS block format (74 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[QK_K/8]` — 32 × u16 packed as 64 raw bytes (little-endian).
//!   Each u16 encodes: lower 9 bits = 9-bit grid index into `IQ2XS_GRID[512]`;
//!   upper 7 bits = sign selector index into `KSIGNS_IQ2XS[128]`.
//! - bytes[66..74]: `scales[QK_K/32]` — 8 bytes, each byte holds two 4-bit scale values.
//!   For super-block `ib32`:
//!   - `db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25`
//!   - `db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25`
//!
//! ## Block size verification
//!
//! ```text
//! sizeof(block_iq2_xs) = 2 + QK_K/8 * 2 + QK_K/32 = 2 + 64 + 8 = 74 bytes ✓
//! ```
//!
//! ## Super-block layout (QK_K/32 = 8 super-blocks of 32 weights each)
//!
//! For super-block `ib32` (0..8), containing 4 groups of 8 weights each:
//! - Grid indices from `qs[4*ib32 .. 4*ib32+4]` — 4 × u16, lower 9 bits each
//! - Sign selectors: upper 7 bits of each u16 word → KSIGNS_IQ2XS index
//! - Scale byte: `scales[ib32]` — low nibble for groups 0-1, high nibble for groups 2-3
//!
//! ## Scale formula
//!
//! ```text
//! db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25   // groups l=0, l=1
//! db[1] = d * (0.5 + (scales[ib32] >>  4)) * 0.25   // groups l=2, l=3
//! ```
//!
//! ## Weight decode per group (4 groups × 8 weights = 32 weights / super-block)
//!
//! For group `l` (0..4):
//! ```text
//! qs_val     = u16 from block at 4*ib32 + l
//! grid_idx   = qs_val & 511
//! sign_idx   = qs_val >> 9
//! magnitudes = IQ2XS_GRID[grid_idx].to_le_bytes()  // 8 × u8
//! sign_byte  = KSIGNS_IQ2XS[sign_idx]
//! dl         = db[l / 2]
//! w[j]       = dl * magnitudes[j] * if sign_byte & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
//! ```

use super::iq_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XS: 256 weights per block (QK_K = 256).
const IQ2XS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XS block: 2 (FP16 d) + 64 (32 × u16 qs) + 8 (scales).
const IQ2XS_BLOCK_BYTES: usize = 74;
/// Number of super-blocks per IQ2_XS block (QK_K/32 = 8).
const IQ2XS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const IQ2XS_SUPER_BLOCK_SIZE: usize = IQ2XS_BLOCK_SIZE / IQ2XS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (each group = 8 weights, 4 groups × 8 = 32).
const IQ2XS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const IQ2XS_WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of `qs` within the block (after 2-byte FP16 scale).
const IQ2XS_QS_OFFSET: usize = 2;
/// Byte offset of `scales` within the block (after d + qs = 2 + 64 = 66).
const IQ2XS_SCALES_OFFSET: usize = 66;

/// Reference (naive scalar) IQ2_XS kernel.
pub struct Iq2XsRef;

impl QuantKernel for Iq2XsRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ2XS_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2XS_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ2XS_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2XS_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        // qs region: 64 bytes of u16 values (32 × u16, stored little-endian).
        let qs_bytes = &block[IQ2XS_QS_OFFSET..IQ2XS_SCALES_OFFSET];
        // scales region: 8 bytes, one per super-block.
        let scales = &block[IQ2XS_SCALES_OFFSET..IQ2XS_BLOCK_BYTES];

        for (ib32, &scale_byte) in scales.iter().enumerate().take(IQ2XS_N_SUPERBLOCKS) {
            // Each super-block has one scale byte: low nibble → db[0], high nibble → db[1].
            let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;

            let weight_base = ib32 * IQ2XS_SUPER_BLOCK_SIZE;

            for l in 0..IQ2XS_GROUPS_PER_SUPER {
                // Read u16 little-endian at position 4*ib32 + l (in u16 units = 8*ib32 + 2*l bytes).
                let byte_pos = 8 * ib32 + 2 * l;
                let qs_val =
                    u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;

                // Lower 9 bits: grid index into IQ2XS_GRID[512].
                let grid_idx = qs_val & 511;
                // Upper 7 bits: sign selector into KSIGNS_IQ2XS[128].
                let sign_idx = qs_val >> 9;

                let magnitudes: [u8; 8] = IQ2XS_GRID[grid_idx].to_le_bytes();
                let sign_byte = KSIGNS_IQ2XS[sign_idx];

                // Groups 0-1 use db[0], groups 2-3 use db[1].
                let dl = if l < 2 { db0 } else { db1 };

                let group_base = weight_base + l * IQ2XS_WEIGHTS_PER_GROUP;
                for j in 0..IQ2XS_WEIGHTS_PER_GROUP {
                    let mag = magnitudes[j] as f32;
                    let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    output[group_base + j] = dl * mag * sign;
                }
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

        let blocks_per_row = n_cols.div_ceil(IQ2XS_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ2XS_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ2XS_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ2XS_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs_bytes = &block[IQ2XS_QS_OFFSET..IQ2XS_SCALES_OFFSET];
                let scales = &block[IQ2XS_SCALES_OFFSET..IQ2XS_BLOCK_BYTES];

                for (ib32, &scale_byte) in scales.iter().enumerate().take(IQ2XS_N_SUPERBLOCKS) {
                    let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;

                    let col_base = blk * IQ2XS_BLOCK_SIZE + ib32 * IQ2XS_SUPER_BLOCK_SIZE;

                    for l in 0..IQ2XS_GROUPS_PER_SUPER {
                        let byte_pos = 8 * ib32 + 2 * l;
                        let qs_val =
                            u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]])
                                as usize;

                        let grid_idx = qs_val & 511;
                        let sign_idx = qs_val >> 9;

                        let magnitudes: [u8; 8] = IQ2XS_GRID[grid_idx].to_le_bytes();
                        let sign_byte = KSIGNS_IQ2XS[sign_idx];

                        let dl = if l < 2 { db0 } else { db1 };
                        let group_col = col_base + l * IQ2XS_WEIGHTS_PER_GROUP;

                        for j in 0..IQ2XS_WEIGHTS_PER_GROUP {
                            let idx = group_col + j;
                            if idx >= n_cols {
                                break;
                            }
                            let mag = magnitudes[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            sum += dl * mag * sign * input[idx];
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
        IQ2XS_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ2XS_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ2_XS"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Construct a minimal IQ2_XS block with a given FP16 scale and zero data.
    fn make_zero_iq2_xs_block(scale: f32) -> [u8; IQ2XS_BLOCK_BYTES] {
        let mut block = [0u8; IQ2XS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_iq2_xs_metadata() {
        assert_eq!(Iq2XsRef.name(), "IQ2_XS");
        assert_eq!(Iq2XsRef.block_size(), 256);
        assert_eq!(Iq2XsRef.block_bytes(), 74);
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 30];
        let mut out = [0.0f32; 256];
        let result = Iq2XsRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_zero_iq2_xs_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq2XsRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_zero_scale() {
        // d = 0 → all outputs must be 0.0.
        let block = make_zero_iq2_xs_block(0.0);
        let mut out = [1.0f32; 256];
        Iq2XsRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_grid0_no_signs() {
        // All-zero qs → qs_val = 0 → grid_idx=0, sign_idx=0.
        // IQ2XS_GRID[0] = 0x0808080808080808 → all magnitudes = 8.
        // KSIGNS_IQ2XS[0] = 0 → all positive.
        // scales all 0 → db0 = d * 0.5 * 0.25 = d * 0.125.
        // Expected: all weights = d * 0.125 * 8.0 = d * 1.0.
        let d = 2.0_f32;
        let block = make_zero_iq2_xs_block(d);
        let mut out = [0.0f32; 256];
        Iq2XsRef.dequant_block(&block, &mut out).unwrap();

        let expected = d * 0.125 * 8.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_block_high_nibble_scale() {
        // Set scales[0] = 0x20 → low nibble = 0, high nibble = 2.
        // db[0] = d * (0.5 + 0) * 0.25 = d * 0.125  (groups l=0,1)
        // db[1] = d * (0.5 + 2) * 0.25 = d * 0.625  (groups l=2,3)
        let d = 1.0_f32;
        let mut block = make_zero_iq2_xs_block(d);
        block[IQ2XS_SCALES_OFFSET] = 0x20; // scales[0] = 0x20

        let mut out = [0.0f32; 256];
        Iq2XsRef.dequant_block(&block, &mut out).unwrap();

        // IQ2XS_GRID[0] magnitudes = [8;8], sign_idx=0 → all positive
        // Groups 0-1 (weights 0..15): db[0] * 8 = 1.0 * 0.125 * 8 = 1.0
        let expected_lo = d * 0.125 * 8.0;
        for (i, &v) in out.iter().enumerate().take(16) {
            assert!(
                (v - expected_lo).abs() < 1e-4,
                "out[{i}]={v}, expected {expected_lo}"
            );
        }
        // Groups 2-3 (weights 16..31): db[1] * 8 = 1.0 * 0.625 * 8 = 5.0
        let expected_hi = d * 0.625 * 8.0;
        for (i, &v) in out.iter().enumerate().take(32).skip(16) {
            assert!(
                (v - expected_hi).abs() < 1e-4,
                "out[{i}]={v}, expected {expected_hi}"
            );
        }
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq2Xs));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        let d = 1.0_f32;
        let block = make_zero_iq2_xs_block(d);

        let mut dequant = [0.0f32; 256];
        Iq2XsRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq2Xs,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq2XsRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
