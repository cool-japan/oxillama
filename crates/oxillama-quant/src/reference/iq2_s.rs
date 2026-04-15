//! IQ2_S reference (naive) implementation.
//!
//! IQ2_S block format (82 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[QK_K/4]` — 64 raw bytes, split into two 32-byte regions:
//!   - `qs[0..32]`:  base grid indices (4 per super-block × 8 super-blocks = 32 bytes)
//!   - `qs[32..64]`: per-group sign masks (4 per super-block × 8 super-blocks = 32 bytes)
//! - bytes[66..74]: `qh[QK_K/32]` — 8 bytes, one per super-block; provides high bits
//!   for the grid index: 2 bits per group × 4 groups = 8 bits = 1 byte per super-block.
//! - bytes[74..82]: `scales[QK_K/32]` — 8 bytes, one per super-block.
//!   Low nibble → `db[0]` (groups 0-1), high nibble → `db[1]` (groups 2-3).
//!
//! ## Block size verification
//!
//! ```text
//! sizeof(block_iq2_s) = 2 + QK_K/4 + QK_K/32 + QK_K/32
//!                     = 2 + 64 + 8 + 8 = 82 bytes ✓
//! ```
//!
//! ## Super-block layout (QK_K/32 = 8 super-blocks of 32 weights each)
//!
//! For super-block `ib32` (0..8), containing 4 groups of 8 weights each:
//! - Base grid bytes at `qs[4*ib32 .. 4*ib32+4]` (first 32-byte region)
//! - Sign mask bytes at `qs[4*ib32+32 .. 4*ib32+36]` (second 32-byte region = signs)
//! - High bits: `qh[ib32]` — bits `[(8 - 2*l) % 9]` for group `l` (bit 8 in 9-bit index)
//! - Scale byte: `scales[ib32]`
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
//! base_idx   = qs_base[4*ib32 + l] as u16
//! high_bit   = ((qh[ib32] as u16) << (8 - 2*l)) & 0x300  // 0 or 256
//! grid_idx   = (base_idx | high_bit) as usize             // 10-bit index 0..1023
//! signs_byte = qs_signs[4*ib32 + l]                        // direct sign mask
//! dl         = db[l / 2]
//! magnitudes = IQ2S_GRID[grid_idx].to_le_bytes()            // 8 × u8
//! w[j]       = dl * magnitudes[j] * if signs_byte & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
//! ```

use super::iq_grids::{IQ2S_GRID, KMASK_IQ2XS};
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_S: 256 weights per block (QK_K = 256).
const IQ2S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_S block: 2 + 64 + 8 + 8 = 82.
const IQ2S_BLOCK_BYTES: usize = 82;
/// Number of super-blocks per IQ2_S block (QK_K/32 = 8).
const IQ2S_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const IQ2S_SUPER_BLOCK_SIZE: usize = IQ2S_BLOCK_SIZE / IQ2S_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (4 groups × 8 weights = 32).
const IQ2S_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const IQ2S_WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of `qs` within the block (after 2-byte FP16 scale).
const IQ2S_QS_OFFSET: usize = 2;
/// Size of the qs region in bytes (QK_K/4 = 64).
const IQ2S_QS_BYTES: usize = 64;
/// Byte offset within qs where sign bytes begin (QK_K/8 = 32).
const IQ2S_SIGNS_IN_QS: usize = 32;
/// Byte offset of `qh` within the block (after d + qs = 2 + 64 = 66).
const IQ2S_QH_OFFSET: usize = 66;
/// Byte offset of `scales` within the block (after d + qs + qh = 2 + 64 + 8 = 74).
const IQ2S_SCALES_OFFSET: usize = 74;

/// Reference (naive scalar) IQ2_S kernel.
pub struct Iq2SRef;

impl QuantKernel for Iq2SRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ2S_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2S_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ2S_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2S_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        // qs region: 64 bytes total.
        let qs = &block[IQ2S_QS_OFFSET..IQ2S_QS_OFFSET + IQ2S_QS_BYTES];
        // First 32 bytes: base grid indices (one per group per super-block).
        let qs_base = &qs[..IQ2S_SIGNS_IN_QS];
        // Second 32 bytes: sign masks (one per group per super-block).
        let qs_signs = &qs[IQ2S_SIGNS_IN_QS..];
        // qh region: 8 bytes, one per super-block.
        let qh = &block[IQ2S_QH_OFFSET..IQ2S_SCALES_OFFSET];
        // scales region: 8 bytes, one per super-block.
        let scales = &block[IQ2S_SCALES_OFFSET..IQ2S_BLOCK_BYTES];

        for ib32 in 0..IQ2S_N_SUPERBLOCKS {
            let scale_byte = scales[ib32];
            let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
            let qh_byte = qh[ib32] as u16;

            let weight_base = ib32 * IQ2S_SUPER_BLOCK_SIZE;

            for l in 0..IQ2S_GROUPS_PER_SUPER {
                // 10-bit grid index: 8 bits from qs_base, 1 high bit from qh.
                // C: qs[l] | (qh[ib32] << (8 - 2*l) & 0x300)
                // In Rust with explicit precedence:
                let base_idx = qs_base[4 * ib32 + l] as u16;
                let shift = 8u16.saturating_sub(2 * l as u16);
                let high_bit = (qh_byte << shift) & 0x300;
                let grid_idx = (base_idx | high_bit) as usize;

                // Sign mask: direct byte from qs_signs (NOT decoded through KSIGNS).
                let signs_byte = qs_signs[4 * ib32 + l];

                let magnitudes: [u8; 8] = IQ2S_GRID[grid_idx].to_le_bytes();

                // Groups 0-1 use db[0], groups 2-3 use db[1].
                let dl = if l < 2 { db0 } else { db1 };

                let group_base = weight_base + l * IQ2S_WEIGHTS_PER_GROUP;
                for j in 0..IQ2S_WEIGHTS_PER_GROUP {
                    let mag = magnitudes[j] as f32;
                    let sign = if signs_byte & KMASK_IQ2XS[j] != 0 {
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

        let blocks_per_row = n_cols.div_ceil(IQ2S_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ2S_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ2S_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ2S_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs = &block[IQ2S_QS_OFFSET..IQ2S_QS_OFFSET + IQ2S_QS_BYTES];
                let qs_base = &qs[..IQ2S_SIGNS_IN_QS];
                let qs_signs = &qs[IQ2S_SIGNS_IN_QS..];
                let qh = &block[IQ2S_QH_OFFSET..IQ2S_SCALES_OFFSET];
                let scales = &block[IQ2S_SCALES_OFFSET..IQ2S_BLOCK_BYTES];

                for ib32 in 0..IQ2S_N_SUPERBLOCKS {
                    let scale_byte = scales[ib32];
                    let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                    let qh_byte = qh[ib32] as u16;

                    let col_base = blk * IQ2S_BLOCK_SIZE + ib32 * IQ2S_SUPER_BLOCK_SIZE;

                    for l in 0..IQ2S_GROUPS_PER_SUPER {
                        let base_idx = qs_base[4 * ib32 + l] as u16;
                        let shift = 8u16.saturating_sub(2 * l as u16);
                        let high_bit = (qh_byte << shift) & 0x300;
                        let grid_idx = (base_idx | high_bit) as usize;

                        let signs_byte = qs_signs[4 * ib32 + l];
                        let magnitudes: [u8; 8] = IQ2S_GRID[grid_idx].to_le_bytes();
                        let dl = if l < 2 { db0 } else { db1 };

                        let group_col = col_base + l * IQ2S_WEIGHTS_PER_GROUP;
                        for j in 0..IQ2S_WEIGHTS_PER_GROUP {
                            let idx = group_col + j;
                            if idx >= n_cols {
                                break;
                            }
                            let mag = magnitudes[j] as f32;
                            let sign = if signs_byte & KMASK_IQ2XS[j] != 0 {
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
        IQ2S_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ2S_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ2_S"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Construct a minimal IQ2_S block with a given FP16 scale and zero data.
    fn make_zero_iq2_s_block(scale: f32) -> [u8; IQ2S_BLOCK_BYTES] {
        let mut block = [0u8; IQ2S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_iq2_s_metadata() {
        assert_eq!(Iq2SRef.name(), "IQ2_S");
        assert_eq!(Iq2SRef.block_size(), 256);
        assert_eq!(Iq2SRef.block_bytes(), 82);
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 40];
        let mut out = [0.0f32; 256];
        let result = Iq2SRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_zero_iq2_s_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq2SRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_zero_scale() {
        let block = make_zero_iq2_s_block(0.0);
        let mut out = [1.0f32; 256];
        Iq2SRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_grid0_no_signs() {
        // All-zero block: grid_idx=0 → IQ2S_GRID[0] = 0x0808080808080808 → magnitudes all 8.
        // signs_byte=0 → all positive.
        // scales[ib32]=0 → db[0] = d * 0.5 * 0.25 = d * 0.125.
        // Expected: all weights = d * 0.125 * 8 = d.
        let d = 2.0_f32;
        let block = make_zero_iq2_s_block(d);
        let mut out = [0.0f32; 256];
        Iq2SRef.dequant_block(&block, &mut out).unwrap();

        let expected = d * 0.125 * 8.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq2S));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        let d = 1.0_f32;
        let block = make_zero_iq2_s_block(d);

        let mut dequant = [0.0f32; 256];
        Iq2SRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq2S,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq2SRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }

    #[test]
    fn test_sign_mask_applied() {
        // Set qs_signs[0] = 1 (bit 0 set) → weight[0] of first group should be negated.
        // IQ2S_GRID[0] = all-8 magnitudes. d=1, scales=0 → db=0.125*1=0.125.
        let d = 1.0_f32;
        let mut block = make_zero_iq2_s_block(d);
        // qs_signs is at qs[32..64] within block bytes [2..66].
        // qs_signs[0] = block[2 + 32] = block[34].
        block[2 + IQ2S_SIGNS_IN_QS] = 1; // sets bit 0 → weight[0] negated
        let mut out = [0.0f32; 256];
        Iq2SRef.dequant_block(&block, &mut out).unwrap();

        let dl = d * 0.5 * 0.25;
        let mag = 8.0_f32;
        // weight[0] should be negative
        assert!(
            (out[0] - (-dl * mag)).abs() < 1e-5,
            "out[0]={}, expected {}",
            out[0],
            -dl * mag
        );
        // weight[1] should be positive (bit 1 not set)
        assert!(
            (out[1] - (dl * mag)).abs() < 1e-5,
            "out[1]={}, expected {}",
            out[1],
            dl * mag
        );
    }
}
