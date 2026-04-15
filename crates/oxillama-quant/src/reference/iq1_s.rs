//! IQ1_S reference (naive) implementation.
//!
//! IQ1_S block format (50 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d` (little-endian)
//! - bytes[2..34]:   `qs[QK_K/8]` — 32 bytes of quantized indices (lower 8 bits)
//! - bytes[34..50]:  `qh[QK_K/32]` — 8 × u16 sub-block headers (little-endian)
//!
//! ## Sub-block layout (8 sub-blocks of 32 weights each)
//!
//! For sub-block `ib` (0..8):
//! - `dl = d * (2 * ((qh[ib] >> 12) & 7) + 1)` — scale magnitude
//! - `delta = if qh[ib] & 0x8000 != 0 { -0.125 } else { 0.125 }` — sign delta
//! - 4 groups of 8 weights, each with 11-bit grid index:
//!   `grid_idx = qs[4*ib + l] | (((qh[ib] >> (3*l)) & 7) << 8)`
//! - Grid value bytes are SIGNED (i8): -1, 0, or +1
//! - `y[j] = dl * (grid_i8[j] as f32 + delta)`
//!
//! ## Block size verification
//!
//! 2 (FP16 d) + QK_K/8 (qs) + QK_K/32 × 2 (qh u16) = 2 + 32 + 16 = 50 ✓

use super::iq1s_grid::IQ1S_GRID;
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ1_S: 256 weights per block (QK_K = 256).
const IQ1S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_S block: 2 (FP16 d) + 32 (qs) + 16 (qh as 8×u16).
const IQ1S_BLOCK_BYTES: usize = 50;
/// Byte offset where `qs` begins.
const IQ1S_QS_OFFSET: usize = 2;
/// Byte offset where `qh` (8 × u16 LE) begins.
const IQ1S_QH_OFFSET: usize = 34;
/// Number of sub-blocks per IQ1_S block (QK_K/32 = 8).
const IQ1S_N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
const IQ1S_SUB_BLOCK_SIZE: usize = IQ1S_BLOCK_SIZE / IQ1S_N_SUBBLOCKS; // 32
/// Number of groups per sub-block (each group = 8 weights).
const IQ1S_GROUPS_PER_SUB: usize = 4;
/// Weights per group.
const IQ1S_WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant (IQ1S_DELTA = 0.125).
const IQ1S_DELTA: f32 = 0.125;

/// Reference (naive scalar) IQ1_S kernel.
pub struct Iq1SRef;

impl QuantKernel for Iq1SRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ1S_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ1S_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ1S_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ1S_BLOCK_SIZE,
                available: output.len(),
            });
        }

        // FP16 global scale d.
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // qs: 32 bytes, lower 8 bits of each 11-bit grid index.
        let qs = &block[IQ1S_QS_OFFSET..IQ1S_QH_OFFSET];
        // qh: 8 × u16 little-endian, each holding scale bits + 3-bit grid selectors.
        let qh_bytes = &block[IQ1S_QH_OFFSET..IQ1S_BLOCK_BYTES];

        for ib in 0..IQ1S_N_SUBBLOCKS {
            // Parse the 16-bit sub-block header.
            let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);

            // Scale: d × (2 × scale_bits + 1), where scale_bits ∈ [0..7] from bits 12-14.
            let scale_bits = ((qh_val >> 12) & 0x7) as f32;
            let dl = d * (2.0 * scale_bits + 1.0);

            // Delta sign from bit 15.
            let delta = if qh_val & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };

            let qs_base = ib * IQ1S_GROUPS_PER_SUB;
            let output_base = ib * IQ1S_SUB_BLOCK_SIZE;

            for l in 0..IQ1S_GROUPS_PER_SUB {
                // 11-bit grid index: lower 8 bits from qs, upper 3 bits from qh.
                let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
                let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);

                // Grid value: each byte is a signed i8 weight (-1, 0, or +1).
                let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

                let group_base = output_base + l * IQ1S_WEIGHTS_PER_GROUP;
                for j in 0..IQ1S_WEIGHTS_PER_GROUP {
                    let gv = grid_raw[j] as i8 as f32;
                    output[group_base + j] = dl * (gv + delta);
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

        let blocks_per_row = n_cols.div_ceil(IQ1S_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ1S_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ1S_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ1S_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs = &block[IQ1S_QS_OFFSET..IQ1S_QH_OFFSET];
                let qh_bytes = &block[IQ1S_QH_OFFSET..IQ1S_BLOCK_BYTES];

                for ib in 0..IQ1S_N_SUBBLOCKS {
                    let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);
                    let scale_bits = ((qh_val >> 12) & 0x7) as f32;
                    let dl = d * (2.0 * scale_bits + 1.0);
                    let delta = if qh_val & 0x8000 != 0 {
                        -IQ1S_DELTA
                    } else {
                        IQ1S_DELTA
                    };

                    let qs_base = ib * IQ1S_GROUPS_PER_SUB;
                    let col_base = blk * IQ1S_BLOCK_SIZE + ib * IQ1S_SUB_BLOCK_SIZE;

                    for l in 0..IQ1S_GROUPS_PER_SUB {
                        let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
                        let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);
                        let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

                        let col = col_base + l * IQ1S_WEIGHTS_PER_GROUP;
                        for (j, &raw_byte) in grid_raw.iter().enumerate() {
                            let idx = col + j;
                            if idx >= n_cols {
                                break;
                            }
                            let gv = raw_byte as i8 as f32;
                            sum += dl * (gv + delta) * input[idx];
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
        IQ1S_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ1S_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ1_S"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Build a minimal IQ1_S block with the given FP16 scale and zero qs/qh.
    fn make_zero_iq1s_block(scale: f32) -> [u8; IQ1S_BLOCK_BYTES] {
        let mut block = [0u8; IQ1S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_iq1_s_metadata() {
        assert_eq!(Iq1SRef.name(), "IQ1_S");
        assert_eq!(Iq1SRef.block_size(), 256);
        assert_eq!(Iq1SRef.block_bytes(), 50);
    }

    #[test]
    fn test_dequant_buffer_too_small_block() {
        let small = [0u8; 30];
        let mut out = [0.0f32; 256];
        let result = Iq1SRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_buffer_too_small_output() {
        let block = make_zero_iq1s_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq1SRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_zero_scale() {
        // d = 0 → all outputs must be 0.0 regardless of grid indices.
        let block = make_zero_iq1s_block(0.0);
        let mut out = [1.0f32; 256];
        Iq1SRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_grid0_positive_delta() {
        // Grid index 0 → IQ1S_GRID[0] = 0xffffffffffffffff → all bytes = 0xff = i8(-1).
        // qh = 0x0000 → scale_bits = 0 → dl = d * 1.0
        //              → bit 15 = 0 → delta = +0.125
        // Expected: dl * (-1.0 + 0.125) = d * (-0.875)
        let d = 2.0_f32;
        let block = make_zero_iq1s_block(d);
        let mut out = [0.0f32; 256];
        Iq1SRef.dequant_block(&block, &mut out).unwrap();

        let expected = d * (-1.0 + IQ1S_DELTA);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_scale_bits_set() {
        // qh[0] = 0x3000 → bits 12-14 = 0x3 → scale_bits = 3
        // dl = d * (2*3 + 1) = d * 7
        // grid index 0 still → grid byte = 0xff = -1
        // delta = +0.125 (bit 15 = 0)
        // Expected: d * 7 * (-1 + 0.125) = d * 7 * (-0.875)
        let d = 1.0_f32;
        let mut block = make_zero_iq1s_block(d);
        // Set qh[0] = 0x3000 (LE: bytes are [0x00, 0x30])
        block[IQ1S_QH_OFFSET] = 0x00;
        block[IQ1S_QH_OFFSET + 1] = 0x30;
        let mut out = [0.0f32; 256];
        Iq1SRef.dequant_block(&block, &mut out).unwrap();

        // Only sub-block 0 changes; rest use qh=0.
        let expected_sb0 = d * 7.0 * (-1.0 + IQ1S_DELTA);
        for (i, &v) in out.iter().enumerate().take(32) {
            assert!(
                (v - expected_sb0).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected_sb0}"
            );
        }
    }

    #[test]
    fn test_dequant_negative_delta() {
        // qh[0] bit 15 = 1 → delta = -0.125.
        // All other sub-blocks: qh = 0 → delta = +0.125.
        let d = 1.0_f32;
        let mut block = make_zero_iq1s_block(d);
        // Set qh[0] = 0x8000 (LE: [0x00, 0x80])
        block[IQ1S_QH_OFFSET] = 0x00;
        block[IQ1S_QH_OFFSET + 1] = 0x80;
        let mut out = [0.0f32; 256];
        Iq1SRef.dequant_block(&block, &mut out).unwrap();

        // Sub-block 0: dl = d*1 (scale_bits=0), delta = -0.125
        let expected_sb0 = d * 1.0 * (-1.0 - IQ1S_DELTA);
        for (i, &v) in out.iter().enumerate().take(32) {
            assert!(
                (v - expected_sb0).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected_sb0}"
            );
        }
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq1S));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        // 1-row × 256-col tensor; gemv with input=ones must equal sum of dequant.
        let d = 1.0_f32;
        let block = make_zero_iq1s_block(d);

        let mut dequant = [0.0f32; 256];
        Iq1SRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq1S,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq1SRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
