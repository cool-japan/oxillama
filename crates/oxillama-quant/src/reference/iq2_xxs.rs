//! IQ2_XXS reference (naive) implementation.
//!
//! IQ2_XXS block format (66 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[QK_K/8]` — 32 × u16 packed as 64 raw bytes (little-endian).
//!   The 256 weights are divided into 8 super-blocks of 32 weights each.
//!   Each super-block occupies 8 bytes (4 × u16 = 8 bytes), using the layout:
//!   - Lower 8-bit per u16 word: grid index for 8 weights (4 group indices)
//!   - Upper half of the second u32 in each pair: scale bits and sign bits.
//!
//! ## Super-block layout (per 8-byte chunk, indices into `qs`)
//!
//! For super-block `ib32` (0..8), the 8 bytes at `qs[4*ib32 .. 4*ib32+8]` are
//! split into two `u32` little-endian words:
//! - `aux32[0]` = `u32::from_le_bytes(qs[4*ib32 .. 4*ib32+4])`
//! - `aux32[1]` = `u32::from_le_bytes(qs[4*ib32+4 .. 4*ib32+8])`
//!
//! Within the 8 bytes seen as `aux8[0..8]`:
//! - `aux8[0..4]` (= bytes of `aux32[0]`): 4 grid indices (1 byte each, 0..255)
//!   Each index selects an entry from `IQ2XXS_GRID[idx]` which yields 8 magnitudes.
//! - `aux32[1] >> 28`: 4-bit sub-scale multiplier (0..15)
//! - `(aux32[1] >> (7*l)) & 0x7F` for l in 0..4: 7-bit sign selector per group
//!
//! ## Scale formula
//!
//! ```text
//! db = d * (0.5 + (aux32[1] >> 28)) * 0.25
//! ```
//!
//! ## Weight decode per group (4 groups × 8 weights = 32 weights / super-block)
//!
//! For group `l` (0..4):
//! ```text
//! magnitudes  = IQ2XXS_GRID[aux8[l]].to_le_bytes()  // 8 × u8
//! sign_byte   = KSIGNS_IQ2XS[(aux32[1] >> 7*l) & 127]
//! w[j]        = db * magnitudes[j] * if sign_byte & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
//! ```

use super::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XXS: 256 weights per block (QK_K = 256).
const IQ2XXS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XXS block: 2 (FP16 d) + 64 (32 × u16 qs).
/// Verification: 2 + QK_K/8 * 2 = 2 + 256/8 * 2 = 2 + 64 = 66.
const IQ2XXS_BLOCK_BYTES: usize = 66;
/// Number of super-blocks per IQ2_XXS block (QK_K/32 = 8).
const IQ2XXS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const IQ2XXS_SUPER_BLOCK_SIZE: usize = IQ2XXS_BLOCK_SIZE / IQ2XXS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (each group = 8 weights, 4 groups × 8 = 32).
#[allow(dead_code)]
const IQ2XXS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const IQ2XXS_WEIGHTS_PER_GROUP: usize = 8;

/// Reference (naive scalar) IQ2_XXS kernel.
pub struct Iq2XxsRef;

impl QuantKernel for Iq2XxsRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ2XXS_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2XXS_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ2XXS_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ2XXS_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        // `qs` starts at byte 2: 64 bytes (32 × u16, stored as raw bytes).
        let qs = &block[2..IQ2XXS_BLOCK_BYTES];

        for ib32 in 0..IQ2XXS_N_SUPERBLOCKS {
            // Each super-block consumes 8 bytes from qs (2 × u32).
            let base = ib32 * 8;

            // aux32[0] = lower 4 bytes → 4 individual grid indices (aux8[0..4])
            let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            // aux32[1] = upper 4 bytes → scale bits (bits 28-31) and sign selectors
            let aux32_1 =
                u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

            // Scale: d * (0.5 + (aux32[1] >> 28)) * 0.25
            let scale_factor = (aux32_1 >> 28) as f32;
            let db = d * (0.5 + scale_factor) * 0.25;

            // The 4 grid indices are the individual bytes of aux32[0].
            let aux8: [u8; 4] = aux32_0.to_le_bytes();

            let weight_base = ib32 * IQ2XXS_SUPER_BLOCK_SIZE;

            for (l, &grid_byte) in aux8.iter().enumerate() {
                // Grid index for this group of 8.
                let grid_idx = grid_byte as usize;
                let grid_entry = IQ2XXS_GRID[grid_idx];
                let magnitudes: [u8; 8] = grid_entry.to_le_bytes();

                // 7-bit sign selector: bits [7*l .. 7*l+6] of aux32[1].
                let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                let sign_byte = KSIGNS_IQ2XS[sign_idx];

                let group_base = weight_base + l * IQ2XXS_WEIGHTS_PER_GROUP;
                for j in 0..IQ2XXS_WEIGHTS_PER_GROUP {
                    let mag = magnitudes[j] as f32;
                    let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    output[group_base + j] = db * mag * sign;
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

        let blocks_per_row = n_cols.div_ceil(IQ2XXS_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ2XXS_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ2XXS_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ2XXS_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs = &block[2..IQ2XXS_BLOCK_BYTES];

                for ib32 in 0..IQ2XXS_N_SUPERBLOCKS {
                    let base = ib32 * 8;
                    let aux32_0 =
                        u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
                    let aux32_1 = u32::from_le_bytes([
                        qs[base + 4],
                        qs[base + 5],
                        qs[base + 6],
                        qs[base + 7],
                    ]);

                    let scale_factor = (aux32_1 >> 28) as f32;
                    let db = d * (0.5 + scale_factor) * 0.25;
                    let aux8: [u8; 4] = aux32_0.to_le_bytes();

                    let col_base = blk * IQ2XXS_BLOCK_SIZE + ib32 * IQ2XXS_SUPER_BLOCK_SIZE;

                    for (l, &grid_byte) in aux8.iter().enumerate() {
                        let grid_idx = grid_byte as usize;
                        let magnitudes: [u8; 8] = IQ2XXS_GRID[grid_idx].to_le_bytes();
                        let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                        let sign_byte = KSIGNS_IQ2XS[sign_idx];

                        let col = col_base + l * IQ2XXS_WEIGHTS_PER_GROUP;
                        for j in 0..IQ2XXS_WEIGHTS_PER_GROUP {
                            let idx = col + j;
                            if idx >= n_cols {
                                break;
                            }
                            let mag = magnitudes[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            sum += db * mag * sign * input[idx];
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
        IQ2XXS_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ2XXS_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ2_XXS"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Construct a minimal IQ2_XXS block with a given FP16 scale and zero qs data.
    fn make_zero_iq2_xxs_block(scale: f32) -> [u8; IQ2XXS_BLOCK_BYTES] {
        let mut block = [0u8; IQ2XXS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    /// Build an IQ2_XXS block where every super-block uses:
    /// - grid index 0 → IQ2XXS_GRID[0] = 0x0808080808080808 → all magnitudes = 8
    /// - aux32[1] = 0 → scale_bits=0 (db = d * 0.5 * 0.25), sign_idx=0 → all positive
    fn make_uniform_block(scale: f32) -> [u8; IQ2XXS_BLOCK_BYTES] {
        // Grid index 0 in all 4 bytes of aux32[0], aux32[1] = 0.
        // This gives: aux8 = [0,0,0,0], scale_bits=0, signs=all_positive.
        make_zero_iq2_xxs_block(scale)
    }

    #[test]
    fn test_kernel_metadata() {
        assert_eq!(Iq2XxsRef.name(), "IQ2_XXS");
        assert_eq!(Iq2XxsRef.block_size(), 256);
        assert_eq!(Iq2XxsRef.block_bytes(), 66);
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 30];
        let mut out = [0.0f32; 256];
        let result = Iq2XxsRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_zero_iq2_xxs_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq2XxsRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_zero_scale() {
        // d = 0 → all outputs must be 0.0 regardless of indices.
        let block = make_zero_iq2_xxs_block(0.0);
        let mut out = [1.0f32; 256]; // pre-fill with non-zero
        Iq2XxsRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_grid0_no_signs() {
        // Grid index 0 → IQ2XXS_GRID[0] = 0x0808080808080808 → all magnitudes = 8.
        // aux32[1] = 0 → scale_bits = 0 (db = d * 0.5 * 0.25 = d * 0.125),
        //                sign_idx = 0 → KSIGNS_IQ2XS[0] = 0 → all positive.
        // Expected: all weights = d * 0.125 * 8.0 = d * 1.0.
        let d = 2.0_f32;
        let block = make_uniform_block(d);
        let mut out = [0.0f32; 256];
        Iq2XxsRef.dequant_block(&block, &mut out).unwrap();

        // IQ2XXS_GRID[0] = 0x0808080808080808 → each byte = 0x08 = 8
        // db = 2.0 * (0.5 + 0) * 0.25 = 2.0 * 0.125 = 0.25
        // weight = 0.25 * 8 * 1.0 = 2.0
        let expected = d * 0.125 * 8.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_block_sign_flip() {
        // Use sign_idx = 1 → KSIGNS_IQ2XS[1] = 129 = 0b10000001.
        // Bit 0 set → weight[0] negated, bit 7 set → weight[7] negated.
        // Grid index 0 → all magnitudes = 8.
        // db = d * 0.5 * 0.25 = d * 0.125.
        let d = 1.0_f32;
        let mut block = make_zero_iq2_xxs_block(d);
        // Set sign_idx = 1 for group 0 of super-block 0:
        //   bits [0..6] of aux32[1] = 1  →  byte at block[2+4..2+8] = [1, 0, 0, 0] LE.
        block[2 + 4] = 1; // aux32[1] = 1 (lower byte)
        block[2 + 5] = 0;
        block[2 + 6] = 0;
        block[2 + 7] = 0;

        let mut out = [0.0f32; 256];
        Iq2XxsRef.dequant_block(&block, &mut out).unwrap();

        // KSIGNS_IQ2XS[1] = 129 = 0b_1000_0001
        // KMASK_IQ2XS[0] = 1 → bit 0 set → weight[0] negative
        // KMASK_IQ2XS[7] = 128 → bit 7 set → weight[7] negative
        let db = d * 0.5 * 0.25;
        let mag = 8.0_f32;
        assert!(
            (out[0] - (-db * mag)).abs() < 1e-5,
            "out[0]={}, expected {}",
            out[0],
            -db * mag
        );
        assert!(
            (out[1] - (db * mag)).abs() < 1e-5,
            "out[1]={}, expected {}",
            out[1],
            db * mag
        );
        assert!(
            (out[7] - (-db * mag)).abs() < 1e-5,
            "out[7]={}, expected {}",
            out[7],
            -db * mag
        );
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq2Xxs));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        // 1-row × 256-col tensor; gemv with input=ones must equal sum of dequant.
        let d = 1.0_f32;
        let block = make_uniform_block(d);

        let mut dequant = [0.0f32; 256];
        Iq2XxsRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq2Xxs,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq2XxsRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
