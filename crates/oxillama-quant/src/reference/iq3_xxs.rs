//! IQ3_XXS reference (naive) implementation.
//!
//! IQ3_XXS block format (98 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..98]:  `qs[3*QK_K/8]` — 96 raw bytes divided into two regions:
//!   - bytes[2..66]:  `qs_grid[QK_K/4]` = 64 bytes — grid indices (2 per 8-weight group)
//!   - bytes[66..98]: `qs_signs[QK_K/8]` = 32 bytes — packed scale + sign data
//!
//! ## Block size verification
//!
//! ```text
//! sizeof(block_iq3_xxs) = 2 + 3*(QK_K/8) = 2 + 3*32 = 2 + 96 = 98 bytes ✓
//! qs_grid  = qs[0..64]    (QK_K/4 = 64 bytes)
//! qs_signs = qs[64..96]   (QK_K/8 = 32 bytes; sits at offset 64 in qs = byte 66 in block)
//! ```
//!
//! ## Super-block layout (QK_K/32 = 8 super-blocks of 32 weights each)
//!
//! For super-block `ib32` (0..8):
//! - Grid indices from `qs_grid[8*ib32 .. 8*ib32+8]` — 8 bytes (2 per group × 4 groups)
//! - `aux32 = u32::from_le_bytes(qs_signs[4*ib32 .. 4*ib32+4])`
//!   - bits 28-31: 4-bit scale multiplier
//!   - bits [7*l .. 7*l+6] for l in 0..4: 7-bit sign selector per group
//!
//! ## Scale formula
//!
//! ```text
//! db = d * (0.5 + (aux32 >> 28)) * 0.5
//! ```
//!
//! ## Weight decode per group (4 groups × 8 weights = 32 weights / super-block)
//!
//! For group `l` (0..4), using grid indices `g1 = qs_grid[8*ib32 + 2*l]` and
//! `g2 = qs_grid[8*ib32 + 2*l + 1]`:
//! ```text
//! mags1 = IQ3XXS_GRID[g1].to_le_bytes()   // 4 × u8 magnitudes
//! mags2 = IQ3XXS_GRID[g2].to_le_bytes()   // 4 × u8 magnitudes
//! sign_byte = KSIGNS_IQ2XS[(aux32 >> 7*l) & 127]
//! w[j+0] = db * mags1[j] * (if sign_byte & KMASK_IQ2XS[j]   != 0 { -1 } else { 1 })
//! w[j+4] = db * mags2[j] * (if sign_byte & KMASK_IQ2XS[j+4] != 0 { -1 } else { 1 })
//! ```

use super::iq_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ3_XXS: 256 weights per block (QK_K = 256).
const IQ3XXS_BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_XXS block: 2 (FP16 d) + 96 (3 * QK_K/8).
const IQ3XXS_BLOCK_BYTES: usize = 98;
/// Number of super-blocks per IQ3_XXS block (QK_K/32 = 8).
const IQ3XXS_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const IQ3XXS_SUPER_BLOCK_SIZE: usize = IQ3XXS_BLOCK_SIZE / IQ3XXS_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (4 groups × 8 weights = 32).
const IQ3XXS_GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const IQ3XXS_WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset within the qs region where sign/scale data begins (QK_K/4 = 64).
const IQ3XXS_SIGNS_OFFSET: usize = 64;

/// Reference (naive scalar) IQ3_XXS kernel.
pub struct Iq3XxsRef;

impl QuantKernel for Iq3XxsRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ3XXS_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ3XXS_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ3XXS_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ3XXS_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        // qs region = block[2..98] (96 bytes).
        let qs = &block[2..IQ3XXS_BLOCK_BYTES];
        // Grid indices: qs[0..64]
        let qs_grid = &qs[..IQ3XXS_SIGNS_OFFSET];
        // Scale + sign data: qs[64..96]
        let qs_signs = &qs[IQ3XXS_SIGNS_OFFSET..];

        for ib32 in 0..IQ3XXS_N_SUPERBLOCKS {
            // Read the 4-byte packed scale/sign word for this super-block.
            let signs_base = ib32 * 4;
            let aux32 = u32::from_le_bytes([
                qs_signs[signs_base],
                qs_signs[signs_base + 1],
                qs_signs[signs_base + 2],
                qs_signs[signs_base + 3],
            ]);

            // Scale: db = d * (0.5 + (aux32 >> 28)) * 0.5
            let scale_bits = (aux32 >> 28) as f32;
            let db = d * (0.5 + scale_bits) * 0.5;

            // Grid indices: 2 per group × 4 groups = 8 bytes per super-block.
            let grid_base = ib32 * 8;
            let weight_base = ib32 * IQ3XXS_SUPER_BLOCK_SIZE;

            for l in 0..IQ3XXS_GROUPS_PER_SUPER {
                let g1 = qs_grid[grid_base + 2 * l] as usize;
                let g2 = qs_grid[grid_base + 2 * l + 1] as usize;
                let mags1: [u8; 4] = IQ3XXS_GRID[g1].to_le_bytes();
                let mags2: [u8; 4] = IQ3XXS_GRID[g2].to_le_bytes();

                // 7-bit sign selector for this group.
                let sign_idx = ((aux32 >> (7 * l)) & 0x7F) as usize;
                let sign_byte = KSIGNS_IQ2XS[sign_idx];

                let group_base = weight_base + l * IQ3XXS_WEIGHTS_PER_GROUP;
                for j in 0..4 {
                    // First 4 weights from grid1; signs from bits 0..3.
                    let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    output[group_base + j] = db * mags1[j] as f32 * sign1;

                    // Second 4 weights from grid2; signs from bits 4..7.
                    let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    output[group_base + j + 4] = db * mags2[j] as f32 * sign2;
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

        let blocks_per_row = n_cols.div_ceil(IQ3XXS_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ3XXS_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ3XXS_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ3XXS_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs = &block[2..IQ3XXS_BLOCK_BYTES];
                let qs_grid = &qs[..IQ3XXS_SIGNS_OFFSET];
                let qs_signs = &qs[IQ3XXS_SIGNS_OFFSET..];

                for ib32 in 0..IQ3XXS_N_SUPERBLOCKS {
                    let signs_base = ib32 * 4;
                    let aux32 = u32::from_le_bytes([
                        qs_signs[signs_base],
                        qs_signs[signs_base + 1],
                        qs_signs[signs_base + 2],
                        qs_signs[signs_base + 3],
                    ]);

                    let scale_bits = (aux32 >> 28) as f32;
                    let db = d * (0.5 + scale_bits) * 0.5;

                    let grid_base = ib32 * 8;
                    let col_base = blk * IQ3XXS_BLOCK_SIZE + ib32 * IQ3XXS_SUPER_BLOCK_SIZE;

                    for l in 0..IQ3XXS_GROUPS_PER_SUPER {
                        let g1 = qs_grid[grid_base + 2 * l] as usize;
                        let g2 = qs_grid[grid_base + 2 * l + 1] as usize;
                        let mags1: [u8; 4] = IQ3XXS_GRID[g1].to_le_bytes();
                        let mags2: [u8; 4] = IQ3XXS_GRID[g2].to_le_bytes();

                        let sign_idx = ((aux32 >> (7 * l)) & 0x7F) as usize;
                        let sign_byte = KSIGNS_IQ2XS[sign_idx];

                        let group_col = col_base + l * IQ3XXS_WEIGHTS_PER_GROUP;
                        for j in 0..4 {
                            let idx1 = group_col + j;
                            if idx1 < n_cols {
                                let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                    -1.0_f32
                                } else {
                                    1.0_f32
                                };
                                sum += db * mags1[j] as f32 * sign1 * input[idx1];
                            }
                            let idx2 = group_col + j + 4;
                            if idx2 < n_cols {
                                let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                                    -1.0_f32
                                } else {
                                    1.0_f32
                                };
                                sum += db * mags2[j] as f32 * sign2 * input[idx2];
                            }
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
        IQ3XXS_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ3XXS_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ3_XXS"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Build a minimal IQ3_XXS block with a given FP16 scale and all-zero qs bytes.
    fn make_zero_iq3_xxs_block(scale: f32) -> [u8; IQ3XXS_BLOCK_BYTES] {
        let mut block = [0u8; IQ3XXS_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_kernel_metadata() {
        assert_eq!(Iq3XxsRef.name(), "IQ3_XXS");
        assert_eq!(Iq3XxsRef.block_size(), 256);
        assert_eq!(Iq3XxsRef.block_bytes(), 98);
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 50];
        let mut out = [0.0f32; 256];
        let result = Iq3XxsRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_zero_iq3_xxs_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq3XxsRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_zero_scale() {
        let block = make_zero_iq3_xxs_block(0.0);
        let mut out = [1.0f32; 256];
        Iq3XxsRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_grid0_no_signs() {
        // Grid index 0 → IQ3XXS_GRID[0] = 0x04040404 → all magnitudes = 4.
        // All-zero qs → grid indices all 0, signs_idx=0 → KSIGNS_IQ2XS[0]=0 → all positive.
        // aux32 = 0 → scale_bits = 0 → db = d * 0.5 * 0.5 = d * 0.25.
        // Expected: all weights = d * 0.25 * 4 = d.
        let d = 2.0_f32;
        let block = make_zero_iq3_xxs_block(d);
        let mut out = [0.0f32; 256];
        Iq3XxsRef.dequant_block(&block, &mut out).unwrap();

        let db = d * 0.5 * 0.5; // = d * 0.25
                                // IQ3XXS_GRID[0] = 0x04040404 → each byte = 4
        let expected = db * 4.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_block_sign_flip_grid0() {
        // Set sign_idx=1 for the first group of super-block 0.
        // KSIGNS_IQ2XS[1] = 129 = 0b10000001 → weight[0] and weight[7] negated.
        // Grid index 0 → magnitudes = [4, 4, 4, 4] for both g1 and g2.
        // aux32 = 1 (sign_bits[0..6] = 1, scale_bits = 0).
        let d = 1.0_f32;
        let mut block = make_zero_iq3_xxs_block(d);
        // Set qs_signs[0..4] (block byte 2+64=66) to aux32 = 1.
        block[2 + IQ3XXS_SIGNS_OFFSET] = 1;

        let mut out = [0.0f32; 256];
        Iq3XxsRef.dequant_block(&block, &mut out).unwrap();

        let db = d * 0.5 * 0.5;
        let mag = 4.0_f32;
        // weight[0]: sign_byte bit 0 set → negative
        assert!(
            (out[0] - (-db * mag)).abs() < 1e-5,
            "out[0]={}, expected {}",
            out[0],
            -db * mag
        );
        // weight[1]: bit 1 not set → positive
        assert!(
            (out[1] - (db * mag)).abs() < 1e-5,
            "out[1]={}, expected {}",
            out[1],
            db * mag
        );
        // weight[7]: sign_byte bit 7 set → negative
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
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq3Xxs));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        let d = 1.0_f32;
        let block = make_zero_iq3_xxs_block(d);

        let mut dequant = [0.0f32; 256];
        Iq3XxsRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq3Xxs,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq3XxsRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
