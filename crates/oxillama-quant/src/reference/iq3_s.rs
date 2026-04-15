//! IQ3_S reference (naive) implementation.
//!
//! IQ3_S block format (110 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d` (little-endian)
//! - bytes[2..66]:   `qs[QK_K/4]` = 64 bytes — grid base indices (8 per super-block)
//! - bytes[66..74]:  `qh[QK_K/32]` = 8 bytes — high bits for grid indices (1 byte per sb)
//! - bytes[74..106]: `signs[QK_K/8]` = 32 bytes — per-group sign masks (4 per super-block)
//! - bytes[106..110]: `scales[QK_K/64]` = 4 bytes — per-pair-of-super-blocks scale nibbles
//!
//! ## Block size verification
//!
//! ```text
//! sizeof(block_iq3_s) = 2 + QK_K/4 + QK_K/32 + QK_K/8 + QK_K/64
//!                     = 2 + 64 + 8 + 32 + 4 = 110 bytes ✓
//! IQ3S_N_SCALE = QK_K/64 = 4
//! ```
//!
//! ## Super-block layout (processes in pairs: ib32 += 2, QK_K/32 = 8 super-blocks total)
//!
//! For each pair (ib32, ib32+1) at index `pair = ib32/2` (0..4):
//! - `db1 = d * (1 + 2 * (scales[pair] & 0xf))` — scale for super-block ib32
//! - `db2 = d * (1 + 2 * (scales[pair] >> 4))` — scale for super-block ib32+1
//!
//! For each of the two super-blocks within the pair:
//! - Grid base indices at `qs[8*ib32 .. 8*ib32+8]` — 8 bytes (2 per group × 4 groups)
//! - High bits: `qh[ib32]` — 2 bits per group, bit `(8-2*l)` for group l
//! - Sign masks at `signs[4*ib32 .. 4*ib32+4]` — 4 bytes (1 per group)
//!
//! ## Weight decode per group (4 groups × 8 weights = 32 weights / super-block)
//!
//! For group `l` (0..4):
//! ```text
//! // Two grid entries per group (4 weights each)
//! idx1 = qs[8*ib32 + 2*l]   | ((qh[k] << (8 - 2*l)) & 256)  // 9-bit
//! idx2 = qs[8*ib32 + 2*l+1] | ((qh[k] << (7 - 2*l)) & 256)  // 9-bit
//! grid1 = IQ3S_GRID[idx1].to_le_bytes()  // 4 × u8
//! grid2 = IQ3S_GRID[idx2].to_le_bytes()  // 4 × u8
//! sign_byte = signs[4*ib32 + l]           // direct bit mask (no ksigns lookup)
//! w[j+0] = db * grid1[j] * (if sign_byte & KMASK_IQ2XS[j]   != 0 { -1 } else { 1 })
//! w[j+4] = db * grid2[j] * (if sign_byte & KMASK_IQ2XS[j+4] != 0 { -1 } else { 1 })
//! ```
//!
//! Note: scale formula is `d * (1 + 2*nibble)` — NOT the `d * (0.5 + nibble) * factor`
//! pattern used by IQ2_XXS / IQ2_XS / IQ3_XXS.

use super::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ3_S: 256 weights per block (QK_K = 256).
const IQ3S_BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_S block: 2 + 64 + 8 + 32 + 4 = 110.
const IQ3S_BLOCK_BYTES: usize = 110;
/// Number of super-blocks per IQ3_S block (QK_K/32 = 8).
const IQ3S_N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const IQ3S_SUPER_BLOCK_SIZE: usize = IQ3S_BLOCK_SIZE / IQ3S_N_SUPERBLOCKS; // 32
/// Number of weight groups per super-block (4 groups × 8 weights = 32).
const IQ3S_GROUPS_PER_SUPER: usize = 4;

/// Byte offset of `qs` within the block.
const IQ3S_QS_OFFSET: usize = 2;
/// Size of qs region (QK_K/4 = 64 bytes).
const IQ3S_QS_BYTES: usize = 64;
/// Byte offset of `qh` within the block.
const IQ3S_QH_OFFSET: usize = 66;
/// Size of qh region (QK_K/32 = 8 bytes).
const IQ3S_QH_BYTES: usize = 8;
/// Byte offset of `signs` within the block.
const IQ3S_SIGNS_OFFSET: usize = 74;
/// Size of signs region (QK_K/8 = 32 bytes).
const IQ3S_SIGNS_BYTES: usize = 32;
/// Byte offset of `scales` within the block.
const IQ3S_SCALES_OFFSET: usize = 106;

/// Reference (naive scalar) IQ3_S kernel.
pub struct Iq3SRef;

impl QuantKernel for Iq3SRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < IQ3S_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: IQ3S_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < IQ3S_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: IQ3S_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[IQ3S_QS_OFFSET..IQ3S_QS_OFFSET + IQ3S_QS_BYTES];
        let qh = &block[IQ3S_QH_OFFSET..IQ3S_QH_OFFSET + IQ3S_QH_BYTES];
        let signs = &block[IQ3S_SIGNS_OFFSET..IQ3S_SIGNS_OFFSET + IQ3S_SIGNS_BYTES];
        let scales = &block[IQ3S_SCALES_OFFSET..IQ3S_BLOCK_BYTES];

        // Process super-blocks in pairs (ib32 += 2).
        let mut ib32 = 0usize;
        while ib32 < IQ3S_N_SUPERBLOCKS {
            let pair = ib32 / 2;
            let scale_byte = scales[pair];
            let db1 = d * (1.0 + 2.0 * (scale_byte & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (scale_byte >> 4) as f32);

            // First super-block in pair: ib32, qh[0] in pair, db1.
            dequant_superblock(
                &qs[8 * ib32..8 * ib32 + 8],
                qh[ib32],
                &signs[4 * ib32..4 * ib32 + 4],
                db1,
                &mut output[ib32 * IQ3S_SUPER_BLOCK_SIZE..(ib32 + 1) * IQ3S_SUPER_BLOCK_SIZE],
            );

            // Second super-block in pair: ib32+1, qh[1] in pair, db2.
            let ib32b = ib32 + 1;
            dequant_superblock(
                &qs[8 * ib32b..8 * ib32b + 8],
                qh[ib32b],
                &signs[4 * ib32b..4 * ib32b + 4],
                db2,
                &mut output[ib32b * IQ3S_SUPER_BLOCK_SIZE..(ib32b + 1) * IQ3S_SUPER_BLOCK_SIZE],
            );

            ib32 += 2;
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

        let blocks_per_row = n_cols.div_ceil(IQ3S_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ3S_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ3S_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ3S_BLOCK_BYTES];

                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let qs = &block[IQ3S_QS_OFFSET..IQ3S_QS_OFFSET + IQ3S_QS_BYTES];
                let qh = &block[IQ3S_QH_OFFSET..IQ3S_QH_OFFSET + IQ3S_QH_BYTES];
                let signs = &block[IQ3S_SIGNS_OFFSET..IQ3S_SIGNS_OFFSET + IQ3S_SIGNS_BYTES];
                let scales = &block[IQ3S_SCALES_OFFSET..IQ3S_BLOCK_BYTES];

                let mut ib32 = 0usize;
                while ib32 < IQ3S_N_SUPERBLOCKS {
                    let pair = ib32 / 2;
                    let scale_byte = scales[pair];
                    let db1 = d * (1.0 + 2.0 * (scale_byte & 0xf) as f32);
                    let db2 = d * (1.0 + 2.0 * (scale_byte >> 4) as f32);

                    // First super-block.
                    sum += gemv_superblock(
                        &qs[8 * ib32..8 * ib32 + 8],
                        qh[ib32],
                        &signs[4 * ib32..4 * ib32 + 4],
                        db1,
                        input,
                        blk * IQ3S_BLOCK_SIZE + ib32 * IQ3S_SUPER_BLOCK_SIZE,
                        n_cols,
                    );

                    // Second super-block.
                    let ib32b = ib32 + 1;
                    sum += gemv_superblock(
                        &qs[8 * ib32b..8 * ib32b + 8],
                        qh[ib32b],
                        &signs[4 * ib32b..4 * ib32b + 4],
                        db2,
                        input,
                        blk * IQ3S_BLOCK_SIZE + ib32b * IQ3S_SUPER_BLOCK_SIZE,
                        n_cols,
                    );

                    ib32 += 2;
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
        IQ3S_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ3S_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ3_S"
    }
}

/// Dequantize one super-block into `out` (32 floats).
///
/// `qs_sb` — the 8-byte qs slice for this super-block (already sliced to [8*ib32..8*ib32+8]).
/// `qh_byte` — the qh byte for this super-block (provides high bits for grid indices).
/// `signs_sb` — 4-byte slice: one sign-mask byte per group.
/// `db` — the scale value for this super-block.
fn dequant_superblock(qs_sb: &[u8], qh_byte: u8, signs_sb: &[u8], db: f32, out: &mut [f32]) {
    let qh = qh_byte as usize;

    for l in 0..IQ3S_GROUPS_PER_SUPER {
        // 9-bit grid indices for the two entries in this group.
        // C: qs[2*l+0] | ((qh << (8 - 2*l)) & 256)
        //    qs[2*l+1] | ((qh << (7 - 2*l)) & 256)
        let qs0 = qs_sb[2 * l] as usize;
        let qs1 = qs_sb[2 * l + 1] as usize;

        let shift0 = 8usize.saturating_sub(2 * l);
        let shift1 = 7usize.saturating_sub(2 * l);
        let idx1 = qs0 | ((qh << shift0) & 256);
        let idx2 = qs1 | ((qh << shift1) & 256);

        let grid1: [u8; 4] = IQ3S_GRID[idx1].to_le_bytes();
        let grid2: [u8; 4] = IQ3S_GRID[idx2].to_le_bytes();

        let sign_byte = signs_sb[l];

        let group_base = l * 8;
        for j in 0..4 {
            let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            out[group_base + j] = db * grid1[j] as f32 * sign1;

            let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            out[group_base + j + 4] = db * grid2[j] as f32 * sign2;
        }
    }
}

/// Accumulate dot-product for one super-block against `input`.
///
/// `qs_sb` — the 8-byte qs slice for this super-block (already sliced).
///
/// Returns the partial dot-product sum.
fn gemv_superblock(
    qs_sb: &[u8],
    qh_byte: u8,
    signs_sb: &[u8],
    db: f32,
    input: &[f32],
    col_base: usize,
    n_cols: usize,
) -> f32 {
    let qh = qh_byte as usize;
    let mut sum = 0.0_f32;

    for l in 0..IQ3S_GROUPS_PER_SUPER {
        let qs0 = qs_sb[2 * l] as usize;
        let qs1 = qs_sb[2 * l + 1] as usize;

        let shift0 = 8usize.saturating_sub(2 * l);
        let shift1 = 7usize.saturating_sub(2 * l);
        let idx1 = qs0 | ((qh << shift0) & 256);
        let idx2 = qs1 | ((qh << shift1) & 256);

        let grid1: [u8; 4] = IQ3S_GRID[idx1].to_le_bytes();
        let grid2: [u8; 4] = IQ3S_GRID[idx2].to_le_bytes();

        let sign_byte = signs_sb[l];
        let group_col = col_base + l * 8;

        for j in 0..4 {
            let col1 = group_col + j;
            if col1 < n_cols {
                let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                sum += db * grid1[j] as f32 * sign1 * input[col1];
            }

            let col2 = group_col + j + 4;
            if col2 < n_cols {
                let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                sum += db * grid2[j] as f32 * sign2 * input[col2];
            }
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Construct a minimal IQ3_S block with a given FP16 scale and zero data.
    fn make_zero_iq3_s_block(scale: f32) -> [u8; IQ3S_BLOCK_BYTES] {
        let mut block = [0u8; IQ3S_BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    #[test]
    fn test_iq3_s_metadata() {
        assert_eq!(Iq3SRef.name(), "IQ3_S");
        assert_eq!(Iq3SRef.block_size(), 256);
        assert_eq!(Iq3SRef.block_bytes(), 110);
    }

    #[test]
    fn test_dequant_block_buffer_too_small_block() {
        let small = [0u8; 50];
        let mut out = [0.0f32; 256];
        let result = Iq3SRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_buffer_too_small_output() {
        let block = make_zero_iq3_s_block(1.0);
        let mut out = [0.0f32; 100];
        let result = Iq3SRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_block_zero_scale() {
        let block = make_zero_iq3_s_block(0.0);
        let mut out = [1.0f32; 256];
        Iq3SRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_block_grid0_no_signs() {
        // All-zero block: qs=0, qh=0, signs=0, scales=0.
        // IQ3S_GRID[0] = 0x01010101 → magnitudes = [1,1,1,1].
        // signs=0 → all positive.
        // scales=0 → db = d * (1 + 2*0) = d * 1.0.
        // Expected: all weights = d * 1 * 1 = d.
        let d = 2.0_f32;
        let block = make_zero_iq3_s_block(d);
        let mut out = [0.0f32; 256];
        Iq3SRef.dequant_block(&block, &mut out).unwrap();

        let expected = d * 1.0 * 1.0;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_scale_nibble() {
        // scales[0] = 0x12 → low nibble = 2, high nibble = 1.
        // db1 = d * (1 + 2*2) = d * 5.0  (super-block 0)
        // db2 = d * (1 + 2*1) = d * 3.0  (super-block 1)
        // IQ3S_GRID[0] = [1,1,1,1].
        let d = 1.0_f32;
        let mut block = make_zero_iq3_s_block(d);
        block[IQ3S_SCALES_OFFSET] = 0x12;
        let mut out = [0.0f32; 256];
        Iq3SRef.dequant_block(&block, &mut out).unwrap();

        // Super-block 0 (weights 0..31): db1 * 1.0 = 5.0
        for (i, &v) in out.iter().enumerate().take(32) {
            assert!((v - 5.0).abs() < 1e-4, "out[{i}]={v}, expected 5.0");
        }
        // Super-block 1 (weights 32..63): db2 * 1.0 = 3.0
        for (i, &v) in out.iter().enumerate().take(64).skip(32) {
            assert!((v - 3.0).abs() < 1e-4, "out[{i}]={v}, expected 3.0");
        }
    }

    #[test]
    fn test_sign_mask_applied() {
        // signs[0] = 1 → bit 0 set → weight[0] of first group negated.
        // scales=0 → db = d, grid0 magnitudes = [1,1,1,1].
        let d = 1.0_f32;
        let mut block = make_zero_iq3_s_block(d);
        block[IQ3S_SIGNS_OFFSET] = 1;
        let mut out = [0.0f32; 256];
        Iq3SRef.dequant_block(&block, &mut out).unwrap();

        // db = d * 1.0 = 1.0, magnitude = 1
        // weight[0]: bit 0 set → negative
        assert!(
            (out[0] - (-1.0_f32)).abs() < 1e-5,
            "out[0]={}, expected -1.0",
            out[0]
        );
        // weight[1]: bit 1 not set → positive
        assert!(
            (out[1] - 1.0_f32).abs() < 1e-5,
            "out[1]={}, expected 1.0",
            out[1]
        );
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq3S));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        let d = 1.0_f32;
        let block = make_zero_iq3_s_block(d);

        let mut dequant = [0.0f32; 256];
        Iq3SRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq3S,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq3SRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
