//! AVX2+FMA accelerated IQ3_XXS quantization kernel.
//!
//! IQ3_XXS block layout (98 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs_grid[64]` — grid base indices (2 per 8-weight group)
//! - bytes[66..98]: `qs_signs[32]` — packed scale + sign data (4 bytes per super-block)
//!
//! The grid/sign decode is inherently scalar; AVX2 is used for the final
//! FMA dot product in gemv, yielding ~4× throughput on that phase.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::simd::avx2::util::hsum_f32_avx;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ3_XXS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_XXS block: 2 (FP16 d) + 96 (3 * QK_K/8).
pub const BLOCK_BYTES: usize = 98;
/// Number of super-blocks per IQ3_XXS block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Number of weight groups per super-block.
const GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset within qs region where sign/scale data begins.
const SIGNS_OFFSET: usize = 64;

/// AVX2+FMA accelerated IQ3_XXS kernel.
pub struct Iq3XxsAvx2;

impl QuantKernel for Iq3XxsAvx2 {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: BLOCK_SIZE,
                available: output.len(),
            });
        }
        dequant_block_scalar(block, output)
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

        let blocks_per_row = n_cols.div_ceil(BLOCK_SIZE);
        let row_bytes = blocks_per_row * BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked; avx2+fma guaranteed by dispatcher.
            *out = unsafe {
                gemv_row_avx2(
                    &quant_matrix.data[row_start..row_start + row_bytes],
                    input,
                    blocks_per_row,
                    n_cols,
                )
            };
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
        BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ3_XXS_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Scalar decode (grid + sign table lookups don't vectorize well)
// ---------------------------------------------------------------------------

/// Decode one IQ3_XXS block (256 weights) into `output` using scalar arithmetic.
fn dequant_block_scalar(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..BLOCK_BYTES];
    let qs_grid = &qs[..SIGNS_OFFSET];
    let qs_signs = &qs[SIGNS_OFFSET..];

    for ib32 in 0..N_SUPERBLOCKS {
        let signs_base = ib32 * 4;
        if signs_base + 3 >= qs_signs.len() {
            return Err(QuantError::BufferTooSmall {
                needed: signs_base + 4,
                available: qs_signs.len(),
            });
        }
        let aux32 = u32::from_le_bytes([
            qs_signs[signs_base],
            qs_signs[signs_base + 1],
            qs_signs[signs_base + 2],
            qs_signs[signs_base + 3],
        ]);

        let scale_bits = (aux32 >> 28) as f32;
        let db = d * (0.5 + scale_bits) * 0.5;

        let grid_base = ib32 * 8;
        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let g1 = qs_grid[grid_base + 2 * l] as usize;
            let g2 = qs_grid[grid_base + 2 * l + 1] as usize;
            let mags1: [u8; 4] = IQ3XXS_GRID[g1].to_le_bytes();
            let mags2: [u8; 4] = IQ3XXS_GRID[g2].to_le_bytes();

            let sign_idx = ((aux32 >> (7 * l)) & 0x7F) as usize;
            let sign_byte = KSIGNS_IQ2XS[sign_idx];

            let group_base = weight_base + l * WEIGHTS_PER_GROUP;
            for j in 0..4 {
                let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                output[group_base + j] = db * mags1[j] as f32 * sign1;

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

// ---------------------------------------------------------------------------
// AVX2+FMA gemv
// ---------------------------------------------------------------------------

/// Compute one row of gemv for IQ3_XXS: scalar decode then AVX2 dot product.
///
/// # Safety
/// All slice bounds must be validated by the caller.
/// `avx2` and `fma` CPU features must be available.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut scalar_tail = 0.0f32;
    let mut buf = [0.0f32; BLOCK_SIZE];

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = row_data.get_unchecked(block_offset..block_offset + BLOCK_BYTES);
        let col_offset = blk * BLOCK_SIZE;
        let remaining = n_cols.saturating_sub(col_offset).min(BLOCK_SIZE);

        // Scalar decode — grid lookups cannot be vectorized.
        let _ = dequant_block_scalar(block, &mut buf);

        let full_chunks = remaining / 8;
        for chunk in 0..full_chunks {
            let w = _mm256_loadu_ps(buf.as_ptr().add(chunk * 8));
            let x = _mm256_loadu_ps(input.as_ptr().add(col_offset + chunk * 8));
            acc = _mm256_fmadd_ps(w, x, acc);
        }
        for j in (full_chunks * 8)..remaining {
            scalar_tail += buf[j] * *input.get_unchecked(col_offset + j);
        }
    }

    hsum_f32_avx(acc) + scalar_tail
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::iq3_xxs::Iq3XxsRef;

    /// Build a minimal valid IQ3_XXS block with scale `d` and all data bytes = 0.
    fn make_zero_block(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_f16 = half::f16::from_f32(d);
        let bytes = d_f16.to_le_bytes();
        block[0] = bytes[0];
        block[1] = bytes[1];
        block
    }

    #[test]
    fn zero_block_matches_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_zero_block(1.0);

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq3XxsRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq3XxsAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn nonzero_block_matches_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut block = make_zero_block(0.25);
        // Set non-trivial sign/scale data.
        let signs_start = 2 + SIGNS_OFFSET;
        block[signs_start] = 0xAB;
        block[signs_start + 1] = 0xCD;
        block[signs_start + 4] = 0x12;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq3XxsRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq3XxsAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn gemv_matches_dequant_dot_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_zero_block(0.5);

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq3XxsAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let input = vec![1.0f32; BLOCK_SIZE];
        let tensor = crate::types::QuantTensor::new(
            block.clone(),
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq3Xxs,
        );
        let mut output = vec![0.0f32; 1];
        Iq3XxsAvx2.gemv(&tensor, &input, &mut output).unwrap();

        assert!(
            (output[0] - expected).abs() < 1e-4,
            "gemv={}, expected={}",
            output[0],
            expected
        );
    }
}
