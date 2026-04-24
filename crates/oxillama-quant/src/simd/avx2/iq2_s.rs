//! AVX2+FMA accelerated IQ2_S quantization kernel.
//!
//! IQ2_S block layout (82 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[64]` — 64 bytes split into base grid indices (32) and signs (32)
//! - bytes[66..74]: `qh[8]`  — high bits for grid indices (1 byte per super-block)
//! - bytes[74..82]: `scales[8]` — one nibble-pair scale per super-block
//!
//! The 10-bit grid/sign decode is inherently scalar; AVX2 is used for the
//! FMA dot product in gemv.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2S_GRID, KMASK_IQ2XS};
use crate::simd::avx2::util::hsum_f32_avx;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_S: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_S block: 2 + 64 + 8 + 8 = 82.
pub const BLOCK_BYTES: usize = 82;
/// Number of super-blocks per IQ2_S block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Number of weight groups per super-block.
const GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;

// Byte offsets within the block.
const QS_OFFSET: usize = 2;
const QS_BYTES: usize = 64;
const SIGNS_IN_QS: usize = 32;
const QH_OFFSET: usize = 66;
const SCALES_OFFSET: usize = 74;

/// AVX2+FMA accelerated IQ2_S kernel.
pub struct Iq2SAvx2;

impl QuantKernel for Iq2SAvx2 {
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
        "IQ2_S_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Scalar decode
// ---------------------------------------------------------------------------

/// Decode one IQ2_S block (256 weights) into `output` using scalar arithmetic.
fn dequant_block_scalar(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
    let qs_base = &qs[..SIGNS_IN_QS];
    let qs_signs = &qs[SIGNS_IN_QS..];
    let qh = &block[QH_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let scale_byte = scales[ib32];
        let db0 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
        let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
        let qh_byte = qh[ib32] as u16;

        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let base_idx = qs_base[4 * ib32 + l] as u16;
            let shift = 8u16.saturating_sub(2 * l as u16);
            let high_bit = (qh_byte << shift) & 0x300;
            let grid_idx = (base_idx | high_bit) as usize;

            let signs_byte = qs_signs[4 * ib32 + l];
            let magnitudes: [u8; 8] = IQ2S_GRID[grid_idx].to_le_bytes();

            let dl = if l < 2 { db0 } else { db1 };
            let group_base = weight_base + l * WEIGHTS_PER_GROUP;

            for j in 0..WEIGHTS_PER_GROUP {
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

// ---------------------------------------------------------------------------
// AVX2+FMA gemv
// ---------------------------------------------------------------------------

/// Compute one row of gemv for IQ2_S: scalar decode then AVX2 dot product.
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

        // Scalar decode — 10-bit grid + sign lookups don't vectorize well.
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
    use crate::reference::iq2_s::Iq2SRef;

    /// Build a minimal valid IQ2_S block with scale `d` and all data bytes = 0.
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
        Iq2SRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq2SAvx2.dequant_block(&block, &mut avx_out).unwrap();

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
        // Set non-trivial qh and scale data to exercise high-bit path.
        block[QH_OFFSET] = 0xAA;
        block[SCALES_OFFSET] = 0x5F;
        block[SCALES_OFFSET + 1] = 0x3C;
        // Set a non-zero sign byte.
        block[QS_OFFSET + SIGNS_IN_QS] = 0x0F;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq2SRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq2SAvx2.dequant_block(&block, &mut avx_out).unwrap();

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
        Iq2SAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let input = vec![1.0f32; BLOCK_SIZE];
        let tensor = crate::types::QuantTensor::new(
            block.clone(),
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq2S,
        );
        let mut output = vec![0.0f32; 1];
        Iq2SAvx2.gemv(&tensor, &input, &mut output).unwrap();

        assert!(
            (output[0] - expected).abs() < 1e-4,
            "gemv={}, expected={}",
            output[0],
            expected
        );
    }
}
