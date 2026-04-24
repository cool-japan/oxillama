//! AVX2+FMA accelerated IQ1_S quantization kernel.
//!
//! IQ1_S block layout (50 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d` (little-endian)
//! - bytes[2..34]:   `qs[32]` — 32 bytes of lower 8-bit grid indices
//! - bytes[34..50]:  `qh[8]` — 8 × u16 sub-block headers
//!
//! The dequantization step is inherently scalar (complex bit extraction),
//! so this kernel uses scalar decode followed by AVX2 FMA in the gemv step.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq1s_grid::IQ1S_GRID;
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ1_S: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_S block: 2 (FP16 d) + 32 (qs) + 16 (qh as 8×u16).
pub const BLOCK_BYTES: usize = 50;
/// Byte offset where `qs` begins.
const QS_OFFSET: usize = 2;
/// Byte offset where `qh` (8 × u16 LE) begins.
const QH_OFFSET: usize = 34;
/// Number of sub-blocks per IQ1_S block.
const N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
const SUB_BLOCK_SIZE: usize = BLOCK_SIZE / N_SUBBLOCKS; // 32
/// Number of groups per sub-block (each group = 8 weights).
const GROUPS_PER_SUB: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant.
const DELTA: f32 = 0.125;

/// AVX2+FMA accelerated IQ1_S kernel.
///
/// Dequantization is scalar (bit extraction is inherently serial).
/// The gemv step uses 8-wide AVX2 FMA for the dot-product accumulation.
pub struct Iq1SAvx2;

impl QuantKernel for Iq1SAvx2 {
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
            // SAFETY: bounds checked above; avx2+fma guaranteed by dispatcher.
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
        "IQ1_S_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Scalar dequant (exact copy of reference logic, inlined here for gemv)
// ---------------------------------------------------------------------------

fn dequant_block_scalar(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[QS_OFFSET..QH_OFFSET];
    let qh_bytes = &block[QH_OFFSET..BLOCK_BYTES];

    for ib in 0..N_SUBBLOCKS {
        let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);
        let scale_bits = ((qh_val >> 12) & 0x7) as f32;
        let dl = d * (2.0 * scale_bits + 1.0);
        let delta = if qh_val & 0x8000 != 0 { -DELTA } else { DELTA };

        let qs_base = ib * GROUPS_PER_SUB;
        let output_base = ib * SUB_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUB {
            let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
            let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);
            let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

            let group_base = output_base + l * WEIGHTS_PER_GROUP;
            for j in 0..WEIGHTS_PER_GROUP {
                let gv = grid_raw[j] as i8 as f32;
                output[group_base + j] = dl * (gv + delta);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// AVX2 inner kernel
// ---------------------------------------------------------------------------

/// GEMV for one row: dot-product of dequantized row with `input`.
///
/// # Safety
/// Bounds checking is done by the caller. CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut col = 0usize;

    for blk in 0..blocks_per_row {
        let block = &row_data[blk * BLOCK_BYTES..(blk + 1) * BLOCK_BYTES];
        let d = f16_to_f32(block);
        let qs = &block[QS_OFFSET..QH_OFFSET];
        let qh_bytes = &block[QH_OFFSET..BLOCK_BYTES];

        for ib in 0..N_SUBBLOCKS {
            let qh_val = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);
            let scale_bits = ((qh_val >> 12) & 0x7) as f32;
            let dl = d * (2.0 * scale_bits + 1.0);
            let delta = if qh_val & 0x8000 != 0 { -DELTA } else { DELTA };

            let qs_base = ib * GROUPS_PER_SUB;
            let delta_vec = _mm256_set1_ps(delta);
            let dl_vec = _mm256_set1_ps(dl);

            for l in 0..GROUPS_PER_SUB {
                let upper_bits = ((qh_val >> (3 * l as u16)) & 0x7) as usize;
                let grid_idx = (qs[qs_base + l] as usize) | (upper_bits << 8);
                let grid_raw = IQ1S_GRID[grid_idx].to_le_bytes();

                // Convert 8 i8 grid bytes → f32, add delta, multiply by dl.
                let mut vals = [0.0f32; WEIGHTS_PER_GROUP];
                for j in 0..WEIGHTS_PER_GROUP {
                    vals[j] = grid_raw[j] as i8 as f32;
                }

                // AVX2: (gv + delta) * dl for all 8 elements.
                let gv_vec = _mm256_loadu_ps(vals.as_ptr());
                let scaled = _mm256_mul_ps(_mm256_add_ps(gv_vec, delta_vec), dl_vec);

                let w_off = col + ib * SUB_BLOCK_SIZE + l * WEIGHTS_PER_GROUP;
                if w_off + WEIGHTS_PER_GROUP > n_cols {
                    // Scalar tail for the last partial group.
                    let mut scaled_arr = [0.0f32; WEIGHTS_PER_GROUP];
                    _mm256_storeu_ps(scaled_arr.as_mut_ptr(), scaled);
                    for k in 0..WEIGHTS_PER_GROUP {
                        let c = w_off + k;
                        if c < n_cols {
                            acc = _mm256_add_ps(acc, _mm256_set1_ps(scaled_arr[k] * input[c]));
                        }
                    }
                } else {
                    let iv = _mm256_loadu_ps(input.as_ptr().add(w_off));
                    acc = _mm256_fmadd_ps(scaled, iv, acc);
                }
            }
        }

        col += BLOCK_SIZE;
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::iq1_s::Iq1SRef;
    use crate::traits::QuantKernel;

    fn make_zero_block(scale: f32) -> Vec<u8> {
        let d_f16 = half::f16::from_f32(scale);
        let [d0, d1] = d_f16.to_le_bytes();
        let mut block = vec![0u8; BLOCK_BYTES];
        block[0] = d0;
        block[1] = d1;
        block
    }

    #[test]
    fn avx2_matches_reference_zero_block() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 1.5_f32;
        let block = make_zero_block(d);

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq1SRef.dequant_block(&block, &mut ref_out).unwrap();
        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq1SAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_nonzero_qh() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.5_f32;
        let mut block = make_zero_block(d);
        // qh[0]: scale_bits = 3 (bits 12-14 = 011), delta positive (bit 15 = 0).
        // Encode: 0x3000 → scale_bits = 3, bit 15 = 0.
        block[QH_OFFSET] = 0x00;
        block[QH_OFFSET + 1] = 0x30;
        // qh[1]: scale_bits = 1, delta negative (bit 15 = 1) → 0x9000.
        block[QH_OFFSET + 2] = 0x00;
        block[QH_OFFSET + 3] = 0x90u8;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq1SRef.dequant_block(&block, &mut ref_out).unwrap();
        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq1SAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn gemv_matches_dequant_dot_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        // Set scale_bits = 2 for sub-block 0.
        block[QH_OFFSET + 1] = 0x20;

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq1SAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block,
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq1S,
        );
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut got = vec![0.0f32; 1];
        Iq1SAvx2.gemv(&tensor, &input, &mut got).unwrap();

        assert!(
            (got[0] - expected).abs() < 1e-2,
            "gemv={}, dequant_sum={}",
            got[0],
            expected
        );
    }
}
