//! AVX2+FMA accelerated IQ3_S quantization kernel.
//!
//! IQ3_S block layout (110 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d` (little-endian)
//! - bytes[2..66]:   `qs[64]` — 64 bytes of grid-base indices, 8 per super-block
//! - bytes[66..74]:  `qh[8]`  — 8 bytes of high bits (1 byte per super-block)
//! - bytes[74..106]: `signs[32]` — 32 bytes of per-weight sign bits, 4 per super-block
//! - bytes[106..110]: `scales[4]` — 4 bytes of packed nibble scales (2 per byte, pairs of super-blocks)
//!
//! For each pair (ib32, ib32+1):
//! - `db1 = d * (1 + 2 * (scales[pair] & 0xF))`
//! - `db2 = d * (1 + 2 * (scales[pair] >> 4))`
//!
//! The AVX2 kernel accelerates the inner-most multiply loop using 8-wide
//! `_mm256_mul_ps` lanes, after scalar grid/sign lookups.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ3_S: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ3_S block: 2 + 64 + 8 + 32 + 4 = 110.
pub const BLOCK_BYTES: usize = 110;
/// Number of super-blocks per IQ3_S block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Groups per super-block (4 groups × 8 weights = 32).
const GROUPS_PER_SUPER: usize = 4;

// Byte offsets within a block.
const QS_OFFSET: usize = 2;
const QS_BYTES: usize = 64;
const QH_OFFSET: usize = 66;
const QH_BYTES: usize = 8;
const SIGNS_OFFSET: usize = 74;
const SIGNS_BYTES: usize = 32;
const SCALES_OFFSET: usize = 106;

/// AVX2+FMA accelerated IQ3_S kernel.
pub struct Iq3SAvx2;

impl QuantKernel for Iq3SAvx2 {
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
        // SAFETY: bounds verified above; avx2+fma guaranteed by dispatcher.
        unsafe { dequant_block_avx2(block, output) }
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
        "IQ3_S_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Inner AVX2 kernels (unsafe, target_feature guaranteed by dispatcher)
// ---------------------------------------------------------------------------

/// Decode one IQ3_S super-block (32 weights) into `output`.
///
/// # Safety
/// Caller must ensure `avx2` is available.  All slice bounds are pre-verified.
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_superblock(
    qs_sb: &[u8],    // 8 bytes: qs for this super-block
    qh_byte: u8,     // 1 byte:  qh for this super-block
    signs_sb: &[u8], // 4 bytes: signs for this super-block (1 per group)
    db: f32,
    output: &mut [f32], // exactly 32 elements
) {
    // Precompute sign-corrected, grid-looked-up floats (8 per group × 4 groups),
    // then batch-multiply by `db` using 8-wide AVX2.
    let mut vals = [0.0f32; SUPER_BLOCK_SIZE];

    for l in 0..GROUPS_PER_SUPER {
        // Reconstruct 9-bit grid indices.
        let idx1 = (qs_sb[2 * l] as usize) | (((qh_byte as usize) << (8 - 2 * l)) & 0x100);
        let idx2 = (qs_sb[2 * l + 1] as usize) | (((qh_byte as usize) << (7 - 2 * l)) & 0x100);

        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
        let sign_byte = signs_sb[l];

        let base = l * 8;
        for j in 0..4usize {
            let sign1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            let sign2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            vals[base + j] = sign1 * grid1[j] as f32;
            vals[base + 4 + j] = sign2 * grid2[j] as f32;
        }
    }

    // AVX2: multiply all 32 floats by `db` in 8-wide chunks.
    let db_vec = _mm256_set1_ps(db);
    for chunk in 0..(SUPER_BLOCK_SIZE / 8) {
        let src = _mm256_loadu_ps(vals.as_ptr().add(chunk * 8));
        let dst = _mm256_mul_ps(src, db_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(chunk * 8), dst);
    }
}

/// Dequantize a full IQ3_S block (256 weights).
///
/// # Safety
/// `block.len() >= BLOCK_BYTES` and `output.len() >= BLOCK_SIZE` must hold.
/// CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
    let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
    let signs = &block[SIGNS_OFFSET..SIGNS_OFFSET + SIGNS_BYTES];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    let mut ib32 = 0usize;
    while ib32 < N_SUPERBLOCKS {
        let pair = ib32 / 2;
        let scale_byte = scales[pair];
        let db1 = d * (1.0 + 2.0 * (scale_byte & 0xF) as f32);
        let db2 = d * (1.0 + 2.0 * (scale_byte >> 4) as f32);

        decode_superblock(
            &qs[8 * ib32..8 * ib32 + 8],
            qh[ib32],
            &signs[4 * ib32..4 * ib32 + 4],
            db1,
            &mut output[ib32 * SUPER_BLOCK_SIZE..(ib32 + 1) * SUPER_BLOCK_SIZE],
        );

        let ib32b = ib32 + 1;
        decode_superblock(
            &qs[8 * ib32b..8 * ib32b + 8],
            qh[ib32b],
            &signs[4 * ib32b..4 * ib32b + 4],
            db2,
            &mut output[ib32b * SUPER_BLOCK_SIZE..(ib32b + 1) * SUPER_BLOCK_SIZE],
        );

        ib32 += 2;
    }
}

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
        let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
        let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
        let signs = &block[SIGNS_OFFSET..SIGNS_OFFSET + SIGNS_BYTES];
        let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

        let mut ib32 = 0usize;
        while ib32 < N_SUPERBLOCKS {
            let pair = ib32 / 2;
            let scale_byte = scales[pair];

            for (sb_idx, &db_raw) in [scale_byte & 0xF, scale_byte >> 4].iter().enumerate() {
                let ib = ib32 + sb_idx;
                let db = d * (1.0 + 2.0 * db_raw as f32);

                let mut vals = [0.0f32; SUPER_BLOCK_SIZE];
                for l in 0..GROUPS_PER_SUPER {
                    let idx1 = (qs[8 * ib + 2 * l] as usize)
                        | (((qh[ib] as usize) << (8 - 2 * l)) & 0x100);
                    let idx2 = (qs[8 * ib + 2 * l + 1] as usize)
                        | (((qh[ib] as usize) << (7 - 2 * l)) & 0x100);

                    let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                    let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                    let sign_byte = signs[4 * ib + l];

                    let base = l * 8;
                    for j in 0..4usize {
                        let s1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        let s2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        vals[base + j] = db * s1 * grid1[j] as f32;
                        vals[base + 4 + j] = db * s2 * grid2[j] as f32;
                    }
                }

                // Dot-product with 8-wide FMA.
                for chunk in 0..(SUPER_BLOCK_SIZE / 8) {
                    let w_off = col + ib * SUPER_BLOCK_SIZE + chunk * 8;
                    if w_off + 8 > n_cols {
                        // scalar tail
                        for k in 0..8usize {
                            let c = w_off + k;
                            if c < n_cols {
                                acc = _mm256_add_ps(
                                    acc,
                                    _mm256_set1_ps(vals[chunk * 8 + k] * input[c]),
                                );
                            }
                        }
                    } else {
                        let wv = _mm256_loadu_ps(vals.as_ptr().add(chunk * 8));
                        let iv = _mm256_loadu_ps(input.as_ptr().add(w_off));
                        acc = _mm256_fmadd_ps(wv, iv, acc);
                    }
                }
            }

            ib32 += 2;
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
    use crate::reference::iq3_s::Iq3SRef;
    use crate::traits::QuantKernel;

    /// Build a minimal zero IQ3_S block with a given FP16 `d`.
    fn make_zero_block(d: f32) -> Vec<u8> {
        let d_f16 = half::f16::from_f32(d);
        let [d0, d1] = d_f16.to_le_bytes();
        let mut block = vec![0u8; BLOCK_BYTES];
        block[0] = d0;
        block[1] = d1;
        block
    }

    #[test]
    fn avx2_matches_reference_zero_block() {
        if !is_x86_feature_detected!("avx2") {
            return; // skip on machines without AVX2
        }
        let d = 1.5_f32;
        let block = make_zero_block(d);

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq3SRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq3SAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_nonzero_scales() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.5_f32;
        let mut block = make_zero_block(d);
        // scales[0] = 0x23 → pair 0: low nibble 3, high nibble 2
        block[SCALES_OFFSET] = 0x23;
        // signs[0] = 0xAA → alternating signs for first group
        block[SIGNS_OFFSET] = 0xAA;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq3SRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq3SAvx2.dequant_block(&block, &mut avx_out).unwrap();

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
        block[SCALES_OFFSET] = 0x12;
        block[SIGNS_OFFSET] = 0x55;

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq3SAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.clone(),
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq3S,
        );
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq3SAvx2.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-2,
            "gemv={}, expected={}",
            out[0],
            expected
        );
    }
}
