//! AVX-512 accelerated IQ3_S quantization kernel.
//!
//! Mirrors `simd/avx2/iq3_s.rs` using 512-bit ZMM accumulators.
//!
//! Block layout (110 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:    FP16 scale `d`
//! - bytes[2..66]:   `qs[64]` — grid base indices, 8 per super-block
//! - bytes[66..74]:  `qh[8]`  — 1 byte per super-block (high bits)
//! - bytes[74..106]: `signs[32]` — 4 bytes per super-block
//! - bytes[106..110]: `scales[4]` — nibble pairs (2 per byte)
//!
//! Scale formula for pair (ib32, ib32+1):
//!   db1 = d * (1 + 2 * (scales[pair] & 0xF))
//!   db2 = d * (1 + 2 * (scales[pair] >> 4))

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
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
/// Groups per super-block.
const GROUPS_PER_SUPER: usize = 4;

const QS_OFFSET: usize = 2;
const QS_BYTES: usize = 64;
const QH_OFFSET: usize = 66;
const QH_BYTES: usize = 8;
const SIGNS_OFFSET: usize = 74;
const SIGNS_BYTES: usize = 32;
const SCALES_OFFSET: usize = 106;

/// AVX-512 accelerated IQ3_S kernel.
pub struct Iq3SAvx512;

impl QuantKernel for Iq3SAvx512 {
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
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return scalar_dequant_block(block, output);
        }
        // SAFETY: bounds verified; avx512f confirmed.
        unsafe { dequant_block_avx512(block, output) }
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

        if !std::arch::is_x86_feature_detected!("avx512f") {
            return scalar_gemv(
                &quant_matrix.data,
                input,
                output,
                n_rows,
                n_cols,
                blocks_per_row,
                row_bytes,
            );
        }

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: bounds checked; avx512f confirmed.
            *out = unsafe {
                gemv_row_avx512(
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
        "IQ3_S"
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

fn scalar_dequant_block(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    use crate::reference::iq3_s::Iq3SRef;
    Iq3SRef.dequant_block(block, output)
}

fn scalar_gemv(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
    blocks_per_row: usize,
    row_bytes: usize,
) -> QuantResult<()> {
    for (row, out) in output.iter_mut().enumerate().take(n_rows) {
        let row_start = row * row_bytes;
        let mut sum = 0.0f32;
        for blk in 0..blocks_per_row {
            let bo = row_start + blk * BLOCK_BYTES;
            let block = &data[bo..bo + BLOCK_BYTES];
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
            let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
            let signs = &block[SIGNS_OFFSET..SIGNS_OFFSET + SIGNS_BYTES];
            let scales = &block[SCALES_OFFSET..BLOCK_BYTES];
            let mut ib32 = 0usize;
            let col = blk * BLOCK_SIZE;
            while ib32 < N_SUPERBLOCKS {
                let pair = ib32 / 2;
                let scale_byte = scales[pair];
                let dbs = [
                    d * (1.0 + 2.0 * (scale_byte & 0xF) as f32),
                    d * (1.0 + 2.0 * (scale_byte >> 4) as f32),
                ];
                for (sb_idx, &db) in dbs.iter().enumerate() {
                    let ib = ib32 + sb_idx;
                    for l in 0..GROUPS_PER_SUPER {
                        let idx1 = (qs[8 * ib + 2 * l] as usize)
                            | (((qh[ib] as usize) << (8 - 2 * l)) & 0x100);
                        let idx2 = (qs[8 * ib + 2 * l + 1] as usize)
                            | (((qh[ib] as usize) << (7 - 2 * l)) & 0x100);
                        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                        let sign_byte = signs[4 * ib + l];
                        let base_col = col + ib * SUPER_BLOCK_SIZE + l * 8;
                        for j in 0..4 {
                            let c1 = base_col + j;
                            let c2 = base_col + 4 + j;
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
                            if c1 < n_cols {
                                sum += db * s1 * grid1[j] as f32 * input[c1];
                            }
                            if c2 < n_cols {
                                sum += db * s2 * grid2[j] as f32 * input[c2];
                            }
                        }
                    }
                }
                ib32 += 2;
            }
            let _ = col;
        }
        *out = sum;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// AVX-512 kernels
// ---------------------------------------------------------------------------

/// Decode one IQ3_S super-block into `output[0..32]` using AVX-512.
///
/// # Safety
/// CPU must support `avx512f`. `output.len() >= 32`.
#[target_feature(enable = "avx512f")]
unsafe fn decode_superblock_avx512(
    qs_sb: &[u8],
    qh_byte: u8,
    signs_sb: &[u8],
    db: f32,
    output: &mut [f32],
) {
    let mut vals = [0.0f32; SUPER_BLOCK_SIZE];

    for l in 0..GROUPS_PER_SUPER {
        let idx1 = (qs_sb[2 * l] as usize) | (((qh_byte as usize) << (8 - 2 * l)) & 0x100);
        let idx2 = (qs_sb[2 * l + 1] as usize) | (((qh_byte as usize) << (7 - 2 * l)) & 0x100);
        let grid1 = IQ3S_GRID[idx1].to_le_bytes();
        let grid2 = IQ3S_GRID[idx2].to_le_bytes();
        let sign_byte = signs_sb[l];

        let base = l * 8;
        for j in 0..4 {
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
            vals[base + j] = s1 * grid1[j] as f32;
            vals[base + 4 + j] = s2 * grid2[j] as f32;
        }
    }

    // AVX-512: multiply all 32 floats by `db` in two 16-wide passes.
    let db_vec = _mm512_set1_ps(db);
    // SAFETY: vals has exactly 32 elements; output.len() >= 32.
    let src0 = _mm512_loadu_ps(vals.as_ptr());
    let src1 = _mm512_loadu_ps(vals.as_ptr().add(16));
    let dst0 = _mm512_mul_ps(src0, db_vec);
    let dst1 = _mm512_mul_ps(src1, db_vec);
    _mm512_storeu_ps(output.as_mut_ptr(), dst0);
    _mm512_storeu_ps(output.as_mut_ptr().add(16), dst1);
}

/// Dequantize one IQ3_S block using AVX-512.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES`, `output.len() >= BLOCK_SIZE`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
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

        decode_superblock_avx512(
            &qs[8 * ib32..8 * ib32 + 8],
            qh[ib32],
            &signs[4 * ib32..4 * ib32 + 4],
            db1,
            &mut output[ib32 * SUPER_BLOCK_SIZE..(ib32 + 1) * SUPER_BLOCK_SIZE],
        );

        let ib32b = ib32 + 1;
        decode_superblock_avx512(
            &qs[8 * ib32b..8 * ib32b + 8],
            qh[ib32b],
            &signs[4 * ib32b..4 * ib32b + 4],
            db2,
            &mut output[ib32b * SUPER_BLOCK_SIZE..(ib32b + 1) * SUPER_BLOCK_SIZE],
        );

        ib32 += 2;
    }
}

/// GEMV for one row using AVX-512 accumulation.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`, `input.len() >= n_cols`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm512_setzero_ps();
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
                    for j in 0..4 {
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

                // Dot-product with 16-wide FMA across the 32 weights in two passes.
                let w_col_base = col + ib * SUPER_BLOCK_SIZE;

                // First 16 weights.
                let w_off0 = w_col_base;
                if w_off0 + 16 <= n_cols {
                    let wv = _mm512_loadu_ps(vals.as_ptr());
                    // SAFETY: w_off0 + 16 <= n_cols <= input.len()
                    let iv = _mm512_loadu_ps(input.as_ptr().add(w_off0));
                    acc = _mm512_fmadd_ps(wv, iv, acc);
                } else {
                    for k in 0..16usize {
                        let c = w_off0 + k;
                        if c < n_cols {
                            acc = _mm512_mask_add_ps(
                                acc,
                                0x0001u16,
                                acc,
                                _mm512_set1_ps(vals[k] * input[c]),
                            );
                        }
                    }
                }

                // Second 16 weights.
                let w_off1 = w_col_base + 16;
                if w_off1 + 16 <= n_cols {
                    let wv = _mm512_loadu_ps(vals.as_ptr().add(16));
                    // SAFETY: w_off1 + 16 <= n_cols
                    let iv = _mm512_loadu_ps(input.as_ptr().add(w_off1));
                    acc = _mm512_fmadd_ps(wv, iv, acc);
                } else {
                    for k in 0..16usize {
                        let c = w_off1 + k;
                        if c < n_cols {
                            acc = _mm512_mask_add_ps(
                                acc,
                                0x0001u16,
                                acc,
                                _mm512_set1_ps(vals[16 + k] * input[c]),
                            );
                        }
                    }
                }
            }

            ib32 += 2;
        }

        col += BLOCK_SIZE;
    }

    hsum_f32_avx512(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::iq3_s::Iq3SRef;

    fn make_zero_block(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_f16 = half::f16::from_f32(d);
        let [d0, d1] = d_f16.to_le_bytes();
        block[0] = d0;
        block[1] = d1;
        block
    }

    fn make_varied_block(d: f32) -> Vec<u8> {
        let mut block = make_zero_block(d);
        block[SCALES_OFFSET] = 0x23;
        block[SCALES_OFFSET + 1] = 0x45;
        block[SIGNS_OFFSET] = 0xAA;
        block[SIGNS_OFFSET + 4] = 0x55;
        block
    }

    #[test]
    fn avx512_iq3_s_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_varied_block(0.5);
        let mut out_avx512 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ref = vec![0.0f32; BLOCK_SIZE];
        Iq3SAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Iq3SRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant");
        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn avx512_iq3_s_matvec_q8_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_varied_block(0.75);
        let n_rows = 4usize;
        let n_cols = BLOCK_SIZE;
        let data: Vec<u8> = block
            .iter()
            .cloned()
            .cycle()
            .take(n_rows * BLOCK_BYTES)
            .collect();

        let tensor_avx512 = crate::types::QuantTensor::new(
            data.clone(),
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Iq3S,
        );
        let tensor_ref = crate::types::QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Iq3S,
        );

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Iq3SAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Iq3SRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-3,
                "gemv mismatch row {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn avx512_iq3_s_kernel_metadata() {
        assert_eq!(Iq3SAvx512.name(), "IQ3_S");
        assert_eq!(Iq3SAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Iq3SAvx512.block_bytes(), BLOCK_BYTES);
    }
}
