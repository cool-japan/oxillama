//! AVX-512 accelerated IQ2_XS quantization kernel.
//!
//! Mirrors `simd/avx2/iq2_xs.rs` using 512-bit ZMM accumulators and the
//! single-instruction `_mm512_reduce_add_ps` horizontal reduction.
//!
//! Block layout (74 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[32]` — 32 × u16 LE (9-bit grid idx | 7-bit sign selector)
//! - bytes[66..74]: `scales[8]` — one byte per super-block
//!   low nibble → db0 (groups 0-1), high nibble → db1 (groups 2-3).
//!   `db = d * (0.5 + nibble) * 0.25`

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XS block: 2 (FP16 d) + 64 (32 × u16 qs) + 8 (scales).
pub const BLOCK_BYTES: usize = 74;
/// Number of super-blocks.
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// u16 qs entries per super-block.
const QS_PER_SUPER: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of qs within block.
const QS_OFFSET: usize = 2;
/// Byte offset of scales within block.
const SCALES_OFFSET: usize = 66;

/// AVX-512 accelerated IQ2_XS kernel.
pub struct Iq2XsAvx512;

impl QuantKernel for Iq2XsAvx512 {
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
        "IQ2_XS"
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

fn scalar_dequant_block(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    use crate::reference::iq2_xs::Iq2XsRef;
    Iq2XsRef.dequant_block(block, output)
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
            let qs_bytes = &block[QS_OFFSET..SCALES_OFFSET];
            let scales = &block[SCALES_OFFSET..BLOCK_BYTES];
            let col_block_base = blk * BLOCK_SIZE;
            for ib32 in 0..N_SUPERBLOCKS {
                let scale_byte = scales[ib32];
                let db0 = d * (0.5 + (scale_byte & 0x0f) as f32) * 0.25;
                let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                for l in 0..QS_PER_SUPER {
                    let byte_pos = 8 * ib32 + 2 * l;
                    let qs_val =
                        u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;
                    let grid_idx = qs_val & 511;
                    let sign_idx = qs_val >> 9;
                    let dl = if l < 2 { db0 } else { db1 };
                    let col = col_block_base + ib32 * SUPER_BLOCK_SIZE + l * WEIGHTS_PER_GROUP;
                    let mags = IQ2XS_GRID[grid_idx].to_le_bytes();
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];
                    for j in 0..WEIGHTS_PER_GROUP {
                        let c = col + j;
                        if c < n_cols {
                            let mag = mags[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            sum += dl * mag * sign * input[c];
                        }
                    }
                }
            }
        }
        *out = sum;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// AVX-512 kernels
// ---------------------------------------------------------------------------

/// Dequantize one IQ2_XS block using AVX-512.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES`, `output.len() >= BLOCK_SIZE`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let qs_bytes = &block[QS_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let scale_byte = scales[ib32];
        let db0 = d * (0.5 + (scale_byte & 0x0f) as f32) * 0.25;
        let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..QS_PER_SUPER {
            let byte_pos = 8 * ib32 + 2 * l;
            let qs_val = u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;
            let grid_idx = qs_val & 511;
            let sign_idx = qs_val >> 9;
            let dl = if l < 2 { db0 } else { db1 };
            let mags = IQ2XS_GRID[grid_idx].to_le_bytes();
            let sign_byte = KSIGNS_IQ2XS[sign_idx];

            let mut vals = [0.0f32; 8];
            for j in 0..8 {
                let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                vals[j] = dl * mags[j] as f32 * sign;
            }

            let group_off = weight_base + l * WEIGHTS_PER_GROUP;
            // SAFETY: group_off + 8 <= 256
            let v = _mm256_loadu_ps(vals.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(group_off), v);
        }
    }
}

/// GEMV for one row using AVX-512 accumulation.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn gemv_row_avx512(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm512_setzero_ps();

    for blk in 0..blocks_per_row {
        let bo = blk * BLOCK_BYTES;
        let block = &row_data[bo..bo + BLOCK_BYTES];
        let d = f16_to_f32(block);
        let qs_bytes = &block[QS_OFFSET..SCALES_OFFSET];
        let scales = &block[SCALES_OFFSET..BLOCK_BYTES];
        let col_block_base = blk * BLOCK_SIZE;

        for ib32 in 0..N_SUPERBLOCKS {
            let scale_byte = scales[ib32];
            let db0 = d * (0.5 + (scale_byte & 0x0f) as f32) * 0.25;
            let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
            let col_super_base = col_block_base + ib32 * SUPER_BLOCK_SIZE;

            for l in 0..QS_PER_SUPER {
                let byte_pos = 8 * ib32 + 2 * l;
                let qs_val =
                    u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;
                let grid_idx = qs_val & 511;
                let sign_idx = qs_val >> 9;
                let dl = if l < 2 { db0 } else { db1 };
                let col = col_super_base + l * WEIGHTS_PER_GROUP;

                let mags = IQ2XS_GRID[grid_idx].to_le_bytes();
                let sign_byte = KSIGNS_IQ2XS[sign_idx];

                if col + WEIGHTS_PER_GROUP <= n_cols {
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        vals[j] = dl * mags[j] as f32 * sign;
                    }
                    // Widen 8-wide to 16-wide ZMM, multiply with input.
                    let w8 = _mm256_loadu_ps(vals.as_ptr());
                    let w16 = _mm512_castps256_ps512(w8);
                    // SAFETY: col + 8 <= n_cols <= input.len()
                    let i8 = _mm256_loadu_ps(input.as_ptr().add(col));
                    let i16 = _mm512_castps256_ps512(i8);
                    let prod = _mm512_mul_ps(w16, i16);
                    acc = _mm512_add_ps(acc, prod);
                } else {
                    let mut partial = 0.0f32;
                    for j in 0..WEIGHTS_PER_GROUP {
                        let c = col + j;
                        if c >= n_cols {
                            break;
                        }
                        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        partial += dl * mags[j] as f32 * sign * input[c];
                    }
                    acc = _mm512_mask_add_ps(acc, 0x0001u16, acc, _mm512_set1_ps(partial));
                }
            }
        }
    }

    hsum_f32_avx512(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::iq2_xs::Iq2XsRef;

    fn make_block(d: f32, qs_words: &[u16; 32], scales: &[u8; 8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &w in qs_words {
            block.extend_from_slice(&w.to_le_bytes());
        }
        block.extend_from_slice(scales);
        block
    }

    fn varied_block() -> Vec<u8> {
        let qs_words: [u16; 32] =
            core::array::from_fn(|i| ((i as u16 * 17 + 3) & 0x1FF) | ((i as u16 * 7) << 9));
        let scales: [u8; 8] = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        make_block(0.5, &qs_words, &scales)
    }

    #[test]
    fn avx512_iq2_xs_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = varied_block();
        let mut out_avx512 = [0.0f32; BLOCK_SIZE];
        let mut out_ref = [0.0f32; BLOCK_SIZE];
        Iq2XsAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Iq2XsRef
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
    fn avx512_iq2_xs_matvec_q8_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = varied_block();
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
            oxillama_gguf::GgufTensorType::Iq2Xs,
        );
        let tensor_ref = crate::types::QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Iq2Xs,
        );

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Iq2XsAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Iq2XsRef
            .gemv(&tensor_ref, &input, &mut out_ref)
            .expect("ref gemv");

        for (i, (&a, &r)) in out_avx512.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "gemv mismatch row {i}: avx512={a}, ref={r}"
            );
        }
    }

    #[test]
    fn avx512_iq2_xs_kernel_metadata() {
        assert_eq!(Iq2XsAvx512.name(), "IQ2_XS");
        assert_eq!(Iq2XsAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Iq2XsAvx512.block_bytes(), BLOCK_BYTES);
    }
}
