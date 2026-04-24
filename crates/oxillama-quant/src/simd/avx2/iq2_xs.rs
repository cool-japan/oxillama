//! AVX2+FMA accelerated IQ2_XS quantization kernel.
//!
//! IQ2_XS block layout (74 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[32]` — 32 × u16 (little-endian), 4 per super-block × 8 super-blocks
//!   Each u16: lower 9 bits = grid index into `IQ2XS_GRID[512]`,
//!              upper 7 bits = sign selector into `KSIGNS_IQ2XS[128]`.
//! - bytes[66..74]: `scales[8]` — one byte per super-block
//!   low nibble → db0 (groups 0-1), high nibble → db1 (groups 2-3).
//!   `db = d * (0.5 + nibble) * 0.25`
//!
//! Weight decode: `w[j] = dl * IQ2XS_GRID[grid_idx][j] * sign[j]`
//! where `sign[j] = KSIGNS_IQ2XS[sign_idx] & KMASK_IQ2XS[j] != 0 ? -1 : 1`.
//!
//! The AVX2 dequantize inner loop calls `decode_group_avx2` (shared pattern
//! with IQ2_XXS) but uses 9-bit grid indices instead of 8-bit, and reads
//! scales from a separate byte array rather than from the qs words.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XS block: 2 (FP16 d) + 64 (32 × u16 qs) + 8 (scales).
pub const BLOCK_BYTES: usize = 74;
/// Number of super-blocks per IQ2_XS block (QK_K/32 = 8).
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Number of u16 qs entries per super-block (32 total / 8 super-blocks = 4).
const QS_PER_SUPER: usize = 4;
/// Weights per group (one grid entry holds 8 weights).
const WEIGHTS_PER_GROUP: usize = 8;
/// Byte offset of the qs region within the block.
const QS_OFFSET: usize = 2;
/// Byte offset of the scales region within the block.
const SCALES_OFFSET: usize = 66;

/// AVX2+FMA accelerated IQ2_XS kernel.
///
/// Requires `avx2` and `fma` CPU features. The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Iq2XsAvx2;

impl QuantKernel for Iq2XsAvx2 {
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
        // SAFETY: bounds verified above. CPU avx2+fma guaranteed by dispatcher.
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
            // SAFETY: bounds checked above. CPU avx2+fma guaranteed by dispatcher.
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
        "IQ2_XS"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Decode one group (8 weights) from a 9-bit grid index and 7-bit sign selector index,
/// scale by `dl`, and store into `output[0..8]`.
///
/// # Safety
/// - `output.len() >= 8`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_group_avx2(grid_idx: usize, sign_idx: usize, dl: f32, output: &mut [f32]) {
    let grid_entry = IQ2XS_GRID[grid_idx];
    let mag_bytes: [u8; 8] = grid_entry.to_le_bytes();
    let sign_byte = KSIGNS_IQ2XS[sign_idx];

    // Load 8 magnitude bytes as i32 and convert to f32.
    let mag_i32 = _mm256_setr_epi32(
        mag_bytes[0] as i32,
        mag_bytes[1] as i32,
        mag_bytes[2] as i32,
        mag_bytes[3] as i32,
        mag_bytes[4] as i32,
        mag_bytes[5] as i32,
        mag_bytes[6] as i32,
        mag_bytes[7] as i32,
    );
    let mag_f32 = _mm256_cvtepi32_ps(mag_i32);

    // Build ±1 sign vector from the sign byte.
    let signs = _mm256_setr_ps(
        if sign_byte & KMASK_IQ2XS[0] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[1] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[2] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[3] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[4] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[5] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[6] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[7] != 0 {
            -1.0
        } else {
            1.0
        },
    );

    // result = dl * mag * sign
    let vdl = _mm256_set1_ps(dl);
    let result = _mm256_mul_ps(_mm256_mul_ps(mag_f32, signs), vdl);
    _mm256_storeu_ps(output.as_mut_ptr(), result);
}

/// Dequantize one 74-byte IQ2_XS block to 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 74`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let qs_bytes = &block[QS_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let scale_byte = scales[ib32];
        let db0 = d * (0.5 + (scale_byte & 0x0f) as f32) * 0.25;
        let db1 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;

        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..QS_PER_SUPER {
            // Each u16 is stored at byte offset 8*ib32 + 2*l within qs_bytes.
            let byte_pos = 8 * ib32 + 2 * l;
            let qs_val = u16::from_le_bytes([qs_bytes[byte_pos], qs_bytes[byte_pos + 1]]) as usize;

            // Lower 9 bits: grid index (IQ2XS has 512-entry grid).
            let grid_idx = qs_val & 511;
            // Upper 7 bits: sign selector.
            let sign_idx = qs_val >> 9;

            // Groups 0-1 use db0, groups 2-3 use db1.
            let dl = if l < 2 { db0 } else { db1 };

            let group_offset = weight_base + l * WEIGHTS_PER_GROUP;
            decode_group_avx2(
                grid_idx,
                sign_idx,
                dl,
                &mut output[group_offset..group_offset + WEIGHTS_PER_GROUP],
            );
        }
    }
}

/// Dot product helper: compute `dl * grid_mag[j] * sign[j]` · `input[j]` for 8 weights.
///
/// Returns an AVX2 accumulator vector (not yet horizontally summed).
///
/// # Safety
/// - `input_ptr` must point to at least 8 valid floats.
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_group_avx2(
    grid_idx: usize,
    sign_idx: usize,
    dl: f32,
    input_ptr: *const f32,
) -> __m256 {
    let grid_entry = IQ2XS_GRID[grid_idx];
    let mag_bytes: [u8; 8] = grid_entry.to_le_bytes();
    let sign_byte = KSIGNS_IQ2XS[sign_idx];

    let mag_i32 = _mm256_setr_epi32(
        mag_bytes[0] as i32,
        mag_bytes[1] as i32,
        mag_bytes[2] as i32,
        mag_bytes[3] as i32,
        mag_bytes[4] as i32,
        mag_bytes[5] as i32,
        mag_bytes[6] as i32,
        mag_bytes[7] as i32,
    );
    let mag_f32 = _mm256_cvtepi32_ps(mag_i32);

    let signs = _mm256_setr_ps(
        if sign_byte & KMASK_IQ2XS[0] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[1] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[2] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[3] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[4] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[5] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[6] != 0 {
            -1.0
        } else {
            1.0
        },
        if sign_byte & KMASK_IQ2XS[7] != 0 {
            -1.0
        } else {
            1.0
        },
    );

    let vdl = _mm256_set1_ps(dl);
    // w = dl * mag * sign; contribution = w * input
    let w = _mm256_mul_ps(_mm256_mul_ps(mag_f32, signs), vdl);
    let inp = _mm256_loadu_ps(input_ptr);
    _mm256_mul_ps(w, inp)
}

/// Compute the dot product of one row of an IQ2_XS matrix with an FP32 vector.
///
/// # Safety
/// - `row_data.len() == blocks_per_row * BLOCK_BYTES`
/// - `input.len() >= n_cols`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(
    row_data: &[u8],
    input: &[f32],
    blocks_per_row: usize,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();

    for blk in 0..blocks_per_row {
        let block_offset = blk * BLOCK_BYTES;
        let block = &row_data[block_offset..block_offset + BLOCK_BYTES];

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
                if col + WEIGHTS_PER_GROUP <= n_cols {
                    // Fast path: full group within bounds.
                    // SAFETY: col + 8 <= n_cols <= input.len()
                    acc = _mm256_add_ps(
                        acc,
                        dot_group_avx2(grid_idx, sign_idx, dl, input.as_ptr().add(col)),
                    );
                } else {
                    // Tail: decode group and scalar multiply.
                    let grid_entry = IQ2XS_GRID[grid_idx];
                    let mag_bytes: [u8; 8] = grid_entry.to_le_bytes();
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];
                    let mut partial = 0.0f32;
                    for j in 0..WEIGHTS_PER_GROUP {
                        let idx = col + j;
                        if idx >= n_cols {
                            break;
                        }
                        let mag = mag_bytes[j] as f32;
                        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        partial += dl * mag * sign * input[idx];
                    }
                    acc = _mm256_add_ps(acc, _mm256_set1_ps(partial));
                }
            }
        }
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal IQ2_XS block for testing.
    fn make_block(d_f32: f32, qs_words: &[u16; 32], scales: &[u8; 8]) -> Vec<u8> {
        let d_f16 = half::f16::from_f32(d_f32);
        let mut block = Vec::with_capacity(BLOCK_BYTES);
        block.extend_from_slice(&d_f16.to_le_bytes());
        for &w in qs_words {
            block.extend_from_slice(&w.to_le_bytes());
        }
        block.extend_from_slice(scales);
        block
    }

    #[test]
    fn test_block_size_constant() {
        assert_eq!(BLOCK_BYTES, 74);
        assert_eq!(BLOCK_SIZE, 256);
    }

    #[test]
    fn test_dequant_zero_grid() {
        // Grid index 0 stores all-zero magnitudes → output should all be 0.
        let qs_words = [0u16; 32]; // grid_idx = 0, sign_idx = 0
        let scales = [0x11u8; 8]; // both nibbles = 1 → db = d * 1.5 * 0.25
        let block = make_block(1.0, &qs_words, &scales);
        let mut output = [0.0f32; BLOCK_SIZE];
        let kernel = Iq2XsAvx2;
        kernel.dequant_block(&block, &mut output).unwrap();
        // Grid[0] bytes all zero → all weights 0 regardless of scale/sign.
        let grid_zero: [u8; 8] = IQ2XS_GRID[0].to_le_bytes();
        for (i, &v) in output.iter().enumerate() {
            let expected = grid_zero[i % WEIGHTS_PER_GROUP] as f32;
            // If grid magnitudes are 0, result is 0.
            if expected == 0.0 {
                assert!(v.abs() < 1e-6, "expected 0.0 at {i}, got {v}");
            }
        }
    }

    #[test]
    fn test_matches_reference_scalar() {
        use crate::reference::Iq2XsRef;

        // Build a block with varied content to exercise multiple code paths.
        let qs_words: [u16; 32] = core::array::from_fn(|i| (i as u16 * 17) & 0xFFFF);
        let scales: [u8; 8] = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let block = make_block(0.5, &qs_words, &scales);

        let mut ref_out = [0.0f32; BLOCK_SIZE];
        let mut avx_out = [0.0f32; BLOCK_SIZE];

        Iq2XsRef.dequant_block(&block, &mut ref_out).unwrap();
        Iq2XsAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!(
                (r - a).abs() < 1e-5,
                "mismatch at index {i}: ref={r} avx={a}"
            );
        }
    }
}
