//! AVX2+FMA accelerated IQ2_XXS quantization kernel.
//!
//! IQ2_XXS block layout (66 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[64]` — 8 super-blocks × 8 bytes each
//!
//! Each super-block (8 bytes, seen as two u32 little-endian words):
//! - `aux32[0]` bytes: 4 grid indices → 4 groups of 8 weights (32 total)
//! - `aux32[1]`: bits 28-31 = sub-scale, bits 0-27 = 4×7-bit sign selectors
//!
//! The grid index references `IQ2XXS_GRID[idx]` (a u64 of 8 magnitude bytes).
//! Scale formula: `db = d * (0.5 + (aux32[1] >> 28)) * 0.25`.
//! Sign byte: `KSIGNS_IQ2XS[(aux32[1] >> 7*l) & 0x7F]`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XXS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XXS block: 2 (FP16 d) + 64 (qs).
pub const BLOCK_BYTES: usize = 66;
/// Number of super-blocks per IQ2_XXS block (QK_K/32 = 8).
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Number of groups per super-block (each group = 8 weights).
const GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;

/// AVX2+FMA accelerated IQ2_XXS kernel.
///
/// Requires `avx2` and `fma` CPU features. The [`crate::dispatch::KernelDispatcher`]
/// checks for these at runtime before constructing this kernel.
pub struct Iq2XxsAvx2;

impl QuantKernel for Iq2XxsAvx2 {
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
        "IQ2_XXS"
    }
}

// ---------------------------------------------------------------------------
// Internal AVX2 kernels
// ---------------------------------------------------------------------------

/// Decode one group (8 weights) from a grid index and sign selector,
/// scale by `db`, and store into `output[0..8]`.
///
/// # Safety
/// - `output.len() >= 8`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_group_avx2(grid_idx: u8, sign_byte: u8, db: f32, output: &mut [f32]) {
    let grid_entry = IQ2XXS_GRID[grid_idx as usize];
    let mag_bytes: [u8; 8] = grid_entry.to_le_bytes();

    // Load 8 magnitude bytes into an AVX2 register via i32 promotion.
    // Using _mm256_setr_epi32 to load 8 values individually.
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

    // Build sign vector: -1.0 or +1.0 per weight based on the sign byte.
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

    let vdb = _mm256_set1_ps(db);

    // result = db * mag * sign
    let result = _mm256_mul_ps(_mm256_mul_ps(mag_f32, signs), vdb);
    _mm256_storeu_ps(output.as_mut_ptr(), result);
}

/// Dequantize one 66-byte IQ2_XXS block to 256 FP32 values using AVX2.
///
/// # Safety
/// - `block.len() >= 66`
/// - `output.len() >= 256`
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    // Read FP16 scale.
    let d = f16_to_f32(block);
    let qs = &block[2..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let base = ib32 * 8;

        let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
        let aux32_1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

        let scale_factor = (aux32_1 >> 28) as f32;
        let db = d * (0.5 + scale_factor) * 0.25;

        let aux8: [u8; 4] = aux32_0.to_le_bytes();
        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let grid_idx = aux8[l];
            let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
            let sign_byte = KSIGNS_IQ2XS[sign_idx];

            let group_offset = weight_base + l * WEIGHTS_PER_GROUP;
            decode_group_avx2(
                grid_idx,
                sign_byte,
                db,
                &mut output[group_offset..group_offset + WEIGHTS_PER_GROUP],
            );
        }
    }
}

/// Compute the dot product of 8 dequantized weights (from one group) with the
/// corresponding 8 input values. Returns the partial sum as a scalar.
///
/// # Safety
/// - `input_ptr` must point to at least 8 valid floats.
/// - CPU must support `avx2` and `fma`
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_group_avx2(grid_idx: u8, sign_byte: u8, db: f32, input_ptr: *const f32) -> __m256 {
    let grid_entry = IQ2XXS_GRID[grid_idx as usize];
    let mag_bytes: [u8; 8] = grid_entry.to_le_bytes();

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

    let vdb = _mm256_set1_ps(db);
    // weight = db * mag * sign
    let weight = _mm256_mul_ps(_mm256_mul_ps(mag_f32, signs), vdb);
    // input
    let inp = _mm256_loadu_ps(input_ptr);
    // weight * input
    _mm256_mul_ps(weight, inp)
}

/// Compute the dot product of one row of an IQ2_XXS matrix with an FP32 vector.
///
/// Returns the scalar result for this row.
///
/// # Safety
/// - `row_data.len() >= blocks_per_row * BLOCK_BYTES`
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
        let qs = &block[2..BLOCK_BYTES];

        for ib32 in 0..N_SUPERBLOCKS {
            let base = ib32 * 8;

            let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            let aux32_1 =
                u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);

            let scale_factor = (aux32_1 >> 28) as f32;
            let db = d * (0.5 + scale_factor) * 0.25;

            let aux8: [u8; 4] = aux32_0.to_le_bytes();
            let col_base = blk * BLOCK_SIZE + ib32 * SUPER_BLOCK_SIZE;

            for l in 0..GROUPS_PER_SUPER {
                let col = col_base + l * WEIGHTS_PER_GROUP;
                let remaining = n_cols.saturating_sub(col);

                if remaining >= WEIGHTS_PER_GROUP {
                    // Fast path: full group within bounds.
                    let grid_idx = aux8[l];
                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];

                    let partial = dot_group_avx2(grid_idx, sign_byte, db, input.as_ptr().add(col));
                    acc = _mm256_add_ps(acc, partial);
                } else if remaining > 0 {
                    // Tail: scalar fallback for partial group.
                    let grid_idx = aux8[l] as usize;
                    let magnitudes: [u8; 8] = IQ2XXS_GRID[grid_idx].to_le_bytes();
                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];

                    let mut partial_sum = 0.0f32;
                    for j in 0..remaining {
                        let mag = magnitudes[j] as f32;
                        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        partial_sum += db * mag * sign * input[col + j];
                    }
                    // Add scalar partial sum to accumulator lane 0.
                    let scalar_vec = _mm256_set1_ps(partial_sum);
                    let mask = _mm256_setr_ps(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    acc = _mm256_fmadd_ps(scalar_vec, mask, acc);
                }
            }
        }
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests (CI only — not executed on aarch64 Darwin build machines)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx2"))]
mod tests {
    use super::*;
    use crate::reference::iq2_xxs::Iq2XxsRef;

    /// Construct a minimal IQ2_XXS block with a given FP16 scale and zero qs data.
    fn make_zero_iq2_xxs_block(scale: f32) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        let d_le = half::f16::from_f32(scale).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    /// Make a block where every super-block uses grid index 0 and no signs.
    fn make_uniform_block(scale: f32) -> [u8; BLOCK_BYTES] {
        make_zero_iq2_xxs_block(scale)
    }

    /// Build a single-row QuantTensor from raw block bytes.
    fn make_tensor(block: Vec<u8>, n_cols: usize) -> QuantTensor {
        QuantTensor::new(
            block,
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Iq2Xxs,
        )
    }

    #[test]
    fn test_kernel_metadata() {
        assert_eq!(Iq2XxsAvx2.name(), "IQ2_XXS");
        assert_eq!(Iq2XxsAvx2.block_size(), 256);
        assert_eq!(Iq2XxsAvx2.block_bytes(), 66);
    }

    #[test]
    fn test_block_size_matches_reference() {
        let ref_kernel = Iq2XxsRef;
        assert_eq!(Iq2XxsAvx2.block_size(), ref_kernel.block_size());
        assert_eq!(Iq2XxsAvx2.block_bytes(), ref_kernel.block_bytes());
    }

    #[test]
    fn test_dequant_buffer_too_small() {
        let small = [0u8; 30];
        let mut out = [0.0f32; 256];
        let result = Iq2XxsAvx2.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));

        let block = make_zero_iq2_xxs_block(1.0);
        let mut small_out = [0.0f32; 100];
        let result = Iq2XxsAvx2.dequant_block(&block, &mut small_out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_zero_scale() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_zero_iq2_xxs_block(0.0);
        let mut out = [1.0f32; 256];
        Iq2XxsAvx2
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_dequant_matches_reference_uniform() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 2.0_f32;
        let block = make_uniform_block(d);

        let mut out_avx2 = [0.0f32; 256];
        let mut out_ref = [0.0f32; 256];

        Iq2XxsAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant failed");
        Iq2XxsRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant failed");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_with_signs() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Build a block with sign_idx = 1 for the first group of super-block 0.
        let d = 1.0_f32;
        let mut block = make_zero_iq2_xxs_block(d);
        block[2 + 4] = 1; // aux32[1] low byte = 1 → sign_idx[0] = 1
        block[2 + 5] = 0;
        block[2 + 6] = 0;
        block[2 + 7] = 0;

        let mut out_avx2 = [0.0f32; 256];
        let mut out_ref = [0.0f32; 256];

        Iq2XxsAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant failed");
        Iq2XxsRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant failed");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "sign dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_dequant_matches_reference_varied_grid_indices() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Build a block with diverse grid indices and scale bits.
        let d = 0.5_f32;
        let mut block = make_zero_iq2_xxs_block(d);

        // Super-block 0: grid indices [1, 2, 3, 4], scale=5 (aux32[1] >> 28 = 5),
        // sign selectors = [2, 3, 0, 1].
        block[2] = 1;
        block[3] = 2;
        block[4] = 3;
        block[5] = 4;
        // aux32[1] = 0x5000_0000 | (1 << 21) | (0 << 14) | (3 << 7) | 2
        // = 0x5020_01C2
        let aux32_1: u32 = (5u32 << 28) | (1u32 << 21) | (3u32 << 7) | 2;
        let aux_bytes = aux32_1.to_le_bytes();
        block[6] = aux_bytes[0];
        block[7] = aux_bytes[1];
        block[8] = aux_bytes[2];
        block[9] = aux_bytes[3];

        // Super-block 1: different indices.
        block[10] = 10;
        block[11] = 20;
        block[12] = 30;
        block[13] = 40;
        let aux32_1b: u32 = (3u32 << 28) | (5u32 << 21) | (2u32 << 14) | (1u32 << 7) | 4;
        let aux_bytes_b = aux32_1b.to_le_bytes();
        block[14] = aux_bytes_b[0];
        block[15] = aux_bytes_b[1];
        block[16] = aux_bytes_b[2];
        block[17] = aux_bytes_b[3];

        let mut out_avx2 = [0.0f32; 256];
        let mut out_ref = [0.0f32; 256];

        Iq2XxsAvx2
            .dequant_block(&block, &mut out_avx2)
            .expect("avx2 dequant failed");
        Iq2XxsRef
            .dequant_block(&block, &mut out_ref)
            .expect("ref dequant failed");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 1e-4,
                "varied grid dequant mismatch at index {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_matches_reference_ones() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 1.0_f32;
        let block = make_uniform_block(d);

        let tensor = make_tensor(block.to_vec(), 256);
        let input = vec![1.0f32; 256];

        let mut out_avx2 = [0.0f32; 1];
        let mut out_ref = [0.0f32; 1];

        Iq2XxsAvx2
            .gemv(&tensor, &input, &mut out_avx2)
            .expect("avx2 gemv failed");
        Iq2XxsRef
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv failed");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 1e-2,
            "gemv mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_matches_reference_random_like() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.75_f32;
        let mut block = make_zero_iq2_xxs_block(d);

        // Fill with some diverse data across all super-blocks.
        for ib32 in 0..N_SUPERBLOCKS {
            let base = 2 + ib32 * 8;
            // Grid indices.
            block[base] = (ib32 * 7) as u8;
            block[base + 1] = (ib32 * 13 + 5) as u8;
            block[base + 2] = (ib32 * 23 + 11) as u8;
            block[base + 3] = (ib32 * 37 + 17) as u8;
            // aux32[1] with varied scale and sign bits.
            let scale_bits = ((ib32 as u32 * 3) % 16) << 28;
            let sign_bits = (ib32 as u32 * 5) & 0x7F;
            let aux32_1 = scale_bits | sign_bits;
            let aux_bytes = aux32_1.to_le_bytes();
            block[base + 4] = aux_bytes[0];
            block[base + 5] = aux_bytes[1];
            block[base + 6] = aux_bytes[2];
            block[base + 7] = aux_bytes[3];
        }

        let tensor = make_tensor(block.to_vec(), 256);

        // Input vector with varying values.
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let mut out_avx2 = [0.0f32; 1];
        let mut out_ref = [0.0f32; 1];

        Iq2XxsAvx2
            .gemv(&tensor, &input, &mut out_avx2)
            .expect("avx2 gemv failed");
        Iq2XxsRef
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv failed");

        assert!(
            (out_avx2[0] - out_ref[0]).abs() < 0.1,
            "gemv random-like mismatch: avx2={}, ref={}",
            out_avx2[0],
            out_ref[0]
        );
    }

    #[test]
    fn test_gemv_multi_row() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 1.5_f32;
        let block1 = make_uniform_block(d);

        let mut block2 = make_zero_iq2_xxs_block(d);
        block2[2 + 4] = 3; // Sign variations.

        let mut data = block1.to_vec();
        data.extend_from_slice(&block2);

        let tensor = QuantTensor::new(data, vec![2, 256], oxillama_gguf::GgufTensorType::Iq2Xxs);
        let input = vec![1.0f32; 256];

        let mut out_avx2 = [0.0f32; 2];
        let mut out_ref = [0.0f32; 2];

        Iq2XxsAvx2
            .gemv(&tensor, &input, &mut out_avx2)
            .expect("avx2 gemv failed");
        Iq2XxsRef
            .gemv(&tensor, &input, &mut out_ref)
            .expect("ref gemv failed");

        for (i, (&a, &r)) in out_avx2.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - r).abs() < 0.1,
                "multi-row gemv mismatch at row {i}: avx2={a}, ref={r}"
            );
        }
    }

    #[test]
    fn test_gemv_dimension_mismatch() {
        let block = make_uniform_block(1.0);
        let tensor = make_tensor(block.to_vec(), 256);

        // Input too short.
        let short_input = vec![1.0f32; 128];
        let mut out = [0.0f32; 1];
        let result = Iq2XxsAvx2.gemv(&tensor, &short_input, &mut out);
        assert!(matches!(result, Err(QuantError::DimensionMismatch { .. })));

        // Output too short.
        let input = vec![1.0f32; 256];
        let mut short_out: [f32; 0] = [];
        let result = Iq2XxsAvx2.gemv(&tensor, &input, &mut short_out);
        assert!(matches!(result, Err(QuantError::DimensionMismatch { .. })));
    }
}
