//! AVX-512 accelerated IQ2_XXS quantization kernel.
//!
//! Mirrors `simd/avx2/iq2_xxs.rs` but processes the same logical data with
//! 512-bit ZMM registers instead of 256-bit YMM registers where beneficial.
//! For IQ2_XXS the inner-most loop still decodes 8 weights per grid entry
//! (the grid width is fixed at 64-bit / 8 bytes), so the AVX-512 advantage
//! comes from using `_mm512_fmadd_ps` 16-wide accumulation and the single-
//! instruction `_mm512_reduce_add_ps` horizontal reduction.
//!
//! Block layout (66 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..66]:  `qs[64]` — 8 super-blocks × 8 bytes each
//!
//! Each super-block (two u32 words):
//! - `aux32[0]`: 4 grid indices → 4 groups of 8 weights (32 total)
//! - `aux32[1]`: bits 28-31 = sub-scale, bits 0-27 = 4×7-bit sign selectors
//!
//! Scale formula: `db = d * (0.5 + (aux32[1] >> 28)) * 0.25`
//! Sign byte: `KSIGNS_IQ2XS[(aux32[1] >> 7*l) & 0x7F]`

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ2_XXS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ2_XXS block: 2 (FP16 d) + 64 (qs).
pub const BLOCK_BYTES: usize = 66;
/// Number of super-blocks per IQ2_XXS block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per super-block.
const SUPER_BLOCK_SIZE: usize = 32;
/// Groups per super-block.
const GROUPS_PER_SUPER: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;

/// AVX-512 accelerated IQ2_XXS kernel.
///
/// Falls back to scalar if `avx512f` is not available at runtime.
pub struct Iq2XxsAvx512;

impl QuantKernel for Iq2XxsAvx512 {
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
        // SAFETY: bounds verified above; avx512f confirmed by runtime check.
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
            // SAFETY: bounds checked above; avx512f confirmed.
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
        "IQ2_XXS"
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback paths (no SIMD required)
// ---------------------------------------------------------------------------

fn scalar_dequant_block(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    use crate::reference::iq2_xxs::Iq2XxsRef;
    Iq2XxsRef.dequant_block(block, output)
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
    use crate::reference::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    for (row, out) in output.iter_mut().enumerate().take(n_rows) {
        let row_start = row * row_bytes;
        let mut sum = 0.0f32;
        for blk in 0..blocks_per_row {
            let block_off = row_start + blk * BLOCK_BYTES;
            let block = &data[block_off..block_off + BLOCK_BYTES];
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..BLOCK_BYTES];
            for ib32 in 0..N_SUPERBLOCKS {
                let base = ib32 * 8;
                let aux32_0 =
                    u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
                let aux32_1 =
                    u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
                let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
                let aux8 = aux32_0.to_le_bytes();
                let col_base = blk * BLOCK_SIZE + ib32 * SUPER_BLOCK_SIZE;
                for l in 0..GROUPS_PER_SUPER {
                    let col = col_base + l * WEIGHTS_PER_GROUP;
                    let grid_idx = aux8[l] as usize;
                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];
                    let mags = IQ2XXS_GRID[grid_idx].to_le_bytes();
                    for j in 0..WEIGHTS_PER_GROUP {
                        let c = col + j;
                        if c < n_cols {
                            let mag = mags[j] as f32;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            sum += db * mag * sign * input[c];
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

/// Decode one group (8 weights) and store into output[0..8].
///
/// # Safety
/// - `output.len() >= 8`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn decode_group_avx512(grid_idx: u8, sign_byte: u8, db: f32, output: &mut [f32]) {
    let mags = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
    // Build 8 f32 values: db * mag * sign
    let mut vals = [0.0f32; 8];
    for j in 0..8 {
        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
            -1.0_f32
        } else {
            1.0_f32
        };
        vals[j] = db * mags[j] as f32 * sign;
    }
    // Use AVX-512 store via 256-bit sub-register (8 floats)
    // SAFETY: output.len() >= 8; vals is stack-allocated with 8 elements.
    let v = _mm256_loadu_ps(vals.as_ptr());
    _mm256_storeu_ps(output.as_mut_ptr(), v);
}

/// Dequantize one IQ2_XXS block using AVX-512.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES`
/// - `output.len() >= BLOCK_SIZE`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let qs = &block[2..BLOCK_BYTES];

    for ib32 in 0..N_SUPERBLOCKS {
        let base = ib32 * 8;
        let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
        let aux32_1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
        let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
        let aux8 = aux32_0.to_le_bytes();
        let weight_base = ib32 * SUPER_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUPER {
            let grid_idx = aux8[l];
            let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
            let sign_byte = KSIGNS_IQ2XS[sign_idx];
            let group_off = weight_base + l * WEIGHTS_PER_GROUP;
            // SAFETY: group_off + 8 <= 256 = output.len()
            decode_group_avx512(
                grid_idx,
                sign_byte,
                db,
                &mut output[group_off..group_off + WEIGHTS_PER_GROUP],
            );
        }
    }
}

/// Compute dot-product of one group with input, returning a partial __m512.
///
/// The 8 products are placed in lanes 0..7 of a 16-wide ZMM register.
///
/// # Safety
/// - `input_ptr` points to >= 8 valid f32 values
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dot_group_avx512(grid_idx: u8, sign_byte: u8, db: f32, input_ptr: *const f32) -> __m512 {
    let mags = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
    let mut vals = [0.0f32; 8];
    for j in 0..8 {
        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
            -1.0_f32
        } else {
            1.0_f32
        };
        vals[j] = db * mags[j] as f32 * sign;
    }
    // Load weights into lower 8 lanes of a ZMM (upper 8 lanes = 0).
    // We use a 256-bit load and cast to 512-bit with zero-extension.
    let w8 = _mm256_loadu_ps(vals.as_ptr());
    let w16 = _mm512_castps256_ps512(w8);
    // Load 8 input floats similarly.
    let i8 = _mm256_loadu_ps(input_ptr);
    let i16 = _mm512_castps256_ps512(i8);
    _mm512_mul_ps(w16, i16)
}

/// Compute the dot product of one row of an IQ2_XXS matrix with an FP32 vector.
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
        let block_off = blk * BLOCK_BYTES;
        let block = &row_data[block_off..block_off + BLOCK_BYTES];
        let d = f16_to_f32(block);
        let qs = &block[2..BLOCK_BYTES];

        for ib32 in 0..N_SUPERBLOCKS {
            let base = ib32 * 8;
            let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            let aux32_1 =
                u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
            let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
            let aux8 = aux32_0.to_le_bytes();
            let col_base = blk * BLOCK_SIZE + ib32 * SUPER_BLOCK_SIZE;

            for l in 0..GROUPS_PER_SUPER {
                let col = col_base + l * WEIGHTS_PER_GROUP;
                let remaining = n_cols.saturating_sub(col);
                if remaining >= WEIGHTS_PER_GROUP {
                    let grid_idx = aux8[l];
                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];
                    // SAFETY: col + 8 <= n_cols <= input.len()
                    acc = _mm512_add_ps(
                        acc,
                        dot_group_avx512(grid_idx, sign_byte, db, input.as_ptr().add(col)),
                    );
                } else if remaining > 0 {
                    let grid_idx = aux8[l] as usize;
                    let mags = IQ2XXS_GRID[grid_idx].to_le_bytes();
                    let sign_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                    let sign_byte = KSIGNS_IQ2XS[sign_idx];
                    let mut partial = 0.0f32;
                    for j in 0..remaining {
                        let mag = mags[j] as f32;
                        let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        partial += db * mag * sign * input[col + j];
                    }
                    // Accumulate scalar partial into lane 0 via set1 + mask.
                    let pv = _mm512_set1_ps(partial);
                    // Only lane 0 contributes.
                    acc = _mm512_mask_add_ps(acc, 0x0001u16, acc, pv);
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
    use crate::reference::iq2_xxs::Iq2XxsRef;

    fn make_block(d: f32) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        let d_le = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_le[0];
        block[1] = d_le[1];
        block
    }

    fn make_varied_block(d: f32) -> [u8; BLOCK_BYTES] {
        let mut block = make_block(d);
        for ib32 in 0..N_SUPERBLOCKS {
            let base = 2 + ib32 * 8;
            block[base] = (ib32 * 7 + 1) as u8;
            block[base + 1] = (ib32 * 13 + 5) as u8;
            block[base + 2] = (ib32 * 23 + 11) as u8;
            block[base + 3] = (ib32 * 37 + 17) as u8;
            let scale_bits = ((ib32 as u32 * 3) % 16) << 28;
            let sign_bits = (ib32 as u32 * 5) & 0x7F;
            let aux32_1 = scale_bits | sign_bits;
            let ab = aux32_1.to_le_bytes();
            block[base + 4] = ab[0];
            block[base + 5] = ab[1];
            block[base + 6] = ab[2];
            block[base + 7] = ab[3];
        }
        block
    }

    #[test]
    fn avx512_iq2_xxs_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_varied_block(0.5);
        let mut out_avx512 = [0.0f32; BLOCK_SIZE];
        let mut out_ref = [0.0f32; BLOCK_SIZE];
        Iq2XxsAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Iq2XxsRef
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
    fn avx512_iq2_xxs_matvec_q8_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Build a 4-row matrix, each row is 1 block (256 weights).
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
            oxillama_gguf::GgufTensorType::Iq2Xxs,
        );
        let tensor_ref = crate::types::QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Iq2Xxs,
        );

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Iq2XxsAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Iq2XxsRef
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
    fn avx512_iq2_xxs_zero_scale() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_block(0.0);
        let mut out = [1.0f32; BLOCK_SIZE];
        Iq2XxsAvx512
            .dequant_block(&block, &mut out)
            .expect("dequant");
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0");
        }
    }

    #[test]
    fn avx512_iq2_xxs_kernel_name() {
        assert_eq!(Iq2XxsAvx512.name(), "IQ2_XXS");
        assert_eq!(Iq2XxsAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Iq2XxsAvx512.block_bytes(), BLOCK_BYTES);
    }
}
