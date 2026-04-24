//! AVX2+FMA accelerated IQ4_XS quantization kernel.
//!
//! IQ4_XS block layout (136 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 delta `d` (little-endian)
//! - bytes[2..4]:   `scales_h` — u16 LE, 2-bit high parts of 8 sub-block scales
//! - bytes[4..8]:   `scales_l` — 4 bytes, 4-bit low parts, two per byte
//! - bytes[8..136]: 128 nibble-bytes: low nibble = weight[2i], high = weight[2i+1]
//!
//! 8 sub-blocks of 32 weights each.  Sub-block scale (6-bit, centred at 0):
//! ```text
//! ls_low  = (scales_l[i/2] >> (4*(i&1))) & 0x0F
//! ls_high = (scales_h >> (2*i)) as u8 & 0x03
//! ls      = ls_low | (ls_high << 4)
//! ls_signed = ls.wrapping_sub(32)   // centred → [-32, 31]
//! scale = d * ls_signed
//! ```
//! `w = scale * KVALUES_IQ4NL[nibble]`
//!
//! The AVX2 kernel loads 8 pre-expanded floats per iteration and multiplies
//! with `_mm256_mul_ps`, providing ~4× throughput over pure scalar.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_shared::KVALUES_IQ4NL;
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ4_XS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ4_XS block: 2 (d) + 2 (scales_h) + 4 (scales_l) + 128 (nibbles) = 136.
pub const BLOCK_BYTES: usize = 136;
/// Number of sub-blocks per IQ4_XS block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per sub-block.
const SUB_BLOCK_SIZE: usize = 32;
/// Nibble-bytes per sub-block.
const NIBBLES_PER_SUB: usize = SUB_BLOCK_SIZE / 2; // 16

/// AVX2+FMA accelerated IQ4_XS kernel.
pub struct Iq4XsAvx2;

// ---------------------------------------------------------------------------
// Scale helpers (purely scalar — no AVX2 needed for 8 values)
// ---------------------------------------------------------------------------

#[inline(always)]
fn unpack_sub_scale(scales_h_u16: u16, scales_l: &[u8], i: usize) -> i32 {
    let ls_low: u8 = (scales_l[i / 2] >> (4 * (i & 1))) & 0x0F;
    let ls_high: u8 = (scales_h_u16 >> (2 * i)) as u8 & 0x03;
    let ls: u8 = ls_low | (ls_high << 4);
    (ls as i32).wrapping_sub(32)
}

// ---------------------------------------------------------------------------
// QuantKernel impl
// ---------------------------------------------------------------------------

impl QuantKernel for Iq4XsAvx2 {
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
        "IQ4_XS_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Inner AVX2 kernels
// ---------------------------------------------------------------------------

/// Expand one IQ4_XS sub-block (16 nibble-bytes → 32 f32) and store to `output`.
///
/// # Safety
/// Caller must have verified `avx2` is available.
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_sub_block(nibbles: &[u8], scale: f32, output: &mut [f32]) {
    // Expand 16 nibble-bytes → 32 raw tableau values in a staging array,
    // then AVX2-multiply by `scale` in 8-wide lanes.
    let mut staging = [0.0f32; SUB_BLOCK_SIZE];

    for i in 0..NIBBLES_PER_SUB {
        let byte = nibbles[i];
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        staging[i * 2] = KVALUES_IQ4NL[lo] as f32;
        staging[i * 2 + 1] = KVALUES_IQ4NL[hi] as f32;
    }

    let scale_vec = _mm256_set1_ps(scale);
    for chunk in 0..(SUB_BLOCK_SIZE / 8) {
        let src = _mm256_loadu_ps(staging.as_ptr().add(chunk * 8));
        let dst = _mm256_mul_ps(src, scale_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(chunk * 8), dst);
    }
}

/// Dequantize a full IQ4_XS block (256 weights).
///
/// # Safety
/// `block.len() >= BLOCK_BYTES` and `output.len() >= BLOCK_SIZE` must hold.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let nibbles = &block[8..BLOCK_BYTES];

    for sub in 0..N_SUPERBLOCKS {
        let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
        let scale = d * ls_signed as f32;

        let nibble_offset = sub * NIBBLES_PER_SUB;
        let weight_offset = sub * SUB_BLOCK_SIZE;

        decode_sub_block(
            &nibbles[nibble_offset..nibble_offset + NIBBLES_PER_SUB],
            scale,
            &mut output[weight_offset..weight_offset + SUB_BLOCK_SIZE],
        );
    }
}

/// GEMV for one row.
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
    // Scalar remainder for boundary chunks — avoids the 8× lane-duplication bug
    // that arises when using _mm256_set1_ps(scalar) + hsum.
    let mut scalar_rem = 0.0f32;
    let mut col = 0usize;

    for blk in 0..blocks_per_row {
        let block = &row_data[blk * BLOCK_BYTES..(blk + 1) * BLOCK_BYTES];
        let d = f16_to_f32(block);
        let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let nibbles = &block[8..BLOCK_BYTES];

        for sub in 0..N_SUPERBLOCKS {
            let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
            let scale = d * ls_signed as f32;
            let scale_vec = _mm256_set1_ps(scale);

            let nibble_offset = sub * NIBBLES_PER_SUB;
            let w_col_base = col + sub * SUB_BLOCK_SIZE;

            // Process in 8-wide chunks: 4 nibble-bytes → 8 weights → 8 input values.
            // SUB_BLOCK_SIZE=32, so 4 chunks of 8 per sub-block.
            for chunk in 0..(SUB_BLOCK_SIZE / 8) {
                let mut w8 = [0.0f32; 8];
                for i in 0..4 {
                    let byte = nibbles[nibble_offset + chunk * 4 + i];
                    w8[i * 2] = KVALUES_IQ4NL[(byte & 0x0F) as usize] as f32;
                    w8[i * 2 + 1] = KVALUES_IQ4NL[((byte >> 4) & 0x0F) as usize] as f32;
                }

                let c_base = w_col_base + chunk * 8;
                if c_base + 8 <= n_cols {
                    // Full 8-wide vectorised dot product: acc += (scale * w8) · x8
                    let w_vec = _mm256_loadu_ps(w8.as_ptr());
                    let sw_vec = _mm256_mul_ps(scale_vec, w_vec);
                    let x_vec = _mm256_loadu_ps(input.as_ptr().add(c_base));
                    acc = _mm256_fmadd_ps(sw_vec, x_vec, acc);
                } else {
                    // Scalar fallback for the partial chunk at the column boundary.
                    // Must NOT go through _mm256_set1_ps + hsum — that would count
                    // each element 8 times.
                    for j in 0..8usize {
                        let c = c_base + j;
                        if c < n_cols {
                            scalar_rem += scale * w8[j] * input[c];
                        }
                    }
                }
            }
        }

        col += BLOCK_SIZE;
    }

    hsum_f32_avx(acc) + scalar_rem
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::iq4_xs::Iq4XsRef;
    use crate::traits::QuantKernel;

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
            return;
        }
        let d = 1.0_f32;
        let block = make_zero_block(d);

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq4XsRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq4XsAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_with_scales() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.25_f32;
        let mut block = make_zero_block(d);
        // scales_h = 0x0001: sub-block 0 gets high bits = 0b01
        block[2] = 0x01;
        block[3] = 0x00;
        // scales_l[0] = 0x3F: sub-block 0 low = 0xF, sub-block 1 low = 0x3
        block[4] = 0x3F;
        // Set some nibble data for sub-block 0
        block[8] = 0xAB;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq4XsRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq4XsAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn gemv_matches_dequant_dot_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.5_f32;
        let mut block = make_zero_block(d);
        block[2] = 0xFF;
        block[3] = 0xFF;
        block[4] = 0xAA;

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq4XsAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.clone(),
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq4Xs,
        );
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Iq4XsAvx2.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-2,
            "gemv={}, expected={}",
            out[0],
            expected
        );
    }
}
