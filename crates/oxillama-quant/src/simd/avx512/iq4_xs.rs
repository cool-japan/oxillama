//! AVX-512 accelerated IQ4_XS quantization kernel.
//!
//! Mirrors `simd/avx2/iq4_xs.rs` with AVX-512 16-wide FMADD passes.
//!
//! Block layout (136 bytes per 256 weights, QK_K = 256):
//! - bytes[0..2]:   FP16 delta `d`
//! - bytes[2..4]:   `scales_h` — u16 LE (2-bit high parts of 8 sub-block scales)
//! - bytes[4..8]:   `scales_l` — 4 bytes (4-bit low parts)
//! - bytes[8..136]: 128 nibble-bytes (low nibble = weight[2i], high = weight[2i+1])
//!
//! Sub-block scale (6-bit, centred at 0):
//!   ls_signed = (ls_low | (ls_high << 4)).wrapping_sub(32)  // [-32, 31]
//!   scale = d * ls_signed
//!   w = scale * KVALUES_IQ4NL[nibble]

#![cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_shared::KVALUES_IQ4NL;
use crate::simd::avx512::util::{f16_to_f32, hsum_f32_avx512};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ4_XS: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ4_XS block: 2 + 2 + 4 + 128 = 136.
pub const BLOCK_BYTES: usize = 136;
/// Number of sub-blocks per IQ4_XS block.
const N_SUPERBLOCKS: usize = 8;
/// Weights per sub-block.
const SUB_BLOCK_SIZE: usize = 32;
/// Nibble-bytes per sub-block.
const NIBBLES_PER_SUB: usize = 16;

/// AVX-512 accelerated IQ4_XS kernel.
pub struct Iq4XsAvx512;

// ---------------------------------------------------------------------------
// Scale helper (purely scalar)
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

impl QuantKernel for Iq4XsAvx512 {
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
        "IQ4_XS"
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

fn scalar_dequant_block(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    use crate::reference::iq4_xs::Iq4XsRef;
    Iq4XsRef.dequant_block(block, output)
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
            let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let nibbles = &block[8..BLOCK_BYTES];
            for sub in 0..N_SUPERBLOCKS {
                let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
                let scale = d * ls_signed as f32;
                let nibble_off = sub * NIBBLES_PER_SUB;
                let col_base = blk * BLOCK_SIZE + sub * SUB_BLOCK_SIZE;
                for i in 0..NIBBLES_PER_SUB {
                    let byte = nibbles[nibble_off + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;
                    let c0 = col_base + i * 2;
                    let c1 = c0 + 1;
                    if c0 < n_cols {
                        sum += scale * KVALUES_IQ4NL[lo] as f32 * input[c0];
                    }
                    if c1 < n_cols {
                        sum += scale * KVALUES_IQ4NL[hi] as f32 * input[c1];
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

/// Expand one IQ4_XS sub-block (16 nibble-bytes → 32 f32) and store using AVX-512.
///
/// # Safety
/// CPU must support `avx512f`. `output.len() >= 32`.
#[target_feature(enable = "avx512f")]
unsafe fn decode_sub_block_avx512(nibbles: &[u8], scale: f32, output: &mut [f32]) {
    let mut staging = [0.0f32; SUB_BLOCK_SIZE];
    for i in 0..NIBBLES_PER_SUB {
        let byte = nibbles[i];
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        staging[i * 2] = KVALUES_IQ4NL[lo] as f32;
        staging[i * 2 + 1] = KVALUES_IQ4NL[hi] as f32;
    }

    // Two 16-wide AVX-512 passes to cover 32 values.
    let sv = _mm512_set1_ps(scale);
    // SAFETY: staging has 32 elements; output.len() >= 32.
    let s0 = _mm512_loadu_ps(staging.as_ptr());
    let s1 = _mm512_loadu_ps(staging.as_ptr().add(16));
    _mm512_storeu_ps(output.as_mut_ptr(), _mm512_mul_ps(s0, sv));
    _mm512_storeu_ps(output.as_mut_ptr().add(16), _mm512_mul_ps(s1, sv));
}

/// Dequantize one IQ4_XS block using AVX-512.
///
/// # Safety
/// - `block.len() >= BLOCK_BYTES`, `output.len() >= BLOCK_SIZE`
/// - CPU must support `avx512f`
#[target_feature(enable = "avx512f")]
unsafe fn dequant_block_avx512(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let scales_h_u16 = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let nibbles = &block[8..BLOCK_BYTES];

    for sub in 0..N_SUPERBLOCKS {
        let ls_signed = unpack_sub_scale(scales_h_u16, scales_l, sub);
        let scale = d * ls_signed as f32;
        let nibble_off = sub * NIBBLES_PER_SUB;
        let weight_off = sub * SUB_BLOCK_SIZE;
        decode_sub_block_avx512(
            &nibbles[nibble_off..nibble_off + NIBBLES_PER_SUB],
            scale,
            &mut output[weight_off..weight_off + SUB_BLOCK_SIZE],
        );
    }
}

/// GEMV for one row using AVX-512.
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
            let sv = _mm512_set1_ps(scale);
            let nibble_off = sub * NIBBLES_PER_SUB;
            let w_col_base = col + sub * SUB_BLOCK_SIZE;

            // Process in 16-wide chunks (2 nibble-bytes → 4 weights... but
            // SUB_BLOCK_SIZE=32, so 2 passes of 16 weights using 8 nibble-bytes each).
            for chunk in 0..2 {
                // 8 nibble-bytes → 16 weights
                let mut w16 = [0.0f32; 16];
                for i in 0..8 {
                    let byte = nibbles[nibble_off + chunk * 8 + i];
                    w16[i * 2] = KVALUES_IQ4NL[(byte & 0x0F) as usize] as f32;
                    w16[i * 2 + 1] = KVALUES_IQ4NL[((byte >> 4) & 0x0F) as usize] as f32;
                }
                let c_base = w_col_base + chunk * 16;
                if c_base + 16 <= n_cols {
                    // Full 16-wide vectorised path.
                    // SAFETY: c_base + 16 <= n_cols <= input.len()
                    let wv = _mm512_loadu_ps(w16.as_ptr());
                    let swv = _mm512_mul_ps(sv, wv);
                    let xv = _mm512_loadu_ps(input.as_ptr().add(c_base));
                    acc = _mm512_fmadd_ps(swv, xv, acc);
                } else {
                    for j in 0..16usize {
                        let c = c_base + j;
                        if c < n_cols {
                            scalar_rem += scale * w16[j] * input[c];
                        }
                    }
                }
            }
        }

        col += BLOCK_SIZE;
    }

    hsum_f32_avx512(acc) + scalar_rem
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64", feature = "simd-avx512"))]
mod tests {
    use super::*;
    use crate::reference::iq4_xs::Iq4XsRef;

    fn make_zero_block(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let [d0, d1] = half::f16::from_f32(d).to_le_bytes();
        block[0] = d0;
        block[1] = d1;
        block
    }

    fn make_varied_block(d: f32) -> Vec<u8> {
        let mut block = make_zero_block(d);
        // Set some non-trivial scales and nibble data.
        block[2] = 0xFF;
        block[3] = 0x3F;
        block[4] = 0xAB;
        block[5] = 0xCD;
        // Fill nibbles with varied data.
        for i in 8..BLOCK_BYTES {
            block[i] = ((i * 7 + 13) & 0xFF) as u8;
        }
        block
    }

    #[test]
    fn avx512_iq4_xs_dequant_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let block = make_varied_block(0.5);
        let mut out_avx512 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ref = vec![0.0f32; BLOCK_SIZE];
        Iq4XsAvx512
            .dequant_block(&block, &mut out_avx512)
            .expect("avx512 dequant");
        Iq4XsRef
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
    fn avx512_iq4_xs_matvec_q8_matches_reference() {
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
            oxillama_gguf::GgufTensorType::Iq4Xs,
        );
        let tensor_ref = crate::types::QuantTensor::new(
            data,
            vec![n_rows, n_cols],
            oxillama_gguf::GgufTensorType::Iq4Xs,
        );

        let input: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut out_avx512 = vec![0.0f32; n_rows];
        let mut out_ref = vec![0.0f32; n_rows];

        Iq4XsAvx512
            .gemv(&tensor_avx512, &input, &mut out_avx512)
            .expect("avx512 gemv");
        Iq4XsRef
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
    fn avx512_iq4_xs_kernel_metadata() {
        assert_eq!(Iq4XsAvx512.name(), "IQ4_XS");
        assert_eq!(Iq4XsAvx512.block_size(), BLOCK_SIZE);
        assert_eq!(Iq4XsAvx512.block_bytes(), BLOCK_BYTES);
    }
}
