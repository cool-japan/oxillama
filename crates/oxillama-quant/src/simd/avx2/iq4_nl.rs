//! AVX2+FMA accelerated IQ4_NL quantization kernel.
//!
//! IQ4_NL block layout (18 bytes per 32 weights):
//! - bytes[0..2]:   FP16 scale `d` (little-endian)
//! - bytes[2..18]:  16 nibble-bytes encoding 32 four-bit weights.
//!   Low nibble  = `weight[2i]`, High nibble = `weight[2i+1]`.
//!
//! Dequantisation: `w = d * KVALUES_IQ4NL[nibble]`
//!
//! The AVX2 kernel expands nibbles scalar-side (table lookup), then
//! multiplies 8 floats at a time using `_mm256_mul_ps`.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq_shared::KVALUES_IQ4NL;
use crate::simd::avx2::util::{f16_to_f32, hsum_f32_avx};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ4_NL: 32 weights per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per IQ4_NL block: 2 (FP16 scale) + 16 (nibble data).
pub const BLOCK_BYTES: usize = 18;
/// Number of nibble-bytes per block.
const NIBBLE_BYTES: usize = 16;

/// AVX2+FMA accelerated IQ4_NL kernel.
pub struct Iq4NlAvx2;

impl QuantKernel for Iq4NlAvx2 {
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
        "IQ4_NL_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Inner AVX2 kernels
// ---------------------------------------------------------------------------

/// Dequantize one IQ4_NL block (32 weights) using AVX2.
///
/// # Safety
/// `block.len() >= BLOCK_BYTES` and `output.len() >= BLOCK_SIZE` must hold.
/// `avx2` and `fma` CPU features must be available.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_block_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(block);
    let scale_vec = _mm256_set1_ps(d);

    // Expand 16 nibble-bytes → 32 floats via scalar table lookup.
    let mut vals = [0.0f32; BLOCK_SIZE];
    for i in 0..NIBBLE_BYTES {
        let byte = *block.get_unchecked(2 + i);
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        vals[i * 2] = *KVALUES_IQ4NL.get_unchecked(lo) as f32;
        vals[i * 2 + 1] = *KVALUES_IQ4NL.get_unchecked(hi) as f32;
    }

    // Multiply in 4 chunks of 8 using AVX2.
    for chunk in 0..(BLOCK_SIZE / 8) {
        let src = _mm256_loadu_ps(vals.as_ptr().add(chunk * 8));
        let dst = _mm256_mul_ps(src, scale_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(chunk * 8), dst);
    }
}

/// Compute one row of gemv for IQ4_NL using AVX2+FMA.
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

        dequant_block_avx2(block, &mut buf);

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
    use crate::reference::iq4_nl::Iq4NlRef;

    /// Build a synthetic IQ4_NL block with scale `d` and all nibbles = 0.
    fn make_zero_block(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_f16 = half::f16::from_f32(d);
        let bytes = d_f16.to_le_bytes();
        block[0] = bytes[0];
        block[1] = bytes[1];
        block
    }

    #[test]
    fn zero_block_dequant() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_zero_block(1.0);
        let mut output = vec![0.0f32; BLOCK_SIZE];
        Iq4NlAvx2.dequant_block(&block, &mut output).unwrap();
        // All nibbles = 0 → KVALUES_IQ4NL[0] * 1.0 for every element.
        let expected = KVALUES_IQ4NL[0] as f32;
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "mismatch at [{i}]: got={v}, expected={expected}"
            );
        }
    }

    #[test]
    fn matches_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 0.5_f32;
        let mut block = make_zero_block(d);
        // Set some nibble patterns.
        block[2] = 0xAB;
        block[3] = 0x12;
        block[10] = 0xFF;

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq4NlRef.dequant_block(&block, &mut ref_out).unwrap();

        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq4NlAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-5, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn gemv_matches_dequant_dot_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 1.0_f32;
        let mut block = make_zero_block(d);
        block[2] = 0x37;
        block[5] = 0xF0;

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq4NlAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let input = vec![1.0f32; BLOCK_SIZE];
        let tensor = crate::types::QuantTensor::new(
            block.clone(),
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq4Nl,
        );
        let mut output = vec![0.0f32; 1];
        Iq4NlAvx2.gemv(&tensor, &input, &mut output).unwrap();

        assert!(
            (output[0] - expected).abs() < 1e-4,
            "gemv={}, expected={}",
            output[0],
            expected
        );
    }
}
