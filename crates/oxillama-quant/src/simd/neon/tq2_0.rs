//! TQ2_0 NEON-optimised kernel.
//!
//! Block format (66 bytes / 256 weights):
//! - bytes[0..64]: qs — 256 × 2-bit ternary codes packed (4 per byte)
//! - bytes[64..66]: FP16 scale `d`
//!
//! Each 2-bit code: 0→-1, 1→0, 2→+1.  Weight = d * (code - 1).

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 66;

/// NEON-accelerated TQ2_0 kernel.
#[allow(non_camel_case_types)]
pub struct Tq2_0Neon;

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn decode_block(block: &[u8], output: &mut [f32]) {
    let qs = &block[0..64];
    let d = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));

    for (i, &byte) in qs.iter().enumerate() {
        let v0 = (byte & 3) as i32 - 1;
        let v1 = ((byte >> 2) & 3) as i32 - 1;
        let v2 = ((byte >> 4) & 3) as i32 - 1;
        let v3 = ((byte >> 6) & 3) as i32 - 1;
        output[i * 4] = d * v0 as f32;
        output[i * 4 + 1] = d * v1 as f32;
        output[i * 4 + 2] = d * v2 as f32;
        output[i * 4 + 3] = d * v3 as f32;
    }
}

impl QuantKernel for Tq2_0Neon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "TQ2_0-NEON"
    }

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
        decode_block(block, output);
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
        let mut scratch = [0.0f32; BLOCK_SIZE];

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            // SAFETY: AArch64 with NEON.
            let mut sum = unsafe { vdupq_n_f32(0.0) };

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * BLOCK_BYTES;
                let block = &quant_matrix.data[bo..bo + BLOCK_BYTES];
                let input_base = blk * BLOCK_SIZE;
                let block_input_len = BLOCK_SIZE.min(n_cols.saturating_sub(input_base));

                decode_block(block, &mut scratch);

                // SAFETY: scratch and input are valid; AArch64 with NEON.
                unsafe {
                    let w_ptr = scratch.as_ptr();
                    let i_ptr = input.as_ptr().add(input_base);
                    let lanes = block_input_len / 4;
                    for k in 0..lanes {
                        let off = k * 4;
                        let wv = vld1q_f32(w_ptr.add(off));
                        let iv = vld1q_f32(i_ptr.add(off));
                        sum = vfmaq_f32(sum, wv, iv);
                    }
                    for k in (lanes * 4)..block_input_len {
                        let s: f32 = scratch[k] * input[input_base + k];
                        sum = vaddq_f32(sum, vdupq_n_f32(s));
                    }
                }
            }

            // SAFETY: AArch64 with NEON.
            *out = unsafe { vaddvq_f32(sum) };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::tq2_0::Tq2_0Ref;
    use oxillama_gguf::GgufTensorType;

    fn make_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[64] = (d_bits & 0xff) as u8;
        block[65] = (d_bits >> 8) as u8;
        block
    }

    #[test]
    fn test_dequant_block_basic() {
        let block = make_zero_block();
        let mut out = vec![0.0f32; BLOCK_SIZE];
        Tq2_0Neon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        // Use only values with bits in [0,1,2] packed — avoid value 3 (invalid ternary)
        for (i, b) in block[..64].iter_mut().enumerate() {
            // Pack deterministic valid ternary codes (0,1,2 only — not 3)
            let v0 = (i as u8) % 3;
            let v1 = (i as u8 + 1) % 3;
            let v2 = (i as u8 + 2) % 3;
            let v3 = (i as u8 + 1) % 3;
            *b = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
        }

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Tq2_0Neon
            .dequant_block(&block, &mut neon_out)
            .expect("neon failed");
        Tq2_0Ref
            .dequant_block(&block, &mut ref_out)
            .expect("ref failed");

        for (i, (&n, &r)) in neon_out.iter().zip(ref_out.iter()).enumerate() {
            assert!((n - r).abs() < 1e-5, "mismatch at {i}: neon={n} ref={r}");
        }
    }

    #[test]
    fn test_gemv_single_row() {
        let block = make_zero_block();
        let data = block.clone();
        let tensor = QuantTensor {
            data,
            shape: vec![1, BLOCK_SIZE],
            tensor_type: GgufTensorType::Tq2_0,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Tq2_0Neon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
