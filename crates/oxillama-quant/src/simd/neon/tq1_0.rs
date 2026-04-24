//! TQ1_0 NEON-optimised kernel.
//!
//! Block format (54 bytes / 256 weights):
//! - bytes[0..48]:  qs — 48 bytes, each encodes 5 ternary values in base-3
//! - bytes[48..52]: qh — 4 bytes, each encodes 4 ternary values as 2-bit codes
//! - bytes[52..54]: FP16 scale `d`
//!
//! Ternary encoding: {0→-1, 1→0, 2→+1}. Weight = d * ternary.

#![cfg(all(feature = "simd-neon", target_arch = "aarch64"))]

use core::arch::aarch64::*;

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 54;
const QS_BYTES: usize = 48;
const QH_BYTES: usize = 4;
const QH_OFFSET: usize = QS_BYTES;
const D_OFFSET: usize = QS_BYTES + QH_BYTES;

/// NEON-accelerated TQ1_0 kernel.
#[allow(non_camel_case_types)]
pub struct Tq1_0Neon;

/// Decode a single `qs` byte into 5 ternary values (-1, 0, or +1).
#[inline]
fn decode_qs_byte(byte: u8) -> [i8; 5] {
    let mut q = byte as u16;
    let mut out = [0i8; 5];
    for v in &mut out {
        *v = (q % 3) as i8 - 1;
        q /= 3;
    }
    out
}

/// Decode a single `qh` byte into 4 ternary values (-1, 0, or +1).
#[inline]
fn decode_qh_byte(byte: u8) -> [i8; 4] {
    [
        (byte & 0x03) as i8 - 1,
        ((byte >> 2) & 0x03) as i8 - 1,
        ((byte >> 4) & 0x03) as i8 - 1,
        ((byte >> 6) & 0x03) as i8 - 1,
    ]
}

fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]).to_f32();

    // Decode qs: 48 bytes → 240 ternary values
    let mut out_idx = 0usize;
    for &qs_byte in &block[..QS_BYTES] {
        let vals = decode_qs_byte(qs_byte);
        for &v in &vals {
            output[out_idx] = d * v as f32;
            out_idx += 1;
        }
    }

    // Decode qh: 4 bytes → 16 ternary values
    for &qh_byte in &block[QH_OFFSET..QH_OFFSET + QH_BYTES] {
        let vals = decode_qh_byte(qh_byte);
        for &v in &vals {
            output[out_idx] = d * v as f32;
            out_idx += 1;
        }
    }
}

impl QuantKernel for Tq1_0Neon {
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn name(&self) -> &'static str {
        "TQ1_0-NEON"
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
    use crate::reference::tq1_0::Tq1_0Ref;
    use oxillama_gguf::GgufTensorType;

    fn make_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_BYTES];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[D_OFFSET] = (d_bits & 0xff) as u8;
        block[D_OFFSET + 1] = (d_bits >> 8) as u8;
        block
    }

    #[test]
    fn test_dequant_block_basic() {
        let block = make_zero_block();
        let mut out = vec![0.0f32; BLOCK_SIZE];
        Tq1_0Neon
            .dequant_block(&block, &mut out)
            .expect("dequant failed");
        assert_eq!(out.len(), BLOCK_SIZE);
    }

    #[test]
    fn test_dequant_cross_validate() {
        let mut block = make_zero_block();
        // Fill qs with values in base-3 range [0..242] for validity
        for (i, b) in block[..QS_BYTES].iter_mut().enumerate() {
            *b = ((i * 7 + 3) % 243) as u8;
        }
        block[QH_OFFSET] = 0b10_01_00_10;

        let mut neon_out = vec![0.0f32; BLOCK_SIZE];
        let mut ref_out = vec![0.0f32; BLOCK_SIZE];

        Tq1_0Neon
            .dequant_block(&block, &mut neon_out)
            .expect("neon failed");
        Tq1_0Ref
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
            tensor_type: GgufTensorType::Tq1_0,
        };
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut out = vec![0.0f32; 1];
        Tq1_0Neon
            .gemv(&tensor, &input, &mut out)
            .expect("gemv failed");
    }
}
