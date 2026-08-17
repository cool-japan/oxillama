//! TQ1_0 NEON-optimised kernel.
//!
//! Block format (54 bytes / 256 weights):
//! - bytes[0..48]:  qs — 48 bytes, each encodes 5 ternary digits
//! - bytes[48..52]: qh — 4 bytes, each encodes 4 ternary digits
//! - bytes[52..54]: FP16 scale `d`
//!
//! Ternary encoding: {0→-1, 1→0, 2→+1}. Weight = d * ternary.
//!
//! Both `qs` and `qh` use upstream's *fixed-point* base-3 packing — the stored
//! byte is `ceil(q * 256 / 243)` for the digit-tuple value `q`, and digit `n`
//! is recovered with `((u8)(byte * 3^n) as u16 * 3) >> 8`.  Decode order is
//! digit-major.  See [`crate::reference::tq1_0`] for the full derivation.

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

/// Powers of three, as the `uint8_t pow3[]` upstream declares.
const POW3: [u8; 5] = [1, 3, 9, 27, 81];
/// Ternary digits packed into one `qs` byte.
const QS_DIGITS: usize = 5;
/// Ternary digits packed into one `qh` byte.
const QH_DIGITS: usize = 4;

/// Recover ternary digit `digit` from a packed TQ1_0 byte.
///
/// Port of upstream's `q = byte * pow3[n]; xi = ((uint16_t) q * 3) >> 8;`,
/// including the `uint8_t` wrap that discards the more significant digits.
#[inline]
fn decode_trit(byte: u8, digit: usize) -> i8 {
    let q = byte.wrapping_mul(POW3[digit]);
    ((((q as u16) * 3) >> 8) as i8) - 1
}

fn decode_block(block: &[u8], output: &mut [f32]) {
    let d = half::f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]).to_f32();

    // Decode qs: 48 bytes → 240 ternary values, digit-major within each group
    // (one 32-byte group then one 16-byte group, upstream's split).
    let mut out_idx = 0usize;
    let mut j = 0usize;
    let qs_head = QS_BYTES - QS_BYTES % 32;
    while j < qs_head {
        for digit in 0..QS_DIGITS {
            for m in 0..32 {
                output[out_idx] = d * decode_trit(block[j + m], digit) as f32;
                out_idx += 1;
            }
        }
        j += 32;
    }
    while j < QS_BYTES {
        for digit in 0..QS_DIGITS {
            for m in 0..16 {
                output[out_idx] = d * decode_trit(block[j + m], digit) as f32;
                out_idx += 1;
            }
        }
        j += 16;
    }

    // Decode qh: 4 bytes → 16 ternary values, also digit-major.
    for digit in 0..QH_DIGITS {
        for m in 0..QH_BYTES {
            output[out_idx] = d * decode_trit(block[QH_OFFSET + m], digit) as f32;
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
        crate::parallel::for_each_row(output, n_rows, n_cols, |row, out| {
            // Per-row scratch: the closure may run on several threads at once.
            let mut scratch = [0.0f32; BLOCK_SIZE];
            let row_start = row * row_bytes;
            // SAFETY: AArch64 with NEON.
            let mut sum = unsafe { vdupq_n_f32(0.0) };
            // Separate scalar accumulator for the sub-4-lane remainder.
            // Folding it in as `vaddq_f32(sum, vdupq_n_f32(s))` would put `s`
            // in all four lanes, and the closing `vaddvq_f32` would then count
            // it four times — the same tail-accumulation bug already fixed in
            // `simd/neon/iq2_xxs.rs`.
            let mut scalar_tail = 0.0f32;

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
                        scalar_tail += scratch[k] * input[input_base + k];
                    }
                }
            }

            // SAFETY: AArch64 with NEON.
            *out = unsafe { vaddvq_f32(sum) } + scalar_tail;
        });

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
            data: data.into(),
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
