//! TQ1_0 reference (naive) implementation — Ternary Quantization 1-bit, type 0.
//!
//! TQ1_0 block format (54 bytes per 256 weights):
//! - 48 bytes (`qs`): 5 ternary digits per byte (48 × 5 = 240 values).
//! - 4 bytes (`qh`): the remaining 16 values, 4 ternary digits per byte.
//! - 2 bytes (`d`): FP16 scale factor.
//!
//! Ternary encoding: `{0 → -1, 1 → 0, 2 → +1}` (i.e. `digit - 1`).
//!
//! Final weight: `w = d * ternary_value`.
//!
//! # Encoding: base-3 in *fixed point*, not plain base-3
//!
//! TQ1_0 is **not** a plain base-3 packing.  `quantize_row_tq1_0_ref` builds
//! the 5-digit value most-significant-digit-first,
//! `q = Σ_n xi_n · 3^(4-n)` in `0..=242`, and then stores the scaled
//! ceiling `ceil(q · 256 / 243)`.  `dequantize_row_tq1_0` recovers digit `n`
//! from the stored byte with
//!
//! ```text
//! let q  = stored_byte.wrapping_mul(3^n);   // u8 arithmetic — wraps mod 256
//! let xi = ((q as u16) * 3) >> 8;           // leading base-3 digit, 0..=2
//! ```
//!
//! Decoding the stored byte with naive `% 3` / `/ 3` arithmetic produces
//! *different values*, not merely a different order: the all-`+1` block stores
//! `ceil(242·256/243) = 255`, which upstream decodes to `+1,+1,+1,+1,+1` and
//! the naive decomposition decodes to `-1,0,0,-1,-1`.
//!
//! `qh` uses the **same** base-3 decomposition — it packs 4 digits shifted up
//! one position (`q *= 3` after the accumulate loop), *not* four 2-bit fields.
//!
//! # Decode order
//!
//! Upstream is **digit-major**: for each 32-byte group it emits digit 0 of all
//! 32 bytes, then digit 1 of all 32, and so on.  So `qs[m]`'s five digits land
//! at output indices `m`, `m + 32`, `m + 64`, `m + 96`, `m + 128` — 32 apart,
//! not adjacent.  `sizeof(qs) = 48` splits into one 32-byte group (160 values)
//! and one 16-byte group (80 values); `qh`'s 4 bytes then contribute
//! `4 digits × 4 bytes = 16` values, also digit-major.
//!
//! This format is part of the llama.cpp ecosystem (GgufTensorType value 34)
//! and supports BitNet b1.58 and similar ternary-weight models.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for TQ1_0: 256 weights per block.
const TQ1_0_BLOCK_SIZE: usize = 256;
/// Bytes per TQ1_0 block: 48 (qs) + 4 (qh) + 2 (d) = 54.
const TQ1_0_BLOCK_BYTES: usize = 54;
/// Number of qs bytes (each encodes 5 ternary values in base-3).
const TQ1_0_QS_BYTES: usize = 48;
/// Number of qh bytes (each encodes 4 ternary values in 2-bit codes).
const TQ1_0_QH_BYTES: usize = 4;
/// Offset to `qh` in the block.
const TQ1_0_QH_OFFSET: usize = TQ1_0_QS_BYTES;
/// Offset to `d` (FP16 scale) in the block.
const TQ1_0_D_OFFSET: usize = TQ1_0_QS_BYTES + TQ1_0_QH_BYTES;

/// Reference (naive scalar) TQ1_0 kernel.
///
/// Implements ternary quantization where each weight is one of {-1, 0, +1}
/// multiplied by a shared FP16 scale factor.
pub struct Tq1_0Ref;

/// Powers of three used by the fixed-point base-3 decode, as `u8` exactly as
/// upstream declares them (`static const uint8_t pow3[6]`).
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Number of ternary digits packed into one `qs` byte.
const TQ1_0_QS_DIGITS: usize = 5;
/// Number of ternary digits packed into one `qh` byte.
const TQ1_0_QH_DIGITS: usize = 4;

/// Recover ternary digit `digit` from a packed TQ1_0 byte.
///
/// Literal port of upstream's two-line kernel:
///
/// ```c
/// uint8_t q  = x[i].qs[j + m] * pow3[n];   // uint8_t: wraps mod 256
/// int16_t xi = ((uint16_t) q * 3) >> 8;    // leading base-3 digit
/// *y++ = (float) (xi - 1) * d;
/// ```
///
/// Multiplying by `3^digit` shifts the requested digit into the leading
/// position (the `u8` wrap discards the more significant digits), and the
/// `(·3) >> 8` reads it back out of the fixed-point representation.
#[inline]
fn decode_trit(byte: u8, digit: usize) -> i8 {
    let q = byte.wrapping_mul(POW3[digit]);
    ((((q as u16) * 3) >> 8) as i8) - 1
}

/// Convert an IEEE 754 FP16 half-precision value to FP32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

impl QuantKernel for Tq1_0Ref {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < TQ1_0_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: TQ1_0_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < TQ1_0_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: TQ1_0_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([
            block[TQ1_0_D_OFFSET],
            block[TQ1_0_D_OFFSET + 1],
        ]));

        // Decode qs: 48 bytes → 240 ternary values, digit-major within each
        // group.  48 = one 32-byte group + one 16-byte group, exactly as
        // upstream's `sizeof(qs) - sizeof(qs) % 32` split.
        let mut out_idx = 0;
        let mut j = 0usize;
        let qs_head = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;
        while j < qs_head {
            for digit in 0..TQ1_0_QS_DIGITS {
                for m in 0..32 {
                    output[out_idx] = d * decode_trit(block[j + m], digit) as f32;
                    out_idx += 1;
                }
            }
            j += 32;
        }
        while j < TQ1_0_QS_BYTES {
            for digit in 0..TQ1_0_QS_DIGITS {
                for m in 0..16 {
                    output[out_idx] = d * decode_trit(block[j + m], digit) as f32;
                    out_idx += 1;
                }
            }
            j += 16;
        }

        // Decode qh: 4 bytes → 16 ternary values, also digit-major.
        for digit in 0..TQ1_0_QH_DIGITS {
            for m in 0..TQ1_0_QH_BYTES {
                output[out_idx] = d * decode_trit(block[TQ1_0_QH_OFFSET + m], digit) as f32;
                out_idx += 1;
            }
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

        let blocks_per_row = n_cols.div_ceil(TQ1_0_BLOCK_SIZE);
        let row_bytes = blocks_per_row * TQ1_0_BLOCK_BYTES;

        crate::parallel::for_each_row(output, n_rows, n_cols, |row, out| {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let bo = row_start + blk * TQ1_0_BLOCK_BYTES;
                let data = &quant_matrix.data;
                let d = f16_to_f32(u16::from_le_bytes([
                    data[bo + TQ1_0_D_OFFSET],
                    data[bo + TQ1_0_D_OFFSET + 1],
                ]));
                let input_offset = blk * TQ1_0_BLOCK_SIZE;
                let inp = &input[input_offset..];

                // Inline dot product for qs portion (240 values), in the same
                // digit-major order `dequant_block` writes.
                let mut in_off = 0;
                let mut j = 0usize;
                let qs_head = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;
                while j < qs_head {
                    for digit in 0..TQ1_0_QS_DIGITS {
                        for m in 0..32 {
                            if input_offset + in_off < n_cols {
                                let v = decode_trit(data[bo + j + m], digit);
                                sum += d * v as f32 * inp[in_off];
                            }
                            in_off += 1;
                        }
                    }
                    j += 32;
                }
                while j < TQ1_0_QS_BYTES {
                    for digit in 0..TQ1_0_QS_DIGITS {
                        for m in 0..16 {
                            if input_offset + in_off < n_cols {
                                let v = decode_trit(data[bo + j + m], digit);
                                sum += d * v as f32 * inp[in_off];
                            }
                            in_off += 1;
                        }
                    }
                    j += 16;
                }

                // Inline dot product for qh portion (16 values)
                for digit in 0..TQ1_0_QH_DIGITS {
                    for m in 0..TQ1_0_QH_BYTES {
                        if input_offset + in_off < n_cols {
                            let v = decode_trit(data[bo + TQ1_0_QH_OFFSET + m], digit);
                            sum += d * v as f32 * inp[in_off];
                        }
                        in_off += 1;
                    }
                }
            }

            *out = sum;
        });

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
        TQ1_0_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        TQ1_0_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "TQ1_0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a TQ1_0 block from raw qs, qh, and scale.
    fn make_tq1_0_block(scale: f32, qs: &[u8; 48], qh: &[u8; 4]) -> Vec<u8> {
        let mut block = Vec::with_capacity(TQ1_0_BLOCK_BYTES);
        block.extend_from_slice(qs);
        block.extend_from_slice(qh);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block
    }

    /// Encode 5 ternary values into one `qs` byte — literal port of the inner
    /// loop of upstream `quantize_row_tq1_0_ref`.
    ///
    /// `vals[n]` becomes digit `n`, i.e. the value the decoder recovers with
    /// `decode_trit(byte, n)`.  Digits are accumulated most-significant-first
    /// and the result is stored as the fixed-point ceiling
    /// `ceil(q · 256 / 243)`, which is what makes the `(·3) >> 8` decode exact.
    fn encode_qs(vals: [i8; 5]) -> u8 {
        let mut q: u8 = 0;
        for &v in &vals {
            q = q * 3 + (v + 1) as u8;
        }
        // ceiling division by 243 == 3^5
        ((q as u16) * 256).div_ceil(243) as u8
    }

    /// Encode 4 ternary values into one `qh` byte — literal port of upstream's
    /// `qh` loop, which uses the *same* base-3 scheme as `qs` but with the
    /// four digits shifted up one position (`q *= 3`), leaving the least
    /// significant trit unused.
    fn encode_qh(vals: [i8; 4]) -> u8 {
        let mut q: u8 = 0;
        for &v in &vals {
            q = q * 3 + (v + 1) as u8;
        }
        q *= 3; // shift the first value to the most significant trit
        ((q as u16) * 256).div_ceil(243) as u8
    }

    /// Output index that carries digit `digit` of `qs[byte_index]`.
    ///
    /// `qs` is decoded digit-major in a 32-byte group (bytes 0..32, output
    /// 0..160) followed by a 16-byte group (bytes 32..48, output 160..240).
    fn qs_out_index(byte_index: usize, digit: usize) -> usize {
        if byte_index < 32 {
            digit * 32 + byte_index
        } else {
            160 + digit * 16 + (byte_index - 32)
        }
    }

    /// Output index that carries digit `digit` of `qh[byte_index]`.
    fn qh_out_index(byte_index: usize, digit: usize) -> usize {
        240 + digit * 4 + byte_index
    }

    #[test]
    fn test_dequant_zeros() {
        // d=0 → all outputs should be zero regardless of ternary values
        let qs = [encode_qs([1, 1, -1, 0, 1]); 48];
        let qh = [encode_qh([1, -1, 0, 1]); 4];
        let block = make_tq1_0_block(0.0, &qs, &qh);
        let kernel = Tq1_0Ref;
        let mut output = vec![f32::NAN; TQ1_0_BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant zeros");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-7, "output[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_dequant_positive() {
        // d=1.0, all ternary values = +1
        // qs byte for [+1,+1,+1,+1,+1]: each digit is 2 → 2 + 2*3 + 2*9 + 2*27 + 2*81 = 242
        let qs = [encode_qs([1, 1, 1, 1, 1]); 48];
        let qh = [encode_qh([1, 1, 1, 1]); 4];
        let block = make_tq1_0_block(1.0, &qs, &qh);
        let kernel = Tq1_0Ref;
        let mut output = vec![0.0f32; TQ1_0_BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant positive");
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "output[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn test_dequant_negative() {
        // d=1.0, all ternary values = -1
        // qs byte for [-1,-1,-1,-1,-1]: each digit is 0 → 0
        let qs = [encode_qs([-1, -1, -1, -1, -1]); 48];
        let qh = [encode_qh([-1, -1, -1, -1]); 4];
        let block = make_tq1_0_block(1.0, &qs, &qh);
        let kernel = Tq1_0Ref;
        let mut output = vec![0.0f32; TQ1_0_BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant negative");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-3,
                "output[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_mixed() {
        // d=2.0, encode known pattern and verify.  Every byte carries the same
        // five digits, so the digit-major decode order makes the output
        // piecewise constant in runs of 32 (then 16) rather than repeating
        // every five weights.
        let qs_digits = [-1i8, 0, 1, -1, 0];
        let qh_digits = [1i8, 0, -1, 1];
        let qs = [encode_qs(qs_digits); 48];
        let qh = [encode_qh(qh_digits); 4];
        let block = make_tq1_0_block(2.0, &qs, &qh);
        let kernel = Tq1_0Ref;
        let mut output = vec![0.0f32; TQ1_0_BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant mixed");

        for byte_index in 0..48 {
            for (digit, &v) in qs_digits.iter().enumerate() {
                let idx = qs_out_index(byte_index, digit);
                let exp = 2.0 * v as f32;
                assert!(
                    (output[idx] - exp).abs() < 1e-2,
                    "qs byte {byte_index} digit {digit} → output[{idx}] = {}, expected {exp}",
                    output[idx]
                );
            }
        }
        for byte_index in 0..4 {
            for (digit, &v) in qh_digits.iter().enumerate() {
                let idx = qh_out_index(byte_index, digit);
                let exp = 2.0 * v as f32;
                assert!(
                    (output[idx] - exp).abs() < 1e-2,
                    "qh byte {byte_index} digit {digit} → output[{idx}] = {}, expected {exp}",
                    output[idx]
                );
            }
        }
    }

    #[test]
    fn test_gemv_tq1_0() {
        let kernel = Tq1_0Ref;

        // Build a 1-row, 256-col matrix with all +1 ternary values, d=0.5
        let qs = [encode_qs([1, 1, 1, 1, 1]); 48];
        let qh = [encode_qh([1, 1, 1, 1]); 4];
        let block = make_tq1_0_block(0.5, &qs, &qh);
        let tensor = QuantTensor::new(
            block.clone(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Tq1_0,
        );

        // Input = all 1.0 → dot product = 256 * 0.5 = 128.0
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: gemv all +1");
        assert!(
            (output[0] - 128.0).abs() < 0.5,
            "got {}, expected 128.0",
            output[0]
        );

        // Verify gemv against dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut dequant)
            .expect("test: dequant for reference");
        let mut ref_dot = 0.0f32;
        for (w, x) in dequant.iter().zip(input.iter()) {
            ref_dot += w * x;
        }
        assert!(
            (output[0] - ref_dot).abs() < 1e-3,
            "gemv={}, dequant ref={}",
            output[0],
            ref_dot
        );
    }

    #[test]
    fn test_gemv_against_dequant_varied() {
        let kernel = Tq1_0Ref;

        // Build varied ternary pattern
        let mut qs = [0u8; 48];
        for (i, byte) in qs.iter_mut().enumerate() {
            // Cycle through different patterns
            let pattern = match i % 3 {
                0 => [-1, 0, 1, -1, 0],
                1 => [1, 1, -1, 0, 0],
                _ => [0, -1, 1, 1, -1],
            };
            *byte = encode_qs(pattern);
        }
        let mut qh = [0u8; 4];
        for (i, byte) in qh.iter_mut().enumerate() {
            let pattern = match i % 2 {
                0 => [1, -1, 0, 1],
                _ => [-1, 0, 1, -1],
            };
            *byte = encode_qh(pattern);
        }
        let block = make_tq1_0_block(0.75, &qs, &qh);

        // Dequant reference
        let mut dequant = vec![0.0f32; 256];
        kernel
            .dequant_block(&block, &mut dequant)
            .expect("test: dequant varied");

        // Build input with varied values
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();

        // Reference dot product
        let ref_dot: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // GEMV
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Tq1_0);
        let mut output = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input, &mut output)
            .expect("test: gemv varied");

        assert!(
            (output[0] - ref_dot).abs() < 1e-3,
            "gemv={}, ref={}",
            output[0],
            ref_dot
        );
    }

    #[test]
    fn test_gemm_tq1_0() {
        let kernel = Tq1_0Ref;

        // 2 rows, 256 cols
        let qs_a = [encode_qs([1, 1, 1, 1, 1]); 48]; // all +1
        let qh_a = [encode_qh([1, 1, 1, 1]); 4];
        let block_a = make_tq1_0_block(1.0, &qs_a, &qh_a);

        let qs_b = [encode_qs([-1, -1, -1, -1, -1]); 48]; // all -1
        let qh_b = [encode_qh([-1, -1, -1, -1]); 4];
        let block_b = make_tq1_0_block(1.0, &qs_b, &qh_b);

        let mut data = Vec::new();
        data.extend_from_slice(&block_a);
        data.extend_from_slice(&block_b);
        let tensor = QuantTensor::new(data, vec![2, 256], oxillama_gguf::GgufTensorType::Tq1_0);

        // 1 input row of 256 ones
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        kernel
            .gemm(&tensor, &input, &mut output, 1, 2, 256)
            .expect("test: gemm tq1_0");

        // Row 0 (all +1, d=1): dot = 256
        assert!(
            (output[0] - 256.0).abs() < 1.0,
            "row0: got {}, expected 256",
            output[0]
        );
        // Row 1 (all -1, d=1): dot = -256
        assert!(
            (output[1] - (-256.0)).abs() < 1.0,
            "row1: got {}, expected -256",
            output[1]
        );
    }

    #[test]
    fn test_block_too_small_errors() {
        let kernel = Tq1_0Ref;
        let block = vec![0u8; 10]; // too small
        let mut output = vec![0.0f32; TQ1_0_BLOCK_SIZE];
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "short block should error"
        );
    }

    #[test]
    fn test_output_too_small_errors() {
        let kernel = Tq1_0Ref;
        let block = vec![0u8; TQ1_0_BLOCK_BYTES];
        let mut output = vec![0.0f32; 10]; // too small
        assert!(
            kernel.dequant_block(&block, &mut output).is_err(),
            "short output should error"
        );
    }

    /// Every one of the 3^5 = 243 `qs` digit tuples and 3^4 = 81 `qh` tuples
    /// survives upstream's encode → decode exactly.
    ///
    /// This is the property the fixed-point `ceil(q·256/243)` / `(·3) >> 8`
    /// pair exists to guarantee, and it is exhaustive: a decoder using naive
    /// `% 3` arithmetic fails it on the very first non-trivial tuple.
    #[test]
    fn test_encode_decode_roundtrip() {
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                for c in -1i8..=1 {
                    for d_val in -1i8..=1 {
                        for e in -1i8..=1 {
                            let vals = [a, b, c, d_val, e];
                            let encoded = encode_qs(vals);
                            let decoded: [i8; 5] = std::array::from_fn(|n| decode_trit(encoded, n));
                            assert_eq!(
                                vals, decoded,
                                "roundtrip failed for {vals:?}: encoded={encoded}, decoded={decoded:?}"
                            );
                        }
                    }
                }
            }
        }

        for a in -1i8..=1 {
            for b in -1i8..=1 {
                for c in -1i8..=1 {
                    for d_val in -1i8..=1 {
                        let vals = [a, b, c, d_val];
                        let encoded = encode_qh(vals);
                        let decoded: [i8; 4] = std::array::from_fn(|n| decode_trit(encoded, n));
                        assert_eq!(
                            vals, decoded,
                            "qh roundtrip failed for {vals:?}: encoded={encoded}, decoded={decoded:?}"
                        );
                    }
                }
            }
        }
    }

    /// The all-`+1` block is the cheapest discriminator between upstream's
    /// fixed-point base-3 decode and a naive `% 3` decomposition.
    ///
    /// `q = 242` → stored `ceil(242·256/243) = 255`.  Upstream decodes `255`
    /// to five `+1`s; the naive decomposition of `255` in base 3 is
    /// `0,1,1,0,0` → `-1,0,0,-1,-1`.
    #[test]
    fn test_all_plus_one_stores_0xff() {
        assert_eq!(encode_qs([1, 1, 1, 1, 1]), 0xFF);
        assert_eq!(encode_qh([1, 1, 1, 1]), 253);
        for n in 0..5 {
            assert_eq!(decode_trit(0xFF, n), 1, "digit {n} of 0xFF");
        }
        for n in 0..4 {
            assert_eq!(decode_trit(253, n), 1, "digit {n} of 253");
        }
    }

    #[test]
    fn test_constants() {
        let kernel = Tq1_0Ref;
        assert_eq!(kernel.block_size(), 256);
        assert_eq!(kernel.block_bytes(), 54);
        assert_eq!(kernel.name(), "TQ1_0");
    }
}
