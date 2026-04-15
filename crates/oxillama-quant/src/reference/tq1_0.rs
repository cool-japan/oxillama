//! TQ1_0 reference (naive) implementation — Ternary Quantization 1-bit, type 0.
//!
//! TQ1_0 block format (54 bytes per 256 weights):
//! - 48 bytes (`qs`): base-3 packed ternary values. Each byte encodes 5 ternary
//!   values via repeated mod-3/div-3 decomposition (48 × 5 = 240 values).
//! - 4 bytes (`qh`): 16 remaining ternary values packed as 2-bit codes
//!   (each byte holds 4 values in bits \[1:0\], \[3:2\], \[5:4\], \[7:6\]).
//! - 2 bytes (`d`): FP16 scale factor.
//!
//! Ternary encoding: `{0 → -1, 1 → 0, 2 → +1}` (i.e. `value - 1`).
//!
//! Final weight: `w = d * ternary_value`.
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

/// Decode a single `qs` byte into 5 ternary values (-1, 0, or +1).
///
/// The byte encodes 5 base-3 digits: `v[i] = (q / 3^i) % 3 - 1`.
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
///
/// Each pair of bits encodes one value: `(bits & 3) - 1`.
#[inline]
fn decode_qh_byte(byte: u8) -> [i8; 4] {
    [
        (byte & 0x03) as i8 - 1,
        ((byte >> 2) & 0x03) as i8 - 1,
        ((byte >> 4) & 0x03) as i8 - 1,
        ((byte >> 6) & 0x03) as i8 - 1,
    ]
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

        // Decode qs: 48 bytes → 240 ternary values
        let mut out_idx = 0;
        for &qs_byte in &block[..TQ1_0_QS_BYTES] {
            let vals = decode_qs_byte(qs_byte);
            for &v in &vals {
                output[out_idx] = d * v as f32;
                out_idx += 1;
            }
        }

        // Decode qh: 4 bytes → 16 ternary values
        for &qh_byte in &block[TQ1_0_QH_OFFSET..TQ1_0_QH_OFFSET + TQ1_0_QH_BYTES] {
            let vals = decode_qh_byte(qh_byte);
            for &v in &vals {
                output[out_idx] = d * v as f32;
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

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
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

                // Inline dot product for qs portion (240 values)
                let mut in_off = 0;
                for qs_idx in 0..TQ1_0_QS_BYTES {
                    let vals = decode_qs_byte(data[bo + qs_idx]);
                    for &v in &vals {
                        if input_offset + in_off < n_cols {
                            sum += d * v as f32 * inp[in_off];
                        }
                        in_off += 1;
                    }
                }

                // Inline dot product for qh portion (16 values)
                for qh_idx in 0..TQ1_0_QH_BYTES {
                    let vals = decode_qh_byte(data[bo + TQ1_0_QH_OFFSET + qh_idx]);
                    for &v in &vals {
                        if input_offset + in_off < n_cols {
                            sum += d * v as f32 * inp[in_off];
                        }
                        in_off += 1;
                    }
                }
            }

            *out = sum;
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

    /// Encode 5 ternary values (-1, 0, +1) into a single qs byte (base-3).
    fn encode_qs(vals: [i8; 5]) -> u8 {
        let mut byte: u8 = 0;
        let mut multiplier: u8 = 1;
        for &v in &vals {
            // ternary: -1→0, 0→1, +1→2
            let encoded = (v + 1) as u8;
            byte += encoded * multiplier;
            multiplier *= 3;
        }
        byte
    }

    /// Encode 4 ternary values (-1, 0, +1) into a single qh byte (2-bit codes).
    fn encode_qh(vals: [i8; 4]) -> u8 {
        let mut byte: u8 = 0;
        for (i, &v) in vals.iter().enumerate() {
            let encoded = (v + 1) as u8; // -1→0, 0→1, +1→2
            byte |= encoded << (i * 2);
        }
        byte
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
        // d=2.0, encode known pattern and verify
        let qs_val = encode_qs([-1, 0, 1, -1, 0]); // should give [-2, 0, 2, -2, 0]
        let qs = [qs_val; 48];
        let qh_val = encode_qh([1, 0, -1, 1]); // should give [2, 0, -2, 2]
        let qh = [qh_val; 4];
        let block = make_tq1_0_block(2.0, &qs, &qh);
        let kernel = Tq1_0Ref;
        let mut output = vec![0.0f32; TQ1_0_BLOCK_SIZE];
        kernel
            .dequant_block(&block, &mut output)
            .expect("test: dequant mixed");

        // Check qs portion pattern repeats
        let expected_qs = [-2.0f32, 0.0, 2.0, -2.0, 0.0];
        for chunk in 0..48 {
            for (j, &exp) in expected_qs.iter().enumerate() {
                let idx = chunk * 5 + j;
                assert!(
                    (output[idx] - exp).abs() < 1e-2,
                    "output[{idx}] = {}, expected {exp}",
                    output[idx]
                );
            }
        }
        // Check qh portion pattern repeats
        let expected_qh = [2.0f32, 0.0, -2.0, 2.0];
        for chunk in 0..4 {
            for (j, &exp) in expected_qh.iter().enumerate() {
                let idx = 240 + chunk * 4 + j;
                assert!(
                    (output[idx] - exp).abs() < 1e-2,
                    "output[{idx}] = {}, expected {exp}",
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

    #[test]
    fn test_encode_decode_roundtrip() {
        // Verify our test encode helpers round-trip with decode
        for a in -1i8..=1 {
            for b in -1i8..=1 {
                for c in -1i8..=1 {
                    for d_val in -1i8..=1 {
                        for e in -1i8..=1 {
                            let vals = [a, b, c, d_val, e];
                            let encoded = encode_qs(vals);
                            let decoded = decode_qs_byte(encoded);
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
                        let decoded = decode_qh_byte(encoded);
                        assert_eq!(
                            vals, decoded,
                            "qh roundtrip failed for {vals:?}: encoded={encoded}, decoded={decoded:?}"
                        );
                    }
                }
            }
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
