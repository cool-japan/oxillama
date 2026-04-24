//! Q4_K reference (naive) implementation.
//!
//! Q4_K block format (144 bytes per 256 weights):
//! - 2 bytes: FP16 super-block scale (d)
//! - 2 bytes: FP16 super-block minimum (dmin)
//! - 12 bytes: 8 sub-block scales + 8 sub-block mins, 6-bit each, packed
//! - 128 bytes: 256 × 4-bit unsigned nibbles packed (2 per byte)
//!
//! 8 sub-blocks of 32 weights each.
//! Weight formula: `w = d * scale_i * q - dmin * min_i` where q is 4-bit (0..15).
//!
//! Effective: 4.5 bits/weight.

use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

/// Reference (naive scalar) Q4_K kernel.
pub struct Q4KRef;

/// Decode the 6-bit packed scales and mins for Q4_K.
///
/// Returns (scales[8], mins[8]) where each is a 6-bit value.
fn decode_scales_mins(scales_raw: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // Sub-blocks 0..3: straightforward 6-bit extraction
    for j in 0..4 {
        sc[j] = scales_raw[j] & 0x3F;
        mn[j] = scales_raw[j + 4] & 0x3F;
    }

    // Sub-blocks 4..7: assembled from high bits of bytes 0..3/4..7 and bytes 8..11
    for j in 4..8 {
        let lo_sc = scales_raw[j + 4] & 0x0F;
        let hi_sc = (scales_raw[j - 4] >> 6) & 0x03;
        sc[j] = lo_sc | (hi_sc << 4);

        let lo_mn = (scales_raw[j + 4] >> 4) & 0x0F;
        let hi_mn = (scales_raw[j] >> 6) & 0x03;
        mn[j] = lo_mn | (hi_mn << 4);
    }

    (sc, mn)
}

impl QuantKernel for Q4KRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        if block.len() < Q4_K_BLOCK_BYTES {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_K_BLOCK_BYTES,
                available: block.len(),
            });
        }
        if output.len() < Q4_K_BLOCK_SIZE {
            return Err(QuantError::BufferTooSmall {
                needed: Q4_K_BLOCK_SIZE,
                available: output.len(),
            });
        }

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales_raw = &block[4..16];
        let qs = &block[16..144]; // 128 bytes of nibble data

        let (sc, mn) = decode_scales_mins(scales_raw);

        // Process 4 groups of 64 weights (2 sub-blocks of 32 per group)
        let mut is = 0usize; // sub-block index
        let mut qs_offset = 0usize;
        let mut out_offset = 0usize;

        for _group in 0..4 {
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            // Low nibbles → first 32 weights (sub-block `is`)
            for l in 0..32 {
                let q = (qs[qs_offset + l] & 0x0F) as f32;
                output[out_offset + l] = d1 * q - m1;
            }

            // High nibbles → next 32 weights (sub-block `is+1`)
            for l in 0..32 {
                let q = ((qs[qs_offset + l] >> 4) & 0x0F) as f32;
                output[out_offset + 32 + l] = d2 * q - m2;
            }

            is += 2;
            qs_offset += 32;
            out_offset += 64;
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

        let blocks_per_row = n_cols.div_ceil(Q4_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q4_K_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + Q4_K_BLOCK_BYTES];

                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales_raw = &block[4..16];
                let qs = &block[16..144];
                let input_offset = blk * Q4_K_BLOCK_SIZE;

                let (sc, mn) = decode_scales_mins(scales_raw);

                let mut is = 0usize;
                let mut qs_off = 0usize;
                let mut w_off = input_offset;

                for _group in 0..4 {
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * mn[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * mn[is + 1] as f32;

                    for l in 0..32 {
                        let idx = w_off + l;
                        if idx < n_cols {
                            let q = (qs[qs_off + l] & 0x0F) as f32;
                            sum += (d1 * q - m1) * input[idx];
                        }
                    }
                    for l in 0..32 {
                        let idx = w_off + 32 + l;
                        if idx < n_cols {
                            let q = ((qs[qs_off + l] >> 4) & 0x0F) as f32;
                            sum += (d2 * q - m2) * input[idx];
                        }
                    }

                    is += 2;
                    qs_off += 32;
                    w_off += 64;
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

    /// Override of `matvec_q8_fused` required because the trait default is wrong for Q4_K.
    ///
    /// # Why the default is broken for Q4_K
    /// The trait default assumes `blocks_per_row = n_cols / block_size` maps 1-to-1 with
    /// Q8_0 blocks.  For Q4_K (block_size=256), each Q4_K weight block spans 256 weights =
    /// 8 Q8_0 activation blocks (each 32 weights).  The default would only allocate
    /// `blocks_per_row * 34` bytes for activations (8× too small) and would panic with
    /// `a_scratch = vec![0f32; 32]` for `valid=256`.
    ///
    /// # Block mapping
    /// - 1 Q4_K weight block (144 bytes, 256 weights) ↔ 8 Q8_0 activation blocks (34 bytes each).
    /// - Sub-block `s` (0..8) of weight block `blk` uses Q8_0 activation at index `blk*8 + s`.
    ///
    /// # Formula per sub-block `s`
    /// `contrib_s = (d·sc[s]·d_a)·Σ(q_w·q_a) − (dmin·mn[s]·d_a)·Σ(q_a)`
    fn matvec_q8_fused(
        &self,
        weights: &[u8],
        acts_q8: &[u8],
        out: &mut [f32],
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<()> {
        if out.len() < n_rows {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows,
                got: out.len(),
            });
        }

        let blocks_per_row = n_cols.div_ceil(Q4_K_BLOCK_SIZE);
        let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
        // Each Q4_K block maps to 8 Q8_0 blocks.
        let q8_blocks_per_row = blocks_per_row * 8;
        let acts_needed = q8_blocks_per_row * Q8_0_BLOCK_BYTES;

        if weights.len() < n_rows * row_bytes {
            return Err(QuantError::BufferTooSmall {
                needed: n_rows * row_bytes,
                available: weights.len(),
            });
        }
        if acts_q8.len() < acts_needed {
            return Err(QuantError::BufferTooSmall {
                needed: acts_needed,
                available: acts_q8.len(),
            });
        }

        for (row, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * Q4_K_BLOCK_BYTES;
                let block = &weights[block_offset..block_offset + Q4_K_BLOCK_BYTES];

                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let (sc, mn) = decode_scales_mins(&block[4..16]);
                let qs = &block[16..144]; // 128 nibble bytes

                let input_offset = blk * Q4_K_BLOCK_SIZE;

                let mut is = 0usize;
                let mut qs_off = 0usize;
                let mut w_off = input_offset;

                // 4 groups of 2 sub-blocks each.
                for _group in 0..4 {
                    // Sub-block `is` (lo nibbles) — Q8_0 activation block index `blk*8 + is`.
                    let a_idx_lo = blk * 8 + is;
                    let a_start_lo = a_idx_lo * Q8_0_BLOCK_BYTES;
                    let a_block_lo = &acts_q8[a_start_lo..a_start_lo + Q8_0_BLOCK_BYTES];
                    let d_a_lo = f16_to_f32(u16::from_le_bytes([a_block_lo[0], a_block_lo[1]]));
                    let q8_lo = &a_block_lo[2..]; // 32 i8 values

                    let da_lo = d * sc[is] as f32;
                    let m_lo = dmin * mn[is] as f32;

                    // Sub-block `is+1` (hi nibbles) — Q8_0 activation block index `blk*8 + is + 1`.
                    let a_idx_hi = blk * 8 + is + 1;
                    let a_start_hi = a_idx_hi * Q8_0_BLOCK_BYTES;
                    let a_block_hi = &acts_q8[a_start_hi..a_start_hi + Q8_0_BLOCK_BYTES];
                    let d_a_hi = f16_to_f32(u16::from_le_bytes([a_block_hi[0], a_block_hi[1]]));
                    let q8_hi = &a_block_hi[2..]; // 32 i8 values

                    let da_hi = d * sc[is + 1] as f32;
                    let m_hi = dmin * mn[is + 1] as f32;

                    // Lo nibbles → first 32 weights of this group (sub-block `is`).
                    // Formula: Σ (da_lo * q_w - m_lo) * q_a_lo
                    //        = da_lo * Σ(q_w * q_a_lo) - m_lo * Σ(q_a_lo)
                    let mut dot_lo = 0.0f32;
                    let mut sum_a_lo = 0.0f32;
                    for l in 0..32 {
                        let idx = w_off + l;
                        if idx < n_cols {
                            let q_w = (qs[qs_off + l] & 0x0F) as f32;
                            let q_a = q8_lo[l] as i8 as f32;
                            dot_lo += q_w * q_a;
                            sum_a_lo += q_a;
                        }
                    }
                    sum += (da_lo * dot_lo - m_lo * sum_a_lo) * d_a_lo;

                    // Hi nibbles → next 32 weights (sub-block `is+1`).
                    let mut dot_hi = 0.0f32;
                    let mut sum_a_hi = 0.0f32;
                    for l in 0..32 {
                        let idx = w_off + 32 + l;
                        if idx < n_cols {
                            let q_w = ((qs[qs_off + l] >> 4) & 0x0F) as f32;
                            let q_a = q8_hi[l] as i8 as f32;
                            dot_hi += q_w * q_a;
                            sum_a_hi += q_a;
                        }
                    }
                    sum += (da_hi * dot_hi - m_hi * sum_a_hi) * d_a_hi;

                    is += 2;
                    qs_off += 32;
                    w_off += 64;
                }
            }

            *out_val += sum; // ACCUMULATE
        }

        Ok(())
    }

    fn block_size(&self) -> usize {
        Q4_K_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        Q4_K_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "Q4_K"
    }
}

/// Q8_0 block constants for the fused GEMV override.
const Q8_0_BLOCK_BYTES: usize = 34;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q4_K_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_dequant_zero_scale() {
        // d=0, dmin=0 → all weights should be 0
        let block = make_q4_k_block(0.0, 0.0, &[0; 12], &[0; 128]);
        let kernel = Q4KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();
        for &v in &output {
            assert!((v).abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_dequant_uniform() {
        // d=1.0, dmin=0.0, all scales=1 (sub-blocks 0..3), all nibbles=8
        // Weight = 1.0 * 1 * 8 - 0 = 8.0
        let mut scales = [0u8; 12];
        // Set sub-block scales 0..3 to 1 (lower 6 bits of bytes 0..3)
        scales[0] = 1;
        scales[1] = 1;
        scales[2] = 1;
        scales[3] = 1;
        // Sub-block scales 4..7: stored in bytes 8..11 lower 4 bits, with high bits from bytes 0..3 upper 2 bits
        scales[8] = 1;
        scales[9] = 1;
        scales[10] = 1;
        scales[11] = 1;

        // All nibbles = 8: byte = 0x88 (lo=8, hi=8)
        let qs = [0x88u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let kernel = Q4KRef;
        let mut output = vec![0.0f32; 256];
        kernel.dequant_block(&block, &mut output).unwrap();

        // All weights should be 1.0 * 1 * 8 = 8.0
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 8.0).abs() < 0.01, "weight[{i}] = {v}, expected 8.0");
        }
    }

    #[test]
    fn test_gemv_q4_k() {
        // Create a simple 1-row, 256-col Q4_K tensor
        // d=1.0, dmin=0, all scales=1, all nibbles=1
        // Weight = 1.0 * 1 * 1 - 0 = 1.0
        let mut scales = [0u8; 12];
        scales[..4].fill(1); // sub-blocks 0..3 scale=1
        scales[8..12].fill(1); // sub-blocks 4..7 scale=1

        // All nibbles = 1: lo=1, hi=1 → byte = 0x11
        let qs = [0x11u8; 128];

        let block = make_q4_k_block(1.0, 0.0, &scales, &qs);
        let tensor = QuantTensor::new(block, vec![1, 256], oxillama_gguf::GgufTensorType::Q4K);

        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        let kernel = Q4KRef;
        kernel.gemv(&tensor, &input, &mut output).unwrap();

        // All 256 weights = 1.0, all inputs = 1.0 → dot = 256.0
        assert!(
            (output[0] - 256.0).abs() < 1.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }

    // ── matvec_q8_fused (Q4_K override) ──────────────────────────────────

    fn make_q8_0_block(scale: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        let d_bits = half::f16::from_f32(scale).to_bits();
        block.extend_from_slice(&d_bits.to_le_bytes());
        for &q in qs {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn test_q4k_fused_zero_activations() {
        // Zero activations → output must stay zero.
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x88u8; 128]; // non-zero weights (q=8)
        let w_block = make_q4_k_block(1.0, 0.0, &scales, &qs);

        // 8 Q8_0 blocks all zero.
        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(1.0, &[0i8; 32]));
        }

        let mut out = vec![0.0f32; 1];
        let kernel = Q4KRef;
        kernel
            .matvec_q8_fused(&w_block, &acts, &mut out, 1, 256)
            .expect("q4k fused zero acts");
        assert!(out[0].abs() < 1e-5, "expected 0, got {}", out[0]);
    }

    #[test]
    fn test_q4k_fused_accumulates() {
        // Verify ACCUMULATE semantics.
        let w_block = make_q4_k_block(0.0, 0.0, &[0u8; 12], &[0u8; 128]);

        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(1.0, &[0i8; 32]));
        }

        let mut out = vec![42.0f32; 1];
        let kernel = Q4KRef;
        kernel
            .matvec_q8_fused(&w_block, &acts, &mut out, 1, 256)
            .expect("q4k fused accumulate");
        assert!(
            (out[0] - 42.0).abs() < 1e-5,
            "accumulation broken: got {}",
            out[0]
        );
    }

    #[test]
    fn test_q4k_fused_matches_unfused() {
        // Q4_K fused GEMV must match dequant + f32 GEMV (unfused) within tol 1e-3.
        let n_cols = 256usize;
        let mut scales = [0u8; 12];
        scales[0] = 5;
        scales[1] = 3;
        scales[2] = 7;
        scales[3] = 2;
        scales[4] = 4;
        scales[5] = 6;
        scales[6] = 1;
        scales[7] = 3;
        scales[8] = 9;
        scales[9] = 11;
        scales[10] = 13;
        scales[11] = 15;
        let qs_nibbles = [0xA5u8; 128]; // lo=5, hi=10
        let d_w = 0.5f32;
        let dmin_w = 0.1f32;
        let w_block = make_q4_k_block(d_w, dmin_w, &scales, &qs_nibbles);

        let d_a = 0.25f32;
        let q8_vals: [i8; 32] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16,
        ];

        // Build 8 Q8_0 blocks (one per Q4_K sub-block).
        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        // Build f32 input for unfused path (256 values = 8 × 32 i8 activations).
        let input: Vec<f32> = (0..8)
            .flat_map(|_| q8_vals.iter().map(|&q| q as f32 * d_a))
            .collect();

        let tensor = QuantTensor::new(
            w_block.clone(),
            vec![1, n_cols],
            oxillama_gguf::GgufTensorType::Q4K,
        );
        let kernel = Q4KRef;

        let mut out_unfused = vec![0.0f32; 1];
        kernel
            .gemv(&tensor, &input, &mut out_unfused)
            .expect("q4k unfused gemv");

        let mut out_fused = vec![0.0f32; 1];
        kernel
            .matvec_q8_fused(&w_block, &acts, &mut out_fused, 1, n_cols)
            .expect("q4k fused");

        let err = (out_fused[0] - out_unfused[0]).abs();
        assert!(
            err < 1e-3,
            "q4k fused vs unfused: fused={} unfused={} err={}",
            out_fused[0],
            out_unfused[0],
            err
        );
    }

    #[test]
    fn test_q4k_fused_multi_row() {
        // 4 rows × 256 cols.
        let n_rows = 4usize;
        let n_cols = 256usize;
        let mut scales = [0u8; 12];
        scales[..4].fill(1);
        scales[8..12].fill(1);
        let qs = [0x55u8; 128]; // lo=5, hi=5

        let d_a = 0.5f32;
        let q8_vals: [i8; 32] = [
            2, -1, 4, -3, 6, -5, 8, -7, 1, -2, 3, -4, 5, -6, 7, -8, -2, 1, -4, 3, -6, 5, -8, 7, -1,
            2, -3, 4, -5, 6, -7, 8,
        ];

        let scales_w = [0.5f32, 1.0f32, 0.25f32, 0.1f32];
        let mut weights: Vec<u8> = Vec::new();
        for &s in &scales_w {
            weights.extend_from_slice(&make_q4_k_block(s, 0.0, &scales, &qs));
        }

        // 8 Q8_0 blocks for 1 Q4_K block.
        let mut acts: Vec<u8> = Vec::new();
        for _ in 0..8 {
            acts.extend_from_slice(&make_q8_0_block(d_a, &q8_vals));
        }

        // Build f32 input (256 values).
        let input: Vec<f32> = (0..8)
            .flat_map(|_| q8_vals.iter().map(|&q| q as f32 * d_a))
            .collect();

        let kernel = Q4KRef;

        let mut out_unfused = vec![0.0f32; n_rows];
        for row in 0..n_rows {
            let row_start = row * Q4_K_BLOCK_BYTES;
            let row_data = weights[row_start..row_start + Q4_K_BLOCK_BYTES].to_vec();
            let tensor = QuantTensor::new(
                row_data,
                vec![1, n_cols],
                oxillama_gguf::GgufTensorType::Q4K,
            );
            kernel
                .gemv(&tensor, &input, &mut out_unfused[row..row + 1])
                .expect("q4k unfused row");
        }

        let mut out_fused = vec![0.0f32; n_rows];
        kernel
            .matvec_q8_fused(&weights, &acts, &mut out_fused, n_rows, n_cols)
            .expect("q4k fused multi-row");

        for i in 0..n_rows {
            let err = (out_fused[i] - out_unfused[i]).abs();
            assert!(
                err < 1e-3,
                "row {i}: fused={} unfused={} err={}",
                out_fused[i],
                out_unfused[i],
                err
            );
        }
    }
}
