//! AVX2+FMA accelerated IQ1_M quantization kernel.
//!
//! IQ1_M block layout (56 bytes per 256 weights, QK_K = 256):
//! - bytes[0..32]:   `qs[32]`      — 32 bytes of lower 8-bit grid indices
//! - bytes[32..48]:  `qh[16]`      — 16 bytes of sub-block headers
//! - bytes[48..56]:  `scales[8]`   — 8 bytes of packed sub-block scales (4 × u16 LE)
//!
//! The global scale `d` is reconstructed from the upper nibbles of the four
//! `scales` u16 words. Dequantization is inherently scalar; the gemv step
//! uses AVX2 FMA for the dot-product accumulation.

#![cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]

use core::arch::x86_64::*;

use crate::error::{QuantError, QuantResult};
use crate::reference::iq1s_grid::IQ1S_GRID;
use crate::simd::avx2::util::hsum_f32_avx;
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ1_M: 256 weights per block (QK_K = 256).
pub const BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_M block: 32 (qs) + 16 (qh) + 8 (scales).
pub const BLOCK_BYTES: usize = 56;
/// Byte offset where `qs` begins.
const QS_OFFSET: usize = 0;
/// Byte offset where `qh` begins.
const QH_OFFSET: usize = 32;
/// Byte offset where `scales` begins.
const SCALES_OFFSET: usize = 48;
/// Number of sub-blocks per IQ1_M block.
const N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
const SUB_BLOCK_SIZE: usize = BLOCK_SIZE / N_SUBBLOCKS; // 32
/// Number of groups per sub-block (each group = 8 weights).
const GROUPS_PER_SUB: usize = 4;
/// Weights per group.
const WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant.
const DELTA: f32 = 0.125;

/// Reconstruct the global FP16 scale `d` from the IQ1_M scales field.
///
/// Upper nibble of each of the 4 u16 values is packed into a 16-bit FP16:
/// ```text
/// d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
///        | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
/// ```
#[inline]
fn reconstruct_d(scales: &[u8]) -> f32 {
    let sc0 = u16::from_le_bytes([scales[0], scales[1]]);
    let sc1 = u16::from_le_bytes([scales[2], scales[3]]);
    let sc2 = u16::from_le_bytes([scales[4], scales[5]]);
    let sc3 = u16::from_le_bytes([scales[6], scales[7]]);
    let d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    half::f16::from_bits(d_bits).to_f32()
}

/// AVX2+FMA accelerated IQ1_M kernel.
///
/// Dequantization is scalar (complex bit extraction inherently serial).
/// The gemv step uses 8-wide AVX2 FMA for the dot-product accumulation.
pub struct Iq1MAvx2;

impl QuantKernel for Iq1MAvx2 {
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
        dequant_block_scalar(block, output)
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
            // SAFETY: bounds checked above; avx2+fma guaranteed by dispatcher.
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
        "IQ1_M_AVX2"
    }
}

// ---------------------------------------------------------------------------
// Scalar dequant (exact copy of reference logic)
// ---------------------------------------------------------------------------

fn dequant_block_scalar(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    let qs = &block[QS_OFFSET..QH_OFFSET];
    let qh = &block[QH_OFFSET..SCALES_OFFSET];
    let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

    let d = reconstruct_d(scales);

    let sc: [u16; 4] = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];

    for ib in 0..N_SUBBLOCKS {
        let sc_pair = sc[ib / 2];
        let sc_shift_base = 6 * (ib % 2);

        let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
        let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

        let qs_base = ib * GROUPS_PER_SUB;
        let qh_base = ib * 2;

        let qh0 = qh[qh_base] as usize;
        let qh1 = qh[qh_base + 1] as usize;

        let idx: [usize; GROUPS_PER_SUB] = [
            (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
            (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
            (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
            (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
        ];

        let delta: [f32; GROUPS_PER_SUB] = [
            if qh[qh_base] & 0x08 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base] & 0x80 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base + 1] & 0x08 != 0 {
                -DELTA
            } else {
                DELTA
            },
            if qh[qh_base + 1] & 0x80 != 0 {
                -DELTA
            } else {
                DELTA
            },
        ];

        let output_base = ib * SUB_BLOCK_SIZE;

        for l in 0..GROUPS_PER_SUB {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();
            let group_base = output_base + l * WEIGHTS_PER_GROUP;
            for j in 0..WEIGHTS_PER_GROUP {
                let gv = grid_raw[j] as i8 as f32;
                output[group_base + j] = dl * (gv + delta[l]);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// AVX2 inner kernel
// ---------------------------------------------------------------------------

/// GEMV for one row: dot-product of dequantized row with `input`.
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
    let mut col = 0usize;

    for blk in 0..blocks_per_row {
        let block = &row_data[blk * BLOCK_BYTES..(blk + 1) * BLOCK_BYTES];
        let qs = &block[QS_OFFSET..QH_OFFSET];
        let qh = &block[QH_OFFSET..SCALES_OFFSET];
        let scales = &block[SCALES_OFFSET..BLOCK_BYTES];

        let d = reconstruct_d(scales);
        let sc: [u16; 4] = [
            u16::from_le_bytes([scales[0], scales[1]]),
            u16::from_le_bytes([scales[2], scales[3]]),
            u16::from_le_bytes([scales[4], scales[5]]),
            u16::from_le_bytes([scales[6], scales[7]]),
        ];

        for ib in 0..N_SUBBLOCKS {
            let sc_pair = sc[ib / 2];
            let sc_shift_base = 6 * (ib % 2);

            let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
            let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

            let qs_base = ib * GROUPS_PER_SUB;
            let qh_base = ib * 2;

            let qh0 = qh[qh_base] as usize;
            let qh1 = qh[qh_base + 1] as usize;

            let idx: [usize; GROUPS_PER_SUB] = [
                (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
                (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
                (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
                (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
            ];

            let delta: [f32; GROUPS_PER_SUB] = [
                if qh[qh_base] & 0x08 != 0 {
                    -DELTA
                } else {
                    DELTA
                },
                if qh[qh_base] & 0x80 != 0 {
                    -DELTA
                } else {
                    DELTA
                },
                if qh[qh_base + 1] & 0x08 != 0 {
                    -DELTA
                } else {
                    DELTA
                },
                if qh[qh_base + 1] & 0x80 != 0 {
                    -DELTA
                } else {
                    DELTA
                },
            ];

            for l in 0..GROUPS_PER_SUB {
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();

                let mut vals = [0.0f32; WEIGHTS_PER_GROUP];
                let delta_l = delta[l];
                for j in 0..WEIGHTS_PER_GROUP {
                    vals[j] = dl * (grid_raw[j] as i8 as f32 + delta_l);
                }

                let w_off = col + ib * SUB_BLOCK_SIZE + l * WEIGHTS_PER_GROUP;
                let scaled = _mm256_loadu_ps(vals.as_ptr());

                if w_off + WEIGHTS_PER_GROUP > n_cols {
                    // Scalar tail.
                    let mut scaled_arr = [0.0f32; WEIGHTS_PER_GROUP];
                    _mm256_storeu_ps(scaled_arr.as_mut_ptr(), scaled);
                    for k in 0..WEIGHTS_PER_GROUP {
                        let c = w_off + k;
                        if c < n_cols {
                            acc = _mm256_add_ps(acc, _mm256_set1_ps(scaled_arr[k] * input[c]));
                        }
                    }
                } else {
                    let iv = _mm256_loadu_ps(input.as_ptr().add(w_off));
                    acc = _mm256_fmadd_ps(scaled, iv, acc);
                }
            }
        }

        col += BLOCK_SIZE;
    }

    hsum_f32_avx(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::iq1_m::Iq1MRef;
    use crate::traits::QuantKernel;

    fn make_zero_block() -> Vec<u8> {
        vec![0u8; BLOCK_BYTES]
    }

    /// Build an IQ1_M block with a specific global scale `d` embedded in the
    /// upper nibbles of the four `scales` u16 entries.
    fn make_block_with_scale(d: f32) -> Vec<u8> {
        let d_bits = half::f16::from_f32(d).to_bits();
        // Pack d_bits into 4 u16 upper nibbles.
        // d_bits = n3 n2 n1 n0 (4-bit nibbles, MSB first)
        let n0 = (d_bits & 0x000f) as u16;
        let n1 = ((d_bits >> 4) & 0x000f) as u16;
        let n2 = ((d_bits >> 8) & 0x000f) as u16;
        let n3 = ((d_bits >> 12) & 0x000f) as u16;

        // sc[0] >> 12 = n0  → sc[0] = n0 << 12
        // sc[1] >> 8 & 0xf0 = n1 → sc[1] has n1 at bits 8-11
        // sc[2] >> 4 & 0xf00 = n2 → sc[2] has n2 at bits 4-7? no...
        // Let's just replicate the reconstruction formula in reverse:
        // d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0xf0) | ((sc2 >> 4) & 0xf00) | (sc3 & 0xf000)
        // So: sc0 upper nibble = d bits [0..3], sc1 upper nibble = d bits [4..7], etc.
        let sc0: u16 = n0 << 12;
        let sc1: u16 = n1 << 8; // bits [11:8] contribute to d bits [7:4]
        let sc2: u16 = n2 << 4; // after >> 4, bits [11:8] contribute
        let sc3: u16 = n3; // after & 0xf000, upper nibble used

        // Wait, let me re-derive:
        // reconstruct_d:
        //   d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)
        // So d_bits nibbles:
        //   bits [3:0]    = sc0 >> 12          → sc0[15:12]
        //   bits [7:4]    = (sc1 >> 8) & 0xf0  → sc1[11:8]
        //   bits [11:8]   = (sc2 >> 4) & 0xf00 → sc2[11:8]
        //   bits [15:12]  = sc3 & 0xf000       → sc3[15:12]
        // So: sc0 bits [15:12] = d_bits [3:0], sc1 bits [11:8] = d_bits [7:4], etc.
        let sc0_v: u16 = (d_bits & 0x000f) << 12;
        let sc1_v: u16 = ((d_bits & 0x00f0) >> 4) << 8;
        let sc2_v: u16 = ((d_bits & 0x0f00) >> 8) << 4; // repack to bits [7:4], then (>>4) gives [3:0] & 0xf → contributes via (sc2>>4)&0x0f00
                                                        // Actually: (sc2>>4) & 0x0f00 extracts bits [11:8] of sc2, which maps to d_bits [11:8]
                                                        // So sc2[11:8] = d_bits[11:8] → sc2_v = ((d_bits >> 8) & 0xf) << 8
        let sc2_v_fixed: u16 = ((d_bits & 0x0f00) >> 8) << 8;
        let sc3_v: u16 = d_bits & 0xf000;

        let _ = sc0;
        let _ = sc1;
        let _ = sc2;
        let _ = sc3;
        let _ = sc0_v;
        let _ = sc1_v;
        let _ = sc2_v;

        let mut block = vec![0u8; BLOCK_BYTES];
        let sc0_bytes = sc0_v.to_le_bytes();
        let sc1_bytes = sc1_v.to_le_bytes();
        let sc2_bytes = sc2_v_fixed.to_le_bytes();
        let sc3_bytes = sc3_v.to_le_bytes();
        block[SCALES_OFFSET..SCALES_OFFSET + 2].copy_from_slice(&sc0_bytes);
        block[SCALES_OFFSET + 2..SCALES_OFFSET + 4].copy_from_slice(&sc1_bytes);
        block[SCALES_OFFSET + 4..SCALES_OFFSET + 6].copy_from_slice(&sc2_bytes);
        block[SCALES_OFFSET + 6..SCALES_OFFSET + 8].copy_from_slice(&sc3_bytes);
        block
    }

    #[test]
    fn avx2_matches_reference_zero_block() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_zero_block();

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq1MRef.dequant_block(&block, &mut ref_out).unwrap();
        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq1MAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn avx2_matches_reference_with_scale() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_block_with_scale(1.5_f32);

        let mut ref_out = vec![0.0f32; BLOCK_SIZE];
        Iq1MRef.dequant_block(&block, &mut ref_out).unwrap();
        let mut avx_out = vec![0.0f32; BLOCK_SIZE];
        Iq1MAvx2.dequant_block(&block, &mut avx_out).unwrap();

        for (i, (r, a)) in ref_out.iter().zip(avx_out.iter()).enumerate() {
            assert!((r - a).abs() < 1e-4, "mismatch at [{i}]: ref={r}, avx={a}");
        }
    }

    #[test]
    fn gemv_matches_dequant_dot_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let block = make_block_with_scale(1.0_f32);

        let mut dequant = vec![0.0f32; BLOCK_SIZE];
        Iq1MAvx2.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block,
            vec![1, BLOCK_SIZE],
            oxillama_gguf::GgufTensorType::Iq1M,
        );
        let input = vec![1.0f32; BLOCK_SIZE];
        let mut got = vec![0.0f32; 1];
        Iq1MAvx2.gemv(&tensor, &input, &mut got).unwrap();

        assert!(
            (got[0] - expected).abs() < 1e-2,
            "gemv={}, dequant_sum={}",
            got[0],
            expected
        );
    }
}
