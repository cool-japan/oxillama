//! IQ1_M reference (naive) implementation.
//!
//! IQ1_M block format (56 bytes per 256 weights, QK_K = 256):
//! - bytes[0..32]:   `qs[QK_K/8]`   — 32 bytes of quantized indices (lower 8 bits)
//! - bytes[32..48]:  `qh[QK_K/16]`  — 16 bytes of sub-block headers
//! - bytes[48..56]:  `scales[QK_K/32]` — 8 bytes of packed sub-block scales
//!
//! There is **no explicit global `d` field**. The global scale is reconstructed
//! from the four upper-nibble bits of the four `scales` u16 words:
//! ```text
//! let sc: [u16; 4] = scales[0..8] as 4×u16 LE;
//! let d_bits: u16 = (sc[0] >> 12)
//!                 | ((sc[1] >> 8)  & 0x00f0)
//!                 | ((sc[2] >> 4)  & 0x0f00)
//!                 | (sc[3]         & 0xf000);
//! let d = f16::from_bits(d_bits).to_f32();
//! ```
//!
//! ## Sub-block layout (8 sub-blocks of 32 weights each)
//!
//! For sub-block `ib` (0..8), sc_pair = `sc[ib/2]`:
//! - `dl1 = d * (2 * ((sc_pair >> (6*(ib%2)+0)) & 0x7) + 1)`  — first-half scale
//! - `dl2 = d * (2 * ((sc_pair >> (6*(ib%2)+3)) & 0x7) + 1)`  — second-half scale
//!
//! Groups 0..2 use dl1; groups 2..4 use dl2.
//!
//! Each sub-block consumes 4 bytes from `qs` and 2 bytes from `qh`:
//! - `idx[0] = qs[4*ib+0] | ((qh[2*ib+0] << 8) & 0x700)`
//! - `idx[1] = qs[4*ib+1] | ((qh[2*ib+0] << 4) & 0x700)`
//! - `idx[2] = qs[4*ib+2] | ((qh[2*ib+1] << 8) & 0x700)`
//! - `idx[3] = qs[4*ib+3] | ((qh[2*ib+1] << 4) & 0x700)`
//! - `delta[0] = if qh[2*ib+0] & 0x08 != 0 { -0.125 } else { 0.125 }`
//! - `delta[1] = if qh[2*ib+0] & 0x80 != 0 { -0.125 } else { 0.125 }`
//! - `delta[2] = if qh[2*ib+1] & 0x08 != 0 { -0.125 } else { 0.125 }`
//! - `delta[3] = if qh[2*ib+1] & 0x80 != 0 { -0.125 } else { 0.125 }`
//!
//! Grid values are SIGNED i8 bytes: `y[j] = dl * (grid_i8[j] as f32 + delta[l])`
//!
//! ## Block size verification
//!
//! QK_K/8 (qs) + QK_K/16 (qh) + QK_K/32 (scales) = 32 + 16 + 8 = 56 ✓

use super::iq1s_grid::IQ1S_GRID;
use crate::error::{QuantError, QuantResult};
use crate::traits::QuantKernel;
use crate::types::QuantTensor;

/// Block size for IQ1_M: 256 weights per block (QK_K = 256).
const IQ1M_BLOCK_SIZE: usize = 256;
/// Bytes per IQ1_M block: 32 (qs) + 16 (qh) + 8 (scales).
const IQ1M_BLOCK_BYTES: usize = 56;
/// Byte offset where `qs` begins.
const IQ1M_QS_OFFSET: usize = 0;
/// Byte offset where `qh` begins.
const IQ1M_QH_OFFSET: usize = 32;
/// Byte offset where `scales` begins.
const IQ1M_SCALES_OFFSET: usize = 48;
/// Number of sub-blocks per IQ1_M block (QK_K/32 = 8).
const IQ1M_N_SUBBLOCKS: usize = 8;
/// Weights per sub-block.
const IQ1M_SUB_BLOCK_SIZE: usize = IQ1M_BLOCK_SIZE / IQ1M_N_SUBBLOCKS; // 32
/// Number of groups per sub-block (each group = 8 weights).
const IQ1M_GROUPS_PER_SUB: usize = 4;
/// Weights per group.
const IQ1M_WEIGHTS_PER_GROUP: usize = 8;
/// Delta constant (IQ1M_DELTA = 0.125).
const IQ1M_DELTA: f32 = 0.125;

/// Reconstruct the global FP16 scale `d` from the IQ1_M scales field.
///
/// The scales field (8 bytes) is interpreted as 4 × u16 little-endian.
/// The upper nibble of each u16 is packed into a 16-bit FP16 value:
/// ```text
/// d_bits = (sc[0] >> 12)
///        | ((sc[1] >> 8)  & 0x00f0)
///        | ((sc[2] >> 4)  & 0x0f00)
///        | (sc[3]         & 0xf000)
/// ```
fn reconstruct_d(scales: &[u8]) -> f32 {
    let sc0 = u16::from_le_bytes([scales[0], scales[1]]);
    let sc1 = u16::from_le_bytes([scales[2], scales[3]]);
    let sc2 = u16::from_le_bytes([scales[4], scales[5]]);
    let sc3 = u16::from_le_bytes([scales[6], scales[7]]);

    let d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    half::f16::from_bits(d_bits).to_f32()
}

/// Dequantize a single IQ1_M block into `output`.
fn dequant_iq1m_block(block: &[u8], output: &mut [f32]) -> QuantResult<()> {
    if block.len() < IQ1M_BLOCK_BYTES {
        return Err(QuantError::BufferTooSmall {
            needed: IQ1M_BLOCK_BYTES,
            available: block.len(),
        });
    }
    if output.len() < IQ1M_BLOCK_SIZE {
        return Err(QuantError::BufferTooSmall {
            needed: IQ1M_BLOCK_SIZE,
            available: output.len(),
        });
    }

    let qs = &block[IQ1M_QS_OFFSET..IQ1M_QH_OFFSET];
    let qh = &block[IQ1M_QH_OFFSET..IQ1M_SCALES_OFFSET];
    let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];

    let d = reconstruct_d(scales);

    // The scales field is also read as 4 × u16 for per-sub-block scales.
    let sc: [u16; 4] = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];

    for ib in 0..IQ1M_N_SUBBLOCKS {
        let sc_pair = sc[ib / 2];
        let sc_shift_base = 6 * (ib % 2);

        // dl1 for groups 0..2, dl2 for groups 2..4.
        let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
        let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

        let qs_base = ib * IQ1M_GROUPS_PER_SUB;
        let qh_base = ib * 2; // 2 qh bytes per sub-block

        // Assemble 4 grid indices from qs + qh.
        let qh0 = qh[qh_base] as usize;
        let qh1 = qh[qh_base + 1] as usize;

        let idx: [usize; 4] = [
            (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
            (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
            (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
            (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
        ];

        // Delta signs from qh bits 3 and 7.
        let delta: [f32; 4] = [
            if qh[qh_base] & 0x08 != 0 {
                -IQ1M_DELTA
            } else {
                IQ1M_DELTA
            },
            if qh[qh_base] & 0x80 != 0 {
                -IQ1M_DELTA
            } else {
                IQ1M_DELTA
            },
            if qh[qh_base + 1] & 0x08 != 0 {
                -IQ1M_DELTA
            } else {
                IQ1M_DELTA
            },
            if qh[qh_base + 1] & 0x80 != 0 {
                -IQ1M_DELTA
            } else {
                IQ1M_DELTA
            },
        ];

        let output_base = ib * IQ1M_SUB_BLOCK_SIZE;

        // Groups 0 and 1 use dl1, groups 2 and 3 use dl2.
        for l in 0..IQ1M_GROUPS_PER_SUB {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();
            let group_base = output_base + l * IQ1M_WEIGHTS_PER_GROUP;
            for j in 0..IQ1M_WEIGHTS_PER_GROUP {
                let gv = grid_raw[j] as i8 as f32;
                output[group_base + j] = dl * (gv + delta[l]);
            }
        }
    }

    Ok(())
}

/// Reference (naive scalar) IQ1_M kernel.
pub struct Iq1MRef;

impl QuantKernel for Iq1MRef {
    fn dequant_block(&self, block: &[u8], output: &mut [f32]) -> QuantResult<()> {
        dequant_iq1m_block(block, output)
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

        let blocks_per_row = n_cols.div_ceil(IQ1M_BLOCK_SIZE);
        let row_bytes = blocks_per_row * IQ1M_BLOCK_BYTES;

        for (row, out) in output.iter_mut().enumerate().take(n_rows) {
            let row_start = row * row_bytes;
            let mut sum = 0.0_f32;

            for blk in 0..blocks_per_row {
                let block_offset = row_start + blk * IQ1M_BLOCK_BYTES;
                let block = &quant_matrix.data[block_offset..block_offset + IQ1M_BLOCK_BYTES];

                let qs = &block[IQ1M_QS_OFFSET..IQ1M_QH_OFFSET];
                let qh = &block[IQ1M_QH_OFFSET..IQ1M_SCALES_OFFSET];
                let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];

                let d = reconstruct_d(scales);
                let sc: [u16; 4] = [
                    u16::from_le_bytes([scales[0], scales[1]]),
                    u16::from_le_bytes([scales[2], scales[3]]),
                    u16::from_le_bytes([scales[4], scales[5]]),
                    u16::from_le_bytes([scales[6], scales[7]]),
                ];

                for ib in 0..IQ1M_N_SUBBLOCKS {
                    let sc_pair = sc[ib / 2];
                    let sc_shift_base = 6 * (ib % 2);
                    let dl1 = d * (2.0 * (((sc_pair >> sc_shift_base) & 0x7) as f32) + 1.0);
                    let dl2 = d * (2.0 * (((sc_pair >> (sc_shift_base + 3)) & 0x7) as f32) + 1.0);

                    let qs_base = ib * IQ1M_GROUPS_PER_SUB;
                    let qh_base = ib * 2;
                    let qh0 = qh[qh_base] as usize;
                    let qh1 = qh[qh_base + 1] as usize;

                    let idx: [usize; 4] = [
                        (qs[qs_base] as usize) | ((qh0 << 8) & 0x700),
                        (qs[qs_base + 1] as usize) | ((qh0 << 4) & 0x700),
                        (qs[qs_base + 2] as usize) | ((qh1 << 8) & 0x700),
                        (qs[qs_base + 3] as usize) | ((qh1 << 4) & 0x700),
                    ];
                    let delta: [f32; 4] = [
                        if qh[qh_base] & 0x08 != 0 {
                            -IQ1M_DELTA
                        } else {
                            IQ1M_DELTA
                        },
                        if qh[qh_base] & 0x80 != 0 {
                            -IQ1M_DELTA
                        } else {
                            IQ1M_DELTA
                        },
                        if qh[qh_base + 1] & 0x08 != 0 {
                            -IQ1M_DELTA
                        } else {
                            IQ1M_DELTA
                        },
                        if qh[qh_base + 1] & 0x80 != 0 {
                            -IQ1M_DELTA
                        } else {
                            IQ1M_DELTA
                        },
                    ];

                    let col_base = blk * IQ1M_BLOCK_SIZE + ib * IQ1M_SUB_BLOCK_SIZE;

                    for l in 0..IQ1M_GROUPS_PER_SUB {
                        let dl = if l < 2 { dl1 } else { dl2 };
                        let grid_raw = IQ1S_GRID[idx[l]].to_le_bytes();
                        let col = col_base + l * IQ1M_WEIGHTS_PER_GROUP;
                        for (j, &raw_byte) in grid_raw.iter().enumerate() {
                            let global_col = col + j;
                            if global_col >= n_cols {
                                break;
                            }
                            let gv = raw_byte as i8 as f32;
                            sum += dl * (gv + delta[l]) * input[global_col];
                        }
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
        IQ1M_BLOCK_SIZE
    }

    fn block_bytes(&self) -> usize {
        IQ1M_BLOCK_BYTES
    }

    fn name(&self) -> &'static str {
        "IQ1_M"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::QuantKernel;

    /// Build a minimal IQ1_M block with all zeroes (scales=0 → d=0).
    fn make_zero_iq1m_block() -> [u8; IQ1M_BLOCK_BYTES] {
        [0u8; IQ1M_BLOCK_BYTES]
    }

    /// Build an IQ1_M block where the global FP16 `d` is encoded in scales.
    ///
    /// We set sc[0] bits [12..15] to the upper 4 bits of the FP16 representation,
    /// and sc[1]/sc[2]/sc[3] each providing the next 4 bits of the FP16 mantissa.
    /// For simplicity, encode the fp16 bits directly:
    ///   d_bits = fp16 bits
    ///   sc[0] bits[15:12] = d_bits[3:0]
    ///   sc[1] bits[15:12] = d_bits[7:4]
    ///   sc[2] bits[15:12] = d_bits[11:8]
    ///   sc[3] bits[15:12] = d_bits[15:12]
    fn make_scaled_iq1m_block(d: f32) -> [u8; IQ1M_BLOCK_BYTES] {
        let mut block = [0u8; IQ1M_BLOCK_BYTES];
        let d_bits = half::f16::from_f32(d).to_bits();

        // Encode the 16-bit FP16 into 4 nibbles across sc[0..3] bits 12-15.
        // reconstruct_d reads: (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)
        // So:
        //   sc[0] >> 12         = d_bits[3:0]   → sc[0] = (d_bits & 0x000f) << 12
        //   (sc[1] >> 8) & 0xf0 = d_bits[7:4]   → sc[1] = (d_bits & 0x00f0) << 8
        //   (sc[2] >> 4) & 0xf00= d_bits[11:8]  → sc[2] = (d_bits & 0x0f00) << 4
        //   sc[3] & 0xf000      = d_bits[15:12] → sc[3] = d_bits & 0xf000
        let sc0: u16 = (d_bits & 0x000f) << 12;
        let sc1: u16 = (d_bits & 0x00f0) << 8;
        let sc2: u16 = (d_bits & 0x0f00) << 4;
        let sc3: u16 = d_bits & 0xf000;

        let sc0_bytes = sc0.to_le_bytes();
        let sc1_bytes = sc1.to_le_bytes();
        let sc2_bytes = sc2.to_le_bytes();
        let sc3_bytes = sc3.to_le_bytes();
        block[48] = sc0_bytes[0];
        block[49] = sc0_bytes[1];
        block[50] = sc1_bytes[0];
        block[51] = sc1_bytes[1];
        block[52] = sc2_bytes[0];
        block[53] = sc2_bytes[1];
        block[54] = sc3_bytes[0];
        block[55] = sc3_bytes[1];

        block
    }

    #[test]
    fn test_iq1_m_metadata() {
        assert_eq!(Iq1MRef.name(), "IQ1_M");
        assert_eq!(Iq1MRef.block_size(), 256);
        assert_eq!(Iq1MRef.block_bytes(), 56);
    }

    #[test]
    fn test_dequant_buffer_too_small_block() {
        let small = [0u8; 30];
        let mut out = [0.0f32; 256];
        let result = Iq1MRef.dequant_block(&small, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_buffer_too_small_output() {
        let block = make_zero_iq1m_block();
        let mut out = [0.0f32; 100];
        let result = Iq1MRef.dequant_block(&block, &mut out);
        assert!(matches!(result, Err(QuantError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_dequant_zero_scale() {
        // d = 0 → all outputs must be 0.0.
        let block = make_zero_iq1m_block();
        let mut out = [1.0f32; 256];
        Iq1MRef.dequant_block(&block, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] should be 0 when d=0, got {v}");
        }
    }

    #[test]
    fn test_reconstruct_d_roundtrip() {
        // Encode a known FP16 value into the scales and verify it's reconstructed.
        let d_in = 2.0_f32;
        let block = make_scaled_iq1m_block(d_in);
        let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];
        let d_out = reconstruct_d(scales);
        assert!(
            (d_out - d_in).abs() < 1e-2,
            "d_out={d_out}, expected {d_in}"
        );
    }

    #[test]
    fn test_dequant_nonzero_scale_grid0() {
        // With a known d, all qs=0/qh=0 (grid index 0 → all bytes 0xff = -1).
        // scales encode d=1.0. sub-block scale bits=0 → dl1=dl2=d*1.
        // delta = +0.125 (qh bits 3 and 7 are 0).
        // Expected per weight: 1.0 * (-1.0 + 0.125) = -0.875.
        let d = 1.0_f32;
        let block = make_scaled_iq1m_block(d);
        let mut out = [0.0f32; 256];
        Iq1MRef.dequant_block(&block, &mut out).unwrap();

        let d_actual = {
            let scales = &block[IQ1M_SCALES_OFFSET..IQ1M_BLOCK_BYTES];
            reconstruct_d(scales)
        };
        // dl = d_actual * (2*0 + 1) = d_actual
        // grid[0] byte = 0xff as i8 = -1
        // delta = +0.125
        let expected = d_actual * (-1.0 + IQ1M_DELTA);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-3,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_supported_by_dispatcher() {
        use crate::dispatch::KernelDispatcher;
        let d = KernelDispatcher::new();
        assert!(d.is_supported(oxillama_gguf::GgufTensorType::Iq1M));
    }

    #[test]
    fn test_gemv_dot_ones_matches_dequant_sum() {
        // 1-row × 256-col tensor; gemv with input=ones must equal sum of dequant.
        let block = make_scaled_iq1m_block(1.0);

        let mut dequant = [0.0f32; 256];
        Iq1MRef.dequant_block(&block, &mut dequant).unwrap();
        let expected: f32 = dequant.iter().sum();

        let tensor = crate::types::QuantTensor::new(
            block.to_vec(),
            vec![1, 256],
            oxillama_gguf::GgufTensorType::Iq1M,
        );
        let input = vec![1.0f32; 256];
        let mut out = [0.0f32; 1];
        Iq1MRef.gemv(&tensor, &input, &mut out).unwrap();

        assert!(
            (out[0] - expected).abs() < 1e-3,
            "gemv={}, expected dequant_sum={}",
            out[0],
            expected
        );
    }
}
